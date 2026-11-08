//! Binary index container: named, 64-byte aligned, checksummed sections,
//! read back zero-copy through a memory map.
//!
//! Layout (little-endian):
//! ```text
//! 0   magic "HOLMIDX\0"           8 bytes
//! 8   version u32, n_sections u32
//! 16  reserved                     16 bytes
//! 32  table: n × { name[32], offset u64, len u64, fnv1a u64 }   (56 bytes each)
//! ..  sections, each starting at a multiple of 64
//! ```

#[cfg(not(target_endian = "little"))]
compile_error!("the HOLM index container assumes a little-endian target");

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use memmap2::Mmap;

use crate::error::{CoreError, Result};
use crate::hash::Fnv1a;

/// Magic bytes at offset 0.
pub const MAGIC: [u8; 8] = *b"HOLMIDX\0";
/// Container format version.
pub const VERSION: u32 = 1;
/// Maximum section name length in bytes.
pub const MAX_NAME: usize = 32;

const HEADER: usize = 32;
const ENTRY: usize = MAX_NAME + 24;
const ALIGN: usize = 64;

/// Payload of a section: borrowed typed slices, or a small owned `u64` vector.
#[derive(Debug, Clone)]
pub enum Payload<'a> {
    /// Bytes.
    U8(&'a [u8]),
    /// Little-endian `u16`.
    U16(&'a [u16]),
    /// Little-endian `u32`.
    U32(&'a [u32]),
    /// Little-endian `u64`.
    U64(&'a [u64]),
    /// Owned little-endian `u64` (metadata).
    OwnedU64(Vec<u64>),
}

impl Payload<'_> {
    fn byte_len(&self) -> usize {
        match self {
            Payload::U8(s) => s.len(),
            Payload::U16(s) => s.len() * 2,
            Payload::U32(s) => s.len() * 4,
            Payload::U64(s) => s.len() * 8,
            Payload::OwnedU64(v) => v.len() * 8,
        }
    }

    /// Calls `f` on consecutive little-endian byte chunks of the payload.
    fn for_each_chunk(&self, mut f: impl FnMut(&[u8]) -> Result<()>) -> Result<()> {
        const CHUNK: usize = 8192;
        let mut buf: Vec<u8> = Vec::with_capacity(CHUNK * 8);
        macro_rules! chunks {
            ($slice:expr) => {
                for c in $slice.chunks(CHUNK) {
                    buf.clear();
                    for x in c {
                        buf.extend_from_slice(&x.to_le_bytes());
                    }
                    f(&buf)?;
                }
            };
        }
        match self {
            Payload::U8(s) => {
                for c in s.chunks(CHUNK * 8) {
                    f(c)?;
                }
            }
            Payload::U16(s) => chunks!(s),
            Payload::U32(s) => chunks!(s),
            Payload::U64(s) => chunks!(s),
            Payload::OwnedU64(v) => chunks!(v),
        }
        Ok(())
    }

    fn checksum(&self) -> u64 {
        let mut h = Fnv1a::new();
        self.for_each_chunk(|b| {
            h.update(b);
            Ok(())
        })
        .expect("hashing cannot fail");
        h.finish()
    }
}

/// Collects sections and writes them to a file.
#[derive(Debug, Default)]
pub struct ContainerWriter<'a> {
    sections: Vec<(String, Payload<'a>)>,
}

fn align_up(x: usize) -> usize {
    x.div_ceil(ALIGN) * ALIGN
}

impl<'a> ContainerWriter<'a> {
    /// Empty writer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a section; names must be unique and at most [`MAX_NAME`] bytes.
    pub fn add(&mut self, name: &str, payload: Payload<'a>) -> Result<()> {
        if name.is_empty() || name.len() > MAX_NAME || name.as_bytes().contains(&0) {
            return Err(CoreError::InvalidParts(format!(
                "invalid section name `{name}`"
            )));
        }
        if self.sections.iter().any(|(n, _)| n == name) {
            return Err(CoreError::InvalidParts(format!(
                "duplicate section `{name}`"
            )));
        }
        self.sections.push((name.to_owned(), payload));
        Ok(())
    }

    /// Writes header, section table and sections to `path`.
    pub fn write_to(&self, path: &Path) -> Result<()> {
        let n = self.sections.len();
        let mut offsets = Vec::with_capacity(n);
        let mut pos = align_up(HEADER + ENTRY * n);
        for (_, p) in &self.sections {
            offsets.push(pos);
            pos = align_up(pos + p.byte_len());
        }

        let mut out = BufWriter::with_capacity(1 << 20, File::create(path)?);
        let mut head = Vec::with_capacity(HEADER + ENTRY * n);
        head.extend_from_slice(&MAGIC);
        head.extend_from_slice(&VERSION.to_le_bytes());
        head.extend_from_slice(&(n as u32).to_le_bytes());
        head.resize(HEADER, 0);
        for ((name, p), &off) in self.sections.iter().zip(&offsets) {
            let mut field = [0u8; MAX_NAME];
            field[..name.len()].copy_from_slice(name.as_bytes());
            head.extend_from_slice(&field);
            head.extend_from_slice(&(off as u64).to_le_bytes());
            head.extend_from_slice(&(p.byte_len() as u64).to_le_bytes());
            head.extend_from_slice(&p.checksum().to_le_bytes());
        }
        out.write_all(&head)?;
        let mut written = head.len();
        let zeros = [0u8; ALIGN];
        for ((_, p), &off) in self.sections.iter().zip(&offsets) {
            out.write_all(&zeros[..off - written])?;
            p.for_each_chunk(|b| out.write_all(b).map_err(CoreError::from))?;
            written = off + p.byte_len();
        }
        out.write_all(&zeros[..align_up(written) - written])?;
        out.flush()?;
        Ok(())
    }
}

/// How much to verify when opening a container.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verify {
    /// Header and section table only (O(sections)).
    Header,
    /// Also recompute every section checksum (reads the whole file).
    Full,
}

#[derive(Debug, Clone)]
struct Entry {
    name: String,
    offset: usize,
    len: usize,
}

/// Memory-mapped read-only container.
#[derive(Debug)]
pub struct Container {
    mmap: Mmap,
    entries: Vec<Entry>,
}

fn le_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().expect("4 bytes"))
}

fn le_u64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

impl Container {
    /// Opens and validates a container file.
    pub fn open(path: &Path, verify: Verify) -> Result<Self> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len() as usize;
        if file_len < HEADER {
            return Err(CoreError::Truncated(format!("{file_len} bytes < header")));
        }
        // SAFETY: the index is opened read-only and is never modified while
        // mapped; concurrent modification of the file by another process is
        // documented as unsupported.
        let mmap = unsafe { Mmap::map(&file)? };
        let b: &[u8] = &mmap;
        if b[..8] != MAGIC {
            return Err(CoreError::BadMagic);
        }
        let version = le_u32(b, 8);
        if version != VERSION {
            return Err(CoreError::BadVersion {
                found: version,
                expected: VERSION,
            });
        }
        let n = le_u32(b, 12) as usize;
        let table_end = HEADER + ENTRY * n;
        if table_end > file_len {
            return Err(CoreError::Truncated("section table".into()));
        }
        let mut entries = Vec::with_capacity(n);
        let mut seen = HashSet::new();
        for i in 0..n {
            let at = HEADER + ENTRY * i;
            let raw = &b[at..at + MAX_NAME];
            let end = raw.iter().position(|&c| c == 0).unwrap_or(MAX_NAME);
            let name = std::str::from_utf8(&raw[..end])
                .map_err(|_| CoreError::InvalidParts("section name is not UTF-8".into()))?
                .to_owned();
            let offset = le_u64(b, at + MAX_NAME) as usize;
            let len = le_u64(b, at + MAX_NAME + 8) as usize;
            let sum = le_u64(b, at + MAX_NAME + 16);
            if offset.checked_add(len).is_none_or(|e| e > file_len) {
                return Err(CoreError::Truncated(format!("section `{name}`")));
            }
            if offset % ALIGN != 0 {
                return Err(CoreError::Misaligned(name));
            }
            if verify == Verify::Full {
                let mut h = Fnv1a::new();
                h.update(&b[offset..offset + len]);
                if h.finish() != sum {
                    return Err(CoreError::Checksum(name));
                }
            }
            if !seen.insert(name.clone()) {
                return Err(CoreError::InvalidParts(format!(
                    "duplicate section `{name}`"
                )));
            }
            entries.push(Entry { name, offset, len });
        }
        Ok(Self { mmap, entries })
    }

    /// Section names in file order.
    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|e| e.name.as_str()).collect()
    }

    /// True if the container has a section called `name`.
    pub fn has(&self, name: &str) -> bool {
        self.entries.iter().any(|e| e.name == name)
    }

    /// Raw bytes of a section.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let e = self
            .entries
            .iter()
            .find(|e| e.name == name)
            .ok_or_else(|| CoreError::MissingSection(name.to_owned()))?;
        Ok(&self.mmap[e.offset..e.offset + e.len])
    }

    /// Section viewed as `u16`.
    pub fn u16s(&self, name: &str) -> Result<&[u16]> {
        let b = self.bytes(name)?;
        // SAFETY: `u16` has no invalid bit patterns; the returned slice
        // borrows `self.mmap`, which lives as long as `self`.
        let (pre, mid, post) = unsafe { b.align_to::<u16>() };
        typed(name, pre, mid, post)
    }

    /// Section viewed as `u32`.
    pub fn u32s(&self, name: &str) -> Result<&[u32]> {
        let b = self.bytes(name)?;
        // SAFETY: `u32` has no invalid bit patterns; the returned slice
        // borrows `self.mmap`, which lives as long as `self`.
        let (pre, mid, post) = unsafe { b.align_to::<u32>() };
        typed(name, pre, mid, post)
    }

    /// Section viewed as `u64`.
    pub fn u64s(&self, name: &str) -> Result<&[u64]> {
        let b = self.bytes(name)?;
        // SAFETY: `u64` has no invalid bit patterns; the returned slice
        // borrows `self.mmap`, which lives as long as `self`.
        let (pre, mid, post) = unsafe { b.align_to::<u64>() };
        typed(name, pre, mid, post)
    }
}

fn typed<'s, T>(name: &str, pre: &[u8], mid: &'s [T], post: &[u8]) -> Result<&'s [T]> {
    if !pre.is_empty() {
        return Err(CoreError::Misaligned(name.to_owned()));
    }
    if !post.is_empty() {
        return Err(CoreError::InvalidParts(format!(
            "section `{name}` is not a whole number of {}-byte elements",
            std::mem::size_of::<T>()
        )));
    }
    Ok(mid)
}
