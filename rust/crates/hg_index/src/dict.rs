//! Term dictionary built by external sort with bounded memory.
//!
//! Terms are buffered up to `chunk_bytes`, sorted and deduplicated, and spilled
//! to temporary run files (`u32` length + bytes). [`DictBuilder::finish`]
//! k-way merges the runs with a binary heap into one sorted, unique blob.
//! The id of a term is its rank in byte order.
//!
//! Memory: O(chunk_bytes + k · max_term) during the build, plus the output
//! blob. Time: O(N log N) for N added terms. Lookup: O(log n) comparisons;
//! `get`: one Elias–Fano range.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use hg_core::container::{Container, ContainerWriter, Payload};
use hg_core::ef::EliasFano;
use hg_core::store::{read_ef, write_ef};

use crate::error::{IndexError, Result};

/// Sorted, deduplicated term list; the id of a term is its rank.
#[derive(Debug, Clone)]
pub struct Dict<'a> {
    blob: Cow<'a, [u8]>,
    offsets: EliasFano<'a>,
}

fn from_sorted_unique<T: AsRef<[u8]>>(terms: &[T]) -> Dict<'static> {
    let mut blob = Vec::with_capacity(terms.iter().map(|t| t.as_ref().len()).sum());
    let mut offsets = Vec::with_capacity(terms.len() + 1);
    offsets.push(0u64);
    for t in terms {
        blob.extend_from_slice(t.as_ref());
        offsets.push(blob.len() as u64);
    }
    Dict {
        blob: Cow::Owned(blob),
        offsets: EliasFano::from_sorted(&offsets).expect("offsets are monotone"),
    }
}

impl<'a> Dict<'a> {
    /// Number of terms.
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// True if empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Term with id `id`. Panics if out of range.
    pub fn get(&self, id: u32) -> &str {
        let (a, b) = self.offsets.range(id as usize);
        std::str::from_utf8(&self.blob[a as usize..b as usize]).expect("dictionary terms are UTF-8")
    }

    /// Id of `term`, or `None`.
    pub fn lookup(&self, term: &str) -> Option<u32> {
        let key = term.as_bytes();
        let (mut lo, mut hi) = (0usize, self.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let (a, b) = self.offsets.range(mid);
            match self.blob[a as usize..b as usize].cmp(key) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid as u32),
            }
        }
        None
    }

    /// Concatenated term bytes (for tests and serialization).
    pub fn blob(&self) -> &[u8] {
        &self.blob
    }

    /// New dictionary with the terms satisfying `keep` (order preserved).
    pub fn filter(&self, mut keep: impl FnMut(&str) -> bool) -> Dict<'static> {
        let kept: Vec<&str> = (0..self.len() as u32)
            .map(|i| self.get(i))
            .filter(|t| keep(t))
            .collect();
        from_sorted_unique(&kept)
    }

    /// Adds the dictionary sections under `name` (`.blob`, `.off`).
    pub fn write<'w>(&'w self, w: &mut ContainerWriter<'w>, name: &str) -> Result<()> {
        w.add(&format!("{name}.blob"), Payload::U8(&self.blob))?;
        write_ef(w, &format!("{name}.off"), &self.offsets)?;
        Ok(())
    }

    /// Reads a dictionary written by [`Dict::write`].
    pub fn read(c: &'a Container, name: &str) -> Result<Self> {
        let blob = c.bytes(&format!("{name}.blob"))?;
        let offsets = read_ef(c, &format!("{name}.off"))?;
        if offsets.is_empty() || offsets.get(offsets.len() - 1) as usize != blob.len() {
            return Err(IndexError::Invalid(format!(
                "dictionary `{name}`: offsets do not match blob"
            )));
        }
        Ok(Self {
            blob: Cow::Borrowed(blob),
            offsets,
        })
    }
}

impl Dict<'static> {
    /// Dictionary of the given terms (in memory; duplicates removed).
    pub fn from_terms<I, S>(terms: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut v: Vec<String> = terms.into_iter().map(|s| s.as_ref().to_owned()).collect();
        v.sort_unstable();
        v.dedup();
        from_sorted_unique(&v)
    }
}

static RUN_ID: AtomicU64 = AtomicU64::new(0);

/// External-sort dictionary builder: spills sorted runs of at most
/// `chunk_bytes` term bytes to `tmp_dir`, then k-way merges them.
#[derive(Debug)]
pub struct DictBuilder {
    tmp_dir: PathBuf,
    chunk_bytes: usize,
    buf: Vec<String>,
    buf_bytes: usize,
    runs: Vec<PathBuf>,
}

fn read_term(r: &mut impl Read) -> Result<Option<Vec<u8>>> {
    let mut len = [0u8; 4];
    match r.read_exact(&mut len) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let mut t = vec![0u8; u32::from_le_bytes(len) as usize];
    r.read_exact(&mut t)?;
    Ok(Some(t))
}

impl DictBuilder {
    /// New builder.
    pub fn new(tmp_dir: &Path, chunk_bytes: usize) -> Self {
        Self {
            tmp_dir: tmp_dir.to_path_buf(),
            chunk_bytes: chunk_bytes.max(1),
            buf: Vec::new(),
            buf_bytes: 0,
            runs: Vec::new(),
        }
    }

    /// Adds one term (duplicates allowed).
    pub fn add(&mut self, term: &str) -> Result<()> {
        self.buf_bytes += term.len() + 24;
        self.buf.push(term.to_owned());
        if self.buf_bytes >= self.chunk_bytes {
            self.spill()?;
        }
        Ok(())
    }

    fn spill(&mut self) -> Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }
        self.buf.sort_unstable();
        self.buf.dedup();
        let id = RUN_ID.fetch_add(1, Ordering::Relaxed);
        let path = self
            .tmp_dir
            .join(format!("holm_dict_run_{}_{id}.bin", std::process::id()));
        let mut w = BufWriter::with_capacity(1 << 20, File::create(&path)?);
        for t in &self.buf {
            let len = u32::try_from(t.len())
                .map_err(|_| IndexError::Invalid("term longer than 4 GiB".into()))?;
            w.write_all(&len.to_le_bytes())?;
            w.write_all(t.as_bytes())?;
        }
        w.flush()?;
        self.runs.push(path);
        self.buf.clear();
        self.buf_bytes = 0;
        Ok(())
    }

    /// Number of spilled runs so far.
    pub fn runs(&self) -> usize {
        self.runs.len()
    }

    /// Merges all runs into the final dictionary and removes temporary files.
    pub fn finish(mut self) -> Result<Dict<'static>> {
        if self.runs.is_empty() {
            self.buf.sort_unstable();
            self.buf.dedup();
            return Ok(from_sorted_unique(&self.buf));
        }
        self.spill()?;
        let result = self.merge();
        for p in &self.runs {
            let _ = std::fs::remove_file(p);
        }
        result
    }

    fn merge(&self) -> Result<Dict<'static>> {
        let mut readers = Vec::with_capacity(self.runs.len());
        let mut heap = BinaryHeap::new();
        for (i, p) in self.runs.iter().enumerate() {
            let mut r = BufReader::with_capacity(1 << 16, File::open(p)?);
            if let Some(t) = read_term(&mut r)? {
                heap.push(Reverse((t, i)));
            }
            readers.push(r);
        }
        let mut blob = Vec::new();
        let mut offsets = vec![0u64];
        let mut last: Option<Vec<u8>> = None;
        while let Some(Reverse((t, i))) = heap.pop() {
            if last.as_deref() != Some(t.as_slice()) {
                blob.extend_from_slice(&t);
                offsets.push(blob.len() as u64);
            }
            if let Some(next) = read_term(&mut readers[i])? {
                heap.push(Reverse((next, i)));
            }
            last = Some(t);
        }
        Ok(Dict {
            blob: Cow::Owned(blob),
            offsets: EliasFano::from_sorted(&offsets)?,
        })
    }
}
