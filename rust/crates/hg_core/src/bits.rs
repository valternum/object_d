//! Succinct bit vector with rank/select.
//!
//! Layout: bits in little-endian `u64` words; a superblock directory stores the
//! number of ones before every block of [`WORDS_PER_BLOCK`] words (512 bits).
//!
//! Space: `len` bits + 64 bits per 512 bits (12.5 % overhead).
//! Time: `rank1` O(1) (≤ 8 popcounts), `select1` O(log blocks + 8 + 64).

use std::borrow::Cow;

use crate::error::{CoreError, Result};

/// Number of words per rank superblock.
pub const WORDS_PER_BLOCK: usize = 8;

fn words_for(len: usize) -> usize {
    len.div_ceil(64)
}

fn blocks_for(n_words: usize) -> usize {
    n_words.div_ceil(WORDS_PER_BLOCK)
}

/// Growable bit vector used to build a [`RankSelect`].
#[derive(Debug, Clone, Default)]
pub struct BitVecBuilder {
    words: Vec<u64>,
    len: usize,
}

impl BitVecBuilder {
    /// Empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder with `len` zero bits.
    pub fn with_len(len: usize) -> Self {
        Self {
            words: vec![0; words_for(len)],
            len,
        }
    }

    /// Appends one bit.
    pub fn push(&mut self, bit: bool) {
        if self.len % 64 == 0 {
            self.words.push(0);
        }
        if bit {
            self.words[self.len / 64] |= 1u64 << (self.len % 64);
        }
        self.len += 1;
    }

    /// Sets bit `i` to one. Panics if `i >= len`.
    pub fn set(&mut self, i: usize) {
        assert!(i < self.len, "bit index {i} out of range {}", self.len);
        self.words[i / 64] |= 1u64 << (i % 64);
    }

    /// Number of bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when no bits were pushed.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Builds the rank/select directory.
    pub fn freeze(self) -> RankSelect<'static> {
        let supers = build_supers(&self.words);
        let ones = *supers.last().unwrap_or(&0) as usize;
        RankSelect {
            words: Cow::Owned(self.words),
            supers: Cow::Owned(supers),
            len: self.len,
            ones,
        }
    }
}

fn build_supers(words: &[u64]) -> Vec<u64> {
    let n_blocks = blocks_for(words.len());
    let mut supers = Vec::with_capacity(n_blocks + 1);
    let mut acc = 0u64;
    supers.push(0);
    for block in words.chunks(WORDS_PER_BLOCK) {
        acc += block.iter().map(|w| u64::from(w.count_ones())).sum::<u64>();
        supers.push(acc);
    }
    supers
}

/// Position of the `k`-th set bit inside `w` (k < popcount(w)).
fn select_in_word(mut w: u64, k: u32) -> u32 {
    for _ in 0..k {
        w &= w - 1;
    }
    w.trailing_zeros()
}

/// Immutable bit vector with rank/select support, over owned or borrowed words.
#[derive(Debug, Clone)]
pub struct RankSelect<'a> {
    words: Cow<'a, [u64]>,
    supers: Cow<'a, [u64]>,
    len: usize,
    ones: usize,
}

impl<'a> RankSelect<'a> {
    /// Rebuilds a bit vector from serialized parts, validating their sizes.
    pub fn from_parts(words: Cow<'a, [u64]>, supers: Cow<'a, [u64]>, len: usize) -> Result<Self> {
        let n_words = words_for(len);
        if words.len() != n_words {
            return Err(CoreError::InvalidParts(format!(
                "bit vector of {len} bits needs {n_words} words, got {}",
                words.len()
            )));
        }
        let n_supers = blocks_for(n_words) + 1;
        if supers.len() != n_supers {
            return Err(CoreError::InvalidParts(format!(
                "rank directory needs {n_supers} entries, got {}",
                supers.len()
            )));
        }
        let ones = supers[n_supers - 1] as usize;
        if ones > len {
            return Err(CoreError::InvalidParts(format!(
                "{ones} ones in {len} bits"
            )));
        }
        Ok(Self {
            words,
            supers,
            len,
            ones,
        })
    }

    /// Number of bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True for an empty vector.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total number of ones.
    pub fn count_ones(&self) -> usize {
        self.ones
    }

    /// Bit `i`. Panics if `i >= len`.
    pub fn get(&self, i: usize) -> bool {
        assert!(i < self.len, "bit index {i} out of range {}", self.len);
        (self.words[i / 64] >> (i % 64)) & 1 == 1
    }

    /// Number of ones in `[0, i)`. Panics if `i > len`.
    pub fn rank1(&self, i: usize) -> usize {
        assert!(i <= self.len, "rank index {i} out of range {}", self.len);
        let word = i / 64;
        let block = word / WORDS_PER_BLOCK;
        let mut r = self.supers[block];
        for w in &self.words[block * WORDS_PER_BLOCK..word] {
            r += u64::from(w.count_ones());
        }
        let bit = i % 64;
        if bit > 0 {
            r += u64::from((self.words[word] & ((1u64 << bit) - 1)).count_ones());
        }
        r as usize
    }

    /// Position of the `k`-th one (0-based), `None` if `k >= count_ones()`.
    pub fn select1(&self, k: usize) -> Option<usize> {
        if k >= self.ones {
            return None;
        }
        let k = k as u64;
        // Last block whose prefix count is <= k.
        let block = self.supers.partition_point(|&s| s <= k) - 1;
        let mut remaining = k - self.supers[block];
        let start = block * WORDS_PER_BLOCK;
        let end = (start + WORDS_PER_BLOCK).min(self.words.len());
        for (offset, &w) in self.words[start..end].iter().enumerate() {
            let c = u64::from(w.count_ones());
            if remaining < c {
                let pos = (start + offset) * 64 + select_in_word(w, remaining as u32) as usize;
                return Some(pos);
            }
            remaining -= c;
        }
        None
    }

    /// Raw words (for serialization).
    pub fn words(&self) -> &[u64] {
        &self.words
    }

    /// Superblock directory (for serialization).
    pub fn supers(&self) -> &[u64] {
        &self.supers
    }
}
