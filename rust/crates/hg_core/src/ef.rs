//! Elias–Fano encoding of non-decreasing `u64` sequences.
//!
//! For `n` values with universe `U = last + 1`, each value is split into
//! `l = floor(log2(U / n))` lower bits (packed) and upper bits stored in unary
//! in a [`RankSelect`] bit vector: value `i` sets bit `(v >> l) + i`.
//!
//! Space: `n·l + n + (last >> l) + 1 ≤ n·(2 + l) + 1` bits plus the rank
//! directory. `get(i)` = one `select1` + one lower-bit read.

use std::borrow::Cow;

use crate::bits::{BitVecBuilder, RankSelect};
use crate::error::{CoreError, Result};

/// Compressed non-decreasing sequence with O(1)-ish random access.
#[derive(Debug, Clone)]
pub struct EliasFano<'a> {
    n: usize,
    low_bits: u32,
    low: Cow<'a, [u64]>,
    high: RankSelect<'a>,
}

fn low_bits_for(n: usize, last: u64) -> u32 {
    if n == 0 {
        return 0;
    }
    let ratio = (u128::from(last) + 1) / n as u128;
    if ratio > 1 {
        ratio.ilog2().min(63)
    } else {
        0
    }
}

fn read_bits(words: &[u64], pos: usize, width: u32) -> u64 {
    if width == 0 {
        return 0;
    }
    let word = pos / 64;
    let offset = (pos % 64) as u32;
    let mask = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let mut v = words[word] >> offset;
    if offset + width > 64 {
        v |= words[word + 1] << (64 - offset);
    }
    v & mask
}

fn write_bits(words: &mut [u64], pos: usize, width: u32, value: u64) {
    if width == 0 {
        return;
    }
    let word = pos / 64;
    let offset = (pos % 64) as u32;
    words[word] |= value << offset;
    if offset + width > 64 {
        words[word + 1] |= value >> (64 - offset);
    }
}

impl EliasFano<'static> {
    /// Encodes `values`; errors if the sequence decreases anywhere.
    pub fn from_sorted(values: &[u64]) -> Result<Self> {
        if let Some(index) = (1..values.len()).find(|&i| values[i] < values[i - 1]) {
            return Err(CoreError::NotMonotone { index });
        }
        let n = values.len();
        let last = values.last().copied().unwrap_or(0);
        let l = low_bits_for(n, last);
        let low_mask = if l == 0 { 0 } else { (1u64 << l) - 1 };

        let mut low = vec![0u64; (n * l as usize).div_ceil(64)];
        let high_len = if n == 0 {
            0
        } else {
            n + (last >> l) as usize + 1
        };
        let mut high = BitVecBuilder::with_len(high_len);
        for (i, &v) in values.iter().enumerate() {
            write_bits(&mut low, i * l as usize, l, v & low_mask);
            high.set((v >> l) as usize + i);
        }
        Ok(Self {
            n,
            low_bits: l,
            low: Cow::Owned(low),
            high: high.freeze(),
        })
    }
}

impl<'a> EliasFano<'a> {
    /// Rebuilds from serialized parts.
    pub fn from_parts(
        n: usize,
        low_bits: u32,
        low: Cow<'a, [u64]>,
        high: RankSelect<'a>,
    ) -> Result<Self> {
        if low_bits > 63 {
            return Err(CoreError::InvalidParts(format!("low_bits {low_bits} > 63")));
        }
        let need = (n * low_bits as usize).div_ceil(64);
        if low.len() != need {
            return Err(CoreError::InvalidParts(format!(
                "Elias–Fano with n={n}, l={low_bits} needs {need} lower words, got {}",
                low.len()
            )));
        }
        if high.count_ones() != n {
            return Err(CoreError::InvalidParts(format!(
                "upper bits hold {} values, expected {n}",
                high.count_ones()
            )));
        }
        Ok(Self {
            n,
            low_bits,
            low,
            high,
        })
    }

    /// Number of values.
    pub fn len(&self) -> usize {
        self.n
    }

    /// True when empty.
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Value `i`. Panics if `i >= len`.
    pub fn get(&self, i: usize) -> u64 {
        assert!(i < self.n, "index {i} out of range {}", self.n);
        let pos = self.high.select1(i).expect("upper bits hold n ones");
        let hi = (pos - i) as u64;
        let lo = read_bits(&self.low, i * self.low_bits as usize, self.low_bits);
        (hi << self.low_bits) | lo
    }

    /// `(get(i), get(i + 1))`. Panics if `i + 1 >= len`.
    pub fn range(&self, i: usize) -> (u64, u64) {
        (self.get(i), self.get(i + 1))
    }

    /// Iterates over all values in order.
    pub fn iter(&self) -> impl Iterator<Item = u64> + '_ {
        (0..self.n).map(move |i| self.get(i))
    }

    /// Bits of the encoded payload: `n·l` lower bits + upper-bit vector length.
    pub fn payload_bits(&self) -> usize {
        self.n * self.low_bits as usize + self.high.len()
    }

    /// Number of lower bits per value.
    pub fn low_bits(&self) -> u32 {
        self.low_bits
    }

    /// Packed lower bits (for serialization).
    pub fn low_words(&self) -> &[u64] {
        &self.low
    }

    /// Upper-bit vector (for serialization).
    pub fn high(&self) -> &RankSelect<'a> {
        &self.high
    }
}
