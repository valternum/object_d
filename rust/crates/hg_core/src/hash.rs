//! 64-bit FNV-1a hash (integrity checksums, not cryptographic).

const OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

/// Streaming FNV-1a hasher.
#[derive(Debug, Clone, Copy)]
pub struct Fnv1a(u64);

impl Default for Fnv1a {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv1a {
    /// Fresh hasher.
    pub fn new() -> Self {
        Self(OFFSET_BASIS)
    }

    /// Feeds bytes.
    pub fn update(&mut self, bytes: &[u8]) {
        let mut h = self.0;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(PRIME);
        }
        self.0 = h;
    }

    /// Current hash value.
    pub fn finish(&self) -> u64 {
        self.0
    }
}

/// FNV-1a of a byte slice.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h = Fnv1a::new();
    h.update(bytes);
    h.finish()
}
