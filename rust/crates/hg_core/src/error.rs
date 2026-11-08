//! Error type shared by `hg_core`.

use thiserror::Error;

/// Errors raised by `hg_core` structures and the index container.
#[derive(Debug, Error)]
pub enum CoreError {
    /// Raw parts handed to a `from_parts` constructor are inconsistent.
    #[error("invalid parts: {0}")]
    InvalidParts(String),
    /// A sequence that must be non-decreasing decreases at `index`.
    #[error("sequence is not monotone at index {index}")]
    NotMonotone {
        /// First index `i` with `values[i] < values[i - 1]`.
        index: usize,
    },
    /// Invalid input given to a graph builder.
    #[error("invalid graph: {0}")]
    InvalidGraph(String),
    /// The file does not start with the container magic bytes.
    #[error("bad magic bytes")]
    BadMagic,
    /// Unsupported container format version.
    #[error("unsupported container version {found} (expected {expected})")]
    BadVersion {
        /// Version stored in the file.
        found: u32,
        /// Version this build understands.
        expected: u32,
    },
    /// File shorter than its header or section table claims.
    #[error("truncated container: {0}")]
    Truncated(String),
    /// A section is not aligned for the requested element type.
    #[error("section `{0}` is misaligned for the requested element type")]
    Misaligned(String),
    /// A required section is absent.
    #[error("missing section `{0}`")]
    MissingSection(String),
    /// Stored and recomputed checksum of a section differ.
    #[error("checksum mismatch in section `{0}`")]
    Checksum(String),
    /// Underlying I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Result alias for `hg_core`.
pub type Result<T> = std::result::Result<T, CoreError>;
