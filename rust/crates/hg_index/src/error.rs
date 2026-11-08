//! Error type of `hg_index`.

use hg_core::CoreError;
use thiserror::Error;

/// Errors raised while reading YAGO files or building the index.
#[derive(Debug, Error)]
pub enum IndexError {
    /// Syntax error in an input file.
    #[error("line {line}: {msg}")]
    Syntax {
        /// 1-based line number.
        line: usize,
        /// Description.
        msg: String,
    },
    /// Inconsistent schema.
    #[error("schema: {0}")]
    Schema(String),
    /// The class graph contains a cycle.
    #[error("class graph has a cycle through `{0}`")]
    Cycle(String),
    /// Invalid builder input or configuration.
    #[error("invalid input: {0}")]
    Invalid(String),
    /// Error from `hg_core`.
    #[error(transparent)]
    Core(#[from] CoreError),
    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// Result alias for `hg_index`.
pub type Result<T> = std::result::Result<T, IndexError>;
