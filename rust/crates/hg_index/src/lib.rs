//! Offline builder of the HOLM knowledge-graph index from YAGO 4.6 Turtle files.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod build;
pub mod dict;
pub mod error;
pub mod index;
pub mod lex;
pub mod renumber;
pub mod schema;
pub mod taxonomy;

pub use error::{IndexError, Result};
