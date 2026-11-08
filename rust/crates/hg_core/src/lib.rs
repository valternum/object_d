//! Core primitives of the HOLM Rust layer: succinct bit vectors, Elias–Fano
//! sequences, the typed incidence hypergraph and the mmap index container.
#![deny(unsafe_op_in_unsafe_fn)]

pub mod attrs;
pub mod bits;
pub mod container;
pub mod ef;
pub mod error;
pub mod graph;
pub mod hash;
pub mod store;

pub use error::{CoreError, Result};
