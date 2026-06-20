//! Error types for the engine.

use std::fmt;

/// Errors returned by index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A vector was added or queried with the wrong number of dimensions.
    DimensionMismatch { expected: usize, got: usize },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => write!(
                f,
                "dimension mismatch: index expects {expected}, got {got}"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Convenience alias for results from this crate.
pub type Result<T> = std::result::Result<T, Error>;
