//! VectorSearch: a vector search engine written in Rust, built from scratch.
//!
//! The public surface today is small: pick a [`Metric`], build a [`FlatIndex`],
//! add vectors, and search. The flat index is exact and serves as the recall
//! baseline for the approximate indexes (HNSW, IVF) on the roadmap.
//!
//! ```
//! use vectorsearch::{FlatIndex, Index, Metric};
//!
//! let mut index = FlatIndex::new(3, Metric::L2);
//! index.add(1, &[0.0, 0.0, 0.0]).unwrap();
//! index.add(2, &[1.0, 0.0, 0.0]).unwrap();
//!
//! let hits = index.search(&[0.1, 0.0, 0.0], 1);
//! assert_eq!(hits[0].id, 1);
//! ```

pub mod distance;
pub mod error;
pub mod index;

pub use distance::Metric;
pub use error::{Error, Result};
pub use index::{FlatIndex, Index, Neighbor};
