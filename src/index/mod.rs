//! Index implementations.
//!
//! Every index ranks results by distance ascending (smaller is closer). The
//! flat index here is exact and acts as the recall baseline for the
//! approximate indexes that come later (HNSW, IVF).

mod flat;

pub use flat::FlatIndex;

/// A single search result: the stored id and its distance to the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u64,
    pub distance: f32,
}

/// The behavior shared by every index.
pub trait Index {
    /// Add a vector under the given id. Errors if the dimension is wrong.
    fn add(&mut self, id: u64, vector: &[f32]) -> crate::error::Result<()>;

    /// Return the `k` nearest neighbors to `query`, sorted closest first.
    ///
    /// Returns fewer than `k` results if the index holds fewer vectors. A query
    /// of the wrong dimension yields an empty result rather than an error, so a
    /// bad query never panics a search loop.
    fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor>;

    /// Number of vectors currently stored.
    fn len(&self) -> usize;

    /// Whether the index holds no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
