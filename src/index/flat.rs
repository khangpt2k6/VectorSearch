//! Flat (brute force) index.
//!
//! Stores every vector and scans all of them on each query. This is O(n * dim)
//! per search, so it does not scale, but it is always exact. That makes it the
//! ground truth that approximate indexes are measured against, and a perfectly
//! reasonable choice for small collections.

use super::{Index, Neighbor};
use crate::distance::Metric;
use crate::error::{Error, Result};

/// An exact, brute-force index.
#[derive(Debug, Clone)]
pub struct FlatIndex {
    dim: usize,
    metric: Metric,
    /// All vectors packed end to end: vector `i` is `data[i*dim .. (i+1)*dim]`.
    data: Vec<f32>,
    ids: Vec<u64>,
}

impl FlatIndex {
    /// Create an empty index for vectors of `dim` dimensions.
    pub fn new(dim: usize, metric: Metric) -> Self {
        FlatIndex {
            dim,
            metric,
            data: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// The dimensionality this index expects.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The metric this index ranks by.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    #[inline]
    fn vector_at(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }
}

impl Index for FlatIndex {
    fn add(&mut self, id: u64, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        self.data.extend_from_slice(vector);
        self.ids.push(id);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Neighbor> {
        if query.len() != self.dim || k == 0 {
            return Vec::new();
        }

        let mut scored: Vec<Neighbor> = (0..self.ids.len())
            .map(|i| Neighbor {
                id: self.ids[i],
                distance: self.metric.distance(query, self.vector_at(i)),
            })
            .collect();

        if scored.is_empty() {
            return scored;
        }

        // Partial sort: we only need the k smallest, in order.
        let k = k.min(scored.len());
        scored.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
            a.distance.total_cmp(&b.distance)
        });
        scored.truncate(k);
        scored.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        scored
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_rejects_wrong_dimension() {
        let mut idx = FlatIndex::new(3, Metric::L2);
        assert_eq!(
            idx.add(1, &[0.0, 0.0]),
            Err(Error::DimensionMismatch { expected: 3, got: 2 })
        );
        assert!(idx.is_empty());
    }

    #[test]
    fn search_returns_closest_first() {
        let mut idx = FlatIndex::new(3, Metric::L2);
        idx.add(1, &[0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[1.0, 0.0, 0.0]).unwrap();
        idx.add(3, &[5.0, 5.0, 5.0]).unwrap();

        let hits = idx.search(&[0.1, 0.0, 0.0], 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id, 1);
        assert_eq!(hits[1].id, 2);
        assert!(hits[0].distance <= hits[1].distance);
    }

    #[test]
    fn search_caps_at_index_size() {
        let mut idx = FlatIndex::new(2, Metric::L2);
        idx.add(1, &[0.0, 0.0]).unwrap();
        let hits = idx.search(&[0.0, 0.0], 10);
        assert_eq!(hits.len(), 1);
    }

    #[test]
    fn search_empty_index_is_empty() {
        let idx = FlatIndex::new(2, Metric::L2);
        assert!(idx.search(&[0.0, 0.0], 5).is_empty());
    }

    #[test]
    fn wrong_dimension_query_is_empty() {
        let mut idx = FlatIndex::new(3, Metric::L2);
        idx.add(1, &[0.0, 0.0, 0.0]).unwrap();
        assert!(idx.search(&[0.0, 0.0], 1).is_empty());
    }
}
