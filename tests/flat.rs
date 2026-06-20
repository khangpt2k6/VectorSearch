//! Integration tests against the public API.

use vectorsearch::{FlatIndex, Index, Metric};

/// A small brute-force "is this the real nearest neighbor" check across metrics.
///
/// Note the expected answer differs by metric on purpose. The query sits right
/// next to id 20, so L2 and cosine (which both measure direction/closeness)
/// rank it first. Dot product is not normalized, so the large-magnitude vector
/// id 50 wins on raw dot. That is the correct, intended behavior.
#[test]
fn flat_index_finds_true_nearest_neighbor() {
    let data = [
        (10u64, vec![0.0, 0.0, 0.0]),
        (20, vec![1.0, 0.0, 0.0]),
        (30, vec![0.0, 1.0, 0.0]),
        (40, vec![0.0, 0.0, 1.0]),
        (50, vec![10.0, 10.0, 10.0]),
    ];

    let cases = [
        (Metric::L2, 20u64),
        (Metric::Cosine, 20),
        (Metric::Dot, 50),
    ];

    for (metric, expected_first) in cases {
        let mut idx = FlatIndex::new(3, metric);
        for (id, v) in &data {
            idx.add(*id, v).unwrap();
        }
        assert_eq!(idx.len(), data.len());

        let hits = idx.search(&[0.95, 0.02, 0.0], 3);
        assert_eq!(
            hits[0].id, expected_first,
            "metric {metric:?} ranked the wrong vector first"
        );
        // Results must be sorted closest first.
        for w in hits.windows(2) {
            assert!(w[0].distance <= w[1].distance);
        }
    }
}

#[test]
fn flat_index_recall_matches_itself() {
    // Trivial but useful invariant: searching for a stored vector returns it.
    let mut idx = FlatIndex::new(4, Metric::Cosine);
    let vectors = [
        (1u64, vec![1.0, 2.0, 3.0, 4.0]),
        (2, vec![-1.0, 0.5, 2.0, 0.0]),
        (3, vec![3.0, 3.0, 0.0, 1.0]),
    ];
    for (id, v) in &vectors {
        idx.add(*id, v).unwrap();
    }
    for (id, v) in &vectors {
        let hits = idx.search(v, 1);
        assert_eq!(hits[0].id, *id);
    }
}
