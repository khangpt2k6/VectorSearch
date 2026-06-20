//! Distance metrics.
//!
//! Every metric is expressed so that a smaller value means "closer". That lets
//! the rest of the engine sort results ascending no matter which metric is in
//! use. For L2 we return the squared distance (the square root is monotonic and
//! would only waste cycles for ranking). For cosine we return `1 - similarity`.
//! For dot product we return `-dot`, since a larger dot product means more
//! similar.

/// The distance metric used by an index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Squared Euclidean distance.
    L2,
    /// Cosine distance, i.e. `1 - cosine_similarity`.
    Cosine,
    /// Negative dot product (so that larger similarity sorts first).
    Dot,
}

impl Metric {
    /// Compute the distance between two equal-length vectors.
    ///
    /// The caller is responsible for passing slices of the same length; the
    /// index layer checks dimensions before getting here.
    #[inline]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len());
        match self {
            Metric::L2 => squared_l2(a, b),
            Metric::Cosine => cosine_distance(a, b),
            Metric::Dot => -dot(a, b),
        }
    }
}

/// Squared Euclidean distance.
#[inline]
pub fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// Dot product.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Cosine distance, `1 - similarity`. Returns `1.0` when either vector has zero
/// magnitude, since the angle is undefined.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let mut dot_ab = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot_ab += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        return 1.0;
    }
    1.0 - dot_ab / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squared_l2_basic() {
        assert_eq!(squared_l2(&[0.0, 0.0], &[3.0, 4.0]), 25.0);
        assert_eq!(squared_l2(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    }

    #[test]
    fn dot_basic() {
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn cosine_identical_is_zero() {
        let v = [1.0, 2.0, 3.0];
        assert!(cosine_distance(&v, &v).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal_is_one() {
        let d = cosine_distance(&[1.0, 0.0], &[0.0, 1.0]);
        assert!((d - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_zero_vector_is_one() {
        assert_eq!(cosine_distance(&[0.0, 0.0], &[1.0, 1.0]), 1.0);
    }

    #[test]
    fn metric_dispatch_smaller_is_closer() {
        // The closer vector should always score lower.
        let q = [1.0, 0.0, 0.0];
        let near = [0.9, 0.1, 0.0];
        let far = [-1.0, 0.0, 0.0];
        for m in [Metric::L2, Metric::Cosine, Metric::Dot] {
            assert!(
                m.distance(&q, &near) < m.distance(&q, &far),
                "metric {:?} ranked the far vector closer",
                m
            );
        }
    }
}
