#pragma once

// Distance metrics.
//
// All metrics are defined so lower values mean closer, allowing the 
// engine to always sort results in ascending order regardless of the metric.

#include <cmath>
#include <cstddef>

namespace vectorsearch {

enum class Metric {
    L2,      // squared Euclidean distance
    Cosine,  // 1 - cosine_similarity
    Dot,     // negative dot product (so larger similarity sorts first)
};

// Squared Euclidean distance.
inline float squared_l2(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// Dot product.
inline float dot(const float* a, const float* b, std::size_t dim) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

// Cosine distance, 1 - similarity. 
// Returns 1.0 when the two vectors are perpendicular.
inline float cosine_distance(const float* a, const float* b, std::size_t dim) {
    const float dot_ab = dot(a, b, dim);
    const float denom =
        std::sqrt(dot(a, a, dim)) * std::sqrt(dot(b, b, dim));
    if (denom == 0.0f) {
        return 1.0f;
    }
    return 1.0f - dot_ab / denom;
}

// Dispatch on the metric. Both slices must have length dim; the index layer
// checks dimensions before getting here.
inline float distance(Metric metric, const float* a, const float* b,
                      std::size_t dim) {
    switch (metric) {
        case Metric::L2:
            return squared_l2(a, b, dim);
        case Metric::Cosine:
            return cosine_distance(a, b, dim);
        case Metric::Dot:
            return -dot(a, b, dim);
    }
    return 0.0f;  // unreachable; silences compiler warnings
}

}  // namespace vectorsearch
