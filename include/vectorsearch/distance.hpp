#pragma once

// Distance metrics.
//
// Every metric is expressed so that a smaller value means "closer". That lets
// the rest of the engine sort results ascending no matter which metric is in
// use. For L2 we return the squared distance (the square root is monotonic and
// would only waste cycles for ranking). For cosine we return 1 - similarity.
// For dot product we return -dot, since a larger dot product means more similar.

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

// Cosine distance, 1 - similarity. Returns 1.0 when either vector has zero
// magnitude, since the angle is undefined.
inline float cosine_distance(const float* a, const float* b, std::size_t dim) {
    float dot_ab = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (std::size_t i = 0; i < dim; ++i) {
        dot_ab += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    const float denom = std::sqrt(norm_a) * std::sqrt(norm_b);
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
