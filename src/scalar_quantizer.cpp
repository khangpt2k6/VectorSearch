#include "vectorsearch/scalar_quantizer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace vectorsearch {

void ScalarQuantizer::train(const float* vectors, std::size_t count,
                            std::size_t dim) {
    dim_ = dim;
    lower_.assign(dim, std::numeric_limits<float>::max());
    std::vector<float> upper(dim, std::numeric_limits<float>::lowest());

    for (std::size_t i = 0; i < count; ++i) {
        const float* v = vectors + i * dim;
        for (std::size_t d = 0; d < dim; ++d) {
            lower_[d] = std::min(lower_[d], v[d]);
            upper[d] = std::max(upper[d], v[d]);
        }
    }

    step_.assign(dim, 0.0f);
    for (std::size_t d = 0; d < dim; ++d) {
        if (count == 0) {
            lower_[d] = 0.0f;  // nothing seen; keep a sane default
        } else if (upper[d] > lower_[d]) {
            step_[d] = (upper[d] - lower_[d]) / 255.0f;
        }
        // else the dimension is constant: step stays 0, decode returns lower_.
    }
    trained_ = true;
}

void ScalarQuantizer::encode(const float* vec, std::uint8_t* code) const {
    for (std::size_t d = 0; d < dim_; ++d) {
        if (step_[d] == 0.0f) {
            code[d] = 0;
            continue;
        }
        const float q = (vec[d] - lower_[d]) / step_[d];
        const float clamped = std::min(255.0f, std::max(0.0f, std::round(q)));
        code[d] = static_cast<std::uint8_t>(clamped);
    }
}

void ScalarQuantizer::decode(const std::uint8_t* code, float* vec) const {
    for (std::size_t d = 0; d < dim_; ++d) {
        vec[d] = lower_[d] + static_cast<float>(code[d]) * step_[d];
    }
}

}  // namespace vectorsearch
