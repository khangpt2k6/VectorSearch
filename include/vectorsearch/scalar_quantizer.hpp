#pragma once

// Scalar quantization (SQ8).
//
// Each f32 component is mapped onto an 8-bit integer using a per-dimension
// affine transform learned from the data: a dimension that only ever holds
// values in [lo, hi] spends all 256 codes across that range, so a tight
// dimension keeps more precision than a wide one. Storage drops to one byte per
// component, a 4x cut versus raw f32.
//
// The reconstruction is lossy, so an index built on codes alone trades a little
// recall for the memory win. A re-rank pass over the original vectors buys that
// recall back; that lives in the quantized index, not here.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectorsearch {

class ScalarQuantizer {
public:
    ScalarQuantizer() = default;

    // Learn per-dimension bounds from count vectors packed end to end. Replaces
    // any previous training.
    void train(const float* vectors, std::size_t count, std::size_t dim);

    bool trained() const { return trained_; }
    std::size_t dim() const { return dim_; }

    // Encode one vector into dim code bytes. Values outside the trained range are
    // clamped to the nearest end. Undefined before train().
    void encode(const float* vec, std::uint8_t* code) const;

    // Reconstruct the approximate f32 vector from its codes.
    void decode(const std::uint8_t* code, float* vec) const;

    // std::vector convenience wrappers.
    void train(const std::vector<float>& vectors, std::size_t dim) {
        train(vectors.data(), dim == 0 ? 0 : vectors.size() / dim, dim);
    }
    std::vector<std::uint8_t> encode(const std::vector<float>& vec) const {
        std::vector<std::uint8_t> code(dim_);
        encode(vec.data(), code.data());
        return code;
    }
    std::vector<float> decode(const std::vector<std::uint8_t>& code) const {
        std::vector<float> vec(dim_);
        decode(code.data(), vec.data());
        return vec;
    }

    // Largest possible reconstruction error on a dimension: half a code step.
    // Handy for tests and for reasoning about the recall hit.
    float max_abs_error(std::size_t d) const { return 0.5f * step_[d]; }

private:
    std::size_t dim_ = 0;
    bool trained_ = false;
    std::vector<float> lower_;  // per-dimension minimum seen in training
    std::vector<float> step_;   // per-dimension code width = (hi - lo) / 255
};

}  // namespace vectorsearch
