#include "vectorsearch/scalar_quantizer.hpp"

namespace vectorsearch {

// stub: returns nothing useful yet so the tests fail.
void ScalarQuantizer::train(const float*, std::size_t, std::size_t dim) {
    dim_ = dim;
    trained_ = true;
    lower_.assign(dim, 0.0f);
    step_.assign(dim, 0.0f);
}

void ScalarQuantizer::encode(const float*, std::uint8_t* code) const {
    for (std::size_t d = 0; d < dim_; ++d) code[d] = 0;
}

void ScalarQuantizer::decode(const std::uint8_t*, float* vec) const {
    for (std::size_t d = 0; d < dim_; ++d) vec[d] = 0.0f;
}

}  // namespace vectorsearch
