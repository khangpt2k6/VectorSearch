#include "vectorsearch/scalar_quantizer.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include "check.hpp"
#include "utils/random_data.hpp"

using namespace vectorsearch;

static void train_records_dim() {
    ScalarQuantizer q;
    CHECK(!q.trained());
    auto data = test::random_data(50, 8, 1);
    q.train(data, 8);
    CHECK(q.trained());
    CHECK(q.dim() == 8);
}

// decode(encode(v)) must stay within half a code step of v on every dimension.
static void roundtrip_within_half_step() {
    const std::size_t dim = 16;
    const std::size_t n = 200;
    ScalarQuantizer q;
    auto data = test::random_data(n, dim, 42);
    q.train(data, dim);

    for (std::size_t i = 0; i < n; ++i) {
        const float* v = data.data() + i * dim;
        std::vector<std::uint8_t> code(dim);
        q.encode(v, code.data());
        std::vector<float> back(dim);
        q.decode(code.data(), back.data());
        for (std::size_t d = 0; d < dim; ++d) {
            const float err = std::fabs(back[d] - v[d]);
            CHECK(err <= q.max_abs_error(d) + 1e-4f);
        }
    }
}

// a dimension that never varies has zero step and must reconstruct exactly.
static void constant_dimension_is_exact() {
    const std::size_t dim = 3;
    // dim 1 is always 7.0; dims 0 and 2 vary.
    std::vector<float> data = {
        0.0f, 7.0f, -2.0f,
        5.0f, 7.0f,  9.0f,
        2.5f, 7.0f,  1.0f,
    };
    ScalarQuantizer q;
    q.train(data, dim);
    CHECK(q.max_abs_error(1) == 0.0f);

    std::vector<std::uint8_t> code(dim);
    q.encode(data.data(), code.data());
    std::vector<float> back(dim);
    q.decode(code.data(), back.data());
    CHECK(back[1] == 7.0f);
}

// values past the trained range clamp to the end codes, not wrap around.
static void clamps_out_of_range() {
    const std::size_t dim = 1;
    std::vector<float> data = {0.0f, 10.0f};  // trained range [0, 10]
    ScalarQuantizer q;
    q.train(data, dim);

    std::vector<std::uint8_t> code(dim);
    std::vector<float> back(dim);

    const float below = -5.0f;
    q.encode(&below, code.data());
    CHECK(code[0] == 0);
    q.decode(code.data(), back.data());
    CHECK(std::fabs(back[0] - 0.0f) < 1e-4f);

    const float above = 99.0f;
    q.encode(&above, code.data());
    CHECK(code[0] == 255);
    q.decode(code.data(), back.data());
    CHECK(std::fabs(back[0] - 10.0f) < 0.1f);
}

int main() {
    train_records_dim();
    roundtrip_within_half_step();
    constant_dimension_is_exact();
    clamps_out_of_range();

    if (test::failures() == 0) {
        std::printf("test_scalar_quantizer: all checks passed\n");
    }
    return test::failures() == 0 ? 0 : 1;
}
