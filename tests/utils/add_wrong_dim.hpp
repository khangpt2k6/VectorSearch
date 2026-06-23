#pragma once

// Shared Index contract: add() must throw std::invalid_argument on a dimension
// mismatch and leave the index empty. Call after any index-specific setup
// (e.g. ScalarQuantizedIndex::train).

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "vectorsearch/index.hpp"

namespace test {

inline void check_add_rejects_wrong_dimension(vectorsearch::Index& idx,
                                              std::size_t wrong_dim = 2) {
    bool threw = false;
    try {
        std::vector<float> bad(wrong_dim, 0.0f);
        idx.add(1, bad.data(), bad.size());
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(idx.empty());
}

}  // namespace test
