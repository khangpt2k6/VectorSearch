#pragma once

// Deterministic synthetic vectors for tests. count vectors of length dim are
// packed end-to-end (count * dim floats). Same mt19937_64 + uniform_real as
// the old per-file copies so fixed seeds produce identical data.

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace test {

inline std::vector<float> random_data(std::size_t count, std::size_t dim,
                                      std::uint64_t seed,
                                      float lo = -1.0f, float hi = 1.0f) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    std::vector<float> out(count * dim);
    for (float& x : out) x = dist(rng);
    return out;
}

}  // namespace test
