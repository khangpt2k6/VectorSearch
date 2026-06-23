#pragma once

// Recall@k of an approximate index against an exact ground-truth index.
// For each query, counts how many truth top-k ids also appear in the
// approximate top-k, then averages over queries. The denominator is the number
// of truth neighbors actually returned (min(k, index size)), not k blindly.

#include <cstddef>
#include <unordered_set>

#include "vectorsearch/index.hpp"

namespace test {

inline double recall_at_k(const vectorsearch::Index& approx,
                          const vectorsearch::Index& truth,
                          const float* queries, std::size_t nq, std::size_t dim,
                          std::size_t k) {
    std::size_t hit = 0;
    std::size_t total = 0;
    for (std::size_t qi = 0; qi < nq; ++qi) {
        const float* q = queries + qi * dim;
        auto truth_nb = truth.search(q, dim, k);
        auto got_nb = approx.search(q, dim, k);

        std::unordered_set<std::uint64_t> got_ids;
        for (const vectorsearch::Neighbor& n : got_nb) {
            got_ids.insert(n.id);
        }
        for (const vectorsearch::Neighbor& t : truth_nb) {
            if (got_ids.count(t.id)) {
                ++hit;
            }
            ++total;
        }
    }
    return total == 0 ? 0.0 : static_cast<double>(hit) / static_cast<double>(total);
}

}  // namespace test
