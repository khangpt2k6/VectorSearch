// recall@10 vs QPS sweep over ef_search.
//
// Builds an HNSW index and a flat (exact) ground-truth index on the same random
// data, then sweeps the query-time ef_search beam width. For each ef it prints
// recall@10 (vs the flat index) and queries/sec, showing the recall/throughput
// trade-off the ef_search knob controls. Synthetic data, so the absolute numbers
// are illustrative only — real benchmarking lives in the Phase 4 harness.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <unordered_set>
#include <vector>

#include "vectorsearch/flat_index.hpp"
#include "vectorsearch/hnsw_index.hpp"

using namespace vectorsearch;

static std::vector<float> random_data(std::size_t count, std::size_t dim,
                                      std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(count * dim);
    for (float& x : out) x = dist(rng);
    return out;
}

int main() {
    const std::size_t dim = 64;
    const std::size_t n = 5000;
    const std::size_t nq = 500;
    const std::size_t k = 10;

    HnswIndex hnsw(dim, Metric::L2);
    FlatIndex flat(dim, Metric::L2);
    auto data = random_data(n, dim, 2024);
    for (std::size_t i = 0; i < n; ++i) {
        hnsw.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
        flat.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
    }

    // ground truth top-k from the exact index, computed once.
    auto queries = random_data(nq, dim, 99);
    std::vector<std::unordered_set<std::uint64_t>> truth(nq);
    for (std::size_t qi = 0; qi < nq; ++qi) {
        auto t = flat.search(queries.data() + qi * dim, dim, k);
        for (const Neighbor& nb : t) truth[qi].insert(nb.id);
    }

    std::printf("n=%zu dim=%zu queries=%zu k=%zu\n", n, dim, nq, k);
    std::printf("%6s | %9s | %12s\n", "ef", "recall@10", "QPS");
    std::printf("-------+-----------+-------------\n");

    for (std::size_t ef : {10u, 20u, 40u, 80u, 160u}) {
        std::size_t hits = 0;
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t qi = 0; qi < nq; ++qi) {
            auto got = hnsw.search(queries.data() + qi * dim, dim, k, ef);
            for (const Neighbor& g : got) {
                if (truth[qi].count(g.id)) ++hits;
            }
        }
        const auto end = std::chrono::steady_clock::now();
        const double secs =
            std::chrono::duration<double>(end - start).count();
        const double recall = static_cast<double>(hits) / (nq * k);
        const double qps = secs > 0.0 ? nq / secs : 0.0;
        std::printf("%6zu | %9.4f | %12.0f\n", ef, recall, qps);
    }
    return 0;
}
