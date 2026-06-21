#include "vectorsearch/hnsw_index.hpp"

#include <cstdint>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "check.hpp"
#include "vectorsearch/distance.hpp"
#include "vectorsearch/flat_index.hpp"

using namespace vectorsearch;

// deterministic random vectors so every run sees the same graph.
static std::vector<float> random_data(std::size_t count, std::size_t dim,
                                      std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(count * dim);
    for (float& x : out) x = dist(rng);
    return out;
}

static void add_grows_size() {
    HnswIndex idx(4, Metric::L2);
    CHECK(idx.empty());
    idx.add(7, std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});
    idx.add(8, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
    CHECK(idx.size() == 2);
    CHECK(idx.entry_point() != HnswIndex::kNoEntry);
}

static void add_rejects_wrong_dimension() {
    HnswIndex idx(3, Metric::L2);
    bool threw = false;
    try {
        idx.add(1, std::vector<float>{0.0f, 0.0f});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
    CHECK(idx.empty());
}

// every node keeps at most 2*M neighbors on layer 0 and M on the upper layers,
// never points at itself, and never lists the same neighbor twice.
static void link_invariants_hold() {
    const std::size_t dim = 16;
    const std::size_t n = 300;
    HnswIndex::Params p;
    p.M = 8;
    HnswIndex idx(dim, Metric::L2, p);
    auto data = random_data(n, dim, 1234);
    for (std::size_t i = 0; i < n; ++i) {
        idx.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
    }
    CHECK(idx.size() == n);

    for (std::size_t i = 0; i < n; ++i) {
        const int top = idx.node_level(i);
        for (int l = 0; l <= top; ++l) {
            const auto& nbrs = idx.neighbors_at(i, l);
            const std::size_t cap = (l == 0) ? 2 * p.M : p.M;
            CHECK(nbrs.size() <= cap);

            std::unordered_set<std::uint32_t> seen;
            for (std::uint32_t nb : nbrs) {
                CHECK(nb != i);             // no self loop
                CHECK(seen.insert(nb).second);  // no duplicate
            }
        }
    }
}

// the layer-0 graph should be one connected component: starting from the entry
// point and walking neighbors must reach every node.
static void graph_is_connected() {
    const std::size_t dim = 12;
    const std::size_t n = 250;
    HnswIndex idx(dim, Metric::L2);
    auto data = random_data(n, dim, 99);
    for (std::size_t i = 0; i < n; ++i) {
        idx.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
    }

    std::vector<bool> seen(n, false);
    std::queue<std::uint32_t> q;
    const auto start = static_cast<std::uint32_t>(idx.entry_point());
    seen[start] = true;
    q.push(start);
    std::size_t reached = 0;
    while (!q.empty()) {
        const std::uint32_t cur = q.front();
        q.pop();
        ++reached;
        for (std::uint32_t nb : idx.neighbors_at(cur, 0)) {
            if (!seen[nb]) {
                seen[nb] = true;
                q.push(nb);
            }
        }
    }
    CHECK(reached == n);
}

// the true nearest neighbor (from the exact flat index) should show up in the
// HNSW result set. stays valid whether search() is exact or approximate.
static void search_contains_true_nearest() {
    const std::size_t dim = 20;
    const std::size_t n = 400;
    HnswIndex hnsw(dim, Metric::L2);
    FlatIndex flat(dim, Metric::L2);
    auto data = random_data(n, dim, 2025);
    for (std::size_t i = 0; i < n; ++i) {
        hnsw.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
        flat.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
    }

    auto queries = random_data(10, dim, 7);
    for (std::size_t qi = 0; qi < 10; ++qi) {
        const float* q = queries.data() + qi * dim;
        auto truth = flat.search(q, dim, 1);
        auto hits = hnsw.search(q, dim, 10);

        CHECK(hits.size() == 10);
        for (std::size_t i = 1; i < hits.size(); ++i) {
            CHECK(hits[i - 1].distance <= hits[i].distance);  // sorted ascending
        }
        bool found = false;
        for (const Neighbor& h : hits) {
            if (h.id == truth[0].id) found = true;
        }
        CHECK(found);
    }
}

// recall@10 against the exact flat index: for each query, the fraction of the
// true top-10 ids that the HNSW result also returns, averaged over queries.
// default params should clear a conservative bar on random data.
static void recall_at_10_vs_flat() {
    const std::size_t dim = 32;
    const std::size_t n = 2000;
    const std::size_t nq = 100;
    const std::size_t k = 10;
    HnswIndex hnsw(dim, Metric::L2);
    FlatIndex flat(dim, Metric::L2);
    auto data = random_data(n, dim, 4242);
    for (std::size_t i = 0; i < n; ++i) {
        hnsw.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
        flat.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
    }

    auto queries = random_data(nq, dim, 13);
    std::size_t hits_total = 0;
    for (std::size_t qi = 0; qi < nq; ++qi) {
        const float* q = queries.data() + qi * dim;
        auto truth = flat.search(q, dim, k);
        auto got = hnsw.search(q, dim, k);

        std::unordered_set<std::uint64_t> truth_ids;
        for (const Neighbor& t : truth) truth_ids.insert(t.id);
        for (const Neighbor& g : got) {
            if (truth_ids.count(g.id)) ++hits_total;
        }
    }

    const double recall = static_cast<double>(hits_total) / (nq * k);
    std::printf("test_hnsw: recall@10 = %.4f\n", recall);
    CHECK(recall >= 0.90);
}

// a parallel build must produce the same node count and a valid graph: no self
// loops, layer-0 degree within bounds, and no isolated nodes.
static void parallel_build_is_valid() {
    const std::size_t dim = 16;
    const std::size_t n = 600;
    HnswIndex::Params p;
    p.M = 8;
    HnswIndex idx(dim, Metric::L2, p);
    auto data = random_data(n, dim, 555);
    std::vector<std::uint64_t> ids(n);
    for (std::size_t i = 0; i < n; ++i) ids[i] = static_cast<std::uint64_t>(i);

    idx.add_parallel(ids.data(), data.data(), n, 4);
    CHECK(idx.size() == n);

    for (std::size_t i = 0; i < n; ++i) {
        const auto& nbrs = idx.neighbors_at(i, 0);
        CHECK(nbrs.size() <= 2 * p.M);
        CHECK(!nbrs.empty());  // no isolated node once the graph has many nodes
        for (std::uint32_t nb : nbrs) CHECK(nb != i);
    }
}

int main() {
    add_grows_size();
    add_rejects_wrong_dimension();
    link_invariants_hold();
    graph_is_connected();
    search_contains_true_nearest();
    recall_at_10_vs_flat();
    parallel_build_is_valid();

    if (test::failures() == 0) {
        std::printf("test_hnsw: all checks passed\n");
    }
    return test::failures() == 0 ? 0 : 1;
}
