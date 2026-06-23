#include "vectorsearch/hnsw_index.hpp"

#include <cstdint>
#include <queue>
#include <unordered_set>
#include <vector>

#include "check.hpp"
#include "utils/add_wrong_dim.hpp"
#include "utils/random_data.hpp"
#include "vectorsearch/distance.hpp"

using namespace vectorsearch;

static void add_grows_size() {
    HnswIndex idx(4, Metric::L2);
    CHECK(idx.empty());
    idx.add(7, std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});
    idx.add(8, std::vector<float>{1.0f, 0.0f, 0.0f, 0.0f});
    CHECK(idx.size() == 2);
    CHECK(idx.entry_point() != HnswIndex::kNoEntry);
}

// every node keeps at most 2*M neighbors on layer 0 and M on the upper layers,
// never points at itself, and never lists the same neighbor twice.
static void link_invariants_hold() {
    const std::size_t dim = 16;
    const std::size_t n = 300;
    HnswIndex::Params p;
    p.M = 8;
    HnswIndex idx(dim, Metric::L2, p);
    auto data = test::random_data(n, dim, 1234);
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
    auto data = test::random_data(n, dim, 99);
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

// a parallel build must produce the same node count and a valid graph: no self
// loops, layer-0 degree within bounds, and no isolated nodes.
static void parallel_build_is_valid() {
    const std::size_t dim = 16;
    const std::size_t n = 600;
    HnswIndex::Params p;
    p.M = 8;
    HnswIndex idx(dim, Metric::L2, p);
    auto data = test::random_data(n, dim, 555);
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
    {
        HnswIndex idx(3, Metric::L2);
        test::check_add_rejects_wrong_dimension(idx);
    }
    link_invariants_hold();
    graph_is_connected();
    parallel_build_is_valid();

    if (test::failures() == 0) {
        std::printf("test_hnsw: all checks passed\n");
    }
    return test::failures() == 0 ? 0 : 1;
}
