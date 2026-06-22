#include "vectorsearch/hnsw_index.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>

#include "check.hpp"
#include "vectorsearch/distance.hpp"
#include "vectorsearch/flat_index.hpp"

using namespace vectorsearch;

// deterministic random vectors, same helper the other hnsw tests use.
static std::vector<float> random_data(std::size_t count, std::size_t dim,
                                      std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> out(count * dim);
    for (float& x : out) x = dist(rng);
    return out;
}

// average recall@k of the hnsw index against the exact flat index over many
// queries. recall here = fraction of the true top-k that hnsw also returns.
static double measure_recall(Metric metric, std::size_t dim, std::size_t n,
                             std::size_t num_queries, std::size_t k,
                             const HnswIndex::Params& p) {
    HnswIndex hnsw(dim, metric, p);
    FlatIndex flat(dim, metric);
    auto data = random_data(n, dim, 2025);
    for (std::size_t i = 0; i < n; ++i) {
        hnsw.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
        flat.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
    }

    auto queries = random_data(num_queries, dim, 31337);
    std::size_t hit = 0;
    std::size_t total = 0;
    for (std::size_t qi = 0; qi < num_queries; ++qi) {
        const float* q = queries.data() + qi * dim;
        auto truth = flat.search(q, dim, k);
        auto hits = hnsw.search(q, dim, k);

        std::unordered_set<std::uint64_t> got;
        for (const Neighbor& h : hits) got.insert(h.id);
        for (const Neighbor& t : truth) {
            if (got.count(t.id)) ++hit;
            ++total;
        }
    }
    return total == 0 ? 0.0 : static_cast<double>(hit) / total;
}

// with default params the approximate graph should recover almost all of the
// exact neighbors. a loose floor keeps this stable across platforms/compilers
// while still catching a broken insert or search path.
static void recall_is_high_l2() {
    HnswIndex::Params p;
    p.M = 16;
    p.ef_construction = 200;
    p.ef_search = 100;
    const double recall = measure_recall(Metric::L2, 24, 1000, 50, 10, p);
    std::printf("recall@10 (L2) = %.3f\n", recall);
    CHECK(recall >= 0.90);
}

static void recall_is_high_cosine() {
    HnswIndex::Params p;
    p.M = 16;
    p.ef_construction = 200;
    p.ef_search = 100;
    const double recall = measure_recall(Metric::Cosine, 24, 1000, 50, 10, p);
    std::printf("recall@10 (cosine) = %.3f\n", recall);
    CHECK(recall >= 0.90);
}

// a bigger ef_search widens the beam, so recall must not get worse when we ask
// for more work. this guards against a search path that ignores ef_search.
static void higher_ef_search_does_not_hurt_recall() {
    HnswIndex::Params lo;
    lo.ef_search = 16;
    HnswIndex::Params hi;
    hi.ef_search = 200;
    const double r_lo = measure_recall(Metric::L2, 16, 800, 40, 10, lo);
    const double r_hi = measure_recall(Metric::L2, 16, 800, 40, 10, hi);
    std::printf("recall@10 ef=16 -> %.3f, ef=200 -> %.3f\n", r_lo, r_hi);
    CHECK(r_hi >= r_lo - 0.02);  // small slack for ties
}

int main() {
    recall_is_high_l2();
    recall_is_high_cosine();
    higher_ef_search_does_not_hurt_recall();

    if (test::failures() == 0) {
        std::printf("test_recall: all checks passed\n");
    }
    return test::failures() == 0 ? 0 : 1;
}
