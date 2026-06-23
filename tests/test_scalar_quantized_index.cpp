#include "vectorsearch/scalar_quantized_index.hpp"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "check.hpp"
#include "utils/add_wrong_dim.hpp"
#include "utils/random_data.hpp"
#include "utils/recall.hpp"
#include "vectorsearch/distance.hpp"
#include "vectorsearch/flat_index.hpp"

using namespace vectorsearch;

static void add_before_train_throws() {
    ScalarQuantizedIndex idx(4, Metric::L2);
    bool threw = false;
    try {
        idx.add(1, std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f});
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw);
}

static void grows_and_searches_sorted() {
    const std::size_t dim = 8;
    ScalarQuantizedIndex idx(dim, Metric::L2);
    auto data = test::random_data(100, dim, 7);
    idx.train(data);
    for (std::size_t i = 0; i < 100; ++i) {
        idx.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
    }
    CHECK(idx.size() == 100);

    auto hits = idx.search(data.data(), dim, 5);
    CHECK(hits.size() == 5);
    for (std::size_t i = 1; i < hits.size(); ++i) {
        CHECK(hits[i - 1].distance <= hits[i].distance);
    }
}

// codes-only storage must be at least 3x smaller than the equivalent f32 store.
static void codes_only_cuts_memory() {
    const std::size_t dim = 128;
    const std::size_t n = 500;
    ScalarQuantizedIndex::Options opt;
    opt.rerank = false;  // codes only, no kept originals
    ScalarQuantizedIndex idx(dim, Metric::L2, opt);
    auto data = test::random_data(n, dim, 3);
    idx.train(data);
    for (std::size_t i = 0; i < n; ++i) {
        idx.add(static_cast<std::uint64_t>(i), data.data() + i * dim, dim);
    }
    CHECK(idx.memory_bytes() * 3 < idx.raw_f32_bytes());
    std::printf("  SQ codes-only memory: %zu bytes vs f32 %zu bytes (%.1fx)\n",
                idx.memory_bytes(), idx.raw_f32_bytes(),
                static_cast<double>(idx.raw_f32_bytes()) /
                    static_cast<double>(idx.memory_bytes()));
}

// the re-rank pass must (a) never lower recall versus codes-only and (b) return
// exact distances for what it ranks. measured over many queries on shared data.
static void rerank_recovers_recall() {
    const std::size_t dim = 32;
    const std::size_t n = 1000;
    const std::size_t nq = 100;
    const std::size_t k = 10;
    auto data = test::random_data(n, dim, 2024);
    auto queries = test::random_data(nq, dim, 99);

    FlatIndex truth(dim, Metric::L2);
    ScalarQuantizedIndex::Options codes_opt;
    codes_opt.rerank = false;
    ScalarQuantizedIndex codes_only(dim, Metric::L2, codes_opt);
    ScalarQuantizedIndex reranked(dim, Metric::L2);  // rerank on by default

    codes_only.train(data);
    reranked.train(data);
    for (std::size_t i = 0; i < n; ++i) {
        const float* v = data.data() + i * dim;
        truth.add(static_cast<std::uint64_t>(i), v, dim);
        codes_only.add(static_cast<std::uint64_t>(i), v, dim);
        reranked.add(static_cast<std::uint64_t>(i), v, dim);
    }

    const double r_codes =
        test::recall_at_k(codes_only, truth, queries.data(), nq, dim, k);
    const double r_rerank =
        test::recall_at_k(reranked, truth, queries.data(), nq, dim, k);
    std::printf("  recall@%zu  codes-only: %.3f   re-ranked: %.3f\n", k, r_codes,
                r_rerank);

    CHECK(r_codes >= 0.5);          // quantization alone is already useful
    CHECK(r_rerank >= r_codes);     // re-rank never hurts
    CHECK(r_rerank >= 0.95);        // and recovers near-exact recall

    // a re-ranked hit must carry the true exact distance, not the quantized one.
    const float* query = queries.data();
    auto exact = truth.search(query, dim, k);
    auto got = reranked.search(query, dim, k);
    CHECK(got.size() == exact.size());
    CHECK(got[0].id == exact[0].id);
    CHECK(std::fabs(got[0].distance - exact[0].distance) < 1e-4f);
}

int main() {
    add_before_train_throws();
    {
        ScalarQuantizedIndex idx(3, Metric::L2);
        idx.train(test::random_data(20, 3, 1));
        test::check_add_rejects_wrong_dimension(idx);
    }
    grows_and_searches_sorted();
    codes_only_cuts_memory();
    rerank_recovers_recall();

    if (test::failures() == 0) {
        std::printf("test_scalar_quantized_index: all checks passed\n");
    }
    return test::failures() == 0 ? 0 : 1;
}
