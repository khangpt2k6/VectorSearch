#include "vectorsearch/flat_index.hpp"

#include <cstdint>
#include <vector>

#include "check.hpp"
#include "utils/add_wrong_dim.hpp"
#include "vectorsearch/distance.hpp"

using namespace vectorsearch;

static void search_returns_closest_first() {
    FlatIndex idx(3, Metric::L2);
    idx.add(1, std::vector<float>{0.0f, 0.0f, 0.0f});
    idx.add(2, std::vector<float>{1.0f, 0.0f, 0.0f});
    idx.add(3, std::vector<float>{5.0f, 5.0f, 5.0f});

    auto hits = idx.search(std::vector<float>{0.1f, 0.0f, 0.0f}, 2);
    CHECK(hits.size() == 2);
    CHECK(hits[0].id == 1);
    CHECK(hits[1].id == 2);
    CHECK(hits[0].distance <= hits[1].distance);
}

static void search_caps_at_index_size() {
    FlatIndex idx(2, Metric::L2);
    idx.add(1, std::vector<float>{0.0f, 0.0f});
    CHECK(idx.search(std::vector<float>{0.0f, 0.0f}, 10).size() == 1);
}

static void search_empty_and_bad_dim() {
    FlatIndex idx(3, Metric::L2);
    CHECK(idx.search(std::vector<float>{0.0f, 0.0f, 0.0f}, 5).empty());
    idx.add(1, std::vector<float>{0.0f, 0.0f, 0.0f});
    // Wrong-dimension query returns empty rather than throwing.
    CHECK(idx.search(std::vector<float>{0.0f, 0.0f}, 1).empty());
}

// The expected answer differs by metric on purpose. The query sits next to
// id 20, so L2 and cosine rank it first. Dot product is not normalized, so the
// large-magnitude vector id 50 wins on raw dot. That is intended behavior.
static void true_nearest_neighbor_per_metric() {
    struct Case {
        Metric metric;
        std::uint64_t expected_first;
    };
    const Case cases[] = {
        {Metric::L2, 20},
        {Metric::Cosine, 20},
        {Metric::Dot, 50},
    };

    for (const Case& c : cases) {
        FlatIndex idx(3, c.metric);
        idx.add(10, std::vector<float>{0.0f, 0.0f, 0.0f});
        idx.add(20, std::vector<float>{1.0f, 0.0f, 0.0f});
        idx.add(30, std::vector<float>{0.0f, 1.0f, 0.0f});
        idx.add(40, std::vector<float>{0.0f, 0.0f, 1.0f});
        idx.add(50, std::vector<float>{10.0f, 10.0f, 10.0f});
        CHECK(idx.size() == 5);

        auto hits = idx.search(std::vector<float>{0.95f, 0.02f, 0.0f}, 3);
        CHECK(hits[0].id == c.expected_first);
        for (std::size_t i = 1; i < hits.size(); ++i) {
            CHECK(hits[i - 1].distance <= hits[i].distance);
        }
    }
}

static void stored_vector_is_its_own_nearest() {
    FlatIndex idx(4, Metric::Cosine);
    std::vector<std::pair<std::uint64_t, std::vector<float>>> vectors = {
        {1, {1.0f, 2.0f, 3.0f, 4.0f}},
        {2, {-1.0f, 0.5f, 2.0f, 0.0f}},
        {3, {3.0f, 3.0f, 0.0f, 1.0f}},
    };
    for (auto& kv : vectors) idx.add(kv.first, kv.second);
    for (auto& kv : vectors) {
        auto hits = idx.search(kv.second, 1);
        CHECK(hits[0].id == kv.first);
    }
}

int main() {
    {
        FlatIndex idx(3, Metric::L2);
        test::check_add_rejects_wrong_dimension(idx);
    }
    search_returns_closest_first();
    search_caps_at_index_size();
    search_empty_and_bad_dim();
    true_nearest_neighbor_per_metric();
    stored_vector_is_its_own_nearest();

    if (test::failures() == 0) {
        std::printf("test_flat: all checks passed\n");
    }
    return test::failures() == 0 ? 0 : 1;
}
