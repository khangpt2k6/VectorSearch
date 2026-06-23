#include "vectorsearch/scalar_quantized_index.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace vectorsearch {

namespace {
using DistIdx = std::pair<float, std::uint32_t>;
bool by_distance(const DistIdx& a, const DistIdx& b) {
    return a.first < b.first;
}
}  // namespace

ScalarQuantizedIndex::ScalarQuantizedIndex(std::size_t dim, Metric metric)
    : ScalarQuantizedIndex(dim, metric, Options()) {}

ScalarQuantizedIndex::ScalarQuantizedIndex(std::size_t dim, Metric metric,
                                           Options options)
    : dim_(dim), metric_(metric), options_(options) {}

void ScalarQuantizedIndex::train(const float* vectors, std::size_t count) {
    quantizer_.train(vectors, count, dim_);
    // re-training invalidates any codes built against the old bounds.
    codes_.clear();
    ids_.clear();
    originals_.clear();
}

void ScalarQuantizedIndex::add(std::uint64_t id, const float* vector,
                               std::size_t dim) {
    if (dim != dim_) {
        throw std::invalid_argument("ScalarQuantizedIndex::add dimension mismatch");
    }
    if (!quantizer_.trained()) {
        throw std::logic_error("ScalarQuantizedIndex::add before train()");
    }
    const std::size_t base = codes_.size();
    codes_.resize(base + dim_);
    quantizer_.encode(vector, codes_.data() + base);
    ids_.push_back(id);
    if (options_.rerank) {
        originals_.insert(originals_.end(), vector, vector + dim_);
    }
}

std::vector<Neighbor> ScalarQuantizedIndex::search(const float* query,
                                                   std::size_t dim,
                                                   std::size_t k) const {
    if (dim != dim_ || ids_.empty() || k == 0) {
        return {};
    }
    const std::size_t n = ids_.size();

    // first pass: rank everything by quantized (decode-on-the-fly) distance.
    std::vector<DistIdx> scored(n);
    std::vector<float> recon(dim_);
    for (std::uint32_t i = 0; i < n; ++i) {
        quantizer_.decode(code_at(i), recon.data());
        scored[i] = {distance(metric_, query, recon.data(), dim_), i};
    }

    const bool do_rerank = options_.rerank && !originals_.empty();
    const std::size_t pool =
        do_rerank ? std::min(n, k * options_.rerank_factor) : std::min(n, k);
    std::partial_sort(scored.begin(), scored.begin() + pool, scored.end(),
                      by_distance);

    std::vector<Neighbor> out;
    if (do_rerank) {
        // refine the shortlist with exact distances on the original vectors.
        std::vector<DistIdx> refined(pool);
        for (std::size_t j = 0; j < pool; ++j) {
            const std::uint32_t i = scored[j].second;
            refined[j] = {distance(metric_, query, original_at(i), dim_), i};
        }
        const std::size_t kk = std::min(k, pool);
        std::partial_sort(refined.begin(), refined.begin() + kk, refined.end(),
                          by_distance);
        out.reserve(kk);
        for (std::size_t j = 0; j < kk; ++j) {
            out.push_back({ids_[refined[j].second], refined[j].first});
        }
    } else {
        out.reserve(pool);
        for (std::size_t j = 0; j < pool; ++j) {
            out.push_back({ids_[scored[j].second], scored[j].first});
        }
    }
    return out;
}

std::size_t ScalarQuantizedIndex::size() const { return ids_.size(); }

std::size_t ScalarQuantizedIndex::memory_bytes() const {
    return codes_.size() * sizeof(std::uint8_t) +
           ids_.size() * sizeof(std::uint64_t) +
           originals_.size() * sizeof(float);
}

std::size_t ScalarQuantizedIndex::raw_f32_bytes() const {
    return ids_.size() * dim_ * sizeof(float) +
           ids_.size() * sizeof(std::uint64_t);
}

}  // namespace vectorsearch
