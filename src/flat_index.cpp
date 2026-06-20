#include "vectorsearch/flat_index.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace vectorsearch {

FlatIndex::FlatIndex(std::size_t dim, Metric metric)
    : dim_(dim), metric_(metric) {}

void FlatIndex::add(std::uint64_t id, const float* vector, std::size_t dim) {
    if (dim != dim_) {
        throw std::invalid_argument(
            "dimension mismatch: index expects " + std::to_string(dim_) +
            ", got " + std::to_string(dim));
    }
    data_.insert(data_.end(), vector, vector + dim);
    ids_.push_back(id);
}

std::vector<Neighbor> FlatIndex::search(const float* query, std::size_t dim,
                                        std::size_t k) const {
    if (dim != dim_ || k == 0 || ids_.empty()) {
        return {};
    }

    std::vector<Neighbor> scored;
    scored.reserve(ids_.size());
    for (std::size_t i = 0; i < ids_.size(); ++i) {
        scored.push_back(
            Neighbor{ids_[i], distance(metric_, query, vector_at(i), dim_)});
    }

    const std::size_t kk = std::min(k, scored.size());
    // Partial sort: we only need the kk smallest, already in order.
    std::partial_sort(
        scored.begin(), scored.begin() + kk, scored.end(),
        [](const Neighbor& a, const Neighbor& b) {
            return a.distance < b.distance;
        });
    scored.resize(kk);
    return scored;
}

std::size_t FlatIndex::size() const { return ids_.size(); }

}  // namespace vectorsearch
