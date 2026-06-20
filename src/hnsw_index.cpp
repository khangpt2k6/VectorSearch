#include "vectorsearch/hnsw_index.hpp"

#include <cmath>
#include <stdexcept>

namespace vectorsearch {

HnswIndex::HnswIndex(std::size_t dim, Metric metric)
    : HnswIndex(dim, metric, Params()) {}

HnswIndex::HnswIndex(std::size_t dim, Metric metric, Params params)
    : dim_(dim),
      metric_(metric),
      params_(params),
      level_mult_(1.0 / std::log(static_cast<double>(params.M))),
      rng_(params.seed) {}

std::size_t HnswIndex::size() const { return nodes_.size(); }

void HnswIndex::add(std::uint64_t id, const float* vector, std::size_t dim) {
    if (dim != dim_) {
        throw std::invalid_argument("HnswIndex::add dimension mismatch");
    }
    // stub: store the node but do not wire any graph edges yet.
    const auto new_idx = static_cast<std::uint32_t>(nodes_.size());
    data_.insert(data_.end(), vector, vector + dim_);
    Node node;
    node.id = id;
    node.level = 0;
    node.links.resize(1);
    nodes_.push_back(std::move(node));
    if (entry_point_ == kNoEntry) {
        entry_point_ = new_idx;
        max_level_ = 0;
    }
}

void HnswIndex::add_parallel(const std::uint64_t* ids, const float* vectors,
                             std::size_t count, unsigned) {
    for (std::size_t i = 0; i < count; ++i) {
        add(ids[i], vectors + i * dim_, dim_);
    }
}

std::vector<Neighbor> HnswIndex::search(const float*, std::size_t,
                                        std::size_t) const {
    return {};  // stub
}

}  // namespace vectorsearch
