#pragma once

// Flat (brute force) index.
//
// Stores every vector and scans all of them on each query. This is O(n * dim)
// per search, so it does not scale, but it is always exact. That makes it the
// ground truth that approximate indexes are measured against, and a perfectly
// reasonable choice for small collections.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vectorsearch/distance.hpp"
#include "vectorsearch/index.hpp"

namespace vectorsearch {

class FlatIndex : public Index {
public:
    FlatIndex(std::size_t dim, Metric metric);

    void add(std::uint64_t id, const float* vector, std::size_t dim) override;
    std::vector<Neighbor> search(const float* query, std::size_t dim,
                                 std::size_t k) const override;
    std::size_t size() const override;

    std::size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }

    // Convenience overloads for std::vector callers.
    void add(std::uint64_t id, const std::vector<float>& vector) {
        add(id, vector.data(), vector.size());
    }
    std::vector<Neighbor> search(const std::vector<float>& query,
                                 std::size_t k) const {
        return search(query.data(), query.size(), k);
    }

private:
    const float* vector_at(std::size_t i) const {
        return data_.data() + i * dim_;
    }

    std::size_t dim_;
    Metric metric_;
    // All vectors packed end to end: vector i is data_[i*dim .. (i+1)*dim].
    std::vector<float> data_;
    std::vector<std::uint64_t> ids_;
};

}  // namespace vectorsearch
