#pragma once

// The behavior shared by every index.
//
// Every index ranks results by distance ascending (smaller is closer). The flat
// index is exact and acts as the recall baseline for the approximate indexes
// that come later (HNSW, IVF).

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectorsearch {

// A single search result: the stored id and its distance to the query.
struct Neighbor {
    std::uint64_t id;
    float distance;
};

class Index {
public:
    virtual ~Index() = default;

    // Add a vector under the given id. Throws std::invalid_argument if the
    // dimension does not match the index.
    virtual void add(std::uint64_t id, const float* vector, std::size_t dim) = 0;

    // Return the k nearest neighbors to query, sorted closest first. Returns
    // fewer than k results if the index holds fewer vectors, and an empty
    // result if the query dimension is wrong (so a bad query never throws in a
    // hot search loop).
    virtual std::vector<Neighbor> search(const float* query, std::size_t dim,
                                         std::size_t k) const = 0;

    // Number of vectors currently stored.
    virtual std::size_t size() const = 0;

    bool empty() const { return size() == 0; }
};

}  // namespace vectorsearch
