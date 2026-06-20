#pragma once

// HNSW index (Hierarchical Navigable Small World).
//
// A multi-layer proximity graph. Upper layers are sparse and let a query take
// long hops across the space; lower layers are dense and refine the result. The
// bottom layer (0) holds every vector. Search descends from the top entry point
// to layer 0, and insertion wires each new node into the graph layer by layer.
//
// This header is the shared structure both halves of the build agree on: the
// node + layer layout, the tunable params, and the introspection hooks the
// tests use. The insert path (level assignment, neighbor heuristic, parallel
// build) lives here. The query-side ef_search beam is built on top of the same
// search_layer routine.

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <vector>

#include "vectorsearch/distance.hpp"
#include "vectorsearch/index.hpp"

namespace vectorsearch {

class HnswIndex : public Index {
public:
    struct Params {
        // neighbors kept per node on the upper layers. layer 0 keeps up to 2*M.
        std::size_t M = 16;
        // candidate pool size while building. bigger = better graph, slower add.
        std::size_t ef_construction = 200;
        // candidate pool size at query time. bigger = better recall, slower.
        std::size_t ef_search = 50;
        // seed for the level-assignment RNG so builds are reproducible.
        std::uint64_t seed = 42;
    };

    HnswIndex(std::size_t dim, Metric metric);
    HnswIndex(std::size_t dim, Metric metric, Params params);

    void add(std::uint64_t id, const float* vector, std::size_t dim) override;
    std::vector<Neighbor> search(const float* query, std::size_t dim,
                                 std::size_t k) const override;
    std::size_t size() const override;

    // Insert many vectors using num_threads worker threads. ids and vectors are
    // parallel arrays; vectors is packed end to end (count * dim floats). Passing
    // num_threads = 0 picks a sensible default from the hardware.
    void add_parallel(const std::uint64_t* ids, const float* vectors,
                      std::size_t count, unsigned num_threads = 0);

    std::size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }
    const Params& params() const { return params_; }

    // Convenience overload for std::vector callers.
    void add(std::uint64_t id, const std::vector<float>& vector) {
        add(id, vector.data(), vector.size());
    }

    // --- introspection, used by the insert-path tests ---------------------
    // internal index of the top entry point, or kNoEntry when the index is empty.
    static constexpr std::size_t kNoEntry = static_cast<std::size_t>(-1);
    std::size_t entry_point() const { return entry_point_; }
    int max_level() const { return max_level_; }
    int node_level(std::size_t internal_idx) const {
        return nodes_[internal_idx].level;
    }
    std::uint64_t id_at(std::size_t internal_idx) const {
        return nodes_[internal_idx].id;
    }
    const std::vector<std::uint32_t>& neighbors_at(std::size_t internal_idx,
                                                   int layer) const {
        return nodes_[internal_idx].links[layer];
    }

private:
    // One vertex of the graph. Its vector lives in data_ at the same index.
    struct Node {
        std::uint64_t id;
        int level;  // top layer this node appears on (0-based)
        // links[l] = internal indices of this node's neighbors on layer l,
        // for l in [0, level]. layer 0 is always present.
        std::vector<std::vector<std::uint32_t>> links;
    };

    const float* vector_at(std::size_t i) const {
        return data_.data() + i * dim_;
    }

    // draw a random top level from the exponential distribution HNSW uses.
    int random_level();

    // greedy beam search of a single layer: returns up to ef candidates closest
    // to query, as (distance, internal_idx) pairs. shared by insert and query.
    std::vector<std::pair<float, std::uint32_t>> search_layer(
        const float* query, std::uint32_t entry, std::size_t ef,
        int layer) const;

    // pick at most m neighbors out of candidates using the HNSW heuristic: keep
    // a candidate only if it is closer to the new node than to anything already
    // kept, which spreads links out instead of clustering them. candidates carry
    // their distance to the query node in .first.
    std::vector<std::uint32_t> select_neighbors(
        std::vector<std::pair<float, std::uint32_t>> candidates,
        std::size_t m) const;

    // wire one node (already stored in nodes_/data_) into the graph.
    void link_node(std::uint32_t new_idx);

    // link accessors that take the per-node lock when a parallel build is in
    // flight, and skip locking otherwise.
    std::vector<std::uint32_t> read_links(std::uint32_t idx, int layer) const;
    void store_links(std::uint32_t idx, int layer,
                     const std::vector<std::uint32_t>& links);
    // add `to` to `from`'s layer links, pruning back to cap if it overflows.
    void connect(std::uint32_t from, int layer, std::uint32_t to,
                 std::size_t cap);

    std::size_t dim_;
    Metric metric_;
    Params params_;
    double level_mult_;  // 1 / ln(M), the level distribution scale

    std::vector<float> data_;   // all vectors packed end to end
    std::vector<Node> nodes_;   // one per stored vector, indexed the same way

    std::size_t entry_point_ = kNoEntry;
    int max_level_ = -1;

    std::mt19937_64 rng_;
    bool concurrent_ = false;  // true only while add_parallel is running
    // per-node locks for the parallel build; index with internal_idx.
    mutable std::vector<std::mutex> link_locks_;
};

}  // namespace vectorsearch
