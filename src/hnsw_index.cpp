#include "vectorsearch/hnsw_index.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <queue>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace vectorsearch {

namespace {
// (distance, internal_idx) ordered so the smallest distance compares "least".
using DistIdx = std::pair<float, std::uint32_t>;

// min-heap by distance: nearest on top.
struct Nearest {
    bool operator()(const DistIdx& a, const DistIdx& b) const {
        return a.first > b.first;
    }
};
// max-heap by distance: farthest on top.
struct Farthest {
    bool operator()(const DistIdx& a, const DistIdx& b) const {
        return a.first < b.first;
    }
};
}  // namespace

HnswIndex::HnswIndex(std::size_t dim, Metric metric)
    : HnswIndex(dim, metric, Params()) {}

HnswIndex::HnswIndex(std::size_t dim, Metric metric, Params params)
    : dim_(dim),
      metric_(metric),
      params_(params),
      level_mult_(1.0 / std::log(static_cast<double>(params.M))),
      rng_(params.seed) {}

std::size_t HnswIndex::size() const { return nodes_.size(); }

int HnswIndex::random_level() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const double r = std::max(dist(rng_), 1e-12);  // avoid log(0)
    return static_cast<int>(-std::log(r) * level_mult_);
}

std::vector<std::uint32_t> HnswIndex::read_links(std::uint32_t idx,
                                                 int layer) const {
    if (concurrent_) {
        std::lock_guard<std::mutex> g(link_locks_[idx]);
        return nodes_[idx].links[layer];
    }
    return nodes_[idx].links[layer];
}

void HnswIndex::store_links(std::uint32_t idx, int layer,
                            const std::vector<std::uint32_t>& links) {
    if (concurrent_) {
        std::lock_guard<std::mutex> g(link_locks_[idx]);
        nodes_[idx].links[layer] = links;
    } else {
        nodes_[idx].links[layer] = links;
    }
}

void HnswIndex::connect(std::uint32_t from, int layer, std::uint32_t to,
                        std::size_t cap) {
    std::unique_lock<std::mutex> lock;
    if (concurrent_) {
        lock = std::unique_lock<std::mutex>(link_locks_[from]);
    }
    auto& links = nodes_[from].links[layer];
    links.push_back(to);
    if (links.size() <= cap) {
        return;
    }
    // overflowed: re-run the heuristic over all of from's neighbors and keep the
    // best cap of them, so the densest nodes stay well spread.
    const float* base = vector_at(from);
    std::vector<DistIdx> candidates;
    candidates.reserve(links.size());
    for (std::uint32_t nb : links) {
        candidates.emplace_back(distance(metric_, base, vector_at(nb), dim_),
                                nb);
    }
    links = select_neighbors(std::move(candidates), cap);
}

std::vector<DistIdx> HnswIndex::search_layer(const float* query,
                                             std::uint32_t entry, std::size_t ef,
                                             int layer) const {
    std::unordered_set<std::uint32_t> visited;
    std::priority_queue<DistIdx, std::vector<DistIdx>, Nearest> candidates;
    std::priority_queue<DistIdx, std::vector<DistIdx>, Farthest> results;

    const float d0 = distance(metric_, query, vector_at(entry), dim_);
    candidates.push({d0, entry});
    results.push({d0, entry});
    visited.insert(entry);

    while (!candidates.empty()) {
        const DistIdx cur = candidates.top();
        candidates.pop();
        // everything left in the candidate heap is farther than the current
        // worst result, so the beam cannot improve: stop.
        if (cur.first > results.top().first) {
            break;
        }
        for (std::uint32_t nb : read_links(cur.second, layer)) {
            if (!visited.insert(nb).second) {
                continue;
            }
            const float d = distance(metric_, query, vector_at(nb), dim_);
            if (results.size() < ef || d < results.top().first) {
                candidates.push({d, nb});
                results.push({d, nb});
                if (results.size() > ef) {
                    results.pop();
                }
            }
        }
    }

    std::vector<DistIdx> out;
    out.reserve(results.size());
    while (!results.empty()) {
        out.push_back(results.top());
        results.pop();
    }
    return out;
}

std::vector<std::uint32_t> HnswIndex::select_neighbors(
    std::vector<DistIdx> candidates, std::size_t m) const {
    std::sort(candidates.begin(), candidates.end(),
              [](const DistIdx& a, const DistIdx& b) {
                  return a.first < b.first;
              });

    std::vector<std::uint32_t> picked;
    picked.reserve(m);
    for (const DistIdx& cand : candidates) {
        if (picked.size() >= m) {
            break;
        }
        // keep cand only if it is closer to the query than to any neighbor we
        // already picked. that drops candidates clustered behind a closer one.
        bool keep = true;
        const float* cand_vec = vector_at(cand.second);
        for (std::uint32_t r : picked) {
            if (distance(metric_, cand_vec, vector_at(r), dim_) < cand.first) {
                keep = false;
                break;
            }
        }
        if (keep) {
            picked.push_back(cand.second);
        }
    }
    return picked;
}

void HnswIndex::link_node(std::uint32_t new_idx) {
    const std::uint32_t entry = static_cast<std::uint32_t>(entry_point_);
    if (entry == new_idx) {
        return;  // first node ever; nothing to link to
    }
    const int top = max_level_;
    const int node_top = nodes_[new_idx].level;
    const float* q = vector_at(new_idx);

    std::uint32_t ep = entry;
    // descend the layers above the new node with a width-1 greedy walk to find
    // the best entry point for the layers we actually join.
    for (int lc = top; lc > node_top; --lc) {
        auto w = search_layer(q, ep, 1, lc);
        if (!w.empty()) {
            ep = std::min_element(w.begin(), w.end())->second;
        }
    }

    for (int lc = std::min(top, node_top); lc >= 0; --lc) {
        auto candidates = search_layer(q, ep, params_.ef_construction, lc);
        const std::size_t cap =
            (lc == 0) ? 2 * params_.M : params_.M;

        auto neighbors = select_neighbors(candidates, params_.M);
        store_links(new_idx, lc, neighbors);
        for (std::uint32_t nb : neighbors) {
            connect(nb, lc, new_idx, cap);
        }
        if (!candidates.empty()) {
            ep = std::min_element(candidates.begin(), candidates.end())->second;
        }
    }
}

void HnswIndex::add(std::uint64_t id, const float* vector, std::size_t dim) {
    if (dim != dim_) {
        throw std::invalid_argument("HnswIndex::add dimension mismatch");
    }
    const auto new_idx = static_cast<std::uint32_t>(nodes_.size());
    const int level = random_level();

    data_.insert(data_.end(), vector, vector + dim_);
    Node node;
    node.id = id;
    node.level = level;
    node.links.resize(static_cast<std::size_t>(level) + 1);
    nodes_.push_back(std::move(node));

    if (entry_point_ == kNoEntry) {
        entry_point_ = new_idx;
        max_level_ = level;
        return;
    }

    link_node(new_idx);
    if (level > max_level_) {
        max_level_ = level;
        entry_point_ = new_idx;
    }
}

void HnswIndex::add_parallel(const std::uint64_t* ids, const float* vectors,
                             std::size_t count, unsigned num_threads) {
    if (count == 0) {
        return;
    }
    // phase 1 (serial): store every vector, assign levels, and fix the entry
    // point to the highest-level node. linking happens in phase 2.
    const std::uint32_t base = static_cast<std::uint32_t>(nodes_.size());
    for (std::size_t i = 0; i < count; ++i) {
        const auto idx = static_cast<std::uint32_t>(nodes_.size());
        const int level = random_level();
        data_.insert(data_.end(), vectors + i * dim_, vectors + (i + 1) * dim_);
        Node node;
        node.id = ids[i];
        node.level = level;
        node.links.resize(static_cast<std::size_t>(level) + 1);
        nodes_.push_back(std::move(node));
        if (entry_point_ == kNoEntry || level > max_level_) {
            max_level_ = level;
            entry_point_ = idx;
        }
    }

    // one lock per node; vector<mutex> is move-assigned wholesale (the mutexes
    // themselves never move).
    link_locks_ = std::vector<std::mutex>(nodes_.size());

    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) {
            num_threads = 4;
        }
    }

    concurrent_ = true;
    std::atomic<std::uint32_t> next{base};
    const std::uint32_t end = base + static_cast<std::uint32_t>(count);
    auto worker = [&]() {
        for (;;) {
            const std::uint32_t i = next.fetch_add(1);
            if (i >= end) {
                break;
            }
            link_node(i);
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(num_threads);
    for (unsigned t = 0; t < num_threads; ++t) {
        pool.emplace_back(worker);
    }
    for (std::thread& th : pool) {
        th.join();
    }
    concurrent_ = false;
}

std::vector<Neighbor> HnswIndex::search(const float* query, std::size_t dim,
                                        std::size_t k) const {
    // temporary exact baseline so the index is usable and testable. the query PR
    // replaces this with the graph-traversal ef_search beam.
    if (dim != dim_ || nodes_.empty()) {
        return {};
    }
    std::vector<DistIdx> scored;
    scored.reserve(nodes_.size());
    for (std::uint32_t i = 0; i < nodes_.size(); ++i) {
        scored.emplace_back(distance(metric_, query, vector_at(i), dim_), i);
    }
    const std::size_t kk = std::min(k, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + kk, scored.end(),
                      [](const DistIdx& a, const DistIdx& b) {
                          return a.first < b.first;
                      });
    std::vector<Neighbor> out;
    out.reserve(kk);
    for (std::size_t i = 0; i < kk; ++i) {
        out.push_back({nodes_[scored[i].second].id, scored[i].first});
    }
    return out;
}

}  // namespace vectorsearch
