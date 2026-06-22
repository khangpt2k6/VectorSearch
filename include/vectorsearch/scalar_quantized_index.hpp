#pragma once

// Flat index over scalar-quantized (int8) vectors.
//
// Stores one byte per component instead of four, so the vectors take 4x less
// memory than the exact flat index. Search is asymmetric: the query stays in
// f32 and each stored vector is reconstructed from its code on the fly. That
// reconstruction is lossy, so a code-only search gives up a little recall.
//
// The re-rank option buys the recall back. It pulls a wider shortlist by
// quantized distance, then recomputes exact distances on the original f32
// vectors and re-sorts, so the top-k it returns is as accurate as the flat
// index. Keeping the originals around costs the memory the codes saved, which
// is the recall-vs-memory trade-off this whole phase is about: leave re-rank off
// for the memory win, turn it on for the accuracy.

#include <cstddef>
#include <cstdint>
#include <vector>

#include "vectorsearch/distance.hpp"
#include "vectorsearch/index.hpp"
#include "vectorsearch/scalar_quantizer.hpp"

namespace vectorsearch {

class ScalarQuantizedIndex : public Index {
public:
    struct Options {
        // keep the original f32 vectors and refine the shortlist with exact
        // distances. accurate but spends the memory the codes saved.
        bool rerank = true;
        // shortlist multiplier: a query for k pulls k * rerank_factor candidates
        // by quantized distance before the exact refine. ignored without rerank.
        std::size_t rerank_factor = 8;
    };

    ScalarQuantizedIndex(std::size_t dim, Metric metric);
    ScalarQuantizedIndex(std::size_t dim, Metric metric, Options options);

    // Fit the quantizer to a representative sample (usually the full dataset),
    // packed end to end. Must run before add(). Re-training resets the index.
    void train(const float* vectors, std::size_t count);

    void add(std::uint64_t id, const float* vector, std::size_t dim) override;
    std::vector<Neighbor> search(const float* query, std::size_t dim,
                                 std::size_t k) const override;
    std::size_t size() const override;

    bool trained() const { return quantizer_.trained(); }
    std::size_t dim() const { return dim_; }
    Metric metric() const { return metric_; }
    const Options& options() const { return options_; }
    const ScalarQuantizer& quantizer() const { return quantizer_; }

    // Bytes held by the stored vectors: codes, ids, and originals when re-rank
    // is on. raw_f32_bytes() is what an exact flat store of the same vectors
    // would cost, so the ratio is the memory win.
    std::size_t memory_bytes() const;
    std::size_t raw_f32_bytes() const;

    // std::vector convenience wrappers.
    void train(const std::vector<float>& vectors) {
        train(vectors.data(), dim_ == 0 ? 0 : vectors.size() / dim_);
    }
    void add(std::uint64_t id, const std::vector<float>& vector) {
        add(id, vector.data(), vector.size());
    }
    std::vector<Neighbor> search(const std::vector<float>& query,
                                 std::size_t k) const {
        return search(query.data(), query.size(), k);
    }

private:
    const std::uint8_t* code_at(std::size_t i) const {
        return codes_.data() + i * dim_;
    }
    const float* original_at(std::size_t i) const {
        return originals_.data() + i * dim_;
    }

    std::size_t dim_;
    Metric metric_;
    Options options_;
    ScalarQuantizer quantizer_;

    std::vector<std::uint8_t> codes_;   // size() * dim_ bytes
    std::vector<std::uint64_t> ids_;
    std::vector<float> originals_;      // size() * dim_ floats, only if rerank
};

}  // namespace vectorsearch
