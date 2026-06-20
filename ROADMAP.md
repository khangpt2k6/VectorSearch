# Roadmap

The plan, in the order it should get built. Each phase leaves the repo in a
working, testable state. Check items off as they land.

## Phase 0 - Foundation (done)

- [x] Cargo project scaffolding
- [x] Distance metrics: squared L2, cosine, dot product
- [x] `Index` trait (add, search, len)
- [x] Flat / brute-force index (exact, used as the recall ground truth)
- [x] Unit + integration tests

## Phase 1 - HNSW index (the core of the project)

- [ ] Graph node + layer structure
- [ ] Insertion: level assignment, greedy search, neighbor selection heuristic
- [ ] Query: layer descent + ef-search beam
- [ ] Tunable params: `M`, `ef_construction`, `ef_search`
- [ ] Correctness test: recall@10 vs the flat index on random data
- [ ] Multi-threaded build (rayon)

## Phase 2 - Persistence

- [ ] Save / load index to disk (binary format)
- [ ] Memory-mapped vectors so a large index does not need to fit in RAM
- [ ] Incremental append without a full rebuild

## Phase 3 - Compression / quantization

- [ ] Scalar quantization (f32 -> int8) with a re-rank pass
- [ ] Product quantization (PQ) for big memory savings
- [ ] Measure the recall vs memory trade-off

## Phase 4 - Benchmark harness (the deliverable that proves it works)

- [ ] Dataset loaders for the ann-benchmarks `.hdf5` format
- [ ] Datasets: SIFT1M, GloVe-100, GIST1M
- [ ] Metrics: recall@k, queries/sec, build time, index size in memory
- [ ] Ground truth from the flat index
- [ ] Comparison runs against FAISS, hnswlib, and Qdrant
- [ ] Output: a recall-vs-QPS table and chart per dataset

## Phase 5 - Server + API (optional, if there is time)

- [ ] In-memory collection with metadata payloads
- [ ] HTTP API: create collection, upsert, search
- [ ] Metadata filtering during search (hybrid)

## Stretch

- [ ] SIMD distance kernels (`std::simd` or `wide`)
- [ ] IVF index as an alternative to HNSW
- [ ] gRPC API

## Design notes

- Search ranks by distance ascending, so smaller is closer for every metric.
  Cosine is stored as `1 - similarity`; dot is stored as `-dot` so that the
  same "smaller is closer" rule holds everywhere.
- The flat index is never thrown away: it is the source of ground truth that
  every approximate index gets scored against.
- Benchmark numbers are only trustworthy on standard public datasets, which is
  why Phase 4 uses the ann-benchmarks datasets instead of synthetic data.
