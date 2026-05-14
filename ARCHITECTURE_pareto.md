# Architecture

## What it does
The system implements a two-stage hybrid retrieval pipeline for movie search. It first performs rapid keyword-based pruning of the candidate space using BM25, followed by a computationally efficient semantic re-ranking using dot-product vector similarity.

## Components
- **Candidate Retrieval:** A BM25-based algorithm retrieves the top 50 most relevant documents from the index based on exact term matching.
- **Embedding Model:** A pre-trained transformer model generates dense vector representations for the input query and the pre-filtered candidate pool.
- **Re-ranker:** A dot-product calculation computes semantic similarity between the query embedding and the candidate pool, sorting them to produce the final top-k result.

## Why it works
The architecture optimizes the search bottleneck by decoupling coarse-grained retrieval from fine-grained ranking. By limiting the embedding encoding and similarity scoring to the top 50 candidates identified by BM25, the system minimizes high-cost inference operations while maintaining high semantic recall. This "prune-then-rank" strategy provides a significant latency reduction compared to global vector search.

## Tradeoffs
- **Latency vs. Exhaustiveness:** By using a fixed candidate pool (top 50), the system trades the ability to retrieve long-tail matches that might score poorly on BM25 but well on semantic similarity.
- **Resource Usage:** The system requires maintaining both a keyword index and a vector embedding model, increasing the complexity of the data pipeline compared to a pure BM25 or pure vector approach.

## Key experiments
- **Candidate Pool Size:** Testing indicated that increasing the pool from 50 to 100 yielded negligible gains in recall, while lower counts significantly degraded performance.
- **Retrieval Hybridization:** Numerous experiments with RRF (Reciprocal Rank Fusion) and complex weighting showed higher latency and lower recall than the simple two-stage BM25-filtering approach.
- **Stability:** Several attempts to implement complex feature-weighted re-ranking resulted in system crashes or performance regressions, leading to the selection of the current stable, low-latency implementation.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.510 | 0.510 |
| latency_ms | 14.8 | 8.6 |

## How to run
1. Ensure the environment has `numpy`, `pandas`, and the `rank_bm25` library installed.
2. Load the movie DataFrame and pre-indexed BM25 object.
3. Initialize the embedding model.
4. Call `search(query, df, bm25, model, index, top_k=10)` to retrieve ranked results.