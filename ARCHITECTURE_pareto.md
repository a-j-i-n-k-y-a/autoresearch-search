# Architecture

## What it does
The system provides high-performance, relevance-aware movie recommendations by performing a candidate expansion via vector similarity search, followed by a lightweight popularity-biased re-ranking.

## Components
*   **Vector Retrieval (FAISS):** Uses a pre-computed index to perform fast L2 distance-based retrieval of 50 candidates (top_k * 5).
*   **Popularity Signal:** Extracts `vote_count` metadata, applying a `log1p` transformation to compress the distribution and prevent highly popular movies from overwhelming semantic relevance.
*   **Scoring Function:** Calculates a hybrid score: $Score = \frac{1}{1 + \text{distance}} + (\log(1 + \text{votes}) \times 0.05)$.
*   **Re-ranking Engine:** Sorts the expanded candidate pool based on the hybrid score and truncates to the final `top_k` results.

## Why it works
By retrieving a pool larger than `top_k`, the system gains enough headroom to inject metadata-based signals without compromising semantic precision. The `log1p` scaling ensures the popularity bias acts as a "tie-breaker" for movies that are already semantically similar, improving user satisfaction without drifting away from the search intent.

## Tradeoffs
*   **Precision vs. Latency:** Increasing the retrieval pool size increases memory overhead and processing time; the current 5x factor is the empirical sweet spot.
*   **Metadata Dependency:** Relies on the availability and quality of `vote_count`. If this data is sparse or noisy, the re-ranking logic may degrade.

## Key experiments
*   **Expansion vs. Direct Search:** Moving from a strict 1:1 retrieval to a 5x candidate pool allowed for significant improvements in recall by enabling effective re-ranking.
*   **Popularity Scaling:** Early iterations using raw `vote_count` or `vote_average` failed to account for long-tail distributions; the `log1p` transformation stabilized the scoring.
*   **Hybrid RRF:** Attempts at complex RRF (Reciprocal Rank Fusion) between BM25 and Vectors were discarded due to high latency and negligible gains compared to the final scoring method.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.412 | 0.800 |
| latency_ms | 14.1 | 11.5 |

## How to run
1. Ensure the FAISS index is loaded into memory as `index`.
2. Provide a pre-encoded `model` capable of generating query embeddings.
3. Pass the `df` containing metadata (`title`, `overview`, `vote_count`) to the `search(query, df, bm25, model, index)` function.