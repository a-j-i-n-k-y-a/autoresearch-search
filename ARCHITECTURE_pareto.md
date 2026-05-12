# Architecture

## What it does
The system performs a semantic-first movie search that balances vector similarity with metadata-driven popularity signals and genre-based query expansion to maintain high recall while optimizing result relevance.

## Components
*   **Vector Search Engine (FAISS):** The core retrieval engine that performs similarity searches using dense embeddings (model-encoded).
*   **Expansion Logic:** A lightweight genre-matching filter that boosts scores based on keyword overlap between the user query and movie genres.
*   **Ranking Engine:** A custom heuristic ranker that combines Euclidean distance-based similarity scores with log-normalized `vote_count` to ensure popular, high-quality results surface without overwhelming niche content.
*   **Candidate Pipeline:** Retrieves a wide pool (50x $K$) to ensure sufficient context for reranking.

## Why it works
The architecture avoids the overhead of complex hybrid retrieval (like BM25 + Vector) which proved costly and error-prone in testing. By utilizing a "widen then weight" strategy, it captures a broad semantic set and then applies domain-specific priors (popularity and genre) to reorder results locally, maintaining high recall while keeping latency stable.

## Tradeoffs
*   **Memory vs. Precision:** By retrieving a wide candidate pool (50 * top_k), the system trades off some compute for higher recall stability.
*   **Heuristic vs. Learned:** The ranker uses hand-tuned weights for popularity and genres. While highly performant and deterministic, it requires manual adjustments if the data distribution shifts significantly.

## Key experiments
*   **Hybrid Failure:** Repeated attempts to integrate BM25 and complex ranking ensembles consistently resulted in lower recall (0.600) compared to pure semantic retrieval.
*   **The "Golden Ratio":** Increasing the candidate pool size was found to be the most critical factor for matching the baseline recall.
*   **Normalization:** Using `np.log1p` for `vote_count` successfully mitigated the "blockbuster bias" where high-vote-count movies drowned out relevant semantic matches.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.900 | 0.900 |
| latency_ms | 39.0 | 39.0 |

## How to run
1. Ensure the FAISS index is initialized and the `df` containing `title`, `overview`, `vote_count`, and `genres` is loaded.
2. Initialize the embedding model.
3. Call `search(query, df, bm25, model, index, top_k=10)`.