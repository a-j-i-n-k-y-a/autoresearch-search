# Architecture

## What it does
The system performs a hybrid semantic search for movies by combining vector-based similarity (via FAISS) with a popularity-based re-ranking mechanism. It retrieves an expanded candidate set from the semantic index and refines the ranking using the movie's `vote_count` to ensure that relevant, well-established content is prioritized.

## Components
*   **Vector Retrieval:** Uses a pre-trained sentence transformer model to encode queries into a dense vector space, querying a FAISS index for high-dimensional similarity.
*   **Candidate Expansion:** Retrieves 5x the target `top_k` candidates to create a sufficient pool for re-ranking without compromising latency.
*   **Popularity-Weighting:** Applies a log-transformation (`log1p`) to `vote_count` to dampen the effect of extreme outliers while favoring popular titles.
*   **Ranking Logic:** Combines inverted L2 distance with normalized popularity scores: `(1.0 / (1.0 + dist)) + (log_pop * 0.05)`.

## Why it works
The architecture avoids the high computational overhead of multi-stage hybrid models (like BM25 + Vector fusion). By expanding the initial search pool and applying a lightweight, mathematically simple popularity boost, the system achieves a balance between "semantic relevance" (vector similarity) and "user trust" (popularity). This approach optimizes for low latency while significantly improving recall compared to baseline methods.

## Tradeoffs
*   **Complexity vs. Performance:** The system trades the potential precision gains of complex hybrid retrieval (e.g., RRF) for architectural simplicity and speed.
*   **Bias:** The inclusion of `vote_count` inherently biases the system toward older, established movies, which may suppress niche or newly released content that has not yet gathered sufficient votes.

## Key experiments
*   **Hybrid RRF Attempts:** Repeatedly showed that complex multi-model scoring (BM25 + Semantic) increased latency significantly without linear gains in recall.
*   **Popularity Boosting:** The discovery that using `vote_count` as a logarithmic weight on the distance-based score provided a substantial jump in recall (from 0.4 to 0.8) while keeping latency under 12ms.
*   **Pool Expansion:** Experimenting with a `top_k * 5` retrieval pool proved optimal, providing enough variety to allow popularity re-ranking to effectively move better results into the final `top_k`.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.412 | 0.800 |
| latency_ms | 14.8 | 11.5 |

## How to run
Ensure the `df` (containing `vote_count`), `model` (embedding model), and `index` (FAISS) are loaded. Call the search function:
```python
results = search(query, df, bm25, model, index, top_k=10)
```