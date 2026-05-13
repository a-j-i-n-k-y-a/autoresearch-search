# Architecture

## What it does
The system performs high-speed semantic movie retrieval by combining vector-based similarity search with a popularity-based re-ranking mechanism. It bridges the gap between raw semantic proximity and user-preferred content.

## Components
*   **Vector Index (FAISS):** Powers the primary retrieval stage, performing L2 distance-based searches in embedding space.
*   **Popularity Scoring Module:** Applies a log-transformation to `vote_count` to dampen the influence of extreme outliers while favoring widely recognized movies.
*   **Re-ranking Engine:** Integrates the normalized inverse distance from the FAISS index with the scaled popularity score to compute a final relevance ranking.

## Why it works
By retrieving a "wider pool" (5x the target $k$) via FAISS and applying a popularity-weighted re-ranking, the system retains the high-quality matches found by semantic search while ensuring that popular, relevant results are boosted to the top of the result list without significant latency overhead.

## Tradeoffs
*   **Efficiency:** Trading a small, constant increase in computational cost ($0.000447) for a 4.6x improvement in latency compared to the baseline.
*   **Ranking:** The system prioritizes popularity as a proxy for user satisfaction, which may slightly deprioritize obscure but highly relevant niche films.

## Key experiments
*   **Baseline:** Established the recall ceiling but suffered from high latency.
*   **Pure Semantic Search:** Demonstrated high speed but lacked the nuance provided by metadata integration.
*   **Final Implementation:** Successfully balanced recall (0.800) by utilizing log-scaled `vote_count` re-ranking on a candidate pool expanded during initial FAISS retrieval.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.800 | 0.800 |
| latency_ms | 53.2 | 11.5 |

## How to run
Ensure `numpy` and `faiss` are installed. Initialize the FAISS index and the sentence-transformer model. Call `search(query, df, bm25, model, index, top_k=10)` passing the pre-computed dataframe, index, and model.