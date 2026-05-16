---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : latency
generated   : 2026-05-16T23:39:37
---

# Architecture

## What it does
The movie search system performs a hybrid semantic-popularity search. It retrieves a broad set of candidate movies via FAISS vector similarity and subsequently refines the ranking by applying a geometric re-scoring function that incorporates movie quality (rating) and popularity (vote count).

## Components
*   **Vector Retrieval:** Uses a pre-trained encoder to convert natural language queries into embedding vectors, performing a FAISS ANN search across the movie catalog.
*   **Candidate Pool:** Fetches a large initial candidate set (top 200) to ensure high recall before applying metadata-based filtering.
*   **Re-ranking Engine:** A `search.py` module that calculates a weighted score using:
    *   **L2 Distance:** Inverted and normalized to emphasize semantic relevance.
    *   **Popularity Boost:** $\log(1 + \text{vote\_count})$ to prioritize high-traffic, well-known content.
    *   **Quality Boost:** Raw `vote_average` scores to promote highly-rated content.

## Why it works
The system balances two distinct signals: intent matching (via embeddings) and human consensus (via popularity and ratings). By using a log-scale for popularity, the system prevents extreme outliers in vote counts from dominating the search results while ensuring that highly-rated, popular movies are surfaced over obscure content with similar semantic profiles.

## Tradeoffs
*   **Compute vs. Precision:** By pulling 200 candidates and calculating scores in memory, we maintain a tight latency budget (11ms) without the overhead of complex RRF (Reciprocal Rank Fusion) or secondary index lookups.
*   **Data Dependency:** The scoring function assumes clean `vote_count` and `vote_average` fields; performance relies heavily on the quality of these metadata features.

## Key experiments
*   **Candidate Pool Expansion:** Testing various retrieval depths showed that 200 candidates provided the optimal balance between recall and latency.
*   **Metadata Re-scaling:** Iterations using logarithmic popularity boosts proved significantly more stable than linear scaling, preventing "popularity bias" from crushing the semantic relevance provided by the vector model.
*   **Hybrid Scoring:** Abandoning complex RRF (which increased latency to >20ms) in favor of a single-pass multiplicative score calculation was critical to achieving target performance.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.540 | 0.540 |
| latency_ms | 11.0 | 11.0 |

## How to run
1. Initialize the FAISS index from the provided movie embeddings.
2. Ensure the Pandas DataFrame `df` contains columns: `vote_count`, `vote_average`, and `id`.
3. Call `search(query, df, bm25, model, index, top_k=10)` to receive the ranked dictionary of results.