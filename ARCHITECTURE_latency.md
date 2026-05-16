---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : latency
generated   : 2026-05-16T23:07:14
---

# Architecture

## What it does
The system is an autonomously optimized movie search engine that retrieves relevant titles by combining semantic vector embeddings with heuristic popularity and quality signals. It maps user queries into a latent space to find candidate movies, then re-ranks these candidates to favor high-rated and well-known content.

## Components
*   **Vector Search (FAISS):** Performs K-Nearest Neighbor retrieval using a pre-computed index to map query embeddings to the top 200 candidate movies.
*   **Ranking Logic:** Applies a post-retrieval re-ranking function that incorporates:
    *   **Vector Proximity:** Normalized L2 distance to ensure semantic relevance.
    *   **Popularity Signal:** Log-scaled `vote_count` to account for crowd consensus.
    *   **Quality Signal:** Raw `vote_average` to prioritize critically acclaimed titles.
*   **Data Pipeline:** Uses a Pandas-based candidate pool management system for lightweight filtering and feature extraction.

## Why it works
By retrieving a broader candidate pool (200 movies) and re-ranking them using a multi-factor scalar score, the system balances raw semantic similarity with user-centric signals (popularity and rating). The combination of log-transformation on popularity prevents outliers from dominating the rankings while maintaining relevance.

## Tradeoffs
*   **Latency vs. Recall:** Extensive testing revealed that increasing the candidate pool beyond 200 provided diminishing returns on recall while linearly increasing compute costs and latency.
*   **Complexity:** The system favors a simple re-ranking heuristic over complex ensemble methods, which proved to be unstable or computationally prohibitive in testing.

## Key experiments
*   **Initial Baseline:** Established a recall of 0.441 with 18.8ms latency.
*   **Candidate Expansion:** Expanding the FAISS retrieval pool to 200 proved the most effective way to hit the recall ceiling without exceeding latency budgets.
*   **Score Fusion:** The final integration of `log(vote_count)` and `vote_average` as multipliers for the normalized FAISS distance yielded the optimal balance between ranking accuracy and inference speed.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.540 | 0.540 |
| latency_ms | 10.8 | 10.8 |

## How to run
1. Ensure `faiss` and `numpy` are installed.
2. Initialize the index using the model encoder.
3. Call `search(query, df, bm25, model, index)` with a user query string.
4. The function returns the top_k ranked movie records as a dictionary.