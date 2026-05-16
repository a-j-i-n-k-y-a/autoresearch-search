---
exp_id      : exp_438
prompt_hash : 56f43cf0
prompt_file : experiments/prompts/56f43cf0.txt
objective   : recall
generated   : 2026-05-17T03:48:08
---

# Architecture

## What it does
The system performs a semantic-first movie search that combines high-recall vector retrieval with a custom re-ranking heuristic. It prioritizes relevant content while adjusting for viewer popularity and critical reception to ensure high-quality recommendations.

## Components
*   **Vector Retrieval (FAISS):** Uses a pre-trained model to encode queries into a dense vector space, retrieving a large candidate pool (500 movies) via L2 distance to ensure high recall.
*   **Semantic Scoring:** Converts L2 distance to a probability-like score: `1.0 / (1.0 + L2)`.
*   **Popularity Re-ranking:** Applies a weighted boost to the semantic score using: `log1p(vote_count) * (vote_average / 10.0)`.
*   **Late Fusion:** Multiplies the semantic score by the popularity boost factor to produce a final ranking score.

## Why it works
The system balances two critical dimensions of discovery:
1.  **Semantic Precision:** The FAISS index captures the intent behind the query, moving beyond simple keyword matching.
2.  **Quality Bias:** By using `log1p` on `vote_count`, the system dampens the noise from extreme outliers while still promoting movies that have a proven, positive track record with audiences.

## Tradeoffs
*   **Recall vs. Complexity:** By retrieving 500 candidates, the system maintains high recall but incurs more overhead than pure top-k retrieval; however, the lack of a heavy second-stage model keeps latency low.
*   **Dependency:** Performance is highly sensitive to the quality of the embedding model and the distribution of the movie ratings.

## Key experiments
*   **Candidate Pool Expansion (exp_438):** Scaling the retrieval pool to 500 items was the primary driver for achieving a recall of 0.550.
*   **Logarithmic Popularity Boosting:** Replacing raw counts with `log1p(vote_count)` prevented popular movies from dominating the rankings while effectively filtering out low-quality, obscure matches.
*   **Hybrid Failure:** Numerous attempts at combining BM25 and FAISS via RRF (Reciprocal Rank Fusion) significantly increased latency and complexity without providing a meaningful boost to recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.441 | 0.550 |
| latency_ms | 18.8 | 5.6 |

## How to run
1. Ensure `faiss` and `sentence-transformers` (or equivalent `model`) are initialized with the pre-indexed movie vectors.
2. Load the movie dataframe containing `vote_count` and `vote_average` metadata.
3. Call the `search(query, df, bm25, model, index)` function, passing the query string and the respective data objects.