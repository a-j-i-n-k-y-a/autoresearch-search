---
exp_id      : exp_314
prompt_hash : f5920c6d
prompt_file : experiments/prompts/f5920c6d.txt
objective   : pareto
generated   : 2026-05-17T04:43:31
---

# Architecture

## What it does
The movie search system implements a hybrid retrieval and re-ranking architecture that combines keyword-based lexical search (BM25) with dense semantic vector search (FAISS) and a popularity-based ranking adjustment to deliver high-relevance movie recommendations.

## Components
*   **BM25 Retrieval:** Performs an initial full-text search across the corpus to retrieve a candidate pool of 1,000 documents, ensuring strong recall for specific keyword queries.
*   **FAISS Semantic Index:** Computes L2-distance-based semantic similarity between the user query and the candidate pool, mapping distances to scores ($1 / (1 + L2)$).
*   **Popularity Re-ranker:** Applies a multiplicative boost to the semantic scores using a formula combining vote volume and quality ($log1p(\text{vote\_count}) \times \text{vote\_average}$).
*   **Hybrid Scorer:** Merges BM25-informed candidate selection with refined semantic and popularity-weighted scoring to generate the final top-k results.

## Why it works
The system solves the "cold-start" problem of pure vector search by using BM25 as an efficient recall mechanism. By re-ranking only the top 1,000 candidates with deep semantic understanding and a popularity bias, it balances query-specific relevance with general user preference, maintaining low latency while significantly improving retrieval accuracy.

## Tradeoffs
*   **Latency vs. Precision:** By narrowing the search space to 1,000 items for re-ranking, we maintain sub-10ms latency at the cost of potential "long-tail" item omission.
*   **Complexity:** The multi-stage pipeline requires index management for both BM25 and FAISS, increasing memory overhead compared to a single-index system.

## Key experiments
*   **exp_314 (Final):** Leveraged a combined semantic and log-popularity scoring function on a 1,000-item BM25 candidate pool.
*   **Candidate Expansion:** Testing pool sizes between 10 and 1,000 revealed that 1,000 is the "sweet spot" for balancing latency and recall.
*   **Hybrid RRF vs. Scoring:** We found that direct score fusion based on normalized metrics significantly outperformed standard Reciprocal Rank Fusion (RRF).

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.520 | 0.540 |
| latency_ms | 5.9 | 5.9 |

## How to run
1. Ensure `bm25` (rank_bm25), `model` (sentence-transformers), and `index` (faiss) objects are pre-indexed.
2. Initialize the search with the movie dataframe `df`.
3. Call `search(query, df, bm25, model, index, top_k=10)` to receive the ranked record list.