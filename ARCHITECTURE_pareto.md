---
exp_id      : exp_314
prompt_hash : f5920c6d
prompt_file : experiments/prompts/f5920c6d.txt
objective   : pareto
generated   : 2026-05-17T04:33:27
---

# Architecture

## What it does
The system implements a hybrid search architecture that combines keyword-based retrieval with semantic vector similarity and popularity-based ranking. It retrieves a broad candidate set using BM25 and refines the relevance using a dense FAISS vector index, ultimately re-weighting results by movie popularity (vote count and average).

## Components
- **BM25 Retrieval**: Performs initial recall by fetching 1,000 candidates based on keyword matching against the movie dataset.
- **FAISS Semantic Scoring**: Computes L2 distances between the query vector and candidate embeddings, converting distances into a 0-1 semantic similarity score ($1 / (1 + L2)$).
- **Popularity Booster**: Applies a multiplicative boost to the semantic score using a logarithmic transformation of `vote_count` scaled by `vote_average` to prioritize high-quality, widely-viewed content.
- **Ranker**: Merges semantic and popularity signals into a final score, sorting candidates to return the top_k results.

## Why it works
The architecture solves the "cold-recall" problem of pure vector search by ensuring relevant keywords are captured via BM25, while the FAISS index provides nuances of semantic meaning. The inclusion of a popularity bias aligns the search results with user expectations for high-quality recommendations, significantly improving recall without sacrificing low-latency performance.

## Tradeoffs
- **Latency vs. Accuracy**: By limiting the re-ranking to a 1,000-candidate pool, the system maintains a low latency (~5.6ms) at the expense of potentially missing relevant results outside the initial BM25 retrieval set.
- **Complexity**: Combining multiple scoring mechanisms increases the surface area for hyperparameter tuning compared to a single-stage search.

## Key experiments
- **Candidate Pool Expansion (exp_314)**: Optimized the balance between BM25 recall and vector re-ranking, leading to the current high-recall configuration.
- **Popularity-Boost Integration**: Iterated on various popularity formulas (log-scaling vs. raw) to find the optimal multiplier that favors quality without drowning out specific semantic matches.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.520 | 0.540 |
| latency_ms | 5.6 | 5.6 |

## How to run
1. Ensure `df`, `bm25` index, `model`, and `faiss_index` are pre-loaded in memory.
2. Call `search(query, df, bm25, model, index, top_k=10)` from `search.py`.
3. The function will return a list of dictionary records containing movie metadata, ranked by the combined semantic and popularity score.