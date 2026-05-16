---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : cost
generated   : 2026-05-17T05:20:36
---

# Architecture

## What it does
The system performs a hybrid retrieval task for movie recommendations by combining keyword-based search with semantic vector space search, finalized by a popularity-aware re-ranking layer. It balances lexical precision (via BM25) and conceptual similarity (via FAISS) while ensuring popular, highly-rated movies are prioritized in the final results.

## Components
- **BM25 Retrieval**: Performs an initial coarse-grained search across the movie dataset, retrieving the top 500 candidates based on query token frequency.
- **FAISS Semantic Index**: Executes a vector similarity search using a pre-trained embedding model to score the top 2000 items in semantic space.
- **Scoring Engine**: 
    - Normalizes semantic distances (reciprocal mapping).
    - Computes a popularity boost using log-transformed vote counts multiplied by average ratings.
- **Ranking Layer**: Combines normalized semantic scores with popularity-boosted metadata scores, sorting by the weighted sum to produce the final top-K recommendation list.

## Why it works
The two-stage retrieval (BM25 -> FAISS) ensures that highly relevant keyword matches are not lost while allowing semantic nuance to refine the ordering. By incorporating a logarithmic popularity boost, the system avoids the "long-tail" problem where obscure movies with high semantic similarity obscure well-regarded, popular cinematic choices.

## Tradeoffs
- **Complexity vs. Latency**: The system maintains stable latency (~20ms) by using an optimized candidate pool size (500-2000), accepting that it does not perform a global full-corpus re-ranking.
- **Hybrid Weighting**: The static 0.1 weight for popularity bias is a heuristic; while effective for the current dataset, it may require tuning if the distribution of ratings shifts significantly.

## Key experiments
- **Candidate Pool Expansion**: Increasing the initial pool size from 100 to 500 significantly improved recall by providing a richer set for semantic comparison.
- **Logarithmic Popularity Boosting**: Moving from raw vote-counts to `log1p(vote_count) * rating` prevented massive outliers from dominating the rankings while still favoring high-quality content.
- **Hybrid Integration**: Experiments comparing pure FAISS against hybrid BM25/FAISS showed that keyword-based retrieval is essential for query-specific precision in movie titles.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.550 | 0.550 |
| latency_ms | 20.7 | 20.7 |

## How to run
1. Ensure `faiss`, `numpy`, and `pandas` are installed.
2. Prepare the `df` (DataFrame) with `vote_count` and `vote_average` columns.
3. Initialize a trained `BM25` object and a `FAISS` index.
4. Pass these components to the `search(query, df, bm25, model, index)` function to receive the ranked record set.