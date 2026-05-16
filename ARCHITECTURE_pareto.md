---
exp_id      : exp_335
prompt_hash : 840bf4d5
prompt_file : experiments/prompts/840bf4d5.txt
objective   : pareto
generated   : 2026-05-17T05:13:57
---

# Architecture

## What it does
The system implements a hybrid search architecture for movie retrieval, combining keyword-based information retrieval with semantic vector space modeling. It performs a multi-stage process: fast recall via BM25, followed by semantic re-ranking using FAISS embeddings, and a final scoring adjustment based on popularity metrics.

## Components
- **BM25 Retrieval**: Uses `rank_bm25` for initial candidate selection, identifying a broad pool of 1,000 documents based on keyword matching to ensure high initial recall.
- **FAISS Semantic Ranking**: Leverages pre-computed embeddings to calculate semantic similarity (L2 distance) for the initial candidates, mapping distances to a semantic score $1/(1+L2)$.
- **Popularity Booster**: Applies a non-linear ranking boost calculated as `log1p(vote_count) * vote_average`, ensuring high-quality, popular content is prioritized over niche or poorly-rated semantic matches.
- **Scoring Logic**: Computes a final score: `semantic_score * (1.0 + 0.05 * popularity_boost)`.

## Why it works
The architecture successfully balances lexical relevance (keyword matching) with semantic intent. BM25 prevents "semantic drift" by ensuring query terms exist in the metadata, while the FAISS index captures latent relationships. The popularity multiplier acts as a quality filter, preventing the system from surfacing irrelevant but semantically "close" results.

## Tradeoffs
- **Complexity vs. Latency**: By offloading initial recall to BM25, we avoid exhaustive vector scans while maintaining consistent latency.
- **Memory**: Requires keeping both an inverted index (BM25) and a vector index (FAISS) in memory.
- **Popularity Bias**: While helpful for search relevance, the popularity boost may suppress newer or low-vote independent films (the "cold-start" problem).

## Key experiments
- **exp_335 (The Winner)**: Stabilized recall at 0.540 while maintaining sub-8ms latency by optimizing the candidate pool size and refining the popularity-based score multiplier.
- **Candidate Pool Tuning**: Increasing the retrieval pool to 1,000 candidates was essential for shifting recall from the baseline 0.441 to current levels.
- **Hybrid Fusion**: Numerous RRF (Reciprocal Rank Fusion) and weighted-score experiments were discarded due to increased latency without proportional improvements in recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.520 | 0.540 |
| latency_ms | 7.7 | 7.7 |

## How to run
1. Ensure the `df` (DataFrame), `bm25` (object), `model` (SentenceTransformer), and `index` (FAISS) are initialized.
2. Call the `search(query, df, bm25, model, index, top_k=10)` function.
3. Ensure the dataframe contains `vote_count` and `vote_average` columns for the popularity-based re-ranking.