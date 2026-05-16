---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : latency
generated   : 2026-05-16T23:29:07
---

# Architecture

## What it does
The system performs a high-performance, quality-aware movie recommendation search. It retrieves candidate movies via semantic vector similarity and then re-ranks them by combining relevance (L2 distance) with popularity (vote count) and quality (average rating) metrics to optimize for user engagement.

## Components
- **Vector Retrieval**: Uses a FAISS index to perform an initial k-nearest neighbor search on the query embedding.
- **Candidate Pool**: A retrieved pool of 200 candidates is processed to ensure semantic coverage while keeping compute overhead low.
- **Re-ranker**: A post-processing step that calculates a custom score: $Score = \frac{1}{1 + dist} \times (1 + 0.1 \times \log(\text{vote\_count})) \times (1 + 0.1 \times \text{vote\_average})$.
- **Data Processor**: A pandas-based dataframe pipeline that handles normalization of popularity and rating features.

## Why it works
By moving from a pure semantic search (which often favors obscure, high-cosine-similarity matches) to a hybrid ranking approach, the system balances relevance with "community validation." The use of `np.log1p` for vote counts prevents hyper-popular movies from drowning out niche content, while the quality boost ensures that within the semantically relevant pool, top-rated movies surface first.

## Tradeoffs
- **Complexity vs. Latency**: Including popularity and rating metadata requires an additional re-ranking step post-retrieval, which could scale poorly if the initial candidate pool is significantly increased.
- **Dependency**: The system relies on the existence of quality/popularity metadata; if this data is missing for new releases, the scoring defaults to the raw vector distance.

## Key experiments
- **Candidate Expansion (Exp #8)**: Determined that a pool of 200 significantly improved performance over smaller sets without inducing substantial latency.
- **Logarithmic Popularity Boost**: Identified that `log1p(vote_count)` provides the most stable improvement to recall by filtering out "noise" while keeping high-quality content visible.
- **Distance Normalization**: Normalized L2 distances to the range [0, 1] to ensure that ranking math remains consistent regardless of index density.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.441 | 0.540 |
| latency_ms | 18.8 | 10.9 |

## How to run
1. Ensure the `faiss` index and `movie_df` are loaded.
2. Initialize the `model` (e.g., SentenceTransformer).
3. Call `search(query, df, bm25, model, index, top_k=10)` to receive the ranked dictionary of records.