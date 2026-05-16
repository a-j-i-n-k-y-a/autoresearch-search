---
exp_id      : exp_541
prompt_hash : fb7b5e6b
prompt_file : experiments/prompts/fb7b5e6b.txt
objective   : pareto
generated   : 2026-05-17T05:18:05
---

# Architecture

## What it does
The system performs a high-performance hybrid movie search that combines keyword-based lexical matching with semantic embedding-based retrieval. It ranks results by synthesizing semantic similarity scores with popularity metrics to deliver relevant, high-quality content efficiently.

## Components
- **BM25 Retrieval**: Operates as a coarse-grained filter to extract the top 500 candidate movies based on tokenized keyword relevance.
- **FAISS Semantic Engine**: Provides dense vector similarity search; retrieves 2,000 top matches to ensure broad semantic coverage.
- **Popularity Booster**: Normalizes `vote_count` (log-scale) and `vote_average` into a single popularity score, which is then blended with normalized semantic distances to adjust final rankings.
- **Fusion Layer**: Calculates a composite score for the candidate pool: $Score = SemanticScore + (0.1 \times PopularityScore)$.

## Why it works
The two-stage approach balances precision and recall: BM25 excels at specific title/keyword matches, while FAISS handles conceptual queries where user intent is implicit. The popularity boost ensures that in cases of semantic ambiguity, the system defaults to well-regarded, community-validated titles, increasing user satisfaction.

## Tradeoffs
- **Latency vs. Accuracy**: By limiting the BM25 retrieval to 500 and performing a localized FAISS search, the system avoids expensive full-index operations, maintaining consistent latency at a slight cost to theoretical recall.
- **Complexity**: The additive scoring logic requires careful scaling; overly aggressive popularity weighting can bias the results toward blockbusters at the expense of niche relevant titles.

## Key experiments
- **exp_541 (Final)**: Implemented the hybrid retrieval and popularity score normalization, which achieved the target pareto efficiency.
- **Baseline**: Established the original BM25-based recall benchmark.
- **Pool Scaling**: Multiple experiments showed that increasing the candidate pool beyond 500-1000 items significantly increased latency without providing proportional gains in recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.580 | 0.550 |
| latency_ms | 20.7 | 20.4 |

## How to run
1. Install dependencies: `pip install numpy pandas faiss-cpu rank-bm25`.
2. Initialize the `bm25` object and `faiss.Index` with your movie corpus.
3. Import the `search` function from `search.py`.
4. Call `search(query, df, bm25, model, index)` to retrieve the top 10 ranked results.