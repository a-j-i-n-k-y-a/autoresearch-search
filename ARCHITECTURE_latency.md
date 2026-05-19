---
exp_id      : exp_339
prompt_hash : 6a31c16b
prompt_file : experiments/prompts/6a31c16b.txt
objective   : latency
generated   : 2026-05-19T14:36:42
---

# Architecture

## What it does
The system performs a hybrid movie retrieval by combining lexical (BM25) and semantic (dense vector) search signals. It aggregates these results using Reciprocal Rank Fusion (RRF), dynamically boosts scores based on genre-query intersection, and applies a log-transformed popularity bias to produce a final ranked list of movies.

## Components
- **Lexical Retrieval:** BM25 implementation using tokenized text from the movie database.
- **Semantic Retrieval:** FAISS-based vector search utilizing a pre-trained encoder to capture semantic intent.
- **RRF Aggregator:** Combines ranks from both retrieval streams to mitigate bias in individual scoring methods.
- **Contextual Re-ranker:** Applies a 1.5x multiplicative boost to results matching identified genres and incorporates a logarithmic popularity bias based on `vote_count` to prioritize established titles.

## Why it works
The hybrid RRF approach addresses the "semantic gap" inherent in pure vector search and the "vocabulary mismatch" in pure BM25.
- **RRF Integration:** Moving from weighted summation to RRF improved recall from 0.545 (baseline) to 0.636 (+0.091).
- **Genre Boosting:** Genre-aware filtering and boosting allowed for sharper alignment with user intent, specifically handling queries that include genre keywords (e.g., "action movies").
- **Efficiency Gains:** The transition to direct NumPy array indexing and reduced DataFrame operations during the scoring pipeline enabled a reduction in latency from 23.4ms to 22.0ms (-1.4ms).

## Tradeoffs
- **Complexity vs. Latency:** Increasing the candidate pool size (to 200) improved recall but required aggressive NumPy-level optimization to maintain sub-25ms latency.
- **Memory Overhead:** The use of pre-computed genre masks reduces runtime compute at the expense of memory consumption, favoring faster retrieval for production environments.

## Key experiments
- **exp_339 (Final):** Optimized candidate selection (200 pool) with RRF and genre-aware popularity dampening. Achieved the optimal recall of 0.636 with a latency of 22.0ms.
- **Initial Baseline:** Standard hybrid search yielding 0.545 recall at 23.4ms.
- **RRF Implementation:** Replacing weighted summation with RRF was critical in stabilizing rank positions and improving overall recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.545 | 0.636 |
| latency_ms | 23.4 | 22.0 |
| optimization_cost_usd | n/a | $0.001517 |

## How to run
1. Initialize the `BM25Okapi` object and FAISS `index` with the movie dataset.
2. Call `search(query, df, bm25, model, index, top_k=10)`.
3. The function will perform tokenization, parallel index querying, RRF aggregation, and finally, vectorized genre/popularity re-ranking on the resulting NumPy arrays.