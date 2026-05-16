---
exp_id      : exp_335
prompt_hash : 840bf4d5
prompt_file : experiments/prompts/840bf4d5.txt
objective   : pareto
generated   : 2026-05-17T05:05:13
---

# Architecture

## What it does
The system performs a high-performance hybrid movie search that combines keyword-based lexical relevance with semantic similarity and popularity-driven ranking. It is designed to maximize recall within strict latency constraints.

## Components
*   **Initial Retrieval (BM25):** Performs a high-recall lexical search on the query to identify a candidate pool of the top 1000 movies.
*   **Semantic Scoring (FAISS):** Computes semantic similarity for the initial 1000 candidates using a pre-indexed vector store. Semantic scores are normalized as $1 / (1 + L2\_distance)$.
*   **Popularity Re-ranking:** Applies a multiplier to the semantic scores based on the product of a log-scaled vote count (`np.log1p(vote_count)`) and the average rating (`vote_average`), ensuring high-quality, popular content is prioritized.
*   **Final Scoring:** Fuses the semantic and popularity metrics via: `score = semantic * (1.0 + 0.05 * popularity)`.

## Why it works
The two-stage retrieval approach balances precision and speed: BM25 acts as a wide-net filter to keep relevant keyword matches in the pool, while FAISS provides semantic depth. The final re-ranking stage corrects for pure relevance by integrating established movie quality metrics, which significantly boosts user-preferred content without introducing additional high-latency model inference steps.

## Tradeoffs
*   **Recall vs. Latency:** By utilizing a fixed candidate pool (1000), the system keeps latency predictable at ~8.4ms, though it may miss long-tail semantic matches outside the BM25 retrieval set.
*   **Popularity Bias:** The popularity boost naturally prioritizes well-known movies, which improves generic search performance but may suppress high-quality, niche, or newly released independent films.

## Key experiments
*   **exp_335 (Final):** Established the winning configuration of BM25 candidate retrieval followed by semantic scoring and logarithmic popularity scaling.
*   **Pool Expansion:** Identified that a candidate pool of 1000 provided the best Pareto efficiency between recall and latency.
*   **Normalization:** Discarded complex RRF and multi-stage re-ranking methods due to high latency overhead and inconsistent recall improvements.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.520 | 0.540 |
| latency_ms | 8.4 | 8.4 |

## How to run
1. Ensure the `bm25` object and `faiss` index are initialized with the movie dataset.
2. Call `search(query, df, bm25, model, index, top_k=10)` from `search.py`.
3. Ensure the `model` supports batch encoding for query vectors.