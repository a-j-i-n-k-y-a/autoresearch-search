---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : latency
generated   : 2026-05-16T23:42:17
---

# Architecture

## What it does
The system performs a high-performance, hybrid semantic search over a movie dataset. It combines vector-based similarity retrieval (FAISS) with metadata-driven ranking (popularity and quality scores) to deliver the most relevant movies while maintaining strict low-latency constraints.

## Components
- **Vector Index (FAISS):** Retrieves an initial candidate pool of 200 movies based on L2 distance between the query embedding and pre-computed movie embeddings.
- **Metadata Scoring Engine:** Post-processes the candidate pool by applying logarithmic transformations to `vote_count` (popularity) and direct scaling to `vote_average` (quality).
- **Ranking Layer:** Combines normalized vector distance with the metadata score to re-sort candidates, ensuring the final list is both contextually relevant and highly rated.

## Why it works
The system solves the "popularity bias" and "cold start" problems common in pure vector search. By re-ranking a broad initial candidate pool (200) using log-scaled popularity and rating boosts, the system surfaces high-quality content that matches user intent, rather than just returning the nearest mathematical neighbor. The use of `np.log1p` for vote counts prevents hyper-popular movies from disproportionately dominating the results.

## Tradeoffs
- **Candidate Pool Size:** Maintaining a pool of 200 items provides a balance between broad recall and the overhead of re-ranking. Smaller pools reduced latency but significantly degraded recall.
- **Compute Overhead:** The system intentionally avoids heavy secondary models (like cross-encoders) in the re-ranking phase to keep total latency under 13ms.
- **Heuristic Weighting:** The current `0.1` boost factors for popularity and rating are fixed; while effective, they do not dynamically adapt to different user intent types.

## Key experiments
- **Baseline:** Established the initial 0.441 recall @ 18.8ms.
- **Candidate Expansion:** Increasing the pool to 200 items was critical to improving recall from baseline levels.
- **Log-Popularity Scaling:** Discarded many attempts at complex hybrid fusion (RRF) in favor of simple metadata re-weighting, which achieved the final recall of 0.540 without increasing latency.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.540 | 0.540 |
| latency_ms | 12.3 | 12.3 |

## How to run
1. Ensure `faiss`, `numpy`, and `pandas` are installed.
2. Initialize the FAISS index and load the movie DataFrame (`df`).
3. Call `search(query, df, bm25, model, index)` to retrieve results.
4. The system is designed for stateless execution; ensure the `index` and `model` are pre-loaded in memory for production environments.