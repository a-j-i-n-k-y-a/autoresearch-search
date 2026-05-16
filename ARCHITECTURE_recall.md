---
exp_id      : exp_314
prompt_hash : f5920c6d
prompt_file : experiments/prompts/f5920c6d.txt
objective   : recall
generated   : 2026-05-17T00:37:57
---

# Architecture

## What it does
The system performs a hybrid semantic-popularity search for movies. It retrieves a broad candidate pool using FAISS vector similarity and then re-ranks these results by integrating signal from popularity (vote count) and quality (vote average) metrics to deliver relevant, user-preferred results.

## Components
- **FAISS (Vector Index):** Handles initial semantic retrieval of candidates based on the user's query embedding.
- **Scoring Engine:** A re-ranking layer that transforms raw FAISS L2 distances into similarity scores and multiplies them by log-scaled popularity and quality weights.
- **Candidate Pool:** Configured to retrieve 200 candidates to balance coverage and re-ranking latency.

## Why it works
The system combines the strengths of deep semantic matching (FAISS) with metadata-based ranking. By using `log1p` on `vote_count`, we suppress the noise of ultra-niche movies while surfacing high-quality, popular content. The L2 normalization ensures that distance scores are transformed into a probability-like space, allowing the popularity and rating multipliers to effectively shift the final rankings.

## Tradeoffs
- **Complexity:** Relies on empirical constants (0.1) for ranking weights, which may require manual tuning if the underlying data distribution changes.
- **Candidate Pool Constraint:** By fixing the initial pool to 200, the system sacrifices recall for extremely long-tail queries that might not appear in the top 200 semantic matches.
- **Cost:** The system incurs a non-zero operational cost per query compared to the baseline, reflecting the overhead of the re-ranking logic and the larger initial retrieval set.

## Key experiments
- **exp_314 (Winner):** Implemented the final log-popularity and rating boost on top of a 200-item FAISS pool, achieving optimal balance in recall and latency.
- **Baseline:** Established the initial recall of 0.441.
- **Candidate Expansion:** Increasing the pool from the baseline to 200 candidates was the primary driver for improving recall, while subsequent re-ranking experiments stabilized the output quality.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.441 | 0.540 |
| latency_ms | 18.8 | 9.4 |

## How to run
1. Ensure `faiss`, `pandas`, and `numpy` are installed.
2. Load the movie metadata into a DataFrame and initialize the FAISS index.
3. Pass the query to the `search(query, df, bm25, model, index, top_k=10)` function in `search.py`.
4. The function returns a sorted list of dictionaries containing the top 10 movies.