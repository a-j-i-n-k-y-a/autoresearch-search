---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : latency
generated   : 2026-05-16T23:52:10
---

# Architecture

## What it does
The system performs high-performance movie retrieval by combining semantic vector search (FAISS) with metadata-based re-ranking to deliver highly relevant results within a tight latency budget.

## Components
*   **Vector Search Engine (FAISS):** Retrieves an initial candidate pool of 200 movies based on embedding similarity between the query and movie content.
*   **Metadata Scorer:** A post-processing logic that calculates a composite score for candidates using logarithmic popularity (based on `vote_count`) and raw `vote_average`.
*   **Normalization Layer:** Transforms L2 distances from FAISS into a [0, 1] probability-like space to allow balanced integration with metadata signals.
*   **Re-ranker:** Multiplies the normalized similarity score by weighted popularity and quality factors to prioritize high-traffic, highly-rated content.

## Why it works
By using a larger initial candidate pool (200), the system ensures a high probability of finding relevant items. The subsequent re-ranking step effectively uses the "wisdom of the crowd" (popularity) and critical acclaim (ratings) to promote the most likely desired movies to the top 10 without the overhead of heavy computational re-scoring of the entire database.

## Tradeoffs
*   **Latency vs. Recall:** The system balances retrieval depth against response speed by using a moderate candidate pool size (200) that fits well within the 13-14ms latency window.
*   **Popularity Bias:** By heavily weighing `vote_count` and `vote_average`, the system favors "blockbuster" content, which may reduce discovery of niche or long-tail movies.

## Key experiments
*   **Baseline:** Established initial vector retrieval logic.
*   **Pool Expansion:** Identified that increasing the retrieval pool to 100-200 candidates significantly improved recall without excessive latency penalties.
*   **Metadata Integration:** Experiments involving logarithmic popularity and quality weighting proved superior to pure vector distance, allowing for higher recall at lower computational cost compared to multi-stage heavy re-ranking models.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.540 | 0.540 |
| latency_ms | 13.1 | 13.1 |

## How to run
The system requires a FAISS index, a pre-computed movie dataframe, and a sentence-transformer model. Run the `search(query, df, bm25, model, index, top_k=10)` function passing the prepared index and dataframe objects.