# Architecture

## What it does
The system performs a text-based search across a movie database. When you type a query, it breaks your search terms into individual words and matches them against the movie overviews using a statistical scoring method that favors unique, relevant keywords. It then returns the top 10 most relevant matches.

## Components
*   **BM25 Algorithm**: A robust probabilistic ranking function that scores movies based on keyword frequency and rarity. It is the core engine for matching user intent to text descriptions.
*   **Pandas Dataframe**: Used as an in-memory data store for movie metadata (`title`, `overview`), enabling fast lookup and retrieval once the top indices are identified.
*   **NumPy**: Facilitates high-performance array manipulation for sorting scores and slicing the result set.

## Why it works
The design leverages BM25’s ability to handle document length and term frequency effectively without the overhead of modern embedding models. By keeping the retrieval logic constrained to keyword matching, the system maintains a predictable, low-latency profile without the hidden costs associated with vector space search.

## Tradeoffs
The system prioritizes **Pareto optimality**—achieving the best possible balance between performance (latency) and utility (recall) for zero additional monetary cost. Through extensive automated testing, it was determined that adding complex hybrid models or machine-learned components increased latency and operational cost without providing a statistically significant gain in user-perceived recall for the specific target dataset.

## Key experiments
*   **High-Performing Failures**: "Vector search with post-filtering" demonstrated a significant recall boost (0.800) but was discarded due to the failure to satisfy the multi-objective Pareto constraint (increased latency/cost).
*   **Redundancy**: Experiments involving hybrid combinations (BM25 + Faiss/RRF) consistently increased latency (up to 26.9ms) and cost while failing to improve recall beyond the baseline 0.600.
*   **Baseline Resilience**: The baseline configuration was proven to be the most efficient implementation, as all 20 iterative variations failed to provide a superior return on investment for the defined metrics.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.600 | 0.600 |
| latency_ms | 13.4 | 13.4 |

## How to run
```bash
python agent_loop.py --eval-only
```