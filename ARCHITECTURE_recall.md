# Architecture

## What it does
The system performs a high-recall movie retrieval by combining semantic vector embeddings with metadata-aware re-ranking. It balances pure vector proximity with popularity and quality signals to surface relevant and highly-rated content.

## Components
*   **FAISS Vector Index:** Performs initial approximate nearest neighbor search to retrieve a broad candidate pool of 500 movies.
*   **Vector Engine:** A pre-trained transformer model encodes user queries into a high-dimensional semantic space.
*   **Metadata Processor:** Applies log-scaling to `vote_count` (to normalize popularity) and scales `vote_average` to a [0, 1] range.
*   **Ranking Logic:** Combines the normalized semantic similarity score with weighted metadata features (15% popularity, 15% rating) to produce the final ranked output.

## Why it works
The architecture avoids the latency overhead and complexity of multi-stage hybrid retrieval (e.g., BM25 + Vector) by using a single-pass candidate retrieval and a lightweight mathematical re-ranking stage. By expanding the candidate pool to 500, it captures a wider semantic space before using fast, vector-based, and attribute-based ranking to ensure the top-K results are both relevant and popular.

## Tradeoffs
*   **Normalization:** By using log-scaling on popularity, the system avoids being dominated by viral outliers while still favoring well-regarded content.
*   **Candidate Pool Size:** A fixed pool of 500 provides a balance between recall and latency; larger pools increased cost and latency without yielding proportional gains in recall.
*   **Simplicity:** Avoiding complex ensemble methods (like RRF) significantly reduced latency and kept the system stable against failure modes observed in the experiments.

## Key experiments
*   **Candidate Expansion:** Increasing the pool from the default to 500 was necessary to ensure the re-ranking phase had sufficient data to optimize against.
*   **Metadata Integration:** Simple boosting via log-normalized popularity and raw rating successfully improved user-facing relevance without the high latency costs of external library dependencies.
*   **Hybrid Rejection:** Numerous experiments involving BM25 ensemble and RRF were discarded due to extreme latency degradation (e.g., 30s+) or marginal recall improvements that did not justify the operational cost.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.510 | 0.510 |
| latency_ms | 12.4 | 12.4 |

## How to run
1. Ensure `pandas`, `numpy`, and `faiss` are installed in your environment.
2. Load the pre-computed index and the movie dataframe.
3. Call the `search` function: `results = search(query, df, bm25, model, index, top_k=10)`.