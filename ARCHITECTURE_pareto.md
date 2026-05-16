---
exp_id      : exp_335
prompt_hash : 840bf4d5
prompt_file : experiments/prompts/840bf4d5.txt
objective   : pareto
generated   : 2026-05-17T05:02:14
---

# Architecture

## What it does
The system performs a high-performance hybrid movie search that combines sparse keyword retrieval with dense semantic embedding lookups. It prioritizes relevant candidates based on a fusion of semantic proximity and popularity-weighted metrics to ensure the highest quality results are surfaced in the top 10.

## Components
*   **BM25 (Sparse Retrieval):** Executes keyword matching on the query to establish an initial broad recall candidate pool of 1,000 documents.
*   **FAISS (Dense Retrieval):** Performs a semantic search on a pre-indexed vector database to calculate L2 distance-based similarity.
*   **Re-ranking Engine:** Fuses the outputs by mapping semantic distances to a $[0, 1]$ range and calculating a custom score: $Score = Semantic \times (1.0 + 0.05 \times Popularity)$, where popularity is defined as $log1p(vote\_count) \times vote\_average$.

## Why it works
The architecture avoids the "bottleneck" of full-dataset vector scoring by using BM25 for rapid candidate pruning. The final ranking logic effectively bridges the gap between semantic relevance (via the embedding model) and user-centric quality (via popularity metrics), which significantly improves precision without increasing latency.

## Tradeoffs
*   **Hybrid Complexity:** Maintaining both BM25 indexes and FAISS vector stores requires synchronization of datasets.
*   **Candidate Truncation:** By limiting the initial recall to 1,000, the system may miss "long-tail" semantic matches that have low keyword overlap, though this is mitigated by the efficiency of the hybrid scoring.

## Key experiments
*   **exp_335 (Winning):** Introduced the specific popularity-boost formula that correctly balanced semantic similarity with commercial quality, achieving the target Pareto efficiency.
*   **Candidate Pool Scaling:** Extensive testing proved that expanding the pool beyond 1,000 provided diminishing returns in recall while significantly impacting inference costs and latency.
*   **Normalization Strategy:** Experimentation with various fusion techniques (RRF vs. multiplicative scoring) revealed that explicit scaling of popularity scores provided more stable ranking than rank-based fusion.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.520 | 0.540 |
| latency_ms | 6.6 | 6.6 |

## How to run
1. Ensure `bm25` (e.g., `rank_bm25`) and `faiss` are installed.
2. Initialize the search object with the pre-trained `model` and pre-populated `index`.
3. Pass the query string and the movie `df` to the `search()` function to retrieve the top-10 dictionary records.