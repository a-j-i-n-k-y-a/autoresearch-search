---
exp_id      : exp_015
prompt_hash : 4e67b07c
prompt_file : experiments/prompts/4e67b07c.txt
objective   : recall
generated   : 2026-05-19T13:27:39
---

# Architecture

## What it does
The system performs a multi-stage hybrid search by fusing lexical (BM25) and semantic (vector similarity) retrieval. It retrieves 500 candidates from each modality, merges them into a candidate pool, and calculates a final ranking score based on a weighted blend of normalized BM25 scores, semantic L2 similarity, and a log-transformed popularity boost.

## Components
*   **BM25 Retrieval:** Uses `BM25Okapi` with custom tokenization (`\b\w\w+\b`) to score candidate relevance based on keyword overlap.
*   **Semantic Retrieval:** Uses FAISS indexing to perform an L2 distance search on pre-computed model embeddings.
*   **Scoring & Fusion Engine:** Implements a normalized weighted sum: `(0.4 * BM25) + (0.6 * Semantic) + (0.1 * Popularity)`.
*   **Popularity Boost:** Applies a `vote_average * log1p(vote_count)` transformation to surface high-quality, popular content without overwhelming relevance signals.

## Why it works
*   **Hybrid Candidate Pooling:** By merging 500 BM25 results with 500 semantic results, we observed an increase in recall from 0.500 to 0.682 (+0.182), successfully capturing both keyword-specific matches and broad semantic concepts.
*   **Normalization:** Explicit MinMax scaling of BM25 scores and L2 similarity mapping ensured that disparate numeric ranges were balanced, preventing the popularity score from dominating the ranking.

## Tradeoffs
*   **Latency:** The additional computational cost of normalization and multi-factor scoring introduces a minor latency penalty of 1.0ms (23.3ms to 24.3ms).
*   **Complexity:** The system relies on fixed weights (0.4/0.6/0.1), which may require recalibration if the underlying embedding model or dataset distribution changes significantly.

## Key experiments
*   **exp_015 (Final):** Combined exact text-field BM25, semantic L2 similarity, and genre-aware popularity boosting. This achieved a recall of 0.682, significantly outperforming the baseline recall of 0.500.
*   **RRF Explorations:** Several attempts to replace weighted summation with Reciprocal Rank Fusion (e.g., experiments 2, 4, 13) resulted in lower recall (ranging 0.091–0.545), suggesting that linear weighting with normalization provided more stable ranking performance for this specific distribution.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.500 | 0.682 |
| latency_ms | 23.3 | 24.3 |

## How to run
1. Ensure the environment contains `rank_bm25`, `numpy`, and `faiss`.
2. Load the movie dataset into a pandas DataFrame.
3. Initialize the BM25 object and the FAISS index.
4. Call `search(query, df, bm25, model, index)` to retrieve the top-k ranked results.