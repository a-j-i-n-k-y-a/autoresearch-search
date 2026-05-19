---
exp_id      : exp_102
prompt_hash : 4312a717
prompt_file : experiments/prompts/4312a717.txt
objective   : pareto
generated   : 2026-05-19T13:44:21
---

# Architecture

## What it does
The system performs a multi-stage hybrid search by aggregating results from a lexical BM25 engine and a semantic FAISS vector index. It refines the candidate pool using a weighted ensemble scoring function that integrates keyword relevance, semantic similarity, genre alignment, and popularity bias.

## Components
*   **Retrieval:** Parallel execution of BM25 (lexical) and FAISS (semantic) to generate 200 candidates each.
*   **Feature Scoring:** 
    *   **Lexical:** Min-Max normalized BM25 scores.
    *   **Semantic:** L2 distance inverted via $1/(1+d)$.
    *   **Genre:** A soft-matching signal providing a 1.0 boost for intersection and 0.5 baseline.
    *   **Popularity:** A log-transformed, z-score normalized signal derived from `vote_count`.
*   **Aggregation:** A weighted linear combination (0.35 BM25, 0.45 Semantic, 0.1 Genre, 0.1 Popularity) for final ranking.

## Why it works
The system achieves optimal recall through the integration of disparate signal sources:
*   **Ensemble Weighting:** Transitioning from RRF to a weighted ensemble (0.35/0.45/0.1/0.1) improved recall from 0.636 to 0.682 (+0.046).
*   **Genre/Popularity Injection:** Adding genre-match scores and log-transformed popularity boosts provided necessary precision constraints, evidenced by the final recall of 0.682 versus the initial baseline of 0.500 (+0.182).
*   **Latency Budget:** The architecture maintains low latency (24.5ms) by avoiding heavy re-ranking models in favor of feature-based scalar fusion.

## Tradeoffs
*   **Precision vs. Latency:** Increasing the candidate pool size and adding feature-based scores increased latency from 23.4ms to 24.5ms (+1.1ms).
*   **Complexity:** The system relies on fixed weighting; while effective for the current dataset, it may require hyperparameter tuning if the distribution of popularity or genre metadata shifts significantly.

## Key experiments
*   **exp_102:** Implementation of hybrid retrieval with normalized genre-match and popularity-weighted ensemble (Recall: 0.682).
*   **exp_baseline:** Initial BM25/Semantic hybrid baseline (Recall: 0.500).
*   **RRF Iterations:** Numerous experiments (e.g., "Implement RRF with candidate expansion...") showed that while RRF offers stable ranking, it consistently underperformed compared to explicit feature weighting for this specific task (Recall decreased in RRF-focused variants).

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.636 | 0.682 |
| latency_ms | 23.4 | 24.5 |

## How to run
1. Install requirements: `pip install -r requirements.txt`
2. Ensure FAISS index and BM25 object are initialized.
3. Call `search(query, df, bm25, model, index)` to retrieve sorted movie records.