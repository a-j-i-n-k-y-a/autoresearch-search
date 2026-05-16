---
exp_id      : exp_314
prompt_hash : f5920c6d
prompt_file : experiments/prompts/f5920c6d.txt
objective   : recall
generated   : 2026-05-17T00:13:26
---

# Architecture

## What it does
The system performs semantic movie search by retrieving a broad candidate pool via FAISS vector similarity, then refining the ranking using a popularity-aware hybrid scoring function. It prioritizes relevant content while elevating high-visibility, well-voted titles to ensure user-friendly results.

## Components
*   **Vector Index (FAISS):** Performs high-speed semantic retrieval on movie embeddings.
*   **Candidate Pool:** Fetches the top 200 candidates from the vector index to ensure sufficient breadth for re-ranking.
*   **Re-ranking Engine:** Applies a log-transformed popularity boost (`log1p(vote_count)`) to the raw L2 distance scores.
*   **Scoring Function:** Calculates the final score as `score = -L2_distance + (0.5 * log_popularity)`, converting the distance-based metric into a standard "higher-is-better" ranking.

## Why it works
The system balances two distinct signals:
1.  **Relevance:** The raw L2 distance ensures that the search results are semantically aligned with the user query.
2.  **Quality/Popularity:** By using `log1p` on `vote_count`, the system incorporates social proof without allowing extreme outliers (blockbusters) to completely drown out relevant, niche content. The negative L2 distance correctly aligns the semantic similarity with the popularity boost.

## Tradeoffs
*   **Candidate Size:** Expanding the pool to 200 candidates provides a better search surface than the baseline but increases memory overhead slightly.
*   **Complexity:** The introduction of a popularity coefficient (0.5) requires empirical tuning; if set too high, it leads to popularity bias; if set too low, the system relies exclusively on semantic similarity, which may surface low-quality results.

## Key experiments
*   **Candidate Pool Expansion (exp_314 context):** Increasing the pool size from the default to 200 candidates was essential for stabilizing recall.
*   **Log-Popularity Integration:** The winning experiment (f5920c6d) successfully integrated a logarithmic popularity boost, which significantly outperformed linear boosting or pure vector search in user-preference alignment.
*   **Constraint Handling:** Numerous experiments involving complex metadata filtering and multi-stage ranking were discarded due to increased latency without proportional improvements in recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.441 | 0.540 |
| latency_ms | 18.8 | 9.1 |

## How to run
1. Ensure the environment has `faiss-cpu`, `numpy`, and `pandas` installed.
2. Load the pre-trained model and the FAISS index.
3. Call `search(query, df, bm25, model, index, top_k=10)` from `search.py`.