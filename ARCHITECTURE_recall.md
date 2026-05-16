---
exp_id      : exp_464
prompt_hash : 3fd25a66
prompt_file : experiments/prompts/3fd25a66.txt
objective   : recall
generated   : 2026-05-17T04:07:07
---

# Architecture

## What it does
The system performs a multi-stage hybrid search to retrieve and rank movies. It leverages keyword-based retrieval for broad recall, followed by semantic re-ranking and a popularity-based bias to surface the most relevant and high-quality results.

## Components
1.  **BM25 Retrieval:** Uses the `bm25` index to retrieve an initial candidate pool of 1000 movies based on keyword overlap with the user query.
2.  **Semantic Re-ranking:** Performs a FAISS vector search on the full index to compute L2 distances, then selects the candidates found in the initial BM25 pool for semantic scoring ($1 / (1 + L2)$).
3.  **Popularity Adjustment:** Calculates a final score by scaling the semantic score with a popularity factor: `log1p(vote_count) * vote_average`.
4.  **Final Ranking:** Sorts the 1000 candidates by the combined score and returns the top_k results.

## Why it works
The hybrid approach compensates for the limitations of each retrieval method. BM25 ensures that specific terms (e.g., titles, keywords) are not missed, while FAISS handles conceptual matches that keywords might overlook. Adding a popularity boost ensures that the ranking doesn't just prioritize relevance, but also surface high-quality, widely recognized content.

## Tradeoffs
*   **Latency vs. Recall:** The final architecture increases latency by ~13ms compared to the simple FAISS baseline to achieve a 3% gain in recall. This is a deliberate trade-off, prioritizing search accuracy for user satisfaction.
*   **Compute:** Adding a two-stage re-ranking process increases the complexity of the query pipeline, requiring both a keyword lookup and a vector similarity lookup per request.

## Key experiments
*   **Initial Baseline:** Simple FAISS index search (Recall: 0.550, Latency: 8.2ms).
*   **Candidate Pool Expansion (exp_8):** Expanding the pool to 100 candidates improved recall to 0.451.
*   **Hybrid BM25 + FAISS (exp_464):** Combining BM25 keyword recall with FAISS semantic distance and popularity boosting yielded the winning recall of 0.580.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.550 | 0.580 |
| latency_ms | 8.2 | 21.6 |

## How to run
1. Ensure the `bm25` object, `model` (for embeddings), and `index` (FAISS) are initialized.
2. Load the movie DataFrame into memory.
3. Pass the query and artifacts to the `search(query, df, bm25, model, index)` function defined in `search.py`.
4. The function returns a dictionary of the top_k records.