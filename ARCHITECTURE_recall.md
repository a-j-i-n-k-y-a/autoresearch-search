# Architecture

## What it does
The system performs high-performance semantic movie retrieval by combining dense vector representations with a popularity-informed re-ranking heuristic. It queries a FAISS index to identify a broad candidate pool of semantically relevant films, which are then refined using metadata-driven scoring to ensure user satisfaction.

## Components
*   **Vector Engine (FAISS):** Encodes user queries into embedding space to perform approximate nearest neighbor search.
*   **Expansion Logic:** Retrieves a larger initial candidate pool ($k=100$) to mitigate the "lost in retrieval" problem of standard vector search.
*   **Popularity Re-ranker:** A custom scoring function that combines normalized semantic distance with a log-scaled popularity boost (`vote_average` * `log1p(vote_count)`).
*   **Inference Pipeline:** A streamlined Python function (`search.py`) optimized for minimal overhead.

## Why it works
The architecture succeeds by shifting the balance from complex hybrid searching (which introduced excessive latency) to an "expand-then-refine" strategy. By increasing the candidate pool to 100, we recover more relevant documents that would otherwise be filtered out. The subsequent re-ranking using a log-scaled popularity boost effectively surfaces high-quality, widely recognized content, while removing the overhead of redundant BM25 calculations.

## Tradeoffs
*   **Precision vs. Compute:** By using a larger initial candidate pool, we increase the semantic recall at the cost of processing slightly more metadata rows per request.
*   **Simplicity:** The system intentionally avoids complex multi-modal fusion (e.g., RRF, BM25) which proved computationally expensive and detrimental to recall in this specific environment.
*   **Popularity Bias:** The ranking heuristic favors established, highly-rated films, which may slightly reduce the visibility of niche or newer content.

## Key experiments
*   **Candidate Pool Expansion (The Winner):** Simply increasing the FAISS search radius to 100 significantly outperformed all hybrid BM25 combinations, proving that retrieval recall was the primary bottleneck.
*   **Hybrid Inefficiency:** Multiple attempts to integrate BM25 scores (either via RRF or linear weighting) consistently increased latency (>20ms) while degrading recall, likely due to feature misalignment between sparse and dense vectors.
*   **Normalization Risks:** Attempting to force L2 normalization or complex ranking math often resulted in instability or severe recall drops, reinforcing the need for the robust, simple heuristic used in the final version.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.441 | 0.451 |
| latency_ms | 18.8 | 8.1 |

## How to run
1. Ensure the environment has `faiss-cpu`, `numpy`, and `pandas` installed.
2. Load the pre-computed `index` and `df` (movie metadata).
3. Import `search` from `search.py`.
4. Invoke the function: `search(query="your search term", df=df, bm25=None, model=model, index=index)`.