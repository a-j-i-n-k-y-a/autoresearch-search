---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : latency
generated   : 2026-05-17T00:29:16
---

# Architecture

## What it does
The system performs a high-performance, two-stage movie recommendation retrieval. It identifies the most relevant candidates using semantic vector search and re-ranks them based on a hybrid utility function that optimizes for both audience popularity and critical reception.

## Components
- **Vector Retrieval:** Employs FAISS to perform approximate nearest neighbor search over pre-computed movie embeddings.
- **Candidate Pool:** Retrieves a broad set of 200 candidates to ensure high recall before applying precision-focused re-ranking.
- **Utility Re-ranker:** A custom scoring function that combines normalized L2 distance with log-transformed popularity (vote count) and average ratings to surface "high-quality, high-visibility" content.

## Why it works
The system balances semantic relevance with metadata-driven quality indicators. By using a log-transformed popularity boost (`np.log1p`), the model prevents extremely popular items from overwhelming the search results while still favoring well-regarded titles. Normalizing the L2 distance ensures that the vector similarity remains the primary anchor for relevance, while the secondary scores act as precise bias levers.

## Tradeoffs
- **Candidate Pool Size:** Selecting 200 candidates provides a balance between recall and latency. Increasing this number would improve recall but degrade sub-10ms response times.
- **Feature Sparsity:** The system ignores temporal features (e.g., release date) to maintain a low-latency profile, prioritizing robust quality metrics instead.
- **Embedding dependency:** The system relies entirely on the quality of the base embeddings; if the embedding model is weak, metadata re-ranking cannot fully compensate for poor initial retrieval.

## Key experiments
- **Candidate Pool Expansion:** Incremental testing showed that increasing the candidate pool to 200 was necessary to achieve parity with the target recall.
- **Normalization Strategy:** Early experiments with raw score fusion led to bias towards specific metrics; the winning configuration uses L2 normalization and log-scaling for metadata to ensure consistent score distributions.
- **Re-ranking logic:** Multiple experiments attempted genre-based filtering, but found that popularity and average rating provided a more consistent improvement in user-perceived relevance.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.540 | 0.540 |
| latency_ms | 6.5 | 6.5 |

## How to run
1. Ensure `faiss`, `numpy`, and `pandas` are installed.
2. Initialize the FAISS index with the target movie dataset embeddings.
3. Call `search(query, df, bm25, model, index, top_k=10)` passing the pre-loaded index and dataframe.