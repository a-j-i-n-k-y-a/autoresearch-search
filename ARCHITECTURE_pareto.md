# Architecture

## What it does
The system performs a high-performance, metadata-aware semantic search for movies. It retrieves candidates using a FAISS vector index and re-ranks them by blending semantic distance with popularity and quality signals to improve relevance while minimizing compute overhead.

## Components
*   **Vector Retrieval:** Uses a pre-trained encoder model to perform a k-NN search on 500 candidates via FAISS, optimized for low-latency retrieval.
*   **Normalization Layer:** 
    *   **Distance:** L2 distances are transformed into a similarity score $1.0 - (d / (1+d))$ to map values to a $[0, 1]$ range.
    *   **Popularity:** `vote_count` is processed via `log1p` and min-max scaled to prevent outliers from dominating the score.
    *   **Quality:** `vote_average` is normalized linearly by factor 10.
*   **Scoring Engine:** Calculates a weighted composite score: $Score = Similarity + (0.15 \times NormalizedPopularity) + (0.15 \times NormalizedRating)$.

## Why it works
The architecture avoids heavy multi-pass re-ranking (like BM25 + FAISS interleaving) which caused high latency in experiments. By expanding the retrieval pool to 500 candidates and applying a lightweight metadata injection during the ranking stage, the system achieves a 15.6% recall improvement while maintaining sub-10ms latency. The log-scaling of popularity ensures that popular movies are promoted without burying niche but highly relevant semantic matches.

## Tradeoffs
*   **Memory vs. Accuracy:** A fixed candidate pool size of 500 is used to maintain performance; very rare results outside this semantic neighborhood are sacrificed.
*   **Complexity:** The scoring function assumes metadata distribution remains stable; significant shifts in the rating/popularity distribution might require a recalibration of the 0.15 weights.

## Key experiments
*   **Baseline (Initial):** Pure FAISS retrieval yielded high latency due to inefficient post-processing.
*   **Candidate Pool Expansion:** Testing pool sizes showed that 500 is the "sweet spot" for balancing latency and recall.
*   **Metadata Integration:** Early attempts to use BM25 for ranking significantly increased latency (often >20ms); the decision to use simple scalar arithmetic for popularity and rating was critical to achieving the final performance.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.441 | 0.510 |
| latency_ms | 13.0 | 8.6 |

## How to run
1. Ensure `faiss`, `pandas`, and `numpy` are installed.
2. Load the pre-computed `index` and the movie `df`.
3. Pass the query and initialized components to `search(query, df, bm25, model, index)`.
4. The function returns a dictionary of the top 10 movies sorted by the composite metadata-aware score.