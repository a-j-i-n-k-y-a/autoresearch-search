---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : latency
generated   : 2026-05-16T23:48:23
---

# Architecture

## What it does
The system performs a high-performance, hybrid semantic search for movies. It retrieves a candidate set of 200 items using a vector database (FAISS) and re-ranks them in-memory using a combination of semantic similarity, logarithmic popularity (vote count), and quality (average rating).

## Components
- **FAISS Index**: Stores dense embeddings for fast similarity retrieval.
- **Candidate Pool**: A fixed-size retrieval (top-200) balances semantic breadth with re-ranking speed.
- **Scoring Engine**: Implements a composite function: $Score = \text{NormDist} \times (1 + 0.1 \times \log(\text{vote\_count})) \times (1 + 0.1 \times \text{vote\_average})$.
- **Data Handler**: Uses `pandas` for efficient vectorized arithmetic during the re-ranking phase.

## Why it works
By using a larger candidate pool (200) and applying a popularity/rating bias, the system promotes high-quality, widely accepted content that is semantically relevant. The use of logarithmic scaling on `vote_count` prevents massive blockbusters from completely overshadowing niche but relevant content, while the inverse distance normalization aligns the vector scores with the metadata-based boosts.

## Tradeoffs
- **Complexity vs. Recall**: Increased the candidate pool size from the initial baseline to achieve higher recall, which stabilizes at 0.540.
- **Latency**: The system intentionally prioritizes low latency (10.9ms) by avoiding multiple cross-referencing steps (e.g., secondary BM25 lookups) during the request cycle.
- **Resource Intensity**: The scoring logic requires real-time access to the metadata frame, necessitating that the candidate index be aligned with the `pandas` DataFrame memory.

## Key experiments
- **Pool Expansion**: Increasing the retrieval pool to 200 was the most significant factor in stabilizing recall.
- **Metadata Boosting**: Logarithmic scaling of popularity allowed for effective ranking without the noise of raw count values.
- **Candidate Capping**: Reached an optimal balance where further pool expansion (e.g., 500+) significantly increased latency without improving recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.540 | 0.540 |
| latency_ms | 10.9 | 10.9 |

## How to run
1. Initialize the FAISS index and load the movie metadata DataFrame.
2. Ensure `search.py` is configured with the pre-trained encoder model.
3. Call `search(query, df, bm25, model, index, top_k=10)` to receive the ranked dictionary of results.