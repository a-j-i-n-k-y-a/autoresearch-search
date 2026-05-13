# Architecture

## What it does
The system performs high-speed semantic movie retrieval by combining vector similarity search with popularity-based re-ranking. It balances relevance with user-preferred content quality to provide a fast, high-quality discovery experience.

## Components
- **Vector Index (FAISS):** Executes efficient approximate nearest neighbor (ANN) search on movie embeddings.
- **Embedding Model:** Encodes user queries into vector space for semantic matching.
- **Popularity Scorer:** Applies a logarithmic transformation to `vote_count` to dampen the impact of extreme outliers while favoring established titles.
- **Re-ranker:** A post-processing logic that combines L2 distance from the FAISS index with popularity metrics to produce the final ranked list.

## Why it works
By retrieving an "oversized" candidate pool (5x `top_k`) from the vector index, the system creates enough variance in the result set to allow for effective re-ranking. The use of `np.log1p` on `vote_count` prevents highly popular titles from completely eclipsing relevant, niche titles, while the `0.05` scaling factor ensures that semantic relevance (distance) remains the primary driver of the search ranking.

## Tradeoffs
- **Candidate Pool Limitation:** By capping the search at 5x `top_k`, we trade potential global recall for significant latency improvements. 
- **Metadata Weighting:** The system assumes that raw `vote_count` is a reliable proxy for content quality, which may bias the system toward older, established movies over newer, less-reviewed ones.

## Key experiments
- **Over-fetching Candidates:** Experimenting with a 5x pool size proved critical for enabling re-ranking without sacrificing recall.
- **Logarithmic Popularity Scaling:** Initial tests using raw vote counts led to poor ranking diversity; the log-transformation successfully normalized the influence of popularity.
- **Hybrid Abandonment:** Attempts to combine BM25 and vector search (RRF or score weighting) were discarded due to high latency overhead ($>20ms$) without providing significant gains in recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.800 | 0.800 |
| latency_ms | 79.6 | 11.5 |

## How to run
1. Ensure `faiss` and `sentence-transformers` are installed.
2. Initialize the FAISS index with pre-computed movie embeddings.
3. Call `search(query, df, bm25, model, index, top_k=10)` passing the prepared dataframes and model objects.