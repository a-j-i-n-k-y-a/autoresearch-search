# Architecture

## What it does
The system implements a high-performance, popularity-aware semantic search engine for movies. It retrieves candidates using dense vector embeddings and performs real-time re-ranking by combining semantic similarity scores with log-transformed popularity metrics to balance relevance and user interest.

## Components
- **FAISS Index**: Performs rapid Approximate Nearest Neighbor (ANN) search on movie embeddings to retrieve a candidate pool ($5 \times \text{top\_k}$).
- **Encoder**: A pre-trained transformer model that maps natural language queries into the same vector space as the movie database.
- **Popularity Scorer**: Calculates a score based on $log(1 + \text{vote\_count})$ to normalize the impact of extremely popular movies without drowning out relevant niche results.
- **Ranker**: A fusion module that calculates a hybrid score: $\text{score} = \frac{1}{1 + \text{dist}} + (\text{pop} \times 0.05)$.

## Why it works
By over-fetching candidates (searching for 50 items to return 10), the system creates a search "buffer." This allows us to inject global metadata (popularity) into the ranking process without losing the semantic precision provided by the embedding model. The log-transformation of vote counts prevents "blockbuster bias" while ensuring highly-voted, relevant movies rank higher than obscure ones with similar semantic scores.

## Tradeoffs
- **Candidate Pool Size**: Fixed at 50 to maintain sub-15ms latency; increasing this would improve recall but degrade real-time performance.
- **Normalization Factor**: The 0.05 weight assigned to popularity is a heuristic; while effective for this dataset, it may require tuning if the distribution of `vote_count` changes significantly.

## Key experiments
- **Baseline**: Established the initial vector-only search performance.
- **Candidate Expansion**: Moving from 10 to 50 candidates proved essential for enabling effective re-ranking.
- **Popularity Integration**: Successfully replaced complex, high-latency hybrid models (like RRF or BM25 combinations) with a simple, computationally cheap log-normalized popularity boost.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.412 | 0.800 |
| latency_ms | 13.3 | 11.5 |

## How to run
1. Ensure the FAISS index is built and the `pandas` DataFrame is loaded.
2. Initialize the model encoder.
3. Pass the query to the `search()` function:
   ```python
   results = search("science fiction space odyssey", df, bm25, model, index, top_k=10)
   ```