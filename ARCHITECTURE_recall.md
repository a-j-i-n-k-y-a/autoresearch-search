# Architecture

## What it does
The system performs high-performance movie retrieval by combining semantic vector search with metadata-driven re-ranking. It balances pure relevance with user-preference signals (popularity and ratings) to deliver highly relevant results within sub-10ms latency.

## Components
- **FAISS Index**: Handles high-speed approximate nearest neighbor (ANN) retrieval using embeddings of movie overviews.
- **Semantic Encoder**: A pre-trained model mapping natural language queries to vector space.
- **Metadata Processor**: Normalizes `vote_count` (using a log-scale) and `vote_average` to ensure signal from popularity and quality is captured without drowning out semantic relevance.
- **Scoring Engine**: Implements a weighted hybrid score: $Score = \text{Norm}(\text{L2\_dist}) + 0.15 \cdot \text{Norm}(\text{Log\_Pop}) + 0.15 \cdot \text{Norm}(\text{Rating})$.

## Why it works
By expanding the candidate pool to 500 in the initial FAISS search, the system captures a broader "long-tail" of relevant items. The subsequent re-ranking step uses light-weight metadata normalization to prioritize high-quality, popular content, which directly correlates with user satisfaction, effectively refining the vector retrieval results.

## Tradeoffs
- **Candidate Pool Size**: Increasing the pool to 500 slightly increases memory usage during retrieval but significantly improves recall.
- **Simple Re-ranking**: We explicitly avoided heavy BM25/hybrid indexing in the final version to maintain sub-10ms latency, favoring arithmetic adjustments to existing metadata over costly text-based re-scoring.

## Key experiments
- **Pool Expansion**: Expanding the FAISS search depth was critical to surpassing the baseline recall (0.441 to 0.510).
- **Metadata Integration**: Normalizing popularity and ratings allowed for a more robust ranking than vector distance alone.
- **Elimination of complex hybrids**: Most complex BM25/RRF fusion experiments were discarded due to high latency (often >20ms) and poor returns on recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.451 | 0.510 |
| latency_ms | 15.1 | 8.6 |

## How to run
1. Ensure `faiss`, `pandas`, and `numpy` are installed.
2. Load the pre-trained model and initialize the FAISS index from your processed dataset.
3. Call `search(query, df, bm25, model, index)` to retrieve the top 10 movies.