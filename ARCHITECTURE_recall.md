# Architecture

The system is an autonomously optimized movie search engine that leverages dense vector embeddings and metadata-aware ranking. It focuses on balancing semantic relevance with movie popularity and quality.

## What it does
The engine takes a natural language query, retrieves the most semantically similar movies from a pre-indexed vector space, and then re-ranks them using movie metadata (popularity and ratings). It returns a list of movies that are both relevant to the query and generally well-regarded by audiences.

## Components
- **FAISS Index**: Used for high-speed semantic retrieval of the top 500 candidate documents using L2 distance.
- **Sentence Transformer Model**: Encodes queries into vector embeddings.
- **Log-Scale Popularity Normalizer**: Processes `vote_count` using a log-transformation to prevent blockbuster movies from disproportionately dominating results while still favoring well-known films.
- **Rating Normalizer**: Scales `vote_average` (0-10) to a 0-1 range to act as a quality signal.
- **Weighted Scoring Engine**: Combines similarity, popularity (15% weight), and rating (15% weight) into a final ranking score.

## Why it works
- **Deep Retrieval**: Searching 500 candidates ensures that even if the most popular movies aren't the most semantically relevant, the system has a large enough pool to find high-quality matches.
- **Metadata Balancing**: By combining semantic similarity with normalized metadata, the system avoids "irrelevant but popular" and "relevant but obscure/poorly-rated" pitfalls.
- **Logarithmic Scaling**: Using `np.log1p` for vote counts accounts for the power-law distribution of movie popularity, making the signal useful across several orders of magnitude.

## Tradeoffs
- **Post-Retrieval Overhead**: Sorting and normalizing 500 candidates in Python adds a small latency overhead compared to returning raw FAISS results.
- **Metadata Dependency**: The ranking quality relies heavily on the presence and accuracy of `vote_count` and `vote_average`.
- **Static Weights**: The 15% weight for popularity and rating is fixed, which may not be ideal for all query types (e.g., searching for extremely niche or new films).

## Key experiments
The autonomous agent attempted 20 iterations to implement **Reciprocal Rank Fusion (RRF)** and **BM25 Hybrid Search**. However, all 20 attempts resulted in crashes (primarily `SyntaxError` and `ImportError`). Consequently, the system retained the stable baseline architecture, which provides a balance of semantic search and metadata boosting.

## Metrics

| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.510 | 0.510 |
| latency_ms | 12.6 | 12.6 |

## How to run
1. Ensure `numpy`, `pandas`, `faiss`, and `sentence_transformers` are installed.
2. Load your movie DataFrame (`df`), pre-trained FAISS `index`, and `model`.
3. Call the search function:
   ```python
   results = search("space exploration movies", df, bm25, model, index, top_k=10)
   ```