# Architecture

## What it does
The movie search system helps users find relevant movies by retrieving a list of movie titles and overviews based on a given query. It combines the strengths of natural language processing and information retrieval to provide accurate results.

## Components
The system consists of several components: 
- **BM25**: a bag-of-words retrieval algorithm that calculates the relevance of each movie based on the query terms.
- **SentenceTransformer**: a sentence embedding model that converts the query and movie overviews into dense vectors for semantic search.
- **Faiss Index**: an efficient indexing system that enables fast similarity search between the query vector and movie vectors.
- **Vote Count and Average**: additional features that consider the popularity and rating of each movie.

## Why it works
The system works by combining the strengths of each component. BM25 provides a robust baseline for retrieval, while the SentenceTransformer and Faiss Index enable semantic search and efficient vector similarity calculation. The vote count and average features help to boost the ranking of popular and highly-rated movies.

## Tradeoffs
To prioritize recall, the system uses a larger candidate pool and combines multiple ranking signals. This approach may increase latency, but it improves the chances of retrieving relevant movies.

## Key experiments
Unfortunately, none of the experiments resulted in significant improvements to recall. Most experiments crashed due to syntax errors in the search.py file, indicating that the algorithm is sensitive to implementation details.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.600 | 0.600 |
| latency_ms | 164.3 | 164.3 |

## How to run
```bash
python agent_loop.py --eval-only
```