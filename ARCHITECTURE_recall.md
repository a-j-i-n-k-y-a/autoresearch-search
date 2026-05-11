# Architecture

## What it does
The system combines two search methods to find movies: keyword matching (for exact terms) and meaning-based matching (for conceptual relationships). It retrieves a set of potential matches from both methods, calculates a score for each based on their relevance and popularity, and returns the top 10 results.

## Components
- **BM25Okapi:** Performs keyword-based retrieval to capture explicit search terms.
- **FAISS (Vector Search):** Uses a pre-trained model to encode queries and documents into vectors, allowing for semantic retrieval that understands synonyms and context.
- **Popularity Boost:** A multiplier derived from `vote_average` and `vote_count` that prioritizes highly-rated, well-known movies to refine the final ranking.

## Why it works
The design balances precision and speed. By pre-calculating vector embeddings, the system performs "approximate" searches in milliseconds. The hybrid approach mitigates the weaknesses of each individual method: BM25 finds exact matches that vector search might miss, while vector search identifies relevant results where the exact keywords don't appear. Scaling by popularity ensures that the highest-quality, most relevant content surfaces at the top of the result list.

## Tradeoffs
To achieve a high recall while maintaining low latency, we sacrificed exhaustive scoring. Instead of scoring the entire database, we retrieve a smaller, high-confidence candidate pool from both search methods and limit calculations to this subset. This "pruning" is why we maintained a $0.000435$ cost while significantly improving performance.

## Key experiments
- **Major Wins:** Removing redundant heavy computations and focusing on an efficient "Candidate-then-Boost" pipeline improved recall from 0.6 to 0.8 while cutting latency by over 85%.
- **Failed Attempts:** Attempts to integrate genre-based matching consistently resulted in system crashes or decreased recall, likely due to index mapping errors or over-constraining the search space. Complex Reciprocal Rank Fusion (RRF) approaches were discarded as they provided similar recall to the final logic but at a higher latency cost.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.600 | 0.800 |
| latency_ms | 169.5 | 25.2 |

## How to run
```bash
python agent_loop.py --eval-only
```