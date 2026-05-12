# Architecture

## What it does
The system performs a semantic search by converting user queries into numerical representations (vectors) to find similar movies. It then refines the results by boosting movies that match the requested genre and prioritizing popular titles based on their total vote counts.

## Components
- **FAISS Index:** A high-speed similarity search engine that narrows down a massive library of movies to a small set of semantic matches.
- **Sentence Transformer Model:** Converts raw text queries into vectors that capture the intent and context of a search.
- **Pandas DataFrame:** Serves as the metadata store, holding genres and popularity (vote count) statistics used for final ranking.
- **Genre/Popularity Re-ranker:** A post-processing logic layer that applies a multiplier to the ranking score if a movie matches the user's requested genre, and weights results by popularity.

## Why it works
By using FAISS for the initial retrieval, the system achieves instant matching across thousands of entries. The "genre boost" acts as a heuristic filter that aligns semantic results with user expectations, while the popularity-based ranking ensures that highly-rated or widely-watched movies occupy top positions, effectively balancing intent with authority.

## Tradeoffs
The architecture prioritizes **Recall** above all else. During optimization, complex hybrid methods (like combining BM25 and FAISS) often introduced latency or logic errors that degraded search quality. By opting for a clean, vector-first approach with simple metadata boosting, the system maintains consistent performance without the cost or complexity of multi-stage retrieval.

## Key experiments
- **Failures:** Most attempts to integrate BM25 (keyword search) resulted in a lower recall (dropping from 0.8 to 0.6) and increased cost, suggesting that the semantic model already captures sufficient keyword intent.
- **Successes:** Simple FAISS retrieval consistently outperformed more complex multi-step fusions. Experiments involving score scaling (Min-Max) improved recall to 0.7, but could not surpass the original baseline performance of 0.8.
- **Crashes:** Attempts to perform heavy mathematical combinations of scores (e.g., product of BM25 and FAISS scores) led to runtime crashes, highlighting the stability of the current implementation.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.800 | 0.800 |
| latency_ms | 54.6 | 54.6 |
| llm_cost_usd | 0.000000 | 0.000000 |

## How to run
```bash
python agent_loop.py --eval-only
```