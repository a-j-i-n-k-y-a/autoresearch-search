# Architecture

## What it does
The system performs keyword-based movie searches by matching user queries against the movie database. It identifies the most relevant titles and descriptions by calculating how often and how uniquely query words appear within the text, returning the top 10 matches.

## Components
- **BM25 Algorithm:** A robust statistical method that ranks documents based on term frequency and document length, serving as the core retrieval engine.
- **Pandas DataFrame:** An in-memory data store providing high-speed access to movie metadata (titles and overviews).
- **NumPy:** Utilized for high-performance numerical operations to sort search scores and extract top-ranking results efficiently.

## Why it works
The design relies on proven statistical information retrieval (BM25) rather than complex vector embeddings. By avoiding the overhead of external model inference and complex multi-stage reranking, the system maintains consistent recall while ensuring minimal execution time and zero operational cost.

## Tradeoffs
The system prioritizes **cost efficiency and simplicity** over incremental quality gains. Experiments showed that adding vector search, genre bias, or metadata boosting either increased latency or introduced costs without significantly improving recall, leading to the selection of the baseline configuration as the Pareto-optimal solution.

## Key experiments
- **Findings:** Over 20 iterations, no configuration surpassed the baseline recall of 0.600. 
- **Failures:** Hybrid search (BM25 + Faiss) and metadata-weighted boosting (vote_average) successfully reduced latency in isolated tests but failed to maintain the recall ceiling, often introducing unnecessary financial overhead.
- **Conclusion:** The complexity of neural-based reranking and feature-weighted scoring provided no measurable performance lift, confirming the baseline as the most efficient architecture.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.600 | 0.600 |
| latency_ms | 20.2 | 20.2 |
| llm_cost_usd | 0.000 | 0.000 |

## How to run
```bash
python agent_loop.py --eval-only
```