# Architecture

## What it does
The system performs an efficient two-stage movie search. First, it uses semantic vector matching to identify a broad set of relevant candidates based on the user's query. Second, it applies a lightweight ranking logic that boosts movies in genres matching the user's query and prioritizes highly popular titles (based on vote count) to ensure the most relevant results appear at the top.

## Components
*   **Faiss Index:** A vector database that handles high-speed similarity search, allowing the system to find relevant movies in milliseconds.
*   **Encoder Model:** Translates user queries into numerical embeddings to bridge the gap between natural language and movie metadata.
*   **Candidate Re-ranker:** A post-processing script that applies a heuristic "genre boost" and rank-based popularity weighting to the top candidates retrieved by the index.

## Why it works
The design succeeds by offloading the heavy lifting of "meaning" to the vector index, while using simple, deterministic post-processing for precision. By querying 5x the desired top-k results from the index and then re-sorting locally, we maintain low latency while significantly improving recall compared to a raw similarity search.

## Tradeoffs
The system prioritizes **latency and recall (Pareto efficiency)** over architectural complexity. It deliberately discards hybrid BM25 approaches, which proved too computationally expensive, in favor of a "Vector + Rerank" strategy that maintains sub-11ms response times while meeting the target recall.

## Key experiments
*   **Successes:** Vector search combined with genre-matching boosts provided the most significant gains in recall (moving from 0.6 to 0.8). Expanding the retrieval pool to 5x top-k before reranking was essential for finding the right candidates.
*   **Failures:** All experiments involving BM25 hybrids or complex Min-Max scaling increased latency significantly (up to 81ms) without providing proportional gains in retrieval accuracy. Pure vector search without reranking failed to provide the necessary precision for the top-10 results.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.600 | 0.800 |
| latency_ms | 14.6 | 10.9 |
| llm_cost_usd  | 0.000000 | 0.000316 |

## How to run
```bash
python agent_loop.py --eval-only
```