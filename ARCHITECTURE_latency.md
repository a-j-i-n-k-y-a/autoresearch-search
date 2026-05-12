# Architecture

## What it does
The system performs an ultra-fast semantic search by converting user queries into mathematical vectors and matching them directly against a pre-indexed collection of movie data. It skips traditional keyword analysis to provide instant search results.

## Components
*   **Encoder Model**: Converts text queries into high-dimensional vectors, capturing the conceptual meaning rather than just matching words.
*   **FAISS Index**: A specialized, high-performance database designed for similarity searching, allowing the system to find the nearest vectors in milliseconds.
*   **DataFrame (`df`)**: A lightweight lookup table that holds the metadata for the retrieved movie IDs, acting as the final data fetcher.

## Why it works
The design works by eliminating heavy compute steps like BM25 text tokenization and complex reranking logic. By delegating the search entirely to the FAISS index (a highly optimized C++ backend), the system avoids Python-level loops and minimizes the memory overhead associated with managing secondary data structures.

## Tradeoffs
To achieve a ~10x improvement in latency, the system sacrificed 33% of its original recall. The architecture prioritizes speed (the system’s primary objective) by favoring "good enough" semantic matching over the precision of hybrid keyword-and-vector reranking.

## Key experiments
*   **The "Simplify" Shift**: Moving from complex hybrid RRF/BM25 pipelines to a single, direct FAISS vector search was the catalyst for latency reduction. 
*   **Failed Approaches**: Attempts to boost accuracy using genre-keyword filtering and popularity weighting consistently crashed the system or introduced prohibitive overhead (e.g., the 4.6s latency spike during multi-field BM25 integration). 
*   **Winner**: The final version—a raw `index.search` call—stripped away all conditional logic, resulting in the most performant state at 8.6ms.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.900 | 0.600 |
| latency_ms | 87.5 | 8.6 |

## How to run
```bash
python agent_loop.py --eval-only
```