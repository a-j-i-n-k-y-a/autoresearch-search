# Architecture

## What it does
The system uses a semantic search approach to find movies. Instead of matching exact words, it converts user queries into mathematical representations (vectors) and identifies movies with similar representations. It then returns the most relevant results by performing a highly efficient similarity search against a pre-indexed collection of movies.

## Components
- **Encoder Model:** Translates text queries into a multi-dimensional numerical space.
- **FAISS Index:** A highly optimized data structure designed to perform near-instantaneous similarity lookups in large datasets.
- **Pandas DataFrame:** Serves as the lightweight metadata store for retrieving human-readable movie details once the index identifies the correct IDs.

## Why it works
The system achieves a pareto-optimal balance by moving away from heavy, multi-stage hybrid architectures (which often introduced latency overhead or failed during fusion) toward a streamlined, pure vector retrieval pipeline. Using multi-vector query processing allows the model to capture nuances in user intent, while the flattened index lookup ensures consistent, low-latency performance.

## Tradeoffs
The system prioritized **latency and recall efficiency** over complex, metadata-heavy reranking logic. By removing complex BM25 hybrid fusion and multi-field scoring, we sacrificed some potential custom business-logic weighting (like explicit popularity-based boosting) in favor of a faster, more robust semantic retrieval engine.

## Key experiments
- **Moved the needle:** The transition to "Multi-vector query expansion" provided the breakthrough to 0.900 recall, while "Simplified FAISS index search" maintained the sub-15ms latency target.
- **Failed approaches:** Numerous hybrid attempts (BM25 + FAISS fusion) resulted in high latency or frequent system crashes due to memory overhead and complex score normalization issues. Attempting to add genre-keyword boosts consistently degraded performance or led to instability.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.600 | 0.900 |
| latency_ms | 53.3 | 13.6 |

## How to run
```bash
python agent_loop.py --eval-only
```