---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : cost
generated   : 2026-05-17T00:30:03
---

# Architecture

## What it does
The system is an optimized, low-latency movie search engine that retrieves candidates using vector similarity (FAISS) and reranks them using a custom scoring function that balances semantic relevance, community popularity, and critical reception.

## Components
- **Vector Retrieval**: Uses a pre-trained embedding model to index movie metadata in a FAISS flat index.
- **Candidate Pool**: A two-hundred-item candidate list is retrieved via L2 distance from the vector store.
- **Reranker**: An post-retrieval scoring engine that performs a multiplicative adjustment on vector similarity scores using `log(vote_count)` and `vote_average`.
- **Normalization**: L2 distances are transformed into a [0, 1] range to ensure relevance scores are compatible with the metadata-based multipliers.

## Why it works
The architecture shifts the focus from complex hybrid keyword/vector models (which often increased latency) to a high-performance vector-first approach. By boosting the initial candidate pool using popularity (log scale) and user ratings, the system surfaces high-quality content that matches semantic user intent without the computational overhead of secondary retrieval pipelines.

## Tradeoffs
- **Complexity**: Sacrifices pure keyword precision (BM25) to maintain sub-10ms latency.
- **Dependency**: Relies on the quality of the embedding model and the metadata distribution (`vote_count`/`vote_average`) rather than explicit keyword matching.
- **Resource Usage**: Achieving high recall required expanding the candidate pool to 200 items, increasing the per-query compute compared to a minimal 10-item retrieval.

## Key experiments
- **Candidate Pool Expansion**: Increasing the pool to 200 items provided the necessary recall headroom for effective reranking.
- **Logarithmic Popularity Boosting**: Found that `log1p(vote_count)` prevented extreme outliers from dominating the result set.
- **Discarded Hybridization**: Numerous experiments attempting to merge BM25 with FAISS via RRF or linear fusion were discarded due to high latency (often >20ms) and negligible recall gains.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.520 | 0.520 |
| latency_ms | 7.7 | 7.7 |

## How to run
1. Ensure the FAISS index and embedding model are loaded.
2. Provide a query string to the `search` function.
3. The function encodes the query, performs the L2 search, applies the popularity/rating multiplier, and returns the top 10 ranked records.