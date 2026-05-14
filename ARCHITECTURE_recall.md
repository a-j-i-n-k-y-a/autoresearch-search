---
exp_id      : exp_069
prompt_hash : 1bffc102
prompt_file : experiments/prompts/1bffc102.txt
objective   : recall
generated   : 2026-05-14T20:09:17
---

# Architecture

## What it does
The system performs high-performance semantic movie retrieval by mapping natural language user queries to a vector embedding space. It leverages a pre-computed FAISS index to identify the most contextually relevant titles based on a learned embedding model, delivering results with low latency.

## Components
- **Encoder Model:** A sentence transformer that generates a `float32` vector representation of the user's input query.
- **FAISS Index:** A high-speed similarity search structure that performs a K-Nearest Neighbors (KNN) lookup against movie vectors.
- **Data Store:** A Pandas DataFrame containing pre-processed movie metadata, mapped to indices in the FAISS structure.
- **Search Engine:** A `search.py` interface that performs encoding, index lookup, and record retrieval in a single pass.

## Why it works
The system relies on pure dense retrieval. By focusing on an optimized vector search (eliminating hybrid reranking or multi-stage pipelines), it minimizes compute overhead. The "winning" configuration balances the precision of semantic embeddings with the extreme efficiency of the FAISS engine, achieving high recall without the latency penalties observed in complex hybrid approaches.

## Tradeoffs
- **Latency vs. Complexity:** By discarding hybrid methods (BM25 fusion/reranking), the system achieves sub-10ms latency but loses the exact-keyword matching benefits that traditional text search engines provide.
- **Data Dependencies:** The system is heavily dependent on the quality of the initial embedding model; if the model fails to represent niche queries, there is no "fallback" text-based mechanism.

## Key experiments
- **`exp_069` (Winning):** Implemented a streamlined metadata rescale that maintained optimal recall while optimizing for search speed and computational cost.
- **`baseline`:** Established the initial recall benchmark and high-latency threshold.
- **`pool_expansion` variants:** Multiple experiments demonstrated that expanding the candidate pool beyond an optimal threshold (around 100) increased latency and cost without improving recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.510 | 0.510 |
| latency_ms | 10.8 | 8.6 |

## How to run
1. Ensure `faiss` and `numpy` are installed.
2. Initialize the `model` (SentenceTransformer) and load the pre-computed `index` and `df` (DataFrame).
3. Call the `search` function:
```python
results = search(query="Sci-fi space travel", df=df, bm25=None, model=model, index=index, top_k=10)
```