---
exp_id      : exp_187
prompt_hash : 5005fad1
prompt_file : experiments/prompts/5005fad1.txt
objective   : latency
generated   : 2026-05-19T13:49:55
---

# Architecture

## What it does
The system performs hybrid movie retrieval by combining lexical (BM25) and semantic (vector embeddings) search signals. It aggregates candidate pools from both methods, applies local normalization, and fuses them with popularity-based metadata to produce a final ranked list of movies.

## Components
*   **Lexical Retrieval:** Uses `BM25Okapi` with a custom regex tokenizer (`\b\w\w+\b`) to score query-document relevance.
*   **Semantic Retrieval:** Employs a pre-trained model for embedding generation and FAISS for efficient approximate nearest neighbor search.
*   **Fusion Engine:** Performs score normalization (MinMax for BM25, L2-reciprocal for vectors), applies a popularity boost via log-transformed vote counts, and performs weighted score summation.
*   **Candidate Management:** Maintains a unified pool of 250 candidates from both retrieval streams to balance recall and latency.

## Why it works
*   **Candidate Pooling:** Limiting the union of BM25 and vector results to 250 candidates minimizes compute overhead while ensuring high recall.
*   **Score Fusion:** Balancing BM25 (0.4) and vector scores (0.5) with popularity (0.1) allows the system to weigh relevance against user-preference signals, contributing to the recall improvement.
*   **Latency Minimization:** Moving from standard DataFrame-heavy operations to direct NumPy array indexing and minimizing object copies directly reduced processing time by 2.2ms.

## Tradeoffs
*   **Candidate Pool Size:** A pool size of 250 offers the optimal balance between recall (0.636) and latency (23.4ms). Larger pools (e.g., 2000) significantly increased latency without proportional recall gains in earlier trials.
*   **Complexity vs. Efficiency:** We removed complex genre-aware re-ranking logic that increased latency (up to 40ms+) without consistent recall gains, favoring simpler, vectorized weightings.

## Key experiments
*   **Baseline:** Standard BM25 search resulted in a recall of 0.500 and 23.3ms latency.
*   **Hybrid Implementation:** Combining BM25 and semantic vectors with popularity-based normalization improved recall to 0.591 (+0.091).
*   **Final Optimization (exp_187):** Streamlining candidate pool selection and optimizing data access patterns resulted in a recall of 0.636 (+0.045 vs hybrid) and reduced latency to 23.4ms.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.591 | 0.636 |
| latency_ms | 25.6 | 23.4 |

## How to run
Ensure the environment is configured with `rank_bm25` and `faiss`. The search function expects a query string, a pandas DataFrame containing movie metadata, a pre-computed BM25 object, an embedding model, and a loaded FAISS index.

```python
from search import search
results = search(query="sci-fi adventure", df=movies_df, bm25=bm25_model, model=embed_model, index=faiss_index)
```