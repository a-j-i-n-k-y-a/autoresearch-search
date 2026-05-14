---
exp_id      : exp_293
prompt_hash : 2d2af823
prompt_file : experiments/prompts/2d2af823.txt
objective   : latency
generated   : 2026-05-14T20:06:11
---

# Architecture

## What it does
The system performs high-speed semantic movie retrieval by mapping natural language queries into a pre-computed vector space. It uses a lean FAISS-based index to retrieve relevant movie candidates based on embedding similarity, bypassing traditional keyword-based scoring for maximum execution efficiency.

## Components
*   **Encoder Model:** Uses a transformer-based model to convert text queries into dense embeddings.
*   **Vector Index:** A memory-resident FAISS index containing pre-computed embeddings of movie titles and overviews.
*   **Search Engine:** A `search.py` script that tokenizes the query via the encoder and performs a k-nearest neighbor (kNN) lookup against the FAISS index to return the top_k relevant records.

## Why it works
By stripping away complex hybrid reranking, metadata filtering, and multi-stage scoring pipelines, the architecture eliminates significant computational overhead. The performance gain is primarily driven by minimizing data transformation steps and reducing the instruction path length in the core retrieval loop.

## Tradeoffs
*   **Latency vs. Recall:** The move to a "pure" FAISS approach drastically reduces latency (from 10.5ms to 6.9ms) but results in a lower recall (0.196 vs 0.441) compared to more complex hybrid systems.
*   **Simplicity:** The system is highly maintainable and scalable due to the absence of interdependent scoring logic or external document-store dependencies.

## Key experiments
*   **Baseline:** Established the initial latency/recall benchmark.
*   **Increase candidate pool to 100:** Demonstrated that expanded pools could recover recall but at higher latency costs.
*   **exp_293 (Final):** Achieved the objective of minimal latency by stripping the architecture to its core vector retrieval components.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.441 | 0.196 |
| latency_ms | 10.5 | 6.9 |

## How to run
1. Ensure `faiss` and `numpy` are installed.
2. Load the pre-trained model and the indexed FAISS database.
3. Call the `search` function:
   ```python
   results = search("science fiction movies", df, bm25, model, index, top_k=10)
   ```