# Architecture

## What it does
The system performs high-precision movie retrieval by combining semantic vector embeddings with heuristic metadata signals. It retrieves a wide pool of candidates using FAISS and refines the ranking by balancing vector proximity, popularity bias, and genre-based keyword matching.

## Components
*   **Encoder:** Uses a sentence-transformer model to map user queries into a high-dimensional semantic vector space.
*   **Vector Index:** A FAISS-based retrieval engine that performs efficient $k$-nearest neighbor search (retrieving 500 candidates for a Top-10 output).
*   **Ranking Logic:** A custom re-ranking function that computes a composite score:
    *   **Semantic Score:** $1 / (1 + \text{distance})$
    *   **Popularity Bias:** $\log(1 + \text{vote\_count}) \times 0.05$ (normalizes high-volume outliers).
    *   **Genre Boost:** Additive constant ($+0.1$) for direct term matches within movie metadata.

## Why it works
The architecture optimizes for recall by acknowledging that pure semantic search often struggles with specific entity preferences (like popularity) or user-specified genre intent. By widening the initial candidate pool and applying post-retrieval reranking, the system preserves high recall while injecting business logic that aligns with user expectations.

## Tradeoffs
*   **Compute Latency:** The decision to maintain the baseline architecture prioritizes recall stability over latency optimization. More complex hybrid approaches (BM25/RRF) consistently regressed recall during testing, suggesting that the current embedding model is highly optimized for this specific dataset.
*   **Operational Cost:** By avoiding complex multi-stage retrieval pipelines that incurred API or processing costs in failed experiments, the final design maintains a $0.00 cost profile.

## Key experiments
*   **Hybrid Failure:** 29 consecutive experiments attempting to integrate BM25 or RRF (Reciprocal Rank Fusion) resulted in lower recall, likely due to feature misalignment or loss of semantic nuance.
*   **Stability:** The baseline configuration proved highly robust, showing that the most effective path was refining the ranking function rather than altering the retrieval foundation.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.900 | 0.900 |
| latency_ms | 208.1 | 208.1 |

## How to run
1. Ensure the movie dataframe and FAISS index are loaded into memory.
2. Initialize the embedding model.
3. Call the `search(query, df, bm25, model, index, top_k)` function with the query string and required dependencies.