---
exp_id      : exp_069
prompt_hash : 1bffc102
prompt_file : experiments/prompts/1bffc102.txt
objective   : pareto
generated   : 2026-05-14T20:04:20
---

# Architecture

## What it does
The system implements a two-stage hybrid search pipeline for movie retrieval. It performs high-speed candidate selection using keyword-based retrieval (BM25) followed by a compute-efficient semantic re-ranking step using dense vector embeddings to ensure relevance.

## Components
- **Candidate Retrieval (BM25):** An initial filtering layer that selects the top 50 candidates from the dataset based on keyword frequency and document length, providing a low-latency "narrowing" of the search space.
- **Semantic Re-ranker (SBERT):** A transformer-based model that encodes the query and the 50 selected candidates into dense vectors. 
- **Dot Product Scorer:** Calculates cosine similarity between the query vector and candidate vectors to determine the final relevance ranking.

## Why it works
By using BM25 for the initial retrieval, the system avoids running expensive transformer inference over the entire movie database. The 50-candidate pool is small enough for the embedding model to process in parallel while providing enough coverage to maintain high recall. This effectively prunes the search space to only those items most likely to be relevant.

## Tradeoffs
- **Candidate Sensitivity:** The performance is highly dependent on the initial recall of the BM25 stage. If relevant movies are not caught in the top 50, the re-ranker cannot recover them.
- **Fixed Pool Size:** Using a static pool size of 50 candidates minimizes latency but limits the model's ability to "see" a wider breadth of potentially relevant documents compared to larger pool sizes.

## Key experiments
- **exp_069 (Winner):** Hybrid search with metadata-informed candidate pool resizing and efficient dot-product scoring.
- **Candidate Pool Scaling:** Multiple experiments revealed that a pool size of 50–100 provided the optimal balance between recall stability and query latency.
- **Fusion Optimization:** Experiments attempting RRF (Reciprocal Rank Fusion) and complex weighting consistently introduced higher latency with diminishing returns on recall, leading to the adoption of the simpler BM25-then-Vector approach.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.510 | 0.510 |
| latency_ms | 17.5 | 8.6 |

## How to run
1. Ensure the environment has `numpy`, `pandas`, and the relevant embedding model dependencies installed.
2. Initialize the `bm25` index using the movie overview corpus.
3. Call `search(query, df, bm25, model, index)` to retrieve the top-k ranked results as a list of dictionaries.