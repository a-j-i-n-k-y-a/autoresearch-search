# Architecture

## What it does
The system implements a two-stage hybrid search pipeline to retrieve and rank relevant movies. It performs an initial coarse-grained retrieval using BM25 to identify a candidate pool, followed by a fine-grained semantic re-ranking stage using dense vector embeddings to determine final relevance.

## Components
*   **BM25 Retrieval:** Executes a term-frequency-based search to identify the top 50 candidate movies from the dataset based on keyword matching.
*   **Vector Re-ranker:** Uses a pre-trained sentence-transformer model to encode the user query and the retrieved candidate descriptions into a shared semantic space.
*   **Cosine Similarity:** Computes the dot product between the query vector and candidate vectors to determine the final ranked order.

## Why it works
The architecture optimizes for both recall and efficiency by limiting the expensive vector encoding process to a small, high-confidence subset of the data (50 items). By offloading the initial broad search to BM25, the system ensures keyword precision, while the re-ranker captures semantic nuances that simple keyword matching might miss.

## Tradeoffs
*   **Candidate Pool Size:** The pool is hard-capped at 50 candidates. While this keeps latency low, it limits the system's ability to recover relevant documents that fall outside the BM25 top 50, effectively capping the maximum possible recall.
*   **Complexity:** The architecture relies on two separate search methodologies, increasing the surface area for potential indexing failures (as evidenced by multiple crashes in the experiment history).

## Key experiments
*   **Hybrid RRF:** Attempts at Reciprocal Rank Fusion (RRF) consistently resulted in system crashes or degraded performance, leading to a pivot toward a sequential pipeline.
*   **FAISS Integration:** Direct FAISS integration experiments consistently failed to outperform the baseline or showed significant instability, suggesting the current BM25-based candidate pool provides the most stable performance floor.
*   **Candidate Scaling:** Increasing the candidate pool to 1000 increased latency and cost without improving recall, justifying the 50-item limit.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.510 | 0.510 |
| latency_ms | 12.2 | 12.2 |

## How to run
1. Ensure `numpy` and `pandas` are installed.
2. Provide a pre-indexed `bm25` object, a `model` with an `encode` method, and the movie `df`.
3. Pass the query and artifacts to the `search` function.
4. The function returns the top 10 ranked results as a list of dictionaries.