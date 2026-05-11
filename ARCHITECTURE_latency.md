# Architecture

## What it does
The system performs a rapid text-based search to find relevant movies. It breaks down your search terms and matches them against movie descriptions using statistical frequency, ignoring complex machine learning models to provide results almost instantly.

## Components
*   **BM25 Algorithm:** A standard information retrieval technique that scores movies based on the presence and frequency of query terms in the "overview" text.
*   **NumPy Indexing:** Used for efficient array sorting and ranking, ensuring the top-k results are retrieved without overhead.
*   **DataFrame Storage:** Holds the movie dataset (titles and overviews), allowing for rapid selection and filtering once the search engine identifies the best candidates.

## Why it works
By stripping away heavy vector embeddings and hybrid scoring layers, the system relies entirely on efficient sparse retrieval. The performance gain comes from bypassing the GPU/CPU-heavy inference required by neural search models, focusing solely on fast token-matching against a pre-computed frequency index.

## Tradeoffs
To achieve a ~82% reduction in latency, the system prioritizes speed over accuracy. The loss in recall is primarily due to the abandonment of semantic matching; the system no longer understands synonyms or context, focusing strictly on exact keyword overlaps.

## Key experiments
*   **What worked:** Stripping out secondary features like genre filtering and complex vote-boost calculations drastically reduced overhead. The "Simple BM25 without vote boost" proved to be the most efficient configuration.
*   **What failed:** Hybrid approaches (FAISS + BM25) and RRF (Reciprocal Rank Fusion) failed to meet the latency objective, as the computational cost of merging two different search outputs outweighed the marginal accuracy benefits.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.800 | 0.600 |
| latency_ms | 67.3 | 12.0 |

## How to run
```bash
python agent_loop.py --eval-only
```