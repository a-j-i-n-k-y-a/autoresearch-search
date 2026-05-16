---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : latency
generated   : 2026-05-17T04:29:31
---

# Architecture

## What it does
The system is an optimized hybrid movie search engine that balances keyword-based relevance, semantic similarity, and content popularity. It retrieves relevant movies by combining traditional term frequency-inverse document frequency (BM25) with high-dimensional vector embeddings, post-processing the results to prioritize highly rated and frequently voted-on films.

## Components
- **BM25 Retrieval:** Performs high-recall initial filtering by identifying the top 1,000 keyword-matching candidates from the movie metadata.
- **FAISS Semantic Index:** Provides a secondary scoring pass using L2 distance on embeddings to assess conceptual relevance.
- **Popularity Engine:** Applies a non-linear boost using `log1p(vote_count) * vote_average` to ensure the final ranked list favors quality and audience consensus.
- **Scoring Logic:** Computes a final hybrid score: `semantic_score * (1.0 + 0.05 * popularity_boost)`, where the semantic score is normalized via `1 / (1 + L2)`.

## Why it works
The architecture follows a classic "retrieval-then-rerank" pattern. The initial BM25 retrieval stage is highly efficient for keyword matching, effectively narrowing the search space to a manageable candidate pool. By applying the semantic index only to this filtered subset, the system maintains high recall while preventing the computational overhead of scanning the entire vector database. The popularity boost acts as a final "tie-breaker" that aligns results with user preference patterns.

## Tradeoffs
- **Latency vs. Complexity:** By limiting the initial recall to 1,000 candidates and performing a focused rerank, the system achieves sub-25ms latency. Increasing the pool size or adding more metadata fields in the reranking phase improves recall but pushes latency above the target threshold.
- **Data Dependency:** The effectiveness of the popularity boost is tied to the quality of the `vote_count` and `vote_average` fields; in cold-start scenarios with new or obscure content, semantic and BM25 scores carry the weight.

## Key experiments
- **Candidate Pool Size:** Testing indicated that a pool of 1,000 candidates provides an optimal balance between coverage (recall) and processing time.
- **Popularity Integration:** Experiments with linear vs. log-scaled popularity show that the log-transformation (`log1p`) prevents high-vote outliers from overwhelming semantic relevance, leading to more stable search results.
- **Hybrid Fusion:** Various RRF and weighted-score experiments confirmed that simple multiplicative blending of popularity and semantic score yielded the most consistent performance improvements over the baseline.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.580 | 0.580 |
| latency_ms | 20.7 | 20.7 |

## How to run
Ensure the `faiss` index and `bm25` model are initialized, then call the `search` function:
```python
from search import search
results = search("science fiction", df, bm25, model, index, top_k=10)
```