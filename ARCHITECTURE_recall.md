---
exp_id      : exp_314
prompt_hash : f5920c6d
prompt_file : experiments/prompts/f5920c6d.txt
objective   : recall
generated   : 2026-05-16T22:44:36
---

# Architecture

## What it does
The system performs high-recall movie retrieval by combining semantic vector search with a metadata-driven popularity heuristic. It leverages FAISS to find top-k semantic matches and then re-ranks them using log-transformed popularity counts to ensure both relevance and audience engagement are surfaced.

## Components
- **FAISS Index**: A dense vector index storing movie embeddings for semantic similarity search.
- **Vector Search Engine**: Uses `model.encode` to convert incoming natural language queries into the same embedding space as the index.
- **Popularity Heuristic**: Applies a logarithmic boost—`log1p(vote_count)`—to the candidates to dampen the impact of extreme outliers while promoting widely recognized content.
- **Re-ranking Logic**: Computes final scores as `(0.5 * log_popularity) - l2_distance` to balance the semantic closeness of the vector search with external popularity metadata.

## Why it works
- **Semantic Mapping**: Vector search effectively captures the intent of complex user queries beyond simple keyword matching.
- **Popularity Smoothing**: By using `log1p`, the system accounts for the power-law distribution of movie popularity, preventing high-vote-count titles from completely overriding the semantic relevance of niche but relevant titles.
- **Balanced Scoring**: Subtracting the L2 distance (which is a minimization objective) from the popularity score (which is a maximization objective) provides a unified ranking surface that prioritizes high-quality, relevant matches.

## Tradeoffs
- **Candidate Pool Size**: Retrieving 200 items to re-rank adds minimal latency compared to the gains in recall, representing a deliberate choice to favor accuracy over the absolute lowest possible latency.
- **Static Weighting**: The `0.5` multiplier for popularity is a tuned hyperparameter; while effective, it may require recalibration if the underlying dataset distribution changes significantly.

## Key experiments
- **exp_314 (Winner)**: Introduced `log1p` popularity boosting combined with L2 distance normalization, achieving the highest recall by tempering the dominance of high-vote movies.
- **Candidate Expansion**: Moving from small pools to a 200-candidate pool was essential to allow the re-ranking logic enough variety to select more relevant results.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.220 | 0.540 |
| latency_ms | 11.2 | 12.2 |

## How to run
1. Ensure `faiss` and `numpy` are installed.
2. Initialize the index with movie embeddings.
3. Pass the `df` containing `vote_count` and the loaded `model` to the `search` function.
4. Call `search(query, df, bm25, model, index)` to retrieve the top 10 ranked results.