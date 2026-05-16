---
exp_id      : exp_314
prompt_hash : f5920c6d
prompt_file : experiments/prompts/f5920c6d.txt
objective   : pareto
generated   : 2026-05-16T22:51:25
---

# Architecture

## What it does
The system performs high-performance movie retrieval by combining semantic vector search with a post-retrieval ranking adjustment. It retrieves a candidate pool of 200 movies using FAISS and refines their order based on a combination of vector similarity, movie popularity (log-scaled vote count), and critical reception (vote average).

## Components
- **Vector Index (FAISS):** An L2-normalized index for low-latency retrieval of top-k semantic candidates.
- **Encoder:** A pre-trained model (via `model.encode`) that transforms natural language queries into dense embeddings.
- **Re-ranking Logic:** A custom scoring function that applies a non-linear boost to vector-based results using `np.log1p(vote_count)` and `vote_average`.
- **Normalization:** L2 distances are transformed into confidence-like scores ($1.0 / (1.0 + \text{dist})$) to allow for multiplicative weighting with metadata.

## Why it works
The architecture shifts from relying solely on semantic similarity to a multi-faceted relevance model. By boosting high-quality, popular movies within the dense-retrieval candidate pool, the system better matches user intent, which often prioritizes well-regarded or popular titles over purely semantically similar descriptions.

## Tradeoffs
- **Recall vs. Precision:** Slight reduction in raw recall (-0.02) in exchange for higher-quality recommendations that favor popular/highly-rated titles.
- **Complexity:** Increased metadata dependency requires keeping additional features (`vote_count`, `vote_average`) in memory alongside the index.

## Key experiments
- **Candidate Pool Expansion:** Testing up to 1000 candidates proved that higher recall is possible, but narrowing to 200 (in `exp_314`) optimized the latency-recall trade-off.
- **Metadata Weighting:** Multi-stage testing revealed that simple vector search was insufficient. Introducing logarithmic popularity scaling (rather than linear) provided the most stable boost to performance.
- **RRF/Hybrid Abandonment:** Several hybrid BM25 experiments were discarded due to the high latency overhead and minimal gains compared to pure vector-based re-ranking.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.540 | 0.520 |
| latency_ms | 12.6 | 11.4 |

## How to run
1. Ensure the `faiss` index and `model` are loaded into memory.
2. Provide the `df` containing metadata features (`vote_count`, `vote_average`).
3. Call `search(query, df, bm25, model, index, top_k=10)`.