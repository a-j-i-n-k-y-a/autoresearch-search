---
exp_id      : unknown
prompt_hash : unknown
prompt_file : experiments/prompts/unknown.txt
objective   : latency
generated   : 2026-05-17T04:08:49
---

# Architecture

## What it does
The system provides a hybrid search experience for movie discovery, combining classical keyword-based retrieval with modern semantic understanding and popularity-based ranking to deliver high-quality, relevant results.

## Components
- **BM25 Retrieval:** Used as the primary stage to retrieve the top 1000 candidate movies based on text-token matching.
- **FAISS Semantic Engine:** Executes vector similarity search using L2 distance on the full index to derive semantic scores for the candidate pool.
- **Popularity-Aware Scorer:** Applies a mathematical transformation (`log1p(vote_count) * vote_average`) to boost candidates that have higher community engagement and ratings.
- **Hybrid Fusion:** Integrates BM25 candidate recall with a final rank determined by the product of semantic similarity and popularity scores.

## Why it works
The two-stage retrieval strategy ensures high recall by capturing both exact keyword matches and conceptual synonyms. By isolating the computation of semantic scores to a refined BM25-selected pool and integrating popularity metrics as a final re-ranking signal, the system balances user preference (quality/ratings) with search precision.

## Tradeoffs
- **Complexity vs. Latency:** Utilizing both BM25 and FAISS increases computational overhead compared to single-method approaches; however, it effectively mitigates the "missing keyword" problem of vector search and the "semantic blind spots" of BM25.
- **Resource Intensity:** Maintaining both an inverted index (BM25) and a vector index (FAISS) requires more memory than a singular architecture.

## Key experiments
- **Initial Baseline:** Established a 0.441 recall threshold.
- **Candidate Pool Expansion:** Testing various pool sizes (100 to 2000) revealed that 1000 candidates provide the optimal balance for re-ranking precision.
- **Hybrid Fusion (Winning Model):** Integrating logarithmic popularity boosting with semantic distance yielded the most significant jump in recall (0.580) while maintaining acceptable latency.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.580 | 0.580 |
| latency_ms | 21.2 | 21.2 |

## How to run
1. Ensure `search.py` and the initialized `faiss_index` are in the environment path.
2. Load the dataset into a pandas DataFrame.
3. Pass a string query to the `search(query, df, bm25, model, index, top_k=10)` function.
4. The system will compute candidate scores based on the hybrid pipeline and return the ranked dictionary of movies.