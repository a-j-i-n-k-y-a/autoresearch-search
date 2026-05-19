---
exp_id      : exp_209
prompt_hash : 52b059e5
prompt_file : experiments/prompts/52b059e5.txt
objective   : cost
generated   : 2026-05-19T13:53:06
---

# Architecture

## What it does
The system performs a high-performance hybrid movie search by combining lexical (BM25) and semantic (vector similarity) retrieval. It dynamically fuses these signals at query time, boosting results based on popularity metrics to surface relevant, well-regarded content within a strict low-latency budget.

## Components
- **Lexical Engine**: Uses `rank_bm25` with a custom regex tokenizer (`\b\w\w+\b`) to perform keyword-based retrieval.
- **Semantic Engine**: Uses a pre-computed FAISS index for high-dimensional vector similarity search.
- **Fusion Pipeline**: A weighted scoring mechanism that normalizes BM25 and vector distances, integrates a log-transformed popularity boost, and performs candidate union.
- **Optimization Layer**: Uses `numpy.argpartition` and direct array indexing to minimize data manipulation overhead and bypass expensive DataFrame operations.

## Why it works
- **Candidate Fusion**: Merging 250-item candidate pools from both lexical and semantic sources improved recall from 0.545 to 0.636 (+0.091).
- **Reduced Overhead**: By shifting from row-based DataFrame processing to vectorized NumPy operations, the system reduced latency from 24.5ms to 23.2ms (-1.3ms), while maintaining the improved recall.
- **Score Normalization**: Min-max normalization of BM25 scores paired with inverse-distance mapping for vectors balanced the disparate signal scales, allowing for meaningful weighted fusion.

## Tradeoffs
- **Precision vs. Recall**: Aggressive candidate pooling (250 items) is prioritized to maximize recall, which inherently limits precision; however, the popularity-weighted boost successfully stabilizes the ranking for top-k results.
- **Computational Efficiency**: Direct NumPy indexing significantly reduces latency but necessitates tighter coupling between the retrieval logic and the underlying index data structures.

## Key experiments
- **exp_209 (Final)**: Implemented BM25 on the full text field and streamlined candidate pool retrieval. This configuration achieved the optimal balance of recall (0.636) and latency (23.2ms).
- **Early Iterations**: Initial attempts to use Reciprocal Rank Fusion (RRF) often resulted in inconsistent recall (e.g., drops to 0.364) and increased compute costs, leading to the selection of the current weighted fusion approach.
- **Latency Refinement**: Several iterations (e.g., using direct array indexing and removing `iloc` overhead) were required to achieve sub-24ms latency without regressing on recall.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | 0.545 | 0.636 |
| latency_ms | 24.5 | 23.2 |

## How to run
1. Ensure `rank_bm25`, `numpy`, and `faiss` are installed.
2. Load the movie DataFrame and pre-indexed BM25/FAISS models.
3. Pass a query string to the `search(query, df, bm25, model, index)` function.
4. The system will return a list of dictionaries containing the top-k matched records.