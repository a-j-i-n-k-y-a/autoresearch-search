# Search Autoresearch

Autonomous agent improving a movie search system across three objectives.

## Setup

1. Agree on a run tag (e.g. `may10`), create branch: `git checkout -b autoresearch/<tag>`
2. Read these files:
   - `prepare.py` — fixed constants, data, evaluation. DO NOT MODIFY.
   - `search.py` — the only file you edit.
3. Verify `data/movies.pkl` and `data/faiss.index` exist. If not, tell the human to run `python prepare.py`.
4. Initialize `results.tsv` with just the header row.
5. Confirm and begin.

## The objective

Optimize a **single score** that balances three constraints:

    score = recall@10 - max(0, (latency_ms - 30) / 100) * 0.2

- **Recall@10** — primary objective. Higher is better. Max is 1.0.
- **Latency** — cost proxy. Queries under 30ms are free. Above that, you pay.
- **Simplicity** — all else equal, fewer lines wins. Removing code that doesn't hurt recall is a win.

The score is what determines keep vs discard. A recall gain that costs too much latency is not worth it.

## Available resources in search()

Fixed signature — never change it:
```python
def search(query, df, bm25, model, index, top_k=10):
```

- `df` — DataFrame: columns title, overview, genres, vote_average, vote_count, text
- `bm25` — BM25Okapi: `bm25.get_scores(query.lower().split())`
- `model` — SentenceTransformer("all-MiniLM-L6-v2"): `model.encode([query])`
- `index` — faiss.IndexFlatL2: `index.search(vec.astype("float32"), k)`

## Environment constraints — violations crash the run

- Allowed imports ONLY: `numpy`, `rank_bm25`, `sentence_transformers`, `faiss`, `sklearn`
- `nltk` is NOT installed
- Use `int()` or `np.int64()`, never `np.int` (deprecated)
- Do NOT import from other project files


## Experiment ideas

These are starting points — not an exhaustive list. Once you've tried them,
think deeper. Ask yourself:

- What information in `df` am I not using? (genres, vote_count, vote_average, release_date)
- Am I searching the right field? (title vs overview vs genres vs full text)
- Can I combine two strategies that each gave partial gains?
- Can I remove something and get equal recall? (simplification wins)
- What would a human do differently when searching for these specific queries?

Known strategies to explore first:
- Reciprocal Rank Fusion (RRF)
- Larger candidate pool
- Title-boosted BM25
- Vote score boosting
- Cosine similarity
- Pure BM25 only

## Simplicity criterion

A 0.01 score gain that adds 50 lines — not worth it.
A 0.01 score gain from a clean 5-line change — keep.
Equal score, less code — always keep.

## Output format

The benchmark prints:
    recall@10: 0.700000
    latency:   27.5ms
    score:     0.695000

## Logging

`results.tsv` columns (tab-separated):
    commit  score   recall  latency_ms  status  description

Status is `keep`, `discard`, or `crash`. Use 0.0 for all metrics on crash.

## NEVER STOP

Once the loop begins, do not ask the human for confirmation. Keep experimenting until manually stopped.