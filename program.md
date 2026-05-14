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

Three constraints tracked separately — never combined:
- **recall@10** — primary. Higher is better. Max 1.0.
- **latency_ms** — lower is better. Under 30ms is ideal.
- **llm_cost_usd** — cost of the API call that generated this code.

The active objective is passed at runtime via --objective flag.
Improving on the active objective = keep. Otherwise = discard.

## Available resources in search()

Fixed signature — never change it:
```python
def search(query, df, bm25, model, index, top_k=10):
```

- `df` — DataFrame: columns title, overview, genres, vote_average, vote_count, text
- `bm25` — BM25Okapi: `bm25.get_scores(query.lower().split())`
- `model` — SentenceTransformer("all-MiniLM-L6-v2"): `model.encode([query])`
- `index` — faiss.IndexFlatL2: `index.search(vec.astype("float32"), k)`

## ENVIRONMENT CONSTRAINTS — violations crash the run:

- Allowed imports ONLY: numpy, rank_bm25, sentence_transformers, faiss, sklearn
- nltk is NOT installed — do not import it
- Use int() or np.int64(), NEVER np.int (deprecated)
- Do NOT import from other project files
- `bm25` is already a BM25Okapi object — do NOT rebuild it, call bm25.get_scores(query.split())
- `model` is already a SentenceTransformer — do NOT reload it, call model.encode([query])
- `index` is already a FAISS index — do NOT rebuild it, call index.search(vec, k)
- `df` has columns: title, overview, genres, vote_count, vote_average
- Function signature must be exactly: def search(query, df, bm25, model, index, top_k=10)
- Return format must be: df[...].to_dict("records") with at least title column

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

## PROMISING DIRECTIONS — not yet fully explored:
- Correct RRF formula: 1/(60+rank_a) + 1/(60+rank_b) — current code has k=0 bug
- Pseudo-relevance feedback: embed top-5 results, use as additional query vectors
- Separate embeddings for title vs overview, combine scores
- Query expansion: generate 2-3 rephrasings of the query, union their FAISS results
- Additive vote boost (normalized): score += vote_score / max_votes * 0.2


## Simplicity criterion

A 0.01 score gain that adds 50 lines — not worth it.
A 0.01 score gain from a clean 5-line change — keep.
Equal score, less code — always keep.

## Output format

The benchmark prints:
    recall@10    : 0.700000
    latency      : 27.5ms
    llm_cost     : $0.001489

## Logging

`results.tsv` columns (tab-separated):
    commit  recall  latency_ms  llm_cost_usd  status  description

Status is `keep`, `discard`, or `crash`. Use 0.0 for all metrics on crash.
Three constraints are tracked separately — never combined into one score.

## NEVER STOP

Once the loop begins, do not ask the human for confirmation. Keep experimenting until manually stopped.

