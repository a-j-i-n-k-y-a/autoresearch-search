# Search Autoresearch

Autonomous agent improving a movie search system across three objectives.

## Setup

1. Agree on a run tag (e.g. `may10`), create branch: `git checkout -b autoresearch/<tag>`
2. Read these files:
   - `prepare.py` — fixed constants, data, evaluation. DO NOT MODIFY.
   - `search.py` — the only file you edit.
3. Verify `data/movies.pkl` and `data/faiss.index` exist. If not, tell the human to run `python prepare.py`.
4. `results.tsv` is created automatically with the correct header on first run.
5. Confirm and begin.

## The objective

Three constraints tracked separately — never combined:
- **recall@10** — primary. Higher is better. Max 1.0.
- **latency_ms** — p50 per-query wall-clock. Under 30ms is ideal.
- **llm_cost_usd** — cost of the API call that generated this code.

Secondary quality metrics tracked on every eval (soft constraints in keep gate):
- **mrr@10** — mean reciprocal rank. Must not drop below 97% of best.
- **ndcg@10** — rank-discounted relevance. Informational.
- **precision@10** — fraction of top-10 that are relevant. Must not drop below 95% of best.
- **top1** — rank-1 hit rate. Must not drop below 95% of best.

The active objective is passed at runtime via `--objective` flag.
Improving on the active objective = keep. Otherwise = discard.

Hard floor: **recall@10 must never drop below 0.50** on any kept experiment,
regardless of objective.

## Available resources in search()

Fixed signature — never change it:
```python
def search(query, df, bm25, model, index, top_k=10):
```

- `df` — DataFrame: columns `title`, `overview`, `genres`, `vote_average`, `vote_count`, `text`
- `bm25` — BM25Okapi built with `_bm25_tokenize` (accent removal → lowercase → punct strip → stopword filter → len>1 filter). **Call `bm25.get_scores(_bm25_tokenize(query))` — NOT `query.lower().split()`**. You cannot import `_bm25_tokenize` from prepare.py; replicate the pipeline in search.py.
- `model` — SentenceTransformer(`"BAAI/bge-small-en-v1.5"`): `model.encode([query])`
- `index` — faiss.IndexFlatL2: `index.search(vec.astype("float32"), k)`

## ENVIRONMENT CONSTRAINTS — violations crash the run:

- Allowed imports ONLY: numpy, rank_bm25, sentence_transformers, faiss, sklearn
- nltk is NOT installed — do not import it
- Use `int()` or `np.int64()`, NEVER `np.int` (deprecated)
- Do NOT import from other project files
- `bm25` is already a BM25Okapi object — do NOT rebuild it
- `model` is already a SentenceTransformer — do NOT reload it, call `model.encode([query])`
- `index` is already a FAISS index — do NOT rebuild it, call `index.search(vec, k)`
- `df` has columns: `title`, `overview`, `genres`, `vote_count`, `vote_average`
- Function signature must be exactly: `def search(query, df, bm25, model, index, top_k=10)`
- Return format must be: `df[...].to_dict("records")` with at least a `title` column
- Never rerank more than top 100 candidates
- Avoid nested loops over candidate pools
- Never call `model.encode()` inside loops
- Avoid repeated dataframe filtering

## Benchmark structure

Queries are split into four sets:

| Split    | Size | Purpose                                              |
|----------|------|------------------------------------------------------|
| train    | 25   | Shown to agent in prompts as examples                |
| dev      | 15   | Used for all keep/discard decisions during runs      |
| test     | 10   | Final benchmark eval only — never used in loop       |
| holdout  | 11   | Private — never shown to agent, never in history     |

All optimization decisions are made on `split="dev"` only.
Call `evaluate_holdout()` only after all optimization is complete.

Queries are also tagged by slice: `exact_title`, `semantic`, `genre`,
`long_tail`, `ambiguous`. Per-slice recall is printed after every eval
and saved to `experiments/log.jsonl`.

## Eval output format

Every benchmark run prints:
```
  mrr@10=0.xxx  ndcg@10=0.xxx  precision@10=0.xxx  top1=0.xxx
  ambiguous      : 0.xxx
  exact_title    : 0.xxx
  genre          : 0.xxx
  long_tail      : 0.xxx
  semantic       : 0.xxx
  recall CI      : [0.xxx, 0.xxx]  noise=0.xxx
  lat p50/p95/p99: xx.x/xx.x/xx.xms
```

`--eval-only` prints the full picture:
```
  recall@10    : 0.xxxxxx  (noise ±0.xxx)
  latency p50  : xx.xms
  latency p95  : xx.xms
  latency p99  : xx.xms
  mrr@10       : 0.xxxxxx
  ndcg@10      : 0.xxxxxx
  precision@10 : 0.xxxxxx
  top1         : 0.xxxxxx
```

## Logging

`results.tsv` columns (tab-separated):
```
exp_id  commit  recall  latency_ms  latency_p95  latency_p99  llm_cost_usd  status  description
```

Status is `keep`, `discard`, `crash`, or `kept-subfloor` (retroactively
marked kept rows that violated the recall floor under audit).

Full experiment records including `search_py`, `full_metrics`, `slice_results`,
and `constraint_trace` are saved to `experiments/log.jsonl`.
Prompts are deduplicated by MD5 hash under `experiments/prompts/`.

## Profile system

Winning implementations are exported to `search_profiles/`:

| Profile        | Objective | Seeded from  |
|----------------|-----------|--------------|
| `high_recall`  | recall    | —            |
| `low_latency`  | latency   | high_recall  |
| `balanced`     | pareto    | high_recall  |
| `low_cost`     | cost      | balanced     |

`search_profiles/registry.py` is auto-regenerated on every KEEP.
Profiles below `RECALL_FLOOR = 0.50` are never exported.

## Keep gate

An experiment is kept only when ALL of the following pass for the active objective:

**recall objective:**
- `recall_ok`: new recall ≥ RECALL_FLOOR AND ≥ best × 0.95
- `recall_improve`: improvement exceeds noise floor (`max(noise×0.5, 1/15)`)
- `mrr_ok`: new mrr ≥ best mrr × 0.97
- `top1_ok`: new top1 ≥ best top1 × 0.95
- `precision_ok`: new precision ≥ best precision × 0.95

**latency objective:** recall_ok + mrr_ok + top1_ok + precision_ok + latency improves > 0.05ms

**cost objective:** recall_ok + mrr_ok + top1_ok + precision_ok + latency ≤ best×1.10 + cost improves

**pareto objective:** recall drop ≤ 0.02 + latency ≤ best×1.10 + cost ≤ best×2.00 + any one improves + mrr_ok + top1_ok + precision_ok

## Experiment ideas

These are starting points — not an exhaustive list. Once you've tried them,
think deeper. Ask yourself:

- What information in `df` am I not using? (`genres`, `vote_count`, `vote_average`)
- Am I searching the right field? (`title` vs `overview` vs `genres` vs full `text`)
- Can I combine two strategies that each gave partial gains?
- Can I remove something and get equal recall? (simplification wins)
- What would a human do differently when searching for these specific queries?
- Does the failure report show a vocabulary mismatch (BM25 failing) or a reranking problem?

Known strategies to explore first:
- Reciprocal Rank Fusion (RRF) — correct formula: `1/(60+rank_a) + 1/(60+rank_b)`
- Larger candidate pool
- Title-boosted BM25
- Vote score boosting (additive, normalized: `score += vote / max_votes * 0.2`)
- Cosine similarity instead of L2
- Pseudo-relevance feedback: embed top-5 results, use as additional query vectors
- Separate embeddings for title vs overview, combine scores
- Query expansion: 2-3 rephrasings, union FAISS results

## Simplicity criterion

A 0.01 score gain that adds 50 lines — not worth it.
A 0.01 score gain from a clean 5-line change — keep.
Equal score, less code — always keep.

## Regression awareness

On every KEEP, a regression report prints which dev queries got worse.
The agent prompt also includes a failure analysis of the current worst
queries (recall < 1.0, sorted worst-first). Use this to reason about
WHY retrieval is failing, not just THAT it is failing.

## NEVER STOP

Once the loop begins, do not ask the human for confirmation.
Keep experimenting until manually stopped.