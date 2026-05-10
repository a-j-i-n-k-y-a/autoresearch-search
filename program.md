# Search Autoresearch

Experiment to have the LLM autonomously improve a search system.

## Setup

To set up a new experiment:

1. Agree on a run tag based on today's date e.g. `may10`
2. Create branch: `git checkout -b autoresearch/<tag>`
3. Read these files for full context:
   - `prepare.py` — fixed constants, data prep, evaluation. DO NOT MODIFY.
   - `search.py` — the only file you modify. The search function.
4. Verify data exists: check that `data/movies.pkl` and `data/faiss.index` exist.
   If not, tell the human to run `python prepare.py` first.
5. Initialize `results.tsv` with just the header row.
6. Confirm setup and begin.

## Experimentation

Each experiment runs the full benchmark and reports recall@K.

**What you CAN do:**
- Modify `search.py` — this is the ONLY file you edit.
- Change anything inside the `search()` function.
- Use any of the resources passed in: df, bm25, model, index.
- Add helper functions above `search()` in the same file.
- Try: query expansion, vector search, hybrid combinations,
  reranking, query preprocessing, different top_k values.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only.
- Modify the benchmark queries or expected results.
- Change the `evaluate()` function.
- Install new packages — use only what's already installed.

**The goal: maximize recall@K.**
Higher recall = more expected movies found in top 10 results.
Current max possible = 1.0

**Simplicity criterion:**
All else being equal, simpler is better.
A 0.01 improvement that adds 50 lines of complex code — not worth it.
A 0.01 improvement from a clean 5-line change — definitely keep.
Removing code and getting equal or better results — great outcome.

**The first run:**
Always run as-is first to establish the baseline.

## Running an experiment

```bash
python agent_loop.py --eval-only
```

This runs the benchmark on the current `search.py` and prints: