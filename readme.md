# Autoresearch — Autonomous Movie Search Optimizer

An autonomous agent that iteratively improves a movie retrieval system across multiple objectives: retrieval quality (`recall@10`), latency (`latency_ms`), and experiment cost (`llm_cost_usd`).

---

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                     OFFLINE LOOP                        │
│                                                         │
│   search.py ──▶ agent_loop.py ──▶ benchmark             │
│       ▲               │               │                 │
│       │          LLM generates    recall / latency /    │
│       │          new search.py    cost / mrr / ndcg     │
│       │               │               │                 │
│       └── keep ◀──────┴──── discard                    │
│                                                         │
│   Every KEEP:                                           │
│     → committed to git                                  │
│     → logged in results.tsv + experiments/log.jsonl     │
│     → prompt archived by MD5 hash                       │
│     → copied into search_profiles/                      │
│     → registry.py regenerated                           │
│     → ARCHITECTURE_<objective>.md updated               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   SERVE TIME (ROUTER)                   │
│                                                         │
│         query + constraint                              │
│               │                                         │
│           router.py                                     │
│               │                                         │
│         registry.py                                     │
│   high_recall / balanced / low_latency / low_cost       │
│               │                                         │
│   search_profiles/<profile>.py → top-k results          │
└─────────────────────────────────────────────────────────┘
```

---

## Project Structure

```text
autoresearch/
├── search.py                   # the only file the agent edits
├── agent_loop.py               # experiment loop, keep gate, logging
├── prepare.py                  # data prep, indexes, evaluation — DO NOT MODIFY
├── router.py                   # serve-time profile selector
├── evaluate_final.py           # holdout evaluation (run after optimization)
├── migrate_legacy.py           # one-time log migration script
├── test_is_improvement.py      # unit tests for the keep gate
├── program.md                  # agent instructions
├── results.tsv                 # high-level experiment log
├── ARCHITECTURE_<objective>.md # auto-generated architecture docs
├── .env
├── data/
│   ├── movies.pkl
│   ├── faiss.index
│   ├── bm25.pkl
│   └── embeddings.pkl
├── experiments/
│   ├── log.jsonl
│   └── prompts/
└── search_profiles/
    ├── __init__.py
    ├── registry.py
    ├── high_recall.py
    ├── balanced.py
    ├── low_latency.py
    └── low_cost.py
```

---

## Setup

**1. Clone and install**

```bash
git clone https://github.com/a-j-i-n-k-y-a/autoresearch-search.git
cd autoresearch-search
pip install -r requirements.txt
```

**2. Configure API key** — create `.env`:

```text
GOOGLE_API_KEY=your_api_key_here
```

To switch to Claude (Anthropic), uncomment the relevant block in `agent_loop.py`.

**3. Build dataset and indexes**

```bash
python prepare.py
```

Downloads ~723k movies, builds BM25 (with stopword-filtered tokenizer), builds FAISS IndexFlatL2, validates benchmark coverage. Cached in `data/`.

**4. Verify**

```bash
python agent_loop.py --eval-only
```

---

## Running Experiments

```bash
# Maximize recall
python agent_loop.py --n 50 --objective recall

# Improve recall/latency tradeoff
python agent_loop.py --n 30 --objective pareto

# Minimize latency
python agent_loop.py --n 20 --objective latency

# Minimize cost
python agent_loop.py --n 20 --objective cost
```

---

## Benchmark

~50 fixed movie retrieval queries split into four sets:

| Split   | Size | Purpose                                          |
|---------|------|--------------------------------------------------|
| train   | 25   | Shown to agent in prompts as examples            |
| dev     | 15   | Used for all keep/discard decisions              |
| test    | 10   | Final benchmark — never used in loop             |
| holdout | 11   | Private — never shown to agent, never in history |

Queries are also tagged by slice: `exact_title`, `semantic`, `genre`, `long_tail`, `ambiguous`. Per-slice recall is printed after every eval.

Dev set uses randomized 75% sub-sampling each run to reduce benchmark memorization pressure.

---

## Metrics

Tracked independently — never collapsed into a single score:

| Metric | Role |
|---|---|
| `recall@10` | Primary gate metric. Hard floor: 0.50. |
| `latency_ms` | p50 per-query wall-clock. p95/p99 also tracked. |
| `llm_cost_usd` | Cost of generating the experiment. |
| `mrr@10` | Soft constraint — must not drop below 97% of best. |
| `precision@10` | Soft constraint — must not drop below 95% of best. |
| `top1` | Soft constraint — must not drop below 95% of best. |
| `ndcg@10` | Informational — logged but not gated. |

Every eval also reports a bootstrap confidence interval on recall to distinguish real improvements from noise (1 dev query flip = 0.067 recall change).

---

## Keep Gate

An experiment is kept only when all constraints pass for the active objective:

**recall:** recall_ok + improvement exceeds noise floor + mrr_ok + top1_ok + precision_ok

**latency:** recall_ok + mrr_ok + top1_ok + precision_ok + latency improves > 0.05ms

**cost:** recall_ok + mrr_ok + top1_ok + precision_ok + latency ≤ best×1.10 + cost improves

**pareto:** recall drop ≤ 0.02 + latency ≤ best×1.10 + cost ≤ best×2.00 + any one improves + mrr_ok + top1_ok + precision_ok

---

## Eval Output

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

`--eval-only` gives the full picture with all six metrics.

---

## Profile System

Winning implementations are exported to `search_profiles/`:

| Profile | Objective | Seeded from |
|---|---|---|
| `high_recall` | recall | — |
| `low_latency` | latency | high_recall |
| `balanced` | pareto | high_recall |
| `low_cost` | cost | balanced |

`registry.py` is auto-regenerated on every KEEP. Profiles below `RECALL_FLOOR = 0.50` are never exported.

---

## Router

```bash
python router.py "psychological thriller unreliable narrator"
python router.py "inception" --constraint low_latency
python router.py --profiles
python router.py --demo
```

From code:

```python
from router import route
response = route("dark sci-fi about consciousness", constraint="high_recall")
for movie in response["results"]:
    print(movie["title"])
```

Constraint is inferred from query characteristics (length, keywords) when not specified.

---

## Final Holdout Evaluation

Run **only after all optimization is complete**:

```bash
python evaluate_final.py --profile high_recall
python evaluate_final.py --all-profiles
```

Reports dev vs holdout recall gap. A gap > 0.05 suggests overfitting to the dev set.

---

## Other Commands

```bash
# Replay a historical experiment
python agent_loop.py --replay exp_042

# Manually export current search.py as a profile
python agent_loop.py --export-profile recall

# Migrate legacy log entries (run once after upgrading)
python migrate_legacy.py

# Run keep-gate unit tests
python test_is_improvement.py
```

---

## Logging

`results.tsv` columns: `exp_id · commit · recall · latency_ms · latency_p95 · latency_p99 · llm_cost_usd · status · description`

Status values: `keep`, `discard`, `crash`, `kept-subfloor`

`experiments/log.jsonl` stores full records: generated code, prompt hash, all metrics, constraint trace, slice results. Cross-referenced with `results.tsv` via `exp_id`.

---

## Design Principles

- Constraints tracked independently, never combined into one score
- Deterministic evaluation with bootstrap noise estimation
- Reversible experiments via git — every KEEP is committed, every DISCARD is restored
- Full experiment replayability from `log.jsonl`
- Objective-isolated optimization with cross-objective seeding
- Profiles below recall floor are never promoted