# Autoresearch — Autonomous Movie Search Optimizer

An autonomous agent that iteratively improves a movie retrieval system across multiple objectives:

- retrieval quality (`recall@10`)
- latency (`latency_ms`)
- experiment generation cost (`llm_cost_usd`)

Inspired by Karpathy's autoresearch concept, but applied to retrieval systems through autonomous code evolution.

---

# Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                     OFFLINE LOOP                        │
│                                                         │
│   search.py  ──▶  agent_loop.py  ──▶  benchmark        │
│       ▲               │                    │            │
│       │          LLM generates         recall          │
│       │          new search.py         latency         │
│       │               │                cost            │
│       └── keep ◀──────┴──── discard                    │
│                                                         │
│   Every KEEP:                                           │
│     → committed to git                                  │
│     → logged in results.tsv                             │
│     → saved in experiments/log.jsonl                    │
│     → prompt archived by hash                           │
│     → copied into search_profiles/                      │
│     → registry.py regenerated                           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   SERVE TIME (ROUTER)                   │
│                                                         │
│         query + constraint                              │
│               │                                         │
│           router.py                                     │
│               │                                         │
│   ┌───────────┴────────────────────┐                    │
│   │        registry.py             │                    │
│   │  high_recall   → max recall    │                    │
│   │  balanced      → mixed tradeoff│                    │
│   │  low_latency   → fast response │                    │
│   │  low_cost      → simple/cheap  │                    │
│   └───────────┬────────────────────┘                    │
│               │                                         │
│   search_profiles/<profile>.py                          │
│               │                                         │
│         top-k movie retrieval                           │
└─────────────────────────────────────────────────────────┘
```

---

# Core Idea

There is no model training.

The agent performs:

- code search
- retrieval algorithm mutation
- evolutionary optimization

`search.py` is the artifact.

The winning retrieval implementation becomes the deployed search system.

---

# Objectives (tracked independently)

These are NEVER collapsed into a single scalar score.

| Metric | Meaning |
|---|---|
| `recall@10` | Fraction of expected movies found in top 10 |
| `latency_ms` | Milliseconds per query |
| `llm_cost_usd` | Cost of generating the experiment |

---

# Project Structure

```text
autoresearch/
├── search.py
├── agent_loop.py
├── prepare.py
├── router.py
├── program.md
├── results.tsv
├── ARCHITECTURE_recall.md
├── ARCHITECTURE_pareto.md
├── ARCHITECTURE_latency.md
├── requirements.txt
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

# Setup

## 1. Clone repository

```bash
git clone https://github.com/a-j-i-n-k-y-a/autoresearch-search.git
cd autoresearch-search
```

---

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Configure API key

Create `.env`:

```text
GOOGLE_API_KEY=your_api_key_here
```

Optional Anthropic support:

```text
ANTHROPIC_API_KEY=your_key_here
```

---

## 4. Build dataset and indexes

```bash
python prepare.py
```

This:

- downloads ~723k movies
- filters the dataset
- builds BM25
- builds FAISS index
- caches embeddings
- validates benchmark integrity

Generated files are stored in `data/`.

---

## 5. Verify evaluation

```bash
python agent_loop.py --eval-only
```

Expected metrics depend on the current evolved `search.py`.

Typical ranges:

```text
recall@10 : 0.4–0.7
latency   : 5–30ms
```

---

# Running Experiments

## Maximize recall

```bash
python agent_loop.py --n 50 --objective recall
```

---

## Improve recall/latency tradeoff

```bash
python agent_loop.py --n 30 --objective pareto
```

---

## Minimize latency

```bash
python agent_loop.py --n 20 --objective latency
```

---

## Minimize code complexity / generation cost

```bash
python agent_loop.py --n 20 --objective cost
```

---

# Experiment Lifecycle

Each experiment:

1. Generates a new `search.py`
2. Benchmarks retrieval quality
3. Measures latency
4. Measures generation cost
5. Either:
   - KEEP
   - DISCARD
   - CRASH

Successful experiments automatically update:

- `search_profiles/`
- `registry.py`
- architecture docs
- git history
- replay logs

---

# Replay Experiments

Restore and benchmark historical experiments:

```bash
python agent_loop.py --replay exp_042
```

Restore current working version afterwards:

```bash
git checkout -- search.py
```

---

# Router

The router dynamically selects retrieval profiles.

---

## List available profiles

```bash
python router.py --profiles
```

---

## Auto-routed query

```bash
python router.py "psychological thriller unreliable narrator"
```

---

## Explicit constraint

```bash
python router.py "inception" --constraint low_latency

python router.py "robot consciousness AI" --constraint high_recall
```

---

# Constraint Guide

| Constraint | Goal | Use when |
|---|---|---|
| `high_recall` | Maximize retrieval quality | Semantic search, recommendations |
| `balanced` | Trade off recall and latency | Default general search |
| `low_latency` | Minimize response time | Autocomplete, typeahead |
| `low_cost` | Simpler/cheaper implementations | Lightweight experimentation |

Profiles are generated dynamically from successful experiments.

Available profiles depend on optimization history.

---

# Python Usage

```python
from router import route

response = route(
    "dark sci-fi about consciousness",
    constraint="high_recall"
)

for movie in response["results"]:
    print(movie["title"])
```

---

# Benchmark

The benchmark consists of ~50 fixed movie retrieval queries.

Each query contains:
- natural language retrieval intent
- expected relevant movies

Examples:

| Query | Expected |
|---|---|
| dream heist movie Leonardo DiCaprio layers of subconscious | Inception |
| astronaut stranded in space wormhole black hole | Interstellar, Gravity |
| psychological thriller unreliable narrator | Shutter Island, Black Swan |

The benchmark is deterministic and version-controlled.

Title normalization and alias resolution are applied during evaluation to avoid impossible matches.

---

# Experiment Logging

## results.tsv

High-level experiment log.

Typical columns:

```text
exp_id
status
recall
latency_ms
llm_cost_usd
description
```

---

## experiments/log.jsonl

Full replay log containing:

- generated code
- prompts
- metrics
- status
- traceback data
- objective lineage

Used for:
- replay
- architecture generation
- profile export
- experiment analysis

---

# Search Profiles

Profiles are exported retrieval implementations optimized for different objectives.

Examples:

```text
search_profiles/high_recall.py
search_profiles/low_latency.py
```

The router dynamically loads these implementations at runtime.

---

# Design Principles

- Constraints tracked independently
- No weighted aggregate score
- Deterministic evaluation
- Reversible experiments
- Git-backed rollback
- Full experiment replayability
- Objective-isolated optimization
- Autonomous architecture evolution

---

# Notes

- Latency comparisons are session-local
- Resources are cached and lazy-loaded
- BM25 and FAISS are prebuilt
- Prompt history is deduplicated
- Crashed experiments are logged and recoverable

---

# Future Work

Potential extensions:

- multi-objective Pareto frontier export
- distributed experiment execution
- semantic benchmark grading
- learned rerankers
- online user feedback optimization
- profile distillation
- benchmark expansion
- retrieval ensembles

---

# License

MIT