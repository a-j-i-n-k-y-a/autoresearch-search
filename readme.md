# Autoresearch — Movie Search Optimizer

An autonomous agent that iteratively improves a movie search system across three constraints: recall, latency, and LLM cost. Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

---

## Architecture

```
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
│   Every ✅ KEEP:                                        │
│     → committed to git                                  │
│     → logged in results.tsv       (exp_id cross-ref)   │
│     → saved in experiments/log.jsonl   (full replay)   │
│     → copied to search_profiles/<objective>.py         │
│     → search_profiles/registry.py regenerated          │
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
│   │  high_recall  → 0.900 recall   │                    │
│   │  balanced     → 0.800 recall   │                    │
│   │  low_latency  → 0.600 recall   │                    │
│   └───────────┬────────────────────┘                    │
│               │                                         │
│   search_profiles/<profile>.py                          │
│               │                                         │
│         top 10 movies                                   │
└─────────────────────────────────────────────────────────┘
```

### Three constraints — tracked separately, never combined

| Constraint | What it measures | Optimized by |
|---|---|---|
| `recall@10` | Fraction of expected movies found in top 10 | `--objective recall` |
| `latency_ms` | Wall clock ms per query | `--objective latency` |
| `llm_cost_usd` | USD cost of the API call that generated the code | `--objective cost` |

### Code is the weights

There is no model training. The agent does code search across possible retrieval algorithms. `search.py` is the artifact — the winning implementation is what gets deployed.

---

## Project structure

```
autoresearch/
├── search.py                      ← agent edits this (current best)
├── agent_loop.py                  ← autonomous experiment loop
├── prepare.py                     ← data prep + evaluation (do not modify)
├── router.py                      ← routes queries to the right profile
├── program.md                     ← agent instructions
├── results.tsv                    ← experiment log (exp_id, recall, latency, cost)
├── ARCHITECTURE_recall.md         ← auto-generated after recall run
├── ARCHITECTURE_pareto.md         ← auto-generated after pareto run
├── requirements.txt
├── .env                           ← GOOGLE_API_KEY
├── data/                          ← dataset + faiss index (gitignored)
│   ├── movies.pkl
│   ├── faiss.index
│   └── embeddings.pkl
├── experiments/
│   ├── log.jsonl                  ← full replay log (code + metrics per exp)
│   └── prompts/                   ← deduplicated prompts keyed by hash
└── search_profiles/
    ├── __init__.py
    ├── registry.py                ← auto-generated, updated on every keep
    ├── high_recall.py             ← best recall implementation
    ├── balanced.py                ← best pareto implementation
    └── low_latency.py             ← best latency implementation
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/a-j-i-n-k-y-a/autoresearch-search.git
cd autoresearch-search
git checkout autoresearchiter2/may10
pip install -r requirements.txt
```

### 2. Set environment variables

Create a `.env` file:

```
GOOGLE_API_KEY=your_google_api_key_here
```

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

For Claude Opus 4:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 3. Build dataset and index (one-time, ~2 min)

```bash
python prepare.py
```

Downloads 100k movies, builds FAISS index, saves to `data/`.

### 4. Verify setup

```bash
python agent_loop.py --eval-only
```

Expected output:
```
recall@10    : 0.900000
latency      : ~14ms
```

---

## Running experiments

### Benchmark current search.py

```bash
python agent_loop.py --eval-only
```

### Run the experiment loop

```bash
# Step 1 — maximize recall (run first)
python agent_loop.py --n 50 --objective recall

# Step 2 — trim latency without dropping recall (run second)
python agent_loop.py --n 30 --objective pareto

# Step 3 — minimize latency (optional)
python agent_loop.py --n 20 --objective latency
```

Every `✅ KEEP` automatically updates `search_profiles/<profile>.py` and `registry.py`.

### Replay a past experiment

```bash
# Restore search.py to a specific experiment's code and benchmark it
python agent_loop.py --replay exp_042

# Find exp_id from a git commit hash
grep -B2 "ca53447" experiments/log.jsonl

# Restore your best version after replay
git checkout -- search.py
```

### Manually export current search.py as a profile

```bash
python agent_loop.py --export-profile recall
python agent_loop.py --export-profile pareto
python agent_loop.py --export-profile latency
```

---

## Using the router

```bash
# List all available profiles
python router.py --profiles

# Query with auto-inferred constraint
python router.py "psychological thriller with a twist"

# Query with explicit constraint
python router.py "inception" --constraint low_latency
python router.py "robot consciousness AI" --constraint high_recall
python router.py "romantic comedy enemies to lovers" --constraint balanced

# Demo mode — runs example queries across all profiles
python router.py --demo
```

### Constraint guide

| Constraint | Recall | Latency | Use when |
|---|---|---|---|
| `high_recall` | 0.900 | ~14ms | Full search page, recommendations |
| `balanced` | 0.800 | ~25ms | Default — general search |
| `low_latency` | 0.600 | ~9ms | Autocomplete, typeahead |

### From Python

```python
from router import route

response = route("dark sci-fi about consciousness", constraint="high_recall")
for movie in response["results"]:
    print(movie["title"])
```

---

## Switching to Claude Opus 4

In `agent_loop.py`, change three lines:

```python
GOOGLE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL          = "claude-opus-4-5"

client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://api.anthropic.com/v1"
)
```

Everything else stays the same — Anthropic's API is OpenAI-compatible.

---

## Benchmark queries

Five fixed queries evaluated against 100k movies. Ground truth — never change.

| Query | Expected |
|---|---|
| dream heist movie Leonardo DiCaprio layers of subconscious | Inception |
| astronaut stranded in space wormhole black hole | Interstellar, Gravity |
| robot humanoid artificial intelligence consciousness | Ex Machina, A.I. Artificial Intelligence |
| psychological thriller unreliable narrator mind bending twist | Shutter Island, Black Swan |
| time machine going back to the future paradox | Back to the Future, Looper |