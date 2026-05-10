# Autoresearch — Movie Search Optimizer



---

## How it works

```
baseline search.py
     ↓
ask LLM for modification
     ↓
apply → benchmark → compute metrics
     ↓
improvement? → keep + commit
no improvement? → discard + restore
     ↓
repeat
```

Three constraints are tracked separately as a tuple — never collapsed into one score:

| Constraint | What it measures |
|---|---|
| `recall@10` | Fraction of expected movies found in top 10 results |
| `latency_ms` | Wall clock milliseconds per query |
| `llm_cost_usd` | USD cost of the API call that generated the search spec |

---

## Project structure

```
search.py           ← the only file the agent edits (your deliverable)
agent_loop.py       ← autonomous experiment loop
prepare.py          ← fixed data prep and evaluation (do not modify)
program.md          ← agent instructions (the research org config)
results.tsv         ← full experiment log (recall, latency, cost per run)
ARCHITECTURE.md     ← auto-generated explainer of the winning algorithm
experiments/
  log.jsonl         ← full replay log (code + metrics per experiment)
  prompts/          ← deduplicated prompt store keyed by hash
requirements.txt    ← dependencies
```

---

## Quick start

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
git checkout autoresearchiter2/may10
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```bash
# For Groq (default)
GROQ_API_KEY=your_groq_api_key_here

# For Anthropic / Claude Opus 4 (interviewer benchmark)
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 4. Download dataset and build index (one-time, ~2 min)

```bash
python prepare.py
```

This downloads the movies dataset, builds the FAISS index, and saves both to `data/`. Only needs to run once.

### 5. Verify setup

```bash
python agent_loop.py --eval-only
```

Expected output:
```
recall@10    : 0.600000
latency      : ~30ms
```

---

## Running experiments

### Run 20 experiments optimizing for recall (default)

```bash
python agent_loop.py --n 20 --objective recall
```

### Run optimizing for latency

```bash
python agent_loop.py --n 20 --objective latency
```

### Run optimizing for LLM cost

```bash
python agent_loop.py --n 20 --objective cost
```

### Run in Pareto mode (improve one constraint without hurting others)

```bash
python agent_loop.py --n 20 --objective pareto
```

### Replay a specific experiment exactly

```bash
python agent_loop.py --replay exp_007
```

---

## Switching to Claude Opus 4

In `agent_loop.py`, change these three lines:

```python
GROQ_API_KEY = os.getenv("ANTHROPIC_API_KEY")
MODEL        = "claude-opus-4-5"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.anthropic.com/v1"
)
```

Everything else stays the same — Anthropic's API is OpenAI-compatible.

---

## Outputs

After a run completes you'll have:

| File | Contents |
|---|---|
| `search.py` | Best search implementation found |
| `results.tsv` | Full experiment trail — recall, latency, cost, status per run |
| `ARCHITECTURE.md` | Auto-generated explanation of the winning algorithm |
| `experiments/log.jsonl` | Complete replay log — every experiment's code and metrics |
| `experiments/prompts/` | Deduplicated prompt store for full auditability |

---

## Benchmark queries

The evaluation runs 5 fixed queries against a 100k movie dataset and measures recall@10 — how many expected movies appear in the top 10 results.

```python
"dream heist movie Leonardo DiCaprio layers of subconscious"  → Inception
"astronaut stranded in space wormhole black hole"             → Interstellar, Gravity
"robot humanoid artificial intelligence consciousness"        → Ex Machina, A.I.
"psychological thriller unreliable narrator mind bending"     → Shutter Island, Black Swan
"time machine going back to the future paradox"              → Back to the Future, Looper
```

---

## Design notes

- **Code is the weights** — there is no model training. The agent does code search across the space of possible retrieval algorithms. `search.py` is the artifact.
- **Replayability** — every experiment's exact code is saved in `experiments/log.jsonl`. Any run can be restored and re-benchmarked with `--replay`.
- **Prompt deduplication** — prompts are stored once by MD5 hash, keeping log records small (~400 bytes each).
- **Architecture documentation** — at the end of every run, the agent writes `ARCHITECTURE.md` explaining what it discovered in plain English.
- **Git-tracked** — every kept experiment is a commit. `git log --oneline` is a readable history of what worked.