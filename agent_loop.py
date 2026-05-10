# agent_loop.py
import time
import sys
import os
import json
import hashlib
import argparse
from dotenv import load_dotenv
import subprocess
from openai import OpenAI
from prepare import load_resources, evaluate, BENCHMARK_QUERIES

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL        = "llama-3.3-70b-versatile"

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ─── LLM COST ───────────────────────────────────────────────
# Groq llama-3.3-70b pricing (per 1k tokens)
COST_PER_1K_INPUT  = 0.00059
COST_PER_1K_OUTPUT = 0.00079

def get_llm_cost(response):
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    return round(
        (input_tokens  / 1000 * COST_PER_1K_INPUT) +
        (output_tokens / 1000 * COST_PER_1K_OUTPUT),
        6
    )

# ─── OBJECTIVE ──────────────────────────────────────────────
def is_improvement(new, best, objective):
    """
    new / best are metric dicts: { recall, latency_ms, llm_cost_usd }
    Constraints are never collapsed into one number.
    The user picks the objective at runtime via --objective.
    """
    if objective == "recall":
        return new["recall"] > best["recall"]

    elif objective == "latency":
        return new["latency_ms"] < best["latency_ms"]

    elif objective == "cost":
        return new["llm_cost_usd"] < best["llm_cost_usd"]

    elif objective == "pareto":
        # Keep only if better on at least one axis, worse on none
        better_on_one = (
            new["recall"]       >  best["recall"]       or
            new["latency_ms"]   <  best["latency_ms"]   or
            new["llm_cost_usd"] <  best["llm_cost_usd"]
        )
        worse_on_none = (
            new["recall"]       >= best["recall"]       and
            new["latency_ms"]   <= best["latency_ms"]   and
            new["llm_cost_usd"] <= best["llm_cost_usd"]
        )
        return better_on_one and worse_on_none

    return False

# ─── EVAL ───────────────────────────────────────────────────
def run_eval():
    """Run benchmark on current search.py — returns (recall, latency_ms)."""
    df, bm25, model, index = load_resources()

    if "search" in sys.modules:
        del sys.modules["search"]
    import search as search_module

    start   = time.time()
    recall  = evaluate(search_module.search, df, bm25, model, index)
    elapsed = (time.time() - start) * 1000 / len(BENCHMARK_QUERIES)

    return recall, elapsed

# ─── FILE / GIT HELPERS ─────────────────────────────────────
def read_file(path):
    with open(path, "r") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)

def git_commit(message):
    subprocess.run(["git", "add", "search.py"])
    subprocess.run(["git", "commit", "-m", message])
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True
    )
    return result.stdout.strip()

def git_restore():
    """Restore search.py to last commit — atomic, keeps git state clean."""
    subprocess.run(["git", "checkout", "--", "search.py"])

# ─── LOGGING ────────────────────────────────────────────────
def log_result(commit, metrics, status, description):
    """Append one row to results.tsv — three separate metric columns."""
    if not os.path.exists("results.tsv"):
        with open("results.tsv", "w") as f:
            f.write("commit\trecall\tlatency_ms\tllm_cost_usd\tstatus\tdescription\n")
    with open("results.tsv", "a") as f:
        f.write(
            f"{commit}\t"
            f"{metrics['recall']:.6f}\t"
            f"{metrics['latency_ms']:.1f}\t"
            f"{metrics['llm_cost_usd']:.6f}\t"
            f"{status}\t"
            f"{description}\n"
        )

# ─── REPLAYABILITY ──────────────────────────────────────────
def save_prompt(prompt):
    """
    Save prompt to experiments/prompts/<hash>.txt.
    Deduplicates — identical prompts share one file.
    Returns the hash for referencing in the log.
    """
    prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
    prompt_dir  = "experiments/prompts"
    os.makedirs(prompt_dir, exist_ok=True)
    prompt_path = f"{prompt_dir}/{prompt_hash}.txt"
    if not os.path.exists(prompt_path):
        with open(prompt_path, "w") as f:
            f.write(prompt)
    return prompt_hash

def save_experiment(exp_id, prompt, search_py, metrics, status, description):
    """
    Save experiment record to experiments/log.jsonl.
    Prompt stored separately by hash — log line stays tiny (~400 bytes).
    """
    os.makedirs("experiments", exist_ok=True)

    prompt_hash = save_prompt(prompt) if prompt else "none"

    record = {
        "exp_id":      exp_id,
        "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_hash": prompt_hash,
        "search_py":   search_py,
        "metrics":     metrics,
        "status":      status,
        "description": description,
    }
    with open("experiments/log.jsonl", "a") as f:
        f.write(json.dumps(record, indent = 2) + "\n\n")

def replay_experiment(exp_id):
    """Restore search.py from a saved experiment and re-run the benchmark."""
    log_path = "experiments/log.jsonl"
    if not os.path.exists(log_path):
        print("No experiments/log.jsonl found.")
        return

    found = None
    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            if record["exp_id"] == exp_id:
                found = record
                break

    if not found:
        print(f"Experiment {exp_id} not found in log.")
        return

    print(f"\n🔁 Replaying: {exp_id}")
    print(f"   Description : {found['description']}")
    print(f"   Original    : recall={found['metrics']['recall']:.3f}  "
          f"latency={found['metrics']['latency_ms']:.1f}ms  "
          f"cost=${found['metrics']['llm_cost_usd']:.6f}")

    prompt_path = f"experiments/prompts/{found['prompt_hash']}.txt"
    if os.path.exists(prompt_path):
        print(f"   Prompt      : {prompt_path}")

    write_file("search.py", found["search_py"])

    try:
        recall, latency = run_eval()
        print(f"   Replayed    : recall={recall:.3f}  latency={latency:.1f}ms")
    except Exception as e:
        print(f"   💥 Replay crashed: {e}")

# ─── API CALL WITH RETRY ────────────────────────────────────
def call_api(messages, retries=3):
    for attempt in range(retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages
            )
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 60 * (attempt + 1)
                print(f"⏳ Rate limit — waiting {wait}s...")
                time.sleep(wait)
            else:
                raise

# ─── AGENT PROMPTS ──────────────────────────────────────────
def format_history_entry(r):
    line = (
        f"- {r['description']}: "
        f"recall={r['metrics']['recall']:.3f}  "
        f"latency={r['metrics']['latency_ms']:.1f}ms  "
        f"cost=${r['metrics']['llm_cost_usd']:.6f}  "
        f"({r['status']})"
    )
    if r.get("traceback"):
        line += f"\n  ERROR:\n{r['traceback'][:500]}"
    return line

def ask_agent(program_md, search_py, history, objective):
    """Ask LLM for next search.py modification. Returns (code, cost, description, prompt)."""

    history_str = "\n".join([format_history_entry(r) for r in history[-10:]])

    objective_guidance = {
        "recall":  "Maximize recall@10. Latency and cost are secondary.",
        "latency": "Minimize latency_ms. Recall must stay above 0.5 or the change is useless.",
        "cost":    "Minimize LLM tokens used to generate the spec — write simpler, shorter search.py so the next prompt is cheaper.",
        "pareto":  "Only keep changes that improve at least one of (recall, latency, cost) without hurting any other.",
    }[objective]

    prompt = f"""
You are an autonomous research agent improving a movie search system.

## Your instructions (program.md):
{program_md}

## Current search.py:
```python
{search_py}
```

## Experiment history (last 10):
{history_str if history_str else "No experiments yet — this is the baseline."}

## Active objective: {objective}
{objective_guidance}

## Three constraints tracked separately (never combined into one score):
- recall@10      — fraction of expected movies found in top 10 results
- latency_ms     — wall clock milliseconds per query
- llm_cost_usd   — cost of this API call to generate the spec

## Your task:
Suggest ONE specific modification to search.py to improve the active objective.
Think step by step. Avoid repeating crashed or discarded ideas.

ENVIRONMENT CONSTRAINTS — violations crash the run:
- Allowed imports ONLY: numpy, rank_bm25, sentence_transformers, faiss, sklearn
- nltk is NOT installed — do not import it
- Use int() or np.int64(), NEVER np.int (deprecated in modern numpy)
- Do NOT import from other project files

Promising strategies:
- Reciprocal Rank Fusion (RRF): combine BM25 + FAISS by rank position, not raw scores
- Larger candidate pool: top_k * 15 or * 20 before re-ranking
- Title-boosted BM25: run BM25 on title field separately, merge results
- Vote score boosting: weight final scores by log(vote_count) * vote_average from df
- Pure BM25 only: skip embedding entirely — faster and cheaper
- Cosine similarity: normalize vectors before FAISS scoring

# DESCRIPTION: <5 words or less describing the change>
Return the description line above FIRST, then immediately the raw Python code.
No markdown fences. No explanation. Just the description comment then the code.
"""

    response = call_api([{"role": "user", "content": prompt}])
    cost     = get_llm_cost(response)
    raw      = response.choices[0].message.content.strip()

    # Strip markdown fences if model ignored instructions
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw   = "\n".join(lines).strip()

    # Extract description from first line
    lines = raw.split("\n")
    if lines[0].startswith("# DESCRIPTION:"):
        description = lines[0].replace("# DESCRIPTION:", "").strip()
        code        = "\n".join(lines[1:]).strip()
    else:
        description = "unlabeled experiment"
        code        = raw

    return code, cost, description, prompt

# ─── ARCHITECTURE DOCUMENTATION ─────────────────────────────
def document_architecture(best_metrics, baseline_metrics, objective, n_experiments, history):
    """
    Generate ARCHITECTURE.md explaining the final winning search implementation.
    Called once at the end of the experiment loop.
    """
    print("\n📄 Documenting final architecture...")

    search_py = read_file("search.py")

    kept     = [r for r in history if r["status"] == "keep" and r["description"] != "baseline"]
    discarded = [r for r in history if r["status"] == "discard"]
    crashed  = [r for r in history if r["status"] == "crash"]

    prompt = f"""
You are documenting the final architecture of an autonomously optimised movie search system.

## Winning search.py:
```python
{search_py}
```

## Experiment summary:
- Baseline  → recall={baseline_metrics['recall']:.3f}  latency={baseline_metrics['latency_ms']:.1f}ms
- Final     → recall={best_metrics['recall']:.3f}  latency={best_metrics['latency_ms']:.1f}ms  cost=${best_metrics['llm_cost_usd']:.6f}
- Objective : {objective}
- Total experiments run : {n_experiments}
- Kept      : {len(kept)}
- Discarded : {len(discarded)}
- Crashed   : {len(crashed)}

## Full experiment history:
{chr(10).join([format_history_entry(r) for r in history])}

Write a concise ARCHITECTURE.md with these sections:

# Architecture

## What it does
Plain English explanation of the retrieval algorithm — no jargon.

## Components
Brief explanation of each component and why it's there.

## Why it works
Intuition behind the design — what each part contributes to recall/latency.

## Tradeoffs
What was sacrificed or prioritised for the active objective ({objective}).

## Key experiments
Which changes actually moved the needle, and what failed. Be specific.

## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | {baseline_metrics['recall']:.3f} | {best_metrics['recall']:.3f} |
| latency_ms | {baseline_metrics['latency_ms']:.1f} | {best_metrics['latency_ms']:.1f} |

## How to run
```bash
python agent_loop.py --eval-only
```

Return only the markdown. No preamble, no commentary.
"""

    response = call_api([{"role": "user", "content": prompt}])
    content  = response.choices[0].message.content.strip()

    write_file(f"ARCHITECTURE_{objective}.md", content)
    subprocess.run(["git", "add", f"ARCHITECTURE_{objective}.md"])

    # Also commit it so it lives in the git history
    subprocess.run(["git", "add", "ARCHITECTURE.md"])
    subprocess.run(["git", "commit", "-m", "docs: add final architecture"])

    print(f"   Saved → ARCHITECTURE.md  (committed to git)")

# ─── MAIN LOOP ──────────────────────────────────────────────
def run_experiment_loop(n_experiments=20, objective="recall"):

    program_md = read_file("program.md")
    history    = []

    print("=" * 60)
    print("AUTORESEARCH — Search System Optimizer")
    print(f"Experiments  : {n_experiments}")
    print(f"Objective    : {objective}")
    print(f"Tracking     : recall | latency_ms | llm_cost_usd  (separately)")
    print("=" * 60)

    # ── Baseline ──
    print("\n📊 Running baseline...")
    baseline_recall, baseline_latency = run_eval()
    baseline_metrics = {
        "recall":       baseline_recall,
        "latency_ms":   baseline_latency,
        "llm_cost_usd": 0.0
    }

    print(f"recall@10    : {baseline_recall:.6f}")
    print(f"latency      : {baseline_latency:.1f}ms")

    commit = git_commit("baseline")
    log_result(commit, baseline_metrics, "keep", "baseline")
    save_experiment("exp_000", "", read_file("search.py"),
                    baseline_metrics, "keep", "baseline")

    best_metrics = baseline_metrics.copy()
    history.append({
        "description": "baseline",
        "metrics":     baseline_metrics,
        "status":      "keep"
    })

    # ── Experiment loop ──
    for i in range(n_experiments):
        exp_id = f"exp_{i + 1:03d}"

        print(f"\n{'=' * 60}")
        print(f"EXPERIMENT {i + 1}/{n_experiments}  [{exp_id}]")
        print(
            f"Best so far  → recall={best_metrics['recall']:.3f}  "
            f"latency={best_metrics['latency_ms']:.1f}ms  "
            f"cost=${best_metrics['llm_cost_usd']:.6f}"
        )
        print("=" * 60)

        current_search_py = read_file("search.py")

        # Ask agent
        print("\n🤖 Asking agent...")
        try:
            new_code, llm_cost, description, prompt = ask_agent(
                program_md, current_search_py, history, objective
            )
        except Exception as e:
            print(f"Agent error: {e}")
            continue

        print(f"Trying       : {description}  (api cost: ${llm_cost:.6f})")

        write_file("search.py", new_code)

        # Benchmark
        print("Running benchmark...")
        try:
            new_recall, new_latency = run_eval()
        except Exception as e:
            import traceback
            crash_reason = traceback.format_exc()
            print(f"💥 Crash: {e}")

            crash_metrics = {
                "recall":       0.0,
                "latency_ms":   0.0,
                "llm_cost_usd": llm_cost
            }
            history.append({
                "description": description,
                "metrics":     crash_metrics,
                "status":      "crash",
                "traceback":   crash_reason
            })
            save_experiment(exp_id, prompt, new_code,
                            crash_metrics, "crash", description)
            git_restore()
            log_result("crash", crash_metrics, "crash", description)
            continue

        new_metrics = {
            "recall":       new_recall,
            "latency_ms":   new_latency,
            "llm_cost_usd": llm_cost
        }

        print(f"recall@10    : {new_recall:.6f}")
        print(f"latency      : {new_latency:.1f}ms")
        print(f"llm_cost     : ${llm_cost:.6f}")

        if is_improvement(new_metrics, best_metrics, objective):
            commit = git_commit(f"experiment: {description}")
            log_result(commit, new_metrics, "keep", description)
            save_experiment(exp_id, prompt, new_code,
                            new_metrics, "keep", description)
            print(f"✅ KEEP — improved on objective '{objective}'")
            best_metrics = new_metrics.copy()
            history.append({
                "description": description,
                "metrics":     new_metrics,
                "status":      "keep"
            })
        else:
            git_restore()
            log_result("discarded", new_metrics, "discard", description)
            save_experiment(exp_id, prompt, new_code,
                            new_metrics, "discard", description)
            print(f"❌ DISCARD — no improvement on '{objective}'")
            history.append({
                "description": description,
                "metrics":     new_metrics,
                "status":      "discard"
            })

    # ── Document final architecture ──
    document_architecture(best_metrics, baseline_metrics, objective, n_experiments, history)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Objective    : {objective}")
    print(
        f"Baseline     → recall={baseline_metrics['recall']:.3f}  "
        f"latency={baseline_metrics['latency_ms']:.1f}ms"
    )
    print(
        f"Best         → recall={best_metrics['recall']:.3f}  "
        f"latency={best_metrics['latency_ms']:.1f}ms  "
        f"cost=${best_metrics['llm_cost_usd']:.6f}"
    )
    print(f"\nFull log     : results.tsv")
    print(f"Replay log   : experiments/log.jsonl")
    print(f"Prompts      : experiments/prompts/")
    print(f"Architecture : ARCHITECTURE.md")
    print(f"Best search  : current state of search.py")


# ─── ENTRYPOINT ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Benchmark current search.py and exit"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of experiments (default: 20)"
    )
    parser.add_argument(
        "--objective",
        choices=["recall", "latency", "cost", "pareto"],
        default="recall",
        help="What to optimize: recall | latency | cost | pareto  (default: recall)"
    )
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Replay a specific experiment by ID e.g. exp_007"
    )
    args = parser.parse_args()

    if args.replay:
        replay_experiment(args.replay)

    elif args.eval_only:
        recall, latency = run_eval()
        print(f"recall@10    : {recall:.6f}")
        print(f"latency      : {latency:.1f}ms")

    else:
        run_experiment_loop(n_experiments=args.n, objective=args.objective)