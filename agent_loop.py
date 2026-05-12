# agent_loop.py
import time
import sys
import os
import json
import hashlib
import argparse
import shutil
from dotenv import load_dotenv
import subprocess
from openai import OpenAI
from prepare import load_resources, evaluate, BENCHMARK_QUERIES

load_dotenv()

# Active: Google Gemini 3.1 Flash-Lite
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL          = "gemini-3.1-flash-lite-preview"

client = OpenAI(
    api_key=GOOGLE_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# ─── LLM COST ───────────────────────────────────────────────
# Google Gemini 3.1 Flash-Lite pricing (per 1k tokens)
COST_PER_1K_INPUT  = 0.000100
COST_PER_1K_OUTPUT = 0.000400

# ── Switch to Claude Opus 4 ──────────────────────────────────
# Comment out the block above and uncomment below:
#
# GOOGLE_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# MODEL          = "claude-opus-4-5"
#
# client = OpenAI(
#     api_key=GOOGLE_API_KEY,
#     base_url="https://api.anthropic.com/v1"
# )
# ─────────────────────────────────────────────────────────────

# ── Claude Opus 4 pricing (per 1k tokens) ────────────────────
# Uncomment if switching to Opus 4:
# COST_PER_1K_INPUT  = 0.015
# COST_PER_1K_OUTPUT = 0.075
# ─────────────────────────────────────────────────────────────


# ─── PROFILE MAPPING ────────────────────────────────────────
# Maps --objective → search_profiles/<name>.py
OBJECTIVE_TO_PROFILE = {
    "recall":  "high_recall",
    "latency": "low_latency",
    "pareto":  "balanced",
    "cost":    "low_cost",
}

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
    """
    if objective == "recall":
        return new["recall"] > best["recall"]

    elif objective == "latency":
        return new["latency_ms"] < best["latency_ms"]

    elif objective == "cost":
        return new["llm_cost_usd"] < best["llm_cost_usd"]

    elif objective == "pareto":
        if new["recall"] < best["recall"]:
            return False
        if new["latency_ms"] > best["latency_ms"] * 1.10:
            return False
        if new["llm_cost_usd"] > best["llm_cost_usd"] * 2.0 and best["llm_cost_usd"] > 0:
            return False
        return (
            new["recall"]     > best["recall"] or
            new["latency_ms"] < best["latency_ms"]
        )

    return False

def is_baseline(description):
    """Robust check — catches 'baseline', 'baseline bm25', etc."""
    return "baseline" in description.lower()

def best_from_history(history, objective):
    """Find the best kept experiment from history for the given objective."""
    kept = [
        r for r in history
        if r["status"] == "keep" and not is_baseline(r["description"])
    ]
    if not kept:
        return None

    if objective == "recall":
        return max(kept, key=lambda r: r["metrics"]["recall"])
    elif objective == "latency":
        return min(kept, key=lambda r: r["metrics"]["latency_ms"])
    elif objective == "cost":
        return min(kept, key=lambda r: r["metrics"]["llm_cost_usd"])
    elif objective == "pareto":
        return max(kept, key=lambda r: r["metrics"]["recall"])
    return None

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

# ─── PROFILE AUTO-UPDATE ────────────────────────────────────
def update_profile(objective, metrics, description):
    """
    Called whenever a new best is kept.
    Copies current search.py to search_profiles/<profile_name>.py
    and regenerates search_profiles/registry.py.
    """
    profile_name = OBJECTIVE_TO_PROFILE.get(objective)
    if not profile_name:
        return

    os.makedirs("search_profiles", exist_ok=True)

    # Create __init__.py if missing
    init_path = "search_profiles/__init__.py"
    if not os.path.exists(init_path):
        write_file(init_path, "")

    # Copy winning search.py to profile
    profile_path = f"search_profiles/{profile_name}.py"
    shutil.copy("search.py", profile_path)
    print(f"   📦 Profile updated → {profile_path}")

    # Regenerate registry.py from all existing profiles
    regenerate_registry(objective, profile_name, metrics, description)

def regenerate_registry(updated_objective, updated_profile, updated_metrics, updated_description):
    """
    Reads all existing profile files and regenerates registry.py.
    Merges new metrics with existing ones.
    """
    registry_path = "search_profiles/registry.py"

    # Load existing registry data if present
    existing = {}
    if os.path.exists(registry_path):
        try:
            import importlib.util
            spec   = importlib.util.spec_from_file_location("registry", registry_path)
            mod    = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            existing = getattr(mod, "PROFILES", {})
        except Exception:
            existing = {}

    # Metadata for each profile
    profile_meta = {
        "high_recall": {
            "objective":    "recall",
            "description":  "maximize recall — best for full search pages",
            "use_when":     "accuracy matters most — full search, recommendation",
        },
        "low_latency": {
            "objective":    "latency",
            "description":  "minimize latency — fastest response",
            "use_when":     "speed matters most — autocomplete, typeahead",
        },
        "balanced": {
            "objective":    "pareto",
            "description":  "best recall/latency tradeoff",
            "use_when":     "default — general search, no specific SLA",
        },
        "low_cost": {
            "objective":    "cost",
            "description":  "minimize LLM cost — simplest implementation",
            "use_when":     "high volume, cost-sensitive workloads",
        },
    }

    # Build updated PROFILES dict
    profiles = dict(existing)

    # Update the profile that just improved
    profiles[updated_profile] = {
        "module":      f"search_profiles.{updated_profile}",
        "recall":      updated_metrics["recall"],
        "latency_ms":  updated_metrics["latency_ms"],
        "cost_usd":    updated_metrics["llm_cost_usd"],
        "description": updated_description,
        **profile_meta.get(updated_profile, {}),
    }

    # Write registry.py
    lines = [
        "# search_profiles/registry.py",
        "# AUTO-GENERATED by agent_loop.py — do not edit manually.",
        "# Updated whenever a new best experiment is kept.",
        "# Cross-reference with experiments/log.jsonl by exp_id.",
        "",
        "PROFILES = {",
    ]

    for name, p in profiles.items():
        lines.append(f"    \"{name}\": {{")
        for k, v in p.items():
            if isinstance(v, str):
                lines.append(f"        \"{k}\": \"{v}\",")
            else:
                lines.append(f"        \"{k}\": {v},")
        lines.append("    },")

    lines += [
        "}",
        "",
        "# Fallback when constraint is unknown or profile file missing",
        "DEFAULT_PROFILE = \"balanced\"",
        "",
    ]

    write_file(registry_path, "\n".join(lines))
    print(f"   📋 Registry updated → {registry_path}")

# ─── LOGGING ────────────────────────────────────────────────
RESULTS_HEADER = "exp_id\tcommit\trecall\tlatency_ms\tllm_cost_usd\tstatus\tdescription\n"

def ensure_results_header():
    """
    Create results.tsv with correct header if missing.
    If file exists with old header (no exp_id), migrate it.
    """
    if not os.path.exists("results.tsv"):
        with open("results.tsv", "w") as f:
            f.write(RESULTS_HEADER)
        return

    with open("results.tsv", "r") as f:
        first_line = f.readline()

    # Migrate old header that's missing exp_id
    if first_line.startswith("commit\t") or not first_line.startswith("exp_id\t"):
        with open("results.tsv", "r") as f:
            content = f.read()
        old_lines = content.split("\n")
        # Replace header, keep data rows
        new_lines = [RESULTS_HEADER.strip()] + old_lines[1:]
        with open("results.tsv", "w") as f:
            f.write("\n".join(new_lines))
        print("   ⚠️  Migrated results.tsv header to include exp_id column")

def log_result(exp_id, commit, metrics, status, description):
    """Append one row to results.tsv — exp_id + three separate metric columns."""
    ensure_results_header()
    with open("results.tsv", "a") as f:
        f.write(
            f"{exp_id}\t"
            f"{commit}\t"
            f"{metrics['recall']:.6f}\t"
            f"{metrics['latency_ms']:.1f}\t"
            f"{metrics['llm_cost_usd']:.6f}\t"
            f"{status}\t"
            f"{description}\n"
        )

# ─── REPLAYABILITY ──────────────────────────────────────────
def save_prompt(prompt):
    """Save prompt deduplicated by hash. Returns hash for log reference."""
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
    Pretty-printed, separated by blank lines.
    Cross-referenceable with results.tsv via exp_id.
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
        f.write(json.dumps(record, indent=2) + "\n\n")

def load_experiments():
    """Load all records from log.jsonl — handles multi-line pretty-printed JSON."""
    log_path = "experiments/log.jsonl"
    if not os.path.exists(log_path):
        return []
    records = []
    with open(log_path) as f:
        content = f.read()
    for chunk in content.strip().split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            try:
                records.append(json.loads(chunk))
            except json.JSONDecodeError:
                continue
    return records

def replay_experiment(exp_id):
    """Restore search.py from a saved experiment and re-run the benchmark."""
    records = load_experiments()
    if not records:
        print("No experiments/log.jsonl found.")
        return

    found = next((r for r in records if r["exp_id"] == exp_id), None)
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

    _, clean_code = parse_agent_response(found["search_py"])
    write_file("search.py", clean_code)

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

def parse_agent_response(raw):
    """
    Robustly extract (description, code) from any LLM response format.
    Handles: bare label, # DESCRIPTION: label, fences, raw code.
    """
    raw = raw.strip()
    lines = raw.split("\n")
    first = lines[0].strip()

    is_code_line = (
        first.startswith("import")
        or first.startswith("from")
        or first.startswith("def ")
        or first.startswith("class ")
        or first.startswith("```")
        or first.startswith("#!")
    )

    if not is_code_line and len(first) < 80:
        description = first.replace("# DESCRIPTION:", "").strip()
        raw         = "\n".join(lines[1:]).strip()
    else:
        description = "unlabeled experiment"

    lines = raw.split("\n")
    lines = [l for l in lines if not l.strip().startswith("```")]
    code  = "\n".join(lines).strip()

    return description, code

def ask_agent(program_md, search_py, history, objective):
    """Ask LLM for next search.py modification. Returns (code, cost, description, prompt)."""

    history_str = "\n".join([format_history_entry(r) for r in history[-10:]])

    tried_descriptions = [
        r['description'] for r in history
        if not is_baseline(r['description'])
    ]
    tried_str = "\n".join(f"  - {d}" for d in tried_descriptions) if tried_descriptions else "  (none yet)"

    already_in_code = []
    if "BM25Okapi(df['title']" in search_py or "title_bm25" in search_py:
        already_in_code.append("Title-boosted BM25")
    if "rrf" in search_py.lower() or "1 / (" in search_py:
        already_in_code.append("Reciprocal Rank Fusion (RRF)")
    if "vote_count" in search_py or "vote_average" in search_py:
        already_in_code.append("Vote score boosting")
    if "top_k * 1" in search_py:
        already_in_code.append("Large candidate pool")
    if "cosine" in search_py.lower():
        already_in_code.append("Cosine similarity")
    if "genres" in search_py:
        already_in_code.append("Genre boosting")
    already_str = "\n".join(f"  - {s}" for s in already_in_code) if already_in_code else "  (none detected)"

    objective_guidance = {
        "recall":  "Maximize recall@10. Latency and cost are secondary.",
        "latency": "Minimize latency_ms. Recall must stay above 0.5 or the change is useless.",
        "cost":    "Minimize LLM tokens — write simpler, shorter search.py so the next prompt is cheaper.",
        "pareto":  "Improve recall or reduce latency. Recall must not drop. Latency must not increase more than 10%.",
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

## ALREADY TRIED across all runs — do not repeat these:
{tried_str}
If your idea is similar to any of the above, pick something structurally different.

## ALREADY IN THE CODE — do not re-add these:
{already_str}

## Think structurally different. Ask yourself:
- Can I remove something and get equal recall? Simpler = better.
- What fields in df am I NOT using? (genres, vote_count, vote_average)
- Can I combine two partial wins from history into one change?
- Would a second embedding pass on the top 20 candidates improve ranking?
- What would a human do differently for these specific queries?

## ENVIRONMENT CONSTRAINTS — violations crash the run:
- Allowed imports ONLY: numpy, rank_bm25, sentence_transformers, faiss, sklearn
- nltk is NOT installed — do not import it
- Use int() or np.int64(), NEVER np.int (deprecated)
- Do NOT import from other project files

# DESCRIPTION: <5 words or less describing the change>
Return the description line above FIRST, then immediately the raw Python code.
No markdown fences. No explanation. Just the description comment then the code.
"""

    response          = call_api([{"role": "user", "content": prompt}])
    cost              = get_llm_cost(response)
    raw               = response.choices[0].message.content.strip()
    description, code = parse_agent_response(raw)

    return code, cost, description, prompt

# ─── ARCHITECTURE DOCUMENTATION ─────────────────────────────
def document_architecture(best_metrics, baseline_metrics, objective, n_experiments, history):
    """Generate ARCHITECTURE_<objective>.md at end of run."""
    print("\n📄 Documenting final architecture...")

    search_py = read_file("search.py")
    kept      = [r for r in history if r["status"] == "keep" and not is_baseline(r["description"])]
    discarded = [r for r in history if r["status"] == "discard"]
    crashed   = [r for r in history if r["status"] == "crash"]

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
- Total experiments : {n_experiments}  |  Kept: {len(kept)}  |  Discarded: {len(discarded)}  |  Crashed: {len(crashed)}

## Full experiment history:
{chr(10).join([format_history_entry(r) for r in history])}

Write a concise ARCHITECTURE.md with sections:
# Architecture
## What it does
## Components
## Why it works
## Tradeoffs
## Key experiments
## Metrics
| Metric | Baseline | Final |
|--------|----------|-------|
| recall@10 | {baseline_metrics['recall']:.3f} | {best_metrics['recall']:.3f} |
| latency_ms | {baseline_metrics['latency_ms']:.1f} | {best_metrics['latency_ms']:.1f} |
## How to run
Return only the markdown. No preamble.
"""

    response = call_api([{"role": "user", "content": prompt}])
    content  = response.choices[0].message.content.strip()

    filename = f"ARCHITECTURE_{objective}.md"
    write_file(filename, content)
    subprocess.run(["git", "add", filename])
    subprocess.run(["git", "commit", "-m", f"docs: architecture for objective={objective}"])
    print(f"   Saved → {filename}")

# ─── MAIN LOOP ──────────────────────────────────────────────
def run_experiment_loop(n_experiments=20, objective="recall"):

    program_md = read_file("program.md")

    # ── Load full history from all previous runs ──
    history      = []
    past_records = load_experiments()
    if past_records:
        for r in past_records:
            history.append({
                "description": r["description"],
                "metrics":     r["metrics"],
                "status":      r["status"],
            })
        print(f"📚 Loaded {len(history)} past experiments from log")
    else:
        print("📚 No past experiments found — starting fresh")

    # Ensure results.tsv has correct header
    ensure_results_header()

    print("=" * 60)
    print("AUTORESEARCH — Search System Optimizer")
    print(f"Experiments  : {n_experiments}")
    print(f"Objective    : {objective}")
    print(f"Profile      : {OBJECTIVE_TO_PROFILE.get(objective, 'unknown')}")
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

    # ── Resume from best prior result for this objective ──
    best_prior = best_from_history(history, objective)
    if best_prior:
        best_metrics = best_prior["metrics"].copy()
        print(
            f"📈 Resuming from best prior: "
            f"recall={best_metrics['recall']:.3f}  "
            f"latency={best_metrics['latency_ms']:.1f}ms"
        )
    else:
        best_metrics = baseline_metrics.copy()
        print("📈 No prior kept experiments — using baseline as starting point")

    # Only log baseline on fresh start
    if not past_records:
        commit = git_commit("baseline")
        log_result("exp_000", commit, baseline_metrics, "keep", "baseline")
        save_experiment("exp_000", "", read_file("search.py"),
                        baseline_metrics, "keep", "baseline")
        history.append({
            "description": "baseline",
            "metrics":     baseline_metrics,
            "status":      "keep"
        })

    # ── Experiment loop ──
    next_exp_num = len(past_records) + 1

    for i in range(n_experiments):
        exp_id = f"exp_{next_exp_num + i:03d}"

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
            save_experiment(exp_id, prompt, new_code, crash_metrics, "crash", description)
            git_restore()
            log_result(exp_id, "crash", crash_metrics, "crash", description)
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
            log_result(exp_id, commit, new_metrics, "keep", description)
            save_experiment(exp_id, prompt, new_code, new_metrics, "keep", description)
            print(f"✅ KEEP — improved on objective '{objective}'")

            # ── Auto-update search profile ──
            update_profile(objective, new_metrics, description)

            best_metrics = new_metrics.copy()
            history.append({
                "description": description,
                "metrics":     new_metrics,
                "status":      "keep"
            })
        else:
            git_restore()
            log_result(exp_id, "discarded", new_metrics, "discard", description)
            save_experiment(exp_id, prompt, new_code, new_metrics, "discard", description)
            print(f"❌ DISCARD — no improvement on '{objective}'")
            history.append({
                "description": description,
                "metrics":     new_metrics,
                "status":      "discard"
            })

    # ── Document final architecture ──
    document_architecture(best_metrics, baseline_metrics, objective, n_experiments, history)

    # ── Summary ──
    cost_str = f"${best_metrics['llm_cost_usd']:.6f}" if best_metrics["llm_cost_usd"] > 0 else "n/a"
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Objective    : {objective}")
    print(f"Profile      : search_profiles/{OBJECTIVE_TO_PROFILE.get(objective)}.py")
    print(
        f"Baseline     → recall={baseline_metrics['recall']:.3f}  "
        f"latency={baseline_metrics['latency_ms']:.1f}ms"
    )
    print(
        f"Best         → recall={best_metrics['recall']:.3f}  "
        f"latency={best_metrics['latency_ms']:.1f}ms  "
        f"cost={cost_str}"
    )
    print(f"\nFull log     : results.tsv")
    print(f"Replay log   : experiments/log.jsonl")
    print(f"Prompts      : experiments/prompts/")
    print(f"Profile      : search_profiles/{OBJECTIVE_TO_PROFILE.get(objective)}.py")
    print(f"Registry     : search_profiles/registry.py")
    print(f"Architecture : ARCHITECTURE_{objective}.md")
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
    parser.add_argument(
        "--export-profile",
        type=str,
        choices=["recall", "latency", "cost", "pareto"],
        default=None,
        help="Manually export current search.py as a profile e.g. --export-profile recall"
    )
    args = parser.parse_args()

    if args.replay:
        replay_experiment(args.replay)

    elif args.export_profile:
        # Manually export current search.py as a profile
        recall, latency = run_eval()
        metrics = {"recall": recall, "latency_ms": latency, "llm_cost_usd": 0.0}
        update_profile(args.export_profile, metrics, "manually exported")
        print(f"Exported search.py → search_profiles/{OBJECTIVE_TO_PROFILE[args.export_profile]}.py")

    elif args.eval_only:
        recall, latency = run_eval()
        print(f"recall@10    : {recall:.6f}")
        print(f"latency      : {latency:.1f}ms")

    else:
        run_experiment_loop(n_experiments=args.n, objective=args.objective)