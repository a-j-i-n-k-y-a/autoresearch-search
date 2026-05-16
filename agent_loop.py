# agent_loop.py
import statistics
import ast
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
import filecmp
# ─── GLOBAL RESOURCE CACHE ─────────────────────────────

CACHED_RESOURCES = None


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

# — Cross-objective seeding.
# When starting pareto or latency runs, seed from the best recall profile
# so the agent refines a strong baseline rather than starting from scratch.
OBJECTIVE_SEED_FROM = {
    "pareto":  "high_recall",   # trim latency without dropping recall
    "latency": "high_recall",   # same logic — start strong, get faster
    "cost":    "balanced",      # balanced is a reasonable cost baseline
    "recall":  None,            # no cross-seeding — own objective
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
    Multi-objective comparison with explicit constraints.

    Metrics:
    {
        "recall": float,
        "latency_ms": float,
        "llm_cost_usd": float
    }

    Philosophy:
    - Constraints are enforced first
    - Then optimize the target metric
    - Never collapse metrics into one score
    """
     # ── Global constraints ─────────────────────────────
    RECALL_FLOOR = 0.50

    # Prevent catastrophic regressions
    MAX_RECALL_DROP_RATIO = 0.95

    # Pareto tolerance thresholds
    MAX_LATENCY_INCREASE = 1.10
    MAX_COST_INCREASE    = 2.00
    MIN_LATENCY_IMPROVEMENT = 0.05

    checks = {
        "recall_floor":    new["recall"] >= RECALL_FLOOR,
        "recall_drop":     new["recall"] >= best["recall"] * MAX_RECALL_DROP_RATIO,
        "latency_improve": (best["latency_ms"] - new["latency_ms"]) > MIN_LATENCY_IMPROVEMENT,
    }


    # ── Helper: quality gate ──────────────────────────
    def recall_ok():
        if new["recall"] < RECALL_FLOOR:
            return False

        # Avoid huge recall regressions even above floor
        if new["recall"] < best["recall"] * MAX_RECALL_DROP_RATIO:
            return False

        return True

    # ── Recall objective ──────────────────────────────
    if objective == "recall":
        return new["recall"] > best["recall"]

    # ── Latency objective ─────────────────────────────
    elif objective == "latency":
        if not recall_ok():
            return False
        return ( best["latency_ms"] - new["latency_ms"] ) > MIN_LATENCY_IMPROVEMENT

    # ── Cost objective ────────────────────────────────
    elif objective == "cost":

        # Cheaper garbage is still garbage
        if not recall_ok():
            return False

        return new["llm_cost_usd"] < best["llm_cost_usd"]

    # ── Pareto objective ──────────────────────────────
    elif objective == "pareto":

        MAX_PARETO_RECALL_DROP = 0.02

        # 1. Recall can drop, but only a tiny bit (max 0.02)
        if new["recall"] < best["recall"] - MAX_PARETO_RECALL_DROP:
            return False

        # 2. Latency can increase, but max 10%
        if new["latency_ms"] > best["latency_ms"] * MAX_LATENCY_INCREASE:
            return False

        # 3. Cost can increase, but max 2x
        if (
            best["llm_cost_usd"] != float("inf") and
            new["llm_cost_usd"] > best["llm_cost_usd"] * MAX_COST_INCREASE
        ):
            return False

        # 4. Must actually improve at least one thing
        return (
            new["recall"] > best["recall"] or
            new["latency_ms"] < best["latency_ms"] - 0.5 or
            new["llm_cost_usd"] < best["llm_cost_usd"]
        )
        

def is_baseline(description):
    """
    True only for the actual baseline experiment.
    Avoid accidental substring matches.
    """

    if not description:
        return False

    return description.strip().lower() == "baseline"
    
def best_from_history(history, objective):
    """Find the best kept experiment from history for the given objective."""
    RECALL_FLOOR = 0.5
    kept = [
        r for r in history
        if r["status"] == "keep"
        and r.get("objective") == objective
        and not is_baseline(r["description"])
        and r["metrics"]["recall"] >= RECALL_FLOOR  # hard gate here too
    ]
    if not kept:
        return None

    if objective == "recall":
        return max(kept, key=lambda r: r["metrics"]["recall"])
    elif objective == "latency":
        feasible = [
            r for r in kept
            if r["metrics"]["recall"] >= 0.50
        ]
        if not feasible:
            return None
        return min(feasible, key=lambda r: r["metrics"]["latency_ms"])
    elif objective == "cost":
        feasible = [
            r for r in kept
            if r["metrics"]["recall"] >= 0.50
        ]
        if not feasible:
            return None
        return min(feasible, key=lambda r: r["metrics"]["llm_cost_usd"])

    # we can also extract rue pareto history - midway recall, miway latency. currently doing
    # with recall max first in the history and reducing latency lateron
    elif objective == "pareto":
        return max(kept, key=lambda r: r["metrics"]["recall"])
    return None

# ─── EVAL ───────────────────────────────────────────────────


def run_eval():
    """
    Run benchmark on current search.py.

    Returns:
        (recall, latency_ms)

    Latency methodology:
    - warmup run
    - repeated trials
    - median latency
    """
    # we load once and cached the resources globally
    global CACHED_RESOURCES

    if CACHED_RESOURCES is None:
        print("📦 Loading shared resources...")
        CACHED_RESOURCES = load_resources()

    df, bm25, model, index = CACHED_RESOURCES

    # python caches the imports, so we need to delete the old one, and import the new one
    if "search" in sys.modules:
        del sys.modules["search"]

    import search as search_module

    # ── Warmup : run the 50 benchmarks but throw the result away, coz it contains noise, coz its the first run
    evaluate(search_module.search, df, bm25, model, index)

    # ── Recall (single deterministic run) ─
    recall = evaluate(search_module.search, df, bm25, model, index)

    # ── Latency trials : calculates per query latency (we divide by the len(benchmark_queries)),
    # and we don't take mean of the 5 trials we take median, coz if pc hiccups once it will skew the mean,
    # not the median
    trials = []

    N_TRIALS = 5

    for _ in range(N_TRIALS):
        start = time.perf_counter()

        evaluate(search_module.search, df, bm25, model, index)

        elapsed = (
            (time.perf_counter() - start)
            * 1000
            / len(BENCHMARK_QUERIES)
        )

        trials.append(elapsed)

    latency_ms = statistics.median(trials)

    return recall, latency_ms

# ─── FILE / GIT HELPERS ─────────────────────────────────────
def read_file(path):
    with open(path, "r") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w") as f:
        f.write(content)

def git_commit(message):
    """
    Commit current experiment state.

    Fails loudly if git operations fail.
    """

    try:

        add_result = subprocess.run(
            ["git", "add", "."],
            capture_output=True,
            text=True,
            check=True,
        )

        commit_result = subprocess.run(
            ["git", "commit", "-m", message],
            capture_output=True,
            text=True,
            check=True,
        )

    except FileNotFoundError:
        raise RuntimeError(
            "Git is not installed or not available in PATH."
        )

    except subprocess.CalledProcessError as e:

        raise RuntimeError(
            "\nGit commit failed.\n\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Return code: {e.returncode}\n\n"
            f"STDOUT:\n{e.stdout}\n\n"
            f"STDERR:\n{e.stderr}"
        )

def git_restore():
    """
    Restore search.py after failed/discarded experiment.

    Failure here is CRITICAL because optimizer state
    becomes corrupted if rollback does not succeed.
    """

    try:

        subprocess.run(
            [
                "git",
                "-c",
                "core.editor=true",
                "checkout",
                "--",
                "search.py"
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )

    except subprocess.TimeoutExpired:

        raise RuntimeError(
            "git restore timed out"
        )

    except FileNotFoundError:

        raise RuntimeError(
            "Git is not installed or unavailable in PATH."
        )

    except subprocess.CalledProcessError as e:

        raise RuntimeError(
            "\nCRITICAL: git restore failed.\n\n"
            f"Command: {' '.join(e.cmd)}\n"
            f"Return code: {e.returncode}\n\n"
            f"STDOUT:\n{e.stdout}\n\n"
            f"STDERR:\n{e.stderr}"
        )


def files_are_identical(path1, path2):
    """
    Byte-level comparison for profile deduplication.
    """
    return (
        os.path.exists(path1)
        and os.path.exists(path2)
        and filecmp.cmp(path1, path2, shallow=False)
    )

# ─── PROFILE AUTO-UPDATE ────────────────────────────────────
def update_profile(objective, metrics, description):
    """
    Called whenever a new best is kept.
    Copies current search.py to search_profiles/<profile_name>.py
    and regenerates search_profiles/registry.py.
    """
    RECALL_FLOOR = 0.5
    # figure out which profile to save to
    profile_name = OBJECTIVE_TO_PROFILE.get(objective)
    if not profile_name:
        return

    
    os.makedirs("search_profiles", exist_ok=True)

    # Create __init__.py if missing
    init_path = "search_profiles/__init__.py"
    if not os.path.exists(init_path):
        write_file(init_path, "")

    if metrics["recall"] < RECALL_FLOOR:
        print(f"⚠️ Skipping profile update — recall {metrics['recall']:.3f} below floor")
        return

    # Copy winning search.py to profile
    profile_path = f"search_profiles/{profile_name}.py"
    # Prevent exporting duplicate profiles
    for other_profile in OBJECTIVE_TO_PROFILE.values():

        other_path = f"search_profiles/{other_profile}.py"

        if (
            other_profile != profile_name
            and files_are_identical("search.py", other_path)
        ):
            print(
                f"   ⚠️ Profile identical to {other_profile}.py "
                f"— skipping export"
            )
            return
    # Save the file
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
        f"DEFAULT_PROFILE = \"{updated_profile}\"",
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

def save_experiment(exp_id, objective, prompt, search_py, metrics, status, description):
    """
    Save experiment record to experiments/log.jsonl.
    Pretty-printed, separated by blank lines.
    Cross-referenceable with results.tsv via exp_id.
    """
    os.makedirs("experiments", exist_ok=True)
    prompt_hash = save_prompt(prompt) if prompt else "none"
    record = {
        "exp_id":      exp_id,
        "objective": objective,
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
                max_tokens=4096,        # FIX 4 — prevent truncated code generation
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
    Parse LLM response into:
        (description, code)

    Expected format:

        # DESCRIPTION: ...
        <full python code>
    """

    raw = raw.strip()

    if not raw:
        raise ValueError("Empty agent response")

    lines = raw.splitlines()

    # ── Primary strict protocol ─────────────────────

    PREFIX = "# DESCRIPTION:"

    first = lines[0].strip()

    if first.startswith(PREFIX):

        description = first[len(PREFIX):].strip()

        code = "\n".join(lines[1:]).strip()

        if not description:
            raise ValueError("Empty description")

        if not code:
            raise ValueError("Missing code block")

        return description, code

    # ── Recovery fallback ───────────────────────────
    # Sometimes model forgets exact prefix but still
    # produces valid code. Recover gracefully.

    code_start = None

    for i, line in enumerate(lines):

        stripped = line.strip()

        if stripped.startswith((
            "import ",
            "from ",
            "def ",
            "class ",
        )):
            code_start = i
            break

    if code_start is not None:

        description = "auto-recovered"

        code = "\n".join(lines[code_start:]).strip()

        return description, code

    raise ValueError(
        "Could not parse agent response"
    )
# FIX — AST-based code feature extraction.
# Replaces hardcoded string matching with deterministic AST parsing.
# Catches anything the agent writes, not just what we anticipated.
# Limitation: variable names may be low-signal (e.g. "Variable: scores").
# Mitigated by only surfacing top-level assignments and function names.
def extract_code_features(search_py: str) -> str:
    """
    Deterministically extract what's in search.py using AST parsing.
    Returns a formatted string for injection into the agent prompt.
    Falls back gracefully if the file has a syntax error.
    """
    try:
        tree = ast.parse(search_py)
    except SyntaxError:
        # Broken search.py — agent wrote invalid code last round.
        # eval will also fail and the run will be rejected, so this
        # is just informational for the prompt.
        return "  (could not parse — search.py has a syntax error)"

    features = []
    seen     = set()

    for node in ast.walk(tree):
        # Helper functions the agent defined beyond the required search() entrypoint
        if isinstance(node, ast.FunctionDef):
            if node.name != "search":
                feat = f"Function: {node.name}()"
                if feat not in seen:
                    seen.add(feat)
                    features.append(feat)

        # All imports — catches any library the agent pulled in
        elif isinstance(node, ast.Import):
            for alias in node.names:
                feat = f"Import: {alias.name}"
                if feat not in seen:
                    seen.add(feat)
                    features.append(feat)

        elif isinstance(node, ast.ImportFrom):
            feat = f"Import: from {node.module} import ..."
            if feat not in seen:
                seen.add(feat)
                features.append(feat)

        # Top-level variable assignments only — skips loop vars and temporaries.
        # isinstance check on col_offset==0 isolates module-level assignments.
        elif (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and getattr(node, "col_offset", 1) == 0
        ):
            feat = f"Variable: {node.targets[0].id}"
            if feat not in seen:
                seen.add(feat)
                features.append(feat)

    # Cap at 20 to avoid prompt bloat on very complex files
    features = features[:20]

    return "\n".join(f"  - {f}" for f in features) if features else "  (none detected)"


def ask_agent(program_md, search_py, history, objective):
    """Ask LLM for next search.py modification. Returns (code, cost, description, prompt)."""

    history_str = "\n".join([format_history_entry(r) for r in history[-10:]])

    tried_descriptions = [
        r['description'] for r in history
        if not is_baseline(r['description'])
    ]
    tried_str = "\n".join(f"  - {d}" for d in tried_descriptions) if tried_descriptions else "  (none yet)"

    # FIX — replaced hardcoded string matching with AST-based extraction
    already_str = extract_code_features(search_py)

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

## EXISTING CODE FEATURES (reference only)

The current implementation already contains:
{already_str}

Guidelines:
- Preserve the existing search() function signature.
- Existing imports may still be required.
- Do NOT remove imports unless you are certain they are unused.
- Avoid duplicate imports.
- Preserve compatibility with the evaluation pipeline.
- Modify only what is necessary to improve the objective.


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

## OUTPUT FORMAT (STRICT)

The FIRST line of your response MUST be EXACTLY in this format:

# DESCRIPTION: short description here

Example:
# DESCRIPTION: add genre weighting

Immediately after that line, output the FULL Python file.

Do NOT include:
- markdown fences
- explanations
- prose
- bullet points
- comments before DESCRIPTION
- partial diffs

Your response MUST begin with:
# DESCRIPTION:
"""

    response          = call_api([{"role": "user", "content": prompt}])
    cost              = get_llm_cost(response)
    raw               = response.choices[0].message.content.strip()
    description, code = parse_agent_response(raw)

    return code, cost, description, prompt

# ─── ARCHITECTURE DOCUMENTATION ─────────────────────────────

# FIX 3 — Live architecture updates on every KEEP.
# No LLM call — just writes a metrics summary so the file is never stale
# even if the run is interrupted. The full LLM-generated doc is written
# at the end of the loop as before.
def update_architecture_file(objective, best_metrics, baseline_metrics, n_kept, n_total, exp_id=None, prompt_hash=None):
    """
    Write a lightweight metrics summary on every KEEP.
    No LLM call — deterministic and fast.
    Overwrites any previous version so the file always reflects current best.
    """
    filename = f"ARCHITECTURE_{objective}.md"
    content  = f"""---
exp_id      : {exp_id or 'unknown'}
prompt_hash : {prompt_hash or 'unknown'}
prompt_file : experiments/prompts/{prompt_hash or 'unknown'}.txt
objective   : {objective}
updated     : {time.strftime("%Y-%m-%dT%H:%M:%S")}
---

# Architecture — {objective}
*Auto-updated on every KEEP. Full LLM-generated doc written at run end.*

## Current best
| Metric | Baseline | Best |
|--------|----------|------|
| recall@10 | {baseline_metrics['recall']:.3f} | {best_metrics['recall']:.3f} |
| latency_ms | {baseline_metrics['latency_ms']:.1f} | {best_metrics['latency_ms']:.1f} |
| llm_cost | ${baseline_metrics['llm_cost_usd']:.6f} | ${best_metrics['llm_cost_usd']:.6f} |

## Progress
- Experiments kept : {n_kept}
- Total so far     : {n_total}

*See experiments/log.jsonl for full history. See search_profiles/{OBJECTIVE_TO_PROFILE.get(objective)}.py for winning implementation.*
"""
    write_file(filename, content)
    print(f"   📄 Architecture updated → {filename}")


def document_architecture(best_metrics, baseline_metrics, objective, n_experiments, history):
    """Generate full LLM-written ARCHITECTURE_<objective>.md at end of run."""
    print("\n📄 Documenting final architecture...")

    search_py = read_file("search.py")
    kept      = [r for r in history if r["status"] == "keep" and not is_baseline(r["description"])]
    discarded = [r for r in history if r["status"] == "discard"]
    crashed   = [r for r in history if r["status"] == "crash"]

    # ── Resolve best experiment's exp_id and prompt_hash ──
    best_kept        = best_from_history(history, objective)
    best_exp_id      = "unknown"
    best_prompt_hash = "unknown"
    if best_kept:
        all_records = load_experiments()
        matched = next((
            r for r in all_records
            if r["description"] == best_kept["description"]
            and r["status"] == "keep"
        ), None)
        if matched:
            best_exp_id      = matched["exp_id"]
            best_prompt_hash = matched.get("prompt_hash", "unknown")

    kept = [r for r in history if r["status"] == "keep" 
                and not is_baseline(r["description"])]
    
    if not kept:
        # honest null result
        prompt = f"""---
            objective  : {objective}
            generated  : {time.strftime("%Y-%m-%dT%H:%M:%S")}
            result     : NO IMPROVEMENT FOUND
            ---

            # No improvements kept for objective: {objective}

            {len(history)} experiments ran. None cleared all constraints.
            Baseline recall: {baseline_metrics['recall']:.3f}
            Baseline latency: {baseline_metrics['latency_ms']:.1f}ms

            See experiments/log.jsonl for full attempt history.
            """
        write_file(f"ARCHITECTURE_{objective}.md", prompt)
        return

    response = call_api([{"role": "user", "content": prompt}])
    content  = response.choices[0].message.content.strip()

    filename = f"ARCHITECTURE_{objective}.md"

    # ── Prepend reference frontmatter before LLM content ──
    header = f"""---
exp_id      : {best_exp_id}
prompt_hash : {best_prompt_hash}
prompt_file : experiments/prompts/{best_prompt_hash}.txt
objective   : {objective}
generated   : {time.strftime("%Y-%m-%dT%H:%M:%S")}
---

"""
    write_file(filename, header + content)

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
                "objective": r.get("objective"),
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

    # FIX 1 — Seed search.py from best prior profile for this objective.
    # Without this, every new run restarts from whatever search.py happens
    # to be on disk, which may be the original baseline rather than the
    # 0.900 recall implementation found in a previous run.
    best_prior = best_from_history(history, objective)
    if best_prior:
        profile_name = OBJECTIVE_TO_PROFILE.get(objective)
        profile_path = f"search_profiles/{profile_name}.py"
        if os.path.exists(profile_path):
            shutil.copy(profile_path, "search.py")
            print(f"   📂 Seeded search.py from {profile_path}")
        else:
            print(f"   ⚠️  Profile {profile_path} not found — using current search.py")
    else:
        # FIX 2 — Cross-objective seeding.
        # If no prior experiments exist for this objective, seed from a
        # related objective's best profile instead of starting cold.
        # e.g. pareto run seeds from high_recall so it refines a strong
        # baseline rather than rediscovering recall improvements from scratch.
        seed_profile = OBJECTIVE_SEED_FROM.get(objective)
        if seed_profile:
            seed_path = f"search_profiles/{seed_profile}.py"
            if os.path.exists(seed_path):
                shutil.copy(seed_path, "search.py")
                print(f"   📂 Cross-objective seed: {seed_path} → search.py")
            else:
                print(f"   ⚠️  Seed profile {seed_path} not found — using current search.py")

    print("\n📦 Preloading resources...")
    global CACHED_RESOURCES
    CACHED_RESOURCES = load_resources()
    
    # ── Baseline — measure what's on disk right now ──
    print("\n📊 Running baseline...")
    baseline_recall, baseline_latency = run_eval()
    baseline_metrics = {
    "recall":       baseline_recall,
    "latency_ms":   baseline_latency,
    "llm_cost_usd": float("inf")
    }
    print(f"recall@10    : {baseline_recall:.6f}")
    print(f"latency      : {baseline_latency:.1f}ms")

    # ── Resume best_metrics from history or use baseline ──
    if best_prior:
        best_metrics = best_prior["metrics"].copy()

        # Latency is not stable across runs/machines.
        # Re-anchor latency to current-session baseline.
        best_metrics["latency_ms"] = baseline_latency
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
        save_experiment("exp_000", objective,  "", read_file("search.py"),
                        baseline_metrics, "keep", "baseline")
        history.append({
            "description": "baseline",
            "metrics":     baseline_metrics,
            "status":      "keep",
            "objective":   objective,
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

            # ── Automatic crash guidance ─────────────────────

            import_related_errors = (
                "NameError",
                "ImportError",
                "ModuleNotFoundError",
            )

            if any(err in crash_reason for err in import_related_errors):

                crash_reason += """

        AUTOMATIC GUIDANCE:
        The previous experiment removed or failed to preserve
        a required dependency/import.

        Ensure:
        - all referenced symbols are imported
        - required imports are preserved
        - existing dependencies remain available
        - the search() signature remains unchanged
        """

            print(f"💥 Crash: {e}")

            # ── Safe restore after crash ─────────────────────

            try:

                git_restore()

            except Exception as restore_error:

                print("\n🚨 CRITICAL RESTORE FAILURE")
                print(str(restore_error))

                raise RuntimeError(
                    "Experiment recovery failed. "
                    "Aborting to prevent optimizer corruption."
                )

            crash_metrics = {
                "recall":       0.0,
                "latency_ms":   float("inf"),
                "llm_cost_usd": llm_cost,
            }

            history.append({
                "exp_id":      exp_id,
                "description": description,
                "metrics":     crash_metrics,
                "status":      "crash",
                "traceback":   crash_reason,
                "objective":   objective,
            })

            save_experiment(
                exp_id,
                objective,
                prompt,
                new_code,
                crash_metrics,
                "crash",
                description,
            )

            log_result(
                exp_id,
                "crash",
                crash_metrics,
                "crash",
                description,
            )

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
            save_experiment(exp_id, objective, prompt, new_code, new_metrics, "keep", description)
            print(f"✅ KEEP — improved on objective '{objective}'")

            # ── Auto-update search profile ──
            update_profile(objective, new_metrics, description)

            best_metrics = new_metrics.copy()
            history.append({
                "description": description,
                "metrics":     new_metrics,
                "status":      "keep",
                "objective":   objective,
            })

            # FIX 3 — Write live architecture summary on every KEEP.
            # No LLM call — just metrics. Survives interrupted runs.
            saved_prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
            n_kept = len([r for r in history if r["status"] == "keep"])
            update_architecture_file(
                objective, best_metrics, baseline_metrics,
                n_kept, i + 1,
                exp_id=exp_id,
                prompt_hash=saved_prompt_hash
            )

        else:
            git_restore()
            log_result(exp_id, "discarded", new_metrics, "discard", description)
            save_experiment(exp_id, objective, prompt, new_code, new_metrics, "discard", description)
            print(f"❌ DISCARD — no improvement on '{objective}'")
            history.append({
                "description": description,
                "metrics":     new_metrics,
                "status":      "discard",
                "objective":   objective,
            })

    # ── Document final architecture (full LLM-generated version) ──
    document_architecture(best_metrics, baseline_metrics, objective, n_experiments, history)

    # ── Summary ──
    cost_value = best_metrics["llm_cost_usd"]

    cost_str = (
        f"${cost_value:.6f}"
        if cost_value != float("inf")
        else "n/a"
    )
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