# agent_loop.py
import re
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
from prepare import load_resources, evaluate, evaluate_metrics, evaluate_by_slice, per_query_results, regression_report, run_full_eval, _run_retrieval_pass, BENCHMARK_QUERIES, TOP_K
import filecmp
import random

# at the top of the file, after imports
RECALL_FLOOR = 0.50
MAX_RECALL_DROP_RATIO = 0.95
MAX_LATENCY_INCREASE = 1.10
MAX_COST_INCREASE = 2.00
MIN_LATENCY_IMPROVEMENT = 0.05
MAX_PARETO_RECALL_DROP = 0.02 
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
def is_improvement(new, best, objective, recall_noise=0.0):
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
    checks = {
        "recall_floor": new["recall"] >= RECALL_FLOOR,
        "recall_drop": new["recall"] >= best["recall"] * MAX_RECALL_DROP_RATIO,
        "latency_improve": (best["latency_ms"] - new["latency_ms"]) > MIN_LATENCY_IMPROVEMENT,
        "mrr_ok": new.get("mrr", 0) >= best.get("mrr", 0) * 0.97,
        "top1_ok": new.get("top1", 0) >= best.get("top1", 0) * 0.95,
        "precision_ok": new.get("precision", 0) >= best.get("precision", 0) * 0.95,
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
    # AFTER
    if objective == "recall":
        checks["recall_ok"]      = recall_ok()   # floor + drop gate, same as all other objectives
        DEV_QUERIES              = [q for q in BENCHMARK_QUERIES if q.get("split") == "dev"]
        min_improvement          = max(recall_noise * 0.5, 1 / len(DEV_QUERIES))
        checks["recall_improve"] = (new["recall"] - best["recall"]) > min_improvement
        passed = (
            checks["recall_ok"]
            and checks["recall_improve"]
            and checks["mrr_ok"]
            and checks["top1_ok"]
            and checks["precision_ok"]
        )
        return passed, checks

    # ── Latency objective ─────────────────────────────
    elif objective == "latency":
        checks["recall_ok"]       = recall_ok()
        checks["latency_improve"] = (best["latency_ms"] - new["latency_ms"]) > MIN_LATENCY_IMPROVEMENT
        passed = (
            checks["recall_ok"]
            and checks["mrr_ok"]
            and checks["top1_ok"]
            and checks["precision_ok"]
            and checks["latency_improve"]
        )

        return passed, checks

    # ── Cost objective ────────────────────────────────
    elif objective == "cost":
        checks["recall_ok"]      = recall_ok()
        checks["latency_ok"]     = new["latency_ms"] <= best["latency_ms"] * MAX_LATENCY_INCREASE
        checks["cost_improve"]   = new["llm_cost_usd"] < best["llm_cost_usd"]
        passed = (
            checks["recall_ok"]
            and checks["mrr_ok"]
            and checks["top1_ok"]
            and checks["precision_ok"]
            and checks["latency_ok"]
            and checks["cost_improve"]
        )

        return passed, checks

    # ── Pareto objective ──────────────────────────────
    elif objective == "pareto":
        checks["recall_drop_ok"]  = new["recall"] >= best["recall"] - MAX_PARETO_RECALL_DROP
        checks["latency_ok"]      = new["latency_ms"] <= best["latency_ms"] * MAX_LATENCY_INCREASE
        checks["cost_ok"]         = (
            best["llm_cost_usd"] == float("inf") or
            new["llm_cost_usd"] <= best["llm_cost_usd"] * MAX_COST_INCREASE
        )
        checks["any_improve"]     = (
            new["recall"] > best["recall"] or
            new["latency_ms"] < best["latency_ms"] - 0.5 or
            new["llm_cost_usd"] < best["llm_cost_usd"]
        )
        passed = all([
            checks["recall_drop_ok"],
            checks["latency_ok"],
            checks["cost_ok"],
            checks["any_improve"],
            checks["mrr_ok"],
            checks["top1_ok"],
            checks["precision_ok"],
        ])
        return passed, checks

    return False, checks
        

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
            if r["metrics"]["recall"] >= RECALL_FLOOR
        ]
        if not feasible:
            return None
        return min(feasible, key=lambda r: r["metrics"]["latency_ms"])
    elif objective == "cost":
        feasible = [
            r for r in kept
            if r["metrics"]["recall"] >= RECALL_FLOOR #intentional redundancy
        ]
        if not feasible:
            return None
        return min(feasible, key=lambda r: r["metrics"]["llm_cost_usd"])

    # we can also extract true pareto history - midway recall, miway latency. currently doing
    # with recall max first in the history and reducing latency lateron
    elif objective == "pareto":
        # True Pareto seed: best combined recall + latency tradeoff.
        # Normalize both axes so neither dominates by scale.
        max_recall  = max(r["metrics"]["recall"]     for r in kept)
        min_latency = min(r["metrics"]["latency_ms"] for r in kept)
        def pareto_score(r):
            norm_recall  = r["metrics"]["recall"]     / max_recall  if max_recall  else 0
            norm_latency = min_latency / r["metrics"]["latency_ms"] if r["metrics"]["latency_ms"] else 0
            return norm_recall + norm_latency
        return max(kept, key=pareto_score)
    return None

# ─── EVAL ───────────────────────────────────────────────────


def run_eval():
    """
    Run benchmark on current search.py.

    Returns:
        (recall, latency_ms, slice_results, full_metrics)

        recall       — recall@10 float (the gate metric used by is_improvement)
        latency_ms   — median per-query wall-clock latency across N_TRIALS
        slice_results — {slice_name: recall} from evaluate_by_slice()
        full_metrics  — {recall, mrr, ndcg, precision, top1} from evaluate_metrics()

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
    run_full_eval(search_module.search, df, bm25, model, index, split='dev')  # warmup (discard)

    # ── Randomized sub-sample eval (gate metric) ───────────
    # 75% of dev queries randomly sampled each run — different subset
    # every experiment so improvements must generalize across queries,
    # not exploit fixed query distribution.
    # Warmup above uses full set intentionally (no sample_size).
    DEV_QUERIES = [q for q in BENCHMARK_QUERIES if q.get("split") == "dev"]
    SAMPLE_SIZE = max(1, int(len(DEV_QUERIES) * 0.75))  # ~11 of 15

    result = run_full_eval(
        search_module.search, df, bm25, model, index,
        split='dev', sample_size=SAMPLE_SIZE, seed=None  # seed=None → different each run
    )
    recall        = result["recall"]
    full_metrics  = result["full_metrics"]
    slice_results = result["slices"]
    per_query     = result["per_query"]

    print(f"  mrr@10={full_metrics['mrr']:.3f}  ndcg@10={full_metrics['ndcg']:.3f}"
          f"  precision@10={full_metrics['precision']:.3f}  top1={full_metrics['top1']:.3f}")
    for slice_name, slice_recall in sorted(slice_results.items()):
        print(f"  {slice_name:<15}: {slice_recall:.3f}")

    # ── Per-query timing for proper percentiles ────────────
    # Time each query individually rather than timing the full evaluate()
    # call and dividing — that only gives an average, not a distribution.
    # 5 trials × 15 queries = 75 data points for stable percentiles.
    per_query_times = []
    DEV_QUERIES     = [q for q in BENCHMARK_QUERIES if q.get("split") == "dev"]
    N_TRIALS        = 5

    for _ in range(N_TRIALS):
        for item in DEV_QUERIES:
            start      = time.perf_counter()
            search_module.search(item["query"], df, bm25, model, index, top_k=TOP_K)
            elapsed_ms = (time.perf_counter() - start) * 1000
            per_query_times.append(elapsed_ms)

    per_query_times.sort()
    n          = len(per_query_times)
    latency_ms = statistics.median(per_query_times)
    latency_p95 = per_query_times[min(int(n * 0.95), n - 1)]
    latency_p99 = per_query_times[min(int(n * 0.99), n - 1)]

    # ── Bootstrap confidence interval on recall ────────────
    # With 15 dev queries, 1 query flip = 0.067 recall change.
    # Bootstrap resampling estimates the noise floor so is_improvement()
    # can require improvements that exceed it.
    # AFTER — pull from result already in hand, no extra retrieval pass
    per_query_recalls = [v["recall"] for v in per_query.values()]
    bootstrap_means = sorted(
        statistics.mean(random.choices(per_query_recalls, k=len(per_query_recalls)))
        for _ in range(1000)
    )
    recall_ci_low  = round(bootstrap_means[24],  6)   # 2.5th percentile
    recall_ci_high = round(bootstrap_means[974], 6)   # 97.5th percentile
    recall_noise   = round(recall_ci_high - recall_ci_low, 6)

    print(f"  recall CI   : [{recall_ci_low:.3f}, {recall_ci_high:.3f}]  noise={recall_noise:.3f}")
    print(f"  lat p50/p95/p99: {latency_ms:.1f}/{latency_p95:.1f}/{latency_p99:.1f}ms")

    return recall, latency_ms, slice_results, full_metrics, per_query, latency_p95, latency_p99, recall_noise

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
    
    # AFTER — extract only the short hash from "[branch abc1234] message" format
    first_line = commit_result.stdout.strip().split('\n')[0]
    match = re.search(r'\[.*?\s+([a-f0-9]+)\]', first_line)
    return match.group(1) if match else first_line

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
    duplicate_profile = None

    for other_profile in OBJECTIVE_TO_PROFILE.values():

        other_path = f"search_profiles/{other_profile}.py"

        if (
            other_profile != profile_name
            and files_are_identical("search.py", other_path)
        ):
            duplicate_profile = other_profile
            break


    if duplicate_profile:

        print(
            f"   ⚠️ Profile identical to {duplicate_profile}.py "
            f"— skipping file copy but updating registry"
        )

    else:

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
        f"DEFAULT_PROFILE = 'balanced'",
        "",
    ]

    write_file(registry_path, "\n".join(lines))
    print(f"   📋 Registry updated → {registry_path}")

# ─── LOGGING ────────────────────────────────────────────────
# AFTER
RESULTS_HEADER = "exp_id\tcommit\trecall\tlatency_ms\tlatency_p95\tlatency_p99\tllm_cost_usd\tstatus\tdescription\n"

def audit_results_tsv():
    """
    Scan results.tsv for kept experiments that violate RECALL_FLOOR.
    Rewrites their status to 'kept-subfloor' so they're visible but
    clearly marked as invalid under the current floor.
    Does not delete rows — preserves full audit trail.
    """
    if not os.path.exists("results.tsv"):
        return

    with open("results.tsv", "r") as f:
        lines = f.readlines()

    if not lines:
        return

    header   = lines[0]
    rows     = lines[1:]
    cols     = header.strip().split("\t")
    patched  = 0
    new_rows = []

    for row in rows:
        parts = row.strip().split("\t")
        if len(parts) != len(cols):
            new_rows.append(row)
            continue

        record = dict(zip(cols, parts))
        try:
            recall = float(record.get("recall", 1.0))
            status = record.get("status", "")
        except ValueError:
            new_rows.append(row)
            continue

        if status == "keep" and recall < RECALL_FLOOR:
            record["status"] = "kept-subfloor"
            new_rows.append("\t".join(record[c] for c in cols) + "\n")
            patched += 1
        else:
            new_rows.append(row)

    if patched:
        with open("results.tsv", "w") as f:
            f.writelines([header] + new_rows)
        print(f"   ⚠️  audit_results_tsv: marked {patched} subfloor kept rows as 'kept-subfloor'")


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
    if (first_line.startswith("commit\t") or not first_line.startswith("exp_id\t") or "latency_p95" not in first_line):
        with open("results.tsv", "r") as f:
            content = f.read()
        old_lines = content.split("\n")
        new_lines = [RESULTS_HEADER.strip()] + old_lines[1:]
        with open("results.tsv", "w") as f:
            f.write("\n".join(new_lines))
        print("   ⚠️  Migrated results.tsv header to include exp_id column")

def log_result(exp_id, commit, metrics, status, description):
    ensure_results_header()
    # p95/p99 absent on baseline and crash records — default to empty string
    p95 = f"{metrics['latency_p95']:.1f}" if metrics.get("latency_p95") is not None else ""
    p99 = f"{metrics['latency_p99']:.1f}" if metrics.get("latency_p99") is not None else ""
    with open("results.tsv", "a") as f:
        f.write(
            f"{exp_id}\t"
            f"{commit}\t"
            f"{metrics['recall']:.6f}\t"
            f"{metrics['latency_ms']:.1f}\t"
            f"{p95}\t"
            f"{p99}\t"
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

def save_experiment(exp_id, objective, prompt, search_py, metrics, status, description, constraint_trace=None, slice_results=None, full_metrics=None):
    """
    Save experiment record to experiments/log.jsonl.
    Pretty-printed, separated by blank lines.
    Cross-referenceable with results.tsv via exp_id.
    """
    os.makedirs("experiments", exist_ok=True)
    prompt_hash = save_prompt(prompt) if prompt else "none"
    record = {
        "exp_id":           exp_id,
        "objective":        objective,
        "timestamp":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "prompt_hash":      prompt_hash,
        "search_py":        search_py,
        "metrics":          metrics,
        "status":           status,
        "description":      description,
        "constraint_trace": constraint_trace or {},
        "slice_results":    slice_results or {},
        "full_metrics":     full_metrics or {},  # MRR, nDCG, precision, top1
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
        recall, latency, _, _, _, _, _, _ = run_eval()
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
# AFTER
def format_history_entry(r):
    m = r["metrics"]
    # Secondary metrics may be absent in older history records — default to None
    mrr       = m.get("mrr")
    top1      = m.get("top1")
    precision = m.get("precision")

    secondary = ""
    if any(v is not None for v in [mrr, top1, precision]):
        secondary = (
            f"  mrr={mrr:.3f}" if mrr is not None else "  mrr=n/a"
        ) + (
            f"  top1={top1:.3f}" if top1 is not None else "  top1=n/a"
        ) + (
            f"  precision={precision:.3f}" if precision is not None else "  precision=n/a"
        )

    line = (
        f"- {r['description']}: "
        f"recall={m['recall']:.3f}  "
        f"latency={m['latency_ms']:.1f}ms  "
        f"cost=${m['llm_cost_usd']:.6f}"
        f"{secondary}  "
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

def format_query_diagnostics(per_query):
    """
    Format per-query results into a prompt-friendly failure report.
    Shows only queries that missed at least one expected title,
    sorted worst-first by recall. Capped at 8 to avoid prompt bloat.
    """
    if not per_query:
        return "  (no diagnostics available)"

    failures = [
        (query, data)
        for query, data in per_query.items()
        if data["recall"] < 1.0
    ]
    failures.sort(key=lambda x: x[1]["recall"])
    failures = failures[:8]

    if not failures:
        return "  ✅ All queries fully recalled — no failures to show"

    lines = []
    for query, data in failures:
        retrieved = ", ".join(data["retrieved_titles"][:5])
        lines.append(
            f"  Query   : {query}\n"
            f"  Recall  : {data['recall']:.2f}  "
            f"MRR: {data['mrr']:.2f}  "
            f"Top1: {data['top1']:.0f}  "
            f"First hit: rank {data['rank_of_first_hit'] or 'none'}\n"
            f"  Got     : [{retrieved}]\n"
        )
    return "\n".join(lines)

def ask_agent(program_md, search_py, history, objective, per_query_diagnostics=None):
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
        "latency": (
                    "Reduce latency WITHOUT collapsing retrieval quality.\n\n"

                    "IMPORTANT:\n"
                    "- Do NOT remove BM25 entirely\n"
                    "- Do NOT use pure FAISS-only retrieval\n"
                    "- Do NOT sacrifice recall for speed\n\n"

                    "Prefer REAL latency optimizations such as:\n"
                    "- reducing rerank candidate pools\n"
                    "- candidate pruning\n"
                    "- simplifying weighting formulas\n"
                    "- avoiding redundant encode() calls\n"
                    "- reducing dataframe operations\n"
                    "- caching intermediate computations\n"
                    "- early stopping\n"
                    "- lightweight reranking\n"
                    "- vector prefilter + BM25 rerank\n\n"

                    "Recall must stay above 0.5."
                ),
        "cost":    (
                    "Reduce runtime retrieval complexity WITHOUT reducing retrieval quality.\n\n"

                    "Do NOT:\n"
                    "- remove BM25 entirely\n"
                    "- collapse to pure FAISS retrieval\n"
                    "- aggressively shrink candidate pools\n"
                    "- simplify retrieval in ways that destroy recall\n\n"

                    "Prefer SMALL structural optimizations such as:\n"
                    "- reducing rerank pool sizes moderately\n"
                    "- avoiding redundant encode() calls\n"
                    "- simplifying weighting formulas\n"
                    "- reducing dataframe operations\n"
                    "- caching intermediate computations\n"
                    "- pruning unnecessary reranking stages\n"
                    "- lightweight candidate filtering\n"
                    "- removing duplicated work\n\n"

                    "Recall must stay above 0.5."
                ),
        "pareto":  "Improve recall or reduce latency. Recall must not drop. Latency must not increase more than 10%.",
    }[objective]

    # Add this variable before the prompt f-string
    diagnostics_str = format_query_diagnostics(per_query_diagnostics) 

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

## AVOID FAILED LATENCY COLLAPSES

Do NOT repeat:
- pure FAISS-only retrieval
- removing BM25 entirely
- minimalist vector-only search
- ultra-small candidate pools that collapse recall
- retrieval simplification that destroys semantic coverage

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

## CURRENT FAILURE ANALYSIS — queries where recall < 1.0 (worst first):
{diagnostics_str}
Use this to reason about WHY retrieval is failing, not just THAT it is failing.
Ask yourself:
- Are failures clustered in a specific slice (long_tail, ambiguous)?
- Does the retrieved list suggest a vocabulary mismatch (BM25 failing)?
- Does rank of first hit suggest reranking is the problem, not retrieval?
- Would genre/vote signals push the right result higher?

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
| optimization_cost_usd | {"n/a" if baseline_metrics['llm_cost_usd'] == float("inf") else f"${baseline_metrics['llm_cost_usd']:.6f}"} | {"n/a" if best_metrics['llm_cost_usd'] == float("inf") else f"${best_metrics['llm_cost_usd']:.6f}"} |

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

    profile_name = OBJECTIVE_TO_PROFILE.get(objective)
    profile_path = f"search_profiles/{profile_name}.py"

    if os.path.exists(profile_path):
        search_py = read_file(profile_path)
    else:
        search_py = read_file("search.py")

    kept      = [r for r in history if r["status"] == "keep" and not is_baseline(r["description"])]
    discarded = [r for r in history if r["status"] == "discard"]
    crashed   = [r for r in history if r["status"] == "crash"]

    # ── Honest null result — no fabricated architecture if nothing improved ──
    if not kept:
        content = f"""---
objective  : {objective}
generated  : {time.strftime("%Y-%m-%dT%H:%M:%S")}
result     : NO IMPROVEMENT FOUND
---

# No improvements kept for objective: {objective}

{len(history)} experiments ran. None cleared all constraints.
Baseline recall : {baseline_metrics['recall']:.3f}
Baseline latency: {baseline_metrics['latency_ms']:.1f}ms

See experiments/log.jsonl for full attempt history.
"""
        write_file(f"ARCHITECTURE_{objective}.md", content)
        subprocess.run(["git", "add", f"ARCHITECTURE_{objective}.md"])
        subprocess.run(["git", "commit", "-m", f"docs: no improvement found for objective={objective}"])
        print(f"   No improvements kept — honest null result written.")
        return

    # ── Resolve best experiment's exp_id and prompt_hash ──
    best_kept = next(
        (
            r for r in reversed(history)
            if r["status"] == "keep"
            and r.get("objective") == objective
            and not is_baseline(r["description"])
        ),
        None
    )
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

    # ── Build LLM prompt ──
    prompt = f"""
You are documenting the final architecture of an autonomously optimised movie search system.

## Winning experiment reference:
- exp_id      : {best_exp_id}
- prompt_hash : {best_prompt_hash}
- Prompt file : experiments/prompts/{best_prompt_hash}.txt

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
| optimization_cost_usd | {"n/a" if baseline_metrics['llm_cost_usd'] == float("inf") else f"${baseline_metrics['llm_cost_usd']:.6f}"} | {"n/a" if best_metrics['llm_cost_usd'] == float("inf") else f"${best_metrics['llm_cost_usd']:.6f}"} |

## How to run
STRICT RULES — violations undermine scientific integrity:
- Every claim about WHY something works MUST cite a metric delta.
  Good: "Adding genre weighting improved recall 0.71 → 0.79 (+0.08)"
  Bad:  "Genre weighting helped the model understand context better"
- Do NOT explain improvements that are smaller than measurement noise.
- Do NOT attribute causality beyond what the metrics show.
- If an experiment's benefit is unclear from the numbers, say so explicitly.
- Prefer "we observed X" over "this works because Y" unless Y is directly evidenced.
Return only the markdown. No preamble.
"""

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

    global CACHED_RESOURCES
    program_md = read_file("program.md")

    # ── Load full history from all previous runs ──
    history      = []
    kept_any = False
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
    audit_results_tsv() 

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
    CACHED_RESOURCES = load_resources()
    
    # ── Baseline — measure what's on disk right now ──
    print("\n📊 Running baseline...")
    baseline_recall, baseline_latency, baseline_slices, baseline_full_metrics, baseline_per_query, baseline_p95, baseline_p99, _ = run_eval()
    current_slices = baseline_slices 
    current_per_query = baseline_per_query 
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
        save_experiment(
            "exp_000", objective, "", read_file("search.py"),
            baseline_metrics, "keep", "baseline",
            constraint_trace={},  # no gate was applied, explicitly empty
            full_metrics=baseline_full_metrics,
        )
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
        _cost_so_far = best_metrics["llm_cost_usd"]
        _cost_so_far_str = f"${_cost_so_far:.6f}" if _cost_so_far != float("inf") else "n/a"
        print(
            f"Best so far  → recall={best_metrics['recall']:.3f}  "
            f"latency={best_metrics['latency_ms']:.1f}ms  "
            f"cost={_cost_so_far_str}"
        )
        print("=" * 60)

        current_search_py = read_file("search.py")

        # Ask agent
        print("\n🤖 Asking agent...")
        try:
            new_code, llm_cost, description, prompt = ask_agent(
                program_md, current_search_py, history, objective, per_query_diagnostics=current_per_query
            )
        except Exception as e:
            print(f"Agent error: {e}")
            continue

        print(f"Trying       : {description}  (api cost: ${llm_cost:.6f})")

        # prev_query_results already held in current_per_query — no extra retrieval pass
        prev_query_results = current_per_query

        write_file("search.py", new_code)

        # Benchmark
        print("Running benchmark...")
        try:
            new_recall, new_latency, new_slices, new_full_metrics, new_per_query, new_p95, new_p99, recall_noise = run_eval()
                    
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
                constraint_trace={"crashed_before_gate": True}  # explains why trace is absent
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
            "latency_p95":  new_p95,
            "latency_p99":  new_p99,
            "llm_cost_usd": llm_cost,
            "mrr":          new_full_metrics["mrr"],
            "ndcg":         new_full_metrics["ndcg"],
            "precision":    new_full_metrics["precision"],
            "top1":         new_full_metrics["top1"],
        }

        print(f"recall@10    : {new_recall:.6f}")
        print(f"latency      : {new_latency:.1f}ms")
        print(f"llm_cost     : ${llm_cost:.6f}")

        improved, constraint_trace = is_improvement(
            new_metrics,
            best_metrics,
            objective,
            recall_noise=recall_noise
        )

        if improved:
            commit = git_commit(f"experiment: {description}")
            log_result(exp_id, commit, new_metrics, "keep", description)
            save_experiment(exp_id, objective, prompt, new_code, new_metrics, "keep", description,
                constraint_trace=constraint_trace, slice_results=new_slices, full_metrics=new_full_metrics)
            passed = [k for k, v in constraint_trace.items() if v]
            print(f"✅ KEEP — objective='{objective}'  passed={passed}")
            kept_any = True

            # ── Auto-update search profile ──
            best_metrics = new_metrics.copy()
            current_per_query = new_per_query 
            current_slices    = new_slices          
            update_profile(objective, best_metrics, description) 
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
            
            # ── Regression report ──────────────────────────

            report = regression_report(prev_query_results, new_per_query)

            if report["regressions"]:
                print(f"   ⚠️  {len(report['regressions'])} queries regressed:")
                for r in report["regressions"]:
                    print(f"      '{r['query'][:60]}'  {r['before']:.2f} → {r['after']:.2f}  (Δ{r['delta']:.2f})")
            else:
                print(f"   ✅ No regressions — {report['net']} net query improvements")

        else:
            git_restore()
            log_result(exp_id, "discarded", new_metrics, "discard", description)
            save_experiment(exp_id, objective, prompt, new_code, new_metrics, "discard", description,
                constraint_trace=constraint_trace)
            failed = [k for k, v in constraint_trace.items() if not v]
            print(f"❌ DISCARD — objective='{objective}'  failed={failed}")
            history.append({
                "description": description,
                "metrics":     new_metrics,
                "status":      "discard",
                "objective":   objective,
            })
    # Add this just before document_architecture() at line 1363
    # ── Sync registry with ACTUAL promoted best ─────────────

    latest_kept = next(
        (
            r for r in reversed(history)
            if r["status"] == "keep"
            and r.get("objective") == objective
            and not is_baseline(r["description"])
        ),
        None
    )

    # AFTER
    if latest_kept and latest_kept["metrics"]["recall"] >= RECALL_FLOOR:
        update_profile(
            objective,
            latest_kept["metrics"],
            latest_kept["description"]
        )
    elif latest_kept:
        print(f"   ⚠️  Skipping final profile sync — "
            f"latest kept recall {latest_kept['metrics']['recall']:.3f} below floor")

    # ── Document final architecture only if promotion occurred ──

    if kept_any:

        document_architecture(
            best_metrics,
            baseline_metrics,
            objective,
            n_experiments,
            history,
        )

    else:

        arch_path = f"ARCHITECTURE_{objective}.md"

        if os.path.exists(arch_path):

            os.remove(arch_path)

            print(f"\n🗑️ Removed stale {arch_path}")

        print("\n📄 No promoted architecture generated.")

    # ── Summary ──
    final_metrics = (latest_kept["metrics"] if latest_kept else best_metrics )

    cost_value = final_metrics["llm_cost_usd"]

    cost_str = (
        f"${cost_value:.6f}"
        if cost_value != float("inf")
        else "n/a"
    )
    print(f"\n{'=' * 60}")
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Objective    : {objective}")

    profile_name = OBJECTIVE_TO_PROFILE.get(objective)
    profile_path = f"search_profiles/{profile_name}.py"

    if kept_any and os.path.exists(profile_path):

        print(f"Profile      : {profile_path}")

    else:

        print("Profile      : no promoted profile yet")

    print(
        f"Baseline     → recall={baseline_metrics['recall']:.3f}  "
        f"latency={baseline_metrics['latency_ms']:.1f}ms"
    )
    print(
        f"Best         → recall={final_metrics['recall']:.3f}  "
        f"latency={final_metrics['latency_ms']:.1f}ms  "
        f"cost={cost_str}"
    )

    # ── Final slice breakdown ──
    if kept_any:
        print("\nRecall by slice (dev set):")
        for slice_name, slice_recall in sorted(current_slices.items()):
            print(f"  {slice_name:<15}: {slice_recall:.3f}")
    
    print(f"\nFull log     : results.tsv")
    print(f"Replay log   : experiments/log.jsonl")
    print(f"Prompts      : experiments/prompts/")
    if kept_any and os.path.exists(profile_path):
        print(f"Profile      : {profile_path}")
    else:
        print("Profile      : no promoted profile yet")
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
        recall, latency, _, _, _, _, _, _ = run_eval()
        metrics = {"recall": recall, "latency_ms": latency, "llm_cost_usd": 0.0}
        update_profile(args.export_profile, metrics, "manually exported")
        print(f"Exported search.py → search_profiles/{OBJECTIVE_TO_PROFILE[args.export_profile]}.py")

    elif args.eval_only:
        recall, latency, slice_results, full_metrics, _, p95, p99, noise = run_eval()
        print(f"recall@10    : {recall:.6f}  (noise ±{noise:.3f})")
        print(f"latency p50  : {latency:.1f}ms")
        print(f"latency p95  : {p95:.1f}ms")
        print(f"latency p99  : {p99:.1f}ms")
        print(f"mrr@10       : {full_metrics['mrr']:.6f}")
        print(f"ndcg@10      : {full_metrics['ndcg']:.6f}")
        print(f"precision@10 : {full_metrics['precision']:.6f}")
        print(f"top1         : {full_metrics['top1']:.6f}")

    else:
        run_experiment_loop(n_experiments=args.n, objective=args.objective)