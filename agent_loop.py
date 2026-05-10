# agent_loop.py
import time
import sys
import os
from dotenv import load_dotenv
import subprocess
from openai import OpenAI
from prepare import load_resources, evaluate, BENCHMARK_QUERIES

#from dotenv import load_dotenv
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "anthropic/claude-3.5-sonnet"

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)
def run_eval():
    """Run benchmark on current search.py and return recall score"""
    df, bm25, model, index = load_resources()
    
    if "search" in sys.modules:
        del sys.modules["search"]
    import search as search_module

    start = time.time()
    recall = evaluate(search_module.search, df, bm25, model, index)
    elapsed = (time.time() - start) * 1000 / len(BENCHMARK_QUERIES)

    return recall, elapsed

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

def git_reset():
    subprocess.run(["git", "reset", "--hard", "HEAD~1"])

def log_result(commit, recall, latency, status, description):
    """Append result to results.tsv"""
    if not os.path.exists("results.tsv"):
        with open("results.tsv", "w") as f:
            f.write("commit\trecall\tlatency_ms\tstatus\tdescription\n")
    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{recall:.6f}\t{latency:.1f}\t{status}\t{description}\n")

def ask_agent(program_md, search_py, recall_history):
    """Ask LLM to suggest next modification to search.py"""
    
    history_str = "\n".join([
        f"- {r['description']}: recall={r['recall']:.3f} ({r['status']})"
        for r in recall_history
    ])

    prompt = f"""
You are an autonomous research agent improving a search system.

## Your instructions (program.md):
{program_md}

## Current search.py:
```python
{search_py}
```

## Experiment history so far:
{history_str if history_str else "No experiments yet — this is the baseline."}

## Your task:
Suggest ONE specific modification to search.py to improve recall@10.
Think step by step about what might work based on the history.

Return ONLY the complete new search.py content — no explanation, 
no markdown code blocks, just the raw Python code.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content.strip()

def extract_description(program_md, search_py, new_search_py):
    """Ask LLM for a short description of what changed"""
    prompt = f"""
Old search.py:
{search_py}

New search.py:
{new_search_py}

In 5 words or less, what did this experiment try?
Reply with ONLY the short description, nothing else.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def run_experiment_loop(n_experiments=20):
    """
    Main autonomous loop:
    1. Get current recall
    2. Ask agent for modification
    3. Apply modification
    4. Measure new recall
    5. Keep if better, discard if worse
    6. Repeat
    """
    
    program_md  = read_file("program.md")
    recall_history = []

    print("="*60)
    print("AUTORESEARCH — Search System Optimizer")
    print(f"Running {n_experiments} experiments")
    print("="*60)

    # Step 1 — Establish baseline
    print("\n📊 Running baseline...")
    baseline_recall, baseline_latency = run_eval()
    print(f"Baseline recall@10: {baseline_recall:.6f}")
    print(f"Baseline latency:   {baseline_latency:.1f}ms")

    commit = git_commit("baseline")
    log_result(commit, baseline_recall, baseline_latency, "keep", "baseline bm25")

    best_recall = baseline_recall
    recall_history.append({
        "description": "baseline bm25",
        "recall": baseline_recall,
        "status": "keep"
    })

    # Step 2 — Experiment loop
    for i in range(n_experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{n_experiments}")
        print(f"Best recall so far: {best_recall:.6f}")
        print("="*60)

        current_search_py = read_file("search.py")

        # Ask agent for modification
        print("\n🤖 Asking agent for next experiment...")
        try:
            new_search_py = ask_agent(program_md, current_search_py, recall_history)
        except Exception as e:
            print(f"Agent error: {e}")
            continue

        # Apply modification
        write_file("search.py", new_search_py)

        # Get description
        description = extract_description(program_md, current_search_py, new_search_py)
        print(f"Trying: {description}")

        # Run benchmark
        print("Running benchmark...")
        try:
            new_recall, new_latency = run_eval()
        except Exception as e:
            print(f"💥 Crash: {e}")
            write_file("search.py", current_search_py)
            log_result("crash", 0.0, 0.0, "crash", description)
            recall_history.append({
                "description": description,
                "recall": 0.0,
                "status": "crash"
            })
            continue

        print(f"recall@10:  {new_recall:.6f}")
        print(f"latency:    {new_latency:.1f}ms")

        # Keep or discard
        if new_recall > best_recall:
            commit = git_commit(f"experiment: {description}")
            log_result(commit, new_recall, new_latency, "keep", description)
            print(f"✅ KEEP — improved from {best_recall:.3f} → {new_recall:.3f}")
            best_recall = new_recall
            recall_history.append({
                "description": description,
                "recall": new_recall,
                "status": "keep"
            })
        else:
            write_file("search.py", current_search_py)
            print(f"❌ DISCARD — no improvement ({new_recall:.3f} <= {best_recall:.3f})")
            log_result("discarded", new_recall, new_latency, "discard", description)
            recall_history.append({
                "description": description,
                "recall": new_recall,
                "status": "discard"
            })

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print("="*60)
    print(f"Baseline recall: {baseline_recall:.6f}")
    print(f"Best recall:     {best_recall:.6f}")
    print(f"Improvement:     +{best_recall - baseline_recall:.6f}")
    print(f"\nFull log: results.tsv")
    print(f"Best search.py is current state of search.py")


if __name__ == "__main__":
    run_experiment_loop(n_experiments=20)