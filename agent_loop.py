# agent_loop.py
# Runs one evaluation pass and prints results.
# The LLM agent calls this after each modification to search.py.

import time
import argparse
import importlib
import sys
from prepare import load_resources, evaluate, BENCHMARK_QUERIES

def run_eval():
    print("Loading resources...")
    df, bm25, model, index = load_resources()

    # Import search function fresh each time
    if "search" in sys.modules:
        del sys.modules["search"]
    import search as search_module

    # Measure latency
    start = time.time()
    recall = evaluate(search_module.search, df, bm25, model, index)
    elapsed = (time.time() - start) * 1000 / len(BENCHMARK_QUERIES)

    print("---")
    print(f"recall@10:    {recall:.6f}")
    print(f"latency_ms:   {elapsed:.1f}")

if __name__ == "__main__":
    run_eval()