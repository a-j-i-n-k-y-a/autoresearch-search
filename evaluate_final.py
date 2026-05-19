# evaluate_final.py
#
# Run this ONLY after all optimization is complete.
# Evaluates the best search profiles against the private holdout set.
#
# Usage:
#   python evaluate_final.py
#   python evaluate_final.py --profile high_recall
#   python evaluate_final.py --all-profiles

import argparse
import importlib
import os
import sys
from prepare import load_resources, evaluate_holdout, evaluate, BENCHMARK_QUERIES

def run_final_eval(profile_name):
    print(f"\n{'=' * 60}")
    print(f"FINAL HOLDOUT EVALUATION — {profile_name}")
    print("=" * 60)
    print("⚠️  This uses the private holdout set.")
    print("    Do not run this mid-optimization.\n")

    # load resources
    df, bm25, model, index = load_resources()

    # load the profile's search function
    module_path = f"search_profiles.{profile_name}"
    try:
        search_module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        print(f"❌ Profile not found: search_profiles/{profile_name}.py")
        return

    search_fn = search_module.search

    # dev score — what the optimizer saw
    dev_recall = evaluate(search_fn, df, bm25, model, index, split="dev")
    print(f"Dev recall@10     : {dev_recall:.4f}  (what optimizer used)")

    # holdout score — honest eval
    holdout_recall, holdout_slices = evaluate_holdout(search_fn, df, bm25, model, index)
    print(f"Holdout recall@10 : {holdout_recall:.4f}  (never seen by agent)")

    # gap — tells you how much the optimizer overfit
    gap = dev_recall - holdout_recall
    if gap > 0.05:
        print(f"\n⚠️  Gap = {gap:.4f} — possible overfitting to dev set")
    else:
        print(f"\n✅ Gap = {gap:.4f} — generalizes well")

    # per slice breakdown
    print("\nHoldout recall by slice:")
    for slice_name, slice_recall in sorted(holdout_slices.items()):
        print(f"  {slice_name:<15}: {slice_recall:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final holdout evaluation")
    parser.add_argument("--profile", default="high_recall",
                        help="Profile to evaluate: high_recall | low_latency | balanced | low_cost")
    parser.add_argument("--all-profiles", action="store_true",
                        help="Evaluate all available profiles")
    args = parser.parse_args()

    if args.all_profiles:
        profiles = ["high_recall", "low_latency", "balanced", "low_cost"]
        for p in profiles:
            if os.path.exists(f"search_profiles/{p}.py"):
                run_final_eval(p)
            else:
                print(f"⏭️  Skipping {p} — profile file not found")
    else:
        run_final_eval(args.profile)