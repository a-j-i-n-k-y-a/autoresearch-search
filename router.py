# router.py
#
# Routes queries to the right search implementation based on a constraint.
# Profiles are managed automatically by agent_loop.py.
#
# Usage:
#   python router.py "dark psychological thriller"
#   python router.py "inception" --constraint low_latency
#   python router.py --profiles
#   python router.py --demo
#
# From code:
#   from router import route
#   response = route("robots and consciousness", constraint="high_recall")

import time
import importlib
import argparse
import os
import sys

# ─── LOAD RESOURCES ONCE ────────────────────────────────────
# Shared across all profiles — one-time cost at startup
from prepare import load_resources
print("Loading search resources...")
df, bm25, model, index = load_resources()
print("Ready.\n")

# ─── CONSTRAINT INFERENCE ───────────────────────────────────
def infer_constraint(query):
    """
    Infer the best constraint from query characteristics.
    Used when no explicit constraint is passed.
    """
    word_count = len(query.strip().split())

    if word_count <= 2:
        # Short query — ambiguous, maximize recall
        return "high_recall"
    elif word_count >= 6:
        # Long descriptive query — semantic, maximize recall
        return "high_recall"
    else:
        return "balanced"

# ─── LOAD REGISTRY ──────────────────────────────────────────
def load_registry():
    """
    Load PROFILES from search_profiles/registry.py.
    Returns empty dict with helpful message if registry not found.
    """
    registry_path = "search_profiles/registry.py"
    if not os.path.exists(registry_path):
        print(
            "⚠️  No registry found at search_profiles/registry.py\n"
            "   Run experiments first:\n"
            "     python agent_loop.py --n 20 --objective recall\n"
            "   Or manually export a profile:\n"
            "     python agent_loop.py --export-profile recall"
        )
        return {}, "balanced"

    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("registry", registry_path)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return getattr(mod, "PROFILES", {}), getattr(mod, "DEFAULT_PROFILE", "balanced")
    except Exception as e:
        print(f"⚠️  Failed to load registry: {e}")
        return {}, "balanced"

# ─── ROUTER ─────────────────────────────────────────────────
def route(query, constraint=None, verbose=True):
    """
    Route a query to the right search implementation.

    Args:
        query      : the search query string
        constraint : "high_recall" | "low_latency" | "balanced" | "low_cost" | None
                     If None, constraint is inferred from the query.
        verbose    : print routing decision and results

    Returns:
        dict with query, constraint, description, metrics, actual_latency_ms, results
        None if profile module not found
    """
    profiles, default_profile = load_registry()

    # Infer constraint if not provided
    if constraint is None:
        constraint = infer_constraint(query)
        if verbose:
            print(f"Constraint inferred : {constraint}")

    # Validate and fallback
    # →
    if constraint not in profiles:
        # Try default, then just pick the first available profile
        if default_profile in profiles:
            if verbose:
                print(f"⚠️  '{constraint}' not found — falling back to '{default_profile}'")
            constraint = default_profile
        elif profiles:
            fallback = list(profiles.keys())[0]
            if verbose:
                print(f"⚠️  '{constraint}' not found — falling back to '{fallback}'")
            constraint = fallback
        else:
            print("⚠️  No profiles available. Run: python agent_loop.py --export-profile recall")
            return None

    profile = profiles[constraint]

    # Load the search module for this profile
    module_path = profile.get("module", f"search_profiles.{constraint}")
    try:
        # Invalidate cached module so we always get the latest file
        if module_path in sys.modules:
            del sys.modules[module_path]
        search_module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        profile_file = module_path.replace(".", "/") + ".py"
        print(
            f"⚠️  Profile module not found: {profile_file}\n"
            f"   Run: python agent_loop.py --export-profile "
            f"{constraint.replace('high_', '').replace('low_', '').replace('_', '')}"
        )
        return None

    # Run search and measure actual latency
    start   = time.time()
    results = search_module.search(query, df, bm25, model, index, top_k=10)
    elapsed = round((time.time() - start) * 1000, 1)

    response = {
        "query":             query,
        "constraint":        constraint,
        "description":       profile.get("description", ""),
        "use_when":          profile.get("use_when", ""),
        "recall_est":        profile.get("recall", 0.0),
        "latency_ms_est":    profile.get("latency_ms", 0.0),
        "actual_latency_ms": elapsed,
        "results":           results,
    }

    if verbose:
        _print_response(response)

    return response

# ─── DISPLAY ────────────────────────────────────────────────
def _print_response(response):
    print(f"Query        : {response['query']}")
    print(f"Constraint   : {response['constraint']}  —  {response['description']}")
    print(f"Latency      : {response['actual_latency_ms']}ms  (expected ~{response['latency_ms_est']}ms)")
    print(f"Recall est.  : {response['recall_est']:.0%}")
    print(f"Top results  :")
    for i, r in enumerate(response["results"], 1):
        print(f"  {i:2}. {r['title']}")
    print()

def list_profiles():
    """Print all available profiles and their metrics."""
    profiles, default = load_registry()
    if not profiles:
        return

    print(f"\n{'Profile':<15} {'Recall':<10} {'Latency':<12} Use when")
    print("─" * 70)
    for name, p in profiles.items():
        marker = "  ← default" if name == default else ""
        print(
            f"{name:<15} "
            f"{p.get('recall', 0):<10.0%} "
            f"{str(round(p.get('latency_ms', 0), 1)) + 'ms':<12} "
            f"{p.get('use_when', '')}{marker}"
        )
    print()

# ─── ENTRYPOINT ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie search router")
    parser.add_argument("query",
                        nargs="?",
                        help="Search query")
    parser.add_argument("--constraint",
                        default=None,
                        help="Profile: high_recall | low_latency | balanced | low_cost")
    parser.add_argument("--profiles",
                        action="store_true",
                        help="List all available profiles and exit")
    parser.add_argument("--demo",
                        action="store_true",
                        help="Run demo queries across all profiles")
    args = parser.parse_args()

    if args.profiles:
        list_profiles()

    elif args.demo:
        print("=" * 60)
        print("ROUTER DEMO")
        print("=" * 60)
        list_profiles()

        demo_queries = [
            ("inception",                                      "low_latency"),
            ("romantic movie with a twist ending",             "balanced"),
            ("psychological thriller unreliable narrator",     "high_recall"),
            ("robot artificial intelligence consciousness",    "high_recall"),
            ("astronaut stranded alone in space",              None),
        ]

        for query, constraint in demo_queries:
            print("-" * 60)
            route(query, constraint=constraint)

    elif args.query:
        route(args.query, constraint=args.constraint)

    else:
        parser.print_help()