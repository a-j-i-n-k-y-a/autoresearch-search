# test_is_improvement.py
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agent_loop import is_improvement, RECALL_FLOOR

# ── Shared fixtures ───────────────────────────────────────────

def make_metrics(recall=0.80, latency=100.0, cost=0.01,
                 mrr=0.80, top1=0.80, precision=0.80):
    return {
        "recall":       recall,
        "latency_ms":   latency,
        "llm_cost_usd": cost,
        "mrr":          mrr,
        "top1":         top1,
        "precision":    precision,
    }

BEST = make_metrics(recall=0.75, latency=120.0, cost=0.02)


# ── Recall objective ──────────────────────────────────────────

def test_recall_improvement_accepted():
    new = make_metrics(recall=0.85)
    passed, checks = is_improvement(new, BEST, "recall")
    assert passed, f"Expected KEEP, checks={checks}"

def test_recall_no_improvement_rejected():
    new = make_metrics(recall=0.75)   # equal, not better
    passed, _ = is_improvement(new, BEST, "recall")
    assert not passed

def test_recall_below_floor_rejected():
    new = make_metrics(recall=RECALL_FLOOR - 0.01)
    passed, checks = is_improvement(new, BEST, "recall")
    assert not passed
    assert not checks["recall_ok"], "recall_ok should fail below floor"

def test_recall_drop_too_large_rejected():
    # Improves vs best in absolute terms but drops vs floor ratio
    best_high = make_metrics(recall=0.95)
    new = make_metrics(recall=0.80)   # 0.80 < 0.95 * 0.95 = 0.9025
    passed, checks = is_improvement(new, best_high, "recall")
    assert not passed
    assert not checks["recall_ok"]

def test_recall_mrr_regression_rejected():
    new = make_metrics(recall=0.85, mrr=0.50)   # mrr drops hard
    passed, checks = is_improvement(new, BEST, "recall")
    assert not passed
    assert not checks["mrr_ok"]


# ── Latency objective ─────────────────────────────────────────

def test_latency_improvement_accepted():
    new = make_metrics(recall=0.75, latency=80.0)   # faster, recall held
    passed, checks = is_improvement(new, BEST, "latency")
    assert passed, f"Expected KEEP, checks={checks}"

def test_latency_recall_too_low_rejected():
    new = make_metrics(recall=RECALL_FLOOR - 0.01, latency=80.0)
    passed, checks = is_improvement(new, BEST, "latency")
    assert not passed
    assert not checks["recall_ok"]

def test_latency_no_improvement_rejected():
    new = make_metrics(recall=0.75, latency=120.0)   # same latency
    passed, _ = is_improvement(new, BEST, "latency")
    assert not passed


# ── Cost objective ────────────────────────────────────────────

def test_cost_improvement_accepted():
    new = make_metrics(recall=0.75, cost=0.01)   # cheaper
    passed, checks = is_improvement(new, BEST, "cost")
    assert passed, f"Expected KEEP, checks={checks}"

def test_cost_recall_too_low_rejected():
    new = make_metrics(recall=RECALL_FLOOR - 0.01, cost=0.001)
    passed, checks = is_improvement(new, BEST, "cost")
    assert not passed
    assert not checks["recall_ok"]

def test_cost_no_improvement_rejected():
    new = make_metrics(recall=0.75, cost=0.02)   # same cost
    passed, _ = is_improvement(new, BEST, "cost")
    assert not passed


# ── Pareto objective ──────────────────────────────────────────

def test_pareto_recall_improvement_accepted():
    new = make_metrics(recall=0.85, latency=120.0)
    passed, checks = is_improvement(new, BEST, "pareto")
    assert passed, f"Expected KEEP, checks={checks}"

def test_pareto_latency_improvement_accepted():
    new = make_metrics(recall=0.75, latency=80.0)
    passed, checks = is_improvement(new, BEST, "pareto")
    assert passed, f"Expected KEEP, checks={checks}"

def test_pareto_recall_drop_too_large_rejected():
    new = make_metrics(recall=0.70, latency=80.0)  # recall drops > MAX_PARETO_RECALL_DROP
    passed, checks = is_improvement(new, BEST, "pareto")
    assert not passed
    assert not checks["recall_drop_ok"]

def test_pareto_no_improvement_rejected():
    new = make_metrics(recall=0.75, latency=120.0, cost=0.02)  # nothing improves
    passed, _ = is_improvement(new, BEST, "pareto")
    assert not passed


# ── Runner ────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed_count = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed_count += 1
        except AssertionError as e:
            print(f"  ❌ {t.__name__}: {e}")
    print(f"\n{passed_count}/{len(tests)} passed")