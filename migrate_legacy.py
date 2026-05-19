"""
migrate_legacy.py — one-time script to flag legacy invalid experiments in log.jsonl.

Marks kept experiments as "legacy_invalid" if they:
  - have recall below RECALL_FLOOR, or
  - predate the multi-metric gate (missing mrr/ndcg/precision/top1).

Does NOT delete records. Safe to run multiple times (idempotent).
Run once: python migrate_legacy.py
"""
import json, os, shutil, time

LOG_PATH         = "experiments/log.jsonl"
RECALL_FLOOR     = 0.50
REQUIRED_METRICS = {"mrr", "ndcg", "precision", "top1"}


def load_records():
    if not os.path.exists(LOG_PATH):
        print("No log.jsonl found.")
        return []
    records = []
    with open(LOG_PATH) as f:
        content = f.read()
    for chunk in content.strip().split("\n\n"):
        chunk = chunk.strip()
        if chunk:
            try:
                records.append(json.loads(chunk))
            except json.JSONDecodeError:
                continue
    return records


def is_legacy_invalid(record):
    if record.get("status") == "legacy_invalid":
        return False  # already migrated — idempotent
    if record.get("status") != "keep":
        return False  # only flag kept experiments
    metrics     = record.get("metrics", {})
    full_metrics = record.get("full_metrics", {})
    if metrics.get("recall", 1.0) < RECALL_FLOOR:
        return True
    if not REQUIRED_METRICS.issubset(full_metrics.keys()):
        return True   # predates multi-metric gate
    return False


def migrate():
    records   = load_records()
    if not records:
        return

    backup = f"{LOG_PATH}.bak_{int(time.time())}"
    shutil.copy(LOG_PATH, backup)
    print(f"Backup written → {backup}")

    n_flagged = 0
    for r in records:
        if is_legacy_invalid(r):
            r["_original_status"] = r["status"]
            r["status"]           = "legacy_invalid"
            r["_migration_note"]  = (
                "Flagged by migrate_legacy.py: recall below floor "
                "or predates multi-metric gate."
            )
            n_flagged += 1

    with open(LOG_PATH, "w") as f:
        for r in records:
            f.write(json.dumps(r, indent=2) + "\n\n")

    print(f"Done — {n_flagged} records flagged out of {len(records)} total.")


if __name__ == "__main__":
    migrate()