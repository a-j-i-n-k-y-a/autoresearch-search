# plot.py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import os

TSV_PATH = "results.tsv"

if not os.path.exists(TSV_PATH):
    print(f"No {TSV_PATH} found. Run some experiments first.")
    sys.exit(1)

df = pd.read_csv(TSV_PATH, sep="\t")
df = df.reset_index(drop=True)
df["exp_num"] = df.index + 1

# ─── COLORS ─────────────────────────────────────────────────
color_map = {
    "keep":    "#1D9E75",
    "discard": "#888780",
    "crash":   "#E24B4A",
}
colors = df["status"].map(color_map).fillna("#888780")

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle("Autoresearch — Experiment Results", fontsize=14, fontweight="bold", y=0.98)

# ─── PLOT 1: RECALL ─────────────────────────────────────────
ax1 = axes[0]
ax1.plot(df["exp_num"], df["recall"], color="#378ADD",
         linewidth=1.5, zorder=1, alpha=0.6)
ax1.scatter(df["exp_num"], df["recall"], c=colors,
            s=60, zorder=2, edgecolors="white", linewidths=0.5)

# Annotate best recall
best_idx = df["recall"].idxmax()
ax1.annotate(
    f"best: {df.loc[best_idx, 'recall']:.3f}\n{df.loc[best_idx, 'description'][:30]}",
    xy=(df.loc[best_idx, "exp_num"], df.loc[best_idx, "recall"]),
    xytext=(8, -20), textcoords="offset points",
    fontsize=8, color="#1D9E75",
    arrowprops=dict(arrowstyle="->", color="#1D9E75", lw=1)
)

ax1.set_ylabel("recall@10", fontsize=11)
ax1.set_ylim(-0.05, 1.1)
ax1.axhline(y=df[df["status"] == "keep"]["recall"].iloc[0] if len(df[df["status"] == "keep"]) else 0,
            color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="baseline")
ax1.grid(axis="y", alpha=0.3, linewidth=0.5)
ax1.spines[["top", "right"]].set_visible(False)

# ─── PLOT 2: LATENCY ────────────────────────────────────────
ax2 = axes[1]
ax2.bar(df["exp_num"], df["latency_ms"], color=colors, alpha=0.85,
        width=0.6, zorder=2)
ax2.set_ylabel("latency (ms)", fontsize=11)
ax2.grid(axis="y", alpha=0.3, linewidth=0.5)
ax2.spines[["top", "right"]].set_visible(False)

# Note crashes as zero bars
crash_count = (df["status"] == "crash").sum()
if crash_count:
    ax2.text(0.01, 0.95, f"{crash_count} crashes shown as 0ms",
             transform=ax2.transAxes, fontsize=8,
             color="#E24B4A", va="top")

# ─── PLOT 3: LLM COST (if column exists) ────────────────────
ax3 = axes[2]
if "llm_cost_usd" in df.columns:
    cumulative_cost = df["llm_cost_usd"].cumsum()
    ax3.fill_between(df["exp_num"], cumulative_cost,
                     alpha=0.3, color="#7F77DD")
    ax3.plot(df["exp_num"], cumulative_cost,
             color="#7F77DD", linewidth=1.5)
    ax3.scatter(df["exp_num"], cumulative_cost,
                c=colors, s=40, zorder=3,
                edgecolors="white", linewidths=0.5)
    total = df["llm_cost_usd"].sum()
    ax3.set_ylabel("cumulative cost ($)", fontsize=11)
    ax3.text(0.99, 0.95, f"total: ${total:.4f}",
             transform=ax3.transAxes, fontsize=9,
             color="#534AB7", va="top", ha="right", fontweight="bold")
else:
    ax3.text(0.5, 0.5, "llm_cost_usd not in results.tsv\n(run with updated agent_loop.py)",
             transform=ax3.transAxes, ha="center", va="center",
             fontsize=10, color="gray")
    ax3.set_ylabel("cumulative cost ($)", fontsize=11)

ax3.grid(axis="y", alpha=0.3, linewidth=0.5)
ax3.spines[["top", "right"]].set_visible(False)
ax3.set_xlabel("experiment number", fontsize=11)

# ─── X AXIS LABELS ──────────────────────────────────────────
ax3.set_xticks(df["exp_num"])
ax3.set_xticklabels(
    [f"#{n}" for n in df["exp_num"]],
    fontsize=8, rotation=45, ha="right"
)

# ─── LEGEND ─────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color="#1D9E75", label="keep"),
    mpatches.Patch(color="#888780", label="discard"),
    mpatches.Patch(color="#E24B4A", label="crash"),
]
fig.legend(handles=legend_patches, loc="upper right",
           fontsize=9, framealpha=0.8,
           bbox_to_anchor=(0.98, 0.95))

# ─── SUMMARY STATS ──────────────────────────────────────────
kept     = (df["status"] == "keep").sum()
discarded = (df["status"] == "discard").sum()
crashed  = (df["status"] == "crash").sum()
best_recall = df["recall"].max()
baseline_recall = df[df["description"].str.contains("baseline", case=False, na=False)]["recall"].mean()

summary = (
    f"experiments: {len(df)}  |  "
    f"kept: {kept}  |  "
    f"discarded: {discarded}  |  "
    f"crashed: {crashed}  |  "
    f"baseline recall: {baseline_recall:.3f}  |  "
    f"best recall: {best_recall:.3f}"
)
fig.text(0.5, 0.01, summary, ha="center", fontsize=8,
         color="gray", style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

output_path = "results_plot.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved → {output_path}")
plt.show()