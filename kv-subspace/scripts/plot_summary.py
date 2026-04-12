"""
Summary plots for KV-subspace compression paper.
Generates ~8 publication-quality figures into results/figures/.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

RESULTS = Path("/home/petty/pruning-research/kv-subspace/results")
FIG = RESULTS / "figures"
FIG.mkdir(exist_ok=True)

STYLE = {
    "baseline": ("#333333", "o", "Baseline"),
    "k128_4bit": ("#2196F3", "s", "k=128, 4-bit"),
    "k112_4bit": ("#FF9800", "^", "k=112, 4-bit"),
    "k96_4bit":  ("#F44336", "D", "k=96, 4-bit"),
    "k64_4bit":  ("#9C27B0", "v", "k=64, 4-bit"),
}

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1: PPL vs k (WikiText-2, 4-bit K-only)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 1: PPL vs k sweep (WikiText-2)")
df24 = pd.read_csv(RESULTS / "exp24_wikitext2_ppl.csv")
konly = df24[df24.compression_type == "K_only"].sort_values("k")
baseline_ppl = df24[df24.compression_type == "baseline"]["ppl"].values[0]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(konly["k"], konly["rel_ppl"], "o-", color="#2196F3", lw=2, ms=7, label="K-only, 4-bit")
ax.axhline(1.0, color="#333", ls="--", lw=1.2, label=f"Baseline (PPL={baseline_ppl:.2f})")
ax.axvline(128, color="#2196F3", ls=":", lw=1, alpha=0.6)
ax.axvline(112, color="#FF9800", ls=":", lw=1, alpha=0.6)
ax.text(128+1, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 1.5 else 7.5, "k=128", color="#2196F3", fontsize=8)
ax.text(112+1, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 1.5 else 7.5, "k=112", color="#FF9800", fontsize=8)
ax.set_xlabel("Subspace rank k")
ax.set_ylabel("Relative PPL (vs baseline)")
ax.set_title("WikiText-2 PPL Degradation vs Subspace Rank\n(4-bit K-only compression, Qwen3-14B-AWQ)")
ax.legend()
ax.set_ylim(bottom=0.8)
fig.tight_layout()
fig.savefig(FIG / "fig1_ppl_vs_k.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2: Compression ratio vs rel-PPL frontier (exp24)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 2: CR vs rel-PPL frontier")
fig, ax = plt.subplots(figsize=(6, 4))
konly2 = konly.copy()
# Compute CR from bits and k (d_head=128): 
# compressed bytes = k*(bits/8)*2 (K+V actually, but here K-only: k*(bits/8))
# But exp24 CSV doesn't have CR column — use k/128 as proxy rank ratio (actual CR ≈ 128/k * 4 accounting for 4-bit vs 16-bit)
# Use k/128 * 4 bits / 16 bits = k/512 fraction of original, so CR = 512/k
konly2["cr"] = 512.0 / konly2["k"]
# Pareto frontier
pareto = konly2.sort_values("cr")
ax.scatter(pareto["cr"], pareto["rel_ppl"], c="#2196F3", s=60, zorder=5, label="K-only 4-bit configs")
for _, row in pareto.iterrows():
    ax.annotate(f"k={int(row['k'])}", (row["cr"], row["rel_ppl"]),
                textcoords="offset points", xytext=(5, 3), fontsize=7, color="#555")
ax.axhline(1.05, color="#888", ls="--", lw=1, label="5% PPL budget")
ax.axhline(1.10, color="#ccc", ls="--", lw=1, label="10% PPL budget")
ax.set_xlabel("Compression Ratio (×)")
ax.set_ylabel("Relative PPL")
ax.set_title("Quality–Compression Tradeoff Frontier\n(4-bit K-only, Qwen3-14B-AWQ)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_cr_vs_ppl.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3: Layer sensitivity (exp16)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 3: Layer sensitivity")
df16 = pd.read_csv(RESULTS / "exp16_layer_sensitivity.csv")
fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#F44336" if d > 0 else "#4CAF50" for d in df16["ppl_delta"]]
ax.bar(df16["layer_idx"], df16["ppl_delta"], color=colors, width=0.8, edgecolor="none")
ax.axhline(0, color="#333", lw=0.8)
ax.set_xlabel("Layer index")
ax.set_ylabel("ΔPPL (compressed – baseline)")
ax.set_title("Per-Layer Sensitivity to K-Subspace Compression\n(k=112, 4-bit; red=hurts, green=free/helps)")
ax.set_xticks(range(0, len(df16), 5))
fig.tight_layout()
fig.savefig(FIG / "fig3_layer_sensitivity.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4: Adaptive scheduling vs uniform (exp18)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 4: Adaptive vs uniform scheduling")
df18 = pd.read_csv(RESULTS / "exp18_adaptive_policy.csv")
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(df18["budget_k"], df18["uniform_rel_ppl"], "o--", color="#555", lw=1.5, ms=5, label="Uniform k")
ax.plot(df18["budget_k"], df18["rank_rel_ppl"], "s-", color="#2196F3", lw=2, ms=6, label="Rank-proportional")
ax.plot(df18["budget_k"], df18["greedy_rel_ppl"], "^-", color="#FF9800", lw=2, ms=6, label="Greedy (layer sensitivity)")
ax.axhline(1.0, color="#333", ls="--", lw=1)
ax.set_xlabel("Mean budget k")
ax.set_ylabel("Relative PPL")
ax.set_title("Adaptive K-Scheduling vs Uniform Allocation\n(Qwen3-14B-AWQ, WikiText-2)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig4_adaptive_scheduling.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5: SubRotQ vs PolarQuant (exp22)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 5: SubRotQ vs PolarQuant")
df22 = pd.read_csv(RESULTS / "exp22_quantizer_comparison.csv")
fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)
for ax, bits in zip(axes, [4, 8]):
    sub = df22[df22["bits"] == bits].sort_values("k")
    for method, color, label in [("subrotq", "#2196F3", "SubRotQ"), ("polarquant", "#F44336", "PolarQuant (Han et al.)")]:
        m = sub[sub["quantizer"] == method]
        ax.plot(m["k"], m["rel_ppl"], "o-", color=color, lw=2, ms=6, label=label)
    ax.axhline(1.0, color="#333", ls="--", lw=1)
    ax.set_xlabel("Subspace rank k")
    ax.set_ylabel("Relative PPL")
    ax.set_title(f"{bits}-bit quantization")
    ax.legend(fontsize=8)
fig.suptitle("SubRotQ vs PolarQuant: Quality Comparison\n(K-only compression, Qwen3-14B-AWQ)", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "fig5_quantizer_comparison.png", bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 6: NIAH accuracy heatmaps (exp25, k128 vs baseline)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 6: NIAH heatmaps")
df25 = pd.read_csv(RESULTS / "exp25_niah_robust.csv")
ctx_levels = sorted(df25["ctx_len"].unique())
depth_levels = sorted(df25["depth"].unique())

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
configs_to_plot = ["baseline", "k128_4bit", "k96_4bit"]
titles = ["Baseline", "k=128, 4-bit", "k=96, 4-bit"]

for ax, cfg, title in zip(axes, configs_to_plot, titles):
    sub = df25[df25["config"] == cfg]
    heat = np.zeros((len(depth_levels), len(ctx_levels)))
    for i, d in enumerate(depth_levels):
        for j, c in enumerate(ctx_levels):
            cell = sub[(sub["depth"] == d) & (sub["ctx_len"] == c)]
            if len(cell) > 0:
                heat[i, j] = cell["correct"].astype(int).mean()
    im = ax.imshow(heat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto", origin="lower")
    ax.set_xticks(range(len(ctx_levels)))
    ax.set_xticklabels([f"{c//1024}K" for c in ctx_levels], fontsize=7, rotation=45)
    ax.set_yticks(range(len(depth_levels)))
    ax.set_yticklabels([f"{d:.0%}" for d in depth_levels], fontsize=7)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Needle depth")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, label="Accuracy")

fig.suptitle("NIAH: Needle-in-a-Haystack Accuracy (n=15/cell)", fontsize=12)
fig.tight_layout()
fig.savefig(FIG / "fig6_niah_heatmap.png", bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 7: Cross-architecture (Llama3 vs Qwen3, exp21)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 7: Cross-arch Llama3 vs Qwen3")
df21 = pd.read_csv(RESULTS / "exp21_llama3_validation.csv")
# K-only configs
konly21 = df21[(df21["compress_v"] == False) | (df21["k_V"] == 0) | df21["viable"].isna()].copy()
# Actually: K-only = k_V column not meaningful OR compress_v=False
konly21 = df21[df21["subexp"].str.contains("k_only|konly|K_only", na=False, case=False)].copy()
if len(konly21) == 0:
    # Fallback: all rows where compress_v is False (K-only)
    konly21 = df21[df21["compress_v"] == False].copy()

# Also get Qwen3 baseline from exp24
qwen3_pts = konly.rename(columns={"rel_ppl": "rel_ppl_qwen3"})

fig, ax = plt.subplots(figsize=(6, 4))
if len(konly21) > 0:
    ax.plot(konly21.sort_values("k_K")["k_K"], konly21.sort_values("k_K")["rel_ppl"],
            "s-", color="#FF9800", lw=2, ms=6, label="Llama-3.1-8B-Instruct-AWQ")
ax.plot(konly["k"], konly["rel_ppl"], "o-", color="#2196F3", lw=2, ms=6, label="Qwen3-14B-AWQ")
ax.axhline(1.0, color="#333", ls="--", lw=1)
ax.set_xlabel("Subspace rank k")
ax.set_ylabel("Relative PPL")
ax.set_title("K-Subspace Compression: Cross-Architecture\n(4-bit K-only)")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "fig7_cross_arch.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 8: Downstream task accuracy (exp27)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 8: Downstream task accuracy")
with open(RESULTS / "exp27_downstream_tasks.json") as f:
    d27 = json.load(f)

tasks = ["arc_challenge", "hellaswag", "arc_easy", "winogrande"]
task_labels = ["ARC-C\n(25-shot)", "HellaSwag\n(10-shot)", "ARC-Easy\n(0-shot)", "Winogrande\n(5-shot)"]
configs = ["baseline", "k128_4bit", "k112_4bit", "k96_4bit"]
cfg_labels = ["Baseline", "k=128 4-bit\n(4× CR)", "k=112 4-bit\n(~3× CR)", "k=96 4-bit\n(~2.3× CR)"]
cfg_colors = ["#333333", "#2196F3", "#FF9800", "#F44336"]

fig, ax = plt.subplots(figsize=(10, 5))
n_tasks = len(tasks)
n_cfgs = len(configs)
w = 0.18
x = np.arange(n_tasks)

for i, (cfg, label, color) in enumerate(zip(configs, cfg_labels, cfg_colors)):
    accs = []
    for t in tasks:
        r = d27.get(cfg, {}).get(t, {})
        acc = r.get("acc")
        accs.append(acc if acc is not None else 0.0)
    bars = ax.bar(x + (i - n_cfgs/2 + 0.5) * w, accs, w*0.9, label=label, color=color, alpha=0.85)
    # Label bars
    for bar, val in zip(bars, accs):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005, f"{val:.2f}",
                    ha="center", va="bottom", fontsize=6.5, color="#333")

ax.set_xticks(x)
ax.set_xticklabels(task_labels)
ax.set_ylabel("Accuracy")
ax.set_ylim(0.3, 0.95)
ax.set_title("Downstream Task Accuracy: KV-Subspace Compression\n(Qwen3-14B-AWQ, limit=300 samples/task)")
ax.legend(loc="lower right", fontsize=9)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
fig.tight_layout()
fig.savefig(FIG / "fig8_downstream_tasks.png")
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Fig 9: Relative accuracy drop (normalized to baseline)
# ─────────────────────────────────────────────────────────────────────────────
print("Fig 9: Relative accuracy drop")
fig, ax = plt.subplots(figsize=(8, 4))
baseline_accs = {t: d27.get("baseline", {}).get(t, {}).get("acc") for t in tasks}

for i, (cfg, label, color) in enumerate(zip(configs[1:], cfg_labels[1:], cfg_colors[1:])):
    rel_accs = []
    for t in tasks:
        b = baseline_accs[t]
        c = d27.get(cfg, {}).get(t, {}).get("acc")
        if b and c:
            rel_accs.append((c - b) / b * 100)
        else:
            rel_accs.append(0.0)
    ax.bar(x + (i - 1) * w*1.1, rel_accs, w, label=label, color=color, alpha=0.85)

ax.axhline(0, color="#333", lw=0.8)
ax.axhline(-5, color="#888", ls="--", lw=1, label="−5% threshold")
ax.set_xticks(x)
ax.set_xticklabels(task_labels)
ax.set_ylabel("Accuracy change vs baseline (%)")
ax.set_title("Relative Accuracy Change vs Baseline\n(per downstream task)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig9_relative_accuracy_drop.png")
plt.close()

print("\nDone! Figures saved to:", FIG)
for f in sorted(FIG.glob("fig*.png")):
    print(f"  {f.name}")
