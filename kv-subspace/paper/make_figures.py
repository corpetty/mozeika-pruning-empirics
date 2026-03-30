#!/usr/bin/env python3
"""Generate all paper figures from CSV results."""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import os, sys

RESULTS = "/home/petty/pruning-research/kv-subspace/results"
OUT = "/home/petty/pruning-research/kv-subspace/paper/figures"
os.makedirs(OUT, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

VIABILITY = 1.20   # 20% PPL budget
RED   = "#d62728"
GREEN = "#2ca02c"
BLUE  = "#1f77b4"
ORANGE= "#ff7f0e"
PURPLE= "#9467bd"
GRAY  = "#7f7f7f"

def hline_viable(ax, y=VIABILITY, label=True):
    ax.axhline(y, color="black", linewidth=0.8, linestyle="--",
               label="20% budget" if label else None)

def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path)
    plt.close("all")
    print(f"  saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Bitrate × K heatmap (truncation vs quantization)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: bitrate_k heatmap")
df = pd.read_csv(f"{RESULTS}/bitrate_k_sweep.csv")
df = df[df.config != "baseline"].copy()
df["k_K"] = df["k_K"].astype(int)

pivot = df.groupby(["k_K","nbits_K"])["rel_ppl"].mean().unstack()
pivot = pivot.reindex(index=sorted(pivot.index), columns=sorted(pivot.columns))

fig, ax = plt.subplots(figsize=(4.2, 2.8))
cmap = plt.cm.RdYlGn_r
# clip display at 4.0 so heatmap isn't washed out by k=64 extremes
vmax = 4.0
data = pivot.values.copy()
im = ax.imshow(data, cmap=cmap, vmin=1.0, vmax=vmax, aspect="auto")

ax.set_xticks(range(len(pivot.columns)))
ax.set_xticklabels([f"{b}-bit" for b in pivot.columns])
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels([f"k={k}" for k in pivot.index])
ax.set_xlabel("Bit depth")
ax.set_ylabel("Subspace dim $k$")
ax.set_title("Rel. PPL heatmap ($k$ × bits) — Qwen3-14B-AWQ")

# Annotate cells
for i, k in enumerate(pivot.index):
    for j, b in enumerate(pivot.columns):
        v = pivot.loc[k, b]
        if np.isnan(v):
            continue
        color = "white" if v > 2.5 else "black"
        ax.text(j, i, f"{v:.2f}×", ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold")

cb = fig.colorbar(im, ax=ax, shrink=0.85)
cb.set_label("Rel. PPL", fontsize=8)
# Mark viable cells with a green border
for i, k in enumerate(pivot.index):
    for j, b in enumerate(pivot.columns):
        v = pivot.loc[k, b]
        if not np.isnan(v) and v <= VIABILITY:
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                  linewidth=2, edgecolor=GREEN,
                                  facecolor="none")
            ax.add_patch(rect)

plt.tight_layout()
save("fig1_bitrate_heatmap.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Long-context PPL stability
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: long context PPL")
df = pd.read_csv(f"{RESULTS}/long_context_ppl.csv")

configs = {
    "k128_4bit": (BLUE,   "k=128/4-bit", "o"),
    "k112_4bit": (ORANGE, "k=112/4-bit", "s"),
    "k96_4bit":  (PURPLE, "k=96/4-bit",  "^"),
    "k64_4bit":  (RED,    "k=64/4-bit",  "D"),
}

fig, ax = plt.subplots(figsize=(4.5, 2.8))
for cfg, (color, label, marker) in configs.items():
    sub = df[df.config == cfg].sort_values("ctx_len")
    if sub.empty:
        continue
    ax.plot(sub.ctx_len / 1000, sub.relative_ppl, color=color,
            label=label, marker=marker, markersize=4)

hline_viable(ax)
ax.set_xlabel("Context length (K tokens)")
ax.set_ylabel("Relative PPL")
ax.set_title("K compression stability over long context (Qwen3-14B-AWQ)")
ax.set_xscale("log")
ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 40])
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.set_ylim(bottom=0.9)
ax.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
save("fig2_long_context.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Cross-architecture comparison (bar)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: cross-arch bar")
df = pd.read_csv(f"{RESULTS}/cross_arch_results.csv")

# Best (min k) viable config per model — use k112/4bit as consistent comparator
k112 = df[(df.k == 112) & (df.n_bits == 4)].copy()
if k112.empty:
    # fallback: use whatever k is closest to 112
    df_nonbase = df[df.k.notna()].copy()
    k112 = df_nonbase[df_nonbase.k == df_nonbase.k.unique()[
        np.argmin(abs(df_nonbase.k.unique() - 112))]]

summary = k112.groupby("model")["rel_ppl"].mean().reset_index()
# Also grab compression ratio
cr = k112.groupby("model")["compression_ratio"].mean().reset_index()
summary = summary.merge(cr, on="model")

# Sort by rel_ppl
summary = summary.sort_values("rel_ppl")

fig, ax = plt.subplots(figsize=(4.5, 2.8))
colors = [GREEN if r <= VIABILITY else RED for r in summary.rel_ppl]
bars = ax.bar(range(len(summary)), summary.rel_ppl, color=colors, width=0.55, zorder=2)
hline_viable(ax)
ax.set_xticks(range(len(summary)))
ax.set_xticklabels([m.split("-")[0] + "\n" + "-".join(m.split("-")[1:3])
                    for m in summary.model], fontsize=7.5)
ax.set_ylabel("Relative PPL (k=112/4-bit)")
ax.set_title("Cross-architecture K compression — k=112/4-bit")
ax.set_ylim(0.9, max(summary.rel_ppl) * 1.1)
ax.grid(axis="y", alpha=0.3, zorder=0)
# Label CR on each bar
for i, (_, row) in enumerate(summary.iterrows()):
    ax.text(i, row.rel_ppl + 0.01, f"{row.compression_ratio:.1f}×",
            ha="center", va="bottom", fontsize=7.5, color="black")
plt.tight_layout()
save("fig3_cross_arch.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Layer sensitivity (Exp 16)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: layer sensitivity")
df = pd.read_csv(f"{RESULTS}/exp16_layer_sensitivity.csv")

fig, ax = plt.subplots(figsize=(5.0, 2.6))
colors = [RED if d > 0 else GREEN for d in df.ppl_delta]
ax.bar(df.layer_idx, df.ppl_delta, color=colors, width=0.7, zorder=2)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Layer index")
ax.set_ylabel("ΔPPL vs. baseline")
ax.set_title("Layer sensitivity to K compression (k=64/4-bit ablation, Qwen3-14B-AWQ)")
ax.grid(axis="y", alpha=0.3, zorder=0)

# Annotate top outliers
top_n = df.nlargest(3, "ppl_delta")
for _, row in top_n.iterrows():
    ax.annotate(f"L{int(row.layer_idx)}", xy=(row.layer_idx, row.ppl_delta),
                xytext=(row.layer_idx, row.ppl_delta + 0.06),
                ha="center", fontsize=7, color=RED)

free_patch = mpatches.Patch(color=GREEN, label="Negative Δ (free to compress)")
sens_patch = mpatches.Patch(color=RED,   label="Positive Δ (sensitive)")
ax.legend(handles=[sens_patch, free_patch], loc="upper left", fontsize=7.5)
plt.tight_layout()
save("fig4_layer_sensitivity.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — V compression failure (Exp 20 + Exp 21 combined)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5: V compression failure")
df20 = pd.read_csv(f"{RESULTS}/exp20_v_threshold.csv")
df21 = pd.read_csv(f"{RESULTS}/exp21_llama3_validation.csv")

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))

# Left: Qwen3 V threshold scan (Exp 20)
sub20 = df20[df20.eval_ctx == 4096].copy()
k_vals = sub20.k_V.values
rel_ppls = sub20.rel_ppl.values
axes[0].plot(k_vals, rel_ppls, "o-", color=RED, markersize=5, label="K=112/4-bit + V=k/4-bit")
axes[0].axhline(VIABILITY, color="black", linewidth=0.8, linestyle="--", label="20% budget")
axes[0].axvline(128, color=GRAY, linewidth=0.8, linestyle=":", label="k=d (full rank)")
axes[0].set_xlabel("V subspace dimension $k_V$")
axes[0].set_ylabel("Relative PPL")
axes[0].set_title("V compression scan — Qwen3-14B-AWQ")
axes[0].set_yscale("log")
axes[0].set_yticks([1, 1.2, 2, 5])
axes[0].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
axes[0].legend(fontsize=7)

# Right: Llama-3.1 K vs V comparison (Exp 21)
# Show K-only, V-only, and K+V at 128 as grouped bar
configs_21 = {
    "K-only\nk=112": ("B_k_only", None, BLUE),
    "V-only\nk=112": ("C_v_only", None, RED),
    "K+V\nk=128":    ("D_kv_full", 128, GREEN),
    "K+V\nk=124":    ("D_kv_full", 124, ORANGE),
}
bar_labels, bar_vals, bar_colors = [], [], []

# K-only
sub = df21[(df21.subexp == "B_k_only") & (df21.k_K == 112)]["rel_ppl"].mean()
if not np.isnan(sub):
    bar_labels.append("K-only\nk=112"); bar_vals.append(sub); bar_colors.append(BLUE)

# V-only
sub = df21[(df21.subexp == "C_v_only") & (df21.k_V == 112)]["rel_ppl"].mean()
if not np.isnan(sub):
    bar_labels.append("V-only\nk=112"); bar_vals.append(sub); bar_colors.append(RED)

# K+V k=128
sub = df21[(df21.subexp == "D_kv_full") & (df21.k_V == 128)]["rel_ppl"].mean()
if not np.isnan(sub):
    bar_labels.append("K+V\nk=128"); bar_vals.append(sub); bar_colors.append(GREEN)

# K+V k=124
sub = df21[(df21.subexp == "D_kv_full") & (df21.k_V == 124)]["rel_ppl"].mean()
if not np.isnan(sub):
    bar_labels.append("K+V\nk=124"); bar_vals.append(sub); bar_colors.append(ORANGE)

axes[1].bar(range(len(bar_labels)), bar_vals, color=bar_colors, width=0.55, zorder=2)
axes[1].axhline(VIABILITY, color="black", linewidth=0.8, linestyle="--")
axes[1].set_xticks(range(len(bar_labels)))
axes[1].set_xticklabels(bar_labels, fontsize=8)
axes[1].set_ylabel("Relative PPL")
axes[1].set_title("K vs V compression — Llama-3.1-8B (no QK-norm)")
axes[1].grid(axis="y", alpha=0.3, zorder=0)
axes[1].set_yscale("log")
axes[1].set_yticks([1, 1.2, 2, 5, 12])
axes[1].get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# Annotate values
for i, v in enumerate(bar_vals):
    axes[1].text(i, v * 1.05, f"{v:.2f}×", ha="center", fontsize=7.5)

plt.suptitle("V compression fails across architectures (QK-norm hypothesis rejected)",
             fontsize=8.5, y=1.01)
plt.tight_layout()
save("fig5_v_failure.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — Needle-in-haystack retrieval (Exp 15)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 6: needle retrieval heatmap")
df = pd.read_csv(f"{RESULTS}/exp15_needle.csv")
df["correct"] = df["correct"].astype(int)

configs = ["baseline", "k128_4bit", "k96_4bit"]
ctx_lens = sorted(df.ctx_len.unique())
depths = sorted(df.depth.unique())

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), sharey=True)
for ax, cfg in zip(axes, configs):
    sub = df[df.config == cfg]
    matrix = np.zeros((len(depths), len(ctx_lens)))
    for i, d in enumerate(depths):
        for j, c in enumerate(ctx_lens):
            cell = sub[(sub.depth == d) & (sub.ctx_len == c)]["correct"]
            matrix[i, j] = cell.mean() if len(cell) > 0 else np.nan
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(ctx_lens)))
    ax.set_xticklabels([f"{c//1024}K" for c in ctx_lens], fontsize=7.5)
    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels([f"{int(d*100)}%" for d in depths], fontsize=7.5)
    ax.set_title(cfg.replace("_", "/"), fontsize=8.5)
    ax.set_xlabel("Context length")
    for i in range(len(depths)):
        for j in range(len(ctx_lens)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, "1.0" if v == 1 else ("0" if v == 0 else f"{v:.0%}"),
                        ha="center", va="center", fontsize=8,
                        color="white" if v < 0.4 else "black")
axes[0].set_ylabel("Needle depth")
fig.colorbar(im, ax=axes[-1], shrink=0.85, label="Accuracy")
plt.suptitle("Needle-in-haystack retrieval accuracy", fontsize=9, y=1.01)
plt.tight_layout()
save("fig6_needle.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Cross-domain calibration transfer (Exp 17)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 7: cross-domain")
df = pd.read_csv(f"{RESULTS}/exp17_cross_domain.csv")

configs_cd = ["k128_4bit", "k96_4bit"]
domains = sorted(df.calib_domain.unique())

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8))
for ax, cfg in zip(axes, configs_cd):
    sub = df[df.config == cfg]
    baseline = df[df.config == "baseline"].groupby("eval_domain")["ppl"].mean()
    pivot = sub.pivot_table(index="calib_domain", columns="eval_domain", values="ppl")
    # Normalize by baseline to get rel_ppl
    rel_pivot = pivot.div(baseline, axis=1)
    im = ax.imshow(rel_pivot.values, vmin=1.0, vmax=min(rel_pivot.values.max(), 4.0),
                   cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=7.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7.5)
    ax.set_title(f"{cfg.replace('_','/')} — calib→eval rel. PPL", fontsize=8.5)
    ax.set_xlabel("Eval domain")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = rel_pivot.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if v > 2.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.85)
axes[0].set_ylabel("Calibration domain")
plt.suptitle("Cross-domain calibration transfer (relative PPL)", fontsize=9, y=1.01)
plt.tight_layout()
save("fig7_cross_domain.pdf")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 8 — Basis drift over context (Exp 13 sub-exp C)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 8: basis drift")
df = pd.read_csv(f"{RESULTS}/long_context_basis_drift.csv")

fig, ax = plt.subplots(figsize=(4.5, 2.8))
for kv, color, label in [("K", BLUE, "Key"), ("V", RED, "Value")]:
    sub = df[df.kv_type == kv]
    mean_per_compare = sub.groupby("compare")["overlap"].mean().reset_index()
    # x-axis = comparison window index
    cmp_order = ["mid_early", "mid", "mid_late", "late"]
    xs = [i+1 for i, c in enumerate(cmp_order) if c in mean_per_compare["compare"].values]
    ys = [mean_per_compare[mean_per_compare["compare"]==c]["overlap"].values[0]
          for c in cmp_order if c in mean_per_compare["compare"].values]
    if ys:
        ax.plot(xs, ys, "o-", color=color, label=label, markersize=5)

ax.set_xticks([1, 2, 3, 4])
ax.set_xticklabels(["0–2K\nvs\n2–10K", "0–2K\nvs\n10–20K",
                     "0–2K\nvs\n20–30K", "0–2K\nvs\n35–40K"], fontsize=7.5)
ax.set_ylabel("PCA basis overlap (subspace cosine²)")
ax.set_title("Basis drift: early vs. later context windows")
ax.set_ylim(0.5, 1.0)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
save("fig8_basis_drift.pdf")


print("\nAll figures saved to", OUT)
print("Files:", sorted(os.listdir(OUT)))
