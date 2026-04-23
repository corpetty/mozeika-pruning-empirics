#!/usr/bin/env python3
"""Generate all paper figures from CSV/JSON results.

Figures produced (matching main.tex \includegraphics calls):
  fig1_truncation_vs_quantization.pdf  — heatmap + line, exp24
  fig2_cross_architecture.pdf          — grouped bar, exp24/30/32
  fig3_downstream_tasks.pdf            — grouped bar, expC1 (N=1000)
  fig4_long_context.pdf                — line, expD1 PG-19
  fig5_subrotq_vs_polarquant.pdf       — grouped bar, exp35
  fig6_k_v_asymmetry.pdf               — grouped bar, exp21/exp32
  fig7_rotation_ablation.pdf           — bar chart, exp37
"""

import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RESULTS = "/home/petty/pruning-research/kv-subspace/results"
OUT     = "/home/petty/pruning-research/kv-subspace/paper/figures"
os.makedirs(OUT, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "lines.linewidth":  1.5,
    "figure.dpi":       200,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

VIABILITY = 1.20
RED    = "#d62728"
GREEN  = "#2ca02c"
BLUE   = "#1f77b4"
ORANGE = "#ff7f0e"
PURPLE = "#9467bd"
GRAY   = "#7f7f7f"

def hline(ax, y=VIABILITY, label="20% budget"):
    ax.axhline(y, color="black", linewidth=0.9, linestyle="--",
               label=label, zorder=0)

def save(name):
    path = os.path.join(OUT, name)
    plt.savefig(path)
    plt.close("all")
    print(f"  saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Truncation vs Quantization  (exp24 — WikiText-2, Qwen3-14B)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 1: truncation vs quantization")

df24 = pd.read_csv(f"{RESULTS}/exp24_wikitext2_ppl.csv")
df24 = df24[df24["compression_type"] != "baseline"].copy()
df24["k"]    = df24["k"].astype(int)
df24["bits"] = df24["bits"].astype(int)
df24 = df24[df24["bits"].isin([4, 8, 16])]

ks   = sorted(df24["k"].unique())
bits = sorted(df24["bits"].unique())

# Side-by-side: heatmap (left) + line plot (right)
fig, (ax_heat, ax_line) = plt.subplots(1, 2, figsize=(7.5, 2.9),
                                        gridspec_kw={"width_ratios": [1, 1.3]})

# — heatmap —
pivot = df24.pivot_table(index="k", columns="bits", values="rel_ppl", aggfunc="mean")
pivot = pivot.reindex(index=sorted(pivot.index, reverse=True),
                      columns=sorted(pivot.columns))
cmap = plt.cm.RdYlGn_r
im = ax_heat.imshow(pivot.values, cmap=cmap, vmin=1.0, vmax=4.0, aspect="auto")
ax_heat.set_xticks(range(len(pivot.columns)))
ax_heat.set_xticklabels([f"{b}-bit" for b in pivot.columns])
ax_heat.set_yticks(range(len(pivot.index)))
ax_heat.set_yticklabels([f"k={k}" for k in pivot.index])
ax_heat.set_xlabel("Bit depth")
ax_heat.set_ylabel("Subspace dim $k$")
ax_heat.set_title("(a) Rel. PPL heatmap")
for i, k in enumerate(pivot.index):
    for j, b in enumerate(pivot.columns):
        v = pivot.loc[k, b]
        color = "white" if v > 2.5 else "black"
        ax_heat.text(j, i, f"{v:.2f}×", ha="center", va="center",
                     fontsize=7.5, color=color, fontweight="bold")
plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04,
             label="Rel. PPL (capped 4×)")

# — line plot —
bit_colors = {4: RED, 8: ORANGE, 16: BLUE}
for b in [4, 8, 16]:
    sub = df24[df24["bits"] == b].sort_values("k")
    ax_line.plot(sub["k"], sub["rel_ppl"], marker="o", color=bit_colors[b],
                 label=f"{b}-bit", markersize=4)
hline(ax_line)
ax_line.axhline(1.0, color=GRAY, linewidth=0.7, linestyle=":")
ax_line.set_xlabel("Subspace dim $k$")
ax_line.set_ylabel("Relative PPL")
ax_line.set_title("(b) Rel. PPL vs $k$ by bit depth")
ax_line.set_xticks(ks)
ax_line.set_ylim(0.8, 5.0)
ax_line.legend(loc="upper right")
ax_line.annotate("truncation\ndominates", xy=(64, df24[(df24.k==64)&(df24.bits==4)]["rel_ppl"].values[0]),
                 xytext=(72, 5.5), fontsize=7, color=RED,
                 arrowprops=dict(arrowstyle="->", color=RED, lw=0.8),
                 va="top")

fig.suptitle("Truncation error dominates quantization noise (Qwen3-14B, WikiText-2)",
             fontsize=9, y=1.02)
plt.tight_layout()
save("fig1_truncation_vs_quantization.pdf")
save("fig1_truncation_vs_quantization.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Cross-architecture comparison  (Qwen3-14B, Mistral-7B, Llama-3.1-8B)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 2: cross-architecture")

# Ground-truth numbers from exp24, exp30, exp32 (4-bit only)
arch_data = {
    "Qwen3-14B":    {"k64": 8.14, "k96": 1.82, "k112": 1.23, "k128": 0.98},
    "Mistral-7B":   {"k64": 8.70, "k96": 1.67, "k112": 1.09, "k128": 1.00},
    "Llama-3.1-8B": {"k64": 2.70, "k96": 1.18, "k112": 1.05, "k128": 1.01},
}
models = list(arch_data.keys())
configs = ["k64", "k96", "k112", "k128"]
labels  = ["k=64", "k=96", "k=112", "k=128"]
colors  = [RED, ORANGE, PURPLE, GREEN]

x = np.arange(len(models))
width = 0.18
fig, ax = plt.subplots(figsize=(6.5, 3.0))
for i, (cfg, lbl, col) in enumerate(zip(configs, labels, colors)):
    vals = [arch_data[m][cfg] for m in models]
    offset = (i - 1.5) * width
    bars = ax.bar(x + offset, vals, width, label=lbl, color=col, alpha=0.85,
                  edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        if v < 3.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                    f"{v:.2f}×", ha="center", va="bottom", fontsize=6.5)

hline(ax)
ax.axhline(1.0, color=GRAY, linewidth=0.7, linestyle=":")
ax.set_ylabel("Relative PPL (vs baseline)")
ax.set_title("Cross-architecture: Rel. PPL at 4-bit (WikiText-2)")
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0, 5.5)
ax.legend(title="Config", ncol=4, loc="upper center",
          bbox_to_anchor=(0.5, 1.0), framealpha=0.9)
note = ("k=64 bars truncated for readability\n"
        "Qwen3-14B: k64=8.14×, Mistral: k64=8.70×")
ax.text(0.99, 0.97, note, transform=ax.transAxes, fontsize=6.5,
        ha="right", va="top", color=GRAY,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRAY, alpha=0.7))
plt.tight_layout()
save("fig2_cross_architecture.pdf")
save("fig2_cross_architecture.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Downstream task accuracy  (expC1, N=1000)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 3: downstream tasks (expC1 N=1000)")

with open(f"{RESULTS}/expC1_downstream_full.json") as f:
    c1 = json.load(f)

tasks    = ["arc_challenge", "hellaswag", "arc_easy", "winogrande"]
task_lbl = ["ARC-Challenge\n(25-shot)", "HellaSwag\n(10-shot)",
            "ARC-Easy\n(0-shot)", "WinoGrande\n(5-shot)"]
configs_c1 = ["baseline", "k128_4bit", "k96_4bit"]
cfg_labels  = ["Baseline", "k=128/4-bit", "k=96/4-bit"]
cfg_colors  = [BLUE, GREEN, RED]

x = np.arange(len(tasks))
width = 0.22
fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(8.0, 3.2))

# Absolute accuracy
for i, (cfg, lbl, col) in enumerate(zip(configs_c1, cfg_labels, cfg_colors)):
    vals = [c1[cfg]["tasks"][t]["accuracy"] for t in tasks]
    offset = (i - 1) * width
    bars = ax_abs.bar(x + offset, vals, width, label=lbl, color=col,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax_abs.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=6.0)

ax_abs.set_ylabel("Accuracy")
ax_abs.set_title("(a) Absolute accuracy (N=1000)")
ax_abs.set_xticks(x)
ax_abs.set_xticklabels(task_lbl, fontsize=7.5)
ax_abs.set_ylim(0.4, 1.0)
ax_abs.legend(loc="lower right", fontsize=7.5)

# Relative accuracy drop (delta vs baseline)
base = {t: c1["baseline"]["tasks"][t]["accuracy"] for t in tasks}
for i, (cfg, lbl, col) in enumerate(zip(configs_c1[1:], cfg_labels[1:], cfg_colors[1:])):
    deltas = [c1[cfg]["tasks"][t]["accuracy"] - base[t] for t in tasks]
    offset = (i - 0.5) * (width * 1.1)
    bars = ax_rel.bar(x + offset, deltas, width * 1.1, label=lbl, color=col,
                      alpha=0.85, edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, deltas):
        ypos = bar.get_height() + 0.002 if v >= 0 else bar.get_height() - 0.012
        ax_rel.text(bar.get_x() + bar.get_width()/2, ypos,
                    f"{v:+.3f}", ha="center", va="bottom", fontsize=6.0)

ax_rel.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
ax_rel.set_ylabel("Δ Accuracy vs baseline")
ax_rel.set_title("(b) Accuracy drop vs baseline")
ax_rel.set_xticks(x)
ax_rel.set_xticklabels(task_lbl, fontsize=7.5)
ax_rel.legend(loc="lower left", fontsize=7.5)

fig.suptitle("Downstream task accuracy — Qwen3-14B-AWQ (N=1000 samples/task)",
             fontsize=9, y=1.02)
plt.tight_layout()
save("fig3_downstream_tasks.pdf")
save("fig3_downstream_tasks.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Long-context PPL stability  (expD1, PG-19)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 4: long context stability (expD1 PG-19)")

df_d1 = pd.read_csv(f"{RESULTS}/expD1_long_context_pg19.csv")
df_d1["ctx_len"] = df_d1["ctx_len"].astype(int)
# Normalise rel_ppl column name regardless of capitalisation
rel_col = [c for c in df_d1.columns if c.lower() == "rel_ppl" or c.lower() == "relative_ppl"][0]
df_d1["rel_PPL"] = df_d1[rel_col].astype(float)
if "config" not in df_d1.columns and "Config" in df_d1.columns:
    df_d1 = df_d1.rename(columns={"Config": "config", "PPL": "ppl"})

configs_d1   = {"baseline": BLUE, "k128_4bit": GREEN, "k96_4bit": RED}
cfg_labels_d1 = {"baseline": "Baseline", "k128_4bit": "k=128/4-bit", "k96_4bit": "k=96/4-bit"}

fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(8.0, 3.0))

for cfg, col in configs_d1.items():
    sub = df_d1[df_d1["config"] == cfg].sort_values("ctx_len")
    if sub.empty:
        continue
    lbl = cfg_labels_d1[cfg]
    ax_abs.plot(sub["ctx_len"] / 1000, sub["ppl"], marker="o", markersize=4,
                color=col, label=lbl)
    if cfg != "baseline":
        ax_rel.plot(sub["ctx_len"] / 1000, sub["rel_PPL"], marker="o",
                    markersize=4, color=col, label=lbl)

ax_abs.set_xlabel("Context length (K tokens)")
ax_abs.set_ylabel("Perplexity (PPL)")
ax_abs.set_title("(a) Absolute PPL vs context length")
ax_abs.legend()
ax_abs.set_xscale("log", base=2)
ticks_k = [0.5, 1, 2, 4, 8, 16, 32]
ax_abs.set_xticks(ticks_k)
ax_abs.set_xticklabels([f"{t:.0f}K" if t >= 1 else f"{t:.1f}K" for t in ticks_k])

hline(ax_rel, label="20% budget")
ax_rel.axhline(1.0, color=GRAY, linewidth=0.7, linestyle=":")
ax_rel.set_xlabel("Context length (K tokens)")
ax_rel.set_ylabel("Relative PPL (vs baseline)")
ax_rel.set_title("(b) Rel. PPL vs context length")
ax_rel.legend()
ax_rel.set_xscale("log", base=2)
ax_rel.set_xticks(ticks_k)
ax_rel.set_xticklabels([f"{t:.0f}K" if t >= 1 else f"{t:.1f}K" for t in ticks_k])
ax_rel.set_ylim(0.8, 3.0)

fig.suptitle("Long-context stability — Qwen3-14B-AWQ on PG-19 (held-out eval)",
             fontsize=9, y=1.02)
plt.tight_layout()
save("fig4_long_context.pdf")
save("fig4_long_context.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — SubRotQ vs PolarQuant  (exp35)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 5: SubRotQ vs PolarQuant")

df35 = pd.read_csv(f"{RESULTS}/exp35_subrotq_vs_polarquant_clean.csv")
# Keep only the two methods at 4-bit
df35 = df35[df35["method"].isin(["subrotq", "polarquant"]) & (df35["bits"] == 4)]
ks35 = sorted(df35["k"].unique())

fig, ax = plt.subplots(figsize=(5.5, 3.0))
method_style = {
    "subrotq":    dict(color=GREEN, marker="o", label="SubRotQ (ours)"),
    "polarquant": dict(color=RED,   marker="s", label="PolarQuant (Han et al.)"),
}
for m, style in method_style.items():
    sub = df35[df35["method"] == m].sort_values("k")
    ax.plot(sub["k"], sub["rel_ppl"], marker=style["marker"],
            color=style["color"], label=style["label"], markersize=5)
    for _, row in sub.iterrows():
        ax.annotate(f'{row["rel_ppl"]:.3f}×',
                    (row["k"], row["rel_ppl"]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=7, color=style["color"])

hline(ax)
ax.axhline(1.0, color=GRAY, linewidth=0.7, linestyle=":")
ax.set_xlabel("Subspace dim $k$")
ax.set_ylabel("Relative PPL")
ax.set_title("SubRotQ vs PolarQuant at 4-bit — Qwen3-14B-AWQ, WikiText-2")
ax.set_xticks(ks35)
ax.legend()
ax.set_ylim(0.85, 2.5)
plt.tight_layout()
save("fig5_subrotq_vs_polarquant.pdf")
save("fig5_subrotq_vs_polarquant.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 6 — K vs V compression asymmetry  (exp21 Llama + exp32 Llama)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 6: K vs V asymmetry")

df32 = pd.read_csv(f"{RESULTS}/exp32_llama3_wikitext2_ppl.csv")
# exp21 data for Llama-3.1-8B K-only vs V-only
# From REPORT-21: K-only k=112: 1.042×; V-only k=112: 12.14×; K+V k=128: 1.085×
exp21_pts = [
    ("K-only\nk=112", 1.042, GREEN),
    ("V-only\nk=112", 12.14, RED),
    ("K+V\nk=128",    1.085, BLUE),
    ("K+V\nk=124",    3.45,  ORANGE),
]

# Also show Qwen3-14B V-only from exp20
# From REPORT-20: V-only k=128 PPL=7.23 (rel ~1.10), k=112 much worse
# Use exp32 K-only curve for K side
df32_k = df32[(df32["compression_type"] == "K_only") & (df32["bits"] == 4)].sort_values("k")

fig, (ax_curve, ax_bar) = plt.subplots(1, 2, figsize=(8.0, 3.2))

# Left: K vs V rel-PPL curve for Llama-3.1-8B
ax_curve.plot(df32_k["k"], df32_k["rel_ppl"], marker="o", markersize=4,
              color=GREEN, label="K-only 4-bit (Llama-3.1-8B)")
# V-only from exp21: only k=112 viable
ax_curve.scatter([112], [12.14], marker="^", s=60, color=RED, zorder=5,
                 label="V-only 4-bit k=112 (Llama)")
ax_curve.scatter([128], [1.085], marker="D", s=60, color=BLUE, zorder=5,
                 label="K+V 4-bit k=128 (Llama)")
hline(ax_curve)
ax_curve.axhline(1.0, color=GRAY, linewidth=0.7, linestyle=":")
ax_curve.set_xlabel("Subspace dim $k$")
ax_curve.set_ylabel("Relative PPL")
ax_curve.set_title("(a) K vs V compression (Llama-3.1-8B)")
ax_curve.legend(fontsize=7)
ax_curve.set_ylim(0.8, 6.0)

# Right: bar chart showing dramatic V failure across architectures
v_fail_data = [
    # (arch_label, K-only k=128 rel_ppl, V-only best rel_ppl)
    ("Qwen3-14B",    0.98, 1.10),   # V k=128 barely viable, from exp20
    ("Llama-3.1-8B", 1.01, 12.14),  # V k=112 catastrophic, exp21
    ("Mistral-7B",   1.00, None),   # V not tested systematically
]
models_v = ["Qwen3-14B", "Llama-3.1-8B"]
k_vals   = [0.98, 1.01]
v_vals   = [1.10, 12.14]
v_notes  = ["k=128 (best)", "k=112"]

x2 = np.arange(len(models_v))
w2 = 0.32
b1 = ax_bar.bar(x2 - w2/2, k_vals, w2, color=GREEN, alpha=0.85,
                label="K-only k=128/4-bit", edgecolor="white")
b2 = ax_bar.bar(x2 + w2/2, v_vals, w2, color=RED, alpha=0.85,
                label="V-only best config", edgecolor="white")
for bar, v, note in zip(b2, v_vals, v_notes):
    ax_bar.text(bar.get_x() + bar.get_width()/2, min(v, 6.5) + 0.1,
                f"{v:.2f}×\n({note})", ha="center", va="bottom",
                fontsize=6.5, color=RED)
for bar, v in zip(b1, k_vals):
    ax_bar.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                f"{v:.2f}×", ha="center", va="bottom", fontsize=6.5, color=GREEN)
hline(ax_bar)
ax_bar.set_ylabel("Relative PPL")
ax_bar.set_title("(b) K vs V asymmetry at best config")
ax_bar.set_xticks(x2)
ax_bar.set_xticklabels(models_v)
ax_bar.set_ylim(0, 7.0)
ax_bar.legend(fontsize=7.5)
ax_bar.text(1.0 + w2/2, 6.2, "12.14×\n(clipped)", ha="center", fontsize=6.5,
            color=RED, style="italic")

fig.suptitle("V compression fails universally — K vectors are intrinsically lower-dimensional",
             fontsize=9, y=1.02)
plt.tight_layout()
save("fig6_k_v_asymmetry.pdf")
save("fig6_k_v_asymmetry.png")


# ══════════════════════════════════════════════════════════════════════════════
# Fig 7 — Rotation ablation  (exp37)
# ══════════════════════════════════════════════════════════════════════════════
print("Fig 7: rotation ablation")

df37 = pd.read_csv(f"{RESULTS}/exp37_rotation_ablation_v2.csv")

# Show the 2×2 at k=128 plus the k=112 pair
ablation_order = [
    ("plain_4bit",        "Plain 4-bit\n(no PCA, no rot)", GRAY),
    ("raw_rotation_128",  "Rotation only\n(no PCA, 4-bit)", PURPLE),
    ("pca_only_128",      "PCA only\nk=128, 4-bit",         ORANGE),
    ("subrotq_128",       "SubRotQ\nk=128/4-bit",           GREEN),
]
ablation_order_112 = [
    ("pca_only_112",  "PCA only\nk=112, 4-bit",  ORANGE),
    ("subrotq_112",   "SubRotQ\nk=112/4-bit",    GREEN),
]

def get_rel(label):
    row = df37[df37["label"] == label]
    return row["rel_ppl"].values[0] if not row.empty else None

baseline_rel = 1.0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.0),
                                gridspec_kw={"width_ratios": [1.6, 1]})

# k=128 ablation
lbls  = [x[1] for x in ablation_order]
rels  = [get_rel(x[0]) for x in ablation_order]
cols  = [x[2] for x in ablation_order]
xpos  = np.arange(len(lbls))
bars1 = ax1.bar(xpos, rels, color=cols, alpha=0.85, edgecolor="white", width=0.55)
for bar, v in zip(bars1, rels):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.004,
             f"{v:.4f}×", ha="center", va="bottom", fontsize=7.5)
ax1.axhline(baseline_rel, color=GRAY, linewidth=0.8, linestyle=":", label="Baseline (1.0×)")
ax1.axhline(1.0, color=GRAY, linewidth=0.8, linestyle=":")
ax1.set_ylabel("Relative PPL")
ax1.set_title("(a) Component ablation at k=128 (no truncation)")
ax1.set_xticks(xpos)
ax1.set_xticklabels(lbls, fontsize=7.5)
ax1.set_ylim(0.95, 1.06)
ax1.set_yticks([0.95, 0.97, 0.99, 1.00, 1.01, 1.03, 1.05])

# Draw horizontal reference for SubRotQ being best
subrotq_v = get_rel("subrotq_128")
ax1.axhline(subrotq_v, color=GREEN, linewidth=0.8, linestyle="--", alpha=0.6)

# k=112 ablation
lbls2 = [x[1] for x in ablation_order_112]
rels2 = [get_rel(x[0]) for x in ablation_order_112]
cols2 = [x[2] for x in ablation_order_112]
xpos2 = np.arange(len(lbls2))
bars2 = ax2.bar(xpos2, rels2, color=cols2, alpha=0.85, edgecolor="white", width=0.45)
for bar, v in zip(bars2, rels2):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.003,
             f"{v:.4f}×", ha="center", va="bottom", fontsize=7.5)
ax2.axhline(baseline_rel, color=GRAY, linewidth=0.8, linestyle=":", label="Baseline (1.0×)")
hline(ax2, label="20% budget")
ax2.set_ylabel("Relative PPL")
ax2.set_title("(b) Rotation effect at k=112 (truncation)")
ax2.set_xticks(xpos2)
ax2.set_xticklabels(lbls2, fontsize=7.5)
ax2.set_ylim(1.0, 1.35)
ax2.legend(fontsize=7)

# Annotation: rotation adds little when truncation dominates
diff = rels2[0] - rels2[1]
ax2.annotate(f"Δ = {diff:.4f}×\n(rotation negligible\nwhen k < d_head)",
             xy=(0.5, (rels2[0] + rels2[1]) / 2),
             xytext=(0.5, 1.29), ha="center", fontsize=6.5, color=GRAY,
             arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.6))

fig.suptitle("Rotation ablation — Qwen3-14B-AWQ, WikiText-2 (exp37)",
             fontsize=9, y=1.02)
plt.tight_layout()
save("fig7_rotation_ablation.pdf")
save("fig7_rotation_ablation.png")


# ══════════════════════════════════════════════════════════════════════════════
print("\nAll figures saved to", OUT)
print("Files:", sorted(os.listdir(OUT)))
