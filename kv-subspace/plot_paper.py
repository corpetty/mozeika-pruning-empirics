#!/usr/bin/env python3
"""
plot_paper.py — Generate all paper figures from experimental results.

Produces 6 figures into results/figures/:
  fig1_truncation_vs_quantization.png  — The core finding: k beats bits
  fig2_pareto_frontier.png             — PPL-vs-compression Pareto across all configs
  fig3_long_context_stability.png      — Relative PPL held flat to 40K ctx
  fig4_cross_arch.png                  — K and V comparison across architectures
  fig5_layer_sensitivity.png           — Per-layer sensitivity heatmap + adaptive vs uniform
  fig6_kv_asymmetry.png               — The K/V asymmetry: why V compression fails

All figures use a consistent style: clean white background, no chartjunk,
colorblind-friendly palette, bold readable labels.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS = Path("/home/petty/pruning-research/kv-subspace/results")
FIGS    = RESULTS / "figures"
FIGS.mkdir(exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        13,
    "axes.labelsize":   14,
    "axes.titlesize":   15,
    "axes.titleweight": "bold",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "legend.fontsize":  12,
    "legend.framealpha":0.9,
    "figure.dpi":       150,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
})

# Colorblind-safe palette (Wong 2011)
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
PURPLE = "#CC79A7"
SKY    = "#56B4E9"
YELLOW = "#F0E442"
BLACK  = "#000000"

VIABILITY_LINE = 1.20   # 20% PPL degradation budget
SAFE_ZONE_COLOR = "#e8f5e9"


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Truncation vs Quantization: the core finding
# ─────────────────────────────────────────────────────────────────────────────
def fig1_truncation_vs_quantization():
    df = pd.read_csv(RESULTS / "bitrate_k_sweep.csv")

    # Mean rel_ppl per (k, nbits_K) — exclude baseline rows
    df = df[df["config"] != "baseline"].copy()
    agg = df.groupby(["k_K", "nbits_K"])["rel_ppl"].mean().reset_index()

    k_vals    = sorted(agg["k_K"].unique())     # [64, 96, 112, 128]
    bit_vals  = sorted(agg["nbits_K"].unique()) # [4, 6, 8, 16]
    bit_colors = {4: RED, 6: ORANGE, 8: GREEN, 16: BLUE}
    bit_labels = {4: "4-bit", 6: "6-bit", 8: "8-bit", 16: "16-bit"}

    fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(13, 5),
                                            gridspec_kw={"width_ratios": [2, 1]})

    # Left: full view with annotation
    for bits in bit_vals:
        sub = agg[agg["nbits_K"] == bits].sort_values("k_K")
        ax_main.plot(sub["k_K"], sub["rel_ppl"],
                     marker="o", linewidth=2.5, markersize=8,
                     color=bit_colors[bits], label=bit_labels[bits])

    ax_main.axhline(VIABILITY_LINE, color="gray", linestyle="--", linewidth=1.5,
                    label="20% budget")
    ax_main.axhspan(1.0, VIABILITY_LINE, alpha=0.08, color=GREEN)

    # Annotate the smoking-gun contrast
    ax_main.annotate("k=64/16-bit\n(pure truncation)\n2.49×",
                     xy=(64, 2.49), xytext=(72, 2.7),
                     fontsize=11, color=BLUE,
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    ax_main.annotate("k=128/4-bit\n(zero truncation)\n1.05×",
                     xy=(128, 1.05), xytext=(108, 1.4),
                     fontsize=11, color=RED,
                     arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    ax_main.set_xlabel("Subspace dimension k  (d_head = 128)")
    ax_main.set_ylabel("Relative PPL  (compressed / baseline)")
    ax_main.set_title("Fig 1a — Truncation Error Dominates Quantization Noise")
    ax_main.set_xticks(k_vals)
    ax_main.set_ylim(0.9, 4.2)
    ax_main.legend(title="Bit depth", loc="upper right")

    # Right: zoomed k=96-128 to show bit lines converge
    for bits in bit_vals:
        sub = agg[(agg["nbits_K"] == bits) & (agg["k_K"] >= 96)].sort_values("k_K")
        ax_zoom.plot(sub["k_K"], sub["rel_ppl"],
                     marker="o", linewidth=2.5, markersize=8,
                     color=bit_colors[bits], label=bit_labels[bits])

    ax_zoom.axhline(VIABILITY_LINE, color="gray", linestyle="--", linewidth=1.5)
    ax_zoom.axhspan(1.0, VIABILITY_LINE, alpha=0.08, color=GREEN)
    ax_zoom.set_xlabel("Subspace dimension k")
    ax_zoom.set_ylabel("Relative PPL")
    ax_zoom.set_title("Fig 1b — Zoom: k ≥ 96")
    ax_zoom.set_xticks([96, 112, 128])
    ax_zoom.set_ylim(0.95, 1.45)
    ax_zoom.text(97, 1.18, "← viable\n   region", fontsize=10, color="gray")

    plt.tight_layout(pad=2)
    out = FIGS / "fig1_truncation_vs_quantization.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Pareto Frontier: PPL vs Compression Ratio
# ─────────────────────────────────────────────────────────────────────────────
def fig2_pareto_frontier():
    df = pd.read_csv(RESULTS / "bitrate_k_sweep.csv")
    df = df[df["config"] != "baseline"].copy()
    agg = df.groupby(["k_K", "nbits_K"]).agg(
        rel_ppl=("rel_ppl", "mean"),
        cr=("compression_ratio", "first")
    ).reset_index()

    # Also load cross-arch data for Mistral/Phi
    ca = pd.read_csv(RESULTS / "cross_arch_results.csv")
    ca = ca[ca["k"].notna() & ca["n_bits"].notna() & (ca["n_bits"] == 4)].copy()
    ca_agg = ca.groupby(["model", "architecture", "k", "k_frac"]).agg(
        rel_ppl=("rel_ppl", "mean"),
        cr=("compression_ratio", "first")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    bit_markers = {4: "o", 6: "s", 8: "^", 16: "D"}
    bit_colors  = {4: RED, 6: ORANGE, 8: GREEN, 16: BLUE}

    # Qwen3-14B sweep
    for bits, grp in agg.groupby("nbits_K"):
        grp = grp.sort_values("cr")
        ax.scatter(grp["cr"], grp["rel_ppl"],
                   marker=bit_markers[bits], s=90,
                   color=bit_colors[bits], zorder=3,
                   label=f"Qwen3-14B {bits}-bit")

    # Connect k values at 4-bit with a line to show the efficient frontier
    front = agg[agg["nbits_K"] == 4].sort_values("cr", ascending=False)
    ax.plot(front["cr"], front["rel_ppl"],
            color=RED, linewidth=2, zorder=2, alpha=0.6, linestyle="-")

    # Mistral/Phi at 4-bit
    arch_style = {"Mistral": (BLUE, "P", "Mistral-7B"), "Phi3": (GREEN, "X", "Phi-4")}
    for arch, (color, mark, label) in arch_style.items():
        sub = ca_agg[ca_agg["architecture"] == arch]
        if len(sub) > 0:
            ax.scatter(sub["cr"], sub["rel_ppl"],
                       marker=mark, s=130, color=color, zorder=4,
                       edgecolors="black", linewidth=0.8, label=f"{label} 4-bit")

    # Viability shading
    ax.axhline(VIABILITY_LINE, color="gray", linestyle="--", linewidth=1.5)
    ax.axhspan(1.0, VIABILITY_LINE, alpha=0.08, color=GREEN, label="Viable (<20% PPL)")

    # Label sweet spots
    sweet = agg[(agg["k_K"] == 112) & (agg["nbits_K"] == 4)]
    if len(sweet):
        r = sweet.iloc[0]
        ax.annotate("k=112/4-bit\n4.27× CR, 1.14× PPL",
                    xy=(r["cr"], r["rel_ppl"]), xytext=(r["cr"]+0.3, r["rel_ppl"]+0.3),
                    fontsize=11, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    sweet2 = agg[(agg["k_K"] == 128) & (agg["nbits_K"] == 4)]
    if len(sweet2):
        r = sweet2.iloc[0]
        ax.annotate("k=128/4-bit\n4.00× CR, 1.05× PPL",
                    xy=(r["cr"], r["rel_ppl"]), xytext=(r["cr"]-1.2, r["rel_ppl"]+0.15),
                    fontsize=11, color=RED,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1.2))

    ax.set_xlabel("Compression Ratio  (× vs fp16 baseline)")
    ax.set_ylabel("Relative PPL  (compressed / baseline)")
    ax.set_title("Fig 2 — PPL–Compression Pareto Frontier")
    ax.set_xlim(0.9, 6.2)
    ax.set_ylim(0.9, 4.5)
    ax.legend(ncol=2, fontsize=10)
    ax.text(1.0, 0.97, "← better quality     more compression →",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color="gray", style="italic")

    plt.tight_layout()
    out = FIGS / "fig2_pareto_frontier.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Long-Context Stability
# ─────────────────────────────────────────────────────────────────────────────
def fig3_long_context_stability():
    df = pd.read_csv(RESULTS / "long_context_ppl.csv")

    configs = {
        "k128_4bit": (BLUE,   "o", "k=128/4-bit  (4.00×)",  "solid"),
        "k112_4bit": (GREEN,  "s", "k=112/4-bit  (4.27×)",  "solid"),
        "k96_4bit":  (ORANGE, "^", "k=96/4-bit   (4.57×)",  "solid"),
        "k64_4bit":  (RED,    "D", "k=64/4-bit   (5.33×)",  "dashed"),
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    for cfg, (color, mark, label, ls) in configs.items():
        sub = df[df["config"] == cfg].sort_values("ctx_len")
        ax.plot(sub["ctx_len"], sub["relative_ppl"],
                marker=mark, color=color, linewidth=2.5, markersize=8,
                linestyle=ls, label=label)

    ax.axhline(VIABILITY_LINE, color="gray", linestyle=":", linewidth=1.5)
    ax.axhspan(1.0, VIABILITY_LINE, alpha=0.07, color=GREEN)

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1024)}K" if x >= 1024 else str(int(x))
    ))
    ax.set_xticks([512, 2048, 8192, 16384, 32768, 40960])

    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Relative PPL  (compressed / baseline)")
    ax.set_title("Fig 3 — Long-Context Stability: k=128/4-bit Flat to 40K Tokens")
    ax.legend(loc="upper right")

    # Callout: k128 is dead flat
    ax.annotate("Flat: 1.05–1.11×\nacross all contexts",
                xy=(40960, 1.09), xytext=(12000, 1.22),
                fontsize=11, color=BLUE,
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))
    # Callout: k64 improves
    ax.annotate("Improves at longer ctx\n(subspace stays representative)",
                xy=(40960, 4.26), xytext=(3000, 4.5),
                fontsize=11, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

    plt.tight_layout()
    out = FIGS / "fig3_long_context_stability.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Cross-Architecture: K vs V asymmetry + model comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig4_cross_arch():
    """
    Two-panel figure:
    Left:  K-only rel PPL by model/arch at k=112/4-bit (all look good)
    Right: V-only rel PPL — catastrophic across ALL architectures including Llama
    """
    # K compression from cross_arch_results (Mistral, Phi3, Qwen3-14B baseline)
    ca = pd.read_csv(RESULTS / "cross_arch_results.csv")
    ca = ca[ca["k"].notna() & (ca["n_bits"] == 4) & (ca["k"] == 112)].copy()
    ca_agg = ca.groupby(["model", "architecture"])["rel_ppl"].mean().reset_index()

    # Also add Llama K-only from exp21
    exp21 = pd.read_csv(RESULTS / "exp21_llama3_validation.csv")
    llama_k_only = exp21[exp21["subexp"] == "B_k_only"]["rel_ppl"].mean()
    llama_v_only = exp21[exp21["subexp"] == "B_v_only"]["rel_ppl"].mean()

    # V-only from exp20 and exp21
    # Qwen3-14B V-only at k=112 from exp20
    exp20 = pd.read_csv(RESULTS / "exp20_v_threshold.csv")
    qwen_v_k112 = exp20[exp20["k_V"] == 112]["rel_ppl"].mean()
    qwen_v_k128 = exp20[exp20["k_V"] == 128]["rel_ppl"].mean()

    fig, (ax_k, ax_v) = plt.subplots(1, 2, figsize=(13, 5))

    # === Left: K compression across architectures ===
    arch_colors = {"Mistral": SKY, "Phi3": GREEN, "Qwen3": ORANGE}
    k_models = []
    k_ppl = []
    k_colors = []

    # Add from cross_arch
    model_map = {
        "Mistral-7B-v0.3": ("Mistral-7B\n(Mistral)", "Mistral"),
        "Phi-4-AWQ":        ("Phi-4\n(Phi3)",         "Phi3"),
    }
    for _, row in ca_agg.iterrows():
        if row["model"] in model_map:
            label, arch = model_map[row["model"]]
            k_models.append(label)
            k_ppl.append(row["rel_ppl"])
            k_colors.append(arch_colors[arch])

    # Qwen3-14B from perplexity_results
    pr = pd.read_csv(RESULTS / "perplexity_results.csv")
    qwen_k112 = pr[pr["config"] == "k112_4bit"]["relative_ppl"].mean()
    k_models.append("Qwen3-14B\n(Qwen3)")
    k_ppl.append(qwen_k112)
    k_colors.append(arch_colors["Qwen3"])

    # Llama-3.1 K-only
    k_models.append("Llama-3.1-8B\n(LLaMA)")
    k_ppl.append(llama_k_only)
    k_colors.append(PURPLE)

    bars_k = ax_k.bar(k_models, k_ppl, color=k_colors, width=0.5, zorder=3)
    ax_k.axhline(VIABILITY_LINE, color="gray", linestyle="--", linewidth=1.5, label="20% budget")
    ax_k.axhspan(1.0, VIABILITY_LINE, alpha=0.08, color=GREEN)
    ax_k.set_ylim(0.9, 1.6)
    ax_k.set_ylabel("Relative PPL  (compressed / baseline)")
    ax_k.set_title("K Compression (k=112/4-bit)\n— All architectures viable")
    ax_k.set_xlabel("Model / Architecture")

    for bar, val in zip(bars_k, k_ppl):
        ax_k.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                  f"{val:.2f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_k.text(0.5, 0.93, "✓ Universally viable", transform=ax_k.transAxes,
              ha="center", fontsize=13, color=GREEN, fontweight="bold")

    # === Right: V compression — show the cliff ===
    v_k_vals = [64, 80, 96, 104, 108, 112, 116, 120, 124, 128]
    exp20_scan = exp20[exp20["experiment"] == "v_k_scan"].sort_values("k_V")

    qwen_v_ppl  = exp20_scan["rel_ppl"].values
    qwen_v_k    = exp20_scan["k_V"].values

    # Llama V data from exp21 C-subexp
    exp21_c = exp21[exp21["subexp"] == "C_v_threshold"].sort_values("k_V")
    llama_v_k   = exp21_c["k_V"].values
    llama_v_ppl = exp21_c["rel_ppl"].values

    ax_v.plot(qwen_v_k, qwen_v_ppl, color=ORANGE, marker="o", linewidth=2.5,
              markersize=8, label="Qwen3-14B (has QK-norm)")
    ax_v.plot(llama_v_k, llama_v_ppl, color=PURPLE, marker="s", linewidth=2.5,
              markersize=8, label="Llama-3.1-8B (no QK-norm)")

    # K-only reference lines
    qwen_k_ref = qwen_k112
    llama_k_ref = llama_k_only
    ax_v.axhline(qwen_k_ref,   color=ORANGE, linestyle=":", linewidth=1.5, alpha=0.7,
                 label=f"Qwen3 K-only ref ({qwen_k_ref:.2f}×)")
    ax_v.axhline(llama_k_ref,  color=PURPLE, linestyle=":", linewidth=1.5, alpha=0.7,
                 label=f"Llama K-only ref ({llama_k_ref:.2f}×)")
    ax_v.axhline(VIABILITY_LINE, color="gray", linestyle="--", linewidth=1.5)

    ax_v.set_ylim(0.9, 6.5)
    ax_v.set_xlabel("V subspace dimension k_V  (d_head = 128)")
    ax_v.set_ylabel("Relative PPL  (compressed / baseline)")
    ax_v.set_title("V Compression — Architecture-Independent Failure\n(QK-norm hypothesis rejected)")
    ax_v.legend(fontsize=10, loc="upper right")
    ax_v.text(0.5, 0.93, "✗ Fails at all k < 128, all architectures",
              transform=ax_v.transAxes, ha="center", fontsize=12, color=RED, fontweight="bold")

    plt.tight_layout(pad=2)
    out = FIGS / "fig4_cross_arch.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Layer Sensitivity + Adaptive vs Uniform
# ─────────────────────────────────────────────────────────────────────────────
def fig5_layer_sensitivity():
    sens = pd.read_csv(RESULTS / "exp16_layer_sensitivity.csv")
    adap = pd.read_csv(RESULTS / "exp18_adaptive_policy.csv")

    fig, (ax_bar, ax_policy) = plt.subplots(1, 2, figsize=(14, 5))

    # === Left: Per-layer sensitivity bar chart ===
    layers = sens["layer_idx"].values
    deltas = sens["ppl_delta"].values
    colors = [RED if d > 0.2 else (GREEN if d < 0 else SKY) for d in deltas]

    ax_bar.bar(layers, deltas, color=colors, width=0.8, zorder=3)
    ax_bar.axhline(0, color="black", linewidth=0.8)
    ax_bar.axhline(0.2, color=RED, linestyle="--", linewidth=1.2, alpha=0.6,
                   label="High sensitivity threshold")

    # Annotate spikes
    for li, d in zip(layers, deltas):
        if d > 0.3:
            ax_bar.text(li, d + 0.02, f"L{li}", ha="center", fontsize=9,
                        color=RED, fontweight="bold")
        elif d < -0.05:
            ax_bar.text(li, d - 0.05, f"L{li}", ha="center", fontsize=9, color=GREEN)

    # Legend patches
    ax_bar.legend(handles=[
        mpatches.Patch(color=RED,   label="High sensitivity (protect)"),
        mpatches.Patch(color=SKY,   label="Normal sensitivity"),
        mpatches.Patch(color=GREEN, label="Negative (free to compress)"),
    ], fontsize=10, loc="upper left")

    ax_bar.set_xlabel("Layer index")
    ax_bar.set_ylabel("PPL delta vs baseline\n(compress this layer alone)")
    ax_bar.set_title("Fig 5a — Layer Sensitivity Profile\n(ablation at k=64/4-bit)")
    ax_bar.set_xlim(-0.5, 39.5)

    # === Right: Adaptive vs uniform PPL at each budget ===
    # budget_k column, uniform_rel_ppl vs rank_rel_ppl
    budgets = adap["budget_k"].values
    uniform_ppl = adap["uniform_rel_ppl"].values
    rank_ppl    = adap["rank_rel_ppl"].values

    x = np.arange(len(budgets))
    width = 0.35
    bars_u = ax_policy.bar(x - width/2, uniform_ppl, width, color=SKY,
                            label="Uniform k (baseline)", zorder=3)
    bars_r = ax_policy.bar(x + width/2, rank_ppl, width, color=ORANGE,
                            label="Rank-proportional adaptive", zorder=3)

    ax_policy.axhline(VIABILITY_LINE, color="gray", linestyle="--", linewidth=1.5)
    ax_policy.set_xticks(x)
    ax_policy.set_xticklabels([f"budget k={b}" for b in budgets], fontsize=10, rotation=15)
    ax_policy.set_ylabel("Relative PPL  (lower = better)")
    ax_policy.set_title("Fig 5b — Adaptive K-Scheduling\nvs Uniform at Same Memory Budget")
    ax_policy.legend(fontsize=11)
    ax_policy.set_ylim(0.9, 2.2)

    # Annotate improvement at budget=112 (the key result)
    if len(budgets) >= 5:
        idx = list(budgets).index(112) if 112 in budgets else -1
        if idx >= 0:
            u_val = uniform_ppl[idx]
            r_val = rank_ppl[idx]
            ax_policy.annotate(f"−{(u_val-r_val)*100:.1f}% PPL\nsame memory",
                               xy=(x[idx] + width/2, r_val),
                               xytext=(x[idx] + 0.6, r_val + 0.12),
                               fontsize=10, color=ORANGE,
                               arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))

    plt.tight_layout(pad=2)
    out = FIGS / "fig5_layer_sensitivity.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — The K/V Asymmetry: compression ratio vs quality, K-only vs V-only
# ─────────────────────────────────────────────────────────────────────────────
def fig6_kv_asymmetry():
    """
    Shows K and V on the same compression-ratio axis so the reader can see
    immediately: K is in the green zone, V is catastrophically above it.
    This is the 'hero' figure that explains the method's main constraint.
    """
    exp20 = pd.read_csv(RESULTS / "exp20_v_threshold.csv")
    exp21 = pd.read_csv(RESULTS / "exp21_llama3_validation.csv")

    # Qwen3-14B K data from bitrate sweep at 4-bit
    bk = pd.read_csv(RESULTS / "bitrate_k_sweep.csv")
    bk = bk[bk["nbits_K"] == 4].copy()
    qwen_k = bk.groupby("k_K").agg(rel_ppl=("rel_ppl","mean"), cr=("compression_ratio","first")).reset_index()

    # Qwen3-14B V data from exp20
    qwen_v_scan = exp20[exp20["experiment"] == "v_k_scan"].sort_values("compression_ratio")

    # Llama K from exp21 subexp A (K+V at matched k — but we want K-only)
    # Use B_k_only as single point
    llama_k_only_ppl = exp21[exp21["subexp"] == "B_k_only"]["rel_ppl"].mean()
    llama_k_only_cr  = exp21[exp21["subexp"] == "B_k_only"]["compression_ratio"].mean()
    # Llama V from exp21 subexp C
    llama_v = exp21[exp21["subexp"] == "C_v_threshold"].sort_values("compression_ratio")

    fig, ax = plt.subplots(figsize=(11, 6))

    # Qwen3 K — solid blue line
    ax.plot(qwen_k["cr"], qwen_k["rel_ppl"],
            color=BLUE, marker="o", linewidth=3, markersize=9,
            label="K compression — Qwen3-14B (has QK-norm)")

    # Qwen3 V — solid orange line
    ax.plot(qwen_v_scan["compression_ratio"], qwen_v_scan["rel_ppl"],
            color=ORANGE, marker="s", linewidth=3, markersize=9,
            label="V compression — Qwen3-14B (has QK-norm)")

    # Llama K — single point
    ax.scatter([llama_k_only_cr], [llama_k_only_ppl],
               color=BLUE, marker="P", s=200, zorder=5,
               edgecolors="black", linewidth=1.2,
               label="K compression — Llama-3.1-8B (no QK-norm)")

    # Llama V — dashed purple line
    ax.plot(llama_v["compression_ratio"], llama_v["rel_ppl"],
            color=PURPLE, marker="X", linewidth=3, markersize=9, linestyle="--",
            label="V compression — Llama-3.1-8B (no QK-norm)")

    # Viability zone
    ax.axhline(VIABILITY_LINE, color="gray", linestyle="--", linewidth=1.5, label="20% budget")
    ax.axhspan(1.0, VIABILITY_LINE, alpha=0.10, color=GREEN)

    ax.set_xlabel("Compression Ratio  (× vs fp16 uncompressed)")
    ax.set_ylabel("Relative PPL  (compressed / baseline)")
    ax.set_title("Fig 6 — K/V Asymmetry: K Compresses, V Resists — Across All Architectures")
    ax.set_xlim(1.0, 8.5)
    ax.set_ylim(0.9, 7.5)
    ax.legend(fontsize=11, loc="upper left")

    # Annotation: K sweet spot
    k112_row = qwen_k[qwen_k["k_K"] == 112]
    if len(k112_row):
        r = k112_row.iloc[0]
        ax.annotate("K sweet spot:\nk=112/4-bit\n4.27× CR, 1.14× PPL",
                    xy=(r["cr"], r["rel_ppl"]),
                    xytext=(r["cr"]+0.4, r["rel_ppl"]+0.5),
                    fontsize=11, color=BLUE,
                    arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.5))

    # Annotation: V cliff
    v_k128 = qwen_v_scan[qwen_v_scan["k_V"] == 128] if "k_V" in qwen_v_scan.columns else pd.DataFrame()
    ax.text(4.8, 5.2, "V cliff:\nEven at full rank (k=128)\nV barely viable at 4K ctx",
            fontsize=11, color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=ORANGE, alpha=0.9))

    plt.tight_layout()
    out = FIGS / "fig6_kv_asymmetry.png"
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Run all figures
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating paper figures...")
    fig1_truncation_vs_quantization()
    fig2_pareto_frontier()
    fig3_long_context_stability()
    fig4_cross_arch()
    fig5_layer_sensitivity()
    fig6_kv_asymmetry()
    print(f"\nAll figures saved to {FIGS}/")
    print("Files:")
    for f in sorted(FIGS.glob("*.png")):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")
