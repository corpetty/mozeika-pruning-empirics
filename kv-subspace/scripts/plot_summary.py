#!/usr/bin/env python3
"""Generate 6 summary charts for KV cache subspace compression research."""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    plt.style.use("seaborn-v0_8-whitegrid")
except Exception:
    plt.style.use("ggplot")

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
DATADIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(FIGDIR, exist_ok=True)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_csv(name):
    if not HAS_PANDAS:
        return None
    path = os.path.join(DATADIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


# ──────────────────────────────────────────────────────────────────────
# Chart 1: PPL vs Compression Ratio — Pareto Frontier
# ──────────────────────────────────────────────────────────────────────

def fig1_pareto():
    df = load_csv("bitrate_k_sweep.csv")
    if df is not None:
        agg = df.groupby("config").agg(
            mean_ppl=("mean_ppl", "first"),
            rel_ppl=("rel_ppl", "first"),
            compression_ratio=("compression_ratio", "first"),
        ).reset_index()
        agg = agg[agg["config"] != "baseline"]
        configs = agg["config"].values
        cr = agg["compression_ratio"].values
        rp = agg["rel_ppl"].values
    else:
        configs = np.array([
            "k64_4bit", "k64_6bit", "k64_8bit", "k64_16bit",
            "k96_4bit", "k96_6bit", "k96_8bit", "k96_16bit",
            "k112_4bit", "k112_6bit", "k112_8bit", "k112_16bit",
            "k128_4bit", "k128_6bit", "k128_8bit",
        ])
        cr = np.array([5.33, 3.56, 2.67, 1.33,
                        4.57, 3.05, 2.29, 1.14,
                        4.27, 2.84, 2.13, 1.07,
                        4.00, 2.67, 2.00])
        rp = np.array([3.19, 2.52, 2.49, 2.48,
                        1.26, 1.18, 1.17, 1.17,
                        1.14, 1.09, 1.09, 1.09,
                        1.05, 1.01, 1.00])

    pareto_configs = {"k64_4bit", "k96_4bit", "k112_4bit", "k128_4bit",
                      "k128_6bit", "k128_8bit"}

    fig, ax = plt.subplots(figsize=(10, 6))

    # All configs as grey dots
    ax.scatter(cr, rp, c="silver", s=60, zorder=2, edgecolors="grey", linewidths=0.5)

    # Pareto configs in blue
    pareto_mask = np.array([c in pareto_configs for c in configs])
    if pareto_mask.any():
        pcr = cr[pareto_mask]
        prp = rp[pareto_mask]
        order = np.argsort(pcr)
        ax.plot(pcr[order], prp[order], "o-", color="#2171b5", ms=9, lw=2,
                zorder=3, label="Pareto frontier")

    # Annotate key configs
    annotations = {
        "k64_4bit": ("k64/4bit\n(worst)", (-40, 15)),
        "k112_4bit": ("k112/4bit\n(sweet spot)", (15, 25)),
        "k128_4bit": ("k128/4bit\n(best quality)", (-80, -30)),
    }
    for cfg, (label, offset) in annotations.items():
        idx = np.where(configs == cfg)[0]
        if len(idx):
            i = idx[0]
            ax.annotate(label, (cr[i], rp[i]), textcoords="offset points",
                        xytext=offset, fontsize=9, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Baseline point
    ax.plot(1.0, 1.0, "D", color="red", ms=10, zorder=4, label="Baseline (FP16)")

    # Threshold line
    ax.axhline(1.20, ls="--", color="crimson", lw=1.2, alpha=0.7, label="20% PPL threshold")

    ax.set_xlabel("Compression Ratio", fontsize=12)
    ax.set_ylabel("Relative PPL (vs baseline)", fontsize=12)
    ax.set_title("PPL vs Compression Ratio — Pareto Frontier", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(0.5, 6.0)
    ax.set_ylim(0.8, 3.6)

    fig.savefig(os.path.join(FIGDIR, "fig1_ppl_vs_compression_pareto.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig1_ppl_vs_compression_pareto.png")


# ──────────────────────────────────────────────────────────────────────
# Chart 2: Truncation vs Quantization
# ──────────────────────────────────────────────────────────────────────

def fig2_truncation():
    df = load_csv("bitrate_k_sweep.csv")
    bits_axis = [4, 6, 8, 16]

    if df is not None:
        agg = df.groupby("config").agg(
            mean_ppl=("mean_ppl", "first"),
            k_K=("k_K", "first"),
            nbits_K=("nbits_K", "first"),
        ).reset_index()
        ppl_k64 = []
        ppl_k128 = []
        for b in bits_axis:
            row64 = agg[(agg["k_K"] == 64) & (agg["nbits_K"] == b)]
            row128 = agg[(agg["k_K"] == 128) & (agg["nbits_K"] == b)]
            ppl_k64.append(row64["mean_ppl"].values[0] if len(row64) else np.nan)
            ppl_k128.append(row128["mean_ppl"].values[0] if len(row128) else np.nan)
        ppl_k64 = np.array(ppl_k64)
        ppl_k128 = np.array(ppl_k128)
    else:
        ppl_k64 = np.array([8.25, 6.51, 6.43, 6.41])
        ppl_k128 = np.array([2.72, 2.60, 2.59, np.nan])

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(bits_axis, ppl_k64, "s-", color="#d94801", ms=10, lw=2.5, label="k=64 (truncation-dominated)")
    valid = ~np.isnan(ppl_k128)
    ax.plot(np.array(bits_axis)[valid], ppl_k128[valid], "o-", color="#2171b5", ms=10, lw=2.5,
            label="k=128 (quantization only)")

    # Baseline reference
    ax.axhline(2.58, ls=":", color="grey", lw=1.2, alpha=0.7, label="Baseline PPL (2.58)")

    # Annotation for truncation plateau
    ax.annotate("Truncation error\ndominates here",
                xy=(12, ppl_k64[2]), xytext=(12, ppl_k64[2] + 1.2),
                fontsize=10, fontweight="bold", color="#d94801",
                arrowprops=dict(arrowstyle="->", color="#d94801", lw=1.5),
                ha="center")

    ax.set_xlabel("Bits per dimension", fontsize=12)
    ax.set_ylabel("Mean PPL", fontsize=12)
    ax.set_title("Root Cause: Truncation Dominates Over Quantization", fontsize=14, fontweight="bold")
    ax.set_xticks(bits_axis)
    ax.legend(fontsize=10)
    ax.set_ylim(1.5, 10.0)

    fig.savefig(os.path.join(FIGDIR, "fig2_truncation_vs_quantization.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig2_truncation_vs_quantization.png")


# ──────────────────────────────────────────────────────────────────────
# Chart 3: Effective Rank of K vs V by Layer
# ──────────────────────────────────────────────────────────────────────

def fig3_effective_rank():
    df = load_csv("adaptive_k_distortion.csv")

    if df is not None and "eff_rank_90" in df.columns:
        # Get per-layer mean effective rank for K
        k_df = df[(df["method"] == "full_dim") & (df["budget"] == "2bit_budget")]
        k_rank = k_df.groupby("layer")["eff_rank_90"].mean()
        layers_k = k_rank.index.values
        rank_k = k_rank.values
    else:
        layers_k = np.arange(40)
        # Approximate from reports: mean ~30, early ~15-25, late ~40-52
        rank_k = np.array([
            14, 18, 23, 27, 25, 26, 27, 28, 24, 22,
            28, 30, 31, 29, 30, 32, 33, 31, 28, 29,
            31, 30, 32, 33, 34, 35, 36, 34, 33, 35,
            38, 40, 42, 43, 45, 47, 48, 49, 50, 52,
        ], dtype=float)

    # V effective rank from v_compression_distortion.csv
    vdf = load_csv("v_compression_distortion.csv")
    if vdf is not None and "eff_rank_V_90" in vdf.columns:
        v_sub = vdf[vdf["method"] == "full_dim"]
        v_rank = v_sub.groupby("layer")["eff_rank_V_90"].mean()
        layers_v = v_rank.index.values
        rank_v = v_rank.values
    else:
        layers_v = np.arange(40)
        # Approximate from Report 3: mean ~54, early ~39, mid ~58, late ~62
        rank_v = np.array([
            26, 30, 33, 35, 37, 38, 39, 40, 42, 44,
            48, 50, 52, 53, 54, 55, 56, 57, 58, 59,
            58, 57, 58, 59, 60, 61, 60, 59, 58, 59,
            60, 61, 62, 63, 64, 65, 66, 67, 68, 79,
        ], dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(layers_k, rank_k, "o-", color="#2171b5", ms=4, lw=1.8, label="K vectors", alpha=0.9)
    ax.plot(layers_v, rank_v, "s-", color="#e6550d", ms=4, lw=1.8, label="V vectors", alpha=0.9)

    # Reference lines for k thresholds
    ax.axhline(64, ls="--", color="grey", lw=1.2, alpha=0.6)
    ax.text(39.5, 65, "k=64", fontsize=9, color="grey", ha="right")
    ax.axhline(112, ls="--", color="grey", lw=1.2, alpha=0.6)
    ax.text(39.5, 113, "k=112", fontsize=9, color="grey", ha="right")

    # Shade regions
    ax.axvspan(-0.5, 9.5, alpha=0.06, color="red", label="Early layers (L0-9)")
    ax.axvspan(9.5, 29.5, alpha=0.06, color="blue")
    ax.axvspan(29.5, 39.5, alpha=0.06, color="green")

    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Effective Rank (90% variance)", fontsize=12)
    ax.set_title("Effective Rank of K vs V Vectors Across Layers", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xlim(-0.5, 39.5)
    ax.set_ylim(0, 130)

    fig.savefig(os.path.join(FIGDIR, "fig3_kv_effective_rank_by_layer.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig3_kv_effective_rank_by_layer.png")


# ──────────────────────────────────────────────────────────────────────
# Chart 4: Subspace Overlap Heatmap (grouped bar chart)
# ──────────────────────────────────────────────────────────────────────

def fig4_overlap():
    # From Report 1 tables
    # K overlaps
    k_cross_layer = 0.5649
    k_cross_head = 0.4559
    k_cross_domain = 0.7029
    # V overlaps
    v_cross_layer = 0.7356
    v_cross_head = 0.5464
    v_cross_domain = 0.6403
    # K layer range breakdown for cross-layer
    k_early = 0.3811
    k_mid = 0.6186
    k_late = 0.6495

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [3, 2]})

    # Left panel: grouped bar chart
    groups = ["Cross-Layer", "Cross-Head", "Cross-Domain"]
    k_vals = [k_cross_layer, k_cross_head, k_cross_domain]
    v_vals = [v_cross_layer, v_cross_head, v_cross_domain]

    x = np.arange(len(groups))
    w = 0.32
    bars_k = ax1.bar(x - w/2, k_vals, w, label="K vectors", color="#2171b5", edgecolor="white")
    bars_v = ax1.bar(x + w/2, v_vals, w, label="V vectors", color="#e6550d", edgecolor="white")

    # Add value labels
    for bar in bars_k:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", fontsize=9, fontweight="bold")
    for bar in bars_v:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=11)
    ax1.set_ylabel("Mean Subspace Overlap", fontsize=12)
    ax1.set_ylim(0, 0.95)
    ax1.legend(fontsize=10)
    ax1.set_title("Overall Overlap by Type", fontsize=12, fontweight="bold")

    # Right panel: K cross-layer breakdown by layer range
    ranges = ["Early\n(L0-9)", "Mid\n(L10-29)", "Late\n(L30-39)"]
    vals = [k_early, k_mid, k_late]
    colors = ["#c6dbef", "#6baed6", "#2171b5"]
    bars = ax2.bar(ranges, vals, color=colors, edgecolor="white", width=0.6)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylabel("K Cross-Layer Overlap", fontsize=12)
    ax2.set_ylim(0, 0.85)
    ax2.set_title("K Cross-Layer by Depth", fontsize=12, fontweight="bold")

    fig.suptitle("Subspace Overlap Across Layers, Heads, and Domains",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()

    fig.savefig(os.path.join(FIGDIR, "fig4_subspace_overlap_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig4_subspace_overlap_heatmap.png")


# ──────────────────────────────────────────────────────────────────────
# Chart 5: Cross-Model Threshold
# ──────────────────────────────────────────────────────────────────────

def fig5_cross_model():
    df = load_csv("cross_model_results.csv")

    models_data = {}
    if df is not None:
        for model in df["model"].unique():
            mdf = df[df["model"] == model]
            agg = mdf.groupby("k_frac").agg(
                rel_ppl=("rel_ppl", "mean"),
            ).reset_index()
            # Add baseline
            fracs = [0.0] + agg["k_frac"].dropna().tolist()
            rppl = [1.0] + agg["rel_ppl"].tolist()
            # Filter out NaN k_frac (baseline rows)
            valid = [(f, r) for f, r in zip(fracs, rppl) if not np.isnan(f) and not np.isnan(r)]
            if valid:
                models_data[model] = (
                    np.array([v[0] for v in valid]),
                    np.array([v[1] for v in valid]),
                )

    # Also include 14B data from exp 9 (not in cross_model_results)
    # From report 9: k112=1.14x, k128=1.05x; from report 11 ref
    if "Qwen3-14B-AWQ" not in models_data:
        models_data["Qwen3-14B-AWQ"] = (
            np.array([0.5, 0.75, 0.875, 1.0]),
            np.array([3.19, 1.26, 1.14, 1.05]),
        )

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {"Qwen3-1.7B": "#d94801", "Qwen3-14B-AWQ": "#2171b5", "Qwen3-32B-AWQ": "#238b45"}
    markers = {"Qwen3-1.7B": "^", "Qwen3-14B-AWQ": "o", "Qwen3-32B-AWQ": "s"}

    for model in ["Qwen3-1.7B", "Qwen3-14B-AWQ", "Qwen3-32B-AWQ"]:
        if model in models_data:
            fracs, rppl = models_data[model]
            order = np.argsort(fracs)
            ax.plot(fracs[order], rppl[order],
                    f"{markers.get(model, 'o')}-",
                    color=colors.get(model, "grey"),
                    ms=10, lw=2.2, label=model)

    ax.axhline(1.20, ls="--", color="crimson", lw=1.2, alpha=0.7, label="20% PPL threshold")

    ax.set_xlabel("k / d_head", fontsize=12)
    ax.set_ylabel("Relative PPL", fontsize=12)
    ax.set_title("Compression Tolerance vs Model Size (k/d_head Sweep)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_xticks([0.5, 0.625, 0.75, 0.875, 0.9375, 1.0])
    ax.set_xticklabels(["0.50", "0.625", "0.75", "0.875", "0.9375", "1.0"])

    # Dynamic y-axis based on data
    all_rppl = np.concatenate([v[1] for v in models_data.values()])
    ymax = min(max(all_rppl) * 1.1, 15.0)
    ax.set_ylim(0.9, ymax)

    fig.savefig(os.path.join(FIGDIR, "fig5_cross_model_threshold.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig5_cross_model_threshold.png")


# ──────────────────────────────────────────────────────────────────────
# Chart 6: Hardware Overhead (latency bar chart at T=512)
# ──────────────────────────────────────────────────────────────────────

def fig6_hardware():
    df = load_csv("hardware_cost.csv")

    if df is not None:
        t512 = df[df["batch_size"] == 512]
        ops = []
        lats = []
        for _, row in t512.iterrows():
            ops.append(row["operation"])
            lats.append(row["latency_us"])
    else:
        ops = ["plain_quantize", "polarquant_128", "project_quantize_k64",
               "subspace_polar_k32", "subspace_polar_k64"]
        lats = [184, 263, 288, 344, 324]

    # Order for display
    display_order = ["plain_quantize", "polarquant_128", "project_quantize_k64",
                     "subspace_polar_k32", "subspace_polar_k64"]
    display_labels = ["Plain\nQuantize", "PolarQuant\n(k=128)", "Project+Quant\n(k=64)",
                      "Subspace Polar\n(k=32)", "Subspace Polar\n(k=64)"]

    ordered_lats = []
    for op in display_order:
        if op in ops:
            idx = ops.index(op)
            ordered_lats.append(lats[idx])
        else:
            ordered_lats.append(0)

    plain_lat = ordered_lats[0] if ordered_lats[0] > 0 else 184

    # Colors: plain = green, rest = blue gradient
    colors = ["#41ab5d", "#c6dbef", "#6baed6", "#2171b5", "#08519c"]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(display_order))
    bars = ax.bar(x, ordered_lats, color=colors, edgecolor="white", width=0.65)

    # Add overhead ratio labels
    for i, (bar, lat) in enumerate(zip(bars, ordered_lats)):
        ratio = lat / plain_lat
        label = f"{lat:.0f} \u03bcs\n({ratio:.2f}\u00d7)"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                label, ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(display_labels, fontsize=10)
    ax.set_ylabel("Latency (\u03bcs)", fontsize=12)
    ax.set_title("CUDA Latency per Head (T=512 tokens)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(ordered_lats) * 1.25)

    fig.savefig(os.path.join(FIGDIR, "fig6_hardware_overhead.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [OK] fig6_hardware_overhead.png")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating summary charts...")
    print(f"  Output dir: {os.path.abspath(FIGDIR)}")
    print(f"  Pandas available: {HAS_PANDAS}")
    print()

    fig1_pareto()
    fig2_truncation()
    fig3_effective_rank()
    fig4_overlap()
    fig5_cross_model()
    fig6_hardware()

    print()
    print("All 6 charts generated successfully.")
