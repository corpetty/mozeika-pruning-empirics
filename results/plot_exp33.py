"""
Experiment 33 (job 68) — Two-phase neuron+weight pruning plots
784→300→100→10 (LeNet-300-100, MNIST)
Phase 1: Neuron Glauber → 784→122→35→10
Phase 2: Weight Glauber on compact net
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

with open("/home/petty/pruning-research/results/33_phase1_records.json") as f:
    p1 = json.load(f)
with open("/home/petty/pruning-research/results/33_phase2_records.json") as f:
    p2 = json.load(f)

# ---------- extract phase 1 ----------
p1_rounds  = [r["round"] for r in p1]
p1_spar    = [r["sparsity"] for r in p1]
p1_acc     = [r["test_acc"] for r in p1]
p1_k1      = [r["k1"] for r in p1]
p1_k2      = [r["k2"] for r in p1]
p1_energy  = [r["energy"] for r in p1]
p1_gap     = [r["gap"] for r in p1]
p1_prune1  = [r["pruned_fc1"] for r in p1]
p1_prune2  = [r["pruned_fc2"] for r in p1]
p1_regrow1 = [r["regrown_fc1"] for r in p1]
p1_regrow2 = [r["regrown_fc2"] for r in p1]

# ---------- extract phase 2 ----------
p2_rounds  = [r["round"] for r in p2]
p2_spar    = [r["sparsity"] for r in p2]
p2_acc     = [r["test_acc"] for r in p2]
p2_energy  = [r["energy"] for r in p2]
p2_gap     = [r["gap"] for r in p2]
p2_pruned  = [r["pruned"] for r in p2]
p2_regrown = [r["regrown"] for r in p2]

BASELINE = 0.9722
PHASE1_END_ACC = p1_acc[-1]
PHASE2_END_ACC = p2_acc[-1]

# ================================================================
# Figure 1: Full two-phase overview (2×2)
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="white")
fig.suptitle("Exp 33 — Two-Phase Neuron + Weight Pruning (LeNet-300-100, MNIST)",
             fontsize=14, fontweight="bold", y=0.98)

# --- 1a: Phase 1 accuracy + neuron counts ---
ax = axes[0, 0]
ax2 = ax.twinx()
ax.axhline(BASELINE, color="gray", lw=1.2, ls="--", label=f"Baseline {BASELINE:.4f}")
ax.plot(p1_rounds, p1_acc, "o-", color="#2196F3", lw=2, ms=5, label="Test acc (Phase 1)")
ax.set_xlabel("Phase 1 Round")
ax.set_ylabel("Test Accuracy")
ax.set_title("Phase 1: Neuron Glauber — Accuracy & Architecture", fontweight="bold")
ax.set_ylim(0.96, 0.995)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
ax2.plot(p1_rounds, p1_k1, "s--", color="#FF5722", lw=1.5, ms=4, label="k1 (fc1 neurons)")
ax2.plot(p1_rounds, p1_k2, "^--", color="#4CAF50", lw=1.5, ms=4, label="k2 (fc2 neurons)")
ax2.set_ylabel("Alive Neurons")
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=8)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)

# --- 1b: Phase 2 accuracy + sparsity ---
ax = axes[0, 1]
ax2 = ax.twinx()
ax.axhline(BASELINE, color="gray", lw=1.2, ls="--", label=f"Baseline {BASELINE:.4f}")
ax.axhline(PHASE1_END_ACC, color="#2196F3", lw=1.2, ls=":", label=f"Phase 1 end {PHASE1_END_ACC:.4f}")
ax.plot(p2_rounds, p2_acc, "o-", color="#9C27B0", lw=2, ms=4, label="Test acc (Phase 2)")
ax.set_xlabel("Phase 2 Round")
ax.set_ylabel("Test Accuracy")
ax.set_title("Phase 2: Weight Glauber — Accuracy & Sparsity", fontweight="bold")
ax.set_ylim(0.96, 0.995)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.3f}"))
ax2.plot(p2_rounds, [s * 100 for s in p2_spar], "s--", color="#FF9800", lw=1.5, ms=3, label="Weight sparsity %")
ax2.set_ylabel("Weight Sparsity (%)")
ax2.set_ylim(0, 105)
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)

# --- 1c: Phase 2 prune/regrow dynamics ---
ax = axes[1, 0]
ax.fill_between(p2_rounds, p2_pruned, alpha=0.4, color="#F44336", label="Pruned")
ax.fill_between(p2_rounds, p2_regrown, alpha=0.4, color="#4CAF50", label="Regrown")
ax.plot(p2_rounds, p2_pruned, color="#F44336", lw=1.5)
ax.plot(p2_rounds, p2_regrown, color="#4CAF50", lw=1.5)
ax.set_xlabel("Phase 2 Round")
ax.set_ylabel("Weight Count")
ax.set_title("Phase 2: Prune / Regrow Dynamics", fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)
ax.set_yscale("log")
ax.set_ylim(bottom=1)

# --- 1d: Energy across both phases ---
ax = axes[1, 1]
# stitch together: p1 energy, then p2 energy with offset rounds
total_p1 = len(p1_rounds)
p2_offset = [r + total_p1 for r in p2_rounds]
combined_rounds = p1_rounds + p2_offset
combined_energy = p1_energy + p2_energy
ax.axvline(total_p1, color="gray", lw=1.2, ls="--", alpha=0.7, label="Phase 1→2 transition")
ax.plot(p1_rounds, p1_energy, "o-", color="#2196F3", lw=2, ms=4, label="Phase 1 energy")
ax.plot(p2_offset, p2_energy, "o-", color="#9C27B0", lw=2, ms=3, label="Phase 2 energy")
ax.set_xlabel("Round (Phase 1 | Phase 2)")
ax.set_ylabel("Energy")
ax.set_title("Energy Across Both Phases", fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = "/home/petty/pruning-research/results/exp33_overview.png"
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out1}")

# ================================================================
# Figure 2: Phase 1 neuron dynamics detail
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
fig.suptitle("Exp 33 Phase 1 — Neuron Glauber Dynamics", fontsize=13, fontweight="bold")

ax = axes[0]
width = 0.35
rounds_arr = np.array(p1_rounds[1:])  # skip R0 (no prune/regrow)
ax.bar(rounds_arr - width/2, p1_prune1[1:], width, label="Pruned fc1", color="#F44336", alpha=0.8)
ax.bar(rounds_arr + width/2, p1_regrow1[1:], width, label="Regrown fc1", color="#4CAF50", alpha=0.8)
ax.set_xlabel("Round")
ax.set_ylabel("Neuron Count")
ax.set_title("FC1 Neuron Prune/Regrow per Round")
ax.legend()
ax.set_facecolor("white")
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.bar(rounds_arr - width/2, p1_prune2[1:], width, label="Pruned fc2", color="#FF9800", alpha=0.8)
ax.bar(rounds_arr + width/2, p1_regrow2[1:], width, label="Regrown fc2", color="#2196F3", alpha=0.8)
ax.set_xlabel("Round")
ax.set_ylabel("Neuron Count")
ax.set_title("FC2 Neuron Prune/Regrow per Round")
ax.legend()
ax.set_facecolor("white")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out2 = "/home/petty/pruning-research/results/exp33_phase1_dynamics.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out2}")

# ================================================================
# Figure 3: Summary comparison bar chart vs all LeNet experiments
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
fig.suptitle("LeNet-300-100 MNIST — All Experiments Comparison", fontsize=13, fontweight="bold")

methods = [
    "Dense\nbaseline",
    "Single-phase\nGlauber\n(56r, 99.1%)",
    "Fisher/OBD\n(mask cmp\n99%)",
    "Magnitude\n(mask cmp\n99%)",
    "Mag+Rewind\n(mask cmp\n99%)",
    "Two-phase\njob65\n(88.5%+~88%)",
    "Two-phase\njob68\n(60.7%+97.5%)",
]
accs = [0.9722, 0.9719, 0.9727, 0.9742, 0.9770, 0.940, 0.9750]
# Weight sparsity (overall) — neuron pruning converted to equivalent weight reduction
weight_spars = [0.0, 99.1, 99.0, 99.0, 99.0, 88.0, 97.52]
colors = ["#9E9E9E", "#2196F3", "#03A9F4", "#00BCD4", "#009688", "#FF5722", "#9C27B0"]

ax = axes[0]
bars = ax.bar(range(len(methods)), [a * 100 for a in accs], color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
ax.axhline(97.22, color="gray", lw=1.5, ls="--", label="Baseline 97.22%")
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=7.5)
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Final Test Accuracy", fontweight="bold")
ax.set_ylim(93, 98.5)
ax.legend(fontsize=8)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3, axis="y")
for i, (bar, acc) in enumerate(zip(bars, accs)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{acc:.4f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

ax = axes[1]
ax.bar(range(len(methods)), weight_spars, color=colors, alpha=0.85, edgecolor="white", linewidth=1.5)
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=7.5)
ax.set_ylabel("Weight Sparsity (%)")
ax.set_title("Achieved Weight Sparsity", fontweight="bold")
ax.set_ylim(0, 105)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3, axis="y")
for i, (spars) in enumerate(weight_spars):
    ax.text(i, spars + 0.5, f"{spars:.1f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")

plt.tight_layout()
out3 = "/home/petty/pruning-research/results/exp33_comparison.png"
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out3}")

print("All plots done.")
