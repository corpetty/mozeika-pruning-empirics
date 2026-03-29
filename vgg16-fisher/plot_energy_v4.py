"""Plot energy components and descent directions vs round for v4 run."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# v4 data — round 0 = resumed state (pre-prune), rounds 1-8 = post-prune/fine-tune
rounds = [0, 1, 2, 3, 4, 5, 6, 7, 8]
sparsity = [0.9515, 0.9613, 0.9690, 0.9752, 0.9802, 0.9842, 0.9873, 0.9899, 0.9919]
E_total  = [56.272304, 45.012721, 36.087643, 28.917754, 23.192336, 18.645192, 14.972845, 12.081390, 9.769684]
E_CE     = [0.2474,    0.2483,    0.2945,    0.2982,    0.3078,    0.3520,    0.3490,    0.3861,    0.4266]
E_L2     = [0.014429,  0.013003,  0.011744,  0.010593,  0.009539,  0.008601,  0.007789,  0.007070,  0.006466]
E_SP     = [56.0105,   44.7514,   35.7814,   28.6089,   22.8750,   18.2846,   14.6160,   11.6883,   9.3366]
# descent_dirs not available at round 0 (pre-prune state)
dd_rounds = [1, 2, 3, 4, 5, 6, 7, 8]
descent_dirs = [6507841, 5200068, 4157285, 3323875, 2657509, 2121471, 1695635, 1355657]
pruned       = [1301844, 1040261,  831770,  665095,  531800,  425080,  339812,  271726]
residual     = [5205997, 4159807, 3325515, 2658780, 2125709, 1696391, 1355823, 1083931]
acc = [0.9163, 0.9187, 0.9024, 0.8991, 0.8934, 0.8800, 0.8848, 0.8694, 0.8572]

test_loss = [0.2474, 0.2483, 0.2945, 0.2982, 0.3078, 0.3520, 0.3490, 0.3861, 0.4266]

fig, axes = plt.subplots(3, 2, figsize=(12, 13), facecolor="white")
fig.suptitle("VGG16 Fisher Pruning: v4 Run (95% → 99% Sparsity)", fontsize=14, fontweight="bold")

# ── Panel 1: Total energy vs round ───────────────────────────────────────────
ax = axes[0, 0]
ax.plot(rounds, E_total, "o-", color="#2196F3", lw=2, ms=7)
ax.set_xlabel("Round")
ax.set_ylabel("Total Energy E")
ax.set_title("Total Energy vs Round")
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)
for r, e in zip(rounds, E_total):
    ax.annotate(f"{e:.1f}", (r, e), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

# ── Panel 2: Energy decomposition ────────────────────────────────────────────
ax = axes[0, 1]
ax.stackplot(rounds, E_SP, E_CE, E_L2,
             labels=["Sparsity (ρ‖w‖₁)", "Cross-entropy", "L2 reg"],
             colors=["#42A5F5", "#EF5350", "#66BB6A"], alpha=0.85)
ax.set_xlabel("Round")
ax.set_ylabel("Energy")
ax.set_title("Energy Decomposition vs Round")
ax.legend(loc="upper right", fontsize=9)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)

# ── Panel 3: Descent directions vs round ─────────────────────────────────────
ax = axes[1, 0]
ax.bar(dd_rounds, [d / 1e6 for d in descent_dirs], color="#7E57C2", alpha=0.8, label="Total")
ax.bar(dd_rounds, [p / 1e6 for p in pruned],       color="#FF7043", alpha=0.9, label="Pruned (20%)")
ax.bar(dd_rounds, [r / 1e6 for r in residual],     color="#26A69A", alpha=0.7, label="Residual")
ax.set_xlabel("Round")
ax.set_ylabel("Count (millions)")
ax.set_title("Energy-Decreasing Directions vs Round")
ax.legend(fontsize=9)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3, axis="y")
for r, d in zip(dd_rounds, descent_dirs):
    ax.text(r, d / 1e6 + 0.1, f"{d/1e6:.2f}M", ha="center", fontsize=8)

# ── Panel 4: Accuracy + sparsity vs round ────────────────────────────────────
ax = axes[1, 1]
color_acc = "#E91E63"
color_sp  = "#FF9800"
ax2 = ax.twinx()
l1, = ax.plot(rounds, [a * 100 for a in acc], "s-", color=color_acc, lw=2, ms=7, label="Test Acc %")
l2, = ax2.plot(rounds, [s * 100 for s in sparsity], "^--", color=color_sp, lw=2, ms=7, label="Sparsity %")
ax.set_xlabel("Round")
ax.set_ylabel("Test Accuracy (%)", color=color_acc)
ax2.set_ylabel("Sparsity (%)", color=color_sp)
ax.set_title("Accuracy & Sparsity vs Round")
ax.tick_params(axis="y", colors=color_acc)
ax2.tick_params(axis="y", colors=color_sp)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)
lines = [l1, l2]
ax.legend(lines, [l.get_label() for l in lines], loc="lower left", fontsize=9)

# ── Panel 5: Total test loss vs round ────────────────────────────────────────
ax = axes[2, 0]
ax.plot(rounds, test_loss, "D-", color="#FF5722", lw=2, ms=7)
ax.set_xlabel("Round")
ax.set_ylabel("Test Loss (CE)")
ax.set_title("Total Test Loss vs Round")
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)
for r, l in zip(rounds, test_loss):
    ax.annotate(f"{l:.3f}", (r, l), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=8)

# ── Panel 6: empty / hide ────────────────────────────────────────────────────
axes[2, 1].set_visible(False)

plt.tight_layout()
out = "/home/petty/pruning-research/vgg16-fisher/vgg16_v4_energy_descent.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved: {out}")
