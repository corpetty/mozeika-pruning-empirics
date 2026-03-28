#!/usr/bin/env python3
"""Plot VGG16 Fisher pruning results: loss, accuracy, sparsity vs. rounds."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Data from /tmp/vgg16_run4.log
rounds    = [0,  1,      2,      3,      4,      5,      6,      7,      8,      9,      10,     11]
loss      = [0.2896, 0.2969, 0.2428, 0.2506, 0.2535, 0.2404, 0.2684, 0.2716, 0.2867, 0.2990, 0.3000, 0.2376]
accuracy  = [89.94,  90.13,  91.87,  91.84,  92.44,  92.88,  92.61,  92.34,  92.37,  92.58,  91.65,  93.06]
sparsity  = [0.00,   38.53,  54.15,  62.42,  68.95,  73.72,  77.67,  81.02,  83.87,  86.29,  88.35,  90.10]

# ── Figure 1: three stacked panels ──────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
fig.suptitle("VGG16 Fisher Pruning — CIFAR-10", fontsize=14, fontweight="bold", y=0.98)

colors = {"sparsity": "#4C72B0", "accuracy": "#55A868", "loss": "#C44E52"}

# Sparsity
ax = axes[0]
ax.plot(rounds, sparsity, "o-", color=colors["sparsity"], linewidth=2, markersize=6)
ax.axhline(90, color=colors["sparsity"], linestyle="--", linewidth=1, alpha=0.5, label="90% target")
ax.fill_between(rounds, sparsity, alpha=0.12, color=colors["sparsity"])
ax.set_ylabel("Sparsity (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
# Annotate final point
ax.annotate(f"90.1%", xy=(11, 90.10), xytext=(9.5, 83),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

# Accuracy
ax = axes[1]
ax.plot(rounds, accuracy, "o-", color=colors["accuracy"], linewidth=2, markersize=6)
ax.axhline(accuracy[0], color="gray", linestyle="--", linewidth=1, alpha=0.6, label=f"Dense baseline {accuracy[0]:.2f}%")
ax.fill_between(rounds, accuracy[0], accuracy, where=[a >= accuracy[0] for a in accuracy],
                alpha=0.15, color=colors["accuracy"], label="Above baseline")
ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_ylim(88, 94)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.annotate(f"93.06% ✓", xy=(11, 93.06), xytext=(8.8, 93.3),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)

# Loss
ax = axes[2]
ax.plot(rounds, loss, "o-", color=colors["loss"], linewidth=2, markersize=6)
ax.axhline(loss[0], color="gray", linestyle="--", linewidth=1, alpha=0.6, label=f"Dense baseline {loss[0]:.4f}")
ax.set_ylabel("Test Loss", fontsize=11)
ax.set_xlabel("Pruning Round", fontsize=11)
ax.set_xticks(rounds)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = "vgg16_metrics_vs_rounds.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: accuracy vs sparsity (the "Pareto" view) ─────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
sc = ax.scatter(sparsity, accuracy, c=rounds, cmap="viridis", s=80, zorder=5)
ax.plot(sparsity, accuracy, "-", color="gray", linewidth=1, alpha=0.5, zorder=4)
cb = fig.colorbar(sc, ax=ax, label="Pruning Round")
ax.axhline(accuracy[0], color="gray", linestyle="--", linewidth=1, alpha=0.6, label=f"Dense baseline {accuracy[0]:.2f}%")
ax.axvline(90, color=colors["sparsity"], linestyle="--", linewidth=1, alpha=0.5, label="90% sparsity target")

# Annotate each point with round number
for r, sp, ac in zip(rounds, sparsity, accuracy):
    ax.annotate(f"R{r}", (sp, ac), textcoords="offset points", xytext=(5, 4), fontsize=8, color="dimgray")

ax.set_xlabel("Sparsity (%)", fontsize=12)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("VGG16 Fisher Pruning — Accuracy vs Sparsity (CIFAR-10)", fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_ylim(88.5, 93.8)

plt.tight_layout()
out2 = "vgg16_accuracy_vs_sparsity.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")

# ── Figure 3: combined dual-axis (accuracy + sparsity on same x) ───────────
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

l1, = ax1.plot(rounds, accuracy, "o-", color=colors["accuracy"], linewidth=2.5, markersize=7, label="Test Accuracy (%)")
ax1.axhline(accuracy[0], color=colors["accuracy"], linestyle=":", linewidth=1, alpha=0.5)
ax1.set_ylabel("Test Accuracy (%)", color=colors["accuracy"], fontsize=12)
ax1.tick_params(axis="y", labelcolor=colors["accuracy"])
ax1.set_ylim(88, 94.5)

l2, = ax2.plot(rounds, sparsity, "s--", color=colors["sparsity"], linewidth=2, markersize=6, label="Sparsity (%)")
ax2.axhline(90, color=colors["sparsity"], linestyle=":", linewidth=1, alpha=0.5)
ax2.set_ylabel("Sparsity (%)", color=colors["sparsity"], fontsize=12)
ax2.tick_params(axis="y", labelcolor=colors["sparsity"])
ax2.set_ylim(0, 105)

ax1.set_xlabel("Pruning Round", fontsize=12)
ax1.set_xticks(rounds)
ax1.set_title("VGG16 Fisher Pruning — Accuracy & Sparsity vs Round (CIFAR-10)",
              fontsize=13, fontweight="bold")
ax1.grid(True, alpha=0.3)

# Joint legend
lines = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="lower right", fontsize=10)

# Annotate target hit
ax1.annotate("Target hit\n90.1% sparse\n93.06% acc",
             xy=(11, 93.06), xytext=(8.5, 93.8),
             arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
             fontsize=9, ha="center",
             bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

plt.tight_layout()
out3 = "vgg16_combined.png"
fig.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out3}")
print("All done.")
