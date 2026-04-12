"""
Exp 33 (job 68) — Train CE vs Test CE across both phases
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

with open("/home/petty/pruning-research/results/33_phase1_records.json") as f:
    p1 = json.load(f)
with open("/home/petty/pruning-research/results/33_phase2_records.json") as f:
    p2 = json.load(f)

p1_rounds   = [r["round"] for r in p1]
p1_train_ce = [r["train_ce"] for r in p1]
p1_test_ce  = [r["test_ce"] for r in p1]
p1_gap      = [r["gap"] for r in p1]

p2_rounds   = [r["round"] for r in p2]
p2_train_ce = [r["train_ce"] for r in p2]
p2_test_ce  = [r["test_ce"] for r in p2]
p2_gap      = [r["gap"] for r in p2]

n_p1 = len(p1_rounds)
p2_offset = [r + n_p1 for r in p2_rounds]

# ================================================================
# Figure 1: Train/Test CE — two panels side by side
# ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
fig.suptitle("Exp 33 — Train vs Test Cross-Entropy Loss", fontsize=13, fontweight="bold")

# Phase 1
ax = axes[0]
ax.plot(p1_rounds, p1_train_ce, "o-", color="#F44336", lw=2, ms=5, label="Train CE")
ax.plot(p1_rounds, p1_test_ce,  "s-", color="#2196F3", lw=2, ms=5, label="Test CE")
ax.fill_between(p1_rounds, p1_train_ce, p1_test_ce,
                where=[t > tr for t, tr in zip(p1_test_ce, p1_train_ce)],
                alpha=0.15, color="#2196F3", label="Generalisation gap")
ax.set_xlabel("Phase 1 Round")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Phase 1: Neuron Glauber (784→122→35→10)", fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)

# Phase 2
ax = axes[1]
ax.plot(p2_rounds, p2_train_ce, "o-", color="#F44336", lw=2, ms=4, label="Train CE")
ax.plot(p2_rounds, p2_test_ce,  "s-", color="#2196F3", lw=2, ms=4, label="Test CE")
ax.fill_between(p2_rounds, p2_train_ce, p2_test_ce,
                where=[t > tr for t, tr in zip(p2_test_ce, p2_train_ce)],
                alpha=0.15, color="#2196F3", label="Generalisation gap")
ax.set_xlabel("Phase 2 Round")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Phase 2: Weight Glauber on compact net", fontweight="bold")
ax.legend(fontsize=9)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)

plt.tight_layout()
out1 = "/home/petty/pruning-research/results/exp33_train_test_loss.png"
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out1}")

# ================================================================
# Figure 2: Full stitched timeline — train CE, test CE, gap
# ================================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 9), facecolor="white")
fig.suptitle("Exp 33 — Full Timeline: Train & Test CE + Generalisation Gap",
             fontsize=13, fontweight="bold")

all_rounds = p1_rounds + p2_offset
all_train  = p1_train_ce + p2_train_ce
all_test   = p1_test_ce  + p2_test_ce
all_gap    = p1_gap + p2_gap
transition = n_p1  # x-position of phase boundary

# Top panel: CE loss
ax = axes[0]
ax.axvline(transition, color="gray", lw=1.5, ls="--", alpha=0.7)
ax.text(transition + 0.3, max(all_test) * 0.98, "Phase 1→2\ntransition",
        fontsize=8, color="gray", va="top")
ax.plot(p1_rounds,  p1_train_ce, "o-", color="#F44336", lw=2, ms=5, label="Train CE (P1)")
ax.plot(p1_rounds,  p1_test_ce,  "s-", color="#2196F3", lw=2, ms=5, label="Test CE (P1)")
ax.plot(p2_offset,  p2_train_ce, "o-", color="#E91E63", lw=2, ms=4, label="Train CE (P2)")
ax.plot(p2_offset,  p2_test_ce,  "s-", color="#3F51B5", lw=2, ms=4, label="Test CE (P2)")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_title("Train CE vs Test CE", fontweight="bold")
ax.legend(fontsize=9, ncol=2)
ax.set_facecolor("white")
ax.grid(True, alpha=0.3)
ax.set_xticks([])

# Bottom panel: gap = train_ce - test_ce (negative = underfitting/generalizing)
ax = axes[1]
ax.axvline(transition, color="gray", lw=1.5, ls="--", alpha=0.7)
ax.axhline(0, color="black", lw=1, ls="-", alpha=0.4)
gap_full = [tr - te for tr, te in zip(all_train, all_test)]
colors_gap = ["#4CAF50" if g < 0 else "#F44336" for g in gap_full]
ax.bar(all_rounds, gap_full, color=colors_gap, alpha=0.7, width=0.7)
ax.plot(all_rounds, gap_full, "o-", color="#333333", lw=1.2, ms=3)
ax.set_xlabel("Round (Phase 1 | Phase 2)")
ax.set_ylabel("Train CE − Test CE")
ax.set_title("Generalisation Gap (negative = test < train = healthy underfitting)", fontweight="bold")
ax.set_facecolor("white")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out2 = "/home/petty/pruning-research/results/exp33_loss_timeline.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved: {out2}")

print("Done.")
