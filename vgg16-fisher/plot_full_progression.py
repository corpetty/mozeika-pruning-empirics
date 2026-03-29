"""
Full VGG16 pruning progression plots — rounds 0-11 (v1/v2) + v3 extension to 95%
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# From VGG16_RESULTS.md (rounds 0-11) + v3 log (resumed at 90.1%, then rounds 12-13)
rounds = list(range(14))
sparsity = [
    0.0000,   # round 0  (baseline)
    0.3853,   # round 1
    0.5415,   # round 2
    0.6242,   # round 3
    0.6895,   # round 4
    0.7372,   # round 5
    0.7767,   # round 6
    0.8102,   # round 7
    0.8387,   # round 8
    0.8629,   # round 9
    0.8835,   # round 10
    0.9010,   # round 11 — checkpoint saved
    0.9307,   # round 12 (v3, resumed from ckpt)
    0.9515,   # round 13 (v3, final)
]
test_acc = [
    0.8994,   # round 0
    0.9013,   # round 1
    0.9187,   # round 2
    0.9184,   # round 3
    0.9244,   # round 4
    0.9288,   # round 5
    0.9261,   # round 6
    0.9234,   # round 7
    0.9237,   # round 8
    0.9258,   # round 9
    0.9165,   # round 10
    0.9306,   # round 11
    0.9249,   # round 12
    0.9163,   # round 13
]
test_loss = [
    0.2896,   # round 0
    0.2969,   # round 1
    0.2428,   # round 2
    0.2506,   # round 3
    0.2535,   # round 4
    0.2404,   # round 5
    0.2684,   # round 6
    0.2716,   # round 7
    0.2867,   # round 8
    0.2990,   # round 9
    0.3000,   # round 10
    0.2376,   # round 11
    0.2452,   # round 12
    0.2474,   # round 13
]

sparsity_pct = [s * 100 for s in sparsity]
acc_pct = [a * 100 for a in test_acc]

# Surviving neuron data (known points)
# round 11: conv_last=502, fc1=3254, fc2=2425
# round 13: conv_last=467, fc1=3026, fc2=1974
conv_orig, fc1_orig, fc2_orig = 512, 4096, 4096
surviving_rounds = [0, 11, 13]
conv_surv = [512, 502, 467]
fc1_surv  = [4096, 3254, 3026]
fc2_surv  = [4096, 2425, 1974]

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "monospace",
    "axes.facecolor": "#0d1117",
    "figure.facecolor": "#0d1117",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#e6edf3",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#e6edf3",
    "grid.color": "#21262d",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BLUE  = "#58a6ff"
GREEN = "#3fb950"
AMBER = "#d29922"
RED   = "#f85149"
PURPLE= "#bc8cff"
GRAY  = "#8b949e"

# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor("#0d1117")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])

def style_ax(ax, title):
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color="#e6edf3", fontsize=11, pad=10, fontweight="bold")
    ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# ── Plot 1: Accuracy vs Sparsity ───────────────────────────────────────────────
style_ax(ax1, "Accuracy vs Sparsity")
ax1.plot(sparsity_pct[:12], acc_pct[:12], color=BLUE, linewidth=2, marker="o",
         markersize=5, label="v1/v2 (→90%)")
ax1.plot(sparsity_pct[11:], acc_pct[11:], color=GREEN, linewidth=2, marker="o",
         markersize=5, linestyle="--", label="v3 extension (→95%)")
ax1.axhline(acc_pct[0], color=GRAY, linewidth=1, linestyle=":", alpha=0.7, label=f"Baseline {acc_pct[0]:.1f}%")
ax1.axvline(90, color=AMBER, linewidth=1, linestyle=":", alpha=0.6)
ax1.axvline(95, color=RED, linewidth=1, linestyle=":", alpha=0.6)
ax1.text(90.3, 89.5, "90%", color=AMBER, fontsize=8)
ax1.text(95.3, 89.5, "95%", color=RED, fontsize=8)
ax1.annotate(f"93.1%\n@90% sparse", xy=(90.10, 93.06), xytext=(70, 91.5),
             arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2),
             color=AMBER, fontsize=8)
ax1.annotate(f"91.6%\n@95% sparse", xy=(95.15, 91.63), xytext=(75, 90.2),
             arrowprops=dict(arrowstyle="->", color=RED, lw=1.2),
             color=RED, fontsize=8)
ax1.set_xlabel("Sparsity (%)")
ax1.set_ylabel("Test Accuracy (%)")
ax1.set_xlim(-2, 100)
ax1.set_ylim(88.5, 94.5)
ax1.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

# ── Plot 2: Accuracy vs Round ──────────────────────────────────────────────────
style_ax(ax2, "Accuracy & Loss per Round")
ax2b = ax2.twinx()
ax2b.set_facecolor("#0d1117")
ax2.plot(rounds[:12], acc_pct[:12], color=BLUE, linewidth=2, marker="o", markersize=5, label="Acc v1/v2")
ax2.plot(rounds[11:], acc_pct[11:], color=GREEN, linewidth=2, marker="o", markersize=5,
         linestyle="--", label="Acc v3")
ax2b.plot(rounds[:12], test_loss[:12], color=PURPLE, linewidth=1.5, marker="s", markersize=4,
          linestyle=":", label="Loss v1/v2", alpha=0.7)
ax2b.plot(rounds[11:], test_loss[11:], color=AMBER, linewidth=1.5, marker="s", markersize=4,
          linestyle=":", label="Loss v3", alpha=0.7)
ax2.axvline(11, color=AMBER, linewidth=1, linestyle=":", alpha=0.6)
ax2.text(11.1, 88.8, "ckpt", color=AMBER, fontsize=8)
ax2.set_xlabel("Pruning Round")
ax2.set_ylabel("Test Accuracy (%)", color=BLUE)
ax2b.set_ylabel("Test Loss", color=PURPLE)
ax2b.tick_params(colors=PURPLE)
ax2.set_xlim(-0.5, 13.5)
ax2.set_ylim(88.5, 94.5)
ax2b.set_ylim(0.20, 0.35)
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2b.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8,
           facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
for spine in ax2b.spines.values():
    spine.set_edgecolor("#30363d")

# ── Plot 3: Sparsity per round ─────────────────────────────────────────────────
style_ax(ax3, "Sparsity Progression per Round")
bars = ax3.bar(rounds, sparsity_pct, color=[GREEN if s < 90.5 else RED if s > 95 else AMBER
                                             for s in sparsity_pct], alpha=0.75, width=0.6)
ax3.plot(rounds, sparsity_pct, color=BLUE, linewidth=1.5, marker="o", markersize=4, zorder=5)
ax3.axhline(90, color=AMBER, linewidth=1.2, linestyle="--", alpha=0.7, label="90% target")
ax3.axhline(95, color=RED, linewidth=1.2, linestyle="--", alpha=0.7, label="95% target")
for i, (r, s) in enumerate(zip(rounds, sparsity_pct)):
    if s > 1:
        ax3.text(r, s + 0.8, f"{s:.1f}", ha="center", va="bottom", fontsize=6.5, color="#8b949e")
ax3.set_xlabel("Pruning Round")
ax3.set_ylabel("Sparsity (%)")
ax3.set_xlim(-0.8, 13.8)
ax3.set_ylim(0, 102)
ax3.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

# ── Plot 4: Surviving neurons ──────────────────────────────────────────────────
style_ax(ax4, "Surviving Neurons at Key Checkpoints")
x = np.arange(3)
width = 0.25
labels_x = ["Round 0\n(dense)", "Round 11\n(90% sparse)", "Round 13\n(95% sparse)"]

b1 = ax4.bar(x - width, conv_surv, width, label=f"Conv last block (/{conv_orig})", color=BLUE, alpha=0.8)
b2 = ax4.bar(x, [f/40.96 for f in fc1_surv], width, label=f"FC1 neurons /{fc1_orig} (÷40.96)", color=GREEN, alpha=0.8)
b3 = ax4.bar(x + width, [f/40.96 for f in fc2_surv], width, label=f"FC2 neurons /{fc2_orig} (÷40.96)", color=AMBER, alpha=0.8)

# Annotate actual values
for bar, val in zip(b1, conv_surv):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, str(val),
             ha="center", va="bottom", fontsize=7.5, color=BLUE)
for bar, val in zip(b2, fc1_surv):
    ax4.text(bar.get_x() + bar.get_width()/2, val/40.96 + 2, str(val),
             ha="center", va="bottom", fontsize=7.5, color=GREEN)
for bar, val in zip(b3, fc2_surv):
    ax4.text(bar.get_x() + bar.get_width()/2, val/40.96 + 2, str(val),
             ha="center", va="bottom", fontsize=7.5, color=AMBER)

ax4.set_xticks(x)
ax4.set_xticklabels(labels_x, fontsize=9)
ax4.set_ylabel("Count (FC scaled ÷40.96 for visibility)")
ax4.set_ylim(0, 140)
ax4.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.suptitle("VGG16 Fisher Pruning on CIFAR-10  ·  Full Progression 0→95% Sparsity",
             fontsize=14, fontweight="bold", color="#e6edf3", y=0.98)

plt.savefig("/home/petty/pruning-research/vgg16-fisher/vgg16_full_progression.png",
            dpi=150, bbox_inches="tight", facecolor="#0d1117")
print("Saved: vgg16_full_progression.png")
