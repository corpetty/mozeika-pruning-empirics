"""
Energy decomposition + energy-descent directions for v3 run.

From v3 log:
  Resumed  | sparsity=0.9010 | E=28.568586 (CE=0.2376 L2=0.021358 SP=28.3097) | no descent_dirs (resume)
  Round 12 | sparsity=0.9307 | E=20.073531 (CE=0.2452 L2=0.017435 SP=19.8109) | descent=13295729 pruned=3990166 residual=9305563
  Round 13 | sparsity=0.9515 | E=14.113050 (CE=0.2474 L2=0.014429 SP=13.8512) | descent=9305677  pruned=2792403 residual=6513274

For rounds 0-11 (v1/v2): only test_loss available, no energy breakdown in logs.
We show the full loss curve (0-13) for context, then zoom into the v3 energy breakdown.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Full run accuracy/loss (all 14 rounds, for context) ──────────────────────
rounds_all = list(range(14))
sparsity_all = [0.000, 0.385, 0.542, 0.624, 0.690, 0.737, 0.777, 0.810,
                0.839, 0.863, 0.884, 0.901, 0.931, 0.952]
loss_all     = [0.2896,0.2969,0.2428,0.2506,0.2535,0.2404,0.2684,0.2716,
                0.2867,0.2990,0.3000,0.2376,0.2452,0.2474]

# ── v3 energy decomposition (3 checkpoints) ──────────────────────────────────
# Round labels: 11 = resumed checkpoint, 12 = first v3 prune, 13 = second v3 prune
v3_rounds     = [11,    12,    13]
v3_sparsity   = [90.10, 93.07, 95.15]  # %
v3_E_total    = [28.569, 20.074, 14.113]
v3_CE         = [0.2376, 0.2452, 0.2474]
v3_L2         = [0.02136, 0.01744, 0.01443]
v3_SP         = [28.310, 19.811, 13.851]

# Fractions of total E
v3_frac_CE = [ce/e for ce,e in zip(v3_CE, v3_E_total)]
v3_frac_L2 = [l2/e for l2,e in zip(v3_L2, v3_E_total)]
v3_frac_SP = [sp/e for sp,e in zip(v3_SP, v3_E_total)]

# ── Energy-descent directions (rounds 12 and 13 only) ────────────────────────
dd_rounds   = [12, 13]
dd_total    = [13_295_729,  9_305_677]
dd_pruned   = [ 3_990_166,  2_792_403]
dd_residual = [ 9_305_563,  6_513_274]
dd_sparsity = [93.07, 95.15]

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
BLUE   = "#58a6ff"
GREEN  = "#3fb950"
AMBER  = "#d29922"
RED    = "#f85149"
PURPLE = "#bc8cff"
TEAL   = "#39d353"
GRAY   = "#8b949e"

fig = plt.figure(figsize=(17, 12))
fig.patch.set_facecolor("#0d1117")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

ax1 = fig.add_subplot(gs[0, :2])   # wide: total energy vs sparsity
ax2 = fig.add_subplot(gs[0, 2])    # energy fractions (stacked bar)
ax3 = fig.add_subplot(gs[1, :2])   # descent directions bar chart
ax4 = fig.add_subplot(gs[1, 2])    # CE + L2 close-up

def style_ax(ax, title):
    ax.set_facecolor("#0d1117")
    ax.set_title(title, color="#e6edf3", fontsize=11, pad=10, fontweight="bold")
    ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# ── Plot 1: Total energy E and CE (cross-entropy) vs sparsity ─────────────────
style_ax(ax1, "Energy Components vs Sparsity  (v3 rounds: full breakdown available)")

# Background: CE loss for all 14 rounds (lighter, for context)
ax1.plot([s*100 for s in sparsity_all], loss_all,
         color=GRAY, linewidth=1.2, marker=".", markersize=4, linestyle=":", alpha=0.5,
         label="Cross-entropy loss L (all 14 rounds)")

# v3 detailed breakdown
x = v3_sparsity
ax1.plot(x, v3_E_total, color=PURPLE, linewidth=2.5, marker="o", markersize=8,
         label=r"$E_{total}$ = CE + L2 + SP  (v3)")
ax1.plot(x, v3_SP,      color=AMBER,  linewidth=2,   marker="s", markersize=7,
         label=r"Sparsity penalty $\frac{1}{2}\sum_i\rho_i h_i$")
ax1.plot(x, v3_L2,      color=GREEN,  linewidth=1.8, marker="^", markersize=7,
         label=r"L2 regularization $\frac{\eta}{2}\|w\|^2$")
ax1.plot(x, v3_CE,      color=BLUE,   linewidth=1.8, marker="D", markersize=6,
         label=r"Cross-entropy loss $L(w \circ h | D)$")

# Shade SP contribution
ax1.fill_between(x, v3_CE, v3_E_total, alpha=0.12, color=AMBER)

# Annotate drop in E
for xi, ei, si in zip(x, v3_E_total, [11,12,13]):
    ax1.annotate(f"R{si}\nE={ei:.1f}", xy=(xi, ei), xytext=(xi-0.5, ei+1.2),
                 fontsize=8, color=PURPLE, ha="center",
                 arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.8, alpha=0.5))

ax1.set_xlabel("Sparsity (%)")
ax1.set_ylabel("Energy / Loss value")
ax1.set_xlim(86, 98)
ax1.legend(fontsize=8.5, facecolor="#161b22", edgecolor="#30363d",
           labelcolor="#e6edf3", loc="upper right")

ax1_r = ax1.twinx()
ax1_r.set_facecolor("#0d1117")
ax1_r.set_ylabel("Fraction of E_total", color=AMBER, fontsize=9)
ax1_r.tick_params(colors=AMBER)
for sp in ax1_r.spines.values():
    sp.set_edgecolor("#30363d")

# ── Plot 2: Stacked bar — energy fractions ─────────────────────────────────────
style_ax(ax2, "Energy Composition\nat Each Checkpoint")

bar_x = np.arange(3)
bar_labels = [f"R11\n90.1%", f"R12\n93.1%", f"R13\n95.2%"]

b_sp = ax2.bar(bar_x, v3_frac_SP, color=AMBER, alpha=0.8, label="SP fraction")
b_l2 = ax2.bar(bar_x, v3_frac_L2, bottom=v3_frac_SP, color=GREEN, alpha=0.8, label="L2 fraction")
b_ce = ax2.bar(bar_x, v3_frac_CE,
               bottom=[sp+l2 for sp,l2 in zip(v3_frac_SP, v3_frac_L2)],
               color=BLUE, alpha=0.8, label="CE fraction")

# Annotate actual values inside bars
for i, (fsp, fl2, fce, etot) in enumerate(zip(v3_frac_SP, v3_frac_L2, v3_frac_CE, v3_E_total)):
    ax2.text(i, fsp/2,        f"SP\n{v3_SP[i]:.2f}", ha="center", va="center", fontsize=7.5, color="#0d1117", fontweight="bold")
    ax2.text(i, fsp+fl2/2,    f"L2\n{v3_L2[i]:.4f}", ha="center", va="center", fontsize=7, color="#0d1117")
    ax2.text(i, fsp+fl2+fce/2,f"CE\n{v3_CE[i]:.4f}", ha="center", va="center", fontsize=7.5, color="#0d1117", fontweight="bold")

ax2.set_xticks(bar_x)
ax2.set_xticklabels(bar_labels, fontsize=9)
ax2.set_ylabel("Fraction of total E")
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3",
           loc="upper right")

# ── Plot 3: Energy-descent directions ─────────────────────────────────────────
style_ax(ax3, "Energy-Descent Directions per Round  (weights where ∂E/∂h < 0 → pruning lowers E)")

x_dd = np.arange(2)
w = 0.28
b_total    = ax3.bar(x_dd - w,   dd_total,    w, color=PURPLE, alpha=0.85, label="Total descent dirs")
b_pruned   = ax3.bar(x_dd,       dd_pruned,   w, color=GREEN,  alpha=0.85, label="Actually pruned (cap limited)")
b_residual = ax3.bar(x_dd + w,   dd_residual, w, color=AMBER,  alpha=0.85, label="Residual (would reduce E, not pruned)")

for bar, val in zip(b_total, dd_total):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+150_000,
             f"{val/1e6:.2f}M", ha="center", fontsize=9, color=PURPLE)
for bar, val in zip(b_pruned, dd_pruned):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+150_000,
             f"{val/1e6:.2f}M", ha="center", fontsize=9, color=GREEN)
for bar, val in zip(b_residual, dd_residual):
    ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+150_000,
             f"{val/1e6:.2f}M", ha="center", fontsize=9, color=AMBER)

# Pruned/total ratio
for i, (p, t) in enumerate(zip(dd_pruned, dd_total)):
    ratio = p/t*100
    ax3.text(i, -800_000, f"pruned\n{ratio:.1f}%", ha="center", fontsize=8.5,
             color=GREEN, va="top")

ax3.set_xticks(x_dd)
ax3.set_xticklabels([f"Round 12  (→93.1% sparse)", f"Round 13  (→95.2% sparse)"], fontsize=10)
ax3.set_ylabel("Number of weights")
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v/1e6:.0f}M"))
ax3.set_ylim(0, 16_000_000)
ax3.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

# Annotate the key insight
ax3.annotate("~70% of eligible weights\ncouldn't be pruned due to\nthe 30% cap → latent capacity\nfor further rounds",
             xy=(0.35, 9_200_000), fontsize=8.5, color=AMBER,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#161b22", edgecolor=AMBER, alpha=0.8))

# ── Plot 4: CE and L2 close-up ─────────────────────────────────────────────────
style_ax(ax4, "CE & L2 Close-up\n(absolute values)")

ax4.plot(v3_sparsity, v3_CE, color=BLUE,  linewidth=2.2, marker="o", markersize=8, label="Cross-entropy (CE)")
ax4.plot(v3_sparsity, v3_L2, color=GREEN, linewidth=2.2, marker="s", markersize=8, label="L2 regularization")

for xi, ce, l2 in zip(v3_sparsity, v3_CE, v3_L2):
    ax4.annotate(f"{ce:.4f}", xy=(xi, ce), xytext=(xi+0.15, ce+0.002),
                 fontsize=8, color=BLUE)
    ax4.annotate(f"{l2:.5f}", xy=(xi, l2), xytext=(xi+0.15, l2-0.002),
                 fontsize=8, color=GREEN)

ax4.set_xlabel("Sparsity (%)")
ax4.set_ylabel("Value")
ax4.set_xlim(88, 97)
ax4.legend(fontsize=9, facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")

fig.suptitle(
    "VGG16 Fisher Pruning — Energy Decomposition & Descent Directions  (v3: 90%→95% sparsity)",
    fontsize=13, fontweight="bold", color="#e6edf3", y=0.99
)

out = "/home/petty/pruning-research/vgg16-fisher/vgg16_energy_descent.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"Saved: {out}")
