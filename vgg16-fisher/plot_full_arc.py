#!/usr/bin/env python3
"""Full arc plot: accuracy and sparsity across all VGG16 pruning rounds (v1 through v4 + finetune)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ───────────────────────────────────────────────────────────────────────
# v1/v2 rounds (from VGG16_RESULTS.md) — rounds 0-11 end at 90.1%
# Round 11 = 90.1% / 93.06% (from context; not in table, adding from summary)
v1v2 = [
    (0,  0.0000, 0.8994),
    (1,  0.3853, 0.9013),
    (2,  0.5415, 0.9187),
    (3,  0.6242, 0.9184),
    (4,  0.6895, 0.9244),
    (5,  0.7372, 0.9288),
    (6,  0.7767, 0.9261),
    (7,  0.8102, 0.9234),
    (8,  0.8387, 0.9237),
    (9,  0.8629, 0.9258),
    (10, 0.8835, 0.9165),
    (11, 0.9010, 0.9306),  # final v2 checkpoint
]

# v3 (from vgg16_v3_run.log) — resumes from round 11
v3 = [
    (12, 0.9307, 0.9249),
    (13, 0.9515, 0.9163),
]

# v4 (from vgg16_v4_run.log) — resumes from v3 checkpoint
v4 = [
    (14, 0.9613, 0.9187),
    (15, 0.9690, 0.9024),
    (16, 0.9752, 0.8991),
    (17, 0.9802, 0.8934),
    (18, 0.9842, 0.8800),
    (19, 0.9873, 0.8848),
    (20, 0.9899, 0.8694),
    (21, 0.9919, 0.8572),
]

# Fine-tune endpoint (best checkpoint, same sparsity)
finetune = [(21.5, 0.9919, 0.9061)]

all_rounds = v1v2 + v3 + v4 + finetune
rounds   = [x[0] for x in all_rounds]
sparsity = [x[1] for x in all_rounds]
accuracy = [x[2] for x in all_rounds]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(13, 6), facecolor='white')
ax1.set_facecolor('white')

# ── Shaded phase regions ───────────────────────────────────────────────────────
phases = [
    (0,  11,  '#E3F2FD', 'v1/v2'),
    (11, 13,  '#E8F5E9', 'v3'),
    (13, 21,  '#FFF3E0', 'v4'),
    (21, 22,  '#FCE4EC', 'fine-tune'),
]
for xstart, xend, color, label in phases:
    ax1.axvspan(xstart - 0.5, xend + 0.0, alpha=0.35, color=color, zorder=0)

# ── Sparsity (left axis) ───────────────────────────────────────────────────────
sp_rounds = [x[0] for x in (v1v2 + v3 + v4)]
sp_vals   = [x[1] for x in (v1v2 + v3 + v4)]
color_sp  = '#9E9E9E'
ax1.plot(sp_rounds, sp_vals, 'o--', color=color_sp, linewidth=1.5,
         markersize=4, label='Sparsity', zorder=2, alpha=0.7)
ax1.set_ylabel('Sparsity', color=color_sp, fontsize=12)
ax1.set_ylim(-0.02, 1.05)
ax1.tick_params(axis='y', labelcolor=color_sp)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax1.set_xlabel('Pruning Round', fontsize=12)

# ── Accuracy (right axis) ──────────────────────────────────────────────────────
ax2 = ax1.twinx()
ax2.set_facecolor('white')

# Segment colors
def plot_seg(data, color, label, ls='-', zorder=3):
    rs = [x[0] for x in data]
    ac = [x[2] for x in data]
    ax2.plot(rs, ac, 'o'+ls, color=color, linewidth=2.5, markersize=7,
             label=label, zorder=zorder)

plot_seg(v1v2,    '#2196F3', 'Fisher/OBD (v1/v2)')
plot_seg(v3,      '#4CAF50', 'Fisher/OBD (v3)')
plot_seg(v3[-1:] + v4[:1], '#FF9800', '', ls='--', zorder=2)  # connector
plot_seg(v4,      '#FF9800', 'Fisher/OBD (v4)')

# Fine-tune point
ax2.scatter([21.5], [0.9061], color='#E91E63', s=120, zorder=5,
            marker='*', label='Post fine-tune (99.2%)', linewidths=0)
ax2.annotate('90.61%\n(post fine-tune)',
             xy=(21.5, 0.9061), xytext=(-40, 15), textcoords='offset points',
             fontsize=8.5, color='#E91E63',
             arrowprops=dict(arrowstyle='->', color='#E91E63', lw=1.2))

# Connector lines between segments
for seg_a, seg_b in [(v1v2, v3), (v3, v4)]:
    ax2.plot([seg_a[-1][0], seg_b[0][0]],
             [seg_a[-1][2], seg_b[0][2]],
             '--', color='#BDBDBD', linewidth=1.2, zorder=1)

# Reference: dense baseline
ax2.axhline(0.8994, color='#795548', linewidth=1.2, linestyle=':',
            alpha=0.8, label=f'Dense baseline (89.94%)')

# Peak annotation
peak_r, peak_sp, peak_acc = v1v2[-1]
ax2.annotate(f'Peak: 93.06%\n@ 90.1% sparse',
             xy=(peak_r, peak_acc), xytext=(10, 10), textcoords='offset points',
             fontsize=8.5, color='#2196F3', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#2196F3', lw=1.2))

ax2.set_ylabel('Test Accuracy (CIFAR-10)', fontsize=12)
ax2.set_ylim(0.83, 0.96)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

# Phase labels at top
for xstart, xend, color, label in phases:
    mid = (xstart + xend) / 2
    ax1.text(mid - 0.3, 1.02, label, fontsize=8.5, color='#555',
             ha='center', transform=ax1.get_xaxis_transform())

# x ticks
xticks = list(range(0, 22))
ax1.set_xticks(xticks + [21.5])
xlabels = [str(i) for i in range(22)] + ['FT']
ax1.set_xticklabels(xlabels, fontsize=8)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# Filter connector
lines2 = [l for l, lb in zip(lines2, labels2) if lb]
labels2 = [lb for lb in labels2 if lb]
ax2.legend(lines1 + lines2, labels1 + labels2,
           loc='lower left', fontsize=9, framealpha=0.92)

plt.title('VGG16 CIFAR-10 — Full Pruning Arc (v1 → v4 → Fine-tune)\nFisher/OBD iterative pruning, all rounds',
          fontsize=12, fontweight='bold')
plt.tight_layout()
out = '/home/petty/pruning-research/vgg16-fisher/vgg16_full_arc.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
