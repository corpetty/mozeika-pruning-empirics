#!/usr/bin/env python3
"""Plot test accuracy and sparsity vs round for v4 pruning run."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Data from vgg16_v4_run.log
# Round 0 = resumed state (start of v4), rounds 1-8 = pruning rounds
rounds     = [0,    1,    2,    3,    4,    5,    6,    7,    8]
sparsity   = [0.9515, 0.9613, 0.9690, 0.9752, 0.9802, 0.9842, 0.9873, 0.9899, 0.9919]
accuracy   = [0.9163, 0.9187, 0.9024, 0.8991, 0.8934, 0.8800, 0.8848, 0.8694, 0.8572]

# Dense baseline and v1/v2 context
DENSE_BASELINE = 0.8994   # round 0 of v1 (initial training)
PEAK_ACCURACY  = 0.9306   # end of v2 (90% sparsity)

fig, ax1 = plt.subplots(figsize=(9, 5), facecolor='white')
ax1.set_facecolor('white')

# Sparsity (left axis, muted)
color_sp = '#9E9E9E'
ax1.set_xlabel('Pruning Round (v4)', fontsize=12)
ax1.set_ylabel('Sparsity', color=color_sp, fontsize=12)
ax1.plot(rounds, sparsity, 'o--', color=color_sp, linewidth=1.5,
         markersize=5, label='Sparsity', zorder=2)
ax1.set_ylim(0.93, 1.00)
ax1.tick_params(axis='y', labelcolor=color_sp)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

# Accuracy (right axis, prominent)
ax2 = ax1.twinx()
ax2.set_facecolor('white')
color_acc = '#2196F3'
ax2.set_ylabel('Test Accuracy (CIFAR-10)', color=color_acc, fontsize=12)
ax2.plot(rounds, accuracy, 'o-', color=color_acc, linewidth=2.5,
         markersize=8, label='Test Accuracy', zorder=3)
ax2.set_ylim(0.83, 0.94)
ax2.tick_params(axis='y', labelcolor=color_acc)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))

# Reference lines
ax2.axhline(DENSE_BASELINE, color='#FF9800', linewidth=1.5, linestyle='--',
            alpha=0.8, label=f'Dense baseline ({DENSE_BASELINE:.1%})')
ax2.axhline(PEAK_ACCURACY, color='#4CAF50', linewidth=1.5, linestyle='--',
            alpha=0.8, label=f'Peak @ 90% sparse ({PEAK_ACCURACY:.1%})')

# Annotate accuracy values
for r, acc, sp in zip(rounds, accuracy, sparsity):
    ax2.annotate(f'{acc:.1%}',
                 xy=(r, acc), xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=8.5, color=color_acc)

# Crossover annotation — where acc drops below dense baseline
crossover_r = None
for i in range(1, len(rounds)):
    if accuracy[i] < DENSE_BASELINE and accuracy[i-1] >= DENSE_BASELINE:
        crossover_r = rounds[i]
        crossover_sp = sparsity[i]
        break
if crossover_r:
    ax2.axvline(crossover_r - 0.5, color='#E53935', linewidth=1, linestyle=':',
                alpha=0.7)
    ax2.text(crossover_r - 0.4, 0.836, f'Drops below\ndense ~{crossover_sp:.1%}',
             fontsize=8, color='#E53935', va='bottom')

# x-axis
ax1.set_xticks(rounds)
ax1.set_xticklabels([f'R{r}' if r > 0 else 'Start\n(95.15%)' for r in rounds])

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='lower left',
           fontsize=9, framealpha=0.9)

plt.title('VGG16 v4: Accuracy & Sparsity by Pruning Round\n(resumed from 95.15% checkpoint)', 
          fontsize=12, fontweight='bold')
plt.tight_layout()
out = '/home/petty/pruning-research/vgg16-fisher/vgg16_v4_progression.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {out}")
