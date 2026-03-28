#!/usr/bin/env python3
"""Plot dead neuron / channel counts per pruning round for VGG16 Fisher run."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

rounds = list(range(1, 12))

# Cumulative dead counts from log (dead = all input weights zeroed out)
dead_conv = [0,  1,  1,  1,  2,  3,  3,  4,  8, 21, 38]
dead_fc1  = [341,388,412,442,471,502,559,616,696,782,842]
dead_fc2  = [390,484,601,746,869,1000,1120,1250,1372,1529,1671]

# Per-round increments (new deaths each round)
def increments(lst):
    return [lst[0]] + [lst[i]-lst[i-1] for i in range(1, len(lst))]

delta_conv = increments(dead_conv)
delta_fc1  = increments(dead_fc1)
delta_fc2  = increments(dead_fc2)

# Totals
total_fc1 = 4096
total_fc2 = 4096
total_conv_ch = 512  # last conv block channels

colors = {"conv": "#E07B39", "fc1": "#4C72B0", "fc2": "#55A868"}

# ── Figure 1: cumulative dead neurons ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("VGG16 Fisher Pruning — Dead Neurons per Round (CIFAR-10)",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(rounds, dead_fc1, "o-", color=colors["fc1"], linewidth=2, markersize=6,
        label=f"fc1 dead (of {total_fc1})")
ax.plot(rounds, dead_fc2, "s-", color=colors["fc2"], linewidth=2, markersize=6,
        label=f"fc2 dead (of {total_fc2})")
ax.axhline(total_fc1, color=colors["fc1"], linestyle=":", alpha=0.4)
ax.axhline(total_fc2, color=colors["fc2"], linestyle=":", alpha=0.4)
ax.set_xlabel("Pruning Round", fontsize=11)
ax.set_ylabel("Cumulative Dead Neurons", fontsize=11)
ax.set_title("FC Layer Dead Neurons (cumulative)", fontsize=11)
ax.set_xticks(rounds)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Add % annotations at final round
ax.annotate(f"{dead_fc1[-1]/total_fc1*100:.1f}% dead",
            xy=(11, dead_fc1[-1]), xytext=(9.2, dead_fc1[-1]+120),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)
ax.annotate(f"{dead_fc2[-1]/total_fc2*100:.1f}% dead",
            xy=(11, dead_fc2[-1]), xytext=(9.2, dead_fc2[-1]+120),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)

ax2 = axes[1]
ax2.plot(rounds, dead_conv, "^-", color=colors["conv"], linewidth=2, markersize=7,
         label=f"conv channels dead (of {total_conv_ch})")
ax2.axhline(total_conv_ch, color=colors["conv"], linestyle=":", alpha=0.3)
ax2.set_xlabel("Pruning Round", fontsize=11)
ax2.set_ylabel("Cumulative Dead Conv Channels", fontsize=11)
ax2.set_title("Conv Channel Deaths (cumulative)", fontsize=11)
ax2.set_xticks(rounds)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.annotate(f"{dead_conv[-1]/total_conv_ch*100:.1f}% dead\n({dead_conv[-1]}/{total_conv_ch})",
             xy=(11, dead_conv[-1]), xytext=(8.5, dead_conv[-1]+5),
             arrowprops=dict(arrowstyle="->", color="gray"), fontsize=8)

plt.tight_layout()
out1 = "vgg16_dead_neurons_cumulative.png"
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")

# ── Figure 2: per-round new deaths (incremental) ────────────────────────────
x = np.array(rounds)
width = 0.3

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("VGG16 Fisher Pruning — New Deaths per Round (CIFAR-10)",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.bar(x - width/2, delta_fc1, width, color=colors["fc1"], alpha=0.8, label="fc1 new deaths")
ax.bar(x + width/2, delta_fc2, width, color=colors["fc2"], alpha=0.8, label="fc2 new deaths")
ax.set_xlabel("Pruning Round", fontsize=11)
ax.set_ylabel("New Dead Neurons This Round", fontsize=11)
ax.set_title("FC Layer New Deaths per Round", fontsize=11)
ax.set_xticks(rounds)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1]
ax.bar(x, delta_conv, width*1.5, color=colors["conv"], alpha=0.8, label="conv new deaths")
ax.set_xlabel("Pruning Round", fontsize=11)
ax.set_ylabel("New Dead Conv Channels This Round", fontsize=11)
ax.set_title("Conv Channel New Deaths per Round", fontsize=11)
ax.set_xticks(rounds)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
# Highlight late-stage acceleration
for i, (r, v) in enumerate(zip(rounds, delta_conv)):
    if v > 0:
        ax.text(r, v + 0.3, str(v), ha="center", fontsize=8, color="darkred", fontweight="bold")

plt.tight_layout()
out2 = "vgg16_dead_neurons_per_round.png"
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out2}")
print("Done.")
