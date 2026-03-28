#!/usr/bin/env python3
"""
Approximate energy E(w,h) = L(w∘h|D) + (η/2)||w||² + (1/2)Σ ρᵢhᵢ
from logged data.

NOTE: The L2 term (η/2)||w||² requires actual weight norms (not logged).
We plot L and the sparsity term (1/2)Σρᵢhᵢ separately, and show
E_approx = L + sparsity_term as a lower bound (missing L2).

Config values from vgg16_pruning.py:
  eta_w = 1e-5, eta_b = 1e-5
  rho_conv_w = 1e-8, rho_conv_b = 1e-8
  rho_fc_w = (5e-8, 5e-8, 1e-8)  [fc1, fc2, fc3]
  rho_fc_b = (5e-8, 5e-8)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Data from run4 log ────────────────────────────────────────────────────
rounds   = list(range(0, 12))
sparsity = [0.0000,0.3853,0.5415,0.6242,0.6895,0.7372,0.7767,0.8102,0.8387,0.8629,0.8835,0.9010]
test_loss= [0.2896,0.2969,0.2428,0.2506,0.2535,0.2404,0.2684,0.2716,0.2867,0.2990,0.3000,0.2376]

# Dead neuron counts (cumulative, from stats)
dead_fc1 = [0, 341, 388, 412, 442, 471, 502, 559, 616, 696, 782, 842]
dead_fc2 = [0, 390, 484, 601, 746, 869, 1000,1120,1250,1372,1529,1671]
dead_conv= [0,   0,   1,   1,   1,   2,   3,   3,   4,   8,  21,  38]

# ── Architecture param counts ─────────────────────────────────────────────
# VGG16 layer sizes (from PyTorch vgg16 definition)
# Conv weights (out, in, kH, kW):
conv_w = [
    (64, 3, 3, 3),     # conv0
    (64, 64, 3, 3),    # conv1
    (128, 64, 3, 3),   # conv2
    (128, 128, 3, 3),  # conv3
    (256, 128, 3, 3),  # conv4
    (256, 256, 3, 3),  # conv5
    (256, 256, 3, 3),  # conv6
    (512, 256, 3, 3),  # conv7
    (512, 512, 3, 3),  # conv8
    (512, 512, 3, 3),  # conv9
    (512, 512, 3, 3),  # conv10
    (512, 512, 3, 3),  # conv11
    (512, 512, 3, 3),  # conv12
]
N_conv_w = sum(o*i*kh*kw for o,i,kh,kw in conv_w)
N_conv_b = sum(o for o,*_ in conv_w)

# FC layers (image_size=224 → 7x7x512=25088 flattened)
N_fc1_w = 4096 * 25088
N_fc2_w = 4096 * 4096
N_fc3_w = 1000 * 4096
N_fc1_b = 4096
N_fc2_b = 4096
# fc3 biases unpruned (excluded from sparsity accounting)

print(f"Conv weights: {N_conv_w:,}")
print(f"FC weights:   {N_fc1_w+N_fc2_w+N_fc3_w:,}")
print(f"Total prunable weights: {N_conv_w+N_fc1_w+N_fc2_w+N_fc3_w:,}")

rho_conv_w = 1e-8
rho_conv_b = 1e-8
rho_fc_w   = [5e-8, 5e-8, 1e-8]   # fc1, fc2, fc3
rho_fc_b   = [5e-8, 5e-8]

# ── Compute sparsity term per round ──────────────────────────────────────
# (1/2) Σᵢ ρᵢ hᵢ  where hᵢ=1 if weight is active
# We know global sparsity and dead FC neuron counts.
# For conv: nearly all channels survive (dead_conv is very small).
# For FC: dead_fc1/fc2 means entire rows/cols are zeroed.

def sparsity_term(r):
    s = sparsity[r]
    N_total_prunable = N_conv_w + N_conv_b + N_fc1_w + N_fc2_w + N_fc3_w + N_fc1_b + N_fc2_b

    # active count from sparsity fraction
    N_active_total = N_total_prunable * (1 - s)

    # Approximate breakdown: use dead neuron info for FC
    # dead fc1 neurons → entire input row (25088 weights) + output col in fc2 (4096 weights) zeroed
    # dead fc2 neurons → entire input row (4096 weights) + output col in fc3 (1000 weights) zeroed
    dead1 = dead_fc1[r]
    dead2 = dead_fc2[r]
    dc    = dead_conv[r]

    # Active weights per layer (approximate)
    # fc1: dead1 rows of 25088 zeroed
    active_fc1_w = N_fc1_w - dead1 * 25088
    active_fc1_b = N_fc1_b - dead1
    # fc2: dead2 rows of (4096 - dead1) zeroed
    active_fc2_w = N_fc2_w - dead2 * (4096 - dead1)
    active_fc2_b = N_fc2_b - dead2
    # fc3: dead2 input neurons zeroed
    active_fc3_w = N_fc3_w - dead2 * 1000
    # conv: dc channels zeroed (small effect)
    # rough: each dead channel removes ~512*9 weights on average
    active_conv_w = N_conv_w - dc * 512 * 9
    active_conv_b = N_conv_b - dc

    term = (
        0.5 * rho_conv_w * max(0, active_conv_w) +
        0.5 * rho_conv_b * max(0, active_conv_b) +
        0.5 * rho_fc_w[0] * max(0, active_fc1_w) +
        0.5 * rho_fc_b[0] * max(0, active_fc1_b) +
        0.5 * rho_fc_w[1] * max(0, active_fc2_w) +
        0.5 * rho_fc_b[1] * max(0, active_fc2_b) +
        0.5 * rho_fc_w[2] * max(0, active_fc3_w)
    )
    return term

sp_terms = [sparsity_term(r) for r in range(12)]
E_approx  = [l + sp for l, sp in zip(test_loss, sp_terms)]

for r, l, sp, e in zip(rounds, test_loss, sp_terms, E_approx):
    print(f"R{r:2d}: L={l:.4f}  sparsity_term={sp:.4f}  E_approx={e:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("VGG16 Fisher Pruning — Energy Decomposition vs Rounds\n"
             r"$E(\mathbf{w},\mathbf{h}) = L(\mathbf{w}\circ\mathbf{h}|D) + \frac{\eta}{2}\|\mathbf{w}\|^2 + \frac{1}{2}\sum_i \rho_i h_i$",
             fontsize=12, fontweight="bold")

c_loss    = "#C44E52"
c_sp      = "#4C72B0"
c_total   = "#8172B3"
c_l2      = "#CCB974"

ax = axes[0]
ax.plot(rounds, test_loss,  "o-", color=c_loss, lw=2, ms=6, label=r"$L$ (cross-entropy loss)")
ax.plot(rounds, sp_terms,   "s-", color=c_sp,   lw=2, ms=6, label=r"$\frac{1}{2}\sum_i\rho_i h_i$ (sparsity penalty)")
ax.plot(rounds, E_approx,   "^-", color=c_total,lw=2, ms=6, label=r"$E_{\rm approx}$ = L + sparsity term")
ax.fill_between(rounds, test_loss, E_approx, alpha=0.12, color=c_sp, label="sparsity term contribution")
ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
ax.set_xlabel("Pruning Round", fontsize=11)
ax.set_ylabel("Energy (nats)", fontsize=11)
ax.set_title("Energy components (L2 term excluded — requires re-run)", fontsize=10)
ax.set_xticks(rounds)
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

ax = axes[1]
# Fractional contribution of each term to E_approx
frac_loss = [l/e for l, e in zip(test_loss, E_approx)]
frac_sp   = [sp/e for sp, e in zip(sp_terms, E_approx)]
ax.stackplot(rounds, frac_sp, frac_loss,
             labels=["Sparsity penalty fraction", "Loss fraction"],
             colors=[c_sp, c_loss], alpha=0.7)
ax.set_xlabel("Pruning Round", fontsize=11)
ax.set_ylabel("Fraction of E_approx", fontsize=11)
ax.set_title("What dominates the energy?", fontsize=11)
ax.set_xticks(rounds)
ax.set_ylim(0, 1)
ax.legend(fontsize=9, loc="lower right")
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
out = "vgg16_energy.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nSaved: {out}")
