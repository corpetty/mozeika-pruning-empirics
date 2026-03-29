# Experiment 19: Online Basis Updating for V Compression

## Overview
V vectors drift more than K vectors over long sequences (overlap 0.702 vs 0.825
from exp13). This experiment tests whether online basis updating can close that
gap, enabling V compression at the same quality level as K compression.

Eval context: 8192 tokens | k=112/4-bit for both K and V
Baseline PPL: 8.1336
K-only reference (V full-dim 4-bit): 9.1012 (1.1190x)

## Results

| Strategy | Update Interval | PPL | Rel PPL | Gap vs K-only | Basis Updates | V Overlap |
|----------|----------------|-----|---------|---------------|---------------|-----------|
| window | 0 | 11.5819 | 1.4239x | +2.4807 | 0 | 1.0000 |
| ema | 0 | 11.5819 | 1.4239x | +2.4807 | 0 | 1.0000 |
| window | 256 | 11.5819 | 1.4239x | +2.4807 | 1 | 0.9978 |
| ema | 256 | 11.5819 | 1.4239x | +2.4807 | 1 | 1.0000 |
| window | 512 | 11.5819 | 1.4239x | +2.4807 | 1 | 0.9978 |
| ema | 512 | 11.5819 | 1.4239x | +2.4807 | 1 | 1.0000 |
| window | 1024 | 11.5819 | 1.4239x | +2.4807 | 1 | 0.9978 |
| ema | 1024 | 11.5819 | 1.4239x | +2.4807 | 1 | 1.0000 |
| window | 2048 | 11.5819 | 1.4239x | +2.4807 | 1 | 0.9978 |
| ema | 2048 | 11.5819 | 1.4239x | +2.4807 | 1 | 1.0000 |

## Key Question
Does any online strategy reduce the PPL gap vs K-only to < 0.05 PPL points?
If so, V compression at k=112/4-bit is viable, pushing total KV compression
from ~5.3x (K+V where V is 4x) to ~6.2x (K subspace k=112 + V subspace k=112).

## Compression Ratio Implication
- Current K+V(full-dim): K at 4.57x, V at 4.0x → harmonic mean ≈ 4.27x
- K+V(both subspace k=112): both at 4.57x → combined 4.57x
- This changes the total KV memory from ~53% reduction to ~78% reduction