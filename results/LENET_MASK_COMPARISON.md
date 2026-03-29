# LeNet-300-100 Mask Comparison Results

**Date:** 2026-03-28  
**Model:** LeNet-300-100  
**Dataset:** MNIST  
**Experiment:** `experiments/30_mask_comparison.py`  
**Checkpoint:** Shared pretrained model (10 epochs, ~97.8% baseline)

## Setup

All three methods start from the **same pretrained checkpoint** and are pruned independently to ~99% sparsity using iterative pruning (20% of active weights per round) with 5-epoch fine-tuning between rounds.

**Methods:**
- **Fisher/OBD:** Saliency = 0.5 × F_ii × w_i² (diagonal Fisher × weight²). Global threshold.
- **Magnitude:** Global magnitude pruning (|w_i|). No weight reset between rounds.
- **Magnitude + Rewind:** Same magnitude saliency, but weights reset to pretrained values after each prune step (only masks carry forward).

## Results at ~99% Sparsity

### Final Accuracy

| Method | Sparsity | Test Accuracy |
|--------|----------|---------------|
| Fisher/OBD | 99.0% | 97.27% |
| Magnitude | 99.0% | 97.42% |
| Magnitude + Rewind | 99.0% | 97.70% |

All methods within 0.5% of each other. Rewind marginally best.

### Mask Jaccard Similarity (global + per-layer)

| Pair | fc1 (784→300) | fc2 (300→100) | Global |
|------|--------------|--------------|--------|
| Fisher vs Magnitude | 0.091 | 0.088 | **0.090** |
| Fisher vs Mag+Rewind | 0.120 | 0.130 | **0.123** |
| Magnitude vs Mag+Rewind | 0.238 | 0.214 | **0.231** |

### Random baseline
At 99% sparsity, random mask overlap expected: ~1% Jaccard (1% survive × 1% survive = ~0.01).  
All pairs are 9–23× above chance — not random, but still very low absolute overlap.

## Key Findings

1. **Fisher and magnitude select almost entirely different weights.** 9% Jaccard means ~91% of each method's surviving weights are unique to that method.

2. **Rewind shifts which weights survive even with the same criterion.** Magnitude vs Mag+Rewind share only 23% despite identical saliency scores — the iterative weight evolution under magnitude pruning (without reset) selects a different subnetwork than restarting from pretrained each time.

3. **High degeneracy at 99% sparsity on MNIST.** Three very different sparse subnetworks achieve nearly identical accuracy, consistent with the task being massively overparameterized even at full density. Multiple "lottery tickets" exist.

4. **Hypothesis for follow-up:** Jaccard similarity between methods should *increase* as task difficulty increases (tighter feasible set of good sparse masks). VGG16/CIFAR-10 at 90–99% sparsity is the next test.

## Files

- `lenet_mask_comparison_jaccard.json` — full Jaccard + accuracy data
- `lenet_mask_comparison_fisher_log.csv` — per-round sparsity/accuracy for Fisher method
- `lenet_mask_comparison_magnitude_log.csv` — same for Magnitude
- `lenet_mask_comparison_mag_rewind_log.csv` — same for Mag+Rewind
- `lenet_mask_comparison_summary.png` — accuracy vs sparsity curves + global Jaccard heatmap
- `lenet_mask_comparison_layerwise.png` — per-layer Jaccard bar chart
