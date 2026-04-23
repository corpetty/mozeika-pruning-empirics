# Task F1: Rotation Ablation — Qwen3-14B WikiText-2

Baseline PPL: 6.5676 | Wall: 1.3 min

| Method | k | bits | PPL | rel PPL | Note |
|--------|---|------|-----|---------|------|
| baseline | 128 | 16 | 6.5676 | 1.0 | no compression |
| plain_4bit | 128 | 4 | 6.4974 | 0.9893 | no PCA, no rotation |
| pca_only | 112 | 4 | 8.1429 | 1.2399 | PCA k=112 → quantize, no rotation |
| rotation_only | 112 | 4 | 6.5387 | 0.9956 | rotate raw K (embed k×k in identity), no PCA |
| subrotq | 112 | 4 | 8.1213 | 1.2366 | PCA k=112 + random rotation + 4-bit |
| pca_only | 128 | 4 | 6.6255 | 1.0088 | PCA k=128 → quantize, no rotation |
| rotation_only | 128 | 4 | 6.6126 | 1.0069 | rotate raw K (embed k×k in identity), no PCA |
| subrotq | 128 | 4 | 6.4468 | 0.9816 | PCA k=128 + random rotation + 4-bit |

## 2×2 Ablation Summary (k=128)

| | No Rotation | With Rotation |
|---|---|---|
| **No PCA** | 0.9893× | 1.0069× |
| **PCA k=128** | 1.0088× | 0.9816× |

## Interpretation

SubRotQ vs PCA-only at k=128: -0.0272x. **Rotation helps at full rank.**
SubRotQ vs plain 4-bit: -0.0077x. **Full pipeline = plain quant at full rank — SubRotQ's value is in truncation.**

## Methodological caution flags

1. **2K eval tokens only** — all differences <0.03x rel PPL are within plausible noise at this token count. The auto-generated interpretation above should not be treated as definitive.

2. **rotation_only implementation** — embeds a k×k rotation in the top-left of a d×d identity. This is not a fair comparison: it preferentially quantizes the first k natural basis directions, not the most informative ones. This conflates "rotation" with "truncation to first k dims."

3. **k=128 is full rank (k=d_head=128)** — at full rank there is no truncation, so PCA-only and rotation-only are mathematically equivalent to different coordinate changes of the same 128-dim space. Differences here reflect quantization grid alignment, not subspace selection. The 2×2 table is most meaningful at k=112 (truncation regime).

4. **Key takeaway at k=112 (truncated regime):**
   - pca_only: 1.2399x (no rotation, truncation hurts)
   - subrotq: 1.2366x (with rotation, essentially same — 0.003x diff)
   - rotation_only: 0.9956x (no truncation, just rotates full K — trivially near-lossless)
   
   At k=112, PCA+rotation ≈ PCA-only. The rotation is not rescuing the truncation error.

5. **What this means for the paper**: SubRotQ's benefit at k<d comes from the PCA subspace selection, not the rotation. The rotation's role needs to be re-examined. Do not update paper claims until A3 results are in and B1/F1 are analyzed jointly.
