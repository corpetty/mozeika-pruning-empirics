# Task F1 v2: Rotation Ablation — Qwen3-14B WikiText-2

Baseline PPL: 6.5676 | Eval tokens: 2048 | Wall: 1.3 min

## Results

| Label | k | bits | PPL | rel PPL | Note |
|-------|---|------|-----|---------|------|
| baseline | 128 | 16 | 6.5676 | 1.0 | fp16 KV cache, no compression |
| plain_4bit | 128 | 4 | 6.4974 | 0.9893 | uniform quant raw K, no PCA, no rotation |
| raw_rotation_128 | 128 | 4 | 6.6126 | 1.0069 | full d×d rotation of raw K + quant all d dims + rotate back |
| pca_only_128 | 128 | 4 | 6.6255 | 1.0088 | PCA k=128 (no truncation) → quant → unproject, no rotation |
| subrotq_128 | 128 | 4 | 6.4468 | 0.9816 | PCA k=128 + 128×128 rotation + quant (full SubRotQ, no truncation) |
| pca_only_112 | 112 | 4 | 8.1429 | 1.2399 | PCA k=112 (truncate 16 dims) → quant → unproject, no rotation |
| subrotq_112 | 112 | 4 | 8.1213 | 1.2366 | PCA k=112 + 112×112 rotation + quant (full SubRotQ, truncated) |

## 2×2 Ablation at k=128 (no truncation)

| | No Rotation | With Rotation |
|---|---|---|
| **No PCA (raw K)** | 0.9893× | 1.0069× |
| **PCA k=128** | 1.0088× | 0.9816× |

## Rotation effect at k=112 (truncation regime)

| pca_only_112 | 1.2399× |
| subrotq_112  | 1.2366× |

Rotation effect at k=112: -0.0033× (helps)

## Notes

- All differences <0.03× should be treated with caution at 2K eval tokens.
- 'rotation_only at k=112' removed (v1 bug: stored all d=128 dims, not a valid truncation ablation).
- raw_rotation_128 uses a full d×d rotation of raw K — valid comparison at k=d only.
