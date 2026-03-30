# Experiment 24: WikiText-2 PPL — Eval Framework Diagnostic

## Purpose

Verify that our `chunked_cross_entropy` evaluation pipeline is correct, and
establish clean reference PPL numbers using a standard benchmark.

**Prior bug:** Experiments 18–23 used War & Peace with calib/eval overlap
(EVAL_OFFSET = CALIB_TOKENS + 100 chars ≈ 572 tokens, inside calib window of
2048 tokens). Additionally, W&P is in Qwen3's training data (baseline PPL ≈ 1.17,
consistent with memorization).

**Fix:** Calibrate on WikiText-2 train split, evaluate on WikiText-2 test split.
No overlap possible. Standard benchmark with known reference values.

## Baseline Verification

| Method | PPL |
|--------|-----|
| chunked_cross_entropy (our pipeline) | 6.5676 |
| HF CausalLM direct loss (cross-check) | 6.5642 |
| Relative difference | 0.05% |

Expected range for Qwen3-14B on WikiText-2 test: **3–7 PPL**

✅ SANITY PASS

✅ Methods agree

## K-Only Compression Results

| k | bits | PPL | rel_PPL | CR |
|---|------|-----|---------|-----|
| 128 | 16 | 6.5676 | 1.000 | 1.00× |
| 64 | 4 | 53.4682 | 8.1412 | 1.60× |
| 64 | 8 | 41.0170 | 6.2454 | 1.33× |
| 64 | 16 | 41.0571 | 6.2515 | 1.00× |
| 96 | 4 | 11.9671 | 1.8221 | 2.29× |
| 96 | 8 | 9.8753 | 1.5036 | 1.60× |
| 96 | 16 | 9.8753 | 1.5036 | 1.00× |
| 112 | 4 | 8.0759 | 1.2297 | 2.91× |
| 112 | 8 | 7.5922 | 1.1560 | 1.78× |
| 112 | 16 | 7.6033 | 1.1577 | 1.00× |
| 128 | 4 | 6.4421 | 0.9809 | 4.00× |
| 128 | 8 | 6.5644 | 0.9995 | 2.00× |
| 128 | 16 | 6.5676 | 1.0000 | 1.00× |

## Key Findings

1. **Eval framework status:** ✅ chunked_cross_entropy is correct
2. **W&P baseline PPL=1.17 was:** memorization artifact (correct eval gives ~6.6)
3. **Calib/eval split:** Clean (train → test, no overlap)
4. **Reference point:** Qwen3-14B WikiText-2 test PPL = 6.5676

## Implications for Prior Experiments

The relative PPL ratios (compressed/baseline) from prior experiments may still be
directionally correct since both baseline and compressed saw the same eval text.
However, absolute PPL values are unreliable due to:
  - Calib/eval text overlap (eval inside calib window)
  - Training data contamination (W&P memorization)

Experiments whose **conclusions** hold regardless of absolute PPL:
  - Exp19: V online updating is a null result (mechanism is structural, not data-dependent)
  - Exp16: Layer sensitivity ranking (relative ordering, same eval for all)
  - Exp22/23: SubRotQ vs PolarQuant comparison (same eval for both)

Experiments needing re-run with clean eval:
  - Core bitrate sweep (exp9 equivalent) → this experiment
  - Cross-arch validation (exp21) → will need Llama re-run on WikiText-2
