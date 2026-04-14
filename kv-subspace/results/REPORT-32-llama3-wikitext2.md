# Experiment 32: Llama-3.1-8B-AWQ — WikiText-2 PPL (Clean Pipeline)

## Purpose

Replaces exp21 (Llama-3.1 validation) which used War & Peace with the broken
calib/eval split. Uses clean WikiText-2 TRAIN→TEST pipeline from exp24.

## Setup
- Model: hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4
- n_layers=32, n_kv_heads=8, d_head=128
- Calibration: WikiText-2 TRAIN split, 2048 tokens
- Evaluation: WikiText-2 TEST split, 2048 tokens

## Baseline
| Method | PPL |
|--------|-----|
| Llama-3.1-8B-AWQ (this exp) | 6.0784 |
| Qwen3-14B-AWQ (exp24) | 6.5676 |
| Mistral-7B-v0.3 (exp30) | 4.26 |

## K-Only Compression Results

| k | bits | PPL | rel_PPL | CR |
|---|------|-----|---------|-----|
| 128 | 16 | 6.0784 | 1.000 | 1.00× |
| 64 | 4 | 16.4057 | 2.6990 | 8.00× |
| 64 | 8 | 16.5666 | 2.7255 | 4.00× |
| 64 | 16 | 16.5343 | 2.7202 | 2.00× |
| 96 | 4 | 7.1480 | 1.1760 | 5.33× |
| 96 | 8 | 7.0304 | 1.1566 | 2.67× |
| 96 | 16 | 7.0321 | 1.1569 | 1.33× |
| 112 | 4 | 6.3856 | 1.0506 | 4.57× |
| 112 | 8 | 6.2927 | 1.0353 | 2.29× |
| 112 | 16 | 6.2897 | 1.0348 | 1.14× |
| 128 | 4 | 6.1260 | 1.0078 | 4.00× |
| 128 | 8 | 6.0769 | 0.9998 | 2.00× |
| 128 | 16 | 6.0813 | 1.0005 | 1.00× |

## V-Only Sanity Check (k=112, 4-bit)

V-only PPL=75.1124, rel_PPL=12.3574, CR=4.57×

## Cross-Architecture Comparison (K-only, 4-bit)

| k | Llama-3.1 | Mistral-7B | Qwen3-14B |
|---|-----------|------------|-----------|
| 64 | 2.70× | 8.70× | 8.14× |
| 96 | 1.18× | 1.67× | 1.82× |
| 112 | 1.05× | 1.09× | 1.23× |
| 128 | 1.01× | 1.00× | 0.98× |

## Key Findings

1. **Headline**: k=128/4-bit rel_PPL = 1.01× (vs 0.98× Qwen3, 1.00× Mistral)
2. **V compression**: V-only k=112/4-bit rel_PPL = 12.36× (confirms arch-independent failure)
3. **Production config**: k=128/4-bit is consistent across all 3 architectures
