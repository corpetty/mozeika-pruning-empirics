"""
experiment.py — Full pipeline: collect KV vectors → analyze → compress → report.

Usage:
    # Full run (loads model, collects KVs, analyzes, compresses):
    python experiment.py --n-tokens 2048

    # Skip collection if kvs.npz already exists:
    python experiment.py --skip-collect

    # Quick test on a few layers only:
    python experiment.py --skip-collect --max-layers 4
"""

import argparse
import csv
import json
import numpy as np
from pathlib import Path


def run_full_experiment(args):
    from collect import get_model_and_tokenizer, collect_kv_vectors, save_kvs, get_sample_text, load_kvs
    from analyze import run_analysis, print_summary
    from compress import compare_compression_methods

    Path('results').mkdir(exist_ok=True)
    kvs_path = 'results/kvs.npz'

    # ── Step 1: Collect KV vectors ────────────────────────────────────────────
    if args.skip_collect and Path(kvs_path).exists():
        print(f"Loading existing KV vectors from {kvs_path}")
        kv_dict = load_kvs(kvs_path)
    else:
        model, tokenizer = get_model_and_tokenizer(args.model)
        text = get_sample_text(n_chars=args.n_tokens * 8)
        kv_dict = collect_kv_vectors(model, tokenizer, text, args.n_tokens)
        save_kvs(kv_dict, kvs_path)
        del model  # free VRAM

    # ── Step 2: PCA / effective rank analysis ─────────────────────────────────
    print("\n=== Effective Rank Analysis ===")
    analysis = run_analysis(kv_dict)
    print_summary(analysis)

    # Save per-layer summary CSV
    rank_rows = []
    for layer_idx, head_results in analysis.items():
        for r in head_results:
            rank_rows.append({
                'layer': layer_idx,
                'head': r['head'],
                'eff_rank_K_90': r['eff_rank_K_90'],
                'eff_rank_K_95': r['eff_rank_K_95'],
                'eff_rank_V_90': r['eff_rank_V_90'],
                'eff_rank_V_95': r['eff_rank_V_95'],
                'participation_ratio_K': r['participation_ratio_K'],
                'participation_ratio_V': r['participation_ratio_V'],
                'd_head': r['d_head'],
                'n_tokens': r['n_tokens'],
            })

    with open('results/effective_rank.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rank_rows[0].keys())
        writer.writeheader()
        writer.writerows(rank_rows)
    print("Saved results/effective_rank.csv")

    # ── Step 3: Compression distortion comparison ─────────────────────────────
    print("\n=== Compression Distortion Comparison ===")

    layers = sorted(kv_dict.keys())
    if args.max_layers:
        # Sample evenly across depth
        idxs = np.linspace(0, len(layers)-1, args.max_layers, dtype=int)
        layers = [layers[i] for i in idxs]
        print(f"Testing {len(layers)} layers (sampled): {layers}")

    bit_budgets = [2, 4, 8]         # bits per scalar
    # k values as fractions of d_head
    sample_K = kv_dict[layers[0]]['K']
    d_head = sample_K.shape[2]
    k_values = [max(2, d_head // 8), d_head // 4, d_head // 2]
    print(f"d_head={d_head}, testing k={k_values}, bits={bit_budgets}")

    compress_rows = []
    for layer_idx in layers:
        K = kv_dict[layer_idx]['K']  # (T, n_heads, d_head)
        V = kv_dict[layer_idx]['V']
        n_heads = K.shape[1]

        # Use first 2 heads for speed (they should be representative)
        for h in range(min(2, n_heads)):
            Kh = K[:, h, :].astype(np.float32)
            Qh = K[:, h, :].astype(np.float32)  # use K as proxy for Q (no Q collected)

            results = compare_compression_methods(Kh, V[:, h, :], Qh, bit_budgets, k_values)
            for r in results:
                compress_rows.append({'layer': layer_idx, 'head': h, **r})
            
            # Print quick summary
            for r in results:
                if r['method'] == 'full_dim' or r['k'] == k_values[1]:
                    print(f"  L{layer_idx:2d} H{h}  {r['method']:<10}  k={r['k']:3d}  "
                          f"{r['bits_per_scalar']}bps  kl={r['K_kl_divergence']:.4f}  "
                          f"top1={r['K_top1_agreement']:.3f}")

    with open('results/compression_distortion.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=compress_rows[0].keys())
        writer.writeheader()
        writer.writerows(compress_rows)
    print("Saved results/compression_distortion.csv")

    # ── Step 4: Summary stats ─────────────────────────────────────────────────
    print("\n=== Key Findings ===")

    # Average effective rank across layers
    erk = [r['eff_rank_K_90'] for r in rank_rows]
    print(f"Mean eff_rank_K @90%: {np.mean(erk):.1f}/{d_head} ({np.mean(erk)/d_head*100:.0f}%)")
    print(f"Range: {min(erk)} - {max(erk)}")

    # Best layer for subspace compression
    low_rank_layers = [(r['layer'], r['eff_rank_K_90']) for r in rank_rows 
                       if r['head'] == 0]
    low_rank_layers.sort(key=lambda x: x[1])
    print(f"\nLowest rank layers (K @90%):")
    for l, er in low_rank_layers[:5]:
        print(f"  Layer {l}: eff_rank={er}/{d_head} ({er/d_head*100:.0f}%)")

    # Distortion comparison: subspace vs full-dim
    for bits in bit_budgets:
        full = [r for r in compress_rows if r['method'] == 'full_dim' and r['bits_per_scalar'] == bits]
        sub  = [r for r in compress_rows if r['method'] == 'subspace' and r['bits_per_scalar'] == bits and r['k'] == k_values[1]]
        if full and sub:
            full_kl = np.mean([r['K_kl_divergence'] for r in full])
            sub_kl  = np.mean([r['K_kl_divergence'] for r in sub])
            print(f"\n{bits}-bit:  full_dim KL={full_kl:.4f}  subspace(k={k_values[1]}) KL={sub_kl:.4f}  "
                  f"ratio={sub_kl/full_kl:.2f}x")

    print("\nDone. Results in results/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-14B-AWQ')
    parser.add_argument('--n-tokens', type=int, default=2048)
    parser.add_argument('--skip-collect', action='store_true')
    parser.add_argument('--max-layers', type=int, default=None,
                        help='Limit to N layers (sampled evenly) for faster testing')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    run_full_experiment(args)


if __name__ == '__main__':
    main()
