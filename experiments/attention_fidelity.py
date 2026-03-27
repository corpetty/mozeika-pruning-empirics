"""
Experiment 7: Attention score fidelity under KV cache compression.

Two forward passes (baseline vs compressed) on the same text; compare the
attention weight distributions per layer/head to measure how much compression
changes the tokens the model attends to.

Usage:
    /home/petty/torch-env/bin/python3 experiments/attention_fidelity.py
"""

import sys
import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import polar_quantize, subspace_polar_quantize, fit_pca
from collect import get_model_and_tokenizer, find_attention_layers, load_kvs


# ── Config ───────────────────────────────────────────────────────────────────

# Recommended compression config: K subspace k=64 4-bit, V full-dim 4-bit
COMP_CONFIG = {
    'K': ('subspace', 64, 4),
    'V': ('full_dim', 128, 4),
}

EVAL_TEXT = (
    "The development of quantum computing represents a fundamental shift in our "
    "approach to computation. Unlike classical computers, which use bits that exist "
    "in one of two states, quantum computers use quantum bits or qubits that can "
    "exist in superpositions of states. This property, combined with quantum "
    "entanglement and interference, allows quantum computers to process certain "
    "types of problems exponentially faster than their classical counterparts. The "
    "concept was first proposed by physicist Richard Feynman in 1982, who suggested "
    "that simulating quantum mechanical systems on classical computers was "
    "inherently inefficient and that a quantum computer could do it naturally. "
    "Peter Shor's 1994 algorithm for factoring large numbers demonstrated that "
    "quantum computers could solve problems that are practically intractable for "
    "classical machines. This result had profound implications for cryptography, as "
    "many encryption systems rely on the difficulty of factoring large numbers. "
    "Quantum error correction, developed in the mid-1990s, showed that quantum "
    "computation could be made fault-tolerant despite the fragility of quantum "
    "states. Today, companies like IBM, Google, and various startups are racing to "
    "build practical quantum computers. Google claimed quantum supremacy in 2019 "
    "with its Sycamore processor, performing a calculation in 200 seconds that "
    "would take classical supercomputers thousands of years. However, the path to "
    "useful quantum computing remains challenging. Current quantum computers are "
    "noisy and limited in the number of qubits they can maintain. The field of "
    "quantum algorithms continues to grow, with applications being explored in drug "
    "discovery, materials science, optimization, and machine learning. Quantum "
    "machine learning algorithms could potentially speed up training of certain "
    "models and enable new approaches to pattern recognition. The intersection of "
    "quantum computing and artificial intelligence represents one of the most "
    "exciting frontiers in computer science, though practical applications remain "
    "years away for most use cases. Meanwhile, quantum-inspired classical algorithms "
    "have already shown improvements in optimization and sampling tasks."
)

MAX_TOKENS = 512


# ── PCA bases ────────────────────────────────────────────────────────────────

def compute_pca_bases(kvs_path, k=64):
    """Compute PCA bases per (layer, head) from calibration KV data."""
    kvs = load_kvs(kvs_path)
    bases = {}
    for layer_idx in sorted(kvs.keys()):
        K = kvs[layer_idx]['K']  # (T, n_heads, d_head)
        V = kvs[layer_idx]['V']
        n_heads = K.shape[1]
        for h in range(n_heads):
            U_k, mean_k = fit_pca(K[:, h, :], k)
            U_v, mean_v = fit_pca(V[:, h, :], k)
            bases[(layer_idx, h)] = {
                'U_K': U_k, 'mean_K': mean_k,
                'U_V': U_v, 'mean_V': mean_v,
            }
    return bases


# ── Compression hooks (same as Exp 6) ────────────────────────────────────────

def compress_head(x_np, method, k, n_bits, U_k, mean):
    if method == 'subspace':
        return subspace_polar_quantize(x_np, k, n_bits, U_k, mean)
    elif method == 'full_dim':
        return polar_quantize(x_np, n_bits)
    return x_np


def install_compression_hooks(model, config, bases, n_kv_heads, d_head):
    hooks = []
    attn_layers = find_attention_layers(model)
    for layer_idx, attn in attn_layers:
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            spec = config[kv_type]
            if spec is None:
                continue
            method, k, n_bits = spec

            def make_hook(li, kvt, m, kk, nb):
                def hook(module, input, output):
                    device, dtype = output.device, output.dtype
                    x = output.detach().cpu().float()
                    batch, seq, _ = x.shape
                    x = x.reshape(batch, seq, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = x[0, :, h, :].numpy()
                        base = bases.get((li, h), {})
                        U = base.get(f'U_{kvt}')
                        mn = base.get(f'mean_{kvt}')
                        xh_comp = compress_head(xh, m, kk, nb, U, mn)
                        x[0, :, h, :] = torch.from_numpy(xh_comp)
                    return x.reshape(batch, seq, -1).to(device=device, dtype=dtype)
                return hook

            proj = getattr(attn, proj_name)
            h = proj.register_forward_hook(
                make_hook(layer_idx, kv_type, method, k, n_bits)
            )
            hooks.append(h)
    return hooks


# ── Attention weight capture via q_proj/k_proj hooks ─────────────────────────
#
# We capture pre-RoPE Q and K from each layer, then manually compute
# attention weights as softmax(Q @ K^T / sqrt(d)).  Because RoPE is an
# isometry applied identically in both passes, the *relative* difference
# between baseline and compressed attention distributions is preserved.

def install_qk_capture_hooks(model, n_kv_heads, d_head):
    """Capture q_proj and k_proj outputs per layer (pre-RoPE)."""
    store = {}   # layer_idx -> {'Q': tensor, 'K': tensor}
    hooks = []
    attn_layers = find_attention_layers(model)
    n_q_heads = model.config.num_attention_heads

    for layer_idx, attn in attn_layers:
        store[layer_idx] = {}

        for which, proj_name in [('Q', 'q_proj'), ('K', 'k_proj')]:
            def make_hook(li, w):
                def hook(module, input, output):
                    store[li][w] = output.detach().cpu().float()
                return hook

            proj = getattr(attn, proj_name)
            h = proj.register_forward_hook(make_hook(layer_idx, which))
            hooks.append(h)

    return store, hooks


def compute_attention_weights(Q_flat, K_flat, n_q_heads, n_kv_heads, d_head):
    """
    Compute pre-RoPE attention weights from raw projection outputs.

    Q_flat: (1, T, n_q_heads * d_head)
    K_flat: (1, T, n_kv_heads * d_head)

    Returns: (n_q_heads, T, T) numpy attention weights.
    """
    T = Q_flat.shape[1]
    Q = Q_flat[0].reshape(T, n_q_heads, d_head).numpy()    # (T, n_q_heads, d)
    K = K_flat[0].reshape(T, n_kv_heads, d_head).numpy()    # (T, n_kv_heads, d)

    scale = 1.0 / np.sqrt(d_head)
    heads_per_group = n_q_heads // n_kv_heads

    attn_all = np.zeros((n_q_heads, T, T), dtype=np.float32)

    for qh in range(n_q_heads):
        kv_h = qh // heads_per_group
        # (T, d) @ (d, T) -> (T, T)
        logits = Q[:, qh, :] @ K[:, kv_h, :].T * scale
        # Apply causal mask
        mask = np.triu(np.full((T, T), -1e9, dtype=np.float32), k=1)
        logits = logits + mask
        # Softmax
        logits -= logits.max(axis=-1, keepdims=True)
        exp_l = np.exp(logits)
        attn_all[qh] = exp_l / exp_l.sum(axis=-1, keepdims=True)

    return attn_all


# ── Fidelity metrics ─────────────────────────────────────────────────────────

def compute_fidelity_metrics(attn_base, attn_comp, n_q_heads, seq_len):
    """
    Compare attention distributions between baseline and compressed.

    attn_base, attn_comp: (n_q_heads, T, T)

    Returns list of dicts with per-layer-head-token_range metrics.
    """
    rows = []
    mid = seq_len // 2

    for qh in range(n_q_heads):
        for tok_range, (lo, hi) in [('early', (1, mid)), ('late', (mid, seq_len))]:
            if hi <= lo:
                continue
            ab = attn_base[qh, lo:hi, :hi]    # (n_tok, context)
            ac = attn_comp[qh, lo:hi, :hi]

            n_tok = ab.shape[0]

            # Top-1 match
            top1_b = np.argmax(ab, axis=-1)
            top1_c = np.argmax(ac, axis=-1)
            top1_match = float(np.mean(top1_b == top1_c))

            # Top-5 Jaccard (per token, then average)
            jaccards = []
            for t in range(n_tok):
                set_b = set(np.argsort(ab[t])[-5:])
                set_c = set(np.argsort(ac[t])[-5:])
                inter = len(set_b & set_c)
                union = len(set_b | set_c)
                jaccards.append(inter / union if union > 0 else 1.0)
            top5_jaccard = float(np.mean(jaccards))

            # KL divergence: KL(base || comp)
            eps = 1e-10
            kl = float(np.mean(
                np.sum(ab * np.log((ab + eps) / (ac + eps)), axis=-1)
            ))

            rows.append({
                'head': qh,
                'token_range': tok_range,
                'top1_match': top1_match,
                'top5_jaccard': top5_jaccard,
                'attn_kl': kl,
                'n_tokens': n_tok,
            })

    return rows


# ── Report ───────────────────────────────────────────────────────────────────

def write_report(all_rows, n_layers, n_q_heads, n_kv_heads):
    lines = [
        "# Experiment 7: Attention Score Fidelity Under KV Compression\n",
        "## Setup\n",
        "- Model: Qwen3-14B-AWQ",
        f"- {n_layers} layers, {n_q_heads} query heads, {n_kv_heads} KV heads, d_head=128",
        "- Compression config: K subspace k=64 4-bit, V full-dim 4-bit (KV_optimal)",
        "- Attention computed from pre-RoPE Q,K projections (isometry preserves relative comparison)",
        f"- Sequence length: {MAX_TOKENS} tokens\n",
    ]

    # Group by layer range
    def layer_range(layer):
        if layer < 10:
            return 'early (L0-9)'
        elif layer < 30:
            return 'mid (L10-29)'
        else:
            return 'late (L30-39)'

    lines.append("## Top-1 Match Rate by Layer Range\n")
    lines.append("| Layer range | Token range | Top-1 match | Top-5 Jaccard | Attn KL |")
    lines.append("|-------------|-------------|-------------|---------------|---------|")

    for lr_name in ['early (L0-9)', 'mid (L10-29)', 'late (L30-39)']:
        for tok_range in ['early', 'late']:
            subset = [r for r in all_rows
                      if layer_range(r['layer']) == lr_name
                      and r['token_range'] == tok_range]
            if not subset:
                continue
            t1 = np.mean([r['top1_match'] for r in subset])
            j5 = np.mean([r['top5_jaccard'] for r in subset])
            kl = np.mean([r['attn_kl'] for r in subset])
            lines.append(f"| {lr_name} | {tok_range} | {t1:.4f} | {j5:.4f} | {kl:.6f} |")

    lines.append("\n## Per-Layer Aggregation (all tokens)\n")
    lines.append("| Layer | Top-1 match | Top-5 Jaccard | Attn KL |")
    lines.append("|-------|-------------|---------------|---------|")

    layers_seen = sorted(set(r['layer'] for r in all_rows))
    for layer in layers_seen:
        subset = [r for r in all_rows if r['layer'] == layer]
        t1 = np.mean([r['top1_match'] for r in subset])
        j5 = np.mean([r['top5_jaccard'] for r in subset])
        kl = np.mean([r['attn_kl'] for r in subset])
        lines.append(f"| {layer:2d} | {t1:.4f} | {j5:.4f} | {kl:.6f} |")

    lines.append("\n## Do Late Layers (30-39) Show Worst Fidelity?\n")
    for lr_name in ['early (L0-9)', 'mid (L10-29)', 'late (L30-39)']:
        subset = [r for r in all_rows if layer_range(r['layer']) == lr_name]
        if subset:
            kl = np.mean([r['attn_kl'] for r in subset])
            t1 = np.mean([r['top1_match'] for r in subset])
            lines.append(f"- **{lr_name}**: mean attn KL = {kl:.6f}, top-1 match = {t1:.4f}")

    late_kl = np.mean([r['attn_kl'] for r in all_rows if layer_range(r['layer']) == 'late (L30-39)']) if any(layer_range(r['layer']) == 'late (L30-39)' for r in all_rows) else 0
    mid_kl = np.mean([r['attn_kl'] for r in all_rows if layer_range(r['layer']) == 'mid (L10-29)']) if any(layer_range(r['layer']) == 'mid (L10-29)' for r in all_rows) else 0
    early_kl = np.mean([r['attn_kl'] for r in all_rows if layer_range(r['layer']) == 'early (L0-9)']) if any(layer_range(r['layer']) == 'early (L0-9)' for r in all_rows) else 0

    worst = max([(early_kl, 'early'), (mid_kl, 'mid'), (late_kl, 'late')], key=lambda x: x[0])
    if worst[1] == 'late':
        lines.append("\nYes — late layers show the highest attention KL, consistent with "
                      "KV compression experiments showing late layers are hardest to compress.")
    else:
        lines.append(f"\nNo — **{worst[1]}** layers show the worst fidelity. This may indicate "
                      "that attention score sensitivity differs from KV reconstruction difficulty.")

    lines.append("\n## Practical Implication\n")
    overall_t1 = np.mean([r['top1_match'] for r in all_rows])
    overall_j5 = np.mean([r['top5_jaccard'] for r in all_rows])
    overall_kl = np.mean([r['attn_kl'] for r in all_rows])

    lines.append(f"- Overall top-1 match rate: **{overall_t1:.4f}** "
                 f"({overall_t1 * 100:.1f}% of tokens attend to the same top token)")
    lines.append(f"- Overall top-5 Jaccard: **{overall_j5:.4f}**")
    lines.append(f"- Overall attention KL: **{overall_kl:.6f}**\n")

    if overall_t1 > 0.95:
        lines.append("Compression with KV_optimal config preserves attention patterns "
                      "with high fidelity — the model attends to nearly the same tokens.")
    elif overall_t1 > 0.85:
        lines.append("Compression moderately perturbs attention patterns but preserves "
                      "the dominant attended tokens for most positions.")
    else:
        lines.append("Compression significantly changes attention patterns. Consider "
                      "less aggressive compression for layers with low fidelity.")

    with open('results/REPORT-7-attention-fidelity.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ── Main ─────────────────────────────────────────────────────────────────────

def main(model=None, tokenizer=None, bases=None, n_kv_heads=None, d_head=None):
    os.chdir(Path(__file__).resolve().parent.parent)
    device = 'cuda'

    print("\n" + "=" * 70)
    print("Experiment 7: Attention Score Fidelity Under KV Compression")
    print("=" * 70)

    if model is None:
        print("\nLoading model...")
        model, tokenizer = get_model_and_tokenizer('Qwen/Qwen3-14B-AWQ')

    if n_kv_heads is None:
        n_kv_heads = model.config.num_key_value_heads
        d_head = model.config.hidden_size // model.config.num_attention_heads

    n_q_heads = model.config.num_attention_heads
    n_layers = model.config.num_hidden_layers
    print(f"  n_layers={n_layers}, n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}, d_head={d_head}")

    if bases is None:
        print("Computing PCA bases...")
        bases = compute_pca_bases('results/kvs.npz', k=64)

    # Tokenize
    inputs = tokenizer(EVAL_TEXT, return_tensors='pt', truncation=True, max_length=MAX_TOKENS)
    input_ids = inputs['input_ids'].to(device)
    seq_len = input_ids.shape[1]
    print(f"  Sequence length: {seq_len} tokens")

    attn_layers = find_attention_layers(model)
    layer_indices = [idx for idx, _ in attn_layers]

    # ── Pass A: Baseline ──
    print("\nPass A: Baseline (no compression)...")
    store_base, hooks_base = install_qk_capture_hooks(model, n_kv_heads, d_head)
    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks_base:
        h.remove()

    # Compute baseline attention weights per layer
    print("  Computing baseline attention weights...")
    attn_baseline = {}
    for li in layer_indices:
        if 'Q' in store_base[li] and 'K' in store_base[li]:
            attn_baseline[li] = compute_attention_weights(
                store_base[li]['Q'], store_base[li]['K'],
                n_q_heads, n_kv_heads, d_head,
            )
    del store_base
    print(f"  Captured attention for {len(attn_baseline)} layers")

    # ── Pass B: Compressed ──
    print("\nPass B: Compressed KV (KV_optimal config)...")
    comp_hooks = install_compression_hooks(model, COMP_CONFIG, bases, n_kv_heads, d_head)
    store_comp, capture_hooks = install_qk_capture_hooks(model, n_kv_heads, d_head)
    with torch.no_grad():
        model(input_ids=input_ids)
    for h in comp_hooks + capture_hooks:
        h.remove()

    print("  Computing compressed attention weights...")
    attn_compressed = {}
    for li in layer_indices:
        if 'Q' in store_comp[li] and 'K' in store_comp[li]:
            attn_compressed[li] = compute_attention_weights(
                store_comp[li]['Q'], store_comp[li]['K'],
                n_q_heads, n_kv_heads, d_head,
            )
    del store_comp

    # ── Compute fidelity metrics ──
    print("\nComputing fidelity metrics...")
    all_rows = []
    for li in sorted(attn_baseline.keys()):
        if li not in attn_compressed:
            continue
        layer_rows = compute_fidelity_metrics(
            attn_baseline[li], attn_compressed[li],
            n_q_heads, seq_len,
        )
        for r in layer_rows:
            r['layer'] = li
        all_rows.extend(layer_rows)
        if li % 10 == 0:
            sub = [r for r in layer_rows]
            t1 = np.mean([r['top1_match'] for r in sub])
            kl = np.mean([r['attn_kl'] for r in sub])
            print(f"  Layer {li:2d}: top1_match={t1:.4f}, attn_kl={kl:.6f}")

    del attn_baseline, attn_compressed

    # Save CSV
    Path('results').mkdir(exist_ok=True)
    fieldnames = ['layer', 'head', 'token_range', 'top1_match', 'top5_jaccard', 'attn_kl', 'n_tokens']
    with open('results/attention_fidelity.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved results/attention_fidelity.csv ({len(all_rows)} rows)")

    # Write report
    write_report(all_rows, n_layers, n_q_heads, n_kv_heads)
    print("Wrote results/REPORT-7-attention-fidelity.md")


if __name__ == '__main__':
    main()
