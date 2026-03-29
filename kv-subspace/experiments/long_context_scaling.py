"""
Experiment 13: Long-context KV cache compression scaling.

Answers: How does compression quality hold as context grows from 512 → 40K tokens?
  - Does relative PPL stay flat (compression is context-length-agnostic)?
  - Does the PCA basis calibrated on short context transfer to long context?
  - Where in the sequence does per-token loss concentrate under compression?

Three sub-experiments:
  A. PPL vs context length for multiple compression configs (baseline, k128/4bit,
     k112/4bit, k64/4bit) at context windows: 512, 1K, 2K, 4K, 8K, 16K, 32K, 40K
  B. Position-aware per-token loss at 16K context — where in the sequence does
     compression error accumulate?
  C. Calibration basis drift — PCA subspace overlap between windows at different
     positions in the document (early vs late, to check for distribution shift)

Usage:
    /home/petty/torch-env/bin/python3 experiments/long_context_scaling.py

Outputs:
    results/long_context_ppl.csv        - PPL vs context length
    results/long_context_per_token.csv  - per-token loss at 16K
    results/long_context_basis_drift.csv - basis overlap vs position
    results/REPORT-13-long-context.md
"""

import sys
import os
import csv
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import polar_quantize, subspace_polar_quantize, fit_pca
from collect import get_model_and_tokenizer, find_attention_layers

try:
    from scipy.linalg import subspace_angles
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _get_transformer_body_and_head(model):
    """
    Return (transformer_body, lm_head) for AWQ-wrapped Qwen3 models.
    AWQ structure: model (AutoAWQForCausalLM)
                     .model (Qwen3ForCausalLM)  <- still calls lm_head in forward()
                       .model (Qwen3Model)       <- transformer body only
                       .lm_head                  <- the projection layer
    Falls back to model.model / model.lm_head for non-AWQ.
    """
    # AWQ wraps: model.model = Qwen3ForCausalLM, model.model.model = Qwen3Model
    causal_lm = getattr(model, 'model', model)
    if hasattr(causal_lm, 'model') and hasattr(causal_lm, 'lm_head'):
        return causal_lm.model, causal_lm.lm_head
    # Fallback
    return causal_lm, model.lm_head


def chunked_cross_entropy(model, input_ids, chunk_size=512):
    """
    Compute cross-entropy loss without materialising the full (seq_len, vocab) logit tensor.
    For large contexts (32K+), the full logit matrix is ~9 GB; chunking avoids the OOM.
    Runs the transformer body once, then projects hidden states to logits in chunks.
    """
    transformer_body, lm_head = _get_transformer_body_and_head(model)

    with torch.no_grad():
        outputs = transformer_body(input_ids=input_ids[:, :-1])
        hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)

    labels = input_ids[:, 1:].view(-1)  # (seq_len,)
    seq_len = hidden.shape[1]
    total_loss = 0.0
    n_chunks = 0

    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_hidden = hidden[:, start:end, :]
            chunk_logits = lm_head(chunk_hidden)
            chunk_labels = labels[start:end]
            chunk_loss = torch.nn.functional.cross_entropy(
                chunk_logits.view(-1, chunk_logits.size(-1)),
                chunk_labels,
            )
            total_loss += float(chunk_loss) * (end - start)
            n_chunks += (end - start)
            del chunk_hidden, chunk_logits, chunk_labels
            torch.cuda.empty_cache()

    del hidden
    torch.cuda.empty_cache()
    return total_loss / n_chunks

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_NAME   = 'Qwen/Qwen3-14B-AWQ'
DATA_FILE    = 'data/war_and_peace.txt'
CALIB_TOKENS = 2048          # calibration window (matches previous experiments)
CALIB_OFFSET = 5000          # skip Gutenberg header

# Context lengths to test in sub-experiment A (tokens)
# Qwen3-14B-AWQ max is 40960; include a few stops below and near max
CTX_LENGTHS  = [512, 1024, 2048, 4096, 8192, 16384, 32768, 40960]

# Context length for per-token analysis (sub-exp B)
PERTOK_CTX   = 16384

# Compression configs: name -> (K_method, K_k, K_bits, V_method, V_k, V_bits)
# None for method means no compression
CONFIGS = {
    'baseline':    (None, None, None, None, None, None),
    'k128_4bit':   ('subspace', 128, 4,  'full_dim', 128, 4),
    'k112_4bit':   ('subspace', 112, 4,  'full_dim', 128, 4),
    'k96_4bit':    ('subspace',  96, 4,  'full_dim', 128, 4),
    'k64_4bit':    ('subspace',  64, 4,  'full_dim', 128, 4),
}


# ── Tokenize long text ────────────────────────────────────────────────────────

def load_tokens(tokenizer, data_file, offset, n_tokens, device):
    """Load a slice of tokenized text starting at character offset."""
    with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    # Start past the Gutenberg header
    text = text[offset:]
    inputs = tokenizer(
        text, return_tensors='pt', truncation=True,
        max_length=n_tokens + 1,  # +1 for label shift
        add_special_tokens=True,
    )
    return inputs['input_ids'].to(device)  # (1, ≤n_tokens+1)


# ── PCA basis fitting ─────────────────────────────────────────────────────────

def collect_kvs_for_basis(model, tokenizer, data_file, offset, n_tokens, device,
                          n_kv_heads, d_head):
    """
    Run a forward pass to collect KV vectors for PCA basis fitting.
    Returns: dict {(layer, head): {'K': np.array(T, d), 'V': np.array(T, d)}}
    """
    input_ids = load_tokens(tokenizer, data_file, offset, n_tokens, device)
    if input_ids.shape[1] > n_tokens:
        input_ids = input_ids[:, :n_tokens]

    kv_store = {}
    hooks = []
    attn_layers = find_attention_layers(model)

    for layer_idx, attn in attn_layers:
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_capture(li, kvt, nh, dh):
                def hook(module, input, output):
                    x = output.detach().cpu().float()
                    # output shape: (batch, seq, nh * dh)
                    b, s, _ = x.shape
                    x = x.reshape(b, s, nh, dh)[0]  # (s, nh, dh)
                    key = (li, kvt)
                    if key not in kv_store:
                        kv_store[key] = []
                    kv_store[key].append(x.numpy())  # won't accumulate — one pass
                return hook
            h = getattr(attn, proj_name).register_forward_hook(
                make_capture(layer_idx, kv_type, n_kv_heads, d_head)
            )
            hooks.append(h)

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    # Merge to {(layer, head): {'K':..., 'V':...}}
    bases_raw = {}
    for (layer_idx, kv_type), arrays in kv_store.items():
        arr = np.concatenate(arrays, axis=0)  # (T, n_heads, d_head)
        for head_idx in range(arr.shape[1]):
            key = (layer_idx, head_idx)
            if key not in bases_raw:
                bases_raw[key] = {}
            bases_raw[key][kv_type] = arr[:, head_idx, :]  # (T, d_head)

    return bases_raw


def fit_bases(kvs_raw, k):
    """Fit PCA bases on collected KV vectors."""
    bases = {}
    for (layer_idx, head_idx), kv in kvs_raw.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        U_v, mean_v = fit_pca(kv['V'], k)
        bases[(layer_idx, head_idx)] = {
            'U_K': U_k, 'mean_K': mean_k,
            'U_V': U_v, 'mean_V': mean_v,
            'k': k,
        }
    return bases


# ── Compression hooks ─────────────────────────────────────────────────────────

def compress_vec(x_np, method, k, n_bits, U, mean):
    if method == 'subspace':
        return subspace_polar_quantize(x_np, k, n_bits, U, mean)
    elif method == 'full_dim':
        return polar_quantize(x_np, n_bits)
    return x_np


def install_hooks(model, cfg_name, cfg, bases, n_kv_heads, d_head):
    K_method, K_k, K_bits, V_method, V_k, V_bits = cfg
    hooks = []
    attn_layers = find_attention_layers(model)
    for layer_idx, attn in attn_layers:
        for kv_type, proj_name, method, k, bits in [
            ('K', 'k_proj', K_method, K_k, K_bits),
            ('V', 'v_proj', V_method, V_k, V_bits),
        ]:
            if method is None:
                continue
            def make_hook(li, kvt, m, kk, nb):
                def hook(module, input, output):
                    device, dtype = output.device, output.dtype
                    x = output.detach().cpu().float()
                    b, s, _ = x.shape
                    x = x.reshape(b, s, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = x[0, :, h, :].numpy()
                        base = bases.get((li, h), {})
                        U = base.get(f'U_{kvt}')
                        mn = base.get(f'mean_{kvt}')
                        x[0, :, h, :] = torch.from_numpy(
                            compress_vec(xh, m, kk, nb, U, mn)
                        )
                    return x.reshape(b, s, -1).to(device=device, dtype=dtype)
                return hook
            proj = getattr(attn, proj_name)
            hooks.append(proj.register_forward_hook(make_hook(layer_idx, kv_type, method, k, bits)))
    return hooks


# ── Sub-experiment A: PPL vs context length ───────────────────────────────────

def load_existing_subexp_A(csv_path):
    """Load already-completed rows from a partial CSV, if it exists."""
    import os
    if not os.path.exists(csv_path):
        return []
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['ctx_len'] = int(row['ctx_len'])
            row['ppl'] = float(row['ppl'])
            row['n_tokens'] = int(row['n_tokens'])
            row['relative_ppl'] = float(row['relative_ppl']) if 'relative_ppl' in row else 1.0
            rows.append(row)
    return rows


def save_subexp_A_partial(rows, csv_path):
    """Save partial results with relative_ppl computed for completed ctx_lens."""
    if not rows:
        return
    # Recompute relative_ppl for all rows with a complete baseline
    baseline_ppl = {r['ctx_len']: r['ppl'] for r in rows if r['config'] == 'baseline'}
    for r in rows:
        r['relative_ppl'] = r['ppl'] / baseline_ppl.get(r['ctx_len'], 1.0)
    Path('results').mkdir(exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ctx_len', 'config', 'ppl', 'n_tokens', 'relative_ppl'])
        w.writeheader()
        w.writerows(rows)


def run_subexp_A(model, tokenizer, bases_by_k, device, data_file, n_kv_heads, d_head):
    """
    For each (config, context_length), compute perplexity on the SAME text slice.
    All configs use the same calibration basis (fitted on first CALIB_TOKENS tokens).
    Supports resume: skips (ctx_len, config) pairs already in results/long_context_ppl.csv.
    Saves incrementally after each ctx_len completes.
    """
    print("\n" + "="*70)
    print("Sub-experiment A: PPL vs Context Length")
    print("="*70)

    csv_path = 'results/long_context_ppl.csv'
    rows = load_existing_subexp_A(csv_path)
    done = {(r['ctx_len'], r['config']) for r in rows}
    if done:
        print(f"  Resuming: {len(done)} (ctx_len, config) pairs already done")

    # Pre-tokenize the full max-context slice once
    full_ids = load_tokens(tokenizer, data_file, CALIB_OFFSET + CALIB_TOKENS + 500,
                           CTX_LENGTHS[-1] + 1, device)

    for ctx_len in CTX_LENGTHS:
        # Check if this ctx_len is fully done already
        configs_needed = [cn for cn in CONFIGS if (ctx_len, cn) not in done]
        if not configs_needed:
            print(f"  ctx={ctx_len:6d}  [all configs done, skipping]")
            continue

        # Use a different text slice than calibration (offset by CALIB_TOKENS + buffer)
        input_ids = full_ids[:, :ctx_len + 1]
        if input_ids.shape[1] < ctx_len + 1:
            print(f"  WARNING: only {input_ids.shape[1]-1} tokens available at ctx={ctx_len}, skipping")
            continue
        eval_ids = input_ids[:, :ctx_len + 1]

        for cfg_name in configs_needed:
            cfg = CONFIGS[cfg_name]
            K_method, K_k, K_bits, V_method, V_k, V_bits = cfg
            # Pick the right basis set
            k_for_basis = K_k if K_k is not None else 128
            bases = bases_by_k.get(k_for_basis, bases_by_k[128])

            hooks = install_hooks(model, cfg_name, cfg, bases, n_kv_heads, d_head)
            loss = chunked_cross_entropy(model, eval_ids, chunk_size=512)
            for h in hooks:
                h.remove()

            ppl = float(torch.exp(torch.tensor(loss)))
            print(f"  ctx={ctx_len:6d}  config={cfg_name:12s}  PPL={ppl:.4f}")
            rows.append({
                'ctx_len': ctx_len,
                'config': cfg_name,
                'ppl': ppl,
                'n_tokens': ctx_len,
                'relative_ppl': 1.0,  # placeholder, recomputed in save
            })
        torch.cuda.empty_cache()
        # Save after each ctx_len completes
        save_subexp_A_partial(rows, csv_path)
        print(f"  [saved partial results to {csv_path}]")

    # Final relative_ppl computation
    baseline_ppl = {r['ctx_len']: r['ppl'] for r in rows if r['config'] == 'baseline'}
    for r in rows:
        r['relative_ppl'] = r['ppl'] / baseline_ppl.get(r['ctx_len'], 1.0)

    return rows


# ── Sub-experiment B: Per-token loss at 16K ───────────────────────────────────

def run_subexp_B(model, tokenizer, bases_by_k, device, data_file, n_kv_heads, d_head):
    """
    At PERTOK_CTX tokens, compute per-token cross-entropy for baseline and two configs.
    Reveals whether compression error concentrates at early or late positions.
    """
    print("\n" + "="*70)
    print(f"Sub-experiment B: Per-token loss at ctx={PERTOK_CTX}")
    print("="*70)

    configs_b = {
        'baseline': CONFIGS['baseline'],
        'k112_4bit': CONFIGS['k112_4bit'],
        'k64_4bit': CONFIGS['k64_4bit'],
    }

    full_ids = load_tokens(tokenizer, data_file, CALIB_OFFSET + CALIB_TOKENS + 500,
                           PERTOK_CTX + 1, device)
    if full_ids.shape[1] < PERTOK_CTX + 1:
        print(f"  WARNING: only {full_ids.shape[1]} tokens, truncating to {full_ids.shape[1]-1}")
    eval_ids = full_ids[:, :PERTOK_CTX + 1]
    actual_ctx = eval_ids.shape[1] - 1

    rows = []
    for cfg_name, cfg in configs_b.items():
        K_method, K_k, K_bits, V_method, V_k, V_bits = cfg
        k_for_basis = K_k if K_k is not None else 128
        bases = bases_by_k.get(k_for_basis, bases_by_k[128])

        transformer_body, lm_head = _get_transformer_body_and_head(model)
        hooks = install_hooks(model, cfg_name, cfg, bases, n_kv_heads, d_head)
        with torch.no_grad():
            outputs = transformer_body(input_ids=eval_ids[:, :-1])
            hidden = outputs.last_hidden_state  # (1, seq_len, hidden_dim)
        for h in hooks:
            h.remove()

        # Compute per-token loss in chunks to avoid ~5 GB logit tensor at 16K ctx
        labels_flat = eval_ids[:, 1:].view(-1)
        per_tok_loss_list = []
        chunk_size = 512
        with torch.no_grad():
            for start in range(0, hidden.shape[1], chunk_size):
                end = min(start + chunk_size, hidden.shape[1])
                chunk_logits = lm_head(hidden[:, start:end, :])
                chunk_labels = labels_flat[start:end]
                chunk_loss = torch.nn.functional.cross_entropy(
                    chunk_logits.view(-1, chunk_logits.size(-1)),
                    chunk_labels,
                    reduction='none',
                )
                per_tok_loss_list.append(chunk_loss.cpu())
                del chunk_logits, chunk_labels, chunk_loss
                torch.cuda.empty_cache()
        del hidden
        torch.cuda.empty_cache()
        per_tok_loss = torch.cat(per_tok_loss_list).numpy()

        print(f"  {cfg_name}: mean_loss={per_tok_loss.mean():.4f}  "
              f"early_50={per_tok_loss[:50].mean():.4f}  "
              f"late_50={per_tok_loss[-50:].mean():.4f}")

        for pos, loss_val in enumerate(per_tok_loss):
            rows.append({
                'config': cfg_name,
                'position': pos,
                'loss': float(loss_val),
                'ppl': float(np.exp(loss_val)),
            })
        torch.cuda.empty_cache()

    return rows


# ── Sub-experiment C: Basis drift across document positions ───────────────────

def run_subexp_C(model, tokenizer, device, data_file, n_kv_heads, d_head):
    """
    Collect KV vectors at multiple document positions and measure PCA subspace overlap.
    Tests whether the calibration basis fitted at position 0 transfers to later positions.
    """
    if not HAS_SCIPY:
        print("  Skipping sub-exp C: scipy not available")
        return []

    print("\n" + "="*70)
    print("Sub-experiment C: Calibration Basis Drift Across Document Positions")
    print("="*70)

    # Sample windows at different positions
    # Each window is CALIB_TOKENS tokens; spaced throughout the document
    char_offsets = [
        CALIB_OFFSET,                          # early (calibration position)
        CALIB_OFFSET + 50000,                  # ~5K tokens in
        CALIB_OFFSET + 150000,                 # ~15K tokens in
        CALIB_OFFSET + 400000,                 # ~40K tokens in (late doc)
    ]
    position_labels = ['early', 'mid_early', 'mid_late', 'late']

    # Collect KV bases for each position
    all_bases = {}
    for pos_label, char_offset in zip(position_labels, char_offsets):
        print(f"  Collecting KVs at {pos_label} (char_offset={char_offset})...")
        kvs_raw = collect_kvs_for_basis(
            model, tokenizer, data_file, char_offset, CALIB_TOKENS,
            device, n_kv_heads, d_head
        )
        all_bases[pos_label] = fit_bases(kvs_raw, k=64)
        torch.cuda.empty_cache()

    # Measure overlap between early (reference) and each other position
    reference = 'early'
    rows = []
    attn_layers = find_attention_layers(model)
    layer_indices = [li for li, _ in attn_layers]

    for compare_label in position_labels:
        if compare_label == reference:
            continue
        print(f"  Computing overlap: {reference} vs {compare_label} ...")
        for layer_idx in layer_indices:
            for head_idx in range(n_kv_heads):
                key = (layer_idx, head_idx)
                ref_bases = all_bases[reference].get(key)
                cmp_bases = all_bases[compare_label].get(key)
                if ref_bases is None or cmp_bases is None:
                    continue
                for kv_type in ['K', 'V']:
                    U_ref = ref_bases[f'U_{kv_type}']  # (d_head, k)
                    U_cmp = cmp_bases[f'U_{kv_type}']
                    if U_ref is None or U_cmp is None:
                        continue
                    angles = subspace_angles(U_ref, U_cmp)
                    overlap = float(np.mean(np.cos(angles) ** 2))
                    rows.append({
                        'reference': reference,
                        'compare': compare_label,
                        'layer': layer_idx,
                        'head': head_idx,
                        'kv_type': kv_type,
                        'overlap': overlap,
                        'n_principal_angles': len(angles),
                        'mean_cos2': overlap,
                    })

    # Summary
    if rows:
        for cmp in [r['compare'] for r in rows]:
            for kv_type in ['K', 'V']:
                subset = [r['overlap'] for r in rows
                          if r['compare'] == cmp and r['kv_type'] == kv_type]
                if subset:
                    print(f"  {reference} vs {cmp}  {kv_type}: mean_overlap={np.mean(subset):.3f}")
    return rows


# ── Report ────────────────────────────────────────────────────────────────────

def write_report(rows_a, rows_b, rows_c):
    lines = [
        "# Experiment 13: Long-Context KV Cache Compression Scaling\n",
        "## Setup\n",
        "- Model: Qwen3-14B-AWQ (40 layers, 8 GQA KV heads, d_head=128)",
        "- Text: War and Peace (Project Gutenberg) — continuous long document",
        f"- Calibration: first {CALIB_TOKENS} tokens of document",
        "- Evaluation: held-out text (offset from calibration to avoid overlap)\n",
        "## Sub-experiment A: PPL vs Context Length\n",
    ]

    if rows_a:
        configs = list(dict.fromkeys(r['config'] for r in rows_a))
        ctx_lens = sorted(set(r['ctx_len'] for r in rows_a))

        lines += [
            "### Absolute PPL\n",
            "| Config | " + " | ".join(str(c) for c in ctx_lens) + " |",
            "|--------|" + "|".join("------" for _ in ctx_lens) + "|",
        ]
        for cfg in configs:
            vals = []
            for c in ctx_lens:
                v = next((r['ppl'] for r in rows_a if r['config']==cfg and r['ctx_len']==c), None)
                vals.append(f"{v:.3f}" if v is not None else "—")
            lines.append(f"| {cfg} | " + " | ".join(vals) + " |")

        lines += [
            "\n### Relative PPL (compressed / baseline)\n",
            "| Config | " + " | ".join(str(c) for c in ctx_lens) + " |",
            "|--------|" + "|".join("------" for _ in ctx_lens) + "|",
        ]
        for cfg in configs:
            if cfg == 'baseline':
                continue
            vals = []
            for c in ctx_lens:
                v = next((r['relative_ppl'] for r in rows_a if r['config']==cfg and r['ctx_len']==c), None)
                vals.append(f"{v:.3f}" if v is not None else "—")
            lines.append(f"| {cfg} | " + " | ".join(vals) + " |")

        # Check for drift
        lines.append("\n### Key Finding: Does Relative PPL Drift With Context Length?\n")
        for cfg in configs:
            if cfg == 'baseline':
                continue
            cfg_rows = sorted([r for r in rows_a if r['config']==cfg], key=lambda x: x['ctx_len'])
            if len(cfg_rows) >= 2:
                short = cfg_rows[0]['relative_ppl']
                long  = cfg_rows[-1]['relative_ppl']
                delta = long - short
                verdict = "STABLE" if abs(delta) < 0.05 else ("DEGRADING" if delta > 0 else "IMPROVING")
                lines.append(f"- **{cfg}**: short-ctx rel_PPL={short:.3f}, long-ctx={long:.3f}, delta={delta:+.3f} → **{verdict}**")

    if rows_b:
        lines += [
            "\n## Sub-experiment B: Per-Token Loss at 16K Context\n",
            "Are compression errors uniform across sequence positions, "
            "or do they concentrate in later tokens?\n",
        ]
        configs_b = list(dict.fromkeys(r['config'] for r in rows_b))
        ctx_len_b = max(r['position'] for r in rows_b) + 1
        # Bin positions into deciles
        bin_size = ctx_len_b // 10
        lines.append("Mean loss per decile of sequence (0%=early, 100%=late):\n")
        lines.append("| Config | " + " | ".join(f"{i*10}-{(i+1)*10}%" for i in range(10)) + " |")
        lines.append("|--------|" + "|".join("------" for _ in range(10)) + "|")
        for cfg in configs_b:
            cfg_losses = sorted([r for r in rows_b if r['config']==cfg], key=lambda x: x['position'])
            bins = []
            for i in range(10):
                start, end = i*bin_size, min((i+1)*bin_size, ctx_len_b)
                bin_losses = [r['loss'] for r in cfg_losses if start <= r['position'] < end]
                bins.append(f"{np.mean(bin_losses):.3f}" if bin_losses else "—")
            lines.append(f"| {cfg} | " + " | ".join(bins) + " |")

    if rows_c:
        lines += [
            "\n## Sub-experiment C: Calibration Basis Drift\n",
            "Subspace overlap (cos²θ, 1=identical, 0=orthogonal) between PCA basis "
            "fitted at document start vs later positions:\n",
            "| Compare Position | K overlap | V overlap |",
            "|-----------------|-----------|-----------|",
        ]
        compare_positions = list(dict.fromkeys(r['compare'] for r in rows_c))
        for cmp in compare_positions:
            k_ovlp = np.mean([r['overlap'] for r in rows_c if r['compare']==cmp and r['kv_type']=='K'])
            v_ovlp = np.mean([r['overlap'] for r in rows_c if r['compare']==cmp and r['kv_type']=='V'])
            lines.append(f"| early vs {cmp} | {k_ovlp:.3f} | {v_ovlp:.3f} |")
        lines.append("\nA drop in overlap indicates the PCA basis fitted on early tokens "
                     "is less representative of late-document KV distributions.")

    lines += [
        "\n## Conclusions and Recommendations\n",
        "See per-section findings above. Key question: does compression quality "
        "hold at long context, or is there a context length beyond which the "
        "calibrated basis and/or cascade effects cause meaningful degradation?\n",
        "---",
        "*Experiment 13 — part of the KV cache subspace compression research.*",
        "*Repo: https://github.com/corpetty/mozeika-pruning-empirics*",
    ]

    with open('results/REPORT-13-long-context.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print("Wrote results/REPORT-13-long-context.md")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    device = 'cuda'

    print("=" * 70)
    print("Experiment 13: Long-Context KV Cache Compression Scaling")
    print("=" * 70)
    print(f"Model:        {MODEL_NAME}")
    print(f"Data file:    {DATA_FILE}")
    print(f"Calib tokens: {CALIB_TOKENS}")
    print(f"Ctx lengths:  {CTX_LENGTHS}")
    print(f"Per-tok ctx:  {PERTOK_CTX}")

    # Load model
    print("\nLoading model...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    print(f"  n_kv_heads={n_kv_heads}, d_head={d_head}")

    # Collect calibration KVs and fit bases for each k we'll need
    print(f"\nCollecting calibration KVs ({CALIB_TOKENS} tokens)...")
    calib_kvs = collect_kvs_for_basis(
        model, tokenizer, DATA_FILE, CALIB_OFFSET, CALIB_TOKENS,
        device, n_kv_heads, d_head
    )
    print(f"  Collected KVs for {len(calib_kvs)} (layer, head) pairs")

    # Fit bases for each k value used in CONFIGS
    k_values = sorted(set(
        cfg[1] for cfg in CONFIGS.values() if cfg[1] is not None
    ))
    print(f"\nFitting PCA bases for k values: {k_values}")
    bases_by_k = {}
    for k in k_values:
        bases_by_k[k] = fit_bases(calib_kvs, k)
        print(f"  k={k}: {len(bases_by_k[k])} bases fitted")

    # Sub-experiment A
    rows_a = run_subexp_A(model, tokenizer, bases_by_k, device, DATA_FILE,
                          n_kv_heads, d_head)

    # Sub-experiment B
    rows_b = run_subexp_B(model, tokenizer, bases_by_k, device, DATA_FILE,
                          n_kv_heads, d_head)

    # Sub-experiment C (basis drift)
    rows_c = run_subexp_C(model, tokenizer, device, DATA_FILE, n_kv_heads, d_head)

    # Save CSVs
    Path('results').mkdir(exist_ok=True)

    # Sub-exp A CSV already saved incrementally by run_subexp_A(); do a final write to ensure complete.
    if rows_a:
        with open('results/long_context_ppl.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows_a[0].keys()))
            w.writeheader(); w.writerows(rows_a)
    print("Saved results/long_context_ppl.csv")

    with open('results/long_context_per_token.csv', 'w', newline='') as f:
        if rows_b:
            w = csv.DictWriter(f, fieldnames=list(rows_b[0].keys()))
            w.writeheader(); w.writerows(rows_b)
    print("Saved results/long_context_per_token.csv")

    with open('results/long_context_basis_drift.csv', 'w', newline='') as f:
        if rows_c:
            w = csv.DictWriter(f, fieldnames=list(rows_c[0].keys()))
            w.writeheader(); w.writerows(rows_c)
    print("Saved results/long_context_basis_drift.csv")

    # Write report
    write_report(rows_a, rows_b, rows_c)

    print("\n" + "="*70)
    print("Experiment 13 complete.")
    print("="*70)


if __name__ == '__main__':
    main()
