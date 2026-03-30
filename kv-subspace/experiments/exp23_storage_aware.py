"""
Experiment 23: Storage-aware quantizer comparison.

The key question from C1: is our PolarQuant implementation correct, and how
do SubRotQ and PolarQuant actually compare when storage overhead is properly
accounted for?

## The storage problem with SubRotQ

SubRotQ (quantize_uniform) uses per-column min/max statistics to scale each
dimension. For a batch of N vectors with d dimensions, it stores:
  - quantized data:  N × d × n_bits  bits
  - scale/offset:    2 × d × 32      bits  (float32 per dimension)

For autoregressive inference (one token at a time, N=1):
  SubRotQ overhead = 2×128×32 = 8192 bits = 1024 bytes
  Quantized data   = 128×4 = 512 bits = 64 bytes
  Total            = 1088 bytes vs 256 bytes FP16 → EXPANSION

For group quantization (G tokens share scale/offset):
  Overhead amortized to 2×128×32/G bits per vector
  At G≥64, overhead < 10% of quantized data at 4-bit

PolarQuant stores:
  - 1 float32 radius (32 bits)
  - (d-1) × n_bits angle bits
  Total per vector at d=128, 4-bit = 32 + 127×4 = 540 bits = 67.5 bytes
  No per-vector or per-group stats needed.

## What this experiment measures

1. PPL quality of SubRotQ at group sizes G = {1, 8, 16, 64, 128, 512, 2048}
   → SubRotQ G=2048 ≈ our existing exp22 results (whole eval set as one group)
   → SubRotQ G=1 = true single-token: requires storing scale/offset per token

2. PPL quality of PolarQuant (always per-vector)

3. Effective bits per element (bpe) for each config:
   bpe = total_bits / (N × d)

4. Quality vs effective bpe tradeoff curves for both methods

Note: PPL measurement is the same as exp22 (batch mode) — all scale/offset
computed over the eval window. What changes is the REPORTED storage cost.
The PPL for SubRotQ G=1 is estimated by simulating group-size-1 quantization
(quantize each vector independently with its own min/max).

Output:
  results/exp23_storage_aware.csv
  results/REPORT-23-storage-aware.md
"""

import sys
import csv
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect import get_model_and_tokenizer
from compress import (fit_pca, random_rotation_matrix,
                      polar_quantize_true, subspace_compress)

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
DATA_FILE    = Path("data/war_and_peace.txt")
RESULTS_DIR  = Path("results")

CALIB_OFFSET = 0
CALIB_TOKENS = 2048
EVAL_OFFSET  = CALIB_TOKENS + 100
EVAL_TOKENS  = 1024

N_KV_HEADS = 8
D_HEAD     = 128
N_LAYERS   = 40

# Configurations to test
K_VALUES        = [112, 128]          # focus on the production-relevant configs
BITS_VALUES     = [4, 8]
GROUP_SIZES     = [1, 8, 16, 64, 128, 512, 2048]  # SubRotQ group sizes
QUANTIZERS      = ['subrotq_grouped', 'polarquant']

# ── Storage cost calculation ───────────────────────────────────────────────────

def subrotq_storage_bpe(d: int, n_bits: int, group_size: int) -> float:
    """
    Effective bits per element for SubRotQ with given group size.
    Per group of G vectors:
      quantized: G × d × n_bits bits
      overhead:  2 × d × 32 bits (scale + offset per dimension)
    bpe = total / (G × d)
    """
    quantized_bits = group_size * d * n_bits
    overhead_bits  = 2 * d * 32  # float32 scale and offset per dimension
    total_bits = quantized_bits + overhead_bits
    return total_bits / (group_size * d)


def polarquant_storage_bpe(d: int, n_bits: int) -> float:
    """
    Effective bits per element for PolarQuant.
    Per vector:
      radius:   32 bits (float32)
      angles:   (d-1) × n_bits bits
    bpe = total / d
    """
    total_bits = 32 + (d - 1) * n_bits
    return total_bits / d


def compute_compression_ratio(d: int, n_bits_fp: int = 16, **kwargs) -> float:
    """CR = fp16_bpe / effective_bpe"""
    bpe = kwargs.get('bpe')
    return n_bits_fp / bpe


# ── SubRotQ group quantization ─────────────────────────────────────────────────

def subrotq_group_quantize(x: np.ndarray, n_bits: int, group_size: int,
                            R: np.ndarray) -> np.ndarray:
    """
    SubRotQ with explicit group quantization.
    x: (N, d)
    Groups N vectors into chunks of group_size, computes per-group scale/offset.
    """
    N, d = x.shape
    xr = x @ R.T
    out = np.zeros_like(xr)
    n_levels = 2 ** n_bits

    for start in range(0, N, group_size):
        end = min(start + group_size, N)
        chunk = xr[start:end]          # (G, d)
        x_min = chunk.min(axis=0)      # (d,)
        x_max = chunk.max(axis=0)      # (d,)
        scale = (x_max - x_min) / (n_levels - 1)
        scale = np.where(scale == 0, 1.0, scale)
        x_int = np.clip(np.round((chunk - x_min) / scale), 0, n_levels - 1)
        out[start:end] = x_int * scale + x_min

    return out @ R


# ── Subspace compression with grouped SubRotQ ─────────────────────────────────

def subspace_compress_grouped(x: np.ndarray, k: int, n_bits: int,
                               U_k: np.ndarray, mean: np.ndarray,
                               R: np.ndarray, group_size: int) -> np.ndarray:
    """Subspace compress with group-quantized SubRotQ."""
    xc = x - mean
    x_proj = xc @ U_k           # (N, k)
    x_proj_q = subrotq_group_quantize(x_proj, n_bits, group_size, R)
    return x_proj_q @ U_k.T + mean


def subspace_compress_polar(x: np.ndarray, k: int, n_bits: int,
                             U_k: np.ndarray, mean: np.ndarray,
                             R: np.ndarray) -> np.ndarray:
    """Subspace compress with PolarQuant."""
    xc = x - mean
    x_proj = xc @ U_k           # (N, k)
    x_proj_q = polar_quantize_true(x_proj, n_bits, R)
    return x_proj_q @ U_k.T + mean


# ── Model helpers ──────────────────────────────────────────────────────────────

def find_attention_layers(model):
    for i, layer in enumerate(model.model.model.layers):
        yield i, layer.self_attn


def collect_kvs_for_basis(model, tokenizer, data_file, char_offset, n_tokens,
                           device, n_kv_heads, d_head):
    text = data_file.read_text(encoding='utf-8', errors='replace')
    chunk = text[char_offset: char_offset + n_tokens * 6]
    inputs = tokenizer(chunk, return_tensors='pt', truncation=True,
                       max_length=n_tokens + 1, add_special_tokens=True)
    input_ids = inputs['input_ids'][:, :n_tokens].to(device)

    kv_store = {}
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_hook(li, kvt, nh, dh):
                def hook(module, input, output):
                    x = output.detach().cpu().float()
                    x = x.reshape(x.shape[0], x.shape[1], nh, dh)[0]
                    for h in range(nh):
                        key = (li, h)
                        if key not in kv_store:
                            kv_store[key] = {'K': [], 'V': []}
                        kv_store[key][kvt].append(x[:, h, :].numpy())
                return hook
            proj = getattr(attn, proj_name)
            hooks.append(proj.register_forward_hook(
                make_hook(layer_idx, kv_type, n_kv_heads, d_head)))

    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()

    return {k: {kv: np.concatenate(v, axis=0) for kv, v in d.items()}
            for k, d in kv_store.items()}


def load_tokens(tokenizer, data_file, char_offset, n_tokens, device):
    text = data_file.read_text(encoding='utf-8', errors='replace')
    chunk = text[char_offset: char_offset + n_tokens * 6]
    inputs = tokenizer(chunk, return_tensors='pt', truncation=True,
                       max_length=n_tokens + 1, add_special_tokens=True)
    return inputs['input_ids'][:, :n_tokens].to(device)


def chunked_cross_entropy(model, input_ids, chunk_size=256):
    transformer_body = model.model.model
    lm_head = model.model.lm_head
    with torch.no_grad():
        outputs = transformer_body(input_ids=input_ids[:, :-1])
        hidden = outputs.last_hidden_state
    labels = input_ids[:, 1:].view(-1)
    total_loss, n_tok = 0.0, 0
    with torch.no_grad():
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            chunk_logits = lm_head(hidden[:, start:end, :])
            chunk_labels = labels[start:end]
            loss = torch.nn.functional.cross_entropy(
                chunk_logits.view(-1, chunk_logits.size(-1)), chunk_labels)
            total_loss += float(loss) * (end - start)
            n_tok += (end - start)
            del chunk_logits, chunk_labels, loss
            torch.cuda.empty_cache()
    del hidden
    torch.cuda.empty_cache()
    return total_loss / n_tok


def eval_ppl_subrotq_grouped(model, tokenizer, data_file, char_offset, n_tokens,
                              device, bases, k, n_bits, group_size):
    """Evaluate PPL with grouped SubRotQ compression hooks."""
    input_ids = load_tokens(tokenizer, data_file, char_offset, n_tokens, device)
    hooks = []
    R_cache = {}

    for layer_idx, attn in find_attention_layers(model):
        for proj_name, kv_type in [('k_proj', 'K'), ('v_proj', 'V')]:
            def make_hook(li, kvt, nh, dh):
                def hook(module, input, output):
                    dev, dty = output.device, output.dtype
                    x = output.detach().cpu().float()
                    b, s, _ = x.shape
                    x = x.reshape(b, s, nh, dh)
                    for h in range(nh):
                        key_bh = (li, h)
                        if key_bh not in bases:
                            continue
                        xh = x[0, :, h, :].numpy()
                        U = bases[key_bh][f'U_{kvt}']
                        mean = bases[key_bh][f'mean_{kvt}']
                        R_key = (li, h, kvt)
                        if R_key not in R_cache:
                            R_cache[R_key] = random_rotation_matrix(k)
                        R = R_cache[R_key]
                        xh_c = subspace_compress_grouped(xh, k, n_bits, U, mean, R,
                                                          group_size=group_size)
                        x[0, :, h, :] = torch.from_numpy(xh_c)
                    return x.reshape(b, s, nh * dh).to(dty).to(dev)
                return hook
            proj = getattr(attn, proj_name)
            hooks.append(proj.register_forward_hook(
                make_hook(layer_idx, kv_type, N_KV_HEADS, D_HEAD)))

    loss = chunked_cross_entropy(model, input_ids)
    for h in hooks:
        h.remove()
    return float(np.exp(loss))


def eval_ppl_polarquant(model, tokenizer, data_file, char_offset, n_tokens,
                         device, bases, k, n_bits):
    """Evaluate PPL with PolarQuant compression hooks."""
    input_ids = load_tokens(tokenizer, data_file, char_offset, n_tokens, device)
    hooks = []
    R_cache = {}

    for layer_idx, attn in find_attention_layers(model):
        for proj_name, kv_type in [('k_proj', 'K'), ('v_proj', 'V')]:
            def make_hook(li, kvt, nh, dh):
                def hook(module, input, output):
                    dev, dty = output.device, output.dtype
                    x = output.detach().cpu().float()
                    b, s, _ = x.shape
                    x = x.reshape(b, s, nh, dh)
                    for h in range(nh):
                        key_bh = (li, h)
                        if key_bh not in bases:
                            continue
                        xh = x[0, :, h, :].numpy()
                        U = bases[key_bh][f'U_{kvt}']
                        mean = bases[key_bh][f'mean_{kvt}']
                        R_key = (li, h, kvt)
                        if R_key not in R_cache:
                            R_cache[R_key] = random_rotation_matrix(k)
                        R = R_cache[R_key]
                        xh_c = subspace_compress_polar(xh, k, n_bits, U, mean, R)
                        x[0, :, h, :] = torch.from_numpy(xh_c)
                    return x.reshape(b, s, nh * dh).to(dty).to(dev)
                return hook
            proj = getattr(attn, proj_name)
            hooks.append(proj.register_forward_hook(
                make_hook(layer_idx, kv_type, N_KV_HEADS, D_HEAD)))

    loss = chunked_cross_entropy(model, input_ids)
    for h in hooks:
        h.remove()
    return float(np.exp(loss))


def fit_bases(kvs, k):
    bases = {}
    for (li, hi), kv in kvs.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        U_v, mean_v = fit_pca(kv['V'], k)
        bases[(li, hi)] = {
            'U_K': U_k, 'mean_K': mean_k,
            'U_V': U_v, 'mean_V': mean_v,
        }
    return bases


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "exp23_storage_aware.csv"
    fieldnames = ["k", "bits", "method", "group_size",
                  "eff_bpe", "compression_ratio", "ppl", "rel_ppl"]

    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["k"]), int(row["bits"]), row["method"],
                          int(row["group_size"])))
        print(f"Resuming: {len(done)} configs done")

    print(f"Loading model {MODEL_NAME}...")
    device = 'cuda'
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    print(f"Collecting calibration KVs ({CALIB_TOKENS} tokens)...")
    initial_kvs = collect_kvs_for_basis(
        model, tokenizer, DATA_FILE, CALIB_OFFSET, CALIB_TOKENS,
        device, N_KV_HEADS, D_HEAD)

    print("Computing baseline PPL...")
    input_ids = load_tokens(tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_TOKENS, device)
    loss = chunked_cross_entropy(model, input_ids)
    baseline_ppl = float(np.exp(loss))
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    print("Pre-fitting PCA bases...")
    bases_by_k = {}
    for k in K_VALUES:
        print(f"  k={k}...", end='', flush=True)
        bases_by_k[k] = fit_bases(initial_kvs, k)
        print(" done")

    # ── SubRotQ group sweep ────────────────────────────────────────────────────
    for k in K_VALUES:
        for n_bits in BITS_VALUES:
            for G in GROUP_SIZES:
                key = (k, n_bits, 'subrotq', G)
                if key in done:
                    print(f"  [skip] subrotq k={k} bits={n_bits} G={G}")
                    continue

                eff_bpe = subrotq_storage_bpe(k, n_bits, G)
                cr      = 16.0 / eff_bpe  # vs FP16 in k-dim subspace
                # For full-dim effective CR: account for PCA truncation too
                # Effective CR over original d-dim FP16:
                # Stored: k vectors of eff_bpe bits vs d vectors of 16 bits
                # eff_cr_full = (d * 16) / (k * eff_bpe)
                eff_cr_full = (D_HEAD * 16) / (k * eff_bpe)

                print(f"  subrotq k={k} bits={n_bits} G={G:5d}  "
                      f"eff_bpe={eff_bpe:.2f}  CR_full={eff_cr_full:.3f}x",
                      end='', flush=True)

                ppl = eval_ppl_subrotq_grouped(
                    model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_TOKENS,
                    device, bases_by_k[k], k, n_bits, group_size=G)
                rel_ppl = ppl / baseline_ppl
                print(f"  PPL={ppl:.4f}  rel={rel_ppl:.4f}")

                row = {"k": k, "bits": n_bits, "method": "subrotq",
                       "group_size": G, "eff_bpe": round(eff_bpe, 4),
                       "compression_ratio": round(eff_cr_full, 4),
                       "ppl": round(ppl, 4), "rel_ppl": round(rel_ppl, 4)}
                file_exists = csv_path.exists()
                with open(csv_path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        w.writeheader()
                    w.writerow(row)

    # ── PolarQuant (no group size — per-vector) ────────────────────────────────
    for k in K_VALUES:
        for n_bits in BITS_VALUES:
            key = (k, n_bits, 'polarquant', 1)  # group_size=1 semantically
            if key in done:
                print(f"  [skip] polarquant k={k} bits={n_bits}")
                continue

            eff_bpe  = polarquant_storage_bpe(k, n_bits)
            eff_cr_full = (D_HEAD * 16) / (k * eff_bpe)

            print(f"  polarquant k={k} bits={n_bits}  "
                  f"eff_bpe={eff_bpe:.2f}  CR_full={eff_cr_full:.3f}x",
                  end='', flush=True)

            ppl = eval_ppl_polarquant(
                model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_TOKENS,
                device, bases_by_k[k], k, n_bits)
            rel_ppl = ppl / baseline_ppl
            print(f"  PPL={ppl:.4f}  rel={rel_ppl:.4f}")

            row = {"k": k, "bits": n_bits, "method": "polarquant",
                   "group_size": 1, "eff_bpe": round(eff_bpe, 4),
                   "compression_ratio": round(eff_cr_full, 4),
                   "ppl": round(ppl, 4), "rel_ppl": round(rel_ppl, 4)}
            file_exists = csv_path.exists()
            with open(csv_path, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    w.writeheader()
                w.writerow(row)

    # ── Report ─────────────────────────────────────────────────────────────────
    all_rows = []
    with open(csv_path) as f:
        all_rows = list(csv.DictReader(f))

    report_path = RESULTS_DIR / "REPORT-23-storage-aware.md"
    with open(report_path, 'w') as f:
        f.write("# Experiment 23: Storage-Aware Quantizer Comparison\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Baseline PPL: {baseline_ppl:.4f}\n\n")

        f.write("## Storage Cost Summary\n\n")
        f.write("For d=128, n_bits=4:\n\n")
        f.write("| Method | Group Size | Effective bpe | True CR |\n")
        f.write("|--------|-----------|--------------|--------|\n")
        f.write(f"| SubRotQ | G=1    | {subrotq_storage_bpe(128,4,1):.1f} | "
                f"{(128*16)/(128*subrotq_storage_bpe(128,4,1)):.3f}x |\n")
        for G in [8, 16, 64, 128]:
            bpe = subrotq_storage_bpe(128, 4, G)
            cr_ = (128*16)/(128*bpe)
            f.write(f"| SubRotQ | G={G:<5} | {bpe:.2f}  | {cr_:.3f}x |\n")
        pq_bpe = polarquant_storage_bpe(128, 4)
        f.write(f"| PolarQuant | per-vec | {pq_bpe:.2f} | "
                f"{(128*16)/(128*pq_bpe):.3f}x |\n")
        f.write(f"| FP16 baseline | — | 16.00 | 1.000x |\n\n")

        f.write("## Quality vs True Compression Ratio\n\n")
        f.write("| k | bits | method | group | eff CR | rel-PPL |\n")
        f.write("|---|------|--------|-------|--------|--------|\n")
        for row in sorted(all_rows,
                          key=lambda r: (int(r['k']), int(r['bits']),
                                         r['method'], int(r['group_size']))):
            f.write(f"| {row['k']} | {row['bits']} | {row['method']} | "
                    f"{row['group_size']} | {float(row['compression_ratio']):.3f}x | "
                    f"{float(row['rel_ppl']):.4f} |\n")

        f.write("\n## Key Findings\n\n")
        f.write("1. **SubRotQ G=1** (autoregressive reality): stores scale/offset per token,\n"
                "   resulting in ~0.24x 'compression ratio' (actually expansion).\n")
        f.write("2. **SubRotQ G≥64** approaches PolarQuant storage cost.\n")
        f.write("3. **PolarQuant** achieves true per-vector quantization with no overhead —\n"
                "   the key practical advantage over SubRotQ for deployment.\n")
        f.write("4. At matched *effective* CR, which achieves better PPL quality?\n")

    print(f"\nReport: {report_path}")
    print(f"CSV:    {csv_path}")


if __name__ == "__main__":
    main()
