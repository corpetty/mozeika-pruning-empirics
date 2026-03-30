"""
Experiment 22: Quantizer comparison — SubRotQ vs PolarQuant.

The current pipeline (experiments 1–21) used SubRotQ (random QR rotation +
uniform scalar quantization), which was mistakenly labelled 'PolarQuant' in early
experiments. This experiment runs both backends head-to-head at the same
(k, bits) configurations used in the core bitrate sweep (Exp 9), producing a
direct comparison of compression quality under matched bit budgets.

Configurations:
  k ∈ {64, 96, 112, 128}  ×  bits ∈ {4, 8}  ×  quantizer ∈ {subrotq, polarquant}
  Plus: k=128 / bits=16 (no quantization noise, pure truncation reference)

Output:
  results/exp22_quantizer_comparison.csv
  results/REPORT-22-quantizer-comparison.md
"""

import sys
import csv
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect import get_model_and_tokenizer
from compress import fit_pca, subspace_compress, random_rotation_matrix

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
DATA_FILE    = Path("data/war_and_peace.txt")
RESULTS_DIR  = Path("results")

CALIB_OFFSET = 0
CALIB_TOKENS = 2048
EVAL_OFFSET  = CALIB_TOKENS + 100   # ensure no calib/eval overlap
EVAL_TOKENS  = 1024

N_KV_HEADS = 8
D_HEAD     = 128
N_LAYERS   = 40

K_VALUES     = [64, 96, 112, 128]
BITS_VALUES  = [4, 8]
QUANTIZERS   = ['subrotq', 'polarquant']

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_attention_layers(model):
    transformer = model.model.model
    for i, layer in enumerate(transformer.layers):
        yield i, layer.self_attn


def collect_kvs_for_basis(model, tokenizer, data_file, char_offset, n_tokens,
                           device, n_kv_heads, d_head):
    """Returns {(layer_idx, head_idx): {'K': (T, d_head), 'V': (T, d_head)}}"""
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
            hooks.append(proj.register_forward_hook(make_hook(layer_idx, kv_type, n_kv_heads, d_head)))

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
    ids = inputs['input_ids'][:, :n_tokens].to(device)
    return ids


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


def eval_ppl_with_hooks(model, tokenizer, data_file, char_offset, n_tokens,
                         device, bases, k, n_bits, quantizer):
    """Evaluate PPL with subspace compression hooks installed."""
    input_ids = load_tokens(tokenizer, data_file, char_offset, n_tokens, device)

    hooks = []
    R_cache = {}  # reuse same rotation per (layer, head) across calls

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
                        xh_c = subspace_compress(xh, k, n_bits, U, mean, R,
                                                  quantizer=quantizer)
                        x[0, :, h, :] = torch.from_numpy(xh_c)
                    return x.reshape(b, s, nh * dh).to(dty).to(dev)
                return hook
            proj = getattr(attn, proj_name)
            hooks.append(proj.register_forward_hook(make_hook(layer_idx, kv_type,
                                                               N_KV_HEADS, D_HEAD)))

    loss = chunked_cross_entropy(model, input_ids)
    for h in hooks:
        h.remove()
    return float(np.exp(loss))


def fit_bases(initial_kvs, k):
    """Fit PCA bases for all (layer, head) pairs at dimension k."""
    bases = {}
    for (li, hi), kv in initial_kvs.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        U_v, mean_v = fit_pca(kv['V'], k)
        bases[(li, hi)] = {
            'U_K': U_k, 'mean_K': mean_k,
            'U_V': U_v, 'mean_V': mean_v,
        }
    return bases


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "exp22_quantizer_comparison.csv"
    fieldnames = ["k", "bits", "quantizer", "ppl", "rel_ppl"]

    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["k"]), int(row["bits"]), row["quantizer"]))
        print(f"Resuming: {len(done)} configs done")

    print(f"Loading model {MODEL_NAME}...")
    device = 'cuda'
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    print(f"Collecting KV calibration basis ({CALIB_TOKENS} tokens)...")
    initial_kvs = collect_kvs_for_basis(
        model, tokenizer, DATA_FILE, CALIB_OFFSET, CALIB_TOKENS,
        device, N_KV_HEADS, D_HEAD)

    print("Computing baseline PPL (no compression)...")
    baseline_ppl = eval_ppl_with_hooks(
        model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_TOKENS,
        device, {}, k=D_HEAD, n_bits=16, quantizer='subrotq')
    # baseline: pass-through (no hooks actually applied since bases is empty)
    # use direct eval instead:
    input_ids = load_tokens(tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_TOKENS, device)
    loss = chunked_cross_entropy(model, input_ids)
    baseline_ppl = float(np.exp(loss))
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # Pre-fit bases for all k values
    bases_by_k = {}
    for k in K_VALUES:
        print(f"  Fitting bases at k={k}...", end='', flush=True)
        bases_by_k[k] = fit_bases(initial_kvs, k)
        print(" done")

    # K-only sweep (V always at full rank 4-bit, as established in Exp 9)
    # Actually run K+V compression to match Exp 9 methodology
    total_configs = len(K_VALUES) * len(BITS_VALUES) * len(QUANTIZERS)
    done_count = len(done)

    for k in K_VALUES:
        bases = bases_by_k[k]
        for n_bits in BITS_VALUES:
            for quantizer in QUANTIZERS:
                key = (k, n_bits, quantizer)
                if key in done:
                    print(f"  [skip] k={k} bits={n_bits} quantizer={quantizer}")
                    continue

                print(f"  k={k:3d} bits={n_bits} quantizer={quantizer:<12}", end='', flush=True)
                ppl = eval_ppl_with_hooks(
                    model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_TOKENS,
                    device, bases, k, n_bits, quantizer)
                rel_ppl = ppl / baseline_ppl
                print(f"  PPL={ppl:.4f}  rel={rel_ppl:.4f}")
                done_count += 1

                row = {"k": k, "bits": n_bits, "quantizer": quantizer,
                       "ppl": round(ppl, 4), "rel_ppl": round(rel_ppl, 4)}
                file_exists = csv_path.exists()
                with open(csv_path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        w.writeheader()
                    w.writerow(row)

    # ── Report ────────────────────────────────────────────────────────────────
    all_rows = []
    with open(csv_path) as f:
        all_rows = list(csv.DictReader(f))

    report_path = RESULTS_DIR / "REPORT-22-quantizer-comparison.md"
    with open(report_path, 'w') as f:
        f.write("# Experiment 22: Quantizer Comparison — SubRotQ vs PolarQuant\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Baseline PPL: {baseline_ppl:.4f}\n\n")
        f.write("## Results\n\n")
        f.write("| k | bits | quantizer | PPL | rel-PPL |\n")
        f.write("|---|------|-----------|-----|---------|\n")
        for row in sorted(all_rows, key=lambda r: (int(r['k']), int(r['bits']), r['quantizer'])):
            f.write(f"| {row['k']} | {row['bits']} | {row['quantizer']} | "
                    f"{float(row['ppl']):.4f} | {float(row['rel_ppl']):.4f} |\n")
        f.write("\n## Key Questions\n\n")
        f.write("1. At matched (k, bits), does PolarQuant reduce PPL vs SubRotQ?\n")
        f.write("2. Is the quantizer gap constant across k values or does it grow at lower k?\n")
        f.write("3. Does the truncation-dominance finding hold: is the k=128→k=112 PPL\n")
        f.write("   gap larger than the SubRotQ→PolarQuant quantizer gap at any k?\n\n")
        # Compute gaps
        subrotq_rows = {(int(r['k']), int(r['bits'])): float(r['rel_ppl'])
                        for r in all_rows if r['quantizer'] == 'subrotq'}
        pq_rows = {(int(r['k']), int(r['bits'])): float(r['rel_ppl'])
                   for r in all_rows if r['quantizer'] == 'polarquant'}
        if subrotq_rows and pq_rows:
            f.write("## Quantizer Gap (PolarQuant rel_ppl - SubRotQ rel_ppl)\n\n")
            f.write("Negative = PolarQuant is better.\n\n")
            f.write("| k | bits | SubRotQ | PolarQuant | gap |\n")
            f.write("|---|------|---------|------------|-----|\n")
            for (k, bits) in sorted(subrotq_rows.keys()):
                if (k, bits) in pq_rows:
                    s = subrotq_rows[(k, bits)]
                    p = pq_rows[(k, bits)]
                    gap = p - s
                    f.write(f"| {k} | {bits} | {s:.4f} | {p:.4f} | {gap:+.4f} |\n")

    print(f"\nReport: {report_path}")
    print(f"CSV:    {csv_path}")
    print(f"\nBaseline PPL: {baseline_ppl:.4f}")


if __name__ == "__main__":
    main()
