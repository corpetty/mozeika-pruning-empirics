"""
Experiment 16: Head and layer sensitivity profiling.

Answers: Which layers are most/least sensitive to KV compression?
  - Ablate one layer at a time with aggressive compression (k64_4bit)
    while keeping all other layers at baseline → PPL delta = layer sensitivity
  - Build an adaptive compression policy: compress insensitive layers more,
    sensitive layers less, with same mean-k budget as k96

Usage:
    python experiments/exp16_head_sensitivity.py

Outputs:
    results/exp16_layer_sensitivity.csv
    results/exp16_adaptive_policy.json
    results/REPORT-16-sensitivity.md
"""

import sys
import csv
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import polar_quantize, subspace_polar_quantize, fit_pca
from collect import get_model_and_tokenizer, find_attention_layers

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
DATA_FILE    = Path("data/war_and_peace.txt")
CALIB_TOKENS = 2048
CALIB_OFFSET = 5000
EVAL_CTX     = 4096
ABLATION_K   = 64
ABLATION_BITS = 4
BUDGET_K     = 96    # target mean k for adaptive policy


def load_tokens(tokenizer, data_file, char_offset, n_tokens, device):
    with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    text = text[char_offset:]
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=n_tokens + 1, add_special_tokens=True)
    return inputs['input_ids'].to(device)


def collect_kvs_for_basis(model, tokenizer, data_file, char_offset, n_tokens,
                           device, n_kv_heads, d_head):
    input_ids = load_tokens(tokenizer, data_file, char_offset, n_tokens, device)
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
                    b, s, _ = x.shape
                    x = x.reshape(b, s, nh, dh)[0]
                    key = (li, kvt)
                    if key not in kv_store:
                        kv_store[key] = []
                    kv_store[key].append(x.numpy())
                return hook
            h = getattr(attn, proj_name).register_forward_hook(
                make_capture(layer_idx, kv_type, n_kv_heads, d_head))
            hooks.append(h)

    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()

    bases_raw = {}
    for (layer_idx, kv_type), arrays in kv_store.items():
        arr = np.concatenate(arrays, axis=0)
        for head_idx in range(arr.shape[1]):
            key = (layer_idx, head_idx)
            if key not in bases_raw:
                bases_raw[key] = {}
            bases_raw[key][kv_type] = arr[:, head_idx, :]
    return bases_raw


def fit_bases(kvs_raw, k):
    bases = {}
    for (layer_idx, head_idx), kv in kvs_raw.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        U_v, mean_v = fit_pca(kv['V'], k)
        bases[(layer_idx, head_idx)] = {
            'U_K': U_k, 'mean_K': mean_k,
            'U_V': U_v, 'mean_V': mean_v,
        }
    return bases


def compress_vec(x_np, method, k, n_bits, U, mean):
    if method == 'subspace':
        return subspace_polar_quantize(x_np, k, n_bits, U, mean)
    elif method == 'full_dim':
        return polar_quantize(x_np, n_bits)
    return x_np


def _get_transformer_body_and_head(model):
    causal_lm = getattr(model, 'model', model)
    if hasattr(causal_lm, 'model') and hasattr(causal_lm, 'lm_head'):
        return causal_lm.model, causal_lm.lm_head
    return causal_lm, model.lm_head


def chunked_cross_entropy(model, input_ids, chunk_size=256):
    transformer_body, lm_head = _get_transformer_body_and_head(model)
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


def install_layer_hooks(model, ablate_layer_idx, bases, n_kv_heads, d_head, k, bits):
    """Hook only the specified layer; everything else runs baseline."""
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        if layer_idx != ablate_layer_idx:
            continue
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_hook(li, kvt, kk, nb):
                def hook(module, input, output):
                    dev, dty = output.device, output.dtype
                    x = output.detach().cpu().float()
                    b, s, _ = x.shape
                    x = x.reshape(b, s, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = x[0, :, h, :].numpy()
                        base = bases.get((li, h), {})
                        x[0, :, h, :] = torch.from_numpy(
                            compress_vec(xh, 'subspace', kk, nb,
                                         base.get(f'U_{kvt}'), base.get(f'mean_{kvt}')))
                    return x.reshape(b, s, -1).to(device=dev, dtype=dty)
                return hook
            hooks.append(getattr(attn, proj_name).register_forward_hook(
                make_hook(layer_idx, kv_type, k, bits)))
    return hooks


def main():
    print("=" * 70)
    print("Experiment 16: Layer Sensitivity Profiling")
    print("=" * 70)

    device = "cuda"
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    attn_layers = find_attention_layers(model)
    n_layers   = len(attn_layers)
    n_kv_heads = model.config.num_key_value_heads
    d_head     = model.config.hidden_size // model.config.num_attention_heads
    print(f"n_layers={n_layers}, n_kv_heads={n_kv_heads}, d_head={d_head}")

    print(f"\nCalibrating ({CALIB_TOKENS} tokens)...")
    calib_kvs = collect_kvs_for_basis(model, tokenizer, DATA_FILE, CALIB_OFFSET,
                                       CALIB_TOKENS, device, n_kv_heads, d_head)
    bases = fit_bases(calib_kvs, ABLATION_K)
    print(f"  Fitted {len(bases)} (layer, head) bases at k={ABLATION_K}")

    # Eval tokens from a non-overlapping window
    eval_offset = CALIB_OFFSET + CALIB_TOKENS * 6
    eval_ids = load_tokens(tokenizer, DATA_FILE, eval_offset, EVAL_CTX, device)
    if eval_ids.shape[1] > EVAL_CTX + 1:
        eval_ids = eval_ids[:, :EVAL_CTX + 1]

    # Baseline PPL
    print("\nComputing baseline PPL...")
    baseline_loss = chunked_cross_entropy(model, eval_ids)
    baseline_ppl  = float(np.exp(baseline_loss))
    print(f"  Baseline PPL = {baseline_ppl:.4f}")

    layer_csv = RESULTS_DIR / "exp16_layer_sensitivity.csv"
    done_layers = set()
    if layer_csv.exists():
        with open(layer_csv) as f:
            for row in csv.DictReader(f):
                done_layers.add(int(row["layer_idx"]))
        print(f"\nResuming: {len(done_layers)}/{n_layers} layers done")

    print(f"\n── Ablating layers one at a time (k={ABLATION_K}_4bit) ──")
    for layer_idx in range(n_layers):
        if layer_idx in done_layers:
            continue
        hooks = install_layer_hooks(model, layer_idx, bases, n_kv_heads, d_head,
                                     ABLATION_K, ABLATION_BITS)
        loss = chunked_cross_entropy(model, eval_ids)
        ppl  = float(np.exp(loss))
        ppl_delta = ppl - baseline_ppl
        for h in hooks:
            h.remove()
        torch.cuda.empty_cache()

        row = {"layer_idx": layer_idx, "ppl": round(ppl, 4),
               "ppl_delta": round(ppl_delta, 4), "baseline_ppl": round(baseline_ppl, 4)}
        file_exists = layer_csv.exists()
        with open(layer_csv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=["layer_idx", "ppl", "ppl_delta", "baseline_ppl"])
            if not file_exists:
                w.writeheader()
            w.writerow(row)

        print(f"  layer {layer_idx:>2}: PPL={ppl:.4f}  delta={ppl_delta:+.4f}", flush=True)

    # Load all results
    all_rows = []
    if layer_csv.exists():
        with open(layer_csv) as f:
            all_rows = sorted(csv.DictReader(f), key=lambda r: int(r["layer_idx"]))

    # Adaptive policy
    deltas = [(int(r["layer_idx"]), float(r["ppl_delta"])) for r in all_rows]
    deltas.sort(key=lambda x: x[1])  # ascending sensitivity

    n = len(deltas)
    policy = {}
    for i, (li, _) in enumerate(deltas):
        frac = i / n
        if frac < 0.25:
            policy[li] = 64    # least sensitive: most aggressive
        elif frac < 0.75:
            policy[li] = 96
        else:
            policy[li] = 128   # most sensitive: least aggressive

    mean_k = sum(policy.values()) / len(policy) if policy else 0
    policy_path = RESULTS_DIR / "exp16_adaptive_policy.json"
    with open(policy_path, 'w') as f:
        json.dump({"per_layer_k": {str(li): k for li, k in policy.items()},
                   "mean_k": round(mean_k, 1),
                   "budget_k": BUDGET_K,
                   "n_layers": n_layers}, f, indent=2)
    print(f"\nAdaptive policy: mean_k={mean_k:.1f} (target={BUDGET_K})")
    print(f"Saved {policy_path}")

    # Report
    delta_dict = {int(r["layer_idx"]): float(r["ppl_delta"]) for r in all_rows}
    report_path = RESULTS_DIR / "REPORT-16-sensitivity.md"
    with open(report_path, 'w') as f:
        f.write("# Experiment 16: Layer Sensitivity Profiling\n\n")
        f.write(f"- Model: Qwen3-14B-AWQ ({n_layers} layers, {n_kv_heads} KV heads, d_head={d_head})\n")
        f.write(f"- Ablation: k={ABLATION_K}_4bit on each layer independently\n")
        f.write(f"- Baseline PPL: {baseline_ppl:.4f}\n\n")

        f.write("## Layer Sensitivity (PPL delta)\n\n")
        f.write("| Layer | PPL | Delta | Tier |\n")
        f.write("|-------|-----|-------|------|\n")
        for r in all_rows:
            delta = float(r["ppl_delta"])
            tier = "HIGH" if delta > 2.0 else ("MED" if delta > 0.5 else "LOW")
            f.write(f"| {int(r['layer_idx']):>5} | {float(r['ppl']):.3f} | {delta:+.3f} | {tier} |\n")

        f.write(f"\n## Adaptive Compression Policy (mean_k={mean_k:.1f})\n\n")
        f.write("| Layer | Assigned k | PPL delta |\n")
        f.write("|-------|-----------|----------|\n")
        for li in sorted(policy.keys()):
            f.write(f"| {li:>5} | {policy[li]:>9} | {delta_dict.get(li, 0):+.3f} |\n")

    print(f"\nSaved {layer_csv}")
    print(f"Wrote {report_path}")
    print("\n" + "=" * 70)
    print("Experiment 16 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
