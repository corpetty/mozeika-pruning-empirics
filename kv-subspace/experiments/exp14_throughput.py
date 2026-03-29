"""
Experiment 14: Memory and throughput benchmark.

Answers: What is the actual VRAM and decode throughput impact of KV compression?
  - KV cache size (GB) at various context lengths
  - Decode tokens/sec at varying context lengths
  - Peak VRAM under prefill + decode

Configs: baseline, k128_4bit, k96_4bit
Context lengths: 4096, 8192, 16384, 32768

Usage:
    python experiments/exp14_throughput.py

Outputs:
    results/exp14_throughput.csv
    results/REPORT-14-throughput.md
"""

import sys
import os
import csv
import time
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
CALIB_OFFSET = 5000   # character offset to skip Gutenberg header
DECODE_STEPS = 128

CTX_LENGTHS  = [4096, 8192, 16384, 32768]

CONFIGS = {
    "baseline":  (None,       None, None, None,       None, None),
    "k128_4bit": ("subspace", 128,  4,    "subspace", 128,  4),
    "k96_4bit":  ("subspace", 96,   4,    "subspace", 96,   4),
}


def load_tokens(tokenizer, data_file, char_offset, n_tokens, device):
    with open(data_file, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    text = text[char_offset:]
    inputs = tokenizer(text, return_tensors='pt', truncation=True,
                       max_length=n_tokens + 1, add_special_tokens=True)
    return inputs['input_ids'].to(device)


def collect_kvs_for_basis(model, tokenizer, data_file, char_offset, n_tokens,
                           device, n_kv_heads, d_head):
    """Collect K/V vectors for PCA basis fitting.
    Returns: {(layer_idx, head_idx): {'K': (T,d), 'V': (T,d)}}
    """
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
                    x = x.reshape(b, s, nh, dh)[0]  # (s, nh, dh)
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
        arr = np.concatenate(arrays, axis=0)  # (T, n_kv_heads, d_head)
        for head_idx in range(arr.shape[1]):
            key = (layer_idx, head_idx)
            if key not in bases_raw:
                bases_raw[key] = {}
            bases_raw[key][kv_type] = arr[:, head_idx, :]  # (T, d_head)
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


def install_hooks(model, cfg, bases, n_kv_heads, d_head):
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
                    device_, dtype_ = output.device, output.dtype
                    x = output.detach().cpu().float()
                    b, s, _ = x.shape
                    x = x.reshape(b, s, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = x[0, :, h, :].numpy()
                        base = bases.get((li, h), {})
                        U  = base.get(f'U_{kvt}')
                        mn = base.get(f'mean_{kvt}')
                        x[0, :, h, :] = torch.from_numpy(
                            compress_vec(xh, m, kk, nb, U, mn))
                    return x.reshape(b, s, -1).to(device=device_, dtype=dtype_)
                return hook
            proj = getattr(attn, proj_name)
            hooks.append(proj.register_forward_hook(make_hook(layer_idx, kv_type, method, k, bits)))
    return hooks


def analytical_kv_gb(n_layers, n_kv_heads, d_dim, ctx_len, bits=16):
    return 2 * n_layers * n_kv_heads * d_dim * ctx_len * (bits / 8) / 1e9


def run_benchmark(model, tokenizer, bases_by_k, n_kv_heads, d_head,
                  device, ctx_len, cfg_name, cfg):
    K_method, K_k, K_bits, V_method, V_k, V_bits = cfg
    k_for_basis = K_k if K_k is not None else 128
    bases = bases_by_k.get(k_for_basis, bases_by_k.get(128, {}))

    # Load eval tokens from a non-overlapping window
    eval_offset = CALIB_OFFSET + CALIB_TOKENS * 6  # skip well past calibration chars
    input_ids = load_tokens(tokenizer, DATA_FILE, eval_offset, ctx_len, device)
    if input_ids.shape[1] < ctx_len:
        # tile if needed
        rep = (ctx_len // input_ids.shape[1]) + 2
        input_ids = input_ids.repeat(1, rep)[:, :ctx_len]
    else:
        input_ids = input_ids[:, :ctx_len]

    hooks = install_hooks(model, cfg, bases, n_kv_heads, d_head)

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    # Prefill
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    past_kv = out.past_key_values
    torch.cuda.synchronize(device)
    t_prefill = time.perf_counter() - t0
    peak_prefill_gb = torch.cuda.max_memory_allocated(device) / 1e9

    # Decode
    next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    for _ in range(DECODE_STEPS):
        with torch.no_grad():
            out = model(input_ids=next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    torch.cuda.synchronize(device)
    t_decode = time.perf_counter() - t0
    peak_decode_gb = torch.cuda.max_memory_allocated(device) / 1e9

    for h in hooks:
        h.remove()
    del past_kv, out, next_tok, input_ids
    torch.cuda.empty_cache()

    return {
        "ctx_len": ctx_len,
        "config": cfg_name,
        "tok_per_sec_decode": round(DECODE_STEPS / t_decode, 2),
        "tok_per_sec_prefill": round(ctx_len / t_prefill, 2),
        "peak_vram_prefill_gb": round(peak_prefill_gb, 3),
        "peak_vram_decode_gb": round(peak_decode_gb, 3),
        "t_prefill_s": round(t_prefill, 3),
        "t_decode_s": round(t_decode, 3),
    }


def main():
    print("=" * 70)
    print("Experiment 14: Throughput and Memory Benchmark")
    print("=" * 70)

    device = "cuda"
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    attn_layers = find_attention_layers(model)
    n_layers   = len(attn_layers)
    n_kv_heads = model.config.num_key_value_heads
    d_head     = model.config.hidden_size // model.config.num_attention_heads
    print(f"n_layers={n_layers}, n_kv_heads={n_kv_heads}, d_head={d_head}")

    # Calibration
    print(f"\nCalibrating ({CALIB_TOKENS} tokens)...")
    calib_kvs = collect_kvs_for_basis(model, tokenizer, DATA_FILE, CALIB_OFFSET,
                                       CALIB_TOKENS, device, n_kv_heads, d_head)
    bases_by_k = {}
    for k in [96, 128]:
        bases_by_k[k] = fit_bases(calib_kvs, k)
    print(f"  Fitted bases: {len(bases_by_k[128])} (layer, head) pairs")

    # Analytical table
    kv_bits = {"baseline": 16, "k128_4bit": 4, "k96_4bit": 4}
    kv_dims  = {"baseline": d_head, "k128_4bit": 128, "k96_4bit": 96}
    print("\n── Analytical KV cache (GB) ──")
    print(f"{'config':<12} | " + " | ".join(f"ctx={c}" for c in CTX_LENGTHS))
    for cfg_name in CONFIGS:
        sizes = [f"{analytical_kv_gb(n_layers, n_kv_heads, kv_dims[cfg_name], c, kv_bits[cfg_name]):.3f}"
                 for c in CTX_LENGTHS]
        print(f"{cfg_name:<12} | " + " | ".join(sizes))

    # Resume
    csv_path = RESULTS_DIR / "exp14_throughput.csv"
    fieldnames = ["ctx_len", "config", "tok_per_sec_decode", "tok_per_sec_prefill",
                  "peak_vram_prefill_gb", "peak_vram_decode_gb", "t_prefill_s", "t_decode_s"]
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["ctx_len"]), row["config"]))
        print(f"\nResuming: {len(done)} pairs done")

    for ctx_len in CTX_LENGTHS:
        for cfg_name, cfg in CONFIGS.items():
            if (ctx_len, cfg_name) in done:
                print(f"  [ctx={ctx_len}, {cfg_name}] skipping (done)")
                continue
            print(f"\n  ctx={ctx_len}  cfg={cfg_name}...", flush=True)
            try:
                result = run_benchmark(model, tokenizer, bases_by_k, n_kv_heads, d_head,
                                       device, ctx_len, cfg_name, cfg)
                print(f"    decode: {result['tok_per_sec_decode']:.1f} tok/s | "
                      f"prefill: {result['tok_per_sec_prefill']:.1f} tok/s | "
                      f"peak VRAM: {result['peak_vram_prefill_gb']:.2f} GB")
                file_exists = csv_path.exists()
                with open(csv_path, 'a', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        w.writeheader()
                    w.writerow(result)
            except torch.cuda.OutOfMemoryError as e:
                print(f"    OOM: {e}")
                torch.cuda.empty_cache()

    # Report
    all_rows = []
    if csv_path.exists():
        with open(csv_path) as f:
            all_rows = list(csv.DictReader(f))

    report_path = RESULTS_DIR / "REPORT-14-throughput.md"
    with open(report_path, 'w') as f:
        f.write("# Experiment 14: Throughput and Memory Benchmark\n\n")
        f.write(f"- Model: Qwen3-14B-AWQ ({n_layers} layers, {n_kv_heads} KV heads, d_head={d_head})\n")
        f.write(f"- Decode steps per trial: {DECODE_STEPS}\n\n")

        f.write("## Decode Throughput (tokens/sec)\n\n")
        f.write("| ctx_len | " + " | ".join(CONFIGS.keys()) + " |\n")
        f.write("|---------|" + "|".join("---" for _ in CONFIGS) + "|\n")
        for ctx_len in CTX_LENGTHS:
            row_vals = {r["config"]: r["tok_per_sec_decode"] for r in all_rows if int(r["ctx_len"]) == ctx_len}
            f.write(f"| {ctx_len} | " + " | ".join(row_vals.get(c, "N/A") for c in CONFIGS) + " |\n")

        f.write("\n## Peak VRAM During Prefill (GB)\n\n")
        f.write("| ctx_len | " + " | ".join(CONFIGS.keys()) + " |\n")
        f.write("|---------|" + "|".join("---" for _ in CONFIGS) + "|\n")
        for ctx_len in CTX_LENGTHS:
            row_vals = {r["config"]: r["peak_vram_prefill_gb"] for r in all_rows if int(r["ctx_len"]) == ctx_len}
            f.write(f"| {ctx_len} | " + " | ".join(row_vals.get(c, "N/A") for c in CONFIGS) + " |\n")

        f.write("\n## Analytical KV Cache Size (GB)\n\n")
        f.write("| Config | " + " | ".join(f"ctx={c}" for c in CTX_LENGTHS) + " |\n")
        f.write("|--------|" + "|".join("---" for _ in CTX_LENGTHS) + "|\n")
        for cfg_name in CONFIGS:
            sizes = [f"{analytical_kv_gb(n_layers, n_kv_heads, kv_dims[cfg_name], c, kv_bits[cfg_name]):.3f}"
                     for c in CTX_LENGTHS]
            f.write(f"| {cfg_name} | " + " | ".join(sizes) + " |\n")

    print(f"\nSaved {csv_path}")
    print(f"Wrote {report_path}")
    print("\n" + "=" * 70)
    print("Experiment 14 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
