"""
Experiment 29: Batched Throughput and Memory Footprint.

MOTIVATION
----------
Exp26 measured single-sequence latency overhead (1.6× prototype, numpy bottleneck).
For the serving story we need:
  1. Batched throughput (tok/s) at batch sizes 1, 4, 8
  2. Peak GPU memory footprint at each batch/context combo
  3. Theoretical KV cache savings at scale

This gives the real serving efficiency numbers: if KV cache is the memory bottleneck
at large batch sizes, subspace compression enables proportionally larger batches.

Outputs:
    results/exp29_batched_throughput.json
    results/REPORT-29-batched-throughput.md
"""
import sys
import json
import time
import numpy as np
import torch
import gc
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from collect import get_model_and_tokenizer, find_attention_layers

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME  = "Qwen/Qwen3-14B-AWQ"
N_KV_HEADS  = 8
D_HEAD      = 128
N_LAYERS    = 40
CALIB_TOKENS = 2048

# ── Data helpers ──────────────────────────────────────────────────────────────

def get_wikitext2_tokens(tokenizer, split, n_tokens):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=True)
    text = "\n\n".join(ds["text"])
    text = "\n".join(line for line in text.split("\n") if line.strip())
    ids = tokenizer.encode(text)
    if len(ids) < n_tokens:
        ids = ids * ((n_tokens // len(ids)) + 1)
    return ids[:n_tokens]

# ── Proven collect_kvs_for_basis from exp24 ───────────────────────────────────

def collect_kvs(model, input_ids, n_kv_heads=8, d_head=128):
    kv_store = {}
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj')]:  # K-only
            proj = getattr(attn, proj_name)
            def make_hook(li, kvt, nh, dh):
                def hook(module, inp, out):
                    x = out.detach().cpu().float()
                    x = x.reshape(x.shape[0], x.shape[1], nh, dh)[0]
                    for h_idx in range(nh):
                        key = (li, h_idx)
                        if key not in kv_store:
                            kv_store[key] = {'K': []}
                        kv_store[key][kvt].append(x[:, h_idx, :].numpy())
                return hook
            hooks.append(proj.register_forward_hook(make_hook(layer_idx, kv_type, n_kv_heads, d_head)))
    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()
    return {k: {kv: np.concatenate(v, axis=0) for kv, v in d.items()}
            for k, d in kv_store.items()}

def fit_bases(k_arrays, k_max=128):
    bases = {}
    for key, X in k_arrays.items():
        if len(X) < k_max:
            continue
        X_c = X - X.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
        bases[key] = (Vt[:, :k_max], X.mean(0))
    return bases

# ── Throughput measurement ────────────────────────────────────────────────────

def measure_baseline(model, ctx_len, batch_size, n_warmup, n_trials, device):
    tokens = get_wikitext2_tokens(next(model.parameters()), ctx_len * (batch_size + n_warmup + n_trials))
    raise NotImplementedError  # placeholder

def measure_baseline_throughput(model, tokenizer, ctx_len, batch_size, n_warmup, n_trials, device):
    tokens = get_wikitext2_tokens(tokenizer, "test", ctx_len * (batch_size + n_warmup + n_trials))
    results = []
    for i in range(n_warmup):
        ids = torch.tensor([tokens[i*ctx_len:(i+1)*ctx_len]] * batch_size, dtype=torch.long, device=device)
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
    for trial in range(n_trials):
        offset = (n_warmup + trial) * ctx_len
        ids = torch.tensor([tokens[offset:offset+ctx_len]] * batch_size, dtype=torch.long, device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        total_tokens = ctx_len * batch_size
        results.append(total_tokens / elapsed)
    return float(np.median(results))

def measure_peak_memory(model, tokenizer, ctx_len, batch_size, device):
    tokens = get_wikitext2_tokens(tokenizer, "test", ctx_len)
    ids = torch.tensor([tokens[:ctx_len]] * batch_size, dtype=torch.long, device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(input_ids=ids, use_cache=False)
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6, torch.cuda.max_memory_reserved() / 1e6

def measure_compressed_throughput_single(model, tokenizer, bases, k_assign, ctx_len, device, n_warmup=2, n_trials=3):
    tokens = get_wikitext2_tokens(tokenizer, "test", ctx_len * (n_warmup + n_trials + 1))
    def install_hooks():
        hooks = []
        for layer_idx, attn_mod in find_attention_layers(model):
            try:
                k_mod = attn_mod.k_proj
            except AttributeError:
                continue
            k_val = k_assign[layer_idx]
            if k_val >= D_HEAD:
                continue
            def make_hook(li, k):
                def hook_fn(module, inp, out):
                    T = out.shape[1]
                    flat = out[0].detach().float().cpu().numpy().reshape(T, N_KV_HEADS, D_HEAD)
                    compressed = np.zeros_like(flat)
                    for hi in range(N_KV_HEADS):
                        key = (li, hi)
                        if key not in bases:
                            compressed[:, hi, :] = flat[:, hi, :]
                            continue
                        U, mean = bases[key]
                        centered = flat[:, hi, :] - mean
                        coords = centered @ U[:, :k]
                        compressed[:, hi, :] = coords @ U[:, :k].T + mean
                    return torch.tensor(
                        compressed.reshape(T, N_KV_HEADS * D_HEAD),
                        dtype=out.dtype, device=out.device
                    ).unsqueeze(0)
                return hook_fn
            hooks.append(k_mod.register_forward_hook(make_hook(layer_idx, k_val)))
        return hooks

    results = []
    for i in range(n_warmup):
        ids = torch.tensor([tokens[i*ctx_len:(i+1)*ctx_len]], dtype=torch.long, device=device)
        h = install_hooks()
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False)
        for hh in h: hh.remove()
        torch.cuda.synchronize()
    for trial in range(n_trials):
        offset = (n_warmup + trial) * ctx_len
        ids = torch.tensor([tokens[offset:offset+ctx_len]], dtype=torch.long, device=device)
        h = install_hooks()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(input_ids=ids, use_cache=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        for hh in h: hh.remove()
        results.append(ctx_len / elapsed)
    return float(np.median(results))

def kv_cache_mb(ctx_len, batch_size, n_layers, n_kv_heads, d_head, bits=16):
    return 2 * n_layers * n_kv_heads * ctx_len * d_head * batch_size * (bits / 8) / 1e6

def kv_cache_compressed_mb(ctx_len, batch_size, n_layers, n_kv_heads, k, d_head=128, bits_k=4, bits_v=16):
    k_b = n_layers * n_kv_heads * ctx_len * k * (bits_k / 8) * batch_size
    v_b = n_layers * n_kv_heads * ctx_len * d_head * (bits_v / 8) * batch_size
    return (k_b + v_b) / 1e6

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print("Loading model...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    device = next(model.parameters()).device

    print("Fitting bases...")
    calib = get_wikitext2_tokens(tokenizer, "train", CALIB_TOKENS)
    calib_t = torch.tensor([calib], dtype=torch.long, device=device)
    kvs = collect_kvs(model, calib_t)
    bases = {k: v['K'] for k, v in kvs.items()}
    bases = fit_bases(bases)
    print(f"  {len(bases)} bases fitted")

    k_assigns = {
        "k128_4bit": {i: 128 for i in range(N_LAYERS)},
        "k112_4bit": {i: 112 for i in range(N_LAYERS)},
    }

    results = {"theoretical": {}, "baseline": {}, "peak_mem": {}, "compressed_single": {}}

    CTX_LENS = [512, 2048, 8192]
    BATCH_SIZES = [1, 4, 8]

    # Theoretical
    print("\n--- Theoretical KV cache ---")
    for ctx in CTX_LENS:
        for bs in BATCH_SIZES:
            k = f"ctx{ctx}_bs{bs}"
            bl = kv_cache_mb(ctx, bs, N_LAYERS, N_KV_HEADS, D_HEAD)
            c128 = kv_cache_compressed_mb(ctx, bs, N_LAYERS, N_KV_HEADS, 128)
            c112 = kv_cache_compressed_mb(ctx, bs, N_LAYERS, N_KV_HEADS, 112)
            results["theoretical"][k] = dict(ctx=ctx, bs=bs, baseline=bl, k128=c128, k112=c112,
                                              r128=bl/c128, r112=bl/c112)
            print(f"  ctx={ctx} bs={bs}: {bl:.0f}MB -> k128={c128:.0f}MB ({bl/c128:.1f}x)")

    # Baseline throughput
    print("\n--- Baseline throughput ---")
    for ctx in CTX_LENS:
        for bs in BATCH_SIZES:
            if ctx >= 8192 and bs > 1:
                continue
            try:
                tps = measure_baseline_throughput(model, tokenizer, ctx, bs, 2, 3, device)
                results["baseline"][f"ctx{ctx}_bs{bs}"] = dict(ctx=ctx, bs=bs, tok_s=tps)
                print(f"  ctx={ctx} bs={bs}: {tps:.0f} tok/s")
            except Exception as e:
                print(f"  ctx={ctx} bs={bs}: FAIL {e}")

    # Peak memory
    print("\n--- Peak memory ---")
    for ctx in CTX_LENS:
        for bs in [1, 4]:
            if ctx >= 8192:
                continue
            try:
                alloc, res = measure_peak_memory(model, tokenizer, ctx, bs, device)
                results["peak_mem"][f"ctx{ctx}_bs{bs}"] = dict(ctx=ctx, bs=bs, alloc=alloc, res=res)
            except Exception:
                pass

    # Compressed single-sequence throughput
    print("\n--- Compressed single-sequence ---")
    for cfg_name, k_a in k_assigns.items():
        for ctx in [512, 2048]:
            try:
                tps = measure_compressed_throughput_single(model, tokenizer, bases, k_a, ctx, device)
                bl_tps = results["baseline"].get(f"ctx{ctx}_bs1", {}).get("tok_s", 1)
                results["compressed_single"][f"{cfg_name}_ctx{ctx}"] = dict(
                    config=cfg_name, ctx=ctx, tok_s=tps, overhead=bl_tps/tps)
                print(f"  {cfg_name} ctx={ctx}: {tps:.0f} tok/s ({bl_tps/tps:.2f}x overhead)")
            except Exception as e:
                print(f"  {cfg_name} ctx={ctx}: FAIL {e}")

    out_path = RESULTS_DIR / "exp29_batched_throughput.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")

    # Report
    lines = ["# Experiment 29: Batched Throughput & Memory", ""]
    lines.append("| ctx,bs | baseline MB | k128 MB | reduction |")
    lines.append("|--------|-----------|---------|-----------|")
    for k, v in results["theoretical"].items():
        lines.append(f"| ctx={v['ctx']},bs={v['bs']} | {v['baseline']:.0f} | {v['k128']:.0f} | {v['r128']:.1f}x |")

    lines += ["\n## Baseline tok/s", "| ctx,bs | tok/s |", "|--------|-------|"]
    for k, v in results["baseline"].items():
        lines.append(f"| ctx={v['ctx']},bs={v['bs']} | {v['tok_s']:.0f} |")

    lines += ["\n## Compressed overhead", "| config,ctx | overhead |", "|------------|----------|"]
    for k, v in results["compressed_single"].items():
        lines.append(f"| {k} | {v['overhead']:.2f}x |")

    report = "\n".join(lines)
    (RESULTS_DIR / "REPORT-29-batched-throughput.md").write_text(report)
    print("\n" + report)

if __name__ == "__main__":
    main()
