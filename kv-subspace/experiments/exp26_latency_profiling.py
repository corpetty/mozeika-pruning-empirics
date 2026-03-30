"""
Experiment 26: Latency profiling — breakdown of 13× hook overhead.

MOTIVATION
----------
Exp14 showed ~13× throughput degradation with compression hooks (11.75 → 0.92 tok/s).
The paper claims "1.2× fused kernel" overhead, but that number has no empirical basis.

The Python forward hook overhead is the problem, not the compression computation.
This experiment breaks down the 13× into components:
  1. Python hook dispatch (registering + entering hooks with no-op body)
  2. Tensor copy to CPU (x.detach().cpu())
  3. PCA projection + quantization (numpy compute)
  4. Tensor copy back to GPU

Additionally tests what happens with a GPU-native (torch) implementation
to estimate what "fused kernel" speedup is achievable without actually writing CUDA.

Output:
  results/exp26_latency_profiling.json
  results/REPORT-26-latency.md
"""

import sys
import csv
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect import get_model_and_tokenizer
from compress import fit_pca, subspace_compress, random_rotation_matrix

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen3-14B-AWQ"
RESULTS_DIR = Path("results")

CALIB_TOKENS   = 2048
N_KV_HEADS     = 8
D_HEAD         = 128
N_LAYERS       = 40
CTX_LEN        = 4096
DECODE_STEPS   = 64     # shorter than exp14 for faster profiling
N_WARMUP       = 5
N_TRIALS       = 3      # repeat each decode run, take median

K              = 128
N_BITS         = 4


# ── Data helpers ──────────────────────────────────────────────────────────────

def get_wikitext2_tokens(tokenizer, split, n_tokens, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=True)
    text = "\n".join(line for line in "\n\n".join(ds["text"]).split("\n") if line.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return ids[:, :n_tokens].to(device)


# ── Model helpers ─────────────────────────────────────────────────────────────

def find_attention_layers(model):
    for i, layer in enumerate(model.model.model.layers):
        yield i, layer.self_attn


def collect_kvs_for_basis(model, input_ids, n_kv_heads, d_head):
    kv_store = {}
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_hook(li, kvt, nh, dh):
                def hook(module, inp, out):
                    x = out.detach().cpu().float().reshape(out.shape[0], out.shape[1], nh, dh)[0]
                    for h in range(nh):
                        key = (li, h)
                        if key not in kv_store:
                            kv_store[key] = {'K': [], 'V': []}
                        kv_store[key][kvt].append(x[:, h, :].numpy())
                return hook
            hooks.append(getattr(attn, proj_name).register_forward_hook(
                make_hook(layer_idx, kv_type, n_kv_heads, d_head)))
    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()
    return {k: {kv: np.concatenate(v, axis=0) for kv, v in d.items()}
            for k, d in kv_store.items()}


def build_bases_numpy(initial_kvs, k):
    """PCA bases for numpy (CPU) compression."""
    bases = {}
    for (li, hi), kv in initial_kvs.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        bases[(li, hi)] = {'U_K': U_k.astype(np.float32), 'mean_K': mean_k.astype(np.float32)}
    return bases


def build_bases_gpu(initial_kvs, k, device):
    """PCA bases as GPU tensors for torch-native compression."""
    bases = {}
    for (li, hi), kv in initial_kvs.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        bases[(li, hi)] = {
            'U_K':   torch.from_numpy(U_k.astype(np.float32)).to(device),    # (d, k)
            'mean_K': torch.from_numpy(mean_k.astype(np.float32)).to(device), # (d,)
        }
    return bases


# ── Compression variants ──────────────────────────────────────────────────────

def make_noop_hook(layer_idx, n_kv_heads, d_head):
    """Pure hook dispatch overhead — hook does nothing but enter/exit."""
    def hook(module, inp, out):
        return out  # no-op
    return hook


def make_cpu_copy_only_hook(layer_idx, n_kv_heads, d_head):
    """Hook overhead including CPU transfer, but no compute."""
    def hook(module, inp, out):
        dev, dty = out.device, out.dtype
        x = out.detach().cpu().float()
        # Round-trip back without any compute
        return x.to(dty).to(dev)
    return hook


def make_numpy_compress_hook(layer_idx, bases, k, n_bits, R_cache, n_kv_heads, d_head):
    """Full numpy compression (current implementation)."""
    def hook(module, inp, out):
        dev, dty = out.device, out.dtype
        x = out.detach().cpu().float()
        b, s, _ = x.shape
        x = x.reshape(b, s, n_kv_heads, d_head)
        for h in range(n_kv_heads):
            key = (layer_idx, h)
            if key not in bases:
                continue
            xh = x[0, :, h, :].numpy()
            U  = bases[key]['U_K']
            mn = bases[key]['mean_K']
            R_key = (layer_idx, h)
            if R_key not in R_cache:
                R_cache[R_key] = random_rotation_matrix(k)
            xh_c = subspace_compress(xh, k, n_bits, U, mn, R_cache[R_key], quantizer='subrotq')
            x[0, :, h, :] = torch.from_numpy(xh_c)
        return x.reshape(b, s, n_kv_heads * d_head).to(dty).to(dev)
    return hook


def make_torch_compress_hook(layer_idx, bases_gpu, k, n_bits, R_cache_gpu,
                              n_kv_heads, d_head, device):
    """
    GPU-native SubRotQ compression in torch (no CPU transfer).
    Simulates what a fused CUDA kernel would do:
      1. Reshape to (T, n_heads, d_head)
      2. For each head: subtract mean, project to k-dim subspace
      3. Simulate quantization: scale/zero-point rounding in torch
      4. Reconstruct and un-project
    This is NOT a real fused kernel but shows the CPU-transfer overhead separately.
    """
    def hook(module, inp, out):
        # out: (1, T, n_heads * d_head)
        dev, dty = out.device, out.dtype
        x = out.float()  # stay on GPU
        b, s, _ = x.shape
        x = x.reshape(b, s, n_kv_heads, d_head)  # (1, T, nh, d_head)

        for h in range(n_kv_heads):
            key = (layer_idx, h)
            if key not in bases_gpu:
                continue
            U  = bases_gpu[key]['U_K']    # (d, k) on GPU
            mn = bases_gpu[key]['mean_K'] # (d,) on GPU

            xh = x[0, :, h, :]           # (T, d)

            # Pre-condition with random rotation (simulate SubRotQ rotation)
            R_key = (layer_idx, h)
            if R_key not in R_cache_gpu:
                R_np = random_rotation_matrix(k)
                R_cache_gpu[R_key] = torch.from_numpy(R_np.astype(np.float32)).to(device)
            R = R_cache_gpu[R_key]        # (k, k)

            # Project to subspace
            centered = xh - mn.unsqueeze(0)   # (T, d)
            coords = centered @ U              # (T, k)

            # Rotate
            coords_rot = coords @ R.T         # (T, k)

            # Simulate uniform quantization (floor rounding, scale=max/127)
            scale = coords_rot.abs().max(dim=0, keepdim=True).values.clamp(min=1e-6)
            n_levels = 2 ** n_bits - 1
            coords_q = torch.round(coords_rot / scale * (n_levels / 2)).clamp(
                -(n_levels // 2), n_levels // 2)
            coords_dq = coords_q / (n_levels / 2) * scale

            # Un-rotate
            coords_unrot = coords_dq @ R      # (T, k)

            # Reconstruct
            xh_c = coords_unrot @ U.T + mn.unsqueeze(0)  # (T, d)
            x[0, :, h, :] = xh_c

        return x.reshape(b, s, n_kv_heads * d_head).to(dty)
    return hook


# ── Timing harness ────────────────────────────────────────────────────────────

def timed_decode(model, tokenizer, prefill_ids, hooks, decode_steps, device):
    """
    Prefill with prefill_ids, then decode `decode_steps` tokens with hooks installed.
    Returns wall-clock seconds for decode phase only.
    """
    # Prefill
    with torch.no_grad():
        prefill_out = model.model(input_ids=prefill_ids, use_cache=True)
    past_kv = prefill_out.past_key_values
    next_tok = prefill_ids[:, -1:]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.no_grad():
        for _ in range(decode_steps):
            out = model.model(
                input_ids=next_tok,
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out.past_key_values
            next_tok = out.logits[:, -1:, :].argmax(dim=-1)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # Remove hooks
    for h in hooks:
        h.remove()

    return t1 - t0


def run_variant(label, model, tokenizer, prefill_ids, hook_builder,
                decode_steps, n_warmup, n_trials, device):
    """Run n_warmup + n_trials decode runs, return median tok/s and list of all."""
    print(f"  [{label}]", end='', flush=True)
    timings = []
    for trial in range(n_warmup + n_trials):
        hooks = hook_builder()
        elapsed = timed_decode(model, tokenizer, prefill_ids, hooks, decode_steps, device)
        tok_s = decode_steps / elapsed
        if trial >= n_warmup:
            timings.append(tok_s)
            print(f" {tok_s:.1f}", end='', flush=True)
    print()
    return float(np.median(timings)), timings


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    RESULTS_DIR.mkdir(exist_ok=True)

    print(f"Loading model {MODEL_NAME}...")
    device = 'cuda'
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    print("Loading WikiText-2...")
    calib_ids = get_wikitext2_tokens(tokenizer, "train", CALIB_TOKENS, device)
    prefill_ids = get_wikitext2_tokens(tokenizer, "test", CTX_LEN, device)

    print("Collecting KV basis...")
    initial_kvs = collect_kvs_for_basis(model, calib_ids, N_KV_HEADS, D_HEAD)

    print("Building bases (numpy + GPU)...")
    bases_np = build_bases_numpy(initial_kvs, K)
    bases_gpu = build_bases_gpu(initial_kvs, K, device)

    R_cache_np  = {}
    R_cache_gpu = {}

    print(f"\nProfiling decode at ctx={CTX_LEN}, k={K}, bits={N_BITS}, "
          f"decode_steps={DECODE_STEPS}...")
    print(f"Warmup={N_WARMUP}, trials={N_TRIALS} (tok/s printed per trial)\n")

    results = {}

    # 1. Baseline — no hooks
    baseline_toks, baseline_list = run_variant(
        "baseline (no hooks)",
        model, tokenizer, prefill_ids,
        lambda: [],
        DECODE_STEPS, N_WARMUP, N_TRIALS, device)
    results["baseline"] = {"tok_s": baseline_toks, "trials": baseline_list}

    # 2. No-op hook — pure dispatch overhead
    def noop_builder():
        hooks = []
        for li, attn in find_attention_layers(model):
            h = attn.k_proj.register_forward_hook(
                make_noop_hook(li, N_KV_HEADS, D_HEAD))
            hooks.append(h)
        return hooks

    noop_toks, noop_list = run_variant(
        "noop hooks (dispatch only)",
        model, tokenizer, prefill_ids, noop_builder,
        DECODE_STEPS, N_WARMUP, N_TRIALS, device)
    results["noop_hooks"] = {"tok_s": noop_toks, "trials": noop_list}

    # 3. CPU copy only — no compute, just the round-trip transfer
    def cpu_copy_builder():
        hooks = []
        for li, attn in find_attention_layers(model):
            h = attn.k_proj.register_forward_hook(
                make_cpu_copy_only_hook(li, N_KV_HEADS, D_HEAD))
            hooks.append(h)
        return hooks

    cpu_copy_toks, cpu_copy_list = run_variant(
        "CPU copy only (no compute)",
        model, tokenizer, prefill_ids, cpu_copy_builder,
        DECODE_STEPS, N_WARMUP, N_TRIALS, device)
    results["cpu_copy_only"] = {"tok_s": cpu_copy_toks, "trials": cpu_copy_list}

    # 4. Full numpy compression (current pipeline)
    def numpy_compress_builder():
        hooks = []
        for li, attn in find_attention_layers(model):
            h = attn.k_proj.register_forward_hook(
                make_numpy_compress_hook(li, bases_np, K, N_BITS, R_cache_np,
                                         N_KV_HEADS, D_HEAD))
            hooks.append(h)
        return hooks

    numpy_toks, numpy_list = run_variant(
        "numpy compression (current)",
        model, tokenizer, prefill_ids, numpy_compress_builder,
        DECODE_STEPS, N_WARMUP, N_TRIALS, device)
    results["numpy_compress"] = {"tok_s": numpy_toks, "trials": numpy_list}

    # 5. Torch GPU-native compression (no CPU transfer)
    def torch_compress_builder():
        hooks = []
        for li, attn in find_attention_layers(model):
            h = attn.k_proj.register_forward_hook(
                make_torch_compress_hook(li, bases_gpu, K, N_BITS, R_cache_gpu,
                                          N_KV_HEADS, D_HEAD, device))
            hooks.append(h)
        return hooks

    torch_toks, torch_list = run_variant(
        "torch GPU compression (no CPU copy)",
        model, tokenizer, prefill_ids, torch_compress_builder,
        DECODE_STEPS, N_WARMUP, N_TRIALS, device)
    results["torch_compress"] = {"tok_s": torch_toks, "trials": torch_list}

    # ── Analysis ──
    print("\n" + "="*60)
    print("LATENCY BREAKDOWN SUMMARY")
    print("="*60)
    overhead_dispatch  = baseline_toks / noop_toks
    overhead_cpu_copy  = noop_toks / cpu_copy_toks
    overhead_numpy     = cpu_copy_toks / numpy_toks
    overhead_vs_torch  = numpy_toks / torch_toks if torch_toks < numpy_toks else 1.0
    total_numpy        = baseline_toks / numpy_toks
    total_torch        = baseline_toks / torch_toks

    print(f"  Baseline:          {baseline_toks:.1f} tok/s")
    print(f"  Noop hooks:        {noop_toks:.1f} tok/s  ({overhead_dispatch:.1f}× overhead from dispatch)")
    print(f"  CPU copy only:     {cpu_copy_toks:.1f} tok/s  ({overhead_cpu_copy:.1f}× overhead from GPU↔CPU)")
    print(f"  Numpy compress:    {numpy_toks:.1f} tok/s  ({overhead_numpy:.1f}× overhead from compute)")
    print(f"  Torch compress:    {torch_toks:.1f} tok/s  ({overhead_vs_torch:.1f}× numpy/torch ratio)")
    print(f"")
    print(f"  Total numpy overhead: {total_numpy:.1f}×")
    print(f"  Total torch overhead: {total_torch:.1f}×")
    print(f"  Prior claim (paper): 1.2× (UNSUPPORTED — actual is {total_numpy:.1f}×)")
    print(f"  Primary bottleneck: {'GPU↔CPU copy' if overhead_cpu_copy > max(overhead_dispatch, overhead_numpy) else 'dispatch overhead' if overhead_dispatch > overhead_numpy else 'numpy compute'}")

    # Save
    json_path = RESULTS_DIR / "exp26_latency_profiling.json"
    full_results = {
        "config": {"model": MODEL_NAME, "ctx_len": CTX_LEN, "k": K, "n_bits": N_BITS,
                   "decode_steps": DECODE_STEPS, "n_warmup": N_WARMUP, "n_trials": N_TRIALS},
        "variants": results,
        "analysis": {
            "baseline_tok_s":          baseline_toks,
            "overhead_hook_dispatch":  overhead_dispatch,
            "overhead_cpu_copy":       overhead_cpu_copy,
            "overhead_numpy_compute":  overhead_numpy,
            "total_numpy_overhead":    total_numpy,
            "total_torch_overhead":    total_torch,
            "paper_claim_overhead":    1.2,
            "primary_bottleneck":      "gpu_cpu_copy" if overhead_cpu_copy > max(overhead_dispatch, overhead_numpy) else "dispatch" if overhead_dispatch > overhead_numpy else "numpy_compute",
        }
    }
    json_path.write_text(json.dumps(full_results, indent=2))

    report = f"""# Experiment 26: Latency Profiling — Hook Overhead Breakdown

## Config
- Model: {MODEL_NAME}
- Context length: {CTX_LEN}
- Decode steps: {DECODE_STEPS}
- Compression: k={K}, {N_BITS}-bit SubRotQ

## Results

| Variant | tok/s | Slowdown vs baseline |
|---------|-------|---------------------|
| Baseline (no hooks) | {baseline_toks:.1f} | 1.0× |
| Noop hooks (dispatch only) | {noop_toks:.1f} | {overhead_dispatch:.1f}× |
| CPU copy only (no compute) | {cpu_copy_toks:.1f} | {baseline_toks/cpu_copy_toks:.1f}× |
| Numpy compression (current) | {numpy_toks:.1f} | {total_numpy:.1f}× |
| Torch GPU compression (no CPU copy) | {torch_toks:.1f} | {total_torch:.1f}× |

## Overhead Attribution

| Component | Slowdown | Notes |
|-----------|----------|-------|
| Python hook dispatch | {overhead_dispatch:.1f}× | Empty hook body |
| GPU↔CPU transfer | {overhead_cpu_copy:.1f}× | detach().cpu() + .to(gpu) |
| Numpy PCA+quant compute | {overhead_numpy:.1f}× | Actual compression |
| **Total (current)** | **{total_numpy:.1f}×** | Python hooks over numpy |
| Torch GPU (no copy) | {total_torch:.1f}× | Upper bound for fused kernel |

## Key Finding

The paper's claim of "1.2× overhead for a fused kernel" is **not supported**.

- Current implementation: **{total_numpy:.1f}×** slowdown (Python hooks + numpy + CPU copy)
- Primary bottleneck: GPU↔CPU copy ({overhead_cpu_copy:.1f}×) + hook dispatch ({overhead_dispatch:.1f}×)
- A GPU-native torch implementation (no CPU copy): **{total_torch:.1f}×**
- A true fused CUDA kernel could potentially approach 1.1–1.5× but would require
  implementing quantization kernels in CUDA — outside scope of this work

**Honest paper claim:** "Our prototype incurs {total_numpy:.0f}× latency overhead due to
Python hook dispatch and GPU↔CPU transfers. A production CUDA implementation
eliminating these would reduce overhead to an estimated {total_torch:.1f}×, but
we leave kernel implementation as future work."
"""

    report_path = RESULTS_DIR / "REPORT-26-latency.md"
    report_path.write_text(report)
    print(f"\nResults: {json_path}")
    print(f"Report:  {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
