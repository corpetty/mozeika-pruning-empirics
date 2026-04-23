"""
Task B1: SubRotQ vs true PolarQuant on clean WikiText-2 (review task B1).

MOTIVATION
----------
Exp22 compared SubRotQ vs PolarQuant on the contaminated W&P pipeline.
This re-runs that comparison on clean WikiText-2, and also adds:
  - Plain 4-bit (from A1: 0.989x) as a lower bound
  - Both quantizers at k=128 (full rank) and k=112 (truncated)

The key question: does SubRotQ's ~5× lower error at 4-bit (exp22) hold on
the clean benchmark, and what is the magnitude relative to plain quantization?

QUANTIZERS TESTED
-----------------
For each k ∈ {112, 128}:
  a) SubRotQ: PCA + random rotation + uniform 4-bit
  b) PolarQuant: PCA + random rotation + recursive polar coordinate encoding (Han et al.)
  c) Plain 4-bit (no PCA, no rotation) — carried over from A1

Output:
  results/exp35_subrotq_vs_polarquant_clean.csv
  results/REPORT-35-subrotq-vs-polarquant-clean.md
"""

import sys
import csv
import os
import time
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect import get_model_and_tokenizer
from compress import (fit_pca, subspace_compress, random_rotation_matrix,
                      quantize_uniform)

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
RESULTS_DIR  = Path("results")
CALIB_TOKENS = 2048
EVAL_TOKENS  = 2048
N_KV_HEADS   = 8
D_HEAD       = 128
N_LAYERS     = 40
K_VALUES     = [112, 128]
N_BITS       = 4


def get_wikitext2_tokens(tokenizer, split, n_tokens, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(ds["text"])
    text = "\n".join(line for line in text.split("\n") if line.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] < n_tokens + 1:
        raise ValueError(f"Only {ids.shape[1]} tokens")
    return ids[:, :n_tokens].to(device)


def find_attention_layers(model):
    for i, layer in enumerate(model.model.model.layers):
        yield i, layer.self_attn


def collect_kvs_for_basis(model, input_ids):
    kv_store = {}
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_hook(li, kvt, nh=N_KV_HEADS, dh=D_HEAD):
                def hook(module, inp, out):
                    x = out.detach().cpu().float()
                    x = x.reshape(x.shape[0], x.shape[1], nh, dh)[0]
                    for h in range(nh):
                        key = (li, h)
                        if key not in kv_store:
                            kv_store[key] = {'K': [], 'V': []}
                        kv_store[key][kvt].append(x[:, h, :].numpy())
                return hook
            hooks.append(getattr(attn, proj_name).register_forward_hook(
                make_hook(layer_idx, kv_type)))
    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()
    return {k: {kv: np.concatenate(v, axis=0) for kv, v in d.items()}
            for k, d in kv_store.items()}


def chunked_cross_entropy(model, input_ids, chunk_size=512):
    transformer_body = model.model.model
    lm_head = model.model.lm_head
    with torch.no_grad():
        outputs = transformer_body(input_ids=input_ids[:, :-1])
        hidden = outputs.last_hidden_state
    labels = input_ids[:, 1:].reshape(-1)
    total_loss = 0.0
    n_tok = 0
    with torch.no_grad():
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            chunk_logits = lm_head(hidden[:, start:end, :])
            chunk_labels = labels[start:end]
            loss = torch.nn.functional.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)), chunk_labels)
            total_loss += float(loss) * (end - start)
            n_tok += (end - start)
            del chunk_logits, chunk_labels, loss
            torch.cuda.empty_cache()
    del hidden
    torch.cuda.empty_cache()
    return total_loss / n_tok


def eval_with_k_hook(model, input_ids, hook_fn):
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        def make_hook(li=layer_idx):
            def hook(module, inp, out):
                dev, dty = out.device, out.dtype
                x = out.detach().cpu().float()
                b, s, _ = x.shape
                x = x.reshape(b, s, N_KV_HEADS, D_HEAD)
                for h in range(N_KV_HEADS):
                    xh = x[0, :, h, :].numpy()
                    xh_q = hook_fn(li, h, xh)
                    x[0, :, h, :] = torch.from_numpy(xh_q)
                return x.reshape(b, s, N_KV_HEADS * D_HEAD).to(dty).to(dev)
            return hook
        hooks.append(attn.k_proj.register_forward_hook(make_hook()))
    loss = chunked_cross_entropy(model, input_ids)
    for h in hooks:
        h.remove()
    return float(np.exp(loss))


def uniform_quant_dequant(x_np, n_bits=4):
    """Per-channel uniform quant."""
    x = x_np.copy().astype(np.float32)
    n_levels = 2 ** n_bits
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    scale = (x_max - x_min) / (n_levels - 1)
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round((x - x_min) / scale), 0, n_levels - 1)
    return (q * scale + x_min).astype(np.float32)


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)

    csv_path    = RESULTS_DIR / "exp35_subrotq_vs_polarquant_clean.csv"
    report_path = RESULTS_DIR / "REPORT-35-subrotq-vs-polarquant-clean.md"
    fieldnames  = ["method", "k", "bits", "ppl", "rel_ppl", "note"]

    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((row["method"], int(row["k"]), int(row["bits"])))
        print(f"Resuming: {len(done)} configs done")

    print(f"=== Task B1: SubRotQ vs PolarQuant (clean WikiText-2) ===")
    print(f"Loading model {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    print("Loading WikiText-2...")
    calib_ids = get_wikitext2_tokens(tokenizer, "train", CALIB_TOKENS, 'cuda')
    eval_ids  = get_wikitext2_tokens(tokenizer, "test",  EVAL_TOKENS,  'cuda')

    # Baseline
    print("\nBaseline PPL...")
    baseline_ppl = float(np.exp(chunked_cross_entropy(model, eval_ids)))
    print(f"  PPL = {baseline_ppl:.4f}")
    expected = (3.0, 10.0)
    assert expected[0] <= baseline_ppl <= expected[1], f"SANITY FAIL: {baseline_ppl}"
    print(f"  SANITY OK")

    rows = [{"method": "baseline", "k": D_HEAD, "bits": 16,
             "ppl": round(baseline_ppl, 4), "rel_ppl": 1.0, "note": ""}]

    # Plain 4-bit (A1 reference — re-run for consistency)
    if ("plain_4bit", D_HEAD, 4) not in done:
        print("\nPlain 4-bit (A1 reference)...")
        def plain_4bit(li, hi, x): return uniform_quant_dequant(x, n_bits=4)
        ppl = eval_with_k_hook(model, eval_ids, plain_4bit)
        rel = ppl / baseline_ppl
        print(f"  PPL={ppl:.4f}  rel={rel:.4f}")
        rows.append({"method": "plain_4bit", "k": D_HEAD, "bits": 4,
                     "ppl": round(ppl, 4), "rel_ppl": round(rel, 4),
                     "note": "per-channel uniform, no PCA/rotation (A1 reference)"})

    # Calibration
    print("\nCollecting KV basis...")
    kvs = collect_kvs_for_basis(model, calib_ids)
    print(f"  {len(kvs)} pairs")

    bases_by_k = {}
    R_by_k = {}
    for k in K_VALUES:
        print(f"  Fitting k={k}...", end="", flush=True)
        bases_by_k[k] = {}
        for (li, hi), kv in kvs.items():
            U_k, mean_k = fit_pca(kv['K'], k)
            bases_by_k[k][(li, hi)] = {'U_K': U_k, 'mean_K': mean_k}
        R_by_k[k] = random_rotation_matrix(k, seed=0)
        print(" done")

    for k in K_VALUES:
        # SubRotQ
        tag = ("subrotq", k, N_BITS)
        if tag not in done:
            print(f"\nSubRotQ k={k}/{N_BITS}-bit...")
            def subrotq_fn(li, hi, x, _k=k):
                b = bases_by_k[_k][(li, hi)]
                return subspace_compress(x, _k, N_BITS, b['U_K'], b['mean_K'],
                                         R_by_k[_k], quantizer='subrotq')
            ppl = eval_with_k_hook(model, eval_ids, subrotq_fn)
            rel = ppl / baseline_ppl
            print(f"  PPL={ppl:.4f}  rel={rel:.4f}")
            rows.append({"method": "subrotq", "k": k, "bits": N_BITS,
                         "ppl": round(ppl, 4), "rel_ppl": round(rel, 4),
                         "note": f"PCA k={k} + random rotation + uniform 4-bit"})

        # PolarQuant (Han et al.)
        tag = ("polarquant", k, N_BITS)
        if tag not in done:
            print(f"\nPolarQuant k={k}/{N_BITS}-bit...")
            def polarquant_fn(li, hi, x, _k=k):
                b = bases_by_k[_k][(li, hi)]
                return subspace_compress(x, _k, N_BITS, b['U_K'], b['mean_K'],
                                         R_by_k[_k], quantizer='polarquant')
            ppl = eval_with_k_hook(model, eval_ids, polarquant_fn)
            rel = ppl / baseline_ppl
            print(f"  PPL={ppl:.4f}  rel={rel:.4f}")
            rows.append({"method": "polarquant", "k": k, "bits": N_BITS,
                         "ppl": round(ppl, 4), "rel_ppl": round(rel, 4),
                         "note": f"PCA k={k} + random rotation + polar coordinate 4-bit"})

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # Report
    wall = (time.time() - t0) / 60

    sr128 = next((r for r in rows if r['method'] == 'subrotq' and r['k'] == 128), None)
    pq128 = next((r for r in rows if r['method'] == 'polarquant' and r['k'] == 128), None)
    plain = next((r for r in rows if r['method'] == 'plain_4bit'), None)

    with open(report_path, "w") as f:
        f.write("# Task B1: SubRotQ vs PolarQuant (clean WikiText-2)\n\n")
        f.write(f"Model: {MODEL_NAME} | Baseline PPL: {baseline_ppl:.4f} | Wall: {wall:.1f} min\n\n")
        f.write("| Method | k | bits | PPL | rel PPL |\n|--------|---|------|-----|------|\n")
        for r in rows:
            f.write(f"| {r['method']} | {r['k']} | {r['bits']} | {r['ppl']} | {r['rel_ppl']} |\n")

        f.write("\n## Key comparisons\n\n")
        if sr128 and pq128:
            diff = float(sr128['rel_ppl']) - float(pq128['rel_ppl'])
            f.write(f"SubRotQ k=128: {sr128['rel_ppl']}x vs PolarQuant k=128: {pq128['rel_ppl']}x  ")
            f.write(f"(diff: {diff:+.4f}x)\n\n")
            if abs(diff) < 0.01:
                f.write("**Essentially identical at k=128 — both are full-rank with rotation, difference is only quant scheme.**\n")
            elif diff < 0:
                f.write(f"**SubRotQ wins by {abs(diff):.4f}x rel PPL at k=128.**\n")
            else:
                f.write(f"**PolarQuant wins by {abs(diff):.4f}x rel PPL at k=128.**\n")
        if sr128 and plain:
            diff2 = float(sr128['rel_ppl']) - float(plain['rel_ppl'])
            f.write(f"\nSubRotQ k=128 vs plain 4-bit: {diff2:+.4f}x rel PPL gap.\n")
            if abs(diff2) < 0.02:
                f.write("At full rank, SubRotQ and plain quant are indistinguishable.")

        # Prior exp22 context
        f.write("\n\n## Prior exp22 reference (contaminated pipeline)\n")
        f.write("exp22 reported SubRotQ k=112/4-bit: 1.0028x vs PolarQuant: 1.0556x on War & Peace.\n")
        f.write("This experiment provides the clean-benchmark equivalent.\n")

    print(f"\nCSV: {csv_path}")
    print(f"Report: {report_path}")
    print(f"Wall time: {wall:.1f} min")
    print("=== Done ===")


if __name__ == "__main__":
    main()
