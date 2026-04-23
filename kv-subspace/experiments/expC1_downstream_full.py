"""
Task C1: Full-N downstream tasks (1000 samples per task).

MOTIVATION
----------
exp27 used LIMIT=300 (~±5.5% CI). C1 uses LIMIT=1000 (~±3.2% CI) to narrow
confidence intervals and confirm the k=128/4-bit vs k=96/4-bit quality cliff
with higher statistical power.

DESIGN
------
Model: Qwen3-14B-AWQ (clean pipeline, WikiText-2 train for calibration)
Configs: baseline, k128_4bit, k96_4bit  (k=112 deferred — borderline, not headline)
Tasks: ARC-Challenge (25-shot), HellaSwag (10-shot), Winogrande (5-shot), ARC-Easy (0-shot)
Limit: 1000 samples per task
Calibration: WikiText-2 train split (2048 tokens) — no overlap with downstream tasks

OUTPUT
------
  results/expC1_downstream_full.json   — per-task accuracy + metadata
  results/REPORT-C1-downstream-full.md — summary table
"""

import sys
import json
import os
import time
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect import get_model_and_tokenizer
from compress import fit_pca, random_rotation_matrix

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
RESULTS_DIR  = Path("results")
CALIB_TOKENS = 2048
LIMIT        = 1000   # samples per task; ~±3.2% CI at 95%
N_KV_HEADS   = 8
D_HEAD       = 128
SEED         = 0

TASKS = [
    ("arc_challenge", 25),
    ("hellaswag",     10),
    ("arc_easy",       0),
    ("winogrande",     5),
]

CONFIGS = {
    "baseline":  {"k": 128, "bits": 16, "compress": False},
    "k128_4bit": {"k": 128, "bits": 4,  "compress": True},
    "k96_4bit":  {"k": 96,  "bits": 4,  "compress": True},
}


def find_attention_layers(model):
    for i, layer in enumerate(model.model.model.layers):
        yield i, layer.self_attn


def get_wikitext2_calib_tokens(tokenizer, n_tokens, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(l for l in "\n\n".join(ds["text"]).split("\n") if l.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] < n_tokens:
        raise ValueError(f"Only {ids.shape[1]} calib tokens available")
    return ids[:, :n_tokens].to(device)


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


def uniform_quant_dequant(x_np, n_bits=4):
    x = x_np.copy().astype(np.float32)
    n_levels = 2 ** n_bits
    x_min = x.min(axis=0, keepdims=True)
    x_max = x.max(axis=0, keepdims=True)
    scale = (x_max - x_min) / (n_levels - 1)
    scale = np.where(scale == 0, 1e-8, scale)
    q = np.clip(np.round((x - x_min) / scale), 0, n_levels - 1)
    return (q * scale + x_min).astype(np.float32)


def compress_subrotq(x, k, U_k, mean_k, R_k, n_bits=4):
    xc      = (x - mean_k) @ U_k[:, :k]
    rotated = xc @ R_k.T
    quant   = uniform_quant_dequant(rotated, n_bits)
    unrot   = quant @ R_k
    return (unrot @ U_k[:, :k].T + mean_k).astype(np.float32)


def install_compression_hooks(model, bases, R, k, n_bits):
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
                    b_data = bases[(li, h)]
                    x[0, :, h, :] = torch.from_numpy(
                        compress_subrotq(xh, k, b_data['U_K'], b_data['mean_K'],
                                         b_data['R_k'], n_bits))
                return x.reshape(b, s, N_KV_HEADS * D_HEAD).to(dty).to(dev)
            return hook
        hooks.append(attn.k_proj.register_forward_hook(make_hook()))
    return hooks


def remove_hooks(hooks):
    for h in hooks:
        h.remove()


def eval_lm_harness(model, tokenizer, task_name, n_fewshot, limit):
    """Run lm-eval-harness on one task, return accuracy."""
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        raise ImportError("lm-eval not installed. Run: pip install lm-eval")

    lm = HFLM(pretrained=model.model, tokenizer=tokenizer, batch_size=1)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=[task_name],
        num_fewshot=n_fewshot,
        limit=limit,
        log_samples=False,
    )
    task_res = results["results"][task_name]
    # Extract primary accuracy metric
    for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
        if key in task_res:
            return float(task_res[key])
    # Fallback: find first float value
    for v in task_res.values():
        if isinstance(v, float):
            return v
    raise ValueError(f"No accuracy found in {task_res}")


def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    t0 = time.time()
    RESULTS_DIR.mkdir(exist_ok=True)

    out_path    = RESULTS_DIR / "expC1_downstream_full.json"
    report_path = RESULTS_DIR / "REPORT-C1-downstream-full.md"

    # Resume support
    results_data = {}
    if out_path.exists():
        with open(out_path) as f:
            results_data = json.load(f)
        print(f"Resuming: {len(results_data)} configs already done")

    print(f"=== Task C1: Full-N Downstream Tasks (N={LIMIT}) ===")
    print(f"Loading model {MODEL_NAME}...")
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()
    device = next(model.parameters()).device

    print("Loading WikiText-2 calibration tokens...")
    calib_ids = get_wikitext2_calib_tokens(tokenizer, CALIB_TOKENS, device)

    print("Collecting KV basis...")
    kvs = collect_kvs_for_basis(model, calib_ids)
    print(f"  {len(kvs)} (layer, head) pairs")

    # Pre-compute bases for all k values needed
    all_k_values = set(cfg['k'] for cfg in CONFIGS.values() if CONFIGS[list(CONFIGS.keys())[0]]['compress'] or True)
    all_k_values = {cfg['k'] for cfg in CONFIGS.values() if cfg['compress']}

    bases_by_k = {}
    for k in all_k_values:
        print(f"  Fitting k={k}...", end="", flush=True)
        R_k = random_rotation_matrix(k, seed=SEED)
        bases_by_k[k] = {}
        for (li, hi), kv in kvs.items():
            U_k, mean_k = fit_pca(kv['K'], k)
            bases_by_k[k][(li, hi)] = {'U_K': U_k, 'mean_K': mean_k, 'R_k': R_k}
        print(" done")

    for config_name, cfg in CONFIGS.items():
        if config_name in results_data:
            print(f"\nSkipping {config_name} (already done)")
            continue

        print(f"\n{'='*50}")
        print(f"Config: {config_name}  (k={cfg['k']}, {cfg['bits']}-bit, compress={cfg['compress']})")
        print(f"{'='*50}")

        # Install hooks if compressing
        hooks = []
        if cfg['compress']:
            hooks = install_compression_hooks(
                model, bases_by_k[cfg['k']], None, cfg['k'], cfg['bits'])
            print(f"  Installed {len(hooks)} compression hooks")

        config_results = {
            "config": config_name,
            "k": cfg['k'],
            "bits": cfg['bits'],
            "compress": cfg['compress'],
            "tasks": {}
        }

        for task_name, n_fewshot in TASKS:
            t_start = time.time()
            print(f"  Task: {task_name} ({n_fewshot}-shot, limit={LIMIT})...")
            try:
                acc = eval_lm_harness(model, tokenizer, task_name, n_fewshot, LIMIT)
                elapsed = time.time() - t_start
                print(f"    acc={acc:.4f}  ({elapsed:.1f}s)")
                config_results["tasks"][task_name] = {
                    "accuracy": acc,
                    "n_fewshot": n_fewshot,
                    "limit": LIMIT,
                    "elapsed_s": round(elapsed, 1)
                }
            except Exception as e:
                print(f"    ERROR: {e}")
                config_results["tasks"][task_name] = {"error": str(e)}

        remove_hooks(hooks)

        results_data[config_name] = config_results

        # Save after each config
        with open(out_path, "w") as f:
            json.dump(results_data, f, indent=2)
        print(f"  Saved checkpoint to {out_path}")

    # Final save
    with open(out_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # Report
    wall = (time.time() - t0) / 60
    task_names = [t for t, _ in TASKS]

    with open(report_path, "w") as f:
        f.write("# Task C1: Full-N Downstream Tasks\n\n")
        f.write(f"Model: {MODEL_NAME} | N={LIMIT}/task | Wall: {wall:.1f} min\n\n")
        f.write("## Accuracy Table\n\n")

        header = "| Config | " + " | ".join(task_names) + " |\n"
        sep    = "|--------|" + "|".join(["--------"] * len(task_names)) + "|\n"
        f.write(header)
        f.write(sep)

        for config_name in CONFIGS:
            if config_name not in results_data:
                continue
            row = f"| {config_name} |"
            for task_name in task_names:
                task_r = results_data[config_name]["tasks"].get(task_name, {})
                if "accuracy" in task_r:
                    row += f" {task_r['accuracy']:.4f} |"
                else:
                    row += " ERROR |"
            f.write(row + "\n")

        f.write("\n## Relative Accuracy (vs baseline)\n\n")
        f.write("| Config | " + " | ".join(task_names) + " |\n")
        f.write("|--------|" + "|".join(["--------"] * len(task_names)) + "|\n")

        baseline = results_data.get("baseline", {}).get("tasks", {})
        for config_name in CONFIGS:
            if config_name == "baseline" or config_name not in results_data:
                continue
            row = f"| {config_name} |"
            for task_name in task_names:
                task_r = results_data[config_name]["tasks"].get(task_name, {})
                base_r = baseline.get(task_name, {})
                if "accuracy" in task_r and "accuracy" in base_r:
                    delta = task_r['accuracy'] - base_r['accuracy']
                    row += f" {delta:+.4f} |"
                else:
                    row += " ? |"
            f.write(row + "\n")

        f.write(f"\n_Calibration: WikiText-2 train split ({CALIB_TOKENS} tokens). "
                f"No overlap with downstream task data._\n")

    print(f"\nOutput: {out_path}")
    print(f"Report: {report_path}")
    print(f"Wall time: {wall:.1f} min")
    print("=== Done ===")


if __name__ == "__main__":
    main()
