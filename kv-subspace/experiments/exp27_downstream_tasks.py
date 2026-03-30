"""
Experiment 27: Downstream task evaluation — MMLU, ARC-Challenge, HellaSwag, GSM8K.

MOTIVATION
----------
All prior quality measurements used perplexity. PPL measures language modeling
quality but doesn't directly answer "does the model still answer questions correctly
with compressed KV cache?" This experiment measures accuracy on standard benchmarks
using the lm-eval-harness framework.

Configs tested:
  - baseline (no compression)
  - k=128, 4-bit (best compression, 4× CR)
  - k=112, 4-bit (borderline compression, ~3× CR)
  - k=96, 4-bit (aggressive compression, ~2.3× CR)

Tasks:
  - MMLU (5-shot): knowledge breadth, 57 subjects
  - ARC-Challenge (25-shot): science reasoning
  - HellaSwag (10-shot): commonsense NLI
  - GSM8K (5-shot, CoT): math reasoning

NOTE: lm-eval can't directly hook into the model like our PPL experiments.
We use a custom model wrapper that installs compression hooks before evaluation.

Output:
  results/exp27_downstream_tasks.json
  results/REPORT-27-downstream.md
"""

import sys
import json
import os
import subprocess
import tempfile
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect import get_model_and_tokenizer
from compress import fit_pca, subspace_compress, random_rotation_matrix

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "Qwen/Qwen3-14B-AWQ"
RESULTS_DIR = Path("results")

CALIB_TOKENS = 2048
N_KV_HEADS   = 8
D_HEAD       = 128

# Tasks: (lm-eval name, num_fewshot)
TASKS = [
    ("mmlu",           5),
    ("arc_challenge",  25),
    ("hellaswag",      10),
    ("gsm8k",          5),
]

CONFIGS = {
    "baseline":  {"k": 128, "bits": 16, "compress": False},
    "k128_4bit": {"k": 128, "bits": 4,  "compress": True},
    "k112_4bit": {"k": 112, "bits": 4,  "compress": True},
    "k96_4bit":  {"k": 96,  "bits": 4,  "compress": True},
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def find_attention_layers(model):
    for i, layer in enumerate(model.model.model.layers):
        yield i, layer.self_attn


def get_wikitext2_tokens(tokenizer, n_tokens, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    text = "\n".join(line for line in "\n\n".join(ds["text"]).split("\n") if line.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return ids[:, :n_tokens].to(device)


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


def fit_bases(initial_kvs, k):
    bases = {}
    for (li, hi), kv in initial_kvs.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        bases[(li, hi)] = {'U_K': U_k.astype(np.float32), 'mean_K': mean_k.astype(np.float32)}
    return bases


def install_compression_hooks(model, bases, k, n_bits, n_kv_heads, d_head):
    """Install K-only compression hooks. Returns list of hook handles."""
    hooks = []
    R_cache = {}
    for layer_idx, attn in find_attention_layers(model):
        def make_hook(li, nh, dh):
            def hook(module, inp, out):
                dev, dty = out.device, out.dtype
                x = out.detach().cpu().float()
                b, s, _ = x.shape
                x = x.reshape(b, s, nh, dh)
                for h in range(nh):
                    key = (li, h)
                    if key not in bases:
                        continue
                    xh = x[0, :, h, :].numpy()
                    U  = bases[key]['U_K']
                    mn = bases[key]['mean_K']
                    R_key = (li, h)
                    if R_key not in R_cache:
                        R_cache[R_key] = random_rotation_matrix(k)
                    xh_c = subspace_compress(xh, k, n_bits, U, mn, R_cache[R_key],
                                             quantizer='subrotq')
                    x[0, :, h, :] = torch.from_numpy(xh_c)
                return x.reshape(b, s, nh * dh).to(dty).to(dev)
            return hook
        hooks.append(attn.k_proj.register_forward_hook(
            make_hook(layer_idx, n_kv_heads, d_head)))
    return hooks


# ── lm-eval integration ───────────────────────────────────────────────────────

def run_lm_eval_with_model(model, tokenizer, task_name, num_fewshot, device):
    """
    Run lm-eval directly using the Python API with our pre-loaded model.
    This avoids subprocess overhead and lets us install hooks into the model.
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    # Wrap model in HFLM (HuggingFace Language Model) interface
    # We pass our already-loaded model directly
    lm = HFLM(
        pretrained=model.model,  # the CausalLM
        tokenizer=tokenizer,
        device=device,
        batch_size=1,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        limit=500,       # cap at 500 samples per task for speed (still statistically meaningful)
        log_samples=False,
    )
    return results["results"][task_name]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    RESULTS_DIR.mkdir(exist_ok=True)

    json_path = RESULTS_DIR / "exp27_downstream_tasks.json"
    all_results = {}
    if json_path.exists():
        all_results = json.loads(json_path.read_text())
        print(f"Resuming: {len(all_results)} configs done")

    print(f"Loading model {MODEL_NAME}...")
    device = 'cuda'
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    print("Loading WikiText-2 for calibration...")
    calib_ids = get_wikitext2_tokens(tokenizer, CALIB_TOKENS, device)

    print("Collecting KV basis...")
    initial_kvs = collect_kvs_for_basis(model, calib_ids, N_KV_HEADS, D_HEAD)

    print("Fitting bases...")
    bases_by_k = {}
    for k in [96, 112, 128]:
        print(f"  k={k}...", end='', flush=True)
        bases_by_k[k] = fit_bases(initial_kvs, k)
        print(" done")

    for config_name, cfg in CONFIGS.items():
        if config_name in all_results:
            print(f"\n[skip] {config_name}")
            continue

        print(f"\n{'='*50}")
        print(f"Config: {config_name}")

        # Install hooks
        hooks = []
        if cfg["compress"]:
            k = cfg["k"]
            hooks = install_compression_hooks(
                model, bases_by_k[k], k, cfg["bits"], N_KV_HEADS, D_HEAD)
            print(f"  Compression hooks: k={k}, {cfg['bits']}-bit ({len(hooks)} hooks)")

        config_results = {}
        for task_name, n_fewshot in TASKS:
            print(f"  Task: {task_name} ({n_fewshot}-shot, limit=500)...", end='', flush=True)
            try:
                task_result = run_lm_eval_with_model(
                    model, tokenizer, task_name, n_fewshot, device)
                # Extract primary metric
                metric_key = "acc,none" if "acc,none" in task_result else list(task_result.keys())[0]
                acc = task_result.get(metric_key, task_result.get("acc", None))
                config_results[task_name] = {
                    "acc": float(acc) if acc is not None else None,
                    "metric_key": metric_key,
                    "full": {k: float(v) if isinstance(v, (int, float)) else v
                             for k, v in task_result.items()},
                }
                print(f" acc={acc:.4f}")
            except Exception as e:
                print(f" ERROR: {e}")
                config_results[task_name] = {"acc": None, "error": str(e)}

        for h in hooks:
            h.remove()

        all_results[config_name] = config_results
        json_path.write_text(json.dumps(all_results, indent=2))
        print(f"  Saved to {json_path}")

    # ── Report ──
    print("\nGenerating report...")
    report_lines = [
        "# Experiment 27: Downstream Task Accuracy\n",
        f"- Model: {MODEL_NAME}",
        "- Tasks: MMLU (5-shot), ARC-Challenge (25-shot), HellaSwag (10-shot), GSM8K (5-shot)",
        "- Limit: 500 samples per task",
        "- Calibration: WikiText-2 train, K-only compression",
        "",
        "## Results\n",
        "| Config | MMLU | ARC-C | HellaSwag | GSM8K | Notes |",
        "|--------|------|-------|-----------|-------|-------|",
    ]

    baseline = all_results.get("baseline", {})
    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        res = all_results[config_name]
        row_parts = [f"| {config_name}"]
        for task_name, _ in TASKS:
            tr = res.get(task_name, {})
            acc = tr.get("acc")
            if acc is None:
                row_parts.append("ERROR")
            else:
                b_acc = baseline.get(task_name, {}).get("acc")
                if b_acc and config_name != "baseline":
                    delta = acc - b_acc
                    row_parts.append(f"{acc:.3f} ({delta:+.3f})")
                else:
                    row_parts.append(f"{acc:.3f}")
        cfg = CONFIGS[config_name]
        note = "no compression" if not cfg["compress"] else f"k={cfg['k']}, {cfg['bits']}-bit"
        row_parts.append(note)
        report_lines.append(" | ".join(row_parts) + " |")

    report_lines += [
        "",
        "## Key Findings",
        "- Delta shown relative to baseline (positive = better, negative = worse)",
        "- Tasks capped at 500 samples each for runtime",
    ]

    report_path = RESULTS_DIR / "REPORT-27-downstream.md"
    report_path.write_text("\n".join(report_lines))
    print(f"Report: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
