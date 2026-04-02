"""
Experiment 27: Downstream task evaluation with rich trajectory tracking.

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
  - k=96, 4-bit  (aggressive compression, ~2.3× CR)

Tasks (MC-only — generate_until tasks like GSM8K are too slow):
  - ARC-Challenge (25-shot): science reasoning
  - HellaSwag (10-shot): commonsense NLI
  - ARC-Easy (0-shot): easier ARC variant, good sanity check
  - WinoGrande (5-shot): coreference / commonsense

TRAJECTORY DATA SAVED (new in this version)
--------------------------------------------
For each (config, task) we save:
  1. per_sample: [{question_idx, log_prob_correct, log_prob_max_distractor,
                    margin, correct, question_len_tokens}]
     - margin = log_prob_correct - log_prob_max_distractor
     - lets us track confidence collapse even when accuracy is stable
  2. accuracy_by_len_bin: accuracy broken down by question token length
     - catches if compression hurts more on longer reasoning chains
  3. cumulative_accuracy: accuracy after each 10-sample window (learning curve)
     - shows if errors are clustered or spread uniformly

DIAGNOSTICS SAVED (per experiment run)
---------------------------------------
  results/exp27_diagnostics.json:
    - variance_explained: {k: fraction} — how much of K variance each subspace captures
    - basis_stability: cosine sim between bases fit on first vs second half of calib data
    - participation_ratio: per-layer intrinsic dimensionality estimate
    - per_layer_recon_error: ||K_compressed - K_orig||_F / ||K_orig||_F per layer
    - truncation_vs_quant_split: fraction of total error from truncation vs quantization

Output:
  results/exp27_downstream_tasks.json      — summary + per-sample trajectories
  results/exp27_diagnostics.json           — basis health metrics
  results/REPORT-27-downstream.md
"""

import sys
import json
import os
import numpy as np
import torch
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from collect import get_model_and_tokenizer
from compress import fit_pca, subspace_compress, random_rotation_matrix

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = "Qwen/Qwen3-14B-AWQ"
RESULTS_DIR  = Path("results")

CALIB_TOKENS = 2048
LIMIT        = 300    # samples per task; ~±5.5% CI at 95%
N_KV_HEADS   = 8
D_HEAD       = 128
LEN_BINS     = [0, 64, 128, 256, 512, 99999]   # token length bins for accuracy-by-len

TASKS = [
    ("arc_challenge", 25),
    ("hellaswag",     10),
    ("arc_easy",       0),
    ("winogrande",     5),
]

CONFIGS = {
    "baseline":  {"k": 128, "bits": 16, "compress": False},
    "k128_4bit": {"k": 128, "bits": 4,  "compress": True},
    "k112_4bit": {"k": 112, "bits": 4,  "compress": True},
    "k96_4bit":  {"k": 96,  "bits": 4,  "compress": True},
}

# ── Model helpers ─────────────────────────────────────────────────────────────

def find_attention_layers(model):
    for i, layer in enumerate(model.model.model.layers):
        yield i, layer.self_attn


def get_wikitext2_tokens(tokenizer, n_tokens, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", trust_remote_code=True)
    text = "\n".join(l for l in "\n\n".join(ds["text"]).split("\n") if l.strip())
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
        bases[(li, hi)] = {
            'U_K':  U_k.astype(np.float32),
            'mean_K': mean_k.astype(np.float32)
        }
    return bases


def install_compression_hooks(model, bases, k, n_bits, n_kv_heads, d_head):
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
                    if (li, h) not in R_cache:
                        R_cache[(li, h)] = random_rotation_matrix(k)
                    xh_c = subspace_compress(xh, k, n_bits, U, mn, R_cache[(li, h)],
                                             quantizer='subrotq')
                    x[0, :, h, :] = torch.from_numpy(xh_c)
                return x.reshape(b, s, nh * dh).to(dty).to(dev)
            return hook
        hooks.append(attn.k_proj.register_forward_hook(
            make_hook(layer_idx, n_kv_heads, d_head)))
    return hooks


# ── Diagnostics ───────────────────────────────────────────────────────────────

def compute_diagnostics(initial_kvs, n_kv_heads, d_head):
    """
    Compute basis health metrics from raw KV vectors.
    Returns a dict saved to exp27_diagnostics.json.
    
    Metrics:
    - variance_explained: fraction of K variance captured at each k
    - basis_stability: cosine sim between bases fit on first vs second half
    - participation_ratio: per-layer intrinsic dimensionality (PR = (Σλ)²/Σλ²)
    - per_layer_recon_error_k128: ||K_proj - K||_F/||K||_F per layer at k=128
    - per_layer_recon_error_k96: same at k=96
    - truncation_vs_quant_split_k128: fraction of error from truncation vs quantization
    """
    print("Computing diagnostics...")
    diag = {
        "model": MODEL_NAME,
        "calib_tokens": CALIB_TOKENS,
        "variance_explained": {},
        "basis_stability_mean_cosine": {},
        "participation_ratio_by_layer": {},
        "per_layer_recon_error": {},
        "truncation_vs_quant_split": {},
    }

    # --- variance explained and participation ratio by layer ---
    # Aggregate eigenvalues per layer (mean across heads)
    layer_pr = {}
    layer_var_k = {k: [] for k in [64, 96, 112, 128]}

    for (li, hi), kv in initial_kvs.items():
        X = kv['K']  # (T, d_head)
        X_centered = X - X.mean(axis=0)
        _, s, _ = np.linalg.svd(X_centered, full_matrices=False)
        eigvals = s ** 2
        total_var = eigvals.sum() + 1e-12

        # Participation ratio
        pr = (eigvals.sum() ** 2) / ((eigvals ** 2).sum() + 1e-12)
        if li not in layer_pr:
            layer_pr[li] = []
        layer_pr[li].append(float(pr))

        # Variance explained at each k
        cumvar = np.cumsum(eigvals) / total_var
        for k in [64, 96, 112, 128]:
            layer_var_k[k].append(float(cumvar[k - 1]) if k <= len(cumvar) else 1.0)

    diag["variance_explained"] = {
        str(k): float(np.mean(vals)) for k, vals in layer_var_k.items()
    }
    diag["participation_ratio_by_layer"] = {
        str(li): float(np.mean(prs)) for li, prs in sorted(layer_pr.items())
    }

    # --- basis stability: fit on first vs second half of calib data ---
    stabilities = []
    for (li, hi), kv in initial_kvs.items():
        X = kv['K']
        T = len(X)
        if T < 4:
            continue
        half = T // 2
        U1, _ = fit_pca(X[:half], 128)
        U2, _ = fit_pca(X[half:], 128)
        # Cosine sim between column spaces: mean abs cosine of principal angles
        cos_sims = np.abs(U1.T @ U2)  # (128, 128)
        # Take max per column (best match), then mean
        stab = float(cos_sims.max(axis=0).mean())
        stabilities.append(stab)
    diag["basis_stability_mean_cosine"] = {
        "mean": float(np.mean(stabilities)),
        "min":  float(np.min(stabilities)),
        "std":  float(np.std(stabilities)),
    }

    # --- per-layer reconstruction error + truncation/quant split ---
    # Use a small random rotation matrix for quantization error estimate
    R = random_rotation_matrix(128)
    R96 = random_rotation_matrix(96)

    recon_k128, recon_k96 = {}, {}
    trunc_fracs = []

    for (li, hi), kv in initial_kvs.items():
        X = kv['K']  # (T, d_head)
        U128, mean128 = fit_pca(X, 128)
        U96,  mean96  = fit_pca(X, 96)

        # k=128 reconstruction error
        X_proj128 = (X - mean128) @ U128 @ U128.T + mean128
        err128 = np.linalg.norm(X - X_proj128, 'fro') / (np.linalg.norm(X, 'fro') + 1e-12)

        # k=96 reconstruction error
        X_proj96 = (X - mean96) @ U96 @ U96.T + mean96
        err96 = np.linalg.norm(X - X_proj96, 'fro') / (np.linalg.norm(X, 'fro') + 1e-12)

        if li not in recon_k128:
            recon_k128[li] = []
            recon_k96[li]  = []
        recon_k128[li].append(float(err128))
        recon_k96[li].append(float(err96))

        # Truncation vs quantization split at k=128, 4-bit
        # truncation error: ||X - X_proj128||
        trunc_err = np.linalg.norm(X - X_proj128, 'fro') ** 2
        # quantization error: quantize the projected coefficients and measure round-trip
        coeffs = (X - mean128) @ U128  # (T, 128)
        # simulate 4-bit quantization per coefficient
        vmin, vmax = coeffs.min(axis=0), coeffs.max(axis=0)
        scale = (vmax - vmin) / 15.0 + 1e-12
        coeffs_q = np.round((coeffs - vmin) / scale) * scale + vmin
        X_quant = coeffs_q @ U128.T + mean128
        quant_err = np.linalg.norm(X_proj128 - X_quant, 'fro') ** 2
        total_err = trunc_err + quant_err + 1e-12
        trunc_fracs.append(float(trunc_err / total_err))

    diag["per_layer_recon_error"] = {
        "k128": {str(li): float(np.mean(v)) for li, v in sorted(recon_k128.items())},
        "k96":  {str(li): float(np.mean(v)) for li, v in sorted(recon_k96.items())},
    }
    diag["truncation_vs_quant_split"] = {
        "truncation_fraction_mean": float(np.mean(trunc_fracs)),
        "truncation_fraction_std":  float(np.std(trunc_fracs)),
        "quantization_fraction_mean": float(1 - np.mean(trunc_fracs)),
    }

    print(f"  Variance explained: k64={diag['variance_explained']['64']:.3f}  "
          f"k96={diag['variance_explained']['96']:.3f}  "
          f"k128={diag['variance_explained']['128']:.3f}")
    print(f"  Basis stability: mean={diag['basis_stability_mean_cosine']['mean']:.3f}  "
          f"min={diag['basis_stability_mean_cosine']['min']:.3f}")
    print(f"  Truncation fraction: {diag['truncation_vs_quant_split']['truncation_fraction_mean']:.3f}")

    diag_path = RESULTS_DIR / "exp27_diagnostics.json"
    diag_path.write_text(json.dumps(diag, indent=2))
    print(f"  Diagnostics saved to {diag_path}")
    return diag


# ── lm-eval integration with per-sample tracking ─────────────────────────────

def run_lm_eval_with_tracking(model, tokenizer, task_name, num_fewshot, device):
    """
    Run lm-eval with log_samples=True to capture per-sample data.
    Returns (summary_dict, trajectory_dict).

    trajectory_dict contains:
      - per_sample: list of {idx, correct, margin, question_len_tokens}
      - cumulative_accuracy: accuracy after each 10-sample window
      - accuracy_by_len_bin: {bin_label: {n, acc}} broken down by question length
    """
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(
        pretrained=model.model,
        tokenizer=tokenizer,
        device=device,
        batch_size=1,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=[task_name],
        num_fewshot=num_fewshot,
        limit=LIMIT,
        log_samples=True,   # <-- enables per-sample data
    )

    summary = results["results"][task_name]
    samples = results.get("samples", {}).get(task_name, [])

    # --- build per-sample records ---
    per_sample = []
    for i, s in enumerate(samples):
        # lm-eval sample structure varies by task; extract robustly
        # resps is a list of (logprob, is_greedy) or similar per answer choice
        correct = int(s.get("target", -1))  # index of correct choice (0-based)
        resps   = s.get("filtered_resps", s.get("resps", []))

        # Extract log probs: lm-eval stores them as [(logprob, ...)] per choice
        log_probs = []
        for r in resps:
            if isinstance(r, (list, tuple)) and len(r) > 0:
                lp = r[0] if isinstance(r[0], float) else (r[0][0] if isinstance(r[0], (list, tuple)) else None)
                log_probs.append(lp)
            elif isinstance(r, float):
                log_probs.append(r)

        lp_correct = log_probs[correct] if correct < len(log_probs) and log_probs[correct] is not None else None
        distractors = [lp for j, lp in enumerate(log_probs) if j != correct and lp is not None]
        lp_max_distractor = max(distractors) if distractors else None
        margin = (lp_correct - lp_max_distractor
                  if lp_correct is not None and lp_max_distractor is not None
                  else None)

        # Question length in tokens
        doc = s.get("doc", {})
        q_text = doc.get("question", doc.get("ctx", doc.get("sentence", "")))
        q_len = len(tokenizer(q_text, add_special_tokens=False)["input_ids"]) if q_text else 0

        # Was it correct? lm-eval stores this in "acc" or we infer from log probs
        is_correct = None
        if "acc" in s:
            is_correct = bool(s["acc"])
        elif log_probs and correct < len(log_probs) and log_probs[correct] is not None:
            is_correct = (lp_correct == max(lp for lp in log_probs if lp is not None))

        per_sample.append({
            "idx":               i,
            "correct":           is_correct,
            "log_prob_correct":  lp_correct,
            "log_prob_max_dist": lp_max_distractor,
            "margin":            margin,
            "question_len_tokens": q_len,
        })

    # --- cumulative accuracy trajectory (every 10 samples) ---
    cumulative_accuracy = []
    n_correct = 0
    for i, ps in enumerate(per_sample):
        if ps["correct"] is True:
            n_correct += 1
        if (i + 1) % 10 == 0 or i == len(per_sample) - 1:
            cumulative_accuracy.append({
                "after_n": i + 1,
                "acc":     round(n_correct / (i + 1), 4),
            })

    # --- accuracy by question length bin ---
    bin_labels = [f"{LEN_BINS[i]}-{LEN_BINS[i+1]}" for i in range(len(LEN_BINS) - 1)]
    acc_by_len = {label: {"n": 0, "correct": 0} for label in bin_labels}
    for ps in per_sample:
        if ps["correct"] is None:
            continue
        q_len = ps["question_len_tokens"]
        for i in range(len(LEN_BINS) - 1):
            if LEN_BINS[i] <= q_len < LEN_BINS[i + 1]:
                label = bin_labels[i]
                acc_by_len[label]["n"] += 1
                acc_by_len[label]["correct"] += int(ps["correct"])
                break
    acc_by_len_final = {
        label: {
            "n":   v["n"],
            "acc": round(v["correct"] / v["n"], 4) if v["n"] > 0 else None,
        }
        for label, v in acc_by_len.items()
    }

    # --- margin distribution summary ---
    margins = [ps["margin"] for ps in per_sample if ps["margin"] is not None]
    margin_stats = {}
    if margins:
        margin_stats = {
            "mean":   float(np.mean(margins)),
            "std":    float(np.std(margins)),
            "p10":    float(np.percentile(margins, 10)),
            "p25":    float(np.percentile(margins, 25)),
            "median": float(np.median(margins)),
            "p75":    float(np.percentile(margins, 75)),
            "p90":    float(np.percentile(margins, 90)),
            "frac_negative": float(np.mean([m < 0 for m in margins])),  # fraction where wrong answer was higher
        }

    trajectory = {
        "per_sample":          per_sample,
        "cumulative_accuracy": cumulative_accuracy,
        "accuracy_by_len_bin": acc_by_len_final,
        "margin_stats":        margin_stats,
    }

    return summary, trajectory


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                        help="Run only this config (baseline|k128_4bit|k112_4bit|k96_4bit). "
                             "If omitted, runs all configs sequentially.")
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    RESULTS_DIR.mkdir(exist_ok=True)

    json_path = RESULTS_DIR / "exp27_downstream_tasks.json"
    all_results = {}
    if json_path.exists():
        all_results = json.loads(json_path.read_text())
        print(f"Resuming: {len(all_results)} configs saved")

    # Filter to requested config only
    configs_to_run = CONFIGS
    if args.config:
        if args.config not in CONFIGS:
            print(f"Unknown config '{args.config}'. Choose from: {list(CONFIGS.keys())}")
            sys.exit(1)
        configs_to_run = {args.config: CONFIGS[args.config]}

    print(f"Loading model {MODEL_NAME}...")
    device = 'cuda'
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    print("Loading WikiText-2 for calibration...")
    calib_ids = get_wikitext2_tokens(tokenizer, CALIB_TOKENS, device)

    print("Collecting KV basis...")
    initial_kvs = collect_kvs_for_basis(model, calib_ids, N_KV_HEADS, D_HEAD)

    # --- Run diagnostics once (unless already done) ---
    diag_path = RESULTS_DIR / "exp27_diagnostics.json"
    if not diag_path.exists():
        compute_diagnostics(initial_kvs, N_KV_HEADS, D_HEAD)
    else:
        print("Diagnostics already exist, skipping.")

    print("Fitting bases...")
    bases_by_k = {}
    for k in [96, 112, 128]:
        print(f"  k={k}...", end='', flush=True)
        bases_by_k[k] = fit_bases(initial_kvs, k)
        print(" done")

    for config_name, cfg in configs_to_run.items():
        if config_name in all_results:
            tasks_done = [t for t in all_results[config_name] if all_results[config_name][t].get("acc") is not None]
            tasks_needed = [t for t, _ in TASKS]
            if all(t in tasks_done for t in tasks_needed):
                print(f"\n[skip] {config_name} (all {len(tasks_done)} tasks done)")
                continue

        print(f"\n{'='*50}")
        print(f"Config: {config_name}")

        hooks = []
        if cfg["compress"]:
            k = cfg["k"]
            hooks = install_compression_hooks(
                model, bases_by_k[k], k, cfg["bits"], N_KV_HEADS, D_HEAD)
            print(f"  Compression: k={k}, {cfg['bits']}-bit ({len(hooks)} hooks)")

        config_results = all_results.get(config_name, {})

        for task_name, n_fewshot in TASKS:
            if task_name in config_results and config_results[task_name].get("acc") is not None:
                print(f"  Task: {task_name} [skip, acc={config_results[task_name]['acc']:.4f}]")
                continue

            print(f"  Task: {task_name} ({n_fewshot}-shot, limit={LIMIT})...", end='', flush=True)
            try:
                task_result, trajectory = run_lm_eval_with_tracking(
                    model, tokenizer, task_name, n_fewshot, device)

                # extract primary metric
                if "acc,none" in task_result:
                    metric_key = "acc,none"
                elif "exact_match,none" in task_result:
                    metric_key = "exact_match,none"
                elif "acc_norm,none" in task_result:
                    metric_key = "acc_norm,none"
                else:
                    metric_key = next(
                        (k for k, v in task_result.items() if isinstance(v, (int, float))),
                        list(task_result.keys())[0]
                    )
                acc = task_result.get(metric_key)
                print(f" acc={acc:.4f}  margin_mean={trajectory['margin_stats'].get('mean', float('nan')):.3f}")

                config_results[task_name] = {
                    "acc":        float(acc) if acc is not None else None,
                    "metric_key": metric_key,
                    "summary":    {k: float(v) if isinstance(v, (int, float)) else v
                                   for k, v in task_result.items()},
                    "trajectory": trajectory,
                }
            except Exception as e:
                import traceback
                print(f" ERROR: {e}")
                traceback.print_exc()
                config_results[task_name] = {"acc": None, "error": str(e)}

            all_results[config_name] = config_results
            json_path.write_text(json.dumps(all_results, indent=2))
            print(f"  Saved → {json_path}")

        for h in hooks:
            h.remove()

        all_results[config_name] = config_results
        json_path.write_text(json.dumps(all_results, indent=2))

    # ── Report ────────────────────────────────────────────────────────────────
    print("\nGenerating report...")

    task_names = [t for t, _ in TASKS]
    baseline = all_results.get("baseline", {})

    lines = [
        "# Experiment 27: Downstream Task Accuracy\n",
        f"- Model: {MODEL_NAME}",
        f"- Tasks: {', '.join(task_names)}",
        f"- Samples/task: {LIMIT} (MC tasks only)",
        "- Calibration: WikiText-2 train split, K-only SubRotQ compression",
        "",
        "## Accuracy Summary\n",
        "| Config | ARC-C | HellaSwag | ARC-Easy | WinoGrande |",
        "|--------|-------|-----------|----------|-----------|",
    ]

    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        res = all_results[config_name]
        row = [f"| {config_name}"]
        for task_name in task_names:
            tr = res.get(task_name, {})
            acc = tr.get("acc")
            if acc is None:
                row.append("—")
            else:
                b_acc = baseline.get(task_name, {}).get("acc")
                if b_acc and config_name != "baseline":
                    row.append(f"{acc:.3f} ({acc - b_acc:+.3f})")
                else:
                    row.append(f"{acc:.3f}")
        lines.append(" | ".join(row) + " |")

    lines += [
        "",
        "## Confidence Margin Summary\n",
        "(mean log-prob margin: correct choice log-prob minus max distractor log-prob)\n",
        "| Config | ARC-C margin | HSwag margin | ARC-Easy margin | WinoGrande margin |",
        "|--------|-------------|-------------|----------------|-----------------|",
    ]

    for config_name in CONFIGS:
        if config_name not in all_results:
            continue
        res = all_results[config_name]
        row = [f"| {config_name}"]
        for task_name in task_names:
            tr = res.get(task_name, {})
            ms = tr.get("trajectory", {}).get("margin_stats", {})
            m  = ms.get("mean")
            row.append(f"{m:.3f}" if m is not None else "—")
        lines.append(" | ".join(row) + " |")

    lines += [
        "",
        "## Key Findings",
        "- Δ shown as (compressed − baseline); negative = degraded",
        "- Margin = log-prob correct − log-prob best distractor; collapse here indicates fragility even when accuracy holds",
        "",
        "## Diagnostics",
    ]

    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
        ve = diag.get("variance_explained", {})
        bs = diag.get("basis_stability_mean_cosine", {})
        tq = diag.get("truncation_vs_quant_split", {})
        lines += [
            f"- Variance explained: k96={ve.get('96', '?'):.3f}, k112={ve.get('112', '?'):.3f}, k128={ve.get('128', '?'):.3f}",
            f"- Basis stability (calib first/second half cosine sim): mean={bs.get('mean', '?'):.3f}, min={bs.get('min', '?'):.3f}",
            f"- Truncation vs quantization error split: truncation={tq.get('truncation_fraction_mean', '?'):.3f}",
        ]

    report_path = RESULTS_DIR / "REPORT-27-downstream.md"
    report_path.write_text("\n".join(lines))
    print(f"Report: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
