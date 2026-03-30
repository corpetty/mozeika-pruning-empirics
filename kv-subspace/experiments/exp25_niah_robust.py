"""
Experiment 25: Needle-in-a-Haystack — statistically robust (n≥10 per cell).

MOTIVATION
----------
Exp15 used only 3 needles per (depth, ctx_len) cell. With n=3 per cell,
a single miss swings accuracy by 33 percentage points — statistically meaningless.
This experiment uses 15 unique needles per cell, giving ≥10 samples per cell
and ±26% 95% CI for each accuracy estimate (still wide, but at least honest).

DESIGN
------
- 15 unique needle facts (alphanumeric codes, all syntactically distinct)
- 5 insertion depths: 10%, 25%, 50%, 75%, 90%
- 4 context lengths: 4096, 8192, 16384, 32768
- 3 configs: baseline, k128_4bit, k96_4bit
- Total: 15 × 5 × 4 × 3 = 900 trials

METHODOLOGY FIXES vs EXP15
--------------------------
- Calibration: WikiText-2 train split (not W&P) — clean, no memorization
- n=15 per cell (was 3)
- Results reported with 95% Wilson confidence intervals, not bare percentages

Output:
  results/exp25_niah_robust.csv
  results/REPORT-25-niah.md
"""

import sys
import csv
import json
import time
import math
import random
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

CALIB_TOKENS  = 2048
N_KV_HEADS    = 8
D_HEAD        = 128
N_LAYERS      = 40
MAX_NEW_TOKENS = 32

DEPTHS      = [0.10, 0.25, 0.50, 0.75, 0.90]
CTX_LENGTHS = [4096, 8192, 16384, 32768]

CONFIGS = {
    "baseline":  {"k": 128, "bits": 16, "compress": False},
    "k128_4bit": {"k": 128, "bits": 4,  "compress": True},
    "k96_4bit":  {"k": 96,  "bits": 4,  "compress": True},
}

# 15 unique needle facts with unambiguous, extractable answers
NEEDLES = [
    ("The authentication token for server ALPHA is: BRAVO-7741.",   "What is the authentication token for server ALPHA?",          "BRAVO-7741"),
    ("The unlock code for vault DELTA is: ECHO-3829.",              "What is the unlock code for vault DELTA?",                   "ECHO-3829"),
    ("The mission confirmation phrase is: FOXTROT-5512.",           "What is the mission confirmation phrase?",                   "FOXTROT-5512"),
    ("The encryption key for channel GOLF is: HOTEL-6647.",         "What is the encryption key for channel GOLF?",               "HOTEL-6647"),
    ("The access passphrase for zone INDIA is: JULIET-2293.",       "What is the access passphrase for zone INDIA?",              "JULIET-2293"),
    ("The recovery phrase for account KILO is: LIMA-8830.",         "What is the recovery phrase for account KILO?",              "LIMA-8830"),
    ("The serial number of unit MIKE is: NOVEMBER-4417.",           "What is the serial number of unit MIKE?",                    "NOVEMBER-4417"),
    ("The distress signal identifier for post OSCAR is: PAPA-9962.","What is the distress signal identifier for post OSCAR?",     "PAPA-9962"),
    ("The frequency override for relay QUEBEC is: ROMEO-3351.",     "What is the frequency override for relay QUEBEC?",           "ROMEO-3351"),
    ("The detonation code for device SIERRA is: TANGO-7703.",       "What is the detonation code for device SIERRA?",             "TANGO-7703"),
    ("The registration tag for asset UNIFORM is: VICTOR-1184.",     "What is the registration tag for asset UNIFORM?",            "VICTOR-1184"),
    ("The failsafe phrase for system WHISKEY is: XRAY-6628.",       "What is the failsafe phrase for system WHISKEY?",            "XRAY-6628"),
    ("The coordination code for operation YANKEE is: ZULU-4459.",   "What is the coordination code for operation YANKEE?",        "ZULU-4459"),
    ("The override command for terminal ALPHA-2 is: BRAVO-9915.",   "What is the override command for terminal ALPHA-2?",         "BRAVO-9915"),
    ("The clearance code for sector CHARLIE is: DELTA-5537.",       "What is the clearance code for sector CHARLIE?",             "DELTA-5537"),
]

assert len(NEEDLES) == 15, "Must have exactly 15 needles"


# ── Stats helpers ─────────────────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return p, max(0, center - half), min(1, center + half)


# ── Model helpers ─────────────────────────────────────────────────────────────

def find_attention_layers(model):
    for i, layer in enumerate(model.model.model.layers):
        yield i, layer.self_attn


def get_wikitext2_tokens(tokenizer, split, n_tokens, device):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, trust_remote_code=True)
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
        bases[(li, hi)] = {'U_K': U_k, 'mean_K': mean_k}
    return bases


def build_compression_hooks(model, bases, k, n_bits, R_cache):
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        def make_hook(li, nh, dh):
            def hook(module, inp, out):
                dev, dty = out.device, out.dtype
                x = out.detach().cpu().float()
                b, s, _ = x.shape
                x = x.reshape(b, s, nh, dh)
                for h in range(nh):
                    if (li, h) not in bases:
                        continue
                    xh = x[0, :, h, :].numpy()
                    U  = bases[(li, h)]['U_K']
                    mn = bases[(li, h)]['mean_K']
                    R_key = (li, h)
                    if R_key not in R_cache:
                        R_cache[R_key] = random_rotation_matrix(k)
                    xh_c = subspace_compress(xh, k, n_bits, U, mn, R_cache[R_key], quantizer='subrotq')
                    x[0, :, h, :] = torch.from_numpy(xh_c)
                return x.reshape(b, s, nh * dh).to(dty).to(dev)
            return hook
        hooks.append(attn.k_proj.register_forward_hook(
            make_hook(layer_idx, N_KV_HEADS, D_HEAD)))
    return hooks


def build_haystack(tokenizer, haystack_text, needle_sentence, ctx_len, depth, device):
    """Insert needle at `depth` fraction of token budget."""
    needle_ids = tokenizer.encode(" " + needle_sentence, add_special_tokens=False)
    budget = ctx_len - len(needle_ids) - 20   # 20 tokens margin for question
    base_ids = tokenizer.encode(haystack_text, add_special_tokens=False)[:budget]
    insert_pos = int(len(base_ids) * depth)
    full_ids = base_ids[:insert_pos] + needle_ids + base_ids[insert_pos:]
    return torch.tensor([full_ids], dtype=torch.long, device=device)


def run_trial(model, tokenizer, haystack_ids, question, expected, device):
    """Run a single needle retrieval trial. Returns (answer_str, correct_bool)."""
    q_ids = tokenizer.encode(
        f"\n\nQuestion: {question}\nAnswer:", add_special_tokens=False)
    full_ids = torch.cat([
        haystack_ids,
        torch.tensor([q_ids], dtype=torch.long, device=device)
    ], dim=1)

    with torch.no_grad():
        output = model.model.generate(
            full_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output[0, full_ids.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    correct = expected.upper() in answer.upper()
    return answer, correct


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path  = RESULTS_DIR / "exp25_niah_robust.csv"
    fieldnames = ["ctx_len", "depth", "config", "needle_idx",
                  "expected", "answer", "correct"]

    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["ctx_len"]), float(row["depth"]),
                          row["config"], int(row["needle_idx"])))
        print(f"Resuming: {len(done)} trials done")

    total = len(CTX_LENGTHS) * len(DEPTHS) * len(CONFIGS) * len(NEEDLES)
    print(f"Total trials: {total} ({len(NEEDLES)} needles × {len(DEPTHS)} depths × "
          f"{len(CTX_LENGTHS)} ctx_lens × {len(CONFIGS)} configs)")

    print(f"\nLoading model {MODEL_NAME}...")
    device = 'cuda'
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    print("Loading WikiText-2 train for calibration...")
    calib_ids = get_wikitext2_tokens(tokenizer, "train", CALIB_TOKENS, device)
    print(f"  {calib_ids.shape[1]} tokens")

    print("Collecting KV basis...")
    initial_kvs = collect_kvs_for_basis(model, calib_ids, N_KV_HEADS, D_HEAD)

    # Pre-fit bases
    bases_by_k = {}
    R_caches = {}
    for k in [96, 128]:
        print(f"  Fitting k={k}...", end='', flush=True)
        bases_by_k[k] = fit_bases(initial_kvs, k)
        R_caches[k] = {}
        print(" done")

    # Load haystack text (WikiText-2 test, long enough for 32K context)
    print("Loading haystack text (WikiText-2 test)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=True)
    haystack_text = "\n".join(line for line in "\n\n".join(ds["text"]).split("\n") if line.strip())
    print(f"  {len(haystack_text)} chars available")

    # Pre-warm generation
    print("Pre-warming generation...")
    dummy_ids = torch.zeros(1, 16, dtype=torch.long, device=device)
    with torch.no_grad():
        model.model.generate(dummy_ids, max_new_tokens=1,
                              do_sample=False, pad_token_id=tokenizer.eos_token_id)
    print("  Warm-up done")

    done_count = len(done)
    t_start = time.time()

    for ctx_len in CTX_LENGTHS:
        for depth in DEPTHS:
            for config_name, cfg in CONFIGS.items():
                for ni, (needle, question, expected) in enumerate(NEEDLES):
                    key = (ctx_len, depth, config_name, ni)
                    if key in done:
                        continue

                    # Build haystack once per (ctx_len, depth, needle)
                    haystack_ids = build_haystack(
                        tokenizer, haystack_text, needle, ctx_len, depth, device)

                    # Install compression hooks if needed
                    hooks = []
                    if cfg["compress"]:
                        k = cfg["k"]
                        hooks = build_compression_hooks(
                            model, bases_by_k[k], k, cfg["bits"], R_caches[k])

                    answer, correct = run_trial(
                        model, tokenizer, haystack_ids, question, expected, device)

                    for h in hooks:
                        h.remove()

                    done_count += 1
                    elapsed = time.time() - t_start
                    rate = done_count / elapsed if elapsed > 0 else 0
                    remaining = (total - done_count) / rate / 60 if rate > 0 else 0
                    print(f"  [{done_count}/{total}] ctx={ctx_len} depth={depth:.0%} "
                          f"{config_name} n={ni} → {'✓' if correct else '✗'} "
                          f"({remaining:.0f}min left)")

                    row = {
                        "ctx_len": ctx_len, "depth": depth,
                        "config": config_name, "needle_idx": ni,
                        "expected": expected, "answer": answer[:80],
                        "correct": int(correct),
                    }
                    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
                    with open(csv_path, 'a', newline='') as f:
                        w = csv.DictWriter(f, fieldnames=fieldnames)
                        if not file_exists:
                            w.writeheader()
                        w.writerow(row)

    # ── Report ──
    print("\nGenerating report...")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    report_lines = [
        "# Experiment 25: Needle-in-a-Haystack (Robust, n=15 per cell)\n",
        f"- Model: {MODEL_NAME}",
        f"- 15 needles × {len(DEPTHS)} depths × {len(CTX_LENGTHS)} ctx_lengths × {len(CONFIGS)} configs",
        f"- Calibration: WikiText-2 train (not W&P — no memorization)",
        "",
        "## Accuracy by Config × Context Length (% correct, 95% Wilson CI)",
        "",
        "| Config | ctx=4096 | ctx=8192 | ctx=16384 | ctx=32768 | Overall |",
        "|--------|----------|----------|-----------|-----------|---------|",
    ]

    for config_name in CONFIGS:
        row_parts = [f"| {config_name}"]
        all_n, all_k = 0, 0
        for ctx in CTX_LENGTHS:
            cell = [r for r in rows if int(r["ctx_len"]) == ctx
                    and r["config"] == config_name]
            n = len(cell)
            k = sum(int(r["correct"]) for r in cell)
            all_n += n; all_k += k
            p, lo, hi = wilson_ci(k, n)
            row_parts.append(f"{p*100:.0f}% ({k}/{n}) [{lo*100:.0f}–{hi*100:.0f}%]")
        p, lo, hi = wilson_ci(all_k, all_n)
        row_parts.append(f"{p*100:.0f}% ({all_k}/{all_n}) [{lo*100:.0f}–{hi*100:.0f}%]")
        report_lines.append(" | ".join(row_parts) + " |")

    report_lines += [
        "",
        "## Accuracy by Config × Depth (% correct, all ctx_lengths pooled)",
        "",
        "| Config | 10% | 25% | 50% | 75% | 90% |",
        "|--------|-----|-----|-----|-----|-----|",
    ]
    for config_name in CONFIGS:
        row_parts = [f"| {config_name}"]
        for depth in DEPTHS:
            cell = [r for r in rows if abs(float(r["depth"]) - depth) < 0.01
                    and r["config"] == config_name]
            n = len(cell)
            k = sum(int(r["correct"]) for r in cell)
            p, lo, hi = wilson_ci(k, n)
            row_parts.append(f"{p*100:.0f}% [{lo*100:.0f}–{hi*100:.0f}%]")
        report_lines.append(" | ".join(row_parts) + " |")

    report_lines += [
        "",
        "## Notes",
        "- All accuracy estimates reported with 95% Wilson confidence intervals",
        "- n=15 per (config, depth, ctx_len) cell",
        "- Prior exp15 had n=3/cell — single miss changed accuracy by 33pp",
        "- W&P-based calibration replaced with WikiText-2 train split",
    ]

    report_path = RESULTS_DIR / "REPORT-25-niah.md"
    report_path.write_text("\n".join(report_lines))
    print(f"Report: {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
