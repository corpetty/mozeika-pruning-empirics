"""
Experiment 15: Needle-in-a-haystack retrieval accuracy.

Answers: Does KV compression degrade fact retrieval at varying positions in long docs?

Methodology:
  - Build synthetic "haystack" documents of varying length (4K, 8K, 16K, 32K tokens)
  - Insert a unique fact ("needle") at depths 10%, 25%, 50%, 75%, 90%
  - Ask the model to retrieve the needle; check if answer contains the expected value
  - Compare: baseline vs k128_4bit vs k96_4bit

Usage:
    python experiments/exp15_needle.py

Outputs:
    results/exp15_needle.csv
    results/REPORT-15-needle.md
"""

import sys
import csv
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

CTX_LENGTHS  = [4096, 8192, 16384, 32768]
DEPTHS       = [0.10, 0.25, 0.50, 0.75, 0.90]

NEEDLES = [
    ("The secret code for the mission is: DELTA-7749.",
     "What is the secret code for the mission?",
     "DELTA-7749"),
    ("The password to the vault is: ORCHID-4821.",
     "What is the password to the vault?",
     "ORCHID-4821"),
    ("The launch confirmation phrase is: GLACIER-9953.",
     "What is the launch confirmation phrase?",
     "GLACIER-9953"),
]

CONFIGS = {
    "baseline":  (None,       None, None, None,       None, None),
    "k128_4bit": ("subspace", 128,  4,    "subspace", 128,  4),
    "k96_4bit":  ("subspace", 96,   4,    "subspace", 96,   4),
}

MAX_NEW_TOKENS = 32


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


def install_hooks(model, cfg, bases, n_kv_heads, d_head):
    K_method, K_k, K_bits, V_method, V_k, V_bits = cfg
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name, method, k, bits in [
            ('K', 'k_proj', K_method, K_k, K_bits),
            ('V', 'v_proj', V_method, V_k, V_bits),
        ]:
            if method is None:
                continue
            def make_hook(li, kvt, m, kk, nb):
                def hook(module, input, output):
                    dev, dty = output.device, output.dtype
                    x = output.detach().cpu().float()
                    b, s, _ = x.shape
                    x = x.reshape(b, s, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = x[0, :, h, :].numpy()
                        base = bases.get((li, h), {})
                        x[0, :, h, :] = torch.from_numpy(
                            compress_vec(xh, m, kk, nb,
                                         base.get(f'U_{kvt}'), base.get(f'mean_{kvt}')))
                    return x.reshape(b, s, -1).to(device=dev, dtype=dty)
                return hook
            hooks.append(getattr(attn, proj_name).register_forward_hook(
                make_hook(layer_idx, kv_type, method, k, bits)))
    return hooks


def build_haystack_ids(tokenizer, base_text, needle_sentence, ctx_len, depth, device):
    """Build input_ids with needle inserted at `depth` fraction of the haystack."""
    needle_tokens = tokenizer.encode(" " + needle_sentence, add_special_tokens=False)
    budget = ctx_len - len(needle_tokens)
    base_tokens = tokenizer.encode(base_text, add_special_tokens=False)
    # Use a window well past calibration
    skip = CALIB_TOKENS + 512
    base_tokens = base_tokens[skip:]
    if len(base_tokens) < budget:
        reps = (budget // len(base_tokens)) + 2
        base_tokens = (base_tokens * reps)
    base_tokens = base_tokens[:budget]
    insert_pos = int(depth * len(base_tokens))
    full = base_tokens[:insert_pos] + needle_tokens + base_tokens[insert_pos:]
    full = full[:ctx_len]
    return torch.tensor([full], dtype=torch.long).to(device)


def run_needle_trial(model, tokenizer, base_text, bases_by_k, n_kv_heads, d_head,
                     device, ctx_len, depth, needle_sentence, question, expected,
                     cfg_name, cfg):
    K_method, K_k, K_bits, V_method, V_k, V_bits = cfg
    k_for_basis = K_k if K_k is not None else 128
    bases = bases_by_k.get(k_for_basis, bases_by_k.get(128, {}))

    input_ids = build_haystack_ids(tokenizer, base_text, needle_sentence, ctx_len, depth, device)

    suffix = f"\n\nAnswer with only the exact value, nothing else.\nQuestion: {question}\nAnswer:"
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    suffix_tensor = torch.tensor([suffix_ids], dtype=torch.long).to(device)
    input_ids = torch.cat([input_ids, suffix_tensor], dim=1)

    hooks = install_hooks(model, cfg, bases, n_kv_heads, d_head)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )

    for h in hooks:
        h.remove()

    generated_ids = output[0, input_ids.shape[1]:]
    answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    correct = expected.lower() in answer.lower()

    del input_ids, output
    torch.cuda.empty_cache()

    return {
        "ctx_len": ctx_len,
        "depth": depth,
        "config": cfg_name,
        "needle_key": needle_sentence[:40],
        "expected": expected,
        "answer": answer[:80],
        "correct": int(correct),
    }


def main():
    print("=" * 70)
    print("Experiment 15: Needle-in-a-Haystack Retrieval")
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
    bases_by_k = {k: fit_bases(calib_kvs, k) for k in [96, 128]}
    print(f"  Fitted {len(bases_by_k[128])} (layer, head) bases")

    base_text = DATA_FILE.read_text(encoding="utf-8", errors="replace")

    csv_path = RESULTS_DIR / "exp15_needle.csv"
    fieldnames = ["ctx_len", "depth", "config", "needle_key", "expected", "answer", "correct"]
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["ctx_len"]), float(row["depth"]), row["config"], row["needle_key"]))
        print(f"\nResuming: {len(done)} trials done")

    for ctx_len in CTX_LENGTHS:
        for depth in DEPTHS:
            for needle_sentence, question, expected in NEEDLES:
                nkey = needle_sentence[:40]
                for cfg_name, cfg in CONFIGS.items():
                    key = (ctx_len, depth, cfg_name, nkey)
                    if key in done:
                        continue
                    print(f"  ctx={ctx_len:>5}  depth={depth:.0%}  cfg={cfg_name:<12}  "
                          f"needle='{nkey[:30]}'", flush=True)
                    try:
                        result = run_needle_trial(
                            model, tokenizer, base_text, bases_by_k, n_kv_heads, d_head,
                            device, ctx_len, depth, needle_sentence, question, expected,
                            cfg_name, cfg)
                        mark = "✓" if result["correct"] else "✗"
                        print(f"    {mark}  answer='{result['answer'][:50]}'")

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

    from collections import defaultdict
    acc_ctx  = defaultdict(list)
    acc_dep  = defaultdict(list)
    for r in all_rows:
        acc_ctx [(r["config"], int(r["ctx_len"]))].append(int(r["correct"]))
        acc_dep [(r["config"], float(r["depth"]))].append(int(r["correct"]))

    report_path = RESULTS_DIR / "REPORT-15-needle.md"
    with open(report_path, 'w') as f:
        f.write("# Experiment 15: Needle-in-a-Haystack Retrieval\n\n")
        f.write(f"- Model: Qwen3-14B-AWQ ({n_layers} layers, {n_kv_heads} KV heads)\n")
        f.write(f"- {len(NEEDLES)} needles × {len(DEPTHS)} depths × {len(CTX_LENGTHS)} ctx lengths\n\n")

        f.write("## Accuracy by Config × Context Length\n\n")
        f.write("| Config | " + " | ".join(f"ctx={c}" for c in CTX_LENGTHS) + " | Overall |\n")
        f.write("|--------|" + "|".join("---" for _ in CTX_LENGTHS) + "|---|\n")
        for cfg_name in CONFIGS:
            vals = [acc_ctx.get((cfg_name, c), []) for c in CTX_LENGTHS]
            strs = [f"{sum(v)/len(v)*100:.0f}%({sum(v)}/{len(v)})" if v else "N/A" for v in vals]
            all_v = [x for v in vals for x in v]
            overall = f"{sum(all_v)/len(all_v)*100:.0f}%" if all_v else "N/A"
            f.write(f"| {cfg_name} | " + " | ".join(strs) + f" | {overall} |\n")

        f.write("\n## Accuracy by Config × Depth\n\n")
        f.write("| Config | " + " | ".join(f"{d:.0%}" for d in DEPTHS) + " |\n")
        f.write("|--------|" + "|".join("---" for _ in DEPTHS) + "|\n")
        for cfg_name in CONFIGS:
            vals = [acc_dep.get((cfg_name, d), []) for d in DEPTHS]
            strs = [f"{sum(v)/len(v)*100:.0f}%" if v else "N/A" for v in vals]
            f.write(f"| {cfg_name} | " + " | ".join(strs) + " |\n")

    print(f"\nSaved {csv_path}")
    print(f"Wrote {report_path}")
    print("\n" + "=" * 70)
    print("Experiment 15 complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
