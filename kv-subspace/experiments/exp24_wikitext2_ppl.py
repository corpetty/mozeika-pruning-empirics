"""
Experiment 24: WikiText-2 PPL — eval framework diagnostic + clean reference numbers.

MOTIVATION
----------
All prior experiments (exp18–23) used War & Peace as eval text with a broken
calib/eval split: EVAL_OFFSET = CALIB_TOKENS + 100 was treating a token count as
a character offset, placing the eval window *inside* the calibration window.

Additionally, W&P is a Gutenberg classic that is almost certainly in Qwen3's
training data — baseline PPL=1.17 on W&P is consistent with memorization, not
genuine language model uncertainty.

This experiment fixes both problems:
  - Calibration: WikiText-2 TRAIN split (first 2048 tokens)
  - Evaluation:  WikiText-2 TEST split (entirely held-out, ~280K tokens, use 2048)
  - No overlap possible — different dataset splits

WHAT WE EXPECT
--------------
Qwen2.5-14B on WikiText-2 test ≈ 5-6 PPL (from public benchmarks).
Qwen3-14B should be similar or better. If we get ≈1.17 here too, chunked_cross_entropy
has a bug. If we get ≈5-6, the W&P numbers were memorization and the eval framework
is structurally correct.

CONFIGS
-------
Same K-only configs as exp22 core sweep:
  k ∈ {64, 96, 112, 128}  ×  bits ∈ {4, 8, 16}
Plus baseline (no compression).

Output:
  results/exp24_wikitext2_ppl.csv
  results/REPORT-24-wikitext2.md
"""

import sys
import csv
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

CALIB_TOKENS = 2048   # tokens from WikiText-2 TRAIN for basis fitting
EVAL_TOKENS  = 2048   # tokens from WikiText-2 TEST for PPL evaluation

N_KV_HEADS = 8
D_HEAD     = 128
N_LAYERS   = 40

K_VALUES    = [64, 96, 112, 128]
BITS_VALUES = [4, 8, 16]

# ── Data loading ──────────────────────────────────────────────────────────────

def get_wikitext2_tokens(tokenizer, split, n_tokens, device):
    """
    Load WikiText-2 split, concatenate all articles, tokenize, return first
    n_tokens as a (1, n_tokens) tensor.

    Uses the standard wikitext-2-raw-v1 config which preserves whitespace.
    The test split is ~282K tokens for Llama-style tokenizers, so 2048 is safe.
    """
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split,
                      trust_remote_code=True)
    # Concatenate all text with newlines (standard practice)
    text = "\n\n".join(ds["text"])
    # Remove empty lines artifact that can produce spurious tokens
    text = "\n".join(line for line in text.split("\n") if line.strip())

    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if ids.shape[1] < n_tokens + 1:
        raise ValueError(
            f"WikiText-2 {split} only has {ids.shape[1]} tokens, need {n_tokens+1}")
    ids = ids[:, :n_tokens].to(device)
    return ids


# ── Model helpers ─────────────────────────────────────────────────────────────

def find_attention_layers(model):
    transformer = model.model.model
    for i, layer in enumerate(transformer.layers):
        yield i, layer.self_attn


def collect_kvs_for_basis(model, input_ids, n_kv_heads, d_head):
    """
    Collect K and V projections for PCA basis fitting.
    Returns {(layer_idx, head_idx): {'K': (T, d_head), 'V': (T, d_head)}}
    """
    kv_store = {}
    hooks = []

    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_hook(li, kvt, nh, dh):
                def hook(module, inp, out):
                    x = out.detach().cpu().float()
                    x = x.reshape(x.shape[0], x.shape[1], nh, dh)[0]  # (T, nh, dh)
                    for h in range(nh):
                        key = (li, h)
                        if key not in kv_store:
                            kv_store[key] = {'K': [], 'V': []}
                        kv_store[key][kvt].append(x[:, h, :].numpy())
                return hook
            proj = getattr(attn, proj_name)
            hooks.append(proj.register_forward_hook(
                make_hook(layer_idx, kv_type, n_kv_heads, d_head)))

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    return {k: {kv: np.concatenate(v, axis=0) for kv, v in d.items()}
            for k, d in kv_store.items()}


def chunked_cross_entropy(model, input_ids, chunk_size=512):
    """
    Compute cross-entropy loss without materializing full logit matrix.
    Uses the transformer body (model.model.model) + lm_head (model.model.lm_head).

    Sanity check: baseline PPL should be ~5-6 for Qwen3-14B on WikiText-2 test.
    If it's ~1, there is a bug in this function.
    """
    transformer_body = model.model.model
    lm_head = model.model.lm_head

    with torch.no_grad():
        outputs = transformer_body(input_ids=input_ids[:, :-1])
        hidden = outputs.last_hidden_state  # (1, T-1, d_model)

    # Labels: tokens 1..T (what we're predicting)
    labels = input_ids[:, 1:].reshape(-1)  # (T-1,)

    total_loss = 0.0
    n_tok = 0
    with torch.no_grad():
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            chunk_logits = lm_head(hidden[:, start:end, :])       # (1, chunk, vocab)
            chunk_labels = labels[start:end]                        # (chunk,)
            loss = torch.nn.functional.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)),    # (chunk, vocab)
                chunk_labels)                                        # (chunk,)
            total_loss += float(loss) * (end - start)
            n_tok += (end - start)
            del chunk_logits, chunk_labels, loss
            torch.cuda.empty_cache()

    del hidden
    torch.cuda.empty_cache()
    return total_loss / n_tok  # mean NLL in nats


def direct_ppl(model, input_ids):
    """
    Cross-check: compute PPL via model.model (CausalLM) with built-in labels.
    This bypasses our chunked implementation and uses HuggingFace's loss directly.
    Should give identical result to chunked_cross_entropy if implementation is correct.
    """
    causal_lm = model.model
    with torch.no_grad():
        out = causal_lm(input_ids=input_ids, labels=input_ids)
    # HF CausalLM loss = mean NLL over all tokens (including first token prediction
    # from BOS, but that's a minor difference for 2048 tokens)
    return float(torch.exp(out.loss))


def eval_ppl_with_hooks(model, input_ids, bases, k, n_bits):
    """Eval PPL with K-only subspace compression hooks."""
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
                    key_bh = (li, h)
                    if key_bh not in bases:
                        continue
                    xh = x[0, :, h, :].numpy()
                    U  = bases[key_bh]['U_K']
                    mn = bases[key_bh]['mean_K']
                    R_key = (li, h)
                    if R_key not in R_cache:
                        R_cache[R_key] = random_rotation_matrix(k)
                    R = R_cache[R_key]
                    xh_c = subspace_compress(xh, k, n_bits, U, mn, R,
                                             quantizer='subrotq')
                    x[0, :, h, :] = torch.from_numpy(xh_c)
                return x.reshape(b, s, nh * dh).to(dty).to(dev)
            return hook
        hooks.append(attn.k_proj.register_forward_hook(
            make_hook(layer_idx, N_KV_HEADS, D_HEAD)))

    loss = chunked_cross_entropy(model, input_ids)
    for h in hooks:
        h.remove()
    return float(np.exp(loss))


def fit_bases(initial_kvs, k):
    bases = {}
    for (li, hi), kv in initial_kvs.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        U_v, mean_v = fit_pca(kv['V'], k)
        bases[(li, hi)] = {
            'U_K': U_k, 'mean_K': mean_k,
            'U_V': U_v, 'mean_V': mean_v,
        }
    return bases


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "exp24_wikitext2_ppl.csv"
    fieldnames = ["k", "bits", "ppl", "rel_ppl", "compression_type"]

    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["k"]), int(row["bits"])))
        print(f"Resuming: {len(done)} configs done")

    print(f"Loading model {MODEL_NAME}...")
    device = 'cuda'
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    print("Loading WikiText-2...")
    print("  Calibration: TRAIN split, first 2048 tokens")
    calib_ids = get_wikitext2_tokens(tokenizer, "train", CALIB_TOKENS, device)
    print(f"  Calib tokens: {calib_ids.shape[1]}")

    print("  Evaluation: TEST split, first 2048 tokens")
    eval_ids = get_wikitext2_tokens(tokenizer, "test", EVAL_TOKENS, device)
    print(f"  Eval tokens: {eval_ids.shape[1]}")

    # ── Baseline PPL — TWO implementations as cross-check ──
    print("\nComputing baseline PPL...")
    print("  Method 1: chunked_cross_entropy (our pipeline)")
    loss_chunked = chunked_cross_entropy(model, eval_ids)
    ppl_chunked = float(np.exp(loss_chunked))
    print(f"    PPL = {ppl_chunked:.4f}  (loss = {loss_chunked:.4f} nats)")

    print("  Method 2: direct HF CausalLM loss (independent verification)")
    ppl_direct = direct_ppl(model, eval_ids)
    print(f"    PPL = {ppl_direct:.4f}")

    delta = abs(ppl_chunked - ppl_direct) / ppl_direct
    print(f"  Relative difference: {delta*100:.2f}%")
    if delta > 0.05:
        print(f"  WARNING: >5% difference between methods — possible chunked_cross_entropy bug!")
    else:
        print(f"  OK: methods agree within 5%")

    expected_range = (3.0, 10.0)
    if not (expected_range[0] <= ppl_chunked <= expected_range[1]):
        print(f"\n  SANITY FAIL: expected baseline PPL in {expected_range}, got {ppl_chunked:.4f}")
        print(f"  If PPL < 3: possible memorization or implementation bug")
        print(f"  If PPL > 10: possible tokenization issue or wrong eval text")
    else:
        print(f"\n  SANITY OK: baseline PPL {ppl_chunked:.4f} is in expected range {expected_range}")

    baseline_ppl = ppl_chunked

    # Write baseline row
    if (-1, 0) not in done:
        with open(csv_path, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not Path(csv_path).stat().st_size:
                w.writeheader()
            w.writerow({"k": 128, "bits": 16, "ppl": round(baseline_ppl, 4),
                        "rel_ppl": 1.0, "compression_type": "baseline"})

    # ── Collect calibration KV basis ──
    print(f"\nCollecting KV basis from train split ({CALIB_TOKENS} tokens)...")
    initial_kvs = collect_kvs_for_basis(model, calib_ids, N_KV_HEADS, D_HEAD)
    print(f"  Collected {len(initial_kvs)} (layer, head) pairs")

    # Pre-fit bases for all k values
    bases_by_k = {}
    for k in K_VALUES:
        print(f"  Fitting bases k={k}...", end='', flush=True)
        bases_by_k[k] = fit_bases(initial_kvs, k)
        print(" done")

    # ── Sweep ──
    print(f"\nRunning K-only compression sweep...")
    for k in K_VALUES:
        for n_bits in BITS_VALUES:
            if (k, n_bits) in done:
                print(f"  [skip] k={k} bits={n_bits}")
                continue

            print(f"  k={k:3d} bits={n_bits:2d}", end='', flush=True)
            ppl = eval_ppl_with_hooks(model, eval_ids, bases_by_k[k], k, n_bits)
            rel_ppl = ppl / baseline_ppl
            cr = (D_HEAD * 16) / (k * n_bits + (D_HEAD - k) * 16)
            print(f"  PPL={ppl:.4f}  rel={rel_ppl:.4f}  CR={cr:.2f}x")

            row = {"k": k, "bits": n_bits, "ppl": round(ppl, 4),
                   "rel_ppl": round(rel_ppl, 4), "compression_type": "K_only"}
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

    baseline_row = next((r for r in rows if r['compression_type'] == 'baseline'), None)
    compressed = [r for r in rows if r['compression_type'] != 'baseline']
    compressed.sort(key=lambda r: (int(r['k']), int(r['bits'])))

    report = f"""# Experiment 24: WikiText-2 PPL — Eval Framework Diagnostic

## Purpose

Verify that our `chunked_cross_entropy` evaluation pipeline is correct, and
establish clean reference PPL numbers using a standard benchmark.

**Prior bug:** Experiments 18–23 used War & Peace with calib/eval overlap
(EVAL_OFFSET = CALIB_TOKENS + 100 chars ≈ 572 tokens, inside calib window of
2048 tokens). Additionally, W&P is in Qwen3's training data (baseline PPL ≈ 1.17,
consistent with memorization).

**Fix:** Calibrate on WikiText-2 train split, evaluate on WikiText-2 test split.
No overlap possible. Standard benchmark with known reference values.

## Baseline Verification

| Method | PPL |
|--------|-----|
| chunked_cross_entropy (our pipeline) | {ppl_chunked:.4f} |
| HF CausalLM direct loss (cross-check) | {ppl_direct:.4f} |
| Relative difference | {delta*100:.2f}% |

Expected range for Qwen3-14B on WikiText-2 test: **3–7 PPL**

{"✅ SANITY PASS" if expected_range[0] <= ppl_chunked <= expected_range[1] else "❌ SANITY FAIL"}

{"✅ Methods agree" if delta <= 0.05 else "❌ WARNING: methods disagree >5%"}

## K-Only Compression Results

| k | bits | PPL | rel_PPL | CR |
|---|------|-----|---------|-----|
| 128 | 16 | {float(baseline_row['ppl']):.4f} | 1.000 | 1.00× |
"""
    for r in compressed:
        k_ = int(r['k'])
        b_ = int(r['bits'])
        cr_ = (D_HEAD * 16) / (k_ * b_ + (D_HEAD - k_) * 16)
        report += f"| {k_} | {b_} | {float(r['ppl']):.4f} | {float(r['rel_ppl']):.4f} | {cr_:.2f}× |\n"

    report += f"""
## Key Findings

1. **Eval framework status:** {"✅ chunked_cross_entropy is correct" if delta <= 0.05 else "❌ Bug in chunked_cross_entropy — investigate"}
2. **W&P baseline PPL=1.17 was:** {"memorization artifact (correct eval gives ~" + f"{ppl_chunked:.1f})" if ppl_chunked > 2 else "also suspicious"}
3. **Calib/eval split:** Clean (train → test, no overlap)
4. **Reference point:** Qwen3-14B WikiText-2 test PPL = {ppl_chunked:.4f}

## Implications for Prior Experiments

The relative PPL ratios (compressed/baseline) from prior experiments may still be
directionally correct since both baseline and compressed saw the same eval text.
However, absolute PPL values are unreliable due to:
  - Calib/eval text overlap (eval inside calib window)
  - Training data contamination (W&P memorization)

Experiments whose **conclusions** hold regardless of absolute PPL:
  - Exp19: V online updating is a null result (mechanism is structural, not data-dependent)
  - Exp16: Layer sensitivity ranking (relative ordering, same eval for all)
  - Exp22/23: SubRotQ vs PolarQuant comparison (same eval for both)

Experiments needing re-run with clean eval:
  - Core bitrate sweep (exp9 equivalent) → this experiment
  - Cross-arch validation (exp21) → will need Llama re-run on WikiText-2
"""

    report_path = RESULTS_DIR / "REPORT-24-wikitext2.md"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")
    print(f"\nBaseline PPL: {ppl_chunked:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
