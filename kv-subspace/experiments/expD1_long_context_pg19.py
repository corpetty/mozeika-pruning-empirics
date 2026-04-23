"""
Experiment D1: Long-Context PPL sweep on PG-19.

Replaces exp13 (War & Peace) which used a potentially memorized text.
PG-19 test split = pre-1919 Project Gutenberg books, unlikely to appear
verbatim in Qwen3-14B training data.

Design:
  - Calibration: first CALIB_TOKENS tokens from PG-19 test doc #0
  - Evaluation:  tokens from doc #1 onward (held-out from calibration)
  - Context lengths: 512, 1K, 2K, 4K, 8K, 16K, 32K
  - Configs: baseline, k128/4-bit, k96/4-bit
  - Compression via k_proj hook (same pattern as exp24, long_context_scaling)
  - Saves incrementally after each ctx_len — safe to kill/resume

Outputs:
    results/expD1_long_context_pg19.csv
    results/expD1_long_context_pg19.json
    results/REPORT-D1-long-context-pg19.md
"""

import sys, os, csv, json, time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import fit_pca, subspace_compress, random_rotation_matrix
from collect  import get_model_and_tokenizer, find_attention_layers

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME   = 'Qwen/Qwen3-14B-AWQ'
N_KV_HEADS   = 8
D_HEAD       = 128
CALIB_TOKENS = 2048
CTX_LENGTHS  = [512, 1024, 2048, 4096, 8192, 16384, 32768]

CONFIGS = {
    'baseline':  (None, None, None),   # (method, k, bits)
    'k128_4bit': ('subrotq', 128, 4),
    'k96_4bit':  ('subrotq',  96, 4),
}

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
OUT_CSV  = RESULTS_DIR / 'expD1_long_context_pg19.csv'
OUT_JSON = RESULTS_DIR / 'expD1_long_context_pg19.json'
OUT_MD   = RESULTS_DIR / 'REPORT-D1-long-context-pg19.md'


# ── Helpers (exact pattern from exp24 / long_context_scaling) ─────────────────

def _get_transformer_body_and_head(model):
    causal_lm = getattr(model, 'model', model)
    if hasattr(causal_lm, 'model') and hasattr(causal_lm, 'lm_head'):
        return causal_lm.model, causal_lm.lm_head
    return causal_lm, model.lm_head


def chunked_cross_entropy(model, input_ids, chunk_size=512):
    """Standard chunked PPL — runs transformer body once, projects in chunks."""
    transformer_body, lm_head = _get_transformer_body_and_head(model)
    with torch.no_grad():
        outputs = transformer_body(input_ids=input_ids[:, :-1])
        hidden  = outputs.last_hidden_state
    labels  = input_ids[:, 1:].view(-1)
    seq_len = hidden.shape[1]
    total_loss, total_n = 0.0, 0
    with torch.no_grad():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            logits = lm_head(hidden[:, start:end, :])
            chunk_labels = labels[start:end]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), chunk_labels)
            total_loss += float(loss) * (end - start)
            total_n    += (end - start)
            del logits, chunk_labels
    del hidden
    torch.cuda.empty_cache()
    return total_loss / total_n


def collect_kvs_for_basis(model, input_ids):
    """Collect K/V projections via k_proj/v_proj hooks for PCA fitting."""
    kv_store = {}
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            def make_hook(li, kvt):
                def hook(module, inp, out):
                    x = out.detach().cpu().float()
                    x = x.reshape(x.shape[0], x.shape[1], N_KV_HEADS, D_HEAD)[0]
                    for h in range(N_KV_HEADS):
                        key = (li, h)
                        if key not in kv_store:
                            kv_store[key] = {'K': [], 'V': []}
                        kv_store[key][kvt].append(x[:, h, :].numpy())
                return hook
            proj = getattr(attn, proj_name)
            hooks.append(proj.register_forward_hook(make_hook(layer_idx, kv_type)))
    with torch.no_grad():
        model(input_ids=input_ids)
    for h in hooks:
        h.remove()
    return {k: {kv: np.concatenate(v, axis=0) for kv, v in d.items()}
            for k, d in kv_store.items()}


def fit_bases(kv_data, k):
    """Fit PCA bases for all (layer, head) pairs at rank k."""
    bases = {}
    for (li, hi), kv in kv_data.items():
        U_k, mean_k = fit_pca(kv['K'], k)
        bases[(li, hi)] = {'U_K': U_k, 'mean_K': mean_k}
    return bases


def install_hooks(model, method, k, n_bits, bases):
    """Install k_proj compression hooks. Returns list of hook handles."""
    if method is None:
        return []
    R_cache = {}
    hooks = []
    for layer_idx, attn in find_attention_layers(model):
        def make_hook(li):
            def hook(module, inp, out):
                device, dtype = out.device, out.dtype
                x = out.detach().cpu().float()
                b, s, _ = x.shape
                x = x.reshape(b, s, N_KV_HEADS, D_HEAD)
                for h in range(N_KV_HEADS):
                    key = (li, h)
                    if key not in bases:
                        continue
                    xh = x[0, :, h, :].numpy()
                    U  = bases[key]['U_K']
                    mn = bases[key]['mean_K']
                    if key not in R_cache:
                        R_cache[key] = random_rotation_matrix(k)
                    R = R_cache[key]
                    xh_c = subspace_compress(xh, k, n_bits, U, mn, R,
                                             quantizer='subrotq')
                    x[0, :, h, :] = torch.from_numpy(xh_c)
                return x.reshape(b, s, N_KV_HEADS * D_HEAD).to(dtype=dtype, device=device)
            return hook
        hooks.append(attn.k_proj.register_forward_hook(make_hook(layer_idx)))
    return hooks


# ── Data loading ───────────────────────────────────────────────────────────────

def load_pg19_tokens(tokenizer, device, max_tokens):
    """
    Load PG-19 test split. Returns:
      calib_ids: (1, CALIB_TOKENS) from doc #0
      eval_ids:  (1, max_tokens+1) from doc #1+ (held-out)
    """
    from datasets import load_dataset
    print('Loading PG-19 test split (streaming)...')
    ds = load_dataset('emozilla/pg19', split='test', streaming=True)
    docs = []
    for i, sample in enumerate(ds):
        docs.append(sample['text'])
        if i >= 5:
            break

    # Calibration from doc #0
    enc = tokenizer(docs[0], return_tensors='pt', truncation=True,
                    max_length=CALIB_TOKENS + 1, add_special_tokens=True)
    calib_ids = enc['input_ids'][:, :CALIB_TOKENS].to(device)
    print(f'  Calibration: {calib_ids.shape[1]} tokens (doc #0)')

    # Evaluation from docs #1+
    eval_text = ' '.join(docs[1:])
    enc2 = tokenizer(eval_text, return_tensors='pt', truncation=True,
                     max_length=max_tokens + 1, add_special_tokens=True)
    eval_ids = enc2['input_ids'].to(device)
    print(f'  Evaluation:  {eval_ids.shape[1]} tokens (docs #1+)')
    return calib_ids, eval_ids


# ── Resume support ─────────────────────────────────────────────────────────────

def load_existing(csv_path):
    if not csv_path.exists():
        return []
    rows = []
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            r['ctx_len'] = int(r['ctx_len'])
            r['ppl']     = float(r['ppl'])
            r['rel_ppl'] = float(r['rel_ppl']) if r['rel_ppl'] != '' else None
            rows.append(r)
    return rows


def save_partial(rows, csv_path):
    # Recompute rel_ppl
    base = {r['ctx_len']: r['ppl'] for r in rows if r['config'] == 'baseline'}
    for r in rows:
        b = base.get(r['ctx_len'])
        r['rel_ppl'] = round(r['ppl'] / b, 4) if b else ''
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['ctx_len', 'config', 'ppl', 'rel_ppl'])
        w.writeheader()
        w.writerows(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print('=== Experiment D1: Long-Context PPL on PG-19 ===')
    print(f'Model: {MODEL_NAME}')
    print(f'Configs: {list(CONFIGS.keys())}')
    print(f'Context lengths: {CTX_LENGTHS}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    print('\nLoading model...')
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    # Load PG-19
    max_ctx = max(CTX_LENGTHS) + 1
    calib_ids, eval_ids = load_pg19_tokens(tokenizer, device, max_ctx)

    # Fit bases
    print('\nCollecting KV for PCA basis (calibration pass)...')
    kv_data = collect_kvs_for_basis(model, calib_ids)
    print(f'  Collected from {len(kv_data)} (layer, head) pairs')
    bases_128 = fit_bases(kv_data, 128)
    bases_96  = fit_bases(kv_data, 96)
    print('  Bases fitted (k=128, k=96)')
    del kv_data
    torch.cuda.empty_cache()

    def get_bases(method, k):
        if method is None: return {}
        return bases_128 if k == 128 else bases_96

    # Resume
    rows = load_existing(OUT_CSV)
    done = {(r['ctx_len'], r['config']) for r in rows}
    if done:
        print(f'\nResuming: {len(done)} (ctx_len, config) pairs already done')

    # Sweep
    print()
    for ctx_len in CTX_LENGTHS:
        needed = [cn for cn in CONFIGS if (ctx_len, cn) not in done]
        if not needed:
            print(f'ctx={ctx_len:6d}  [all done, skipping]')
            continue

        if eval_ids.shape[1] < ctx_len + 1:
            print(f'ctx={ctx_len:6d}  [not enough tokens, skipping]')
            continue

        input_ids = eval_ids[:, :ctx_len + 1]

        for cfg_name in needed:
            method, k, bits = CONFIGS[cfg_name]
            bases = get_bases(method, k)

            hooks = install_hooks(model, method, k, bits, bases)
            t_s = time.time()
            loss = chunked_cross_entropy(model, input_ids)
            for h in hooks:
                h.remove()

            ppl = float(np.exp(loss))
            elapsed = time.time() - t_s
            print(f'ctx={ctx_len:6d}  {cfg_name:12s}  PPL={ppl:.4f}  ({elapsed:.1f}s)')
            rows.append({'ctx_len': ctx_len, 'config': cfg_name,
                         'ppl': round(ppl, 4), 'rel_ppl': ''})
            torch.cuda.empty_cache()

        save_partial(rows, OUT_CSV)
        print(f'  [saved → {OUT_CSV}]')

    # Final rel_ppl pass
    base_ppl = {r['ctx_len']: r['ppl'] for r in rows if r['config'] == 'baseline'}
    for r in rows:
        b = base_ppl.get(r['ctx_len'])
        r['rel_ppl'] = round(r['ppl'] / b, 4) if b else ''
    save_partial(rows, OUT_CSV)

    # JSON
    out_data = {'model': MODEL_NAME, 'calib_tokens': CALIB_TOKENS,
                'corpus': 'emozilla/pg19 test', 'ctx_lengths': CTX_LENGTHS,
                'configs': list(CONFIGS.keys()), 'rows': rows}
    with open(OUT_JSON, 'w') as f:
        json.dump(out_data, f, indent=2)
    print(f'Saved: {OUT_JSON}')

    # Markdown report
    total_min = (time.time() - t0) / 60
    k128_rels = [r['rel_ppl'] for r in rows
                 if r['config'] == 'k128_4bit' and isinstance(r['rel_ppl'], float)]
    k96_rels  = [r['rel_ppl'] for r in rows
                 if r['config'] == 'k96_4bit'  and isinstance(r['rel_ppl'], float)]

    md = f"""# Experiment D1: Long-Context PPL on PG-19

**Model:** {MODEL_NAME}
**Corpus:** emozilla/pg19 test split (pre-1919 Project Gutenberg books)
**Calibration:** {CALIB_TOKENS} tokens from doc #0
**Evaluation:** docs #1+ (held-out from calibration)
**Wall time:** {total_min:.1f} min

## Results

| ctx_len | config | PPL | rel_PPL |
|---------|--------|-----|---------|
"""
    for ctx_len in CTX_LENGTHS:
        for cfg_name in CONFIGS:
            r = next((x for x in rows if x['ctx_len'] == ctx_len and x['config'] == cfg_name), None)
            if r:
                md += f"| {ctx_len} | {cfg_name} | {r['ppl']:.4f} | {r['rel_ppl']} |\n"

    md += f"""
## Key Findings

- **k128/4-bit** rel_PPL: {f"{min(k128_rels):.4f}–{max(k128_rels):.4f}×" if k128_rels else "pending"}
- **k96/4-bit** rel_PPL:  {f"{min(k96_rels):.4f}–{max(k96_rels):.4f}×" if k96_rels else "pending"}
- Calibration: WikiText → PG-19 transfer (held-out eval split)
"""
    with open(OUT_MD, 'w') as f:
        f.write(md)
    print(f'Saved: {OUT_MD}')
    print(f'\n=== Done in {total_min:.1f} min ===')


if __name__ == '__main__':
    main()
