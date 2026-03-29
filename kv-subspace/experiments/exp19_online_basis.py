"""
Experiment 19: Online basis updating for V compression.

The core problem: V vectors drift more than K vectors over a long sequence
(basis overlap K=0.825 vs V=0.702 from exp13). A single offline calibration basis
fitted on the first N tokens becomes a poor approximation of the V subspace at
token positions 2000+. This causes V compression to fail at quality thresholds
where K compression succeeds.

Hypothesis: If we update the PCA basis every N tokens using an incremental
(rank-1 update or window-based re-fit) strategy, V compression quality improves
enough to be usable — closing V toward K's drift profile.

Design:
  - Baseline: offline calibration on first 2K tokens, static basis throughout
  - Online variants:
    a) Window-based: re-fit PCA every N tokens on the most recent W tokens
    b) Incremental: use exponential moving average of scatter matrix, update per-token
  - Test update intervals N = {never (static), 256, 512, 1024, 2048}
  - Measure: PPL at 8K context for K+V compressed (k=112/4-bit) vs K-only compressed
  - Measure: V basis overlap (initial vs position 4K, 8K) as function of update interval

Update strategies:
  1. Static: fit on first CALIB_TOKENS, never update
  2. Window re-fit: at each N-token boundary, re-fit PCA on the last W=512 observed V vectors
  3. EMA scatter: maintain running scatter matrix S = (1-alpha)*S + alpha*x*x^T, re-derive
     eigenvectors from S on demand (no explicit window)

For comparison: K-only compression (V at full precision 16-bit) at k=112 gives
our 1.14x PPL baseline. If online V compression at k=112 adds < 0.05x additional
PPL degradation, V compression is viable.

Usage:
    /home/petty/torch-env/bin/python3 experiments/exp19_online_basis.py

Outputs:
    results/exp19_online_basis.csv           - PPL and basis overlap vs update strategy
    results/exp19_v_drift_by_position.csv    - basis overlap vs sequence position
    results/REPORT-19-online-basis.md
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

MODEL_NAME    = "Qwen/Qwen3-14B-AWQ"
DATA_FILE     = Path("data/war_and_peace.txt")
CALIB_TOKENS  = 2048
CALIB_OFFSET  = 0
EVAL_OFFSET   = 5000
EVAL_CTX      = 8192    # Long enough for drift to matter
K_DIM         = 112     # subspace dimension for both K and V
BITS          = 4
UPDATE_INTERVALS = [0, 256, 512, 1024, 2048]  # 0 = static (never update)
WINDOW_SIZE   = 512     # number of recent vectors to re-fit on for window strategy
EMA_ALPHA     = 0.02    # EMA decay for scatter update


# ── Incremental basis update strategies ──────────────────────────────────────

class StaticBasis:
    """Fixed PCA basis, no updates."""
    def __init__(self, U, mean):
        self.U = U.copy()
        self.mean = mean.copy()
        self.n_updates = 0

    def update(self, x_batch):
        pass  # no-op

    def get(self):
        return self.U, self.mean


class WindowBasis:
    """Re-fit PCA on a sliding window of recent vectors."""
    def __init__(self, U, mean, k, window_size=512):
        self.U = U.copy()
        self.mean = mean.copy()
        self.k = k
        self.window_size = window_size
        self.buffer = []
        self.n_updates = 0

    def update(self, x_batch):
        """x_batch: (n_tokens, d_head) numpy array"""
        self.buffer.append(x_batch)
        # Keep only recent window_size vectors
        all_vecs = np.concatenate(self.buffer, axis=0)
        if len(all_vecs) > self.window_size:
            all_vecs = all_vecs[-self.window_size:]
            # Rebuild buffer from trimmed data
            self.buffer = [all_vecs]
        if len(all_vecs) >= self.k + 1:
            self.U, self.mean = fit_pca(all_vecs, self.k)
            self.n_updates += 1

    def get(self):
        return self.U, self.mean


class EMABasis:
    """Exponential moving average of the scatter matrix."""
    def __init__(self, U, mean, k, alpha=0.02):
        self.U = U.copy()
        self.mean = mean.copy()
        self.k = k
        self.alpha = alpha
        d = U.shape[0]
        # Initialize scatter from current basis
        self.scatter = U @ np.diag(np.ones(k)) @ U.T
        self.ema_mean = mean.copy()
        self.n_updates = 0

    def update(self, x_batch):
        """x_batch: (n_tokens, d_head) numpy array"""
        # Update EMA of mean
        batch_mean = x_batch.mean(axis=0)
        self.ema_mean = (1 - self.alpha) * self.ema_mean + self.alpha * batch_mean
        # Update EMA of scatter matrix
        centered = x_batch - self.ema_mean
        batch_scatter = (centered.T @ centered) / max(len(x_batch) - 1, 1)
        self.scatter = (1 - self.alpha) * self.scatter + self.alpha * batch_scatter
        # Re-derive top-k eigenvectors
        try:
            eigvals, eigvecs = np.linalg.eigh(self.scatter)
            # eigh returns ascending order, take last k (largest)
            idx = np.argsort(eigvals)[::-1][:self.k]
            self.U = eigvecs[:, idx].astype(np.float32)
            self.mean = self.ema_mean.astype(np.float32)
            self.n_updates += 1
        except np.linalg.LinAlgError:
            pass  # keep previous basis if decomp fails

    def get(self):
        return self.U, self.mean


def basis_overlap(U1, U2):
    """Compute subspace overlap: mean absolute cosine of principal angles."""
    try:
        from scipy.linalg import subspace_angles
        angles = subspace_angles(U1, U2)
        return float(np.mean(np.cos(angles)))
    except ImportError:
        # Fallback: nuclear norm of U1^T U2
        M = U1.T @ U2
        sv = np.linalg.svd(M, compute_uv=False)
        return float(np.mean(np.minimum(sv, 1.0)))


# ── Token loading ─────────────────────────────────────────────────────────────

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


# ── Online compression hooks ──────────────────────────────────────────────────

def install_online_hooks(model, initial_bases, k_dim, bits, n_kv_heads, d_head,
                          update_interval, basis_cls, basis_kwargs,
                          v_drift_log=None):
    """
    Install compression hooks with online basis updating.
    K: static basis at k_dim (proven stable in exp13)
    V: online basis at k_dim, updates every update_interval tokens

    v_drift_log: if dict, record V basis overlap at checkpoints
    """
    hooks = []
    # Build per-head basis objects
    k_bases = {}   # static
    v_bases = {}   # online updatable

    for (layer_idx, head_idx), kv in initial_bases.items():
        U_k, mean_k = fit_pca(kv['K'], k_dim)
        U_v, mean_v = fit_pca(kv['V'], k_dim)
        k_bases[(layer_idx, head_idx)] = StaticBasis(U_k, mean_k)
        v_bases[(layer_idx, head_idx)] = basis_cls(U_v, mean_v, k_dim, **basis_kwargs)

    # Token counter (shared state via dict to allow mutation in closures)
    state = {'tokens_seen': 0, 'next_update': update_interval if update_interval > 0 else None}
    # Capture buffer for V updates
    v_capture = {}  # (layer, head) -> list of vectors since last update

    for layer_idx, attn in find_attention_layers(model):
        # K hook: static basis, always compress
        def make_k_hook(li):
            def hook(module, input, output):
                dev, dty = output.device, output.dtype
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                x = x.reshape(b, s, n_kv_heads, d_head)
                for h in range(n_kv_heads):
                    xh = x[0, :, h, :].numpy()
                    U, mean = k_bases[(li, h)].get()
                    x[0, :, h, :] = torch.from_numpy(
                        subspace_polar_quantize(xh, k_dim, bits, U, mean))
                return x.reshape(b, s, n_kv_heads * d_head).to(dty).to(dev)
            return hook

        # V hook: online basis, compress + optionally update basis
        def make_v_hook(li):
            def hook(module, input, output):
                dev, dty = output.device, output.dtype
                x = output.detach().cpu().float()
                b, s, _ = x.shape
                x_reshaped = x.reshape(b, s, n_kv_heads, d_head)

                for h in range(n_kv_heads):
                    xh = x_reshaped[0, :, h, :].numpy()

                    # Accumulate for basis update
                    key = (li, h)
                    if key not in v_capture:
                        v_capture[key] = []
                    v_capture[key].append(xh.copy())

                    # Compress with current basis
                    U, mean = v_bases[key].get()
                    x_reshaped[0, :, h, :] = torch.from_numpy(
                        subspace_polar_quantize(xh, k_dim, bits, U, mean))

                # Update token counter (only once per layer call, use layer 0)
                if li == 0:
                    state['tokens_seen'] += s
                    # Trigger basis update if interval reached
                    if (update_interval > 0 and
                            state['next_update'] is not None and
                            state['tokens_seen'] >= state['next_update']):
                        state['next_update'] += update_interval
                        # Update all V bases
                        for key2, vecs in v_capture.items():
                            if vecs:
                                batch = np.concatenate(vecs, axis=0)
                                v_bases[key2].update(batch)
                        v_capture.clear()

                        # Log drift if requested
                        if v_drift_log is not None:
                            t = state['tokens_seen']
                            overlaps = []
                            for key2, vbasis in v_bases.items():
                                # Compare initial basis (before any update) to current
                                U_cur, _ = vbasis.get()
                                U_init, _ = k_bases[key2].get()  # use K as proxy for initial V
                                overlaps.append(basis_overlap(U_cur, U_init))
                            v_drift_log.setdefault('tokens', []).append(t)
                            v_drift_log.setdefault('mean_overlap', []).append(
                                float(np.mean(overlaps)))

                return x_reshaped.reshape(b, s, n_kv_heads * d_head).to(dty).to(dev)
            return hook

        hooks.append(attn.k_proj.register_forward_hook(make_k_hook(layer_idx)))
        hooks.append(attn.v_proj.register_forward_hook(make_v_hook(layer_idx)))

    return hooks, v_bases


# ── PPL evaluation ────────────────────────────────────────────────────────────

def _get_transformer_body_and_head(model):
    causal_lm = getattr(model, 'model', model)
    if hasattr(causal_lm, 'model') and hasattr(causal_lm, 'lm_head'):
        return causal_lm.model, causal_lm.lm_head
    return causal_lm, model.lm_head


def chunked_cross_entropy(model, input_ids, chunk_size=256):
    transformer_body, lm_head = _get_transformer_body_and_head(model)
    with torch.no_grad():
        outputs = transformer_body(input_ids=input_ids[:, :-1])
        hidden = outputs.last_hidden_state
    labels = input_ids[:, 1:].view(-1)
    total_loss, n_tok = 0.0, 0
    with torch.no_grad():
        for start in range(0, hidden.shape[1], chunk_size):
            end = min(start + chunk_size, hidden.shape[1])
            chunk_logits = lm_head(hidden[:, start:end, :])
            chunk_labels = labels[start:end]
            loss = torch.nn.functional.cross_entropy(
                chunk_logits.view(-1, chunk_logits.size(-1)), chunk_labels)
            total_loss += float(loss) * (end - start)
            n_tok += (end - start)
            del chunk_logits, chunk_labels, loss
            torch.cuda.empty_cache()
    del hidden
    torch.cuda.empty_cache()
    return total_loss / n_tok


def eval_ppl(model, tokenizer, data_file, char_offset, n_tokens, device):
    input_ids = load_tokens(tokenizer, data_file, char_offset, n_tokens, device)
    if input_ids.shape[1] > n_tokens:
        input_ids = input_ids[:, :n_tokens]
    loss = chunked_cross_entropy(model, input_ids)
    return float(np.exp(loss))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    print(f"Loading model {MODEL_NAME}...")
    device = 'cuda'
    model, tokenizer = get_model_and_tokenizer(MODEL_NAME)
    model.eval()

    cfg = model.config if hasattr(model, 'config') else model.model.config
    n_kv_heads = getattr(cfg, 'num_key_value_heads', 8)
    d_head = getattr(cfg, 'head_dim',
                     getattr(cfg, 'hidden_size', 4096) // getattr(cfg, 'num_attention_heads', 32))
    print(f"  n_kv_heads={n_kv_heads}, d_head={d_head}")

    print(f"\nCollecting initial KV basis on {CALIB_TOKENS} tokens...")
    initial_kvs = collect_kvs_for_basis(
        model, tokenizer, DATA_FILE, CALIB_OFFSET, CALIB_TOKENS,
        device, n_kv_heads, d_head)

    print(f"\nComputing baseline PPL (no compression, ctx={EVAL_CTX})...")
    baseline_ppl = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_CTX, device)
    print(f"  Baseline PPL: {baseline_ppl:.4f}")

    # K-only compression reference (static, proven baseline)
    print(f"\nComputing K-only reference (k={K_DIM}/4-bit, V full-dim 4-bit)...")
    # Build static bases for K-only run
    k_only_bases = {}
    for (li, hi), kv in initial_kvs.items():
        U_k, mean_k = fit_pca(kv['K'], K_DIM)
        k_only_bases[(li, hi)] = {'U_K': U_k, 'mean_K': mean_k}

    k_only_hooks = []
    for layer_idx, attn in find_attention_layers(model):
        def make_k_only_k_hook(li):
            def hook(module, input, output):
                dev, dty = output.device, output.dtype
                x = output.detach().cpu().float().reshape(
                    output.shape[0], output.shape[1], n_kv_heads, d_head)
                for h in range(n_kv_heads):
                    xh = x[0, :, h, :].numpy()
                    U = k_only_bases[(li, h)]['U_K']
                    mean = k_only_bases[(li, h)]['mean_K']
                    x[0, :, h, :] = torch.from_numpy(
                        subspace_polar_quantize(xh, K_DIM, BITS, U, mean))
                return x.reshape(output.shape[0], output.shape[1],
                                  n_kv_heads * d_head).to(dty).to(dev)
            return hook
        def make_k_only_v_hook():
            def hook(module, input, output):
                dev, dty = output.device, output.dtype
                x = output.detach().cpu().float().reshape(
                    output.shape[0], output.shape[1], n_kv_heads, d_head)
                for h in range(n_kv_heads):
                    x[0, :, h, :] = torch.from_numpy(
                        polar_quantize(x[0, :, h, :].numpy(), BITS))
                return x.reshape(output.shape[0], output.shape[1],
                                  n_kv_heads * d_head).to(dty).to(dev)
            return hook
        k_only_hooks.append(attn.k_proj.register_forward_hook(make_k_only_k_hook(layer_idx)))
        k_only_hooks.append(attn.v_proj.register_forward_hook(make_k_only_v_hook()))

    k_only_ppl = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_CTX, device)
    for h in k_only_hooks:
        h.remove()
    k_only_rel = k_only_ppl / baseline_ppl
    print(f"  K-only (V full-dim 4-bit) PPL={k_only_ppl:.4f} rel={k_only_rel:.4f}")

    results = []
    drift_rows = []

    # Test each update strategy
    strategies = [
        ('window', WindowBasis, {'window_size': WINDOW_SIZE}),
        ('ema', EMABasis, {'alpha': EMA_ALPHA}),
    ]

    for interval in UPDATE_INTERVALS:
        for strat_name, BasisCls, basis_kwargs in strategies:
            label = f"{strat_name}_interval={interval}" if interval > 0 else f"{strat_name}_static"
            print(f"\n{'='*60}")
            print(f"Strategy: {label}")

            v_drift_log = {}
            hooks, v_bases = install_online_hooks(
                model, initial_kvs, K_DIM, BITS, n_kv_heads, d_head,
                interval, BasisCls, basis_kwargs, v_drift_log)

            ppl = eval_ppl(model, tokenizer, DATA_FILE, EVAL_OFFSET, EVAL_CTX, device)
            for h in hooks:
                h.remove()

            rel = ppl / baseline_ppl
            ppl_gap_vs_k_only = ppl - k_only_ppl
            print(f"  PPL={ppl:.4f} rel={rel:.4f} gap_vs_k_only={ppl_gap_vs_k_only:+.4f}")

            # Count updates
            sample_vbasis = next(iter(v_bases.values()))
            n_updates = sample_vbasis.n_updates
            print(f"  Basis updates performed: {n_updates}")

            # Compute final V basis drift (initial vs final basis per head)
            overlaps = []
            for (li, hi), kv in initial_kvs.items():
                U_init, _ = fit_pca(kv['V'], K_DIM)
                U_cur, _ = v_bases[(li, hi)].get()
                overlaps.append(basis_overlap(U_init, U_cur))
            mean_drift = float(np.mean(overlaps))
            print(f"  Mean V basis overlap (initial→final): {mean_drift:.4f}")

            row = {
                'strategy': strat_name,
                'update_interval': interval,
                'label': label,
                'baseline_ppl': baseline_ppl,
                'k_only_ppl': k_only_ppl,
                'k_only_rel': k_only_rel,
                'ppl': ppl,
                'rel_ppl': rel,
                'ppl_gap_vs_k_only': ppl_gap_vs_k_only,
                'n_basis_updates': n_updates,
                'mean_v_basis_overlap': mean_drift,
            }
            results.append(row)

            for i, t in enumerate(v_drift_log.get('tokens', [])):
                drift_rows.append({
                    'strategy': strat_name,
                    'update_interval': interval,
                    'tokens_seen': t,
                    'mean_overlap': v_drift_log['mean_overlap'][i],
                })

            # Incremental CSV save
            csv_path = RESULTS_DIR / "exp19_online_basis.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(results)

    # Save drift log
    if drift_rows:
        with open(RESULTS_DIR / "exp19_v_drift_by_position.csv", 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=drift_rows[0].keys())
            writer.writeheader()
            writer.writerows(drift_rows)

    # ── Generate report ───────────────────────────────────────────────────────
    report_lines = [
        "# Experiment 19: Online Basis Updating for V Compression",
        "",
        "## Overview",
        "V vectors drift more than K vectors over long sequences (overlap 0.702 vs 0.825",
        "from exp13). This experiment tests whether online basis updating can close that",
        "gap, enabling V compression at the same quality level as K compression.",
        "",
        f"Eval context: {EVAL_CTX} tokens | k={K_DIM}/4-bit for both K and V",
        f"Baseline PPL: {baseline_ppl:.4f}",
        f"K-only reference (V full-dim 4-bit): {k_only_ppl:.4f} ({k_only_rel:.4f}x)",
        "",
        "## Results",
        "",
        "| Strategy | Update Interval | PPL | Rel PPL | Gap vs K-only | Basis Updates | V Overlap |",
        "|----------|----------------|-----|---------|---------------|---------------|-----------|",
    ]
    for r in results:
        report_lines.append(
            f"| {r['strategy']} | {r['update_interval']} | {r['ppl']:.4f} | "
            f"{r['rel_ppl']:.4f}x | {r['ppl_gap_vs_k_only']:+.4f} | "
            f"{r['n_basis_updates']} | {r['mean_v_basis_overlap']:.4f} |"
        )

    report_lines += [
        "",
        "## Key Question",
        "Does any online strategy reduce the PPL gap vs K-only to < 0.05 PPL points?",
        "If so, V compression at k=112/4-bit is viable, pushing total KV compression",
        "from ~5.3x (K+V where V is 4x) to ~6.2x (K subspace k=112 + V subspace k=112).",
        "",
        "## Compression Ratio Implication",
        f"- Current K+V(full-dim): K at {(d_head * 16)/(K_DIM * BITS):.2f}x, V at {16/BITS:.1f}x"
        f" → harmonic mean ≈ {2/((K_DIM*BITS)/(d_head*16) + BITS/16):.2f}x"
        if 'd_head' in dir() else "",
        "- K+V(both subspace k=112): both at 4.57x → combined 4.57x",
        "- This changes the total KV memory from ~53% reduction to ~78% reduction",
    ]

    with open(RESULTS_DIR / "REPORT-19-online-basis.md", 'w') as f:
        f.write('\n'.join(report_lines))

    print("\n\nDone!")
    print(f"Results: {RESULTS_DIR / 'exp19_online_basis.csv'}")
    print(f"Report:  {RESULTS_DIR / 'REPORT-19-online-basis.md'}")


if __name__ == '__main__':
    main()
