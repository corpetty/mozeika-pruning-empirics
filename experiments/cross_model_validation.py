"""
Experiment 11: Cross-Model Validation — k/d_head generalization.

Tests whether the k/d_head >= 0.875 rule found on Qwen3-14B-AWQ (d_head=128)
generalizes to:
  - Qwen3-1.7B  (28 layers, 8 KV heads, d_head=128) — same d_head, different scale
  - Qwen3-32B-AWQ (64 layers, 8 KV heads, d_head=128) — different scale

For each model, sweeps k/d_head fractions: 0.50, 0.75, 0.875, 0.9375, 1.0
at 4-bit quantization (V always full-dim 4-bit).

Usage:
    /home/petty/torch-env/bin/python3 experiments/cross_model_validation.py
"""

import sys
import os
import gc
import csv
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import polar_quantize, subspace_polar_quantize, fit_pca, quantize_uniform
from collect import find_attention_layers, get_sample_text


# ── Models ─────────────────────────────────────────────────────────────────

MODELS = [
    {
        'name': 'Qwen3-1.7B',
        'path': '/home/petty/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e',
        'is_awq': False,
        'n_layers': 28,
        'n_kv_heads': 8,
        'd_head': 128,
    },
    {
        'name': 'Qwen3-32B-AWQ',
        'path': '/home/petty/.cache/huggingface/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/0499c3ac83fdef8810b907a23894ba91e95eddd8',
        'is_awq': True,
        'n_layers': 64,
        'n_kv_heads': 8,
        'd_head': 128,  # head_dim is explicitly 128, NOT hidden_size//num_attention_heads=80
    },
]

# k/d_head fractions to test
K_FRACS = [0.50, 0.75, 0.875, 0.9375, 1.0]

N_BITS = 4  # 4-bit only


# ── Eval passages (same 3 as all previous experiments) ─────────────────────

EVAL_PASSAGES = [
    # 0: Scientific / biology
    (
        "The mitochondria are membrane-bound organelles found in the cytoplasm of "
        "eukaryotic cells. They generate most of the cell's supply of adenosine "
        "triphosphate, used as a source of chemical energy. Mitochondria have their "
        "own DNA, known as mitochondrial DNA, which is separate from the nuclear DNA "
        "found in the cell nucleus. This organelle has its own ribosomes and can "
        "synthesize some of its own proteins. The number of mitochondria in a cell "
        "varies widely by organism and tissue type. Many cells have only a single "
        "mitochondrion, whereas others can contain several thousand mitochondria. "
        "The organelle is composed of compartments that carry out specialized "
        "functions. These compartments or regions include the outer membrane, the "
        "intermembrane space, the inner membrane, the cristae, and the matrix. "
        "Although most of a cell's DNA is contained in the cell nucleus, the "
        "mitochondrion has its own genome that is substantially different from the "
        "nuclear genome. The human mitochondrial genome contains 16,569 base pairs "
        "and encodes 37 genes: 13 for subunits of respiratory complexes I, III, IV, "
        "and V, 22 for mitochondrial tRNA, and 2 for rRNA. The mitochondrion is "
        "thought to have originated from an ancient endosymbiotic event in which an "
        "ancestral eukaryotic cell engulfed an aerobic bacterium. Over evolutionary "
        "time, the engulfed bacterium transferred many of its genes to the host "
        "cell's nuclear genome. This endosymbiotic theory is supported by several "
        "lines of evidence, including the double membrane structure of mitochondria, "
        "their own circular DNA, and the similarity of their ribosomes to bacterial "
        "ribosomes. Mitochondria play a central role in cellular respiration, the "
        "metabolic process by which cells convert nutrients into energy. The process "
        "begins with glycolysis in the cytoplasm, which breaks down glucose into "
        "pyruvate. Pyruvate then enters the mitochondrion, where it is converted to "
        "acetyl-CoA by the pyruvate dehydrogenase complex. Acetyl-CoA enters the "
        "citric acid cycle, also known as the Krebs cycle, which takes place in the "
        "mitochondrial matrix. The citric acid cycle generates NADH and FADH2, which "
        "donate electrons to the electron transport chain located in the inner "
        "mitochondrial membrane. The electron transport chain consists of a series of "
        "protein complexes that transfer electrons from NADH and FADH2 to molecular "
        "oxygen, generating a proton gradient across the inner membrane. This proton "
        "gradient drives ATP synthase, which produces ATP from ADP and inorganic "
        "phosphate. The entire process of oxidative phosphorylation can produce "
        "approximately 30 to 32 ATP molecules per glucose molecule, making it far "
        "more efficient than glycolysis alone. Beyond energy production, mitochondria "
        "are involved in numerous other cellular processes, including regulation of "
        "the cell cycle, cell growth, and cell death through apoptosis."
    ),

    # 1: Historical narrative
    (
        "The construction of the Panama Canal stands as one of the most ambitious "
        "engineering projects in human history. The idea of creating a waterway "
        "across the narrow isthmus connecting North and South America had been "
        "discussed since the early sixteenth century, when Spanish explorers first "
        "recognized the potential for such a route. The first serious attempt to "
        "build the canal was made by the French, led by Ferdinand de Lesseps, who "
        "had successfully overseen the construction of the Suez Canal in Egypt. In "
        "1881, the French began excavation work on a sea-level canal through Panama, "
        "which was then a province of Colombia. The project was plagued from the "
        "start by inadequate planning, tropical diseases, and the challenging terrain "
        "of the Panamanian jungle. Malaria and yellow fever claimed the lives of "
        "thousands of workers, with estimates suggesting that between 20,000 and "
        "22,000 workers died during the French construction period. Financial "
        "mismanagement and engineering difficulties led to the collapse of the French "
        "canal company in 1889, resulting in one of the largest financial scandals "
        "of the nineteenth century. The United States took over the canal project in "
        "1904, following Panama's independence from Colombia, which was supported by "
        "the United States government. Under the leadership of chief engineer John "
        "Frank Stevens and later George Washington Goethals, the Americans adopted a "
        "radically different approach. Instead of a sea-level canal, they designed a "
        "lock-based system that would raise ships 85 feet above sea level through a "
        "series of locks to an artificial lake created by damming the Chagres River. "
        "The American effort also prioritized disease prevention, with Colonel "
        "William Crawford Gorgas implementing extensive sanitation measures that "
        "dramatically reduced the incidence of malaria and yellow fever. The "
        "construction of the Gatun Dam, which created Gatun Lake, was a massive "
        "undertaking in itself. At the time of its completion, it was the largest dam "
        "and Gatun Lake was the largest artificial body of water in the world. The "
        "Culebra Cut, later renamed the Gaillard Cut, required the excavation of "
        "nearly 100 million cubic yards of earth and rock through the Continental "
        "Divide. The canal opened to traffic on August 15, 1914, just as World War I "
        "was beginning in Europe. The Panama Canal reduced the sailing distance "
        "between New York and San Francisco by approximately 8,000 miles, "
        "transforming global shipping patterns and trade routes."
    ),

    # 2: Philosophical / epistemology
    (
        "In the realm of epistemology, the question of how we acquire knowledge has "
        "been debated by philosophers for millennia. The rationalist tradition, "
        "championed by thinkers such as Descartes, Leibniz, and Spinoza, holds that "
        "certain fundamental truths can be known through reason alone, independent of "
        "sensory experience. Descartes famously employed his method of radical doubt, "
        "systematically questioning all beliefs that could possibly be false, until "
        "he arrived at the one thing he could not doubt: his own existence as a "
        "thinking being. This led to his celebrated declaration, cogito ergo sum, I "
        "think therefore I am. From this foundation, Descartes attempted to rebuild "
        "knowledge on a purely rational basis, arguing that clear and distinct ideas "
        "perceived by the intellect must be true, guaranteed by the existence of a "
        "non-deceptive God. In contrast, the empiricist tradition, developed by "
        "philosophers such as Locke, Berkeley, and Hume, maintains that all knowledge "
        "ultimately derives from sensory experience. John Locke argued that the mind "
        "at birth is a tabula rasa, a blank slate, upon which experience writes. He "
        "distinguished between primary qualities, such as shape and size, which exist "
        "in objects themselves, and secondary qualities, such as color and taste, "
        "which are produced by the interaction between objects and our senses. David "
        "Hume pushed empiricism to its logical extreme, arguing that even our belief "
        "in causation is not rationally justified but is merely a habit of mind "
        "formed by the repeated observation of one event following another. Hume's "
        "skepticism posed a fundamental challenge to both science and philosophy, "
        "questioning whether we can ever truly know that the future will resemble the "
        "past. Immanuel Kant attempted to reconcile rationalism and empiricism in his "
        "Critique of Pure Reason, published in 1781. Kant argued that while all "
        "knowledge begins with experience, it does not all arise from experience. He "
        "proposed that the mind actively structures experience through innate "
        "categories of understanding, such as causality, space, and time. These "
        "categories are not derived from experience but are the very conditions that "
        "make experience possible. Kant called this his Copernican revolution in "
        "philosophy: rather than our knowledge conforming to objects, objects conform "
        "to our ways of knowing them. This transcendental idealism, as Kant termed "
        "it, suggests that we can never know things as they are in themselves, only "
        "as they appear to us through the lens of our cognitive faculties."
    ),
]


# ── Model loading ──────────────────────────────────────────────────────────

def load_model(model_info):
    """Load a model by its info dict. Returns (model, tokenizer)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = model_info['path']
    name = model_info['name']

    tokenizer = AutoTokenizer.from_pretrained(path)

    if model_info['is_awq']:
        from awq import AutoAWQForCausalLM
        # Constrain per-GPU memory to leave room for other processes (ollama)
        max_memory = {i: "9GiB" for i in range(torch.cuda.device_count())}
        max_memory["cpu"] = "24GiB"
        model = AutoAWQForCausalLM.from_quantized(
            path, fuse_layers=False, device_map="auto",
            max_memory=max_memory,
        )
        print(f"Loaded AWQ model: {name}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print(f"Loaded model: {name}")

    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory."""
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("Model unloaded, CUDA cache cleared")


# ── Calibration KV collection (in-memory) ─────────────────────────────────

def collect_calibration_kvs(model, tokenizer, n_tokens=2048, device='cuda'):
    """
    Collect K/V vectors from a calibration forward pass.
    Returns dict: {layer_idx: {'K': np.ndarray(T, n_heads, d_head),
                                'V': np.ndarray(T, n_heads, d_head)}}
    """
    attention_layers = find_attention_layers(model)
    print(f"  Found {len(attention_layers)} attention layers")

    model_config = getattr(model, 'config', None)
    config_n_kv_heads = getattr(model_config, 'num_key_value_heads', None)
    config_d_head = getattr(model_config, 'head_dim', None)
    if config_d_head is None and model_config is not None:
        n_heads = getattr(model_config, 'num_attention_heads', None)
        hidden = getattr(model_config, 'hidden_size', None)
        if n_heads and hidden:
            config_d_head = hidden // n_heads

    kv_raw = {}
    hooks = []

    def make_kv_hook(layer_idx, which):
        def hook(module, input, output):
            if layer_idx not in kv_raw:
                kv_raw[layer_idx] = {'K': None, 'V': None}
            kv_raw[layer_idx][which] = output.detach().cpu().float()
        return hook

    for layer_idx, attn in attention_layers:
        if hasattr(attn, 'k_proj'):
            h = attn.k_proj.register_forward_hook(make_kv_hook(layer_idx, 'K'))
            hooks.append(h)
        if hasattr(attn, 'v_proj'):
            h = attn.v_proj.register_forward_hook(make_kv_hook(layer_idx, 'V'))
            hooks.append(h)

    text = get_sample_text(n_chars=n_tokens * 8)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    actual_tokens = inputs['input_ids'].shape[1]
    print(f"  Calibration forward pass: {actual_tokens} tokens")

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    result = {}
    for layer_idx, attn in attention_layers:
        if layer_idx not in kv_raw:
            continue
        raw = kv_raw[layer_idx]
        if raw['K'] is None or raw['V'] is None:
            continue

        n_kv_heads = config_n_kv_heads
        d_head = config_d_head

        K = raw['K'][0]  # (seq, n_kv_heads * d_head)
        V = raw['V'][0]
        T = K.shape[0]
        K = K.reshape(T, n_kv_heads, d_head).numpy()
        V = V.reshape(T, n_kv_heads, d_head).numpy()
        result[layer_idx] = {'K': K, 'V': V}

    print(f"  Collected KVs from {len(result)} layers")
    return result


# ── PCA bases (in-memory) ─────────────────────────────────────────────────

def compute_pca_bases(kvs, max_k):
    """Compute PCA bases per (layer, head) from in-memory calibration KVs."""
    bases = {}
    for layer_idx in sorted(kvs.keys()):
        K = kvs[layer_idx]['K']  # (T, n_heads, d_head)
        V = kvs[layer_idx]['V']
        n_heads = K.shape[1]
        for h in range(n_heads):
            U_k, mean_k = fit_pca(K[:, h, :], max_k)
            U_v, mean_v = fit_pca(V[:, h, :], max_k)
            bases[(layer_idx, h)] = {
                'U_K': U_k,
                'mean_K': mean_k,
                'U_V': U_v,
                'mean_V': mean_v,
            }
    return bases


def get_basis_for_k(bases, layer_idx, head_idx, kv_type, k):
    """Slice stored basis to get top-k components."""
    base = bases.get((layer_idx, head_idx), {})
    U = base.get(f'U_{kv_type}')
    mean = base.get(f'mean_{kv_type}')
    if U is not None and k < U.shape[1]:
        U = U[:, :k]
    return U, mean


# ── Compression hooks ─────────────────────────────────────────────────────

def compress_head(x_np, k, n_bits, U_k, mean, d_head):
    """Compress-decompress roundtrip for a single head's (T, d) vectors."""
    if k == d_head:
        return polar_quantize(x_np, n_bits)
    else:
        return subspace_polar_quantize(x_np, k, n_bits, U_k, mean)


def install_compression_hooks(model, k_K, n_bits, bases, n_kv_heads, d_head):
    """Install hooks on k_proj/v_proj that apply compress-decompress roundtrip."""
    hooks = []
    attn_layers = find_attention_layers(model)

    for layer_idx, attn in attn_layers:
        for kv_type, proj_name, k_dim in [
            ('K', 'k_proj', k_K),
            ('V', 'v_proj', d_head),  # V always full-dim
        ]:
            def make_hook(li, kvt, kk):
                def hook(module, input, output):
                    device, dtype = output.device, output.dtype
                    x = output.detach().cpu().float()
                    batch, seq, _ = x.shape
                    x = x.reshape(batch, seq, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = x[0, :, h, :].numpy()
                        U, mn = get_basis_for_k(bases, li, h, kvt, kk)
                        xh_comp = compress_head(xh, kk, n_bits, U, mn, d_head)
                        x[0, :, h, :] = torch.from_numpy(xh_comp)
                    return x.reshape(batch, seq, -1).to(device=device, dtype=dtype)
                return hook

            proj = getattr(attn, proj_name)
            h = proj.register_forward_hook(make_hook(layer_idx, kv_type, k_dim))
            hooks.append(h)

    return hooks


# ── Compression ratio ─────────────────────────────────────────────────────

def compute_compression_ratio(k_K, n_bits, d_head):
    """
    CR = FP16_total / compressed_total
    FP16: (d_head_K + d_head_V) × 16 = 2 × d_head × 16
    Compressed: k_K × 4 + d_head × 4  (K subspace + V full-dim, both 4-bit)
    """
    fp16_bits = 2 * d_head * 16
    k_bits = k_K * n_bits
    v_bits = d_head * n_bits
    return fp16_bits / (k_bits + v_bits)


# ── Perplexity computation ────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, text, max_tokens=512, device='cuda'):
    """Compute perplexity of text under model (with any active hooks)."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens)
    input_ids = inputs['input_ids'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return float(torch.exp(loss)), input_ids.shape[1]


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    device = 'cuda'
    max_tokens = 512

    print("=" * 70)
    print("Experiment 11: Cross-Model Validation — k/d_head Generalization")
    print("=" * 70)

    all_rows = []

    for model_info in MODELS:
        model_name = model_info['name']
        d_head = model_info['d_head']
        n_kv_heads = model_info['n_kv_heads']

        # k values for this model
        k_values = [int(round(f * d_head)) for f in K_FRACS]
        max_k = max(k for k in k_values if k < d_head)  # max subspace k (exclude full-dim)

        print(f"\n{'=' * 70}")
        print(f"Model: {model_name}")
        print(f"  d_head={d_head}, n_kv_heads={n_kv_heads}, n_layers={model_info['n_layers']}")
        print(f"  k values: {k_values} (fracs: {K_FRACS})")
        print(f"  max_k for PCA: {max_k}")
        print("=" * 70)

        # Load model
        print(f"\nLoading {model_name}...")
        model, tokenizer = load_model(model_info)

        # Collect calibration KVs in-memory
        # Use fewer tokens for large models to avoid OOM during forward pass
        cal_tokens = 512 if model_info['n_layers'] > 40 else 2048
        print(f"\nCollecting calibration KVs ({cal_tokens} tokens)...")
        cal_kvs = collect_calibration_kvs(model, tokenizer, n_tokens=cal_tokens, device=device)

        # Compute PCA bases
        print(f"\nComputing PCA bases (max_k={max_k})...")
        bases = compute_pca_bases(cal_kvs, max_k)
        print(f"  Computed bases for {len(bases)} (layer, head) pairs")
        del cal_kvs  # free calibration data

        # Configs: baseline + each k value
        configs = [('baseline', None, None)]
        for k_frac, k in zip(K_FRACS, k_values):
            configs.append((f"k{k}_{N_BITS}bit", k, k_frac))

        # Evaluate each config × passage
        for cfg_name, k_K, k_frac in configs:
            print(f"\n--- {model_name} / {cfg_name} ---")

            if k_K is not None:
                cr = compute_compression_ratio(k_K, N_BITS, d_head)
                print(f"  k={k_K}, k/d_head={k_frac:.4f}, CR={cr:.2f}x")

            for pidx, passage in enumerate(EVAL_PASSAGES):
                if k_K is None:
                    hooks = []
                else:
                    hooks = install_compression_hooks(
                        model, k_K, N_BITS, bases, n_kv_heads, d_head
                    )

                ppl, n_tok = compute_perplexity(model, tokenizer, passage, max_tokens, device)

                for h in hooks:
                    h.remove()

                print(f"  Passage {pidx}: PPL = {ppl:.2f}  ({n_tok} tokens)")
                all_rows.append({
                    'model': model_name,
                    'k': k_K,
                    'k_frac': k_frac,
                    'n_bits': N_BITS if k_K is not None else None,
                    'passage_idx': pidx,
                    'ppl': ppl,
                    'd_head': d_head,
                })

        # Unload model before loading next
        del bases
        unload_model(model, tokenizer)

    # ── Compute derived metrics ────────────────────────────────────────────

    # Baseline PPL per model per passage
    baselines = {}
    for r in all_rows:
        if r['k'] is None:
            baselines[(r['model'], r['passage_idx'])] = r['ppl']

    baseline_means = {}
    for model_info in MODELS:
        mn = model_info['name']
        ppls = [baselines[(mn, i)] for i in range(len(EVAL_PASSAGES))]
        baseline_means[mn] = np.mean(ppls)

    for r in all_rows:
        mn = r['model']
        # Mean PPL across passages for this config
        same_cfg = [rr['ppl'] for rr in all_rows
                    if rr['model'] == mn and rr['k'] == r['k']]
        r['mean_ppl'] = np.mean(same_cfg)
        # Relative PPL vs baseline for this passage
        r['rel_ppl'] = r['ppl'] / baselines[(mn, r['passage_idx'])]
        # Compression ratio
        if r['k'] is not None:
            r['compression_ratio'] = compute_compression_ratio(r['k'], N_BITS, r['d_head'])
        else:
            r['compression_ratio'] = 1.0

    # ── Save CSV ───────────────────────────────────────────────────────────

    Path('results').mkdir(exist_ok=True)
    fieldnames = ['model', 'k', 'k_frac', 'n_bits', 'passage_idx',
                  'ppl', 'mean_ppl', 'rel_ppl', 'compression_ratio']
    with open('results/cross_model_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved results/cross_model_results.csv")

    # ── Write report ───────────────────────────────────────────────────────

    write_report(all_rows, baseline_means)
    print("Wrote results/REPORT-11-cross-model.md")

    # ── Summary ────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_info in MODELS:
        mn = model_info['name']
        d_head = model_info['d_head']
        bm = baseline_means[mn]
        print(f"\n{mn} (d_head={d_head}, baseline mean PPL={bm:.2f}):")
        print(f"  {'Config':<15} {'k':>4} {'k/d':>6} {'Mean PPL':>10} {'Rel PPL':>10} {'CR':>8}")
        print(f"  {'-' * 57}")

        seen = set()
        for r in all_rows:
            if r['model'] != mn:
                continue
            cfg_key = (r['model'], r['k'])
            if cfg_key in seen:
                continue
            seen.add(cfg_key)

            if r['k'] is None:
                print(f"  {'baseline':<15} {'—':>4} {'—':>6} {r['mean_ppl']:>10.2f} {'1.00x':>10} {'1.00x':>8}")
            else:
                cr = compute_compression_ratio(r['k'], N_BITS, d_head)
                rel = r['mean_ppl'] / bm
                label = f"k{r['k']}_{N_BITS}bit"
                print(f"  {label:<15} {r['k']:>4} {r['k_frac']:>6.4f} {r['mean_ppl']:>10.2f} {rel:>9.2f}x {cr:>7.2f}x")


# ── Report ────────────────────────────────────────────────────────────────

def write_report(all_rows, baseline_means):
    """Write results/REPORT-11-cross-model.md."""

    # Build per-model lookup: (model, k_frac) -> mean_ppl, rel_ppl_mean
    model_data = {}  # model_name -> {k_frac: {mean_ppl, rel_ppl_mean, k, cr}}
    for model_info in MODELS:
        mn = model_info['name']
        d_head = model_info['d_head']
        bm = baseline_means[mn]
        model_data[mn] = {}

        for r in all_rows:
            if r['model'] != mn or r['k'] is None:
                continue
            kf = r['k_frac']
            if kf not in model_data[mn]:
                same = [rr['ppl'] for rr in all_rows
                        if rr['model'] == mn and rr['k'] == r['k']]
                mppl = np.mean(same)
                cr = compute_compression_ratio(r['k'], N_BITS, d_head)
                model_data[mn][kf] = {
                    'mean_ppl': mppl,
                    'rel_ppl': mppl / bm,
                    'k': r['k'],
                    'cr': cr,
                }

    lines = [
        "# Experiment 11: Cross-Model Validation — k/d_head Generalization\n",
        "## Question\n",
        "Is the k/d_head >= 0.875 rule (found on Qwen3-14B-AWQ, d_head=128) a general",
        "principle or model-specific?\n",
        "## Models Tested\n",
        "| Model | Layers | KV Heads | d_head | AWQ |",
        "|-------|--------|----------|--------|-----|",
        "| Qwen3-1.7B | 28 | 8 | 128 | No |",
        "| Qwen3-32B-AWQ | 64 | 8 | 128 | Yes |",
        "| Qwen3-14B-AWQ (Exp 9) | 40 | 8 | 128 | Yes |\n",
        "## Setup\n",
        "- K compression: subspace PCA + PolarQuant at 4-bit",
        "- V compression: full-dim PolarQuant at 4-bit",
        "- k/d_head fractions tested: 0.50, 0.75, 0.875, 0.9375, 1.0",
        "- Calibration: Project Gutenberg text, in-memory PCA (2048 tokens for 1.7B; 512 for 32B)",
        "- Evaluation: 3 passages (scientific, historical, philosophical), 512 tokens each\n",
    ]

    # 1. Per-model PPL vs k/d_head table
    lines.append("## 1. PPL vs k/d_head (Side by Side)\n")

    # 14B reference data from Exp 9
    ref_14b = {
        0.50:   {'k': 64,  'rel_ppl': None},  # will fill from context
        0.75:   {'k': 96,  'rel_ppl': None},
        0.875:  {'k': 112, 'rel_ppl': 1.14},
        0.9375: {'k': 120, 'rel_ppl': None},
        1.0:    {'k': 128, 'rel_ppl': 1.05},
    }

    header = "| k/d_head |"
    sep = "|----------|"
    for model_info in MODELS:
        mn = model_info['name']
        bm = baseline_means[mn]
        header += f" {mn} (base={bm:.2f}) |"
        sep += "---------------------------|"
    header += " 14B-AWQ (Exp 9) |"
    sep += "-----------------|"
    lines.append(header)
    lines.append(sep)

    for kf in K_FRACS:
        row = f"| {kf:.4f} |"
        for model_info in MODELS:
            mn = model_info['name']
            d = model_data[mn].get(kf)
            if d:
                marker = " **" if d['rel_ppl'] <= 1.20 else ""
                row += f" k={d['k']}: {d['mean_ppl']:.2f} ({d['rel_ppl']:.2f}x, CR={d['cr']:.2f}x){marker} |"
            else:
                row += " — |"
        # 14B reference
        ref = ref_14b.get(kf)
        if ref and ref['rel_ppl'] is not None:
            row += f" k={ref['k']}: {ref['rel_ppl']:.2f}x |"
        else:
            row += " — |"
        lines.append(row)

    lines.append("\n**Bold** = within 20% PPL degradation threshold.\n")

    # 2. k/d_head >= 0.875 analysis
    lines.append("## 2. Does k/d_head >= 0.875 Hold?\n")

    for model_info in MODELS:
        mn = model_info['name']
        d_head = model_info['d_head']
        lines.append(f"### {mn} (d_head={d_head})\n")

        d_0875 = model_data[mn].get(0.875)
        if d_0875:
            within = "YES" if d_0875['rel_ppl'] <= 1.20 else "NO"
            lines.append(f"- At k/d_head=0.875 (k={d_0875['k']}): rel_ppl = {d_0875['rel_ppl']:.2f}x → within 20%? **{within}**")
        else:
            lines.append("- k/d_head=0.875: no data")

        # Find threshold: smallest k_frac where rel_ppl <= 1.20
        sorted_fracs = sorted(model_data[mn].items(), key=lambda x: x[0])
        threshold_frac = None
        for kf, d in sorted_fracs:
            if d['rel_ppl'] <= 1.20:
                threshold_frac = kf
                break
        if threshold_frac is not None:
            lines.append(f"- Smallest k/d_head within 20%: {threshold_frac:.4f} (k={model_data[mn][threshold_frac]['k']})")
        else:
            lines.append("- No k/d_head fraction achieves <= 20% PPL degradation at 4-bit")
        lines.append("")

    lines.append(f"### 14B-AWQ (reference from Exp 9)\n")
    lines.append(f"- At k/d_head=0.875 (k=112): rel_ppl = 1.14x → within 20%? **YES**")
    lines.append(f"- At k/d_head=1.0 (k=128): rel_ppl = 1.05x\n")

    # 3. Pareto frontier per model
    lines.append("## 3. PPL vs Compression Pareto\n")

    for model_info in MODELS:
        mn = model_info['name']
        bm = baseline_means[mn]
        lines.append(f"### {mn}\n")
        lines.append("| k/d_head | k | Mean PPL | Rel PPL | CR | Pareto? |")
        lines.append("|----------|---|----------|---------|-----|---------|")

        points = sorted(model_data[mn].items(), key=lambda x: -x[1]['cr'])
        pareto = []
        best_ppl = float('inf')
        for kf, d in points:
            if d['mean_ppl'] <= best_ppl:
                pareto.append(kf)
                best_ppl = d['mean_ppl']

        for kf, d in points:
            is_pareto = "YES" if kf in pareto else "no"
            lines.append(f"| {kf:.4f} | {d['k']} | {d['mean_ppl']:.2f} | {d['rel_ppl']:.2f}x | {d['cr']:.2f}x | {is_pareto} |")
        lines.append("")

    # 4. Comparison with 14B
    lines.append("## 4. Cross-Model Comparison at k/d_head=0.875\n")
    lines.append("| Model | d_head | k | Rel PPL | CR |")
    lines.append("|-------|--------|---|---------|-----|")
    lines.append("| Qwen3-14B-AWQ | 128 | 112 | 1.14x | 4.27x |")

    for model_info in MODELS:
        mn = model_info['name']
        d_head = model_info['d_head']
        d_0875 = model_data[mn].get(0.875)
        if d_0875:
            lines.append(f"| {mn} | {d_head} | {d_0875['k']} | {d_0875['rel_ppl']:.2f}x | {d_0875['cr']:.2f}x |")

    lines.append("")

    # Pattern analysis
    lines.append("### Pattern: Model Size vs Compression Tolerance\n")
    lines.append("At k/d_head=0.875, 4-bit:\n")
    rel_ppls = {}
    for model_info in MODELS:
        mn = model_info['name']
        d_0875 = model_data[mn].get(0.875)
        if d_0875:
            rel_ppls[mn] = d_0875['rel_ppl']
            lines.append(f"- {mn}: {d_0875['rel_ppl']:.2f}x")
    lines.append("- Qwen3-14B-AWQ: 1.14x\n")

    if len(rel_ppls) >= 2:
        models_sorted = ['Qwen3-1.7B', 'Qwen3-32B-AWQ']
        r17 = rel_ppls.get('Qwen3-1.7B')
        r32 = rel_ppls.get('Qwen3-32B-AWQ')
        if r17 is not None and r32 is not None:
            if r17 > 1.14 and r32 < 1.14:
                lines.append("Trend: larger models tolerate compression better (lower rel_ppl).\n")
            elif r17 < 1.14 and r32 > 1.14:
                lines.append("Trend: smaller models tolerate compression better (lower rel_ppl).\n")
            elif r17 <= 1.20 and r32 <= 1.20:
                lines.append("All three models stay within 20% at k/d_head=0.875 — rule generalizes.\n")
            else:
                lines.append("Mixed results — no clear size trend.\n")

    # 5. Conclusion
    lines.append("## 5. Conclusion\n")

    all_within = True
    for model_info in MODELS:
        mn = model_info['name']
        d_0875 = model_data[mn].get(0.875)
        if d_0875 is None or d_0875['rel_ppl'] > 1.20:
            all_within = False

    if all_within:
        lines.append("**k/d_head >= 0.875 is a general principle.** Both Qwen3-1.7B (d_head=128)")
        lines.append("and Qwen3-32B-AWQ (d_head=128) stay within 20% PPL degradation at")
        lines.append("k/d_head=0.875 with 4-bit quantization, consistent with Qwen3-14B-AWQ.")
        lines.append("")
        lines.append("This suggests the rule holds across:")
        lines.append("- Different model sizes (1.7B → 14B → 32B)")
        lines.append("- Both AWQ and non-AWQ models")
    else:
        lines.append("**k/d_head >= 0.875 does NOT universally hold.** Results vary by model:\n")
        for model_info in MODELS:
            mn = model_info['name']
            d_0875 = model_data[mn].get(0.875)
            if d_0875:
                status = "within 20%" if d_0875['rel_ppl'] <= 1.20 else f"exceeds 20% ({d_0875['rel_ppl']:.2f}x)"
                lines.append(f"- {mn}: {status}")
        lines.append("")
        lines.append("The threshold may need to be adjusted per model size — smaller models")
        lines.append("are more sensitive to subspace truncation and may need k/d_head closer to 1.0.")

    with open('results/REPORT-11-cross-model.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
