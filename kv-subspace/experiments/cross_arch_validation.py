"""
Experiment 12: Cross-Architecture Validation — Mistral-7B + Phi-4

Tests whether the k/d_head >= 0.875 rule generalizes across architectures:
  - Mistral-7B-v0.3: Mistral architecture, d_head=128, 32 layers, 8 KV heads
  - Phi-4-AWQ: Phi3 architecture, d_head=128, 40 layers, 10 KV heads

Usage:
    /home/petty/torch-env/bin/python3 experiments/cross_arch_validation.py
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
        'name': 'Mistral-7B-v0.3',
        'architecture': 'Mistral',
        'path': '/home/petty/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1',
        'is_awq': False,
        'trust_remote_code': False,
        'n_layers': 32,
        'n_kv_heads': 8,
        'd_head': 128,
        'params': '7B',
    },
    {
        'name': 'Phi-4-AWQ',
        'architecture': 'Phi3',
        'path': '/home/petty/.cache/huggingface/hub/models--stelterlab--phi-4-AWQ/snapshots/075b93fe5ab0d2e86004a5d68c7575ec3bb5a88b',
        'is_awq': True,
        'trust_remote_code': True,
        'n_layers': 40,
        'n_kv_heads': 10,
        'd_head': 128,
        'params': '14B',
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
    trust_remote = model_info.get('trust_remote_code', False)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote)

    max_memory = {i: "12GiB" for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = "24GiB"

    if model_info['is_awq']:
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(
            path, fuse_layers=False, device_map="auto",
            max_memory=max_memory,
            trust_remote_code=trust_remote,
        )
        print(f"Loaded AWQ model: {name}")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, device_map="auto",
            max_memory=max_memory,
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


# ── Extended attention layer discovery ─────────────────────────────────────

def find_attention_layers_extended(model):
    """
    Find attention layers, handling AWQ wrappers and various architectures.
    Returns list of (layer_idx, attn_module).
    """
    # Try standard function first
    layers = find_attention_layers(model)
    if layers:
        return layers

    # Try going deeper through nested .model wrappers (AWQ -> CausalLM -> Model)
    inner = model
    for _ in range(3):
        if hasattr(inner, 'model'):
            inner = inner.model
        if hasattr(inner, 'layers'):
            layers = []
            for i, layer in enumerate(inner.layers):
                if hasattr(layer, 'self_attn'):
                    layers.append((i, layer.self_attn))
            if layers:
                return layers

    # Last resort: scan all named_modules
    layers = []
    for name, module in model.named_modules():
        has_kv = hasattr(module, 'k_proj') and hasattr(module, 'v_proj')
        has_qkv = hasattr(module, 'qkv_proj')
        if has_kv or has_qkv:
            parts = name.split('.')
            idx = len(layers)
            for p in parts:
                if p.isdigit():
                    idx = int(p)
                    break
            layers.append((idx, module))

    return layers


def detect_proj_style(attn_module):
    """Detect whether attention uses separate k_proj/v_proj or fused qkv_proj."""
    if hasattr(attn_module, 'k_proj') and hasattr(attn_module, 'v_proj'):
        return 'separate'
    elif hasattr(attn_module, 'qkv_proj'):
        return 'fused_qkv'
    else:
        children = [n for n, _ in attn_module.named_children()]
        raise ValueError(f"Unknown projection style. Children: {children}")


# ── Calibration KV collection (in-memory) ─────────────────────────────────

def collect_calibration_kvs(model, tokenizer, model_info, n_tokens=2048, device='cuda'):
    """
    Collect K/V vectors from a calibration forward pass.
    Handles both separate k_proj/v_proj and fused qkv_proj architectures.
    Returns dict: {layer_idx: {'K': np.ndarray(T, n_heads, d_head),
                                'V': np.ndarray(T, n_heads, d_head)}}
    """
    attention_layers = find_attention_layers_extended(model)
    print(f"  Found {len(attention_layers)} attention layers")

    if not attention_layers:
        print("  ERROR: No attention layers found!")
        # Print model structure for debugging
        for name, module in model.named_modules():
            if 'attn' in name.lower():
                print(f"    {name}: {type(module).__name__}")
        return {}

    # Detect projection style
    _, first_attn = attention_layers[0]
    proj_style = detect_proj_style(first_attn)
    children = [n for n, _ in first_attn.named_children()]
    print(f"  Projection style: {proj_style}")
    print(f"  Attention module children: {children}")

    # Get config for splitting fused QKV
    model_config = getattr(model, 'config', None)
    num_q_heads = getattr(model_config, 'num_attention_heads', None)
    num_kv_heads = model_info['n_kv_heads']
    d_head = model_info['d_head']

    kv_raw = {}
    hooks = []

    if proj_style == 'separate':
        def make_kv_hook(layer_idx, which):
            def hook(module, input, output):
                if layer_idx not in kv_raw:
                    kv_raw[layer_idx] = {'K': None, 'V': None}
                kv_raw[layer_idx][which] = output.detach().cpu().float()
            return hook

        for layer_idx, attn in attention_layers:
            h = attn.k_proj.register_forward_hook(make_kv_hook(layer_idx, 'K'))
            hooks.append(h)
            h = attn.v_proj.register_forward_hook(make_kv_hook(layer_idx, 'V'))
            hooks.append(h)

    elif proj_style == 'fused_qkv':
        q_dim = num_q_heads * d_head
        k_dim = num_kv_heads * d_head
        # v_dim = num_kv_heads * d_head  (same as k_dim)

        def make_qkv_hook(layer_idx):
            def hook(module, input, output):
                if layer_idx not in kv_raw:
                    kv_raw[layer_idx] = {'K': None, 'V': None}
                full = output.detach().cpu().float()
                # QKV layout: [Q | K | V] along last dim
                kv_raw[layer_idx]['K'] = full[:, :, q_dim:q_dim + k_dim]
                kv_raw[layer_idx]['V'] = full[:, :, q_dim + k_dim:q_dim + 2 * k_dim]
            return hook

        for layer_idx, attn in attention_layers:
            h = attn.qkv_proj.register_forward_hook(make_qkv_hook(layer_idx))
            hooks.append(h)

    # Run calibration forward pass
    text = get_sample_text(n_chars=n_tokens * 8)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=n_tokens)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    actual_tokens = inputs['input_ids'].shape[1]
    print(f"  Calibration forward pass: {actual_tokens} tokens")

    with torch.no_grad():
        model(**inputs)

    for h in hooks:
        h.remove()

    # Reshape to (T, n_kv_heads, d_head)
    result = {}
    for layer_idx, attn in attention_layers:
        if layer_idx not in kv_raw:
            continue
        raw = kv_raw[layer_idx]
        if raw['K'] is None or raw['V'] is None:
            continue

        K = raw['K'][0]  # (seq, n_kv_heads * d_head)
        V = raw['V'][0]
        T = K.shape[0]

        # Validate dimensions
        expected_dim = num_kv_heads * d_head
        if K.shape[1] != expected_dim:
            print(f"  WARNING: Layer {layer_idx} K dim {K.shape[1]} != expected {expected_dim}")
            continue

        K = K.reshape(T, num_kv_heads, d_head).numpy()
        V = V.reshape(T, num_kv_heads, d_head).numpy()
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


def install_compression_hooks(model, k_K, n_bits, bases, model_info):
    """Install hooks on k_proj/v_proj (or qkv_proj) that apply compress-decompress roundtrip."""
    hooks = []
    attn_layers = find_attention_layers_extended(model)
    n_kv_heads = model_info['n_kv_heads']
    d_head = model_info['d_head']

    _, first_attn = attn_layers[0]
    proj_style = detect_proj_style(first_attn)

    if proj_style == 'separate':
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

    elif proj_style == 'fused_qkv':
        model_config = getattr(model, 'config', None)
        num_q_heads = getattr(model_config, 'num_attention_heads', None)
        q_dim = num_q_heads * d_head
        kv_dim = n_kv_heads * d_head

        for layer_idx, attn in attn_layers:
            def make_qkv_hook(li):
                def hook(module, input, output):
                    device, dtype = output.device, output.dtype
                    x = output.detach().cpu().float()
                    batch, seq, total_dim = x.shape

                    # Split into Q, K, V
                    q_part = x[:, :, :q_dim]
                    k_part = x[:, :, q_dim:q_dim + kv_dim]
                    v_part = x[:, :, q_dim + kv_dim:q_dim + 2 * kv_dim]

                    # Compress K
                    k_part = k_part.reshape(batch, seq, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = k_part[0, :, h, :].numpy()
                        U, mn = get_basis_for_k(bases, li, h, 'K', k_K)
                        xh_comp = compress_head(xh, k_K, n_bits, U, mn, d_head)
                        k_part[0, :, h, :] = torch.from_numpy(xh_comp)
                    k_part = k_part.reshape(batch, seq, kv_dim)

                    # Compress V (full-dim)
                    v_part = v_part.reshape(batch, seq, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = v_part[0, :, h, :].numpy()
                        U, mn = get_basis_for_k(bases, li, h, 'V', d_head)
                        xh_comp = compress_head(xh, d_head, n_bits, U, mn, d_head)
                        v_part[0, :, h, :] = torch.from_numpy(xh_comp)
                    v_part = v_part.reshape(batch, seq, kv_dim)

                    # Reassemble
                    out = torch.cat([q_part, k_part, v_part], dim=-1)
                    return out.to(device=device, dtype=dtype)
                return hook

            h = attn.qkv_proj.register_forward_hook(make_qkv_hook(layer_idx))
            hooks.append(h)

    return hooks


# ── Compression ratio ─────────────────────────────────────────────────────

def compute_compression_ratio(k_K, n_bits, d_head):
    """
    CR = FP16_total / compressed_total
    FP16: (d_head_K + d_head_V) x 16 = 2 x d_head x 16
    Compressed: k_K x 4 + d_head x 4  (K subspace + V full-dim, both 4-bit)
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
    print("Experiment 12: Cross-Architecture Validation — Mistral-7B + Phi-4")
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
        print(f"Model: {model_name} ({model_info['architecture']}, {model_info['params']})")
        print(f"  d_head={d_head}, n_kv_heads={n_kv_heads}, n_layers={model_info['n_layers']}")
        print(f"  k values: {k_values} (fracs: {K_FRACS})")
        print(f"  max_k for PCA: {max_k}")
        print("=" * 70)

        # Load model
        print(f"\nLoading {model_name}...")
        model, tokenizer = load_model(model_info)

        # Collect calibration KVs in-memory (512 tokens to fit GPU memory)
        cal_tokens = 512
        print(f"\nCollecting calibration KVs ({cal_tokens} tokens)...")
        cal_kvs = collect_calibration_kvs(model, tokenizer, model_info,
                                          n_tokens=cal_tokens, device=device)

        if not cal_kvs:
            print(f"  SKIPPING {model_name} — no KVs collected")
            unload_model(model, tokenizer)
            continue

        # Compute PCA bases
        print(f"\nComputing PCA bases (max_k={max_k})...")
        bases = compute_pca_bases(cal_kvs, max_k)
        print(f"  Computed bases for {len(bases)} (layer, head) pairs")
        del cal_kvs  # free calibration data

        # Configs: baseline + each k value
        configs = [('baseline', None, None)]
        for k_frac, k in zip(K_FRACS, k_values):
            configs.append((f"k{k}_{N_BITS}bit", k, k_frac))

        # Evaluate each config x passage
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
                        model, k_K, N_BITS, bases, model_info
                    )

                ppl, n_tok = compute_perplexity(model, tokenizer, passage, max_tokens, device)

                for h in hooks:
                    h.remove()

                print(f"  Passage {pidx}: PPL = {ppl:.2f}  ({n_tok} tokens)")
                all_rows.append({
                    'model': model_name,
                    'architecture': model_info['architecture'],
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
        ppls = [baselines.get((mn, i)) for i in range(len(EVAL_PASSAGES))]
        ppls = [p for p in ppls if p is not None]
        if ppls:
            baseline_means[mn] = np.mean(ppls)

    for r in all_rows:
        mn = r['model']
        if mn not in baseline_means:
            continue
        # Mean PPL across passages for this config
        same_cfg = [rr['ppl'] for rr in all_rows
                    if rr['model'] == mn and rr['k'] == r['k']]
        r['mean_ppl'] = np.mean(same_cfg)
        # Relative PPL vs baseline for this passage
        bl = baselines.get((mn, r['passage_idx']))
        r['rel_ppl'] = r['ppl'] / bl if bl else None
        # Compression ratio
        if r['k'] is not None:
            r['compression_ratio'] = compute_compression_ratio(r['k'], N_BITS, r['d_head'])
        else:
            r['compression_ratio'] = 1.0

    # ── Save CSV ───────────────────────────────────────────────────────────

    Path('results').mkdir(exist_ok=True)
    fieldnames = ['model', 'architecture', 'k', 'k_frac', 'n_bits', 'passage_idx',
                  'ppl', 'mean_ppl', 'rel_ppl', 'compression_ratio']
    with open('results/cross_arch_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved results/cross_arch_results.csv")

    # ── Write report ───────────────────────────────────────────────────────

    write_report(all_rows, baseline_means)
    print("Wrote results/REPORT-12-cross-arch.md")

    # ── Summary ────────────────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model_info in MODELS:
        mn = model_info['name']
        d_head = model_info['d_head']
        bm = baseline_means.get(mn)
        if bm is None:
            print(f"\n{mn}: SKIPPED (no data)")
            continue
        print(f"\n{mn} ({model_info['architecture']}, d_head={d_head}, baseline mean PPL={bm:.2f}):")
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

# Reference data from previous experiments
REF_14B = {
    0.50:   {'k': 64,  'rel_ppl': 3.19, 'mean_ppl': 8.25},
    0.75:   {'k': 96,  'rel_ppl': 1.26, 'mean_ppl': 3.25},
    0.875:  {'k': 112, 'rel_ppl': 1.14, 'mean_ppl': 2.95},
    1.0:    {'k': 128, 'rel_ppl': 1.05, 'mean_ppl': 2.72},
}

REF_CROSS_MODEL = {
    'Qwen3-1.7B':     {0.875: 1.32},
    'Qwen3-14B-AWQ':  {0.875: 1.14},
    'Qwen3-32B-AWQ':  {0.875: 1.05},
}


def write_report(all_rows, baseline_means):
    """Write results/REPORT-12-cross-arch.md."""

    # Build per-model lookup: model_name -> {k_frac: {mean_ppl, rel_ppl, k, cr}}
    model_data = {}
    for model_info in MODELS:
        mn = model_info['name']
        d_head = model_info['d_head']
        bm = baseline_means.get(mn)
        if bm is None:
            continue
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
        "# Experiment 12: Cross-Architecture Validation — Mistral-7B + Phi-4\n",
        "## Question\n",
        "Is the k/d_head >= 0.875 rule architecture-dependent or universal?",
        "Does it hold for Mistral and Phi3 architectures (not just Qwen3)?\n",
        "## Models Tested\n",
        "| Model | Architecture | Params | Layers | KV Heads | d_head | Quantization |",
        "|-------|-------------|--------|--------|----------|--------|-------------|",
        "| Mistral-7B-v0.3 | Mistral | 7B | 32 | 8 | 128 | BF16 |",
        "| Phi-4-AWQ | Phi3 | 14B | 40 | 10 | 128 | AWQ |",
        "| Qwen3-14B-AWQ (ref) | Qwen3 | 14B | 40 | 8 | 128 | AWQ |\n",
        "## Setup\n",
        "- K compression: subspace PCA + PolarQuant at 4-bit",
        "- V compression: full-dim PolarQuant at 4-bit",
        "- k/d_head fractions tested: 0.50, 0.75, 0.875, 0.9375, 1.0",
        "- Calibration: Project Gutenberg text, in-memory PCA (512 tokens)",
        "- Evaluation: 3 passages (scientific, historical, philosophical), 512 tokens each\n",
    ]

    # 1. PPL vs k/d_head table
    lines.append("## 1. PPL vs k/d_head (Side by Side)\n")

    header = "| k/d_head |"
    sep = "|----------|"
    for model_info in MODELS:
        mn = model_info['name']
        bm = baseline_means.get(mn, 0)
        header += f" {mn} (base={bm:.2f}) |"
        sep += "---------------------------|"
    header += " 14B-AWQ (ref) |"
    sep += "----------------|"
    lines.append(header)
    lines.append(sep)

    for kf in K_FRACS:
        row = f"| {kf:.4f} |"
        for model_info in MODELS:
            mn = model_info['name']
            d = model_data.get(mn, {}).get(kf)
            if d:
                marker = " **" if d['rel_ppl'] <= 1.20 else ""
                row += f" k={d['k']}: {d['mean_ppl']:.2f} ({d['rel_ppl']:.2f}x){marker} |"
            else:
                row += " — |"
        # 14B reference
        ref = REF_14B.get(kf)
        if ref:
            row += f" k={ref['k']}: {ref['mean_ppl']:.2f} ({ref['rel_ppl']:.2f}x) |"
        else:
            row += " — |"
        lines.append(row)

    lines.append("\n**Bold** = within 20% PPL degradation threshold.\n")

    # 2. k/d_head >= 0.875 analysis
    lines.append("## 2. Does k/d_head >= 0.875 Hold?\n")

    for model_info in MODELS:
        mn = model_info['name']
        d_head = model_info['d_head']
        lines.append(f"### {mn} ({model_info['architecture']}, {model_info['params']}, d_head={d_head})\n")

        md = model_data.get(mn, {})
        d_0875 = md.get(0.875)
        if d_0875:
            within = "YES" if d_0875['rel_ppl'] <= 1.20 else "NO"
            lines.append(f"- At k/d_head=0.875 (k={d_0875['k']}): rel_ppl = {d_0875['rel_ppl']:.2f}x -> within 20%? **{within}**")
        else:
            lines.append("- k/d_head=0.875: no data")

        # Find threshold: smallest k_frac where rel_ppl <= 1.20
        sorted_fracs = sorted(md.items(), key=lambda x: x[0])
        threshold_frac = None
        for kf_val, d in sorted_fracs:
            if d['rel_ppl'] <= 1.20:
                threshold_frac = kf_val
                break
        if threshold_frac is not None:
            lines.append(f"- Smallest k/d_head within 20%: {threshold_frac:.4f} (k={md[threshold_frac]['k']})")
        else:
            lines.append("- No k/d_head fraction achieves <= 20% PPL degradation at 4-bit")
        lines.append("")

    lines.append("### Qwen3-14B-AWQ (reference from Exp 9)\n")
    lines.append("- At k/d_head=0.875 (k=112): rel_ppl = 1.14x -> within 20%? **YES**")
    lines.append("- At k/d_head=1.0 (k=128): rel_ppl = 1.05x\n")

    # 3. Cross-architecture comparison at k/d_head=0.875
    lines.append("## 3. Cross-Architecture Comparison at k/d_head=0.875\n")
    lines.append("| Model | Architecture | Params | Rel PPL | CR |")
    lines.append("|-------|-------------|--------|---------|-----|")
    lines.append("| Qwen3-1.7B | Qwen3 | 1.7B | 1.32x | 4.27x |")

    # Mistral-7B
    d_mistral = model_data.get('Mistral-7B-v0.3', {}).get(0.875)
    if d_mistral:
        lines.append(f"| Mistral-7B-v0.3 | Mistral | 7B | {d_mistral['rel_ppl']:.2f}x | {d_mistral['cr']:.2f}x |")
    else:
        lines.append("| Mistral-7B-v0.3 | Mistral | 7B | ? | ? |")

    lines.append("| Qwen3-14B-AWQ | Qwen3 | 14B | 1.14x | 4.27x |")

    # Phi-4
    d_phi = model_data.get('Phi-4-AWQ', {}).get(0.875)
    if d_phi:
        lines.append(f"| Phi-4-AWQ | Phi3 | 14B | {d_phi['rel_ppl']:.2f}x | {d_phi['cr']:.2f}x |")
    else:
        lines.append("| Phi-4-AWQ | Phi3 | 14B | ? | ? |")

    lines.append("| Qwen3-32B-AWQ | Qwen3 | 32B | 1.05x | 4.27x |\n")

    # 4. Analysis: model-size-dependent vs architecture-dependent
    lines.append("## 4. Is the Pattern Model-Size-Dependent, Architecture-Dependent, or Both?\n")

    if d_mistral and d_phi:
        mistral_rel = d_mistral['rel_ppl']
        phi_rel = d_phi['rel_ppl']

        lines.append(f"- **Mistral-7B (7B, Mistral arch)**: {mistral_rel:.2f}x")
        lines.append(f"- **Phi-4 (14B, Phi3 arch)**: {phi_rel:.2f}x")
        lines.append(f"- **Qwen3-14B (14B, Qwen3 arch)**: 1.14x")
        lines.append(f"- **Qwen3-1.7B (1.7B, Qwen3 arch)**: 1.32x")
        lines.append(f"- **Qwen3-32B (32B, Qwen3 arch)**: 1.05x\n")

        # Size comparison: Mistral-7B vs Qwen3-14B (different arch, different size)
        lines.append("### Size vs Architecture Analysis\n")

        # Compare same-size different-arch: Phi-4 (14B, Phi3) vs Qwen3-14B (14B, Qwen3)
        phi_vs_qwen14 = abs(phi_rel - 1.14)
        lines.append(f"**Same size, different arch** (Phi-4 vs Qwen3-14B, both 14B):")
        lines.append(f"  Phi-4 = {phi_rel:.2f}x, Qwen3-14B = 1.14x, diff = {phi_vs_qwen14:.2f}\n")

        # Compare different-size same-arch: Qwen3-1.7B vs Qwen3-14B vs Qwen3-32B
        lines.append("**Same arch, different size** (Qwen3 family):")
        lines.append("  1.7B = 1.32x, 14B = 1.14x, 32B = 1.05x\n")

        # Interpretation
        if phi_vs_qwen14 < 0.10:
            lines.append("**Conclusion**: At matched size (14B), Phi-4 and Qwen3 show similar compression tolerance.")
            lines.append("This suggests the k/d_head rule is **primarily size-dependent**, not architecture-dependent.")
        elif phi_rel < 1.20 and mistral_rel < 1.20:
            lines.append("**Conclusion**: Both new architectures stay within 20% at k/d_head=0.875.")
            lines.append("The rule appears to generalize across architectures.")
        else:
            lines.append("**Conclusion**: Results vary across architectures.")
            if phi_rel > 1.20:
                lines.append(f"  Phi-4 exceeds 20% threshold ({phi_rel:.2f}x) — architecture may matter.")
            if mistral_rel > 1.20:
                lines.append(f"  Mistral-7B exceeds 20% threshold ({mistral_rel:.2f}x) — could be size or arch.")
        lines.append("")

    else:
        lines.append("Insufficient data for cross-architecture comparison.\n")

    # 5. Revised k/d_head rule
    lines.append("## 5. Revised k/d_head Rule\n")

    all_results_at_0875 = []
    for mn, md in model_data.items():
        d_0875 = md.get(0.875)
        if d_0875:
            arch = next(m['architecture'] for m in MODELS if m['name'] == mn)
            params = next(m['params'] for m in MODELS if m['name'] == mn)
            all_results_at_0875.append((mn, arch, params, d_0875['rel_ppl']))

    # Include reference models
    all_results_at_0875.append(('Qwen3-1.7B', 'Qwen3', '1.7B', 1.32))
    all_results_at_0875.append(('Qwen3-14B-AWQ', 'Qwen3', '14B', 1.14))
    all_results_at_0875.append(('Qwen3-32B-AWQ', 'Qwen3', '32B', 1.05))

    lines.append("All models at k/d_head=0.875, 4-bit:\n")
    lines.append("| Model | Arch | Params | Rel PPL | Within 20%? |")
    lines.append("|-------|------|--------|---------|-------------|")
    for mn, arch, params, rel in sorted(all_results_at_0875, key=lambda x: x[3]):
        within = "YES" if rel <= 1.20 else "NO"
        lines.append(f"| {mn} | {arch} | {params} | {rel:.2f}x | {within} |")
    lines.append("")

    # Count passes/fails
    passes = sum(1 for _, _, _, rel in all_results_at_0875 if rel <= 1.20)
    total = len(all_results_at_0875)
    lines.append(f"**{passes}/{total} models pass** the k/d_head >= 0.875 rule at 4-bit.\n")

    if passes == total:
        lines.append("The k/d_head >= 0.875 rule holds universally across all tested architectures and sizes.")
    elif passes >= total - 1:
        fails = [mn for mn, _, _, rel in all_results_at_0875 if rel > 1.20]
        lines.append(f"The rule holds for most models. Exception: {', '.join(fails)}.")
        lines.append("For these models, k/d_head >= 0.9375 may be needed.")
    else:
        lines.append("The rule does NOT universally hold. Revised recommendations:")
        lines.append("- Models >= 14B: k/d_head >= 0.875 is sufficient")
        lines.append("- Models < 14B: k/d_head >= 0.9375 may be needed")
        lines.append("- Very small models (< 3B): full-dim quantization recommended")

    # 6. Recommended config per architecture
    lines.append("\n## 6. Recommended Config per Architecture\n")

    lines.append("| Architecture | Model | Recommended k/d_head | k (d_head=128) | Notes |")
    lines.append("|-------------|-------|---------------------|-----------------|-------|")

    for model_info in MODELS:
        mn = model_info['name']
        md = model_data.get(mn, {})
        # Find smallest k_frac within 20%
        best_kf = None
        for kf in K_FRACS:
            d = md.get(kf)
            if d and d['rel_ppl'] <= 1.20:
                best_kf = kf
                break
        if best_kf is not None:
            d = md[best_kf]
            lines.append(f"| {model_info['architecture']} | {mn} | {best_kf:.4f} | {d['k']} | {d['rel_ppl']:.2f}x rel PPL |")
        else:
            lines.append(f"| {model_info['architecture']} | {mn} | 1.0 (full-dim) | 128 | No subspace saves within 20% |")

    lines.append("| Qwen3 | Qwen3-14B-AWQ | 0.875 | 112 | 1.14x rel PPL (Exp 9) |")
    lines.append("| Qwen3 | Qwen3-32B-AWQ | 0.75 | 96 | 1.09x rel PPL (Exp 11) |")
    lines.append("| Qwen3 | Qwen3-1.7B | 0.9375+ | 120+ | 1.32x at 0.875 — needs higher k (Exp 11) |")
    lines.append("")

    with open('results/REPORT-12-cross-arch.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
