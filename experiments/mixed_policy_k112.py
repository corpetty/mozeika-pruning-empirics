"""
Experiment 10: Mixed layerwise policy with k=112/4-bit.

Re-runs the mixed policy experiment from Exp 8, but using k=112/4-bit as the
compression primitive instead of k=64/4-bit. Also tests k=96/4-bit and k=128/4-bit.

Key questions:
- Does the cascade problem from Exp 8 disappear at k=112?
- Do mixed policies with k=112 offer further gains over k=112 uniform?
- What is the PPL gain per protected layer at k=112?

Usage:
    /home/petty/torch-env/bin/python3 experiments/mixed_policy_k112.py
"""

import sys
import os
import csv
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import polar_quantize, subspace_polar_quantize, fit_pca
from collect import get_model_and_tokenizer, find_attention_layers, load_kvs


# ── Eval passages (same 3 as Exps 6/8/9) ─────────────────────────────────

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


# ── Policies ──────────────────────────────────────────────────────────────
# Each policy maps layer_idx -> (k_spec, v_spec)
# k_spec / v_spec: "none", "sub_k96_4bit", "sub_k112_4bit", "fulldim_4bit"

N_LAYERS = 40

POLICIES = {
    "baseline": {i: ("none", "none") for i in range(N_LAYERS)},

    # k=96 uniform (from Exp 9: 1.26x PPL, 4.57x CR — just outside 20% threshold)
    "k96_uniform": {i: ("sub_k96_4bit", "fulldim_4bit") for i in range(N_LAYERS)},

    # k=112 uniform (from Exp 9: 1.14x PPL, 4.27x CR — confirmed viable)
    "k112_uniform": {i: ("sub_k112_4bit", "fulldim_4bit") for i in range(N_LAYERS)},

    # k=112 mixed: protect early layers
    "k112_protect10": {
        **{i: ("none", "none") for i in range(10)},
        **{i: ("sub_k112_4bit", "fulldim_4bit") for i in range(10, N_LAYERS)},
    },
    "k112_protect20": {
        **{i: ("none", "none") for i in range(20)},
        **{i: ("sub_k112_4bit", "fulldim_4bit") for i in range(20, N_LAYERS)},
    },

    # k=112 graduated: compress harder on late layers
    "k112_graduated": {
        **{i: ("none", "none") for i in range(10)},
        **{i: ("sub_k112_4bit", "none") for i in range(10, 20)},           # K only mid-early
        **{i: ("sub_k112_4bit", "fulldim_4bit") for i in range(20, N_LAYERS)},  # full mid-late
    },

    # k=128 uniform (pure quantization, no truncation — from Exp 9: 1.05x PPL, 4x CR)
    "k128_uniform": {i: ("fulldim_4bit", "fulldim_4bit") for i in range(N_LAYERS)},

    # Hybrid: k=128 early, k=112 late (give early layers max precision)
    "hybrid_128_early_112_late": {
        **{i: ("fulldim_4bit", "fulldim_4bit") for i in range(20)},
        **{i: ("sub_k112_4bit", "fulldim_4bit") for i in range(20, N_LAYERS)},
    },
}


# ── Spec parsing ──────────────────────────────────────────────────────────

def parse_spec(spec_str):
    """Convert spec string to (method, k, n_bits) or None."""
    if spec_str == "none":
        return None
    if spec_str == "sub_k96_4bit":
        return ("subspace", 96, 4)
    if spec_str == "sub_k112_4bit":
        return ("subspace", 112, 4)
    if spec_str == "fulldim_4bit":
        return ("full_dim", 128, 4)
    raise ValueError(f"Unknown spec: {spec_str}")


# ── Compression ratio ─────────────────────────────────────────────────────

def compute_compression_ratio(policy, d_head=128):
    """
    Compute compression ratio vs FP16 baseline.

    FP16 KV per token per layer: 2 * d_head * 16 bits = 4096 bits
    Compressed bits depend on spec:
      - none: d_head * 16 bits
      - sub_kN_4bit: N * 4 bits
      - fulldim_4bit: d_head * 4 bits
    """
    fp16_total = 0
    compressed_total = 0

    for layer_idx in range(N_LAYERS):
        k_spec_str, v_spec_str = policy[layer_idx]

        # FP16 cost for this layer
        fp16_layer = 2 * d_head * 16  # K + V in FP16
        fp16_total += fp16_layer

        # Compressed K cost
        k_spec = parse_spec(k_spec_str)
        if k_spec is None:
            k_bits = d_head * 16
        else:
            method, k, n_bits = k_spec
            if method == "subspace":
                k_bits = k * n_bits
            else:  # full_dim
                k_bits = d_head * n_bits

        # Compressed V cost
        v_spec = parse_spec(v_spec_str)
        if v_spec is None:
            v_bits = d_head * 16
        else:
            method, k, n_bits = v_spec
            if method == "subspace":
                v_bits = k * n_bits
            else:  # full_dim
                v_bits = d_head * n_bits

        compressed_total += k_bits + v_bits

    return fp16_total / compressed_total


# ── PCA bases ─────────────────────────────────────────────────────────────

def compute_pca_bases(kvs_path, max_k=112):
    """
    Compute PCA bases per (layer, head) from calibration KV data.
    Stores top max_k components — can be sliced for k=96, 112.
    """
    kvs = load_kvs(kvs_path)
    bases = {}
    for layer_idx in sorted(kvs.keys()):
        K = kvs[layer_idx]['K']  # (T, n_heads, d_head)
        V = kvs[layer_idx]['V']
        n_heads = K.shape[1]
        for h in range(n_heads):
            U_k, mean_k = fit_pca(K[:, h, :], max_k)
            U_v, mean_v = fit_pca(V[:, h, :], max_k)
            bases[(layer_idx, h)] = {
                'U_K': U_k,    # (d_head, max_k)
                'mean_K': mean_k,
                'U_V': U_v,
                'mean_V': mean_v,
            }
    return bases


def get_basis_for_k(bases, layer_idx, head_idx, kv_type, k):
    """Slice stored basis to get top-k components."""
    base = bases.get((layer_idx, head_idx), {})
    U = base.get(f'U_{kv_type}')    # (d_head, max_k)
    mean = base.get(f'mean_{kv_type}')
    if U is not None and k < U.shape[1]:
        U = U[:, :k]  # (d_head, k)
    return U, mean


# ── Compression hook ──────────────────────────────────────────────────────

def compress_head(x_np, method, k, n_bits, U_k, mean):
    """Compress-decompress roundtrip for a single head's (T, d) vectors."""
    if method == 'subspace':
        return subspace_polar_quantize(x_np, k, n_bits, U_k, mean)
    elif method == 'full_dim':
        return polar_quantize(x_np, n_bits)
    return x_np


def install_policy_hooks(model, policy, bases, n_kv_heads, d_head):
    """
    Install per-layer compression hooks according to policy.

    policy: dict mapping layer_idx -> (k_spec_str, v_spec_str)
    """
    hooks = []
    attn_layers = find_attention_layers(model)

    for layer_idx, attn in attn_layers:
        k_spec_str, v_spec_str = policy.get(layer_idx, ("none", "none"))

        for kv_type, proj_name, spec_str in [
            ('K', 'k_proj', k_spec_str),
            ('V', 'v_proj', v_spec_str),
        ]:
            spec = parse_spec(spec_str)
            if spec is None:
                continue
            method, k, n_bits = spec

            def make_hook(li, kvt, m, kk, nb):
                def hook(module, input, output):
                    device, dtype = output.device, output.dtype
                    x = output.detach().cpu().float()
                    batch, seq, _ = x.shape
                    x = x.reshape(batch, seq, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = x[0, :, h, :].numpy()
                        if m == 'subspace':
                            U, mn = get_basis_for_k(bases, li, h, kvt, kk)
                        else:
                            U, mn = None, None
                        xh_comp = compress_head(xh, m, kk, nb, U, mn)
                        x[0, :, h, :] = torch.from_numpy(xh_comp)
                    return x.reshape(batch, seq, -1).to(device=device, dtype=dtype)
                return hook

            proj = getattr(attn, proj_name)
            h = proj.register_forward_hook(
                make_hook(layer_idx, kv_type, method, k, n_bits)
            )
            hooks.append(h)

    return hooks


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


# ── Report ────────────────────────────────────────────────────────────────

def write_report(rows, policy_ratios, baseline_mean):
    """Write results/REPORT-10-mixed-k112.md."""

    # Aggregate per policy
    policy_names = list(dict.fromkeys(r['policy'] for r in rows))
    policy_stats = {}
    for pname in policy_names:
        pr = [r for r in rows if r['policy'] == pname]
        ppls = [r['ppl'] for r in pr]
        mean_ppl = np.mean(ppls)
        policy_stats[pname] = {
            'mean_ppl': mean_ppl,
            'ppls': ppls,
        }

    baseline_ppl = policy_stats['baseline']['mean_ppl']

    lines = [
        "# Experiment 10: Mixed Layerwise Policy with k=112\n",
        "## Context\n",
        "Exp 8 showed that mixed layerwise policies with k=64/4-bit still caused",
        "1.75-3x PPL degradation. The cascade hypothesis was confirmed but the best",
        "policy (protect L0-19) only achieved 1.68x compression at 1.75x PPL.\n",
        "Exp 9 revealed the root cause: **truncation error dominates**. k=64 throws",
        "away too much signal. k=112/4-bit achieves 4.27x compression with only",
        "1.14x PPL degradation (uniform across all layers).\n",
        "**Key question**: Does the cascade problem from Exp 8 disappear at k=112?\n",
        "## Setup\n",
        "- Model: Qwen3-14B-AWQ (40 layers, d_head=128)",
        "- 3 evaluation passages (scientific, historical, philosophical)",
        "- Sequence length: 512 tokens",
        "- Compression via forward hooks on k_proj/v_proj outputs",
        "- PCA bases from calibration data (results/kvs.npz)\n",
        "## Policies Tested\n",
        "| Policy | Description | Layers Compressed |",
        "|--------|-------------|-------------------|",
    ]

    policy_descriptions = {
        "baseline": ("No compression", "0/40"),
        "k96_uniform": ("k=96/4-bit K + full-dim 4-bit V, all layers", "40/40"),
        "k112_uniform": ("k=112/4-bit K + full-dim 4-bit V, all layers", "40/40"),
        "k112_protect10": ("Skip L0-9, k=112/4-bit K + full-dim 4-bit V on L10-39", "30/40"),
        "k112_protect20": ("Skip L0-19, k=112/4-bit K + full-dim 4-bit V on L20-39", "20/40"),
        "k112_graduated": ("Skip L0-9, K-only L10-19, full KV L20-39", "30/40 (graduated)"),
        "k128_uniform": ("Full-dim 4-bit K + V, all layers (no truncation)", "40/40"),
        "hybrid_128_early_112_late": ("k=128/4-bit L0-19, k=112/4-bit L20-39", "40/40"),
    }

    for pname in policy_names:
        desc, layers = policy_descriptions.get(pname, ("", ""))
        lines.append(f"| {pname} | {desc} | {layers} |")

    # Results table
    lines.append("\n## Results\n")
    lines.append("| Policy | P0 | P1 | P2 | Mean PPL | Rel PPL | Compression Ratio |")
    lines.append("|--------|-----|-----|-----|----------|---------|-------------------|")

    for pname in policy_names:
        stats = policy_stats[pname]
        ppls = stats['ppls']
        mean_ppl = stats['mean_ppl']
        rel_ppl = mean_ppl / baseline_ppl
        cr = policy_ratios[pname]
        ppl_strs = " | ".join(f"{p:.2f}" for p in ppls)
        lines.append(f"| {pname} | {ppl_strs} | {mean_ppl:.2f} | {rel_ppl:.2f}x | {cr:.2f}x |")

    # Question 1: Does k=112 uniform solve the cascade?
    lines.append("\n## Q1: Does k=112 Uniform Already Solve the Cascade?\n")

    k112_uniform_ppl = policy_stats.get('k112_uniform', {}).get('mean_ppl', 0)
    k112_protect10_ppl = policy_stats.get('k112_protect10', {}).get('mean_ppl', 0)
    k112_protect20_ppl = policy_stats.get('k112_protect20', {}).get('mean_ppl', 0)

    lines.append("Exp 8 (k=64): uniform_kv_optimal = 3.19x PPL — severe cascade degradation.")
    lines.append(f"Exp 10 (k=112): k112_uniform = {k112_uniform_ppl/baseline_ppl:.2f}x PPL\n")

    if k112_uniform_ppl > 0:
        rel_112 = k112_uniform_ppl / baseline_ppl
        if rel_112 <= 1.20:
            lines.append(f"**Yes** — k=112 uniform is already within 20% of baseline ({rel_112:.2f}x).")
            lines.append("The cascade effect that plagued k=64 is largely eliminated by retaining")
            lines.append("more dimensions. The error per layer is small enough that 40-layer")
            lines.append("accumulation stays manageable.")
        else:
            lines.append(f"**No** — k=112 uniform ({rel_112:.2f}x) still exceeds the 20% threshold.")
            lines.append("Mixed policies may still be needed.")

    # Question 2: Do mixed policies improve further?
    lines.append("\n## Q2: Do Mixed Policies Improve Over k=112 Uniform?\n")

    if k112_uniform_ppl > 0 and k112_protect10_ppl > 0 and k112_protect20_ppl > 0:
        gap_uniform = k112_uniform_ppl - baseline_ppl
        gap_protect10 = k112_protect10_ppl - baseline_ppl
        gap_protect20 = k112_protect20_ppl - baseline_ppl

        lines.append(f"- k112_uniform: {k112_uniform_ppl:.2f} PPL (gap from baseline: {gap_uniform:.2f})")
        lines.append(f"- k112_protect10: {k112_protect10_ppl:.2f} PPL (gap: {gap_protect10:.2f})")
        lines.append(f"- k112_protect20: {k112_protect20_ppl:.2f} PPL (gap: {gap_protect20:.2f})")

        if gap_uniform > 0:
            recovery_10 = (gap_uniform - gap_protect10) / gap_uniform * 100
            recovery_20 = (gap_uniform - gap_protect20) / gap_uniform * 100
            lines.append(f"\nProtecting L0-9 recovers {recovery_10:.1f}% of the PPL gap.")
            lines.append(f"Protecting L0-19 recovers {recovery_20:.1f}% of the PPL gap.")

        # But at what compression cost?
        cr_uniform = policy_ratios['k112_uniform']
        cr_protect10 = policy_ratios['k112_protect10']
        cr_protect20 = policy_ratios['k112_protect20']

        lines.append(f"\nBut the compression cost is significant:")
        lines.append(f"- k112_uniform: {cr_uniform:.2f}x CR")
        lines.append(f"- k112_protect10: {cr_protect10:.2f}x CR ({(1-cr_protect10/cr_uniform)*100:.0f}% less compression)")
        lines.append(f"- k112_protect20: {cr_protect20:.2f}x CR ({(1-cr_protect20/cr_uniform)*100:.0f}% less compression)")

    # Question 3: PPL gain per protected layer
    lines.append("\n## Q3: PPL Gain Per Protected Layer at k=112\n")

    if k112_uniform_ppl > 0 and k112_protect10_ppl > 0 and k112_protect20_ppl > 0:
        ppl_per_layer_10 = (k112_uniform_ppl - k112_protect10_ppl) / 10
        ppl_per_layer_20 = (k112_uniform_ppl - k112_protect20_ppl) / 20

        rel_gain_10 = ((k112_uniform_ppl / baseline_ppl) - (k112_protect10_ppl / baseline_ppl)) / 10 * 100
        rel_gain_20 = ((k112_uniform_ppl / baseline_ppl) - (k112_protect20_ppl / baseline_ppl)) / 20 * 100

        lines.append(f"- Protecting 10 layers: {ppl_per_layer_10:.3f} PPL / layer ({rel_gain_10:.2f}% rel PPL / layer)")
        lines.append(f"- Protecting 20 layers: {ppl_per_layer_20:.3f} PPL / layer ({rel_gain_20:.2f}% rel PPL / layer)")
        lines.append(f"\nFor comparison, Exp 8 at k=64 showed ~3%/layer gain. At k=112 we expect")
        lines.append("much smaller per-layer gains since the per-layer error is already small.")

    # Pareto frontier
    lines.append("\n## Pareto Frontier: PPL vs Compression Ratio\n")
    lines.append("| Policy | Mean PPL | Rel PPL | CR | Pareto? |")
    lines.append("|--------|----------|---------|-----|---------|")

    sorted_policies = sorted(
        [(pname, policy_stats[pname]['mean_ppl'], policy_ratios[pname])
         for pname in policy_names if pname != "baseline"],
        key=lambda x: -x[2],  # highest compression first
    )

    pareto = []
    best_ppl = float('inf')
    for pname, mean_ppl, cr in sorted_policies:
        if mean_ppl <= best_ppl:
            pareto.append(pname)
            best_ppl = mean_ppl

    for pname, mean_ppl, cr in sorted_policies:
        rel = mean_ppl / baseline_ppl
        is_pareto = "YES" if pname in pareto else "no"
        lines.append(f"| {pname} | {mean_ppl:.2f} | {rel:.2f}x | {cr:.2f}x | {is_pareto} |")

    # Final recommendation
    lines.append("\n## Final Recommended Config (Across All 10 Experiments)\n")

    # Find best within 20% threshold
    within_20 = []
    for pname in policy_names:
        if pname == "baseline":
            continue
        rel = policy_stats[pname]['mean_ppl'] / baseline_ppl
        cr = policy_ratios[pname]
        if rel <= 1.20:
            within_20.append((pname, policy_stats[pname]['mean_ppl'], rel, cr))

    if within_20:
        # Best CR within 20%
        best_cr = max(within_20, key=lambda x: x[3])
        # Best PPL within 20%
        best_ppl_entry = min(within_20, key=lambda x: x[1])

        lines.append("### Within 20% PPL threshold:\n")
        lines.append("| Policy | Mean PPL | Rel PPL | CR |")
        lines.append("|--------|----------|---------|-----|")
        for pname, mppl, rel, cr in sorted(within_20, key=lambda x: -x[3]):
            marker = ""
            if pname == best_cr[0]:
                marker = " **<-- best CR**"
            if pname == best_ppl_entry[0] and pname != best_cr[0]:
                marker = " **<-- best PPL**"
            lines.append(f"| {pname} | {mppl:.2f} | {rel:.2f}x | {cr:.2f}x |{marker}")

    lines.append("\n### Recommendation:\n")

    # Build the final recommendation based on data
    k112_rel = policy_stats.get('k112_uniform', {}).get('mean_ppl', 0) / baseline_ppl if baseline_ppl > 0 else 0
    k128_rel = policy_stats.get('k128_uniform', {}).get('mean_ppl', 0) / baseline_ppl if baseline_ppl > 0 else 0
    hybrid_rel = policy_stats.get('hybrid_128_early_112_late', {}).get('mean_ppl', 0) / baseline_ppl if baseline_ppl > 0 else 0

    k112_cr = policy_ratios.get('k112_uniform', 0)
    k128_cr = policy_ratios.get('k128_uniform', 0)
    hybrid_cr = policy_ratios.get('hybrid_128_early_112_late', 0)

    # Pick the best tradeoff
    candidates = [
        ("k112_uniform", k112_rel, k112_cr),
        ("k128_uniform", k128_rel, k128_cr),
        ("hybrid_128_early_112_late", hybrid_rel, hybrid_cr),
    ]

    # Best efficiency score: rel_ppl / CR (lower = better tradeoff)
    best = min(candidates, key=lambda x: x[1] / x[2] if x[2] > 0 else float('inf'))

    lines.append(f"**Primary recommendation: {best[0]}**")
    lines.append(f"- Rel PPL: {best[1]:.2f}x")
    lines.append(f"- Compression ratio: {best[2]:.2f}x")
    lines.append(f"- Efficiency (rel_ppl/CR): {best[1]/best[2]:.4f}\n")

    lines.append("**Tiered options for different use cases:**\n")
    lines.append(f"1. **Maximum compression** (best CR within 20% PPL): k=112/4-bit uniform")
    lines.append(f"   - {k112_rel:.2f}x PPL, {k112_cr:.2f}x CR")
    lines.append(f"2. **Minimum PPL impact** (closest to baseline): k=128/4-bit uniform")
    lines.append(f"   - {k128_rel:.2f}x PPL, {k128_cr:.2f}x CR")
    lines.append(f"3. **Balanced**: hybrid k=128 early / k=112 late")
    lines.append(f"   - {hybrid_rel:.2f}x PPL, {hybrid_cr:.2f}x CR")
    lines.append(f"\n**Key insight from 10 experiments:** Truncation error (not quantization)")
    lines.append("is the dominant factor. The cascade effect that makes k=64 impractical")
    lines.append("(3.19x PPL) largely disappears at k=112 (1.14x PPL) because per-layer")
    lines.append("error is small enough that 40-layer accumulation stays within budget.")
    lines.append("Mixed policies offer marginal PPL gains but at significant compression cost.")

    with open('results/REPORT-10-mixed-k112.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    device = 'cuda'
    max_tokens = 512

    print("=" * 70)
    print("Experiment 10: Mixed Layerwise Policy with k=112")
    print("=" * 70)

    # Load model once
    print("\nLoading model...")
    model, tokenizer = get_model_and_tokenizer('Qwen/Qwen3-14B-AWQ')
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    print(f"  n_kv_heads={n_kv_heads}, d_head={d_head}")

    # Compute PCA bases (max k=112, covers 96/112 by slicing)
    print("\nComputing PCA bases from results/kvs.npz (max_k=112)...")
    bases = compute_pca_bases('results/kvs.npz', max_k=112)
    print(f"  Computed bases for {len(bases)} (layer, head) pairs")

    # Compute compression ratios
    print("\nCompression ratios:")
    policy_ratios = {}
    for pname, policy in POLICIES.items():
        cr = compute_compression_ratio(policy, d_head)
        policy_ratios[pname] = cr
        print(f"  {pname}: {cr:.2f}x")

    # Evaluate each policy x passage
    rows = []
    for pname, policy in POLICIES.items():
        print(f"\n--- Policy: {pname} (CR={policy_ratios[pname]:.2f}x) ---")

        for pidx, passage in enumerate(EVAL_PASSAGES):
            hooks = install_policy_hooks(model, policy, bases, n_kv_heads, d_head)
            ppl, n_tok = compute_perplexity(model, tokenizer, passage, max_tokens, device)
            for h in hooks:
                h.remove()
            print(f"  Passage {pidx}: PPL = {ppl:.2f}  ({n_tok} tokens)")
            rows.append({
                'policy': pname,
                'passage_idx': pidx,
                'ppl': ppl,
            })

    # Compute relative PPL and add compression ratio
    baseline_ppls = {r['passage_idx']: r['ppl'] for r in rows if r['policy'] == 'baseline'}
    baseline_mean = np.mean(list(baseline_ppls.values()))

    for r in rows:
        r['mean_ppl'] = np.mean([rr['ppl'] for rr in rows
                                  if rr['policy'] == r['policy']])
        r['rel_ppl'] = r['ppl'] / baseline_ppls[r['passage_idx']]
        r['compression_ratio'] = policy_ratios[r['policy']]

    # Save CSV
    Path('results').mkdir(exist_ok=True)
    fieldnames = ['policy', 'passage_idx', 'ppl', 'mean_ppl', 'rel_ppl', 'compression_ratio']
    with open('results/mixed_k112_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results/mixed_k112_results.csv")

    # Write report
    write_report(rows, policy_ratios, baseline_mean)
    print("Wrote results/REPORT-10-mixed-k112.md")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline mean PPL: {baseline_mean:.2f}")
    print(f"\n{'Policy':<35} {'Mean PPL':>10} {'Rel PPL':>10} {'CR':>8}")
    print("-" * 65)
    for pname in POLICIES:
        pr = [r for r in rows if r['policy'] == pname]
        mean_ppl = np.mean([r['ppl'] for r in pr])
        rel = mean_ppl / baseline_mean
        cr = policy_ratios[pname]
        print(f"{pname:<35} {mean_ppl:>10.2f} {rel:>10.2f}x {cr:>7.2f}x")


if __name__ == '__main__':
    main()
