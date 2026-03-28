"""
Experiment 8: Layerwise mixed compression policy.

Hypothesis: compressing early layers causes error cascades that destroy
mid/late-layer attention.  If we skip or barely compress early layers,
perplexity should stay much closer to baseline.

Tests 7 policies (including baseline) and reports PPL + compression ratio.

Usage:
    /home/petty/torch-env/bin/python3 experiments/mixed_policy_perplexity.py
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


# ── Eval passages (same 3 as Exp 6) ────────────────────────────────────────

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


# ── Policies ────────────────────────────────────────────────────────────────
# Each policy maps layer_idx -> (k_spec, v_spec)
# k_spec / v_spec: "none", "sub_k64_4bit", "sub_k64_2bit", "fulldim_4bit"

N_LAYERS = 40

POLICIES = {
    "baseline": {i: ("none", "none") for i in range(N_LAYERS)},

    # Uniform configs (re-run for comparison)
    "uniform_k64_4bit": {i: ("sub_k64_4bit", "none") for i in range(N_LAYERS)},
    "uniform_kv_optimal": {i: ("sub_k64_4bit", "fulldim_4bit") for i in range(N_LAYERS)},

    # Mixed: protect early layers
    "protect_early_10": {
        **{i: ("none", "none") for i in range(10)},
        **{i: ("sub_k64_4bit", "fulldim_4bit") for i in range(10, N_LAYERS)},
    },
    "protect_early_20": {
        **{i: ("none", "none") for i in range(20)},
        **{i: ("sub_k64_4bit", "fulldim_4bit") for i in range(20, N_LAYERS)},
    },
    "graduated": {
        **{i: ("none", "none") for i in range(10)},
        **{i: ("sub_k64_4bit", "none") for i in range(10, 20)},
        **{i: ("sub_k64_4bit", "fulldim_4bit") for i in range(20, 30)},
        **{i: ("sub_k64_2bit", "fulldim_4bit") for i in range(30, N_LAYERS)},
    },
    "k_only_all": {i: ("sub_k64_4bit", "none") for i in range(N_LAYERS)},
}


# ── Spec parsing ────────────────────────────────────────────────────────────

def parse_spec(spec_str):
    """Convert spec string to (method, k, n_bits) or None."""
    if spec_str == "none":
        return None
    if spec_str == "sub_k64_4bit":
        return ("subspace", 64, 4)
    if spec_str == "sub_k64_2bit":
        return ("subspace", 64, 2)
    if spec_str == "fulldim_4bit":
        return ("full_dim", 128, 4)
    raise ValueError(f"Unknown spec: {spec_str}")


# ── Compression ratio ──────────────────────────────────────────────────────

def compute_compression_ratio(policy, d_head=128):
    """
    Compute compression ratio vs FP16 baseline.

    FP16 KV per token per layer: 2 * d_head * 16 bits = 4096 bits
    Compressed bits depend on spec:
      - none: d_head * 16 bits
      - sub_k64_Nbit: 64 * N bits  (+ PCA basis stored once, amortized away)
      - fulldim_Nbit: d_head * N bits
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


# ── PCA bases ──────────────────────────────────────────────────────────────

def compute_pca_bases(kvs_path, k=64):
    """Compute PCA bases per (layer, head) from calibration KV data."""
    kvs = load_kvs(kvs_path)
    bases = {}
    for layer_idx in sorted(kvs.keys()):
        K = kvs[layer_idx]['K']  # (T, n_heads, d_head)
        V = kvs[layer_idx]['V']
        n_heads = K.shape[1]
        for h in range(n_heads):
            U_k, mean_k = fit_pca(K[:, h, :], k)
            U_v, mean_v = fit_pca(V[:, h, :], k)
            bases[(layer_idx, h)] = {
                'U_K': U_k, 'mean_K': mean_k,
                'U_V': U_v, 'mean_V': mean_v,
            }
    return bases


# ── Compression hook ───────────────────────────────────────────────────────

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
                        base = bases.get((li, h), {})
                        U = base.get(f'U_{kvt}')
                        mn = base.get(f'mean_{kvt}')
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


# ── Perplexity computation ─────────────────────────────────────────────────

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


# ── Report ─────────────────────────────────────────────────────────────────

def write_report(rows, policy_ratios):
    """Write results/REPORT-8-mixed-policy.md."""

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
        "# Experiment 8: Layerwise Mixed Compression Policy\n",
        "## Hypothesis\n",
        "Compressing early layers causes error cascades that destroy mid/late-layer",
        "attention fidelity (Exp 7 showed early layers at 63% top-1 match, mid/late",
        "collapsing to 26-29%). Protecting early layers from compression should keep",
        "the hidden state clean and reduce perplexity degradation.\n",
        "## Setup\n",
        "- Model: Qwen3-14B-AWQ (40 layers)",
        "- 3 evaluation passages (scientific, historical, philosophical)",
        "- Sequence length: 512 tokens",
        "- Compression via forward hooks on k_proj/v_proj outputs",
        "- PCA bases from calibration data (results/kvs.npz)\n",
        "## Policies Tested\n",
        "| Policy | L0-9 | L10-19 | L20-29 | L30-39 |",
        "|--------|------|--------|--------|--------|",
        "| baseline | none | none | none | none |",
        "| uniform_k64_4bit | K:sub64/4b | K:sub64/4b | K:sub64/4b | K:sub64/4b |",
        "| uniform_kv_optimal | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b |",
        "| protect_early_10 | none | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b |",
        "| protect_early_20 | none | none | K:sub64/4b V:fd/4b | K:sub64/4b V:fd/4b |",
        "| graduated | none | K:sub64/4b | K:sub64/4b V:fd/4b | K:sub64/2b V:fd/4b |",
        "| k_only_all | K:sub64/4b | K:sub64/4b | K:sub64/4b | K:sub64/4b |\n",
        "## Results\n",
        "| Policy | P0 | P1 | P2 | Mean PPL | Rel PPL | Compression Ratio |",
        "|--------|-----|-----|-----|----------|---------|-------------------|",
    ]

    for pname in policy_names:
        stats = policy_stats[pname]
        ppls = stats['ppls']
        mean_ppl = stats['mean_ppl']
        rel_ppl = mean_ppl / baseline_ppl
        cr = policy_ratios[pname]
        ppl_strs = " | ".join(f"{p:.2f}" for p in ppls)
        lines.append(f"| {pname} | {ppl_strs} | {mean_ppl:.2f} | {rel_ppl:.2f}x | {cr:.2f}x |")

    # Threshold analysis
    lines.append("\n## Threshold Analysis\n")
    for threshold, label in [(0.10, "10%"), (0.20, "20%"), (0.50, "50%")]:
        within = []
        for pname in policy_names:
            if pname == "baseline":
                continue
            rel = policy_stats[pname]['mean_ppl'] / baseline_ppl
            if rel - 1.0 <= threshold:
                within.append(f"{pname} ({(rel-1)*100:.1f}%)")
        if within:
            lines.append(f"**Within {label} of baseline:** {', '.join(within)}")
        else:
            lines.append(f"**Within {label} of baseline:** none")

    # Pareto frontier
    lines.append("\n## PPL vs Compression Ratio (Pareto Frontier)\n")
    lines.append("| Policy | Mean PPL | Rel PPL | Compression Ratio | Pareto? |")
    lines.append("|--------|----------|---------|-------------------|---------|")

    # Sort by compression ratio descending
    sorted_policies = sorted(
        [(pname, policy_stats[pname]['mean_ppl'], policy_ratios[pname])
         for pname in policy_names if pname != "baseline"],
        key=lambda x: -x[2],  # highest compression first
    )

    # Find Pareto-optimal: best PPL at each compression level
    pareto = []
    best_ppl_so_far = float('inf')
    for pname, mean_ppl, cr in sorted_policies:
        if mean_ppl <= best_ppl_so_far:
            pareto.append(pname)
            best_ppl_so_far = mean_ppl

    for pname, mean_ppl, cr in sorted_policies:
        rel = mean_ppl / baseline_ppl
        is_pareto = "YES" if pname in pareto else "no"
        lines.append(f"| {pname} | {mean_ppl:.2f} | {rel:.2f}x | {cr:.2f}x | {is_pareto} |")

    # Cascade analysis
    lines.append("\n## Does Protecting Early Layers Reduce the Cascade?\n")

    uniform_kv = policy_stats.get('uniform_kv_optimal', {}).get('mean_ppl', 0)
    protect10 = policy_stats.get('protect_early_10', {}).get('mean_ppl', 0)
    protect20 = policy_stats.get('protect_early_20', {}).get('mean_ppl', 0)

    if uniform_kv > 0:
        lines.append(f"- uniform_kv_optimal (all 40 layers): PPL = {uniform_kv:.2f} ({uniform_kv/baseline_ppl:.2f}x baseline)")
    if protect10 > 0:
        lines.append(f"- protect_early_10 (skip L0-9): PPL = {protect10:.2f} ({protect10/baseline_ppl:.2f}x baseline)")
        if uniform_kv > 0:
            reduction = (uniform_kv - protect10) / (uniform_kv - baseline_ppl) * 100 if uniform_kv != baseline_ppl else 0
            lines.append(f"  - Recovered {reduction:.1f}% of the PPL gap by protecting L0-9")
    if protect20 > 0:
        lines.append(f"- protect_early_20 (skip L0-19): PPL = {protect20:.2f} ({protect20/baseline_ppl:.2f}x baseline)")
        if uniform_kv > 0:
            reduction = (uniform_kv - protect20) / (uniform_kv - baseline_ppl) * 100 if uniform_kv != baseline_ppl else 0
            lines.append(f"  - Recovered {reduction:.1f}% of the PPL gap by protecting L0-19")

    if protect10 > 0 and uniform_kv > 0:
        if protect10 < uniform_kv * 0.8:
            lines.append("\n**Yes** — protecting early layers significantly reduces the cascade effect.")
        elif protect10 < uniform_kv * 0.95:
            lines.append("\n**Partially** — protecting early layers helps but doesn't eliminate the cascade.")
        else:
            lines.append("\n**No** — protecting early layers does not significantly reduce PPL degradation.")
            lines.append("The cascade may originate from mid-layer interactions rather than early-layer errors.")

    # Recommendation
    lines.append("\n## Recommended Policy\n")

    # Find best PPL-per-bit tradeoff among non-baseline policies
    best_tradeoff = None
    best_score = float('inf')
    for pname in policy_names:
        if pname == "baseline":
            continue
        rel = policy_stats[pname]['mean_ppl'] / baseline_ppl
        cr = policy_ratios[pname]
        # Score: lower is better.  rel_ppl / compression_ratio
        # (want low PPL degradation and high compression)
        score = rel / cr
        if score < best_score:
            best_score = score
            best_tradeoff = pname

    if best_tradeoff:
        rel = policy_stats[best_tradeoff]['mean_ppl'] / baseline_ppl
        cr = policy_ratios[best_tradeoff]
        lines.append(f"**Best PPL-per-bit tradeoff: {best_tradeoff}**")
        lines.append(f"- Mean PPL: {policy_stats[best_tradeoff]['mean_ppl']:.2f} ({rel:.2f}x baseline)")
        lines.append(f"- Compression ratio: {cr:.2f}x")
        lines.append(f"- Efficiency score (rel_ppl / compression_ratio): {best_score:.4f}")

    # Also recommend best among those within 20%
    within_20 = [(pname, policy_ratios[pname])
                 for pname in policy_names if pname != "baseline"
                 and policy_stats[pname]['mean_ppl'] / baseline_ppl <= 1.20]
    if within_20:
        best_cr = max(within_20, key=lambda x: x[1])
        lines.append(f"\n**Best compression within 20% PPL: {best_cr[0]}** (CR={best_cr[1]:.2f}x)")

    with open('results/REPORT-8-mixed-policy.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    device = 'cuda'
    max_tokens = 512

    print("=" * 70)
    print("Experiment 8: Layerwise Mixed Compression Policy")
    print("=" * 70)

    # Load model once
    print("\nLoading model...")
    model, tokenizer = get_model_and_tokenizer('Qwen/Qwen3-14B-AWQ')
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    print(f"  n_kv_heads={n_kv_heads}, d_head={d_head}")

    # Compute PCA bases
    print("\nComputing PCA bases from results/kvs.npz ...")
    bases = compute_pca_bases('results/kvs.npz', k=64)
    print(f"  Computed bases for {len(bases)} (layer, head) pairs")

    # Compute compression ratios
    print("\nCompression ratios:")
    policy_ratios = {}
    for pname, policy in POLICIES.items():
        cr = compute_compression_ratio(policy, d_head)
        policy_ratios[pname] = cr
        print(f"  {pname}: {cr:.2f}x")

    # Evaluate each policy × passage
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
    with open('results/mixed_policy_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results/mixed_policy_results.csv")

    # Write report
    write_report(rows, policy_ratios)
    print("Wrote results/REPORT-8-mixed-policy.md")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Policy':<25} {'Mean PPL':>10} {'Rel PPL':>10} {'CR':>8}")
    print("-" * 55)
    for pname in POLICIES:
        pr = [r for r in rows if r['policy'] == pname]
        mean_ppl = np.mean([r['ppl'] for r in pr])
        rel = mean_ppl / baseline_mean
        cr = policy_ratios[pname]
        print(f"{pname:<25} {mean_ppl:>10.2f} {rel:>10.2f}x {cr:>7.2f}x")


if __name__ == '__main__':
    main()
