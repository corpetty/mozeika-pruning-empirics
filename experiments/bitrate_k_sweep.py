"""
Experiment 9: Bitrate and subspace dimension sweep.

Systematically sweeps bit depth (4, 6, 8, 16) and subspace dimension k (64, 96, 112, 128)
to find the practical operating region where PPL stays within 20% of baseline.

Key questions:
- Does higher k help more than higher bits at equal bit budget?
- Which error source dominates: truncation or quantization?
- What is the minimum viable config for ≤20% PPL degradation?

Usage:
    /home/petty/torch-env/bin/python3 experiments/bitrate_k_sweep.py
"""

import sys
import os
import csv
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from compress import polar_quantize, subspace_polar_quantize, fit_pca, quantize_uniform
from collect import get_model_and_tokenizer, find_attention_layers, load_kvs


# ── Eval passages (same 3 as Exps 6, 8) ─────────────────────────────────────

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


# ── Compression configs ─────────────────────────────────────────────────────
# (name, k_K, nbits_K, nbits_V)
# k_K=None means baseline (no compression)
# For k_K=128: full-dim polar_quantize (no PCA projection)

CONFIGS = [
    # baseline
    ("baseline", None, None, None),

    # k=64 sweep
    ("k64_4bit",   64,  4,  4),
    ("k64_6bit",   64,  6,  6),
    ("k64_8bit",   64,  8,  8),
    ("k64_16bit",  64, 16, 16),

    # k=96 sweep
    ("k96_4bit",   96,  4,  4),
    ("k96_6bit",   96,  6,  6),
    ("k96_8bit",   96,  8,  8),
    ("k96_16bit",  96, 16, 16),

    # k=112 sweep
    ("k112_4bit",  112,  4,  4),
    ("k112_6bit",  112,  6,  6),
    ("k112_8bit",  112,  8,  8),
    ("k112_16bit", 112, 16, 16),

    # k=128 (full dim, no truncation) — pure quantization effect
    ("k128_4bit",  128,  4,  4),
    ("k128_6bit",  128,  6,  6),
    ("k128_8bit",  128,  8,  8),
]


# ── PCA bases ───────────────────────────────────────────────────────────────

def compute_pca_bases(kvs_path, max_k=112):
    """
    Compute PCA bases per (layer, head) from calibration KV data.
    Stores top max_k components — can be sliced for k=64, 96, 112.
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


# ── Compression hooks ───────────────────────────────────────────────────────

def compress_head(x_np, k, n_bits, U_k, mean):
    """
    Compress-decompress roundtrip for a single head's (T, d) vectors.

    k < 128: subspace projection + PolarQuant
    k = 128: full-dim PolarQuant (no projection)
    """
    if k == 128:
        return polar_quantize(x_np, n_bits)
    else:
        return subspace_polar_quantize(x_np, k, n_bits, U_k, mean)


def install_compression_hooks(model, k_K, nbits_K, nbits_V, bases, n_kv_heads, d_head):
    """Install hooks on k_proj/v_proj that apply compress-decompress roundtrip."""
    hooks = []
    attn_layers = find_attention_layers(model)

    for layer_idx, attn in attn_layers:
        for kv_type, proj_name, k_dim, n_bits in [
            ('K', 'k_proj', k_K, nbits_K),
            ('V', 'v_proj', 128, nbits_V),  # V always full-dim at same n_bits
        ]:
            if n_bits is None:
                continue

            def make_hook(li, kvt, kk, nb):
                def hook(module, input, output):
                    device, dtype = output.device, output.dtype
                    x = output.detach().cpu().float()
                    batch, seq, _ = x.shape
                    x = x.reshape(batch, seq, n_kv_heads, d_head)
                    for h in range(n_kv_heads):
                        xh = x[0, :, h, :].numpy()
                        U, mn = get_basis_for_k(bases, li, h, kvt, kk)
                        xh_comp = compress_head(xh, kk, nb, U, mn)
                        x[0, :, h, :] = torch.from_numpy(xh_comp)
                    return x.reshape(batch, seq, -1).to(device=device, dtype=dtype)
                return hook

            proj = getattr(attn, proj_name)
            h = proj.register_forward_hook(
                make_hook(layer_idx, kv_type, k_dim, n_bits)
            )
            hooks.append(h)

    return hooks


# ── Compression ratio ───────────────────────────────────────────────────────

def compute_compression_ratio(k_K, nbits_K, nbits_V, d_head=128):
    """
    Compute compression ratio vs FP16 baseline.

    FP16 KV per token per layer: 2 * d_head * 16 = 4096 bits
    K: k_K * nbits_K bits (subspace) or d_head * nbits_K (full-dim k=128)
    V: d_head * nbits_V bits (always full-dim)
    """
    fp16_bits = 2 * d_head * 16  # K + V in FP16

    if k_K == 128:
        k_bits = d_head * nbits_K
    else:
        k_bits = k_K * nbits_K
    v_bits = d_head * nbits_V

    return fp16_bits / (k_bits + v_bits)


# ── Perplexity computation ──────────────────────────────────────────────────

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


# ── Report ──────────────────────────────────────────────────────────────────

def write_report(rows, baseline_mean):
    """Write results/REPORT-9-bitrate-k-sweep.md."""
    d_head = 128

    # Build lookup: (k, nbits) -> mean_ppl
    grid = {}  # (k, nbits) -> mean_ppl
    for r in rows:
        if r['config'] == 'baseline':
            continue
        key = (r['k_K'], r['nbits_K'])
        if key not in grid:
            grid[key] = []
        grid[key].append(r['ppl'])
    for key in grid:
        grid[key] = np.mean(grid[key])

    k_values = [64, 96, 112, 128]
    bit_values = [4, 6, 8, 16]

    lines = [
        "# Experiment 9: Bitrate and Subspace Dimension Sweep\n",
        "## Setup\n",
        "- Model: Qwen3-14B-AWQ (40 layers, d_head=128)",
        "- K compression: subspace (k < 128) or full-dim (k=128) PolarQuant",
        "- V compression: full-dim PolarQuant at same n_bits as K",
        "- 3 evaluation passages (scientific, historical, philosophical)",
        "- Sequence length: 512 tokens",
        f"- Baseline mean PPL: {baseline_mean:.2f}\n",
    ]

    # 1. PPL heatmap table
    lines.append("## PPL Heatmap (Mean PPL)\n")
    lines.append("| k \\ n_bits | " + " | ".join(str(b) for b in bit_values) + " |")
    lines.append("|------------|" + "|".join("--------" for _ in bit_values) + "|")
    for k in k_values:
        cells = []
        for b in bit_values:
            if k == 128 and b == 16:
                cells.append("—")  # k=128/16bit not tested (essentially FP16)
            else:
                val = grid.get((k, b))
                if val is not None:
                    rel = val / baseline_mean
                    marker = ""
                    if rel <= 1.20:
                        marker = " **"
                    cells.append(f"{val:.2f} ({rel:.2f}x){marker}")
                else:
                    cells.append("—")
        lines.append(f"| k={k} | " + " | ".join(cells) + " |")
    lines.append("\n**Bold** = within 20% of baseline.\n")

    # 2. Compression ratio table
    lines.append("## Compression Ratio Table\n")
    lines.append("| k \\ n_bits | " + " | ".join(str(b) for b in bit_values) + " |")
    lines.append("|------------|" + "|".join("--------" for _ in bit_values) + "|")
    for k in k_values:
        cells = []
        for b in bit_values:
            if k == 128 and b == 16:
                cells.append("1.00x")
            else:
                cr = compute_compression_ratio(k, b, b, d_head)
                cells.append(f"{cr:.2f}x")
        lines.append(f"| k={k} | " + " | ".join(cells) + " |")

    # 3. Threshold analysis: 20%
    lines.append("\n## Configs Within 20% PPL Threshold (rel_ppl ≤ 1.20)\n")
    within_20 = []
    for (k, b), mean_ppl in sorted(grid.items()):
        rel = mean_ppl / baseline_mean
        if rel <= 1.20:
            cr = compute_compression_ratio(k, b, b, d_head)
            within_20.append((k, b, mean_ppl, rel, cr))
    if within_20:
        lines.append("| Config | k | n_bits | Mean PPL | Rel PPL | CR |")
        lines.append("|--------|---|--------|----------|---------|-----|")
        for k, b, mppl, rel, cr in sorted(within_20, key=lambda x: -x[4]):
            lines.append(f"| k{k}_{b}bit | {k} | {b} | {mppl:.2f} | {rel:.2f}x | {cr:.2f}x |")
    else:
        lines.append("**None** — no config achieves ≤20% PPL degradation.\n")

    # 4. 50% threshold
    lines.append("\n## Configs Within 50% PPL Threshold (rel_ppl ≤ 1.50)\n")
    within_50 = []
    for (k, b), mean_ppl in sorted(grid.items()):
        rel = mean_ppl / baseline_mean
        if rel <= 1.50:
            cr = compute_compression_ratio(k, b, b, d_head)
            within_50.append((k, b, mean_ppl, rel, cr))
    if within_50:
        lines.append("| Config | k | n_bits | Mean PPL | Rel PPL | CR |")
        lines.append("|--------|---|--------|----------|---------|-----|")
        for k, b, mppl, rel, cr in sorted(within_50, key=lambda x: -x[4]):
            lines.append(f"| k{k}_{b}bit | {k} | {b} | {mppl:.2f} | {rel:.2f}x | {cr:.2f}x |")
    else:
        lines.append("**None** — no config achieves ≤50% PPL degradation.\n")

    # 5. Pareto frontier
    lines.append("\n## PPL vs Compression Pareto Frontier\n")
    all_points = []
    for (k, b), mean_ppl in grid.items():
        cr = compute_compression_ratio(k, b, b, d_head)
        all_points.append((k, b, mean_ppl, cr))

    # Sort by CR descending, find Pareto-optimal (best PPL at each CR level)
    all_points.sort(key=lambda x: -x[3])
    pareto = []
    best_ppl = float('inf')
    for k, b, mppl, cr in all_points:
        if mppl <= best_ppl:
            pareto.append((k, b, mppl, cr))
            best_ppl = mppl

    lines.append("| Config | k | n_bits | Mean PPL | Rel PPL | CR | Pareto? |")
    lines.append("|--------|---|--------|----------|---------|-----|---------|")
    pareto_set = {(p[0], p[1]) for p in pareto}
    for k, b, mppl, cr in sorted(all_points, key=lambda x: -x[3]):
        rel = mppl / baseline_mean
        is_pareto = "YES" if (k, b) in pareto_set else "no"
        lines.append(f"| k{k}_{b}bit | {k} | {b} | {mppl:.2f} | {rel:.2f}x | {cr:.2f}x | {is_pareto} |")

    # 6. Equal bit-budget comparison
    lines.append("\n## Equal Bit-Budget Comparison: Truncation vs Quantization\n")
    lines.append("At roughly equal total bits per KV pair, which wins on PPL?\n")

    comparisons = [
        ("k64_8bit",  64,  8, "K: 64×8=512b, V: 128×8=1024b → 1536b total"),
        ("k96_6bit",  96,  6, "K: 96×6=576b, V: 128×6=768b → 1344b total"),
        ("k112_4bit", 112, 4, "K: 112×4=448b, V: 128×4=512b → 960b total"),
        ("k128_4bit", 128, 4, "K: 128×4=512b, V: 128×4=512b → 1024b total"),
    ]

    lines.append("| Config | K bits | V bits | Total bits | Mean PPL | Rel PPL |")
    lines.append("|--------|--------|--------|------------|----------|---------|")
    for name, k, b, desc in comparisons:
        mppl = grid.get((k, b))
        if mppl is not None:
            k_bits = k * b if k < 128 else 128 * b
            v_bits = 128 * b
            rel = mppl / baseline_mean
            lines.append(f"| {name} | {k_bits} | {v_bits} | {k_bits + v_bits} | {mppl:.2f} | {rel:.2f}x |")

    # Determine which factor dominates
    k64_8 = grid.get((64, 8))
    k128_4 = grid.get((128, 4))
    k96_6 = grid.get((96, 6))
    if k64_8 is not None and k128_4 is not None:
        if k64_8 < k128_4:
            lines.append("\nAt ~equal bit budget, k=64/8bit < k=128/4bit → **more bits per dim wins** (quantization error dominates).")
        elif k128_4 < k64_8:
            lines.append("\nAt ~equal bit budget, k=128/4bit < k=64/8bit → **more dimensions wins** (truncation error dominates).")
        else:
            lines.append("\nAt ~equal bit budget, both approaches yield similar PPL.")

    # 7. Isolating truncation vs quantization
    lines.append("\n## Isolating Truncation vs Quantization Error\n")

    k128_4 = grid.get((128, 4))
    k64_16 = grid.get((64, 16))

    lines.append("| Config | Error source | Mean PPL | Rel PPL |")
    lines.append("|--------|-------------|----------|---------|")
    if k128_4 is not None:
        rel = k128_4 / baseline_mean
        lines.append(f"| k128_4bit | Pure quantization (no truncation) | {k128_4:.2f} | {rel:.2f}x |")
    if k64_16 is not None:
        rel = k64_16 / baseline_mean
        lines.append(f"| k64_16bit | Pure truncation (16-bit ≈ lossless quant) | {k64_16:.2f} | {rel:.2f}x |")

    if k128_4 is not None and k64_16 is not None:
        if k128_4 < k64_16:
            lines.append(
                f"\n**Truncation error dominates**: pure truncation (k64/16bit, {k64_16:.2f}) is worse "
                f"than pure quantization (k128/4bit, {k128_4:.2f}). "
                "Retaining all 128 dims matters more than having high bit precision."
            )
        elif k64_16 < k128_4:
            lines.append(
                f"\n**Quantization error dominates**: pure quantization (k128/4bit, {k128_4:.2f}) is worse "
                f"than pure truncation (k64/16bit, {k64_16:.2f}). "
                "4-bit quantization noise is more damaging than discarding 50% of dimensions."
            )
        else:
            lines.append("\nBoth error sources contribute roughly equally.")

    # 8. Recommendation
    lines.append("\n## Recommendation\n")

    if within_20:
        # Best compression ratio within 20%
        best = max(within_20, key=lambda x: x[4])  # highest CR
        k, b, mppl, rel, cr = best
        lines.append(f"**Minimum viable config (best CR within ≤20% PPL):** k={k}, {b}-bit")
        lines.append(f"- Mean PPL: {mppl:.2f} ({rel:.2f}x baseline)")
        lines.append(f"- Compression ratio: {cr:.2f}x")
        lines.append(f"- K storage: {k}×{b} = {k*b} bits, V storage: 128×{b} = {128*b} bits")
        lines.append(f"- Total: {k*b + 128*b} bits vs 4096 bits FP16")
    else:
        lines.append("**No config achieves ≤20% PPL degradation.**")
        if within_50:
            best_50 = max(within_50, key=lambda x: x[4])
            k, b, mppl, rel, cr = best_50
            lines.append(f"\nBest within 50%: k={k}, {b}-bit (PPL={mppl:.2f}, {rel:.2f}x, CR={cr:.2f}x)")
        lines.append("\nNext steps: try larger k (> 112), higher bits (> 8), or mixed per-layer policies.")

    with open('results/REPORT-9-bitrate-k-sweep.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    device = 'cuda'
    max_tokens = 512

    print("=" * 70)
    print("Experiment 9: Bitrate and Subspace Dimension Sweep")
    print("=" * 70)

    # Load model once
    print("\nLoading model...")
    model, tokenizer = get_model_and_tokenizer('Qwen/Qwen3-14B-AWQ')
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    print(f"  n_kv_heads={n_kv_heads}, d_head={d_head}")

    # Compute PCA bases (max k=112, covers 64/96/112 by slicing)
    print("\nComputing PCA bases from results/kvs.npz (k=112)...")
    bases = compute_pca_bases('results/kvs.npz', max_k=112)
    print(f"  Computed bases for {len(bases)} (layer, head) pairs")

    # Evaluate each config × passage
    rows = []
    for cfg_name, k_K, nbits_K, nbits_V in CONFIGS:
        print(f"\n--- Config: {cfg_name} ---")

        if k_K is not None:
            cr = compute_compression_ratio(k_K, nbits_K, nbits_V, d_head)
            print(f"  CR = {cr:.2f}x")

        for pidx, passage in enumerate(EVAL_PASSAGES):
            if k_K is None:
                # Baseline: no hooks
                hooks = []
            else:
                hooks = install_compression_hooks(
                    model, k_K, nbits_K, nbits_V, bases, n_kv_heads, d_head
                )

            ppl, n_tok = compute_perplexity(model, tokenizer, passage, max_tokens, device)

            for h in hooks:
                h.remove()

            print(f"  Passage {pidx}: PPL = {ppl:.2f}  ({n_tok} tokens)")
            rows.append({
                'config': cfg_name,
                'k_K': k_K,
                'nbits_K': nbits_K,
                'nbits_V': nbits_V,
                'passage_idx': pidx,
                'ppl': ppl,
            })

    # Compute baseline and derived metrics
    baseline_ppls = {r['passage_idx']: r['ppl'] for r in rows if r['config'] == 'baseline'}
    baseline_mean = np.mean(list(baseline_ppls.values()))

    for r in rows:
        r['mean_ppl'] = np.mean([rr['ppl'] for rr in rows if rr['config'] == r['config']])
        r['rel_ppl'] = r['ppl'] / baseline_ppls[r['passage_idx']]
        if r['k_K'] is not None:
            r['compression_ratio'] = compute_compression_ratio(
                r['k_K'], r['nbits_K'], r['nbits_V'], d_head
            )
        else:
            r['compression_ratio'] = 1.0

    # Save CSV
    Path('results').mkdir(exist_ok=True)
    fieldnames = ['config', 'k_K', 'nbits_K', 'nbits_V', 'passage_idx',
                  'ppl', 'mean_ppl', 'rel_ppl', 'compression_ratio']
    with open('results/bitrate_k_sweep.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results/bitrate_k_sweep.csv")

    # Write report
    write_report(rows, baseline_mean)
    print("Wrote results/REPORT-9-bitrate-k-sweep.md")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Baseline mean PPL: {baseline_mean:.2f}")
    print(f"\n{'Config':<15} {'k':>4} {'bits':>5} {'Mean PPL':>10} {'Rel PPL':>10} {'CR':>8}")
    print("-" * 55)
    for cfg_name, k_K, nbits_K, nbits_V in CONFIGS:
        cr_rows = [r for r in rows if r['config'] == cfg_name]
        mean_ppl = np.mean([r['ppl'] for r in cr_rows])
        rel = mean_ppl / baseline_mean
        if k_K is not None:
            cr = compute_compression_ratio(k_K, nbits_K, nbits_V, d_head)
            print(f"{cfg_name:<15} {k_K:>4} {nbits_K:>5} {mean_ppl:>10.2f} {rel:>10.2f}x {cr:>7.2f}x")
        else:
            print(f"{cfg_name:<15} {'—':>4} {'—':>5} {mean_ppl:>10.2f} {rel:>10.2f}x {'1.00x':>8}")


if __name__ == '__main__':
    main()
