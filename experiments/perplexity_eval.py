"""
Experiment 6: End-to-end perplexity with compressed KV cache.

Replaces real KV cache with compress-decompress roundtrip at k_proj/v_proj
outputs and measures perplexity degradation across compression configs.

Usage:
    /home/petty/torch-env/bin/python3 experiments/perplexity_eval.py
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


# ── Eval passages (diverse text NOT in calibration set) ──────────────────────

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


# ── Compression configs ──────────────────────────────────────────────────────
# Each config: {'K': (method, k_dim, n_bits) or None, 'V': ... or None}

CONFIGS = {
    'baseline':        {'K': None, 'V': None},
    'K_sub_k64_4bit':  {'K': ('subspace', 64, 4), 'V': None},
    'KV_optimal':      {'K': ('subspace', 64, 4), 'V': ('full_dim', 128, 4)},
    'KV_aggressive':   {'K': ('subspace', 64, 2), 'V': ('full_dim', 128, 2)},
}


# ── PCA bases ────────────────────────────────────────────────────────────────

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


# ── Compression hooks ────────────────────────────────────────────────────────

def compress_head(x_np, method, k, n_bits, U_k, mean):
    """Compress-decompress roundtrip for a single head's (T, d) vectors."""
    if method == 'subspace':
        return subspace_polar_quantize(x_np, k, n_bits, U_k, mean)
    elif method == 'full_dim':
        return polar_quantize(x_np, n_bits)
    return x_np


def install_compression_hooks(model, config, bases, n_kv_heads, d_head):
    """Install hooks on k_proj/v_proj that apply compress-decompress roundtrip."""
    hooks = []
    attn_layers = find_attention_layers(model)

    for layer_idx, attn in attn_layers:
        for kv_type, proj_name in [('K', 'k_proj'), ('V', 'v_proj')]:
            spec = config[kv_type]
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


# ── Perplexity computation ───────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, text, max_tokens=512, device='cuda'):
    """Compute perplexity of text under model (with any active hooks)."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_tokens)
    input_ids = inputs['input_ids'].to(device)
    n_tokens = input_ids.shape[1]

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # (1, seq, vocab)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return float(torch.exp(loss)), n_tokens


# ── Report ───────────────────────────────────────────────────────────────────

def write_report(rows):
    configs = list(dict.fromkeys(r['config'] for r in rows))
    passages = sorted(set(r['passage_idx'] for r in rows))

    lines = [
        "# Experiment 6: End-to-End Perplexity with Compressed KV Cache\n",
        "## Setup\n",
        "- Model: Qwen3-14B-AWQ",
        "- Evaluation: 3 text passages (scientific, historical, philosophical)",
        "- Sequence length: 512 tokens",
        "- Compression applied to k_proj/v_proj outputs via forward hooks\n",
        "## Compression Configs\n",
        "| Config | K compression | V compression |",
        "|--------|--------------|--------------|",
        "| baseline | none | none |",
        "| K_sub_k64_4bit | subspace k=64, 4-bit | none |",
        "| KV_optimal | subspace k=64, 4-bit | full-dim 4-bit |",
        "| KV_aggressive | subspace k=64, 2-bit | full-dim 2-bit |\n",
        "## Perplexity Results\n",
        "| Config | " + " | ".join(f"P{p}" for p in passages) + " | Mean PPL | Rel. PPL |",
        "|--------|" + "|".join("--------" for _ in passages) + "|----------|----------|",
    ]

    for cfg in configs:
        cfg_rows = sorted(
            [r for r in rows if r['config'] == cfg],
            key=lambda x: x['passage_idx'],
        )
        ppls = [r['ppl'] for r in cfg_rows]
        mean_ppl = np.mean(ppls)
        mean_rel = np.mean([r['relative_ppl'] for r in cfg_rows])
        ppl_strs = " | ".join(f"{p:.2f}" for p in ppls)
        lines.append(f"| {cfg} | {ppl_strs} | {mean_ppl:.2f} | {mean_rel:.4f} |")

    lines.append("\n## PPL Degradation vs Baseline\n")
    for cfg in configs:
        if cfg == 'baseline':
            continue
        cfg_rows = [r for r in rows if r['config'] == cfg]
        mean_rel = np.mean([r['relative_ppl'] for r in cfg_rows])
        pct = (mean_rel - 1.0) * 100
        safe = "SAFE (< 5%)" if pct < 5.0 else "EXCEEDS 5% threshold"
        lines.append(f"- **{cfg}**: {pct:+.2f}% — {safe}")

    lines.append("\n## KL Proxy Correlation\n")
    lines.append(
        "Previous experiments measured KL divergence on reconstructed KV vectors "
        "as a proxy for compression quality. The expected rank order of degradation "
        "(from proxy KL, least to most aggressive):"
    )
    lines.append("1. K_sub_k64_4bit (K-only, least aggressive)")
    lines.append("2. KV_optimal (K+V at 4-bit)")
    lines.append("3. KV_aggressive (K+V at 2-bit, most aggressive)\n")

    cfg_ppls = {}
    for cfg in configs:
        if cfg == 'baseline':
            continue
        cfg_ppls[cfg] = np.mean([r['relative_ppl'] for r in rows if r['config'] == cfg])

    ranked = sorted(cfg_ppls.items(), key=lambda x: x[1])
    lines.append("Actual PPL degradation rank order (least to most):")
    for i, (cfg, rel) in enumerate(ranked, 1):
        lines.append(f"{i}. {cfg}: {(rel - 1) * 100:+.2f}%")

    kl_order = ['K_sub_k64_4bit', 'KV_optimal', 'KV_aggressive']
    ppl_order = [cfg for cfg, _ in ranked]
    if ppl_order == kl_order:
        lines.append("\nThe PPL rank order **matches** the KL proxy rank order — proxy is validated.")
    else:
        lines.append(
            f"\nPPL rank order ({ppl_order}) differs from KL proxy order ({kl_order}). "
            "Partial correlation — the proxy is directionally useful but not perfectly predictive."
        )

    lines.append("\n## Conclusion\n")
    for cfg in configs:
        if cfg == 'baseline':
            continue
        rel = np.mean([r['relative_ppl'] for r in rows if r['config'] == cfg])
        pct = (rel - 1.0) * 100
        if pct < 5.0:
            lines.append(f"- **{cfg}** is safe for deployment (PPL within 5% of baseline).")
        else:
            lines.append(
                f"- **{cfg}** exceeds the 5% PPL threshold — "
                "not recommended without further tuning."
            )

    with open('results/REPORT-6-perplexity.md', 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.chdir(Path(__file__).resolve().parent.parent)
    device = 'cuda'
    max_tokens = 512

    print("=" * 70)
    print("Experiment 6: End-to-End Perplexity with Compressed KV Cache")
    print("=" * 70)

    # Load model
    print("\nLoading model...")
    model, tokenizer = get_model_and_tokenizer('Qwen/Qwen3-14B-AWQ')
    n_kv_heads = model.config.num_key_value_heads
    d_head = model.config.hidden_size // model.config.num_attention_heads
    print(f"  n_kv_heads={n_kv_heads}, d_head={d_head}")

    # Compute PCA bases from calibration data
    print("\nComputing PCA bases from results/kvs.npz ...")
    bases = compute_pca_bases('results/kvs.npz', k=64)
    print(f"  Computed bases for {len(bases)} (layer, head) pairs")

    # Evaluate each config × passage
    rows = []
    for cfg_name, cfg in CONFIGS.items():
        print(f"\n--- Config: {cfg_name} ---")
        for pidx, passage in enumerate(EVAL_PASSAGES):
            hooks = install_compression_hooks(model, cfg, bases, n_kv_heads, d_head)
            ppl, n_tok = compute_perplexity(model, tokenizer, passage, max_tokens, device)
            for h in hooks:
                h.remove()
            print(f"  Passage {pidx}: PPL = {ppl:.2f}  ({n_tok} tokens)")
            rows.append({'config': cfg_name, 'passage_idx': pidx, 'ppl': ppl})

    # Compute relative PPL
    baseline = {r['passage_idx']: r['ppl'] for r in rows if r['config'] == 'baseline'}
    for r in rows:
        r['relative_ppl'] = r['ppl'] / baseline[r['passage_idx']]

    # Save CSV
    Path('results').mkdir(exist_ok=True)
    with open('results/perplexity_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['config', 'passage_idx', 'ppl', 'relative_ppl'])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results/perplexity_results.csv")

    # Write report
    write_report(rows)
    print("Wrote results/REPORT-6-perplexity.md")

    return model, tokenizer, bases, n_kv_heads, d_head


if __name__ == '__main__':
    main()
