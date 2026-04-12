"""
Experiment 35 — VGG16/CIFAR-10: per-layer calibrated rho, deterministic OBD pruning.

Root cause of exp 34 collapse:
    A single global rho could not span the 15-order-of-magnitude saliency range across
    layers.  Exp 34 diagnostic (job 77) measured per-layer p10 saliencies after 1-epoch
    fine-tune of ImageNet-pretrained VGG16 on CIFAR-10:

        features.0  (conv1)       p10 = 4.4e-8
        features.2–7 (conv2-4)    p10 ~ 1.4e-11
        features.10–28 (conv5-13) p10 ~ 1.6e-13
        classifier.0 (FC1, 93M)   p10 = 2.2e-24  ← wildly different scale
        classifier.3 (FC2, 15M)   p10 = 1.4e-19
        classifier.6 (FC3)        p10 = 6.3e-13

    rho was set to 5e-6 (conv) / 2e-5 (FC) in exp 34 — orders of magnitude above
    p90 for most layers, so Glauber pruned everything immediately.

Fix (exp 35):
    Per-layer rho ≈ 0.01-0.1 × per-layer p10, keeping only the bottom ~1-5% per round.
    Deterministic zero-temperature OBD: prune iff saliency < rho/2.
    Per-layer fraction cap (10%) to bound single-round pruning.

Starting point: ImageNet pretrained + 1 epoch CIFAR-10 fine-tune (same as exp 34).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vgg16-fisher"))

from vgg16_pruning_v4 import (
    MaskedConv2d,
    MaskedLinear,
    VGGPruningConfig,
    make_masked_vgg16,
    build_cifar10_loaders,
    evaluate,
    train_model,
    estimate_diag_fisher,
    named_masked_modules,
    sparsity_report,
    compute_full_energy,
    compress_masked_vgg16,
    cleanup_dead_structure_,
    set_seed,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_ROOT = "/home/petty/.openclaw/workspace-ai-research/data"

# ---------------------------------------------------------------------------
# Per-layer rho: calibrated at ~0.01–0.1× per-layer p10 saliency
# Keys match the output of `named_masked_modules(model)` — i.e. the name
# attribute returned for each MaskedConv2d / MaskedLinear.
# ---------------------------------------------------------------------------
PER_LAYER_RHO: Dict[str, float] = {
    # conv1 — very high saliency, can afford a larger rho
    "features.0":   1e-9,
    # conv2-4 — mid saliency
    "features.2":   1e-12,
    "features.5":   1e-12,
    "features.7":   1e-12,
    # conv5-13 — lower saliency, fine-grain rho
    "features.10":  1e-14,
    "features.12":  1e-14,
    "features.14":  1e-14,
    "features.17":  1e-14,
    "features.19":  1e-14,
    "features.21":  1e-14,
    "features.24":  1e-14,
    "features.26":  1e-14,
    "features.28":  1e-14,
    # FC1 — 93M params, p10 = 2.2e-24; extremely conservative rho
    "classifier.0": 1e-25,
    # FC2 — p10 = 1.4e-19
    "classifier.3": 1e-20,
    # FC3 — output layer, keep similar scale to deep conv
    "classifier.6": 1e-13,
}

# Per-layer fraction cap: max fraction of active weights pruned in one round.
# Slows down pruning in the first few rounds; loosens as sparsity climbs.
PER_LAYER_FRAC_CAP: float = 0.10   # global default; can override per-layer below
LAYER_FRAC_CAPS: Dict[str, float] = {
    "features.0":   0.05,   # conv1 is tiny (1728 params), be careful
    "classifier.0": 0.05,   # FC1 is huge — cap to avoid mass pruning
}


# ---------------------------------------------------------------------------
# Deterministic per-layer OBD prune round
# ---------------------------------------------------------------------------

@torch.no_grad()
def perlayer_prune_round(
    model: nn.Module,
    fisher: Dict[str, torch.Tensor],
    rho_dict: Dict[str, float] = PER_LAYER_RHO,
    frac_cap: float = PER_LAYER_FRAC_CAP,
    layer_frac_caps: Dict[str, float] = LAYER_FRAC_CAPS,
) -> Dict[str, object]:
    """
    Zero-temperature OBD with per-layer rho and a fraction cap.

    Pruning criterion:
        prune weight i  iff  0.5 * F_i * w_i^2  <  rho_i / 2
                         iff  saliency_i < rho_i / 2

    The fraction cap prevents pruning more than `frac_cap` of active weights
    in any single layer per round (sorted ascending by saliency, take the
    bottom `frac_cap * n_active` weights that also satisfy the threshold).
    """
    stats: Dict[str, object] = {
        "total_pruned": 0,
        "total_active_start": 0,
        "layer_stats": {},
    }

    for name, m in named_masked_modules(model):
        rho_w = rho_dict.get(name)
        if rho_w is None:
            print(f"  [WARN] No rho for layer {name!r} — skipping.")
            continue

        layer_cap = layer_frac_caps.get(name, frac_cap)

        # ── Weights ──────────────────────────────────────────────────────
        eff_w    = m.effective_weight()
        sal_w    = 0.5 * fisher[f"{name}.weight"] * (eff_w ** 2)
        active_w = m.weight_mask.bool()
        n_active = int(active_w.sum().item())
        stats["total_active_start"] = stats["total_active_start"] + n_active

        # threshold
        threshold_w = rho_w / 2.0
        candidate_w = active_w & (sal_w < threshold_w)

        # fraction cap
        n_candidate = int(candidate_w.sum().item())
        max_prune   = max(1, int(layer_cap * n_active)) if n_active > 0 else 0

        if n_candidate > max_prune:
            # Sort candidates by saliency ascending, prune only the bottom `max_prune`
            sal_candidates = sal_w[candidate_w]
            sorted_vals, _ = torch.sort(sal_candidates)
            cutoff_val = sorted_vals[max_prune - 1].item()
            prune_w = active_w & (sal_w <= cutoff_val)
            # In case of ties, clip to max_prune
            if int(prune_w.sum().item()) > max_prune:
                # zero out the extras by taking the exact bottom indices
                sal_flat    = sal_w.view(-1)
                active_flat = active_w.view(-1)
                indices     = active_flat.nonzero(as_tuple=False).squeeze(1)
                sal_at_idx  = sal_flat[indices]
                _, order    = torch.sort(sal_at_idx)
                prune_indices = indices[order[:max_prune]]
                prune_w = torch.zeros_like(active_w.view(-1), dtype=torch.bool)
                prune_w[prune_indices] = True
                prune_w = prune_w.view_as(active_w)
        else:
            prune_w = candidate_w

        pruned_w = m.prune_weights_(prune_w)
        stats["total_pruned"] = stats["total_pruned"] + pruned_w

        # ── Biases ───────────────────────────────────────────────────────
        pruned_b = 0
        if m.bias_mask is not None and f"{name}.bias" in fisher:
            eff_b    = m.effective_bias()
            sal_b    = 0.5 * fisher[f"{name}.bias"] * (eff_b ** 2)
            active_b = m.bias_mask.bool()
            prune_b  = active_b & (sal_b < threshold_w)   # same rho for bias
            pruned_b = m.prune_biases_(prune_b)
            stats["total_pruned"] = stats["total_pruned"] + pruned_b

        stats["layer_stats"][name] = {
            "n_active_start": n_active,
            "n_candidates":   n_candidate,
            "n_pruned_w":     pruned_w,
            "n_pruned_b":     pruned_b,
            "rho":            rho_w,
            "threshold":      threshold_w,
            "frac_cap":       layer_cap,
        }

    stats.update(cleanup_dead_structure_(model))
    return stats


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    base_cfg: VGGPruningConfig,
    max_rounds: int = 80,
    target_sparsity: float = 0.99,
    train_epochs_per_round: int = 3,
    fisher_batches: int = 5,
    seed: int = 0,
    rho_dict: Optional[Dict[str, float]] = None,
    frac_cap: float = PER_LAYER_FRAC_CAP,
    layer_frac_caps: Optional[Dict[str, float]] = None,
    resume_checkpoint: Optional[str] = None,
) -> List[Dict]:
    if rho_dict is None:
        rho_dict = PER_LAYER_RHO
    if layer_frac_caps is None:
        layer_frac_caps = LAYER_FRAC_CAPS

    set_seed(seed)
    train_loader, test_loader = build_cifar10_loaders(base_cfg)

    # ── Model init ────────────────────────────────────────────────────────
    if resume_checkpoint is not None:
        print(f"Resuming from: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=base_cfg.device)
        cfg_nopre = VGGPruningConfig(**{**base_cfg.__dict__, "use_pretrained": False})
        model = make_masked_vgg16(cfg_nopre).to(base_cfg.device)
        model.load_state_dict(ckpt["masked_state_dict"], strict=False)
        records = ckpt.get("records", [])
        start_round = len(records) + 1
    else:
        print("Starting from ImageNet pretrained VGG16 — fine-tuning 1 epoch on CIFAR-10...")
        model = make_masked_vgg16(base_cfg).to(base_cfg.device)
        train_model(model, train_loader, base_cfg, epochs=1)
        records = []
        start_round = 1

    test_loss, test_acc = evaluate(model, test_loader, base_cfg.device)
    report = sparsity_report(model)
    ce0, l2_0, sp0, e0 = compute_full_energy(model, test_loader, base_cfg)
    print(
        f"Start | test acc={test_acc:.4f} | sparsity={report['global_prunable_sparsity']:.4f} | "
        f"E={e0:.6f}"
    )
    print(f"Per-layer rho: {json.dumps({k: f'{v:.2e}' for k, v in rho_dict.items()}, indent=2)}")

    for round_idx in range(start_round, max_rounds + 1):
        # Update config's fisher_batches field so estimate_diag_fisher uses it
        base_cfg.fisher_batches = fisher_batches
        fisher = estimate_diag_fisher(model, train_loader, base_cfg)

        stats = perlayer_prune_round(
            model, fisher,
            rho_dict=rho_dict,
            frac_cap=frac_cap,
            layer_frac_caps=layer_frac_caps,
        )

        train_model(model, train_loader, base_cfg, epochs=train_epochs_per_round)

        test_loss, test_acc = evaluate(model, test_loader, base_cfg.device)
        train_loss, train_acc = evaluate(model, train_loader, base_cfg.device)
        report = sparsity_report(model)
        ce, l2, sp, energy = compute_full_energy(model, test_loader, base_cfg)

        # Dead filter/neuron counts
        dead_neurons: Dict[str, int] = {}
        for n, m in named_masked_modules(model):
            if isinstance(m, MaskedConv2d):
                wm = m.weight_mask.view(m.weight_mask.shape[0], -1)
                dead_neurons[n] = int((wm.sum(dim=1) == 0).sum().item())
            else:
                dead_neurons[n] = int((m.weight_mask.sum(dim=1) == 0).sum().item())

        # Compact per-layer prune counts for the record
        layer_pruned = {
            k: v["n_pruned_w"] for k, v in stats["layer_stats"].items()
        }

        rec = {
            "round":           round_idx,
            "sparsity":        report["global_prunable_sparsity"],
            "test_acc":        test_acc,
            "test_loss":       test_loss,
            "train_acc":       train_acc,
            "train_loss":      train_loss,
            "gap":             train_acc - test_acc,
            "energy":          energy,
            "ce":              ce,
            "l2":              l2,
            "sp_penalty":      sp,
            "total_pruned":    stats["total_pruned"],
            "layer_pruned":    layer_pruned,
            "dead_neurons":    dead_neurons,
            "total_dead":      sum(dead_neurons.values()),
        }
        records.append(rec)

        # Print per-layer prune summary for first 5 rounds (debug) then compact
        layer_summary = "  ".join(
            f"{k.split('.')[-1]}:{v}" for k, v in layer_pruned.items() if v > 0
        )
        print(
            f"R{round_idx:02d} | spar={rec['sparsity']:.4f} | "
            f"test={rec['test_acc']:.4f} | train={rec['train_acc']:.4f} | "
            f"gap={rec['gap']:+.4f} | "
            f"E={energy:.5f} (CE={ce:.4f} L2={l2:.5f}) | "
            f"pruned={rec['total_pruned']:,} dead={rec['total_dead']} | "
            f"{layer_summary}"
        )

        # Intermediate checkpoint every 10 rounds
        if round_idx % 10 == 0:
            ckpt_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "vgg16-fisher",
                f"vgg16_exp35_r{round_idx:02d}.pt",
            )
            torch.save({
                "masked_state_dict": model.state_dict(),
                "records":           records,
                "round":             round_idx,
            }, ckpt_path)
            print(f"  >> Checkpoint: {ckpt_path}")

        # Save rolling JSON
        out_json = os.path.join(RESULTS_DIR, "35_records.json")
        with open(out_json, "w") as f:
            json.dump(records, f, indent=2)

        # Early stop
        if report["global_prunable_sparsity"] >= target_sparsity:
            print(f"  → Target sparsity {target_sparsity} reached at round {round_idx}.")
            break

        if stats["total_pruned"] == 0:
            print("  >> Zero weights pruned — converged.")
            break

    # Final checkpoint
    ckpt_final = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "vgg16-fisher",
        "vgg16_exp35_final.pt",
    )
    torch.save({
        "masked_state_dict": model.state_dict(),
        "records":           records,
    }, ckpt_final)

    summary = {
        "experiment":       "35_vgg16_perlayer_rho",
        "rho_strategy":     "per-layer calibrated at ~0.01-0.1x p10 saliency",
        "rho_dict":         {k: f"{v:.2e}" for k, v in rho_dict.items()},
        "final_sparsity":   records[-1]["sparsity"] if records else None,
        "final_test_acc":   records[-1]["test_acc"] if records else None,
        "rounds":           len(records),
        "peak_acc":         max(r["test_acc"] for r in records) if records else None,
        "peak_acc_round":   max(range(len(records)), key=lambda i: records[i]["test_acc"]) + 1 if records else None,
    }
    out_summary = os.path.join(RESULTS_DIR, "35_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", json.dumps(summary, indent=2))

    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 35: VGG16 per-layer calibrated rho")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--max-rounds", type=int, default=80)
    parser.add_argument("--target-sparsity", type=float, default=0.99)
    parser.add_argument("--train-epochs", type=int, default=3)
    parser.add_argument("--fisher-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_cfg = VGGPruningConfig(
        device=device,
        data_root=args.data_root,
        use_pretrained=True,
        fisher_batches=args.fisher_batches,
        train_epochs_per_round=args.train_epochs,
        finetune_epochs=args.train_epochs,
        prune_fraction_cap=None,   # we enforce fraction cap per-layer in perlayer_prune_round
    )

    run_experiment(
        base_cfg=base_cfg,
        max_rounds=args.max_rounds,
        target_sparsity=args.target_sparsity,
        train_epochs_per_round=args.train_epochs,
        fisher_batches=args.fisher_batches,
        seed=args.seed,
        resume_checkpoint=args.resume,
    )
