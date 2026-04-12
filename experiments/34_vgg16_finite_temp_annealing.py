"""
Experiment 34 — VGG16/CIFAR-10: Finite-temperature Glauber pruning with annealing.

Zero-temperature baseline (v4): prune all weights with saliency < rho/2 per round,
subject to a fraction cap. Deterministic, greedy.

This experiment: replace the hard threshold with stochastic Metropolis-Glauber acceptance.
At each round we compute delta_E for every candidate flip (prune or revive), then accept
with probability:
    P(flip) = 1 / (1 + exp(beta * delta_E))   [logistic / Fermi-Dirac]

where beta = 1/T and T anneals from T_start → T_end over max_pruning_rounds.

Key differences from v4:
1. Stochastic acceptance: weights with delta_E < 0 (energy-descending) are accepted with
   probability > 0.5, but NOT guaranteed. Weights with delta_E > 0 (energy-ascending) can
   still be pruned with small probability (exploration at high T).
2. Regrowth: currently-pruned weights can be revived if their delta_E (for flipping 0→1)
   is negative. This closes the loop between pruning and regrowth in each round.
3. Temperature schedule: linear, geometric, or cosine annealing — configurable.
4. The fraction cap from v4 is removed (T controls the pruning rate naturally).

delta_E for pruning weight i (h_i: 1 → 0):
    delta_E_prune = -saliency_i + rho_i/2
                  = -(1/2) * F_i * w_i^2 + rho_i/2

    Pruning decreases energy iff saliency < rho/2 — same threshold as v4, but now we
    sample rather than threshold.

delta_E for regrowing weight i (h_i: 0 → 1):
    delta_E_regrow = +saliency_i - rho_i/2
                   = +(1/2) * F_i * w_i^2 - rho_i/2

    Reviving a pruned weight increases energy iff saliency < rho/2, but at high T the
    acceptance probability smooths this out — the network can explore different masks.

Note on saliency for pruned weights: Fisher is re-estimated each round using current
(masked) activations. Pruned weights have zero effective weight, so their OBD saliency
is exactly 0. delta_E_regrow = -rho/2 for all pruned weights at any T, meaning regrowth
is always energy-ascending (costs rho/2). At low T this is almost never accepted; at
high T it's accepted ~50% of the time (when delta_E → 0 limit isn't hit). In practice
we set T_start small enough that regrowth is controlled but non-zero.

Resume: supports --resume path --start-round N to continue from a checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import everything from v4 — masked layers, model builder, data loaders, etc.
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

VGG16_CHECKPOINT_99 = "/home/petty/pruning-research/vgg16-fisher/vgg16_pruned_and_compressed_v4_99pct.pt"
VGG16_CHECKPOINT_95 = "/home/petty/pruning-research/vgg16-fisher/vgg16_pruned_and_compressed_v3_95pct.pt"
VGG16_CHECKPOINT_90 = "/home/petty/.openclaw/workspace-ai-research/vgg16_pruned_and_compressed.pt"
DATA_ROOT = "/home/petty/.openclaw/workspace-ai-research/data"


# ------------------------------------------------------------------
# Annealing schedules
# ------------------------------------------------------------------

def linear_schedule(T_start: float, T_end: float, total_rounds: int, r: int) -> float:
    """T decreases linearly from T_start (r=1) to T_end (r=total_rounds)."""
    if total_rounds <= 1:
        return T_end
    frac = (r - 1) / (total_rounds - 1)
    return T_start * (1.0 - frac) + T_end * frac


def geometric_schedule(T_start: float, T_end: float, total_rounds: int, r: int) -> float:
    """T decreases geometrically (log-linear) from T_start to T_end."""
    if total_rounds <= 1 or T_end <= 0:
        return T_end
    frac = (r - 1) / (total_rounds - 1)
    return T_start * (T_end / T_start) ** frac


def cosine_schedule(T_start: float, T_end: float, total_rounds: int, r: int) -> float:
    """Cosine annealing: T = T_end + 0.5*(T_start-T_end)*(1 + cos(pi*frac))."""
    if total_rounds <= 1:
        return T_end
    frac = (r - 1) / (total_rounds - 1)
    return T_end + 0.5 * (T_start - T_end) * (1.0 + math.cos(math.pi * frac))


SCHEDULES = {
    "linear": linear_schedule,
    "geometric": geometric_schedule,
    "cosine": cosine_schedule,
}


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class FiniteTempConfig:
    """Finite-temperature annealing config on top of VGGPruningConfig."""

    # Temperature schedule.
    # Calibration: typical saliency = 0.5*F*w^2 ~ 1e-15 to 1e-9 (median ~5e-15 at 90% sparsity).
    # rho/2 ~ 2.5e-6 for conv weights. delta_E for most active weights ≈ +2.5e-6 (should NOT prune).
    # For T to be "just exploratory": beta*delta_E ~ 0.5, so T ~ delta_E / 0.5 ~ 5e-6.
    # We want T_start well below rho/2 so only clearly energy-descending flips are favoured.
    # T_start=1e-7: beta=1e7, P(prune energy-ascending by 2.5e-6) = sigmoid(-25) ≈ 1.4e-11 (safe)
    #               P(prune energy-descending by 2.5e-6) = sigmoid(+25) ≈ 1 (greedy for these)
    # T_end=1e-10: effectively zero temperature throughout — this is really a warm start then greedy.
    T_start: float = 1e-7          # initial temperature (calibrated to rho/2 ~ 2.5e-6 scale)
    T_end: float = 1e-10           # final temperature (effectively greedy)
    schedule: str = "geometric"    # "linear" | "geometric" | "cosine"

    # Pruning rounds
    max_pruning_rounds: int = 80
    target_global_sparsity: Optional[float] = 0.99

    # rho values (same as v4 for fair comparison)
    rho_conv_w: float = 5e-6
    rho_conv_b: float = 5e-6
    rho_fc_w: Tuple[float, float, float] = (2e-5, 2e-5, 5e-6)
    rho_fc_b: Tuple[float, float] = (2e-5, 2e-5)

    # Training
    train_epochs_per_round: int = 1
    finetune_epochs: int = 1
    fisher_batches: int = 3

    # Allow regrowth (flip 0→1 via Glauber)
    allow_regrowth: bool = True

    # Per-round regrowth cap: max fraction of currently-pruned weights to consider
    # for regrowth. Limits compute. Set to None for unlimited.
    regrowth_cap: Optional[float] = 0.05

    # Seed
    seed: int = 0

    # Resume
    resume_checkpoint: Optional[str] = None
    start_round: int = 1

    def T_at(self, r: int) -> float:
        fn = SCHEDULES[self.schedule]
        return fn(self.T_start, self.T_end, self.max_pruning_rounds, r)

    def beta_at(self, r: int) -> float:
        T = self.T_at(r)
        return float("inf") if T <= 0 else 1.0 / T


# ------------------------------------------------------------------
# Stochastic Glauber sweep
# ------------------------------------------------------------------

@torch.no_grad()
def glauber_prune_round(
    model: nn.Module,
    fisher: Dict[str, torch.Tensor],
    cfg: FiniteTempConfig,
    base_cfg: VGGPruningConfig,
    round_idx: int,
) -> Dict[str, int]:
    """
    One Glauber sweep over all weights.

    For each active weight i:
        delta_E = -saliency_i + rho_i/2
        P(prune) = sigmoid(-beta * delta_E)  [= 1/(1+exp(beta*delta_E))]
        Sample Bernoulli(P(prune)) to decide.

    For each pruned weight i (if allow_regrowth):
        delta_E_revive = +saliency_i - rho_i/2 = -(delta_E_prune)
        But saliency of pruned weight = 0 (zero effective weight), so:
        delta_E_revive = 0 - rho_i/2 = -rho_i/2
        P(revive) = sigmoid(beta * rho_i/2)  -- always < 0.5 since rho>0
        At T→0: P→0. At high T: P→0.5.
    """
    T = cfg.T_at(round_idx)
    beta = cfg.beta_at(round_idx)

    stats: Dict[str, int] = {
        "T": T,
        "beta": beta,
        "total_pruned": 0,
        "total_revived": 0,
        "total_descent_dirs": 0,   # |delta_E < 0| for active weights
        "total_ascent_pruned": 0,  # pruned despite delta_E > 0 (thermal noise)
    }

    conv_index = 0
    fc_index = 0

    for name, m in named_masked_modules(model):
        if isinstance(m, MaskedConv2d):
            rho_w = base_cfg.rho_conv_w
            rho_b = base_cfg.rho_conv_b
            is_conv = True
        else:
            rho_w = base_cfg.rho_fc_w[fc_index]
            rho_b = base_cfg.rho_fc_b[fc_index] if fc_index < 2 else None
            is_conv = False

        # --- Weight pruning ---
        eff_w = m.effective_weight()
        sal_w = 0.5 * fisher[f"{name}.weight"] * (eff_w ** 2)
        active_w = m.weight_mask.bool()

        delta_E_w = -sal_w + rho_w / 2.0  # prune lowers energy iff delta_E < 0
        stats["total_descent_dirs"] += int((active_w & (delta_E_w < 0)).sum().item())

        if beta == float("inf"):
            # Zero temperature: deterministic threshold
            prune_w = active_w & (delta_E_w < 0)
        else:
            # Stochastic acceptance: P(prune) = sigmoid(-beta * delta_E)
            log_p_prune = -beta * delta_E_w[active_w]  # logit for pruning
            log_p_prune = log_p_prune.clamp(-30, 30)
            p_prune = torch.sigmoid(log_p_prune)
            samples = torch.bernoulli(p_prune).bool()
            prune_w_active = samples
            prune_w = torch.zeros_like(active_w)
            prune_w[active_w] = prune_w_active

        stats["total_ascent_pruned"] += int((prune_w & (delta_E_w > 0)).sum().item())
        pruned_w = m.prune_weights_(prune_w)
        stats[f"{'conv' if is_conv else 'fc'}{conv_index if is_conv else fc_index+1}.weights_pruned"] = pruned_w
        stats["total_pruned"] += pruned_w

        # --- Weight regrowth (0→1 flip) ---
        if cfg.allow_regrowth:
            dead_w = ~m.weight_mask.bool()
            if dead_w.any() and beta < float("inf"):
                # delta_E_revive = -rho/2 for all pruned weights (sal=0)
                # P(revive) = sigmoid(-beta * rho/2)
                p_revive_scalar = torch.sigmoid(torch.tensor(-beta * rho_w / 2.0)).item()

                # Fast path: if expected revives ≈ 0, skip all the expensive indexing.
                # At T=1e-7, rho=5e-6: P(revive)=sigmoid(-25)≈1.4e-11 → skip immediately.
                n_dead_approx = int(dead_w.sum().item())
                cap = cfg.regrowth_cap if cfg.regrowth_cap is not None else 1.0
                max_candidates = max(1, int(n_dead_approx * cap)) * 4
                expected_revives = p_revive_scalar * max_candidates
                if expected_revives >= 0.5 and p_revive_scalar > 0 and n_dead_approx > 0:
                    max_revive = max(1, int(n_dead_approx * (cfg.regrowth_cap or 1.0)))
                    # Flatten weight mask for uniform dead-weight sampling
                    dead_flat = dead_w.view(-1).nonzero(as_tuple=False).squeeze(1)
                    # Cap candidates for speed
                    if len(dead_flat) > max_revive * 4:
                        perm = torch.randperm(len(dead_flat), device=dead_flat.device)[:max_revive * 4]
                        dead_flat = dead_flat[perm]
                    accept = torch.bernoulli(
                        torch.full((len(dead_flat),), p_revive_scalar, device=dead_w.device)
                    ).bool()
                    revive_flat = dead_flat[accept]
                    if len(revive_flat) > 0:
                        revive_mask = torch.zeros_like(m.weight_mask.view(-1), dtype=torch.bool)
                        revive_mask[revive_flat] = True
                        revive_mask = revive_mask.view_as(m.weight_mask)
                        m.weight_mask[revive_mask] = 1.0
                        revived_w = int(revive_mask.sum().item())
                        stats[f"{'conv' if is_conv else 'fc'}{conv_index if is_conv else fc_index+1}.weights_revived"] = revived_w
                        stats["total_revived"] += revived_w

        # --- Bias handling ---
        if m.bias_mask is not None:
            eff_b = m.effective_bias()
            sal_b = 0.5 * fisher[f"{name}.bias"] * (eff_b ** 2)
            active_b = m.bias_mask.bool()
            delta_E_b = -sal_b + rho_b / 2.0 if rho_b is not None else torch.full_like(sal_b, float("inf"))

            if rho_b is not None:
                if beta == float("inf"):
                    prune_b = active_b & (delta_E_b < 0)
                else:
                    log_p_b = -beta * delta_E_b[active_b]
                    log_p_b = log_p_b.clamp(-30, 30)
                    p_b = torch.sigmoid(log_p_b)
                    samples_b = torch.bernoulli(p_b).bool()
                    prune_b = torch.zeros_like(active_b)
                    prune_b[active_b] = samples_b

                pruned_b = m.prune_biases_(prune_b)
                stats[f"{'conv' if is_conv else 'fc'}{conv_index if is_conv else fc_index+1}.biases_pruned"] = pruned_b
                stats["total_pruned"] += pruned_b

        if is_conv:
            conv_index += 1
        else:
            fc_index += 1

    stats.update(cleanup_dead_structure_(model))
    return stats


# ------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------

def run_finite_temp_experiment(
    ft_cfg: FiniteTempConfig,
    base_cfg: VGGPruningConfig,
) -> List[Dict]:
    set_seed(ft_cfg.seed)
    train_loader, test_loader = build_cifar10_loaders(base_cfg)

    # Load checkpoint or start from pretrained dense
    resume = ft_cfg.resume_checkpoint
    if resume is not None:
        print(f"Resuming from: {resume}")
        ckpt = torch.load(resume, map_location=base_cfg.device)
        cfg_nopre = VGGPruningConfig(**{**base_cfg.__dict__, "use_pretrained": False})
        model = make_masked_vgg16(cfg_nopre).to(base_cfg.device)
        model.load_state_dict(ckpt["masked_state_dict"], strict=False)
    else:
        print("No checkpoint provided — loading ImageNet pretrained VGG16 and fine-tuning 1 epoch on CIFAR-10...")
        cfg_pre = VGGPruningConfig(**{**base_cfg.__dict__, "use_pretrained": True})
        model = make_masked_vgg16(cfg_pre).to(base_cfg.device)
        train_model(model, train_loader, base_cfg, epochs=1)
        print("Dense fine-tune complete.")

    test_loss, test_acc = evaluate(model, test_loader, base_cfg.device)
    report = sparsity_report(model)
    ce0, l2_0, sp0, e0 = compute_full_energy(model, test_loader, base_cfg)
    print(
        f"Start | test acc={test_acc:.4f} | sparsity={report['global_prunable_sparsity']:.4f} | "
        f"E={e0:.6f} (CE={ce0:.4f} L2={l2_0:.6f} SP={sp0:.4f})"
    )
    print(f"Schedule: {ft_cfg.schedule} | T_start={ft_cfg.T_start:.2e} → T_end={ft_cfg.T_end:.2e}")
    print(f"Regrowth: {ft_cfg.allow_regrowth} | regrowth_cap={ft_cfg.regrowth_cap}")

    records = []

    for round_idx in range(ft_cfg.start_round, ft_cfg.max_pruning_rounds + 1):
        T = ft_cfg.T_at(round_idx)
        fisher = estimate_diag_fisher(model, train_loader, base_cfg)
        stats = glauber_prune_round(model, fisher, ft_cfg, base_cfg, round_idx)
        train_model(model, train_loader, base_cfg, epochs=base_cfg.finetune_epochs)
        test_loss, test_acc = evaluate(model, test_loader, base_cfg.device)
        train_loss, train_acc = evaluate(model, train_loader, base_cfg.device)
        report = sparsity_report(model)
        ce, l2, sp, energy = compute_full_energy(model, test_loader, base_cfg)

        # Count dead neurons across all conv/fc layers (filters/neurons with all weights zeroed)
        dead_neurons: Dict[str, int] = {}
        for name, m in named_masked_modules(model):
            if isinstance(m, MaskedConv2d):
                # dead filter = all weights in that filter (out_channel) are zero
                w_mask = m.weight_mask.view(m.weight_mask.shape[0], -1)  # [out_ch, k*k*in]
                dead_neurons[name] = int((w_mask.sum(dim=1) == 0).sum().item())
            else:
                # dead neuron (row) = all outgoing weights zeroed
                dead_neurons[name] = int((m.weight_mask.sum(dim=1) == 0).sum().item())

        rec = {
            "round": round_idx,
            "T": T,
            "sparsity": report["global_prunable_sparsity"],
            "test_acc": test_acc,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "train_loss": train_loss,
            "train_test_gap": train_acc - test_acc,
            "energy": energy,
            "ce": ce,
            "l2": l2,
            "sp_penalty": sp,
            "total_pruned": stats["total_pruned"],
            "total_revived": stats.get("total_revived", 0),
            "descent_dirs": stats.get("total_descent_dirs", 0),
            "ascent_pruned": stats.get("total_ascent_pruned", 0),
            "dead_neurons": dead_neurons,
            "total_dead_neurons": sum(dead_neurons.values()),
        }
        records.append(rec)

        print(
            f"R{round_idx:02d} | T={T:.2e} | spar={rec['sparsity']:.4f} | "
            f"test_acc={rec['test_acc']:.4f} test_loss={rec['test_loss']:.4f} | "
            f"train_acc={rec['train_acc']:.4f} train_loss={rec['train_loss']:.4f} | "
            f"gap={rec['train_test_gap']:+.4f} | "
            f"E={energy:.5f} (CE={ce:.4f} L2={l2:.5f} SP={sp:.5f}) | "
            f"prune={rec['total_pruned']} revive={rec['total_revived']} "
            f"ascent_prune={rec['ascent_pruned']} dead_neurons={rec['total_dead_neurons']}"
        )

        # Save intermediate results
        out_json = os.path.join(RESULTS_DIR, "34_finite_temp_records.json")
        with open(out_json, "w") as f:
            json.dump(records, f, indent=2)

        if (
            ft_cfg.target_global_sparsity is not None
            and report["global_prunable_sparsity"] >= ft_cfg.target_global_sparsity
        ):
            print(f"  → Target sparsity {ft_cfg.target_global_sparsity} reached.")
            break

        if stats["total_pruned"] == 0 and stats.get("total_revived", 0) == 0 and T < 1e-10:
            print("  >> Converged (zero moves at near-zero T).")
            break

    # Save checkpoint
    ckpt_out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "vgg16-fisher",
        "vgg16_finite_temp_annealed.pt",
    )
    torch.save({"masked_state_dict": model.state_dict(), "records": records, "ft_cfg": ft_cfg.__dict__}, ckpt_out)
    print(f"Checkpoint saved: {ckpt_out}")

    # Summary
    summary = {
        "schedule": ft_cfg.schedule,
        "T_start": ft_cfg.T_start,
        "T_end": ft_cfg.T_end,
        "allow_regrowth": ft_cfg.allow_regrowth,
        "final_sparsity": records[-1]["sparsity"] if records else None,
        "final_acc": records[-1]["test_acc"] if records else None,
        "rounds": len(records),
        "peak_acc": max(r["test_acc"] for r in records) if records else None,
        "peak_acc_round": max(range(len(records)), key=lambda i: records[i]["test_acc"]) + 1 if records else None,
    }
    out_summary = os.path.join(RESULTS_DIR, "34_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print("Summary:", json.dumps(summary, indent=2))

    return records


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 34: VGG16 finite-temperature Glauber annealing")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint to resume from (default: 90pct checkpoint)")
    parser.add_argument("--start-round", type=int, default=1)
    parser.add_argument("--T-start", type=float, default=1e-7)
    parser.add_argument("--T-end", type=float, default=1e-10)
    parser.add_argument("--schedule", type=str, default="geometric",
                        choices=["linear", "geometric", "cosine"])
    parser.add_argument("--max-rounds", type=int, default=80)
    parser.add_argument("--target-sparsity", type=float, default=0.99)
    parser.add_argument("--no-regrowth", action="store_true")
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_cfg = VGGPruningConfig(
        device=device,
        data_root=args.data_root,
        use_pretrained=True,   # used only when no --resume; overridden to False when loading ckpt
        fisher_batches=3,
        train_epochs_per_round=1,
        finetune_epochs=1,
        # No fraction cap — temperature controls pruning rate
        prune_fraction_cap=None,
    )

    ft_cfg = FiniteTempConfig(
        T_start=args.T_start,
        T_end=args.T_end,
        schedule=args.schedule,
        max_pruning_rounds=args.max_rounds,
        target_global_sparsity=args.target_sparsity,
        allow_regrowth=not args.no_regrowth,
        resume_checkpoint=args.resume,
        start_round=args.start_round,
        seed=args.seed,
    )

    run_finite_temp_experiment(ft_cfg, base_cfg)
