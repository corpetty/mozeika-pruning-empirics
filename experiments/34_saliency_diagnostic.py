"""
34_saliency_diagnostic.py
=========================
Loads ImageNet-pretrained VGG16, fine-tunes 1 epoch on CIFAR-10 (same setup
as exp 34 no-resume path), estimates diagonal Fisher, then prints per-layer
and global saliency (0.5 * F * w^2) statistics.

Goal: calibrate rho for exp 35.  We want rho << p10 saliency so that only
genuinely unimportant weights are pruned in early rounds.

Run:
    sbatch /home/petty/pruning-research/vgg16-fisher/jobs/saliency_diagnostic.sh
or interactively:
    python3 experiments/34_saliency_diagnostic.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vgg16-fisher"))

from vgg16_pruning_v4 import (
    VGGPruningConfig,
    make_masked_vgg16,
    estimate_diag_fisher,
    build_cifar10_loaders,
)

DATA_ROOT   = "/home/petty/.openclaw/workspace-ai-research/data"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# 1-epoch fine-tune (identical to exp 34 dense-start path)
# ---------------------------------------------------------------------------

def finetune_1epoch(model, loader):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
    ce  = nn.CrossEntropyLoss()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        opt.zero_grad()
        loss = ce(model(imgs), labels)
        loss.backward()
        opt.step()
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


@torch.no_grad()
def eval_acc(model, loader):
    model.eval()
    correct = total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        preds = model(imgs).argmax(1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ---------------------------------------------------------------------------
# Saliency analysis
# ---------------------------------------------------------------------------

def analyse_saliency(model, fisher):
    """
    Compute 0.5 * F * w^2 for every active (unmasked) parameter.
    Returns:
        per_layer: dict of {param_name: {"n": int, pct stats ...}}
        global_vals: flat numpy array of all saliencies
    """
    per_layer = {}
    global_vals = []

    for name, param in model.named_parameters():
        if name not in fisher:
            continue
        F    = fisher[name]
        w    = param.data

        # For masked layers, only count active weights
        # (zero weights have zero saliency anyway, but separate tracking is useful)
        sal  = (0.5 * F * w ** 2).cpu().float().numpy().ravel()
        active_sal = sal[sal > 0]   # non-zero saliencies

        all_sal = sal  # includes dead-weight zeros

        if active_sal.size == 0:
            stats = {"n_total": int(sal.size), "n_active": 0}
        else:
            pcts = np.percentile(all_sal, [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100])
            stats = {
                "n_total":  int(sal.size),
                "n_active": int(active_sal.size),
                "mean":     float(np.mean(active_sal)),
                "std":      float(np.std(active_sal)),
                "p00":  float(pcts[0]),
                "p01":  float(pcts[1]),
                "p05":  float(pcts[2]),
                "p10":  float(pcts[3]),
                "p25":  float(pcts[4]),
                "p50":  float(pcts[5]),
                "p75":  float(pcts[6]),
                "p90":  float(pcts[7]),
                "p95":  float(pcts[8]),
                "p99":  float(pcts[9]),
                "p100": float(pcts[10]),
            }
        per_layer[name] = stats
        global_vals.append(all_sal)

    global_arr = np.concatenate(global_vals)
    return per_layer, global_arr


def print_report(per_layer, global_arr):
    # ── Global stats ──────────────────────────────────────────────────────
    active = global_arr[global_arr > 0]
    g_pcts = np.percentile(active, [1, 5, 10, 25, 50, 75, 90, 95, 99])
    print("\n" + "=" * 70)
    print("GLOBAL SALIENCY STATS  (active weights only)")
    print("=" * 70)
    print(f"  total params : {global_arr.size:,}")
    print(f"  active params: {active.size:,}  ({100*active.size/global_arr.size:.1f}%)")
    print(f"  mean  : {active.mean():.4e}")
    print(f"  p01   : {g_pcts[0]:.4e}")
    print(f"  p05   : {g_pcts[1]:.4e}")
    print(f"  p10   : {g_pcts[2]:.4e}  ← rho should be well below this")
    print(f"  p25   : {g_pcts[3]:.4e}")
    print(f"  p50   : {g_pcts[4]:.4e}")
    print(f"  p75   : {g_pcts[5]:.4e}")
    print(f"  p90   : {g_pcts[6]:.4e}")
    print(f"  p95   : {g_pcts[7]:.4e}")
    print(f"  p99   : {g_pcts[8]:.4e}")

    # ── Suggested rho ─────────────────────────────────────────────────────
    p10 = g_pcts[2]
    p01 = g_pcts[0]
    rho_suggestions = {
        "p01 / 100":  p01 / 100,
        "p01 / 10":   p01 / 10,
        "p01":        p01,
        "p05 / 10":   g_pcts[1] / 10,
        "p10 / 100":  p10 / 100,
        "p10 / 10":   p10 / 10,
    }
    print("\n── Suggested rho values (rho < p10 keeps pruning selective) ──")
    for label, val in rho_suggestions.items():
        print(f"  {label:20s}  {val:.4e}")

    # ── Per-layer summary ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"{'Layer':<40} {'n_active':>10} {'p10':>12} {'p50':>12} {'p90':>12}")
    print("-" * 70)
    for name, s in per_layer.items():
        if s.get("n_active", 0) == 0:
            print(f"  {name:<38} {'ALL DEAD':>10}")
            continue
        print(f"  {name:<38} {s['n_active']:>10,} {s['p10']:>12.4e} {s['p50']:>12.4e} {s['p90']:>12.4e}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}, "
              f"{torch.cuda.get_device_properties(0).total_memory // 1024**2} MiB")

    cfg = VGGPruningConfig(
        device="cuda" if torch.cuda.is_available() else "cpu",
        data_root=DATA_ROOT,
        use_pretrained=True,
        num_classes=10,
        image_size=224,
        fisher_batches=10,   # more batches → more stable Fisher estimate
    )

    print("\nLoading ImageNet pretrained VGG16 + CIFAR-10 adapter...")
    model = make_masked_vgg16(cfg).to(DEVICE)

    train_loader, test_loader = build_cifar10_loaders(cfg)

    print("Fine-tuning 1 epoch on CIFAR-10...")
    train_acc = finetune_1epoch(model, train_loader)
    test_acc  = eval_acc(model, test_loader)
    print(f"  After 1 epoch: train={train_acc:.4f}  test={test_acc:.4f}")

    print(f"\nEstimating diagonal Fisher ({cfg.fisher_batches} batches)...")
    fisher = estimate_diag_fisher(model, train_loader, cfg)
    print(f"  Fisher keys: {len(fisher)}")

    per_layer, global_arr = analyse_saliency(model, fisher)
    print_report(per_layer, global_arr)

    # ── Save results ──────────────────────────────────────────────────────
    out = {
        "test_acc_after_1epoch": test_acc,
        "per_layer": per_layer,
        "global": {
            "n_total":  int(global_arr.size),
            "n_active": int((global_arr > 0).sum()),
            "mean":  float(np.mean(global_arr[global_arr > 0])),
        },
    }
    out_path = os.path.join(RESULTS_DIR, "34_saliency_diagnostic.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
