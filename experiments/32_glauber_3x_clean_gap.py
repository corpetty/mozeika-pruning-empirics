"""
32_glauber_3x_clean_gap.py

Re-run of the 3x Glauber experiment (LeNet-300-100, MNIST) with a CLEAN
generalisation gap:

    train_ce  = CE only (no L2), evaluated on training data  [no grad]
    test_ce   = CE only (no L2), evaluated on test data      [no grad]
    gap       = train_ce - test_ce   (positive = overfitting)

The optimiser still uses CE + L2 internally — the clean eval is a separate
no-grad forward pass after each fine-tune step.

Config: 3 sweeps/round, linear T anneal 1e-7 → 0 over 20 steps,
        max_rounds=60, target_sparsity=0.99

Output: results/glauber_3x_clean_gap.json
"""

import json
import math
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, prune_bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("weight_mask", torch.ones(out_features, in_features))
        self.bias_mask = (
            self._create_bias_mask(out_features) if bias and prune_bias else None
        )
        self.reset_parameters()

    def _create_bias_mask(self, n):
        buf = torch.ones(n)
        self.register_buffer("bias_mask", buf)
        return self.get_buffer("bias_mask")

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_weight(self):
        return self.weight * self.weight_mask

    def effective_bias(self):
        if self.bias is None:
            return None
        return self.bias if self.bias_mask is None else self.bias * self.bias_mask

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.effective_bias())

    @torch.no_grad()
    def zero_masked_parameters_(self):
        self.weight.data.mul_(self.weight_mask)
        if self.bias is not None and self.bias_mask is not None:
            self.bias.data.mul_(self.bias_mask)

    @torch.no_grad()
    def zero_masked_grads_(self):
        if self.weight.grad is not None:
            self.weight.grad.mul_(self.weight_mask)
        if self.bias is not None and self.bias.grad is not None and self.bias_mask is not None:
            self.bias.grad.mul_(self.bias_mask)


class MaskedLeNet300100(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = MaskedLinear(784, 300, bias=True, prune_bias=True)
        self.fc2 = MaskedLinear(300, 100, bias=True, prune_bias=True)
        self.fc3 = MaskedLinear(100, 10,  bias=True, prune_bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def layers(self):
        return [self.fc1, self.fc2, self.fc3]

    @torch.no_grad()
    def cleanup_dead_neurons_(self):
        stats = {"dead_fc1": 0, "dead_fc2": 0}
        dead1 = (self.fc1.weight_mask.sum(1) == 0)
        if self.fc1.bias_mask is not None:
            dead1 &= (self.fc1.bias_mask == 0)
        stats["dead_fc1"] = int(dead1.sum())
        if dead1.any():
            cols = dead1.unsqueeze(0).expand_as(self.fc2.weight_mask)
            newly = self.fc2.weight_mask.bool() & cols
            self.fc2.weight_mask[newly] = 0.0
            self.fc2.weight.data[newly] = 0.0

        dead2 = (self.fc2.weight_mask.sum(1) == 0)
        if self.fc2.bias_mask is not None:
            dead2 &= (self.fc2.bias_mask == 0)
        stats["dead_fc2"] = int(dead2.sum())
        if dead2.any():
            cols = dead2.unsqueeze(0).expand_as(self.fc3.weight_mask)
            newly = self.fc3.weight_mask.bool() & cols
            self.fc3.weight_mask[newly] = 0.0
            self.fc3.weight.data[newly] = 0.0
        return stats


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Cfg:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 1e-3
    batch_size: int = 128
    train_epochs_per_round: int = 2
    finetune_epochs: int = 2
    max_rounds: int = 60
    fisher_batches: int = 50
    eta_w: float = 1e-4
    eta_b: float = 1e-4
    rho_w: Tuple = (1e-7, 2e-7, 5e-8)
    rho_b: Tuple = (1e-7, 2e-7)
    target_sparsity: float = 0.99
    eps_curv: float = 1e-12
    allow_regrowth: bool = True
    max_flip_fraction: float = 0.2
    seed: int = 0
    T_start: float = 1e-7
    temp_steps: int = 20
    sweeps_per_round: int = 3
    data_root: str = "/home/petty/.openclaw/workspace-ai-research/data"
    out_path: str = os.path.join(os.path.dirname(__file__), "..", "results",
                                  "glauber_3x_clean_gap.json")

    def T_at(self, round_idx: int) -> float:
        step = min(round_idx - 1, self.temp_steps - 1)
        frac = step / max(self.temp_steps - 1, 1)
        return self.T_start * (1.0 - frac)

    def beta_at(self, round_idx: int) -> float:
        T = self.T_at(round_idx)
        return float("inf") if T <= 0 else 1.0 / T


def set_seed(s):
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def masked_l2(model, cfg):
    dev = next(model.parameters()).device
    out = torch.zeros((), device=dev)
    out += 0.5 * cfg.eta_w * sum((l.effective_weight() ** 2).sum() for l in model.layers())
    out += 0.5 * cfg.eta_b * (
        (model.fc1.effective_bias() ** 2).sum()
        + (model.fc2.effective_bias() ** 2).sum()
        + (model.fc3.bias ** 2).sum()
    )
    return out


def train_epoch(model, loader, cfg) -> None:
    """Train one epoch with CE + L2. No loss logging here — eval separately."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y) + masked_l2(model, cfg)
        loss.backward()
        for l in model.layers():
            l.zero_masked_grads_()
        opt.step()
        with torch.no_grad():
            for l in model.layers():
                l.zero_masked_parameters_()


def train_model(model, loader, cfg, epochs) -> None:
    for _ in range(epochs):
        train_epoch(model, loader, cfg)


@torch.no_grad()
def eval_ce(model, loader, device) -> Tuple[float, float]:
    """
    Pure CE evaluation — NO L2.
    Returns (mean_ce, accuracy). Use for both train and test sets.
    """
    model.eval()
    total_ce = total_correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_ce      += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (logits.argmax(1) == y).sum().item()
        total         += y.size(0)
    return total_ce / total, total_correct / total


def energy(model, test_loader, cfg):
    """Free energy = CE + L2 + density term (for Glauber diagnostics only)."""
    model.eval()
    ce_sum = n = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            ce_sum += F.cross_entropy(model(x), y, reduction="sum").item()
            n      += y.size(0)
    ce  = ce_sum / n
    l2  = masked_l2(model, cfg).item()
    rho = sum(l.weight_mask.float().mean().item() for l in model.layers()) / 3
    return ce + l2, ce, l2, rho


# ---------------------------------------------------------------------------
# Fisher / Glauber
# ---------------------------------------------------------------------------

def estimate_fisher(model, loader, cfg):
    model.train()
    names  = ["fc1", "fc2", "fc3"]
    layers = [model.fc1, model.fc2, model.fc3]
    stats  = {}
    for n, l in zip(names, layers):
        stats[f"{n}.weight.grad"]   = torch.zeros_like(l.weight)
        stats[f"{n}.weight.fisher"] = torch.zeros_like(l.weight)
        if l.bias is not None and l.bias_mask is not None:
            stats[f"{n}.bias.grad"]   = torch.zeros_like(l.bias)
            stats[f"{n}.bias.fisher"] = torch.zeros_like(l.bias)
    batches = 0
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        model.zero_grad(set_to_none=True)
        F.cross_entropy(model(x), y).backward()
        for n, l in zip(names, layers):
            stats[f"{n}.weight.grad"]   += l.weight.grad.detach()
            stats[f"{n}.weight.fisher"] += l.weight.grad.detach() ** 2
            if l.bias is not None and l.bias.grad is not None and l.bias_mask is not None:
                stats[f"{n}.bias.grad"]   += l.bias.grad.detach()
                stats[f"{n}.bias.fisher"] += l.bias.grad.detach() ** 2
        batches += 1
        if batches >= cfg.fisher_batches:
            break
    for k in stats:
        stats[k] /= max(1, batches)
    return stats


def glauber_sweep(model, stats, cfg, round_idx):
    beta = cfg.beta_at(round_idx)
    results = {}
    layer_cfg = [
        ("fc1", model.fc1, cfg.rho_w[0], cfg.rho_b[0]),
        ("fc2", model.fc2, cfg.rho_w[1], cfg.rho_b[1]),
        ("fc3", model.fc3, cfg.rho_w[2], None),
    ]
    for name, layer, rho_w, rho_b in layer_cfg:
        eff_w   = layer.effective_weight()
        fish_w  = stats[f"{name}.weight.fisher"] + cfg.eps_curv
        grad_w  = stats[f"{name}.weight.grad"]
        active  = layer.weight_mask.bool()

        delta = torch.empty_like(layer.weight_mask)
        delta[active]  = -0.5 * rho_w + 0.5 * fish_w[active]  * eff_w[active] ** 2
        if cfg.allow_regrowth:
            delta[~active] = 0.5 * rho_w - 0.5 * grad_w[~active] ** 2 / fish_w[~active]
        else:
            delta[~active] = float("inf")

        if math.isinf(beta):
            p = (delta < 0).float()
        else:
            p = torch.sigmoid(-beta * delta)

        flips = torch.rand_like(p) < p.clamp(0, 1)
        if cfg.max_flip_fraction is not None:
            max_k = int(cfg.max_flip_fraction * flips.numel())
            if flips.sum() > max_k:
                idx  = torch.nonzero(flips.flatten(), as_tuple=False).flatten()
                keep = idx[torch.randperm(idx.numel(), device=idx.device)[:max_k]]
                flips = torch.zeros_like(flips.flatten(), dtype=torch.bool)
                flips[keep] = True
                flips = flips.view_as(layer.weight_mask)
        layer.weight_mask[flips] = 1.0 - layer.weight_mask[flips]
        layer.weight.data.mul_(layer.weight_mask)

        results[f"{name}.pruned"]  = int((flips & active).sum())
        results[f"{name}.regrown"] = int((flips & ~active).sum())

        if layer.bias is not None and layer.bias_mask is not None and rho_b is not None:
            eff_b  = layer.effective_bias()
            fish_b = stats[f"{name}.bias.fisher"] + cfg.eps_curv
            grad_b = stats[f"{name}.bias.grad"]
            act_b  = layer.bias_mask.bool()
            db     = torch.empty_like(layer.bias_mask)
            db[act_b]  = -0.5 * rho_b + 0.5 * fish_b[act_b]  * eff_b[act_b] ** 2
            if cfg.allow_regrowth:
                db[~act_b] = 0.5 * rho_b - 0.5 * grad_b[~act_b] ** 2 / fish_b[~act_b]
            else:
                db[~act_b] = float("inf")
            if math.isinf(beta):
                pb = (db < 0).float()
            else:
                pb = torch.sigmoid(-beta * db)
            fb = torch.rand_like(pb) < pb.clamp(0, 1)
            layer.bias_mask[fb] = 1.0 - layer.bias_mask[fb]
            layer.bias.data.mul_(layer.bias_mask)

    results.update(model.cleanup_dead_neurons_())
    results["total_pruned"]  = sum(v for k, v in results.items() if k.endswith(".pruned"))
    results["total_regrown"] = sum(v for k, v in results.items() if k.endswith(".regrown"))
    return results


def global_sparsity(model):
    masks  = [model.fc1.weight_mask, model.fc2.weight_mask, model.fc3.weight_mask,
              model.fc1.bias_mask, model.fc2.bias_mask]
    total  = sum(m.numel() for m in masks)
    active = sum(int(m.sum()) for m in masks)
    return 1.0 - active / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = Cfg()
    set_seed(cfg.seed)

    transform    = transforms.Compose([transforms.ToTensor()])
    train_ds     = datasets.MNIST(cfg.data_root, train=True,  download=True, transform=transform)
    test_ds      = datasets.MNIST(cfg.data_root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=2)
    # Separate non-shuffled loader for clean CE eval on training set
    train_eval_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    model = MaskedLeNet300100().to(cfg.device)
    print(f"Device: {cfg.device} | sweeps/round: {cfg.sweeps_per_round} | T_start: {cfg.T_start}")
    print("Gap = train_ce - test_ce  (CE only, no L2, separate eval pass)")

    records = []

    # Round 0 — pretrain
    train_model(model, train_loader, cfg, cfg.train_epochs_per_round)
    train_ce, train_acc = eval_ce(model, train_eval_loader, cfg.device)
    test_ce,  test_acc  = eval_ce(model, test_loader,       cfg.device)
    E, _, l2, rho = energy(model, test_loader, cfg)
    spar = global_sparsity(model)
    gap  = train_ce - test_ce
    rec  = {
        "round": 0, "T": float("nan"),
        "train_ce": train_ce, "train_acc": train_acc,
        "test_ce":  test_ce,  "test_acc":  test_acc,
        "gap": gap,
        "energy": E, "l2": l2, "rho": rho,
        "sparsity": spar, "dead": 0,
        "pruned": 0, "regrown": 0,
    }
    records.append(rec)
    print(f"R00 | spar={spar:.4f} | acc={test_acc:.4f} | "
          f"train_ce={train_ce:.4f} | test_ce={test_ce:.4f} | gap={gap:+.4f}")

    for r in range(1, cfg.max_rounds + 1):
        T = cfg.T_at(r)

        # Pre-sweep training
        train_model(model, train_loader, cfg, cfg.train_epochs_per_round)

        # Glauber sweeps
        agg = {"total_pruned": 0, "total_regrown": 0, "dead_fc1": 0, "dead_fc2": 0}
        for _ in range(cfg.sweeps_per_round):
            st = estimate_fisher(model, train_loader, cfg)
            sw = glauber_sweep(model, st, cfg, r)
            for k in agg:
                agg[k] += sw.get(k, 0)

        # Post-sweep fine-tune
        train_model(model, train_loader, cfg, cfg.finetune_epochs)

        # Clean eval — CE only, no L2, separate passes
        train_ce, train_acc = eval_ce(model, train_eval_loader, cfg.device)
        test_ce,  test_acc  = eval_ce(model, test_loader,       cfg.device)
        E, _, l2, rho = energy(model, test_loader, cfg)
        spar = global_sparsity(model)
        dead = agg["dead_fc1"] + agg["dead_fc2"]
        gap  = train_ce - test_ce

        rec = {
            "round": r, "T": T,
            "train_ce": train_ce, "train_acc": train_acc,
            "test_ce":  test_ce,  "test_acc":  test_acc,
            "gap": gap,
            "energy": E, "l2": l2, "rho": rho,
            "sparsity": spar, "dead": dead,
            "pruned": agg["total_pruned"], "regrown": agg["total_regrown"],
        }
        records.append(rec)
        print(f"R{r:02d} | T={T:.2e} | spar={spar:.4f} | acc={test_acc:.4f} | "
              f"train_ce={train_ce:.4f} | test_ce={test_ce:.4f} | gap={gap:+.4f} | "
              f"prune={agg['total_pruned']} regrow={agg['total_regrown']} dead={dead}")

        if spar >= cfg.target_sparsity:
            print(f"Target sparsity {cfg.target_sparsity} reached at round {r}.")
            break

    os.makedirs(os.path.dirname(cfg.out_path), exist_ok=True)
    with open(cfg.out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\nResults saved → {cfg.out_path}")


if __name__ == "__main__":
    main()
