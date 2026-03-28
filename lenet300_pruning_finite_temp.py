import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
except Exception:
    datasets = None
    transforms = None


class MaskedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, prune_bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.register_buffer("weight_mask", torch.ones(out_features, in_features))
        if bias and prune_bias:
            self.register_buffer("bias_mask", torch.ones(out_features))
        else:
            self.bias_mask = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.weight_mask

    def effective_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        return self.bias if self.bias_mask is None else self.bias * self.bias_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.effective_bias())

    @torch.no_grad()
    def zero_masked_parameters_(self) -> None:
        self.weight.data.mul_(self.weight_mask)
        if self.bias is not None and self.bias_mask is not None:
            self.bias.data.mul_(self.bias_mask)

    @torch.no_grad()
    def zero_masked_grads_(self) -> None:
        if self.weight.grad is not None:
            self.weight.grad.mul_(self.weight_mask)
        if self.bias is not None and self.bias.grad is not None and self.bias_mask is not None:
            self.bias.grad.mul_(self.bias_mask)


class MaskedLeNet300100(nn.Module):
    def __init__(self, activation: str = "relu"):
        super().__init__()
        self.fc1 = MaskedLinear(28 * 28, 300, bias=True, prune_bias=True)
        self.fc2 = MaskedLinear(300, 100, bias=True, prune_bias=True)
        self.fc3 = MaskedLinear(100, 10, bias=True, prune_bias=False)
        if activation == "relu":
            self.phi = F.relu
        elif activation == "tanh":
            self.phi = torch.tanh
        else:
            raise ValueError("activation must be 'relu' or 'tanh'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        z1 = self.phi(self.fc1(x))
        z2 = self.phi(self.fc2(z1))
        return self.fc3(z2)

    def layers(self):
        return [self.fc1, self.fc2, self.fc3]

    @torch.no_grad()
    def cleanup_dead_neurons_(self) -> Dict[str, int]:
        stats = {"dead_fc1": 0, "dead_fc2": 0, "pruned_outgoing_fc2": 0, "pruned_outgoing_fc3": 0}
        dead_fc1 = (self.fc1.weight_mask.sum(dim=1) == 0)
        if self.fc1.bias_mask is not None:
            dead_fc1 &= (self.fc1.bias_mask == 0)
        stats["dead_fc1"] = int(dead_fc1.sum().item())
        if dead_fc1.any():
            cols = dead_fc1.unsqueeze(0).expand_as(self.fc2.weight_mask)
            newly = self.fc2.weight_mask.bool() & cols
            self.fc2.weight_mask[newly] = 0.0
            self.fc2.weight.data[newly] = 0.0
            stats["pruned_outgoing_fc2"] = int(newly.sum().item())

        dead_fc2 = (self.fc2.weight_mask.sum(dim=1) == 0)
        if self.fc2.bias_mask is not None:
            dead_fc2 &= (self.fc2.bias_mask == 0)
        stats["dead_fc2"] = int(dead_fc2.sum().item())
        if dead_fc2.any():
            cols = dead_fc2.unsqueeze(0).expand_as(self.fc3.weight_mask)
            newly = self.fc3.weight_mask.bool() & cols
            self.fc3.weight_mask[newly] = 0.0
            self.fc3.weight.data[newly] = 0.0
            stats["pruned_outgoing_fc3"] = int(newly.sum().item())
        return stats


@dataclass
class PruningConfig:
    device: str = "cpu"
    lr: float = 1e-3
    batch_size: int = 128
    train_epochs_per_round: int = 2
    finetune_epochs: int = 2
    max_rounds: int = 20
    fisher_batches: int = 50
    eta_w: float = 1e-4
    eta_b: float = 1e-4
    rho_w: Tuple[float, float, float] = (1e-7, 2e-7, 5e-8)
    rho_b: Tuple[float, float] = (1e-7, 2e-7)
    activation: str = "relu"
    seed: int = 0
    target_global_sparsity: Optional[float] = 0.95
    pin_memory: bool = False
    # Finite-temperature pruning dynamics
    T_h: float = 1e-7
    sweeps_per_round: int = 1
    max_flip_fraction_per_sweep: Optional[float] = 0.2
    allow_regrowth: bool = True
    eps_curv: float = 1e-12

    @property
    def beta_h(self) -> float:
        if self.T_h <= 0.0:
            return float("inf")
        return 1.0 / self.T_h


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_l2_penalty(model: MaskedLeNet300100, cfg: PruningConfig) -> torch.Tensor:
    device = next(model.parameters()).device
    out = torch.zeros((), device=device)
    out = out + 0.5 * cfg.eta_w * sum((layer.effective_weight() ** 2).sum() for layer in model.layers())
    out = out + 0.5 * cfg.eta_b * (
        (model.fc1.effective_bias() ** 2).sum() + (model.fc2.effective_bias() ** 2).sum() + (model.fc3.bias ** 2).sum()
    )
    return out


def compute_loss(model: MaskedLeNet300100, x: torch.Tensor, y: torch.Tensor, cfg: PruningConfig) -> torch.Tensor:
    return F.cross_entropy(model(x), y) + masked_l2_penalty(model, cfg)


def train_model(model: MaskedLeNet300100, loader: DataLoader, cfg: PruningConfig, epochs: int) -> None:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            opt.zero_grad(set_to_none=True)
            loss = compute_loss(model, x, y, cfg)
            loss.backward()
            for layer in model.layers():
                layer.zero_masked_grads_()
            opt.step()
            with torch.no_grad():
                for layer in model.layers():
                    layer.zero_masked_parameters_()


@torch.no_grad()
def evaluate(model: MaskedLeNet300100, loader: DataLoader, device: str):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, total_correct / total


def estimate_grad_and_fisher(model: MaskedLeNet300100, loader: DataLoader, cfg: PruningConfig) -> Dict[str, torch.Tensor]:
    model.train()
    stats = {
        "fc1.weight.grad": torch.zeros_like(model.fc1.weight),
        "fc1.weight.fisher": torch.zeros_like(model.fc1.weight),
        "fc1.bias.grad": torch.zeros_like(model.fc1.bias),
        "fc1.bias.fisher": torch.zeros_like(model.fc1.bias),
        "fc2.weight.grad": torch.zeros_like(model.fc2.weight),
        "fc2.weight.fisher": torch.zeros_like(model.fc2.weight),
        "fc2.bias.grad": torch.zeros_like(model.fc2.bias),
        "fc2.bias.fisher": torch.zeros_like(model.fc2.bias),
        "fc3.weight.grad": torch.zeros_like(model.fc3.weight),
        "fc3.weight.fisher": torch.zeros_like(model.fc3.weight),
    }
    batches = 0
    for x, y in loader:
        x, y = x.to(cfg.device), y.to(cfg.device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        for prefix, layer in [("fc1", model.fc1), ("fc2", model.fc2), ("fc3", model.fc3)]:
            stats[f"{prefix}.weight.grad"] += layer.weight.grad.detach()
            stats[f"{prefix}.weight.fisher"] += layer.weight.grad.detach() ** 2
            if layer.bias is not None and layer.bias.grad is not None and layer.bias_mask is not None:
                stats[f"{prefix}.bias.grad"] += layer.bias.grad.detach()
                stats[f"{prefix}.bias.fisher"] += layer.bias.grad.detach() ** 2
        batches += 1
        if batches >= cfg.fisher_batches:
            break
    for k in stats:
        stats[k] /= max(1, batches)
    return stats


def _glauber_flip_probability(delta: torch.Tensor, beta_h: float) -> torch.Tensor:
    if math.isinf(beta_h):
        return (delta < 0).to(delta.dtype)
    return torch.sigmoid(-beta_h * delta)


@torch.no_grad()
def _apply_binary_mask_update(mask: torch.Tensor, proposal_prob: torch.Tensor, max_flip_fraction: Optional[float] = None) -> torch.Tensor:
    proposal_prob = proposal_prob.clamp(0.0, 1.0)
    flips = torch.rand_like(proposal_prob) < proposal_prob
    if max_flip_fraction is not None and flips.any():
        max_flips = int(max_flip_fraction * flips.numel())
        if max_flips < flips.sum().item():
            flip_idx = torch.nonzero(flips.flatten(), as_tuple=False).flatten()
            perm = flip_idx[torch.randperm(flip_idx.numel(), device=flip_idx.device)]
            keep = perm[:max_flips]
            new_flips = torch.zeros_like(flips.flatten(), dtype=torch.bool)
            new_flips[keep] = True
            flips = new_flips.view_as(flips)
    mask[flips] = 1.0 - mask[flips]
    return flips


@torch.no_grad()
def finite_temperature_mask_sweep(model: MaskedLeNet300100, stats: Dict[str, torch.Tensor], cfg: PruningConfig) -> Dict[str, int]:
    results: Dict[str, int] = {}
    beta_h = cfg.beta_h

    layer_info = [
        ("fc1", model.fc1, cfg.rho_w[0], cfg.rho_b[0]),
        ("fc2", model.fc2, cfg.rho_w[1], cfg.rho_b[1]),
        ("fc3", model.fc3, cfg.rho_w[2], None),
    ]

    for name, layer, rho_w, rho_b in layer_info:
        eff_w = layer.effective_weight()
        fisher_w = stats[f"{name}.weight.fisher"] + cfg.eps_curv
        grad_w = stats[f"{name}.weight.grad"]
        active_w = layer.weight_mask.bool()
        delta_w = torch.empty_like(layer.weight_mask)
        delta_w[active_w] = -0.5 * rho_w + 0.5 * fisher_w[active_w] * eff_w[active_w] ** 2
        grow_gain_w = 0.5 * (grad_w[~active_w] ** 2) / fisher_w[~active_w]
        delta_w[~active_w] = 0.5 * rho_w - grow_gain_w
        if not cfg.allow_regrowth:
            delta_w[~active_w] = torch.full_like(delta_w[~active_w], float("inf"))
        p_flip_w = _glauber_flip_probability(delta_w, beta_h)
        flips_w = _apply_binary_mask_update(layer.weight_mask, p_flip_w, cfg.max_flip_fraction_per_sweep)
        layer.weight.data.mul_(layer.weight_mask)
        results[f"{name}.weight_flips"] = int(flips_w.sum().item())
        results[f"{name}.weights_pruned"] = int((flips_w & active_w).sum().item())
        results[f"{name}.weights_regrown"] = int((flips_w & (~active_w)).sum().item())

        if layer.bias is not None and layer.bias_mask is not None and rho_b is not None:
            eff_b = layer.effective_bias()
            fisher_b = stats[f"{name}.bias.fisher"] + cfg.eps_curv
            grad_b = stats[f"{name}.bias.grad"]
            active_b = layer.bias_mask.bool()
            delta_b = torch.empty_like(layer.bias_mask)
            delta_b[active_b] = -0.5 * rho_b + 0.5 * fisher_b[active_b] * eff_b[active_b] ** 2
            grow_gain_b = 0.5 * (grad_b[~active_b] ** 2) / fisher_b[~active_b]
            delta_b[~active_b] = 0.5 * rho_b - grow_gain_b
            if not cfg.allow_regrowth:
                delta_b[~active_b] = torch.full_like(delta_b[~active_b], float("inf"))
            p_flip_b = _glauber_flip_probability(delta_b, beta_h)
            flips_b = _apply_binary_mask_update(layer.bias_mask, p_flip_b, cfg.max_flip_fraction_per_sweep)
            layer.bias.data.mul_(layer.bias_mask)
            results[f"{name}.bias_flips"] = int(flips_b.sum().item())
            results[f"{name}.biases_pruned"] = int((flips_b & active_b).sum().item())
            results[f"{name}.biases_regrown"] = int((flips_b & (~active_b)).sum().item())

    total_prune_flips = 0
    total_regrowth_flips = 0
    for key, value in list(results.items()):
        if key.endswith("_pruned"):
            total_prune_flips += int(value)
        elif key.endswith("_regrown"):
            total_regrowth_flips += int(value)

    results["prune_flips"] = total_prune_flips
    results["regrowth_flips"] = total_regrowth_flips
    results.update(model.cleanup_dead_neurons_())
    return results


@torch.no_grad()
def sparsity_report(model: MaskedLeNet300100) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, layer in [("fc1", model.fc1), ("fc2", model.fc2), ("fc3", model.fc3)]:
        out[f"{name}.weight_sparsity"] = 1.0 - layer.weight_mask.mean().item()
        if layer.bias_mask is not None:
            out[f"{name}.bias_sparsity"] = 1.0 - layer.bias_mask.mean().item()
    prunable_total = model.fc1.weight_mask.numel() + model.fc2.weight_mask.numel() + model.fc3.weight_mask.numel() + model.fc1.bias_mask.numel() + model.fc2.bias_mask.numel()
    prunable_active = int(model.fc1.weight_mask.sum() + model.fc2.weight_mask.sum() + model.fc3.weight_mask.sum() + model.fc1.bias_mask.sum() + model.fc2.bias_mask.sum())
    out["global_prunable_sparsity"] = 1.0 - (prunable_active / prunable_total)
    return out


def build_mnist_loaders(batch_size: int, root: str = "./data", pin_memory: bool = False):
    if datasets is None or transforms is None:
        raise RuntimeError("torchvision is required")
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin_memory)
    return train_loader, test_loader


def run_pruning_experiment(cfg: PruningConfig) -> MaskedLeNet300100:
    set_seed(cfg.seed)
    train_loader, test_loader = build_mnist_loaders(cfg.batch_size, pin_memory=cfg.pin_memory)
    model = MaskedLeNet300100(activation=cfg.activation).to(cfg.device)

    print(f"Using finite-temperature pruning dynamics with T_h={cfg.T_h} and beta_h={cfg.beta_h}")
    print("Pretraining...")
    train_model(model, train_loader, cfg, epochs=cfg.train_epochs_per_round)
    test_loss, test_acc = evaluate(model, test_loader, cfg.device)
    print(f"Round 0 | test loss={test_loss:.4f} | test acc={test_acc:.4f} | sparsity={sparsity_report(model)['global_prunable_sparsity']:.4f}")

    for round_idx in range(1, cfg.max_rounds + 1):
        train_model(model, train_loader, cfg, epochs=cfg.train_epochs_per_round)
        aggregate: Dict[str, int] = {}
        for _ in range(cfg.sweeps_per_round):
            stats = estimate_grad_and_fisher(model, train_loader, cfg)
            sweep_stats = finite_temperature_mask_sweep(model, stats, cfg)
            for k, v in sweep_stats.items():
                aggregate[k] = aggregate.get(k, 0) + int(v)
        train_model(model, train_loader, cfg, epochs=cfg.finetune_epochs)

        report = sparsity_report(model)
        test_loss, test_acc = evaluate(model, test_loader, cfg.device)
        print(f"Round {round_idx} | test loss={test_loss:.4f} | test acc={test_acc:.4f} | sparsity={report['global_prunable_sparsity']:.4f}")
        print(f"  stats: {aggregate}")
        total_flips = aggregate.get("prune_flips", 0) + aggregate.get("regrowth_flips", 0)
        if total_flips > 0:
            flip_balance = (aggregate.get("regrowth_flips", 0) - aggregate.get("prune_flips", 0)) / total_flips
        else:
            flip_balance = 0.0
        print(f"    prune flips: {aggregate.get('prune_flips', 0)}, regrowth flips: {aggregate.get('regrowth_flips', 0)}")
        print(f"    flip balance: {flip_balance:.4f}")
        print(f"  layer sparsities: {report}")

        if cfg.target_global_sparsity is not None and report["global_prunable_sparsity"] >= cfg.target_global_sparsity:
            break
        if sum(v for k, v in aggregate.items() if k.endswith("_flips")) == 0:
            break
    return model


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    cfg = PruningConfig(device=device)
    run_pruning_experiment(cfg)
