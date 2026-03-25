import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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
    """
    Linear layer with multiplicative binary masks.

    - weight mask is always present
    - bias mask is optional (used for hidden layers only in this project)
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, prune_bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prune_bias = prune_bias

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
        if self.bias_mask is None:
            return self.bias
        return self.bias * self.bias_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.effective_bias())

    @torch.no_grad()
    def prune_weights_(self, prune_mask: torch.Tensor) -> int:
        """
        prune_mask=True means set corresponding binary mask entry to zero.
        Returns number of newly pruned weights.
        """
        prune_mask = prune_mask.to(dtype=torch.bool, device=self.weight_mask.device)
        newly_pruned = (self.weight_mask.bool() & prune_mask).sum().item()
        self.weight_mask[prune_mask] = 0.0
        self.weight.data[prune_mask] = 0.0
        return int(newly_pruned)

    @torch.no_grad()
    def prune_biases_(self, prune_mask: torch.Tensor) -> int:
        if self.bias is None or self.bias_mask is None:
            return 0
        prune_mask = prune_mask.to(dtype=torch.bool, device=self.bias_mask.device)
        newly_pruned = (self.bias_mask.bool() & prune_mask).sum().item()
        self.bias_mask[prune_mask] = 0.0
        self.bias.data[prune_mask] = 0.0
        return int(newly_pruned)

    @torch.no_grad()
    def zero_masked_grads_(self) -> None:
        if self.weight.grad is not None:
            self.weight.grad.mul_(self.weight_mask)
        if self.bias is not None and self.bias.grad is not None and self.bias_mask is not None:
            self.bias.grad.mul_(self.bias_mask)

    def num_active_weights(self) -> int:
        return int(self.weight_mask.sum().item())

    def num_total_weights(self) -> int:
        return self.weight_mask.numel()

    def num_active_biases(self) -> int:
        if self.bias_mask is None:
            return 0
        return int(self.bias_mask.sum().item())

    def num_total_biases(self) -> int:
        if self.bias_mask is None:
            return 0
        return self.bias_mask.numel()


class MaskedLeNet300100(nn.Module):
    """
    LeNet-300-100 variant with phi(0)=0. Default activation is ReLU.

    Pruning policy:
    - prune all weights in fc1, fc2, fc3
    - prune biases in fc1, fc2
    - do not prune output bias in fc3
    """

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
        logits = self.fc3(z2)
        return logits

    @torch.no_grad()
    def cleanup_dead_neurons_(self) -> Dict[str, int]:
        """
        If all incoming weights and hidden bias of a neuron are pruned, the neuron is dead
        because phi(0)=0. Then all outgoing weights can be pruned too.
        """
        stats = {"dead_fc1": 0, "dead_fc2": 0, "pruned_outgoing_fc2": 0, "pruned_outgoing_fc3": 0}

        dead_fc1 = (self.fc1.weight_mask.sum(dim=1) == 0)
        if self.fc1.bias_mask is not None:
            dead_fc1 &= (self.fc1.bias_mask == 0)
        stats["dead_fc1"] = int(dead_fc1.sum().item())
        if dead_fc1.any():
            col_mask = dead_fc1.unsqueeze(0).expand_as(self.fc2.weight_mask)
            stats["pruned_outgoing_fc2"] = self.fc2.prune_weights_(col_mask)

        dead_fc2 = (self.fc2.weight_mask.sum(dim=1) == 0)
        if self.fc2.bias_mask is not None:
            dead_fc2 &= (self.fc2.bias_mask == 0)
        stats["dead_fc2"] = int(dead_fc2.sum().item())
        if dead_fc2.any():
            col_mask = dead_fc2.unsqueeze(0).expand_as(self.fc3.weight_mask)
            stats["pruned_outgoing_fc3"] = self.fc3.prune_weights_(col_mask)

        return stats

    def layers(self) -> List[MaskedLinear]:
        return [self.fc1, self.fc2, self.fc3]


@dataclass
class PruningConfig:
    device: str = "cpu"
    lr: float = 1e-3
    batch_size: int = 128
    train_epochs_per_round: int = 2
    finetune_epochs: int = 2
    max_pruning_rounds: int = 20
    fisher_batches: int = 50
    rho_w: Tuple[float, float, float] = (1e-7, 2e-7, 5e-8)
    rho_b: Tuple[float, float] = (1e-7, 2e-7)
    eta_w: float = 1e-4
    eta_b: float = 1e-4
    prune_fraction_cap: Optional[float] = 0.2
    target_global_sparsity: Optional[float] = 0.95
    activation: str = "relu"
    seed: int = 0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def masked_l2_penalty(model: MaskedLeNet300100, eta_w: float, eta_b: float) -> torch.Tensor:
    device = next(model.parameters()).device
    penalty = torch.zeros((), device=device)
    penalty = penalty + 0.5 * eta_w * sum((layer.effective_weight() ** 2).sum() for layer in model.layers())
    penalty = penalty + 0.5 * eta_b * ((model.fc1.effective_bias() ** 2).sum() + (model.fc2.effective_bias() ** 2).sum() + (model.fc3.bias ** 2).sum())
    return penalty


def compute_loss(model: MaskedLeNet300100, x: torch.Tensor, y: torch.Tensor, cfg: PruningConfig) -> torch.Tensor:
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    reg = masked_l2_penalty(model, cfg.eta_w, cfg.eta_b)
    return ce + reg


def train_model(model: MaskedLeNet300100, loader: DataLoader, cfg: PruningConfig, epochs: int) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model, x, y, cfg)
            loss.backward()
            for layer in model.layers():
                layer.zero_masked_grads_()
            optimizer.step()
            with torch.no_grad():
                for layer in model.layers():
                    layer.weight.data.mul_(layer.weight_mask)
                    if layer.bias is not None and layer.bias_mask is not None:
                        layer.bias.data.mul_(layer.bias_mask)


@torch.no_grad()
def evaluate(model: MaskedLeNet300100, loader: DataLoader, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total += y.size(0)
    return total_loss / total, total_correct / total


def estimate_diag_fisher(model: MaskedLeNet300100, loader: DataLoader, cfg: PruningConfig) -> Dict[str, torch.Tensor]:
    """
    Diagonal Fisher / Gauss-Newton proxy:
    E[g^2] over minibatches.
    """
    model.train()
    stats = {
        "fc1.weight": torch.zeros_like(model.fc1.weight),
        "fc1.bias": torch.zeros_like(model.fc1.bias),
        "fc2.weight": torch.zeros_like(model.fc2.weight),
        "fc2.bias": torch.zeros_like(model.fc2.bias),
        "fc3.weight": torch.zeros_like(model.fc3.weight),
    }

    batches = 0
    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        stats["fc1.weight"] += model.fc1.weight.grad.detach() ** 2
        stats["fc1.bias"] += model.fc1.bias.grad.detach() ** 2
        stats["fc2.weight"] += model.fc2.weight.grad.detach() ** 2
        stats["fc2.bias"] += model.fc2.bias.grad.detach() ** 2
        stats["fc3.weight"] += model.fc3.weight.grad.detach() ** 2
        batches += 1
        if batches >= cfg.fisher_batches:
            break

    for k in stats:
        stats[k] /= max(batches, 1)
    return stats


def _fraction_threshold(active_scores: torch.Tensor, fraction_cap: Optional[float]) -> Optional[torch.Tensor]:
    if fraction_cap is None or active_scores.numel() == 0:
        return None
    k = int(fraction_cap * active_scores.numel())
    if k <= 0:
        return None
    values, _ = torch.kthvalue(active_scores, k)
    return values


@torch.no_grad()
def prune_round(model: MaskedLeNet300100, fisher: Dict[str, torch.Tensor], cfg: PruningConfig) -> Dict[str, int]:
    """
    Fast pruning rule: prune active parameter q if S_q < rho_q / 2,
    where S_q = 0.5 * kappa_q * u_q^2.

    Optional fraction cap prevents pruning too many parameters in one round.
    """
    stats: Dict[str, int] = {}

    layers = [
        ("fc1", model.fc1, cfg.rho_w[0], cfg.rho_b[0]),
        ("fc2", model.fc2, cfg.rho_w[1], cfg.rho_b[1]),
        ("fc3", model.fc3, cfg.rho_w[2], None),
    ]

    for name, layer, rho_w, rho_b in layers:
        eff_w = layer.effective_weight()
        sal_w = 0.5 * fisher[f"{name}.weight"] * (eff_w ** 2)
        active_w = layer.weight_mask.bool()
        candidate_w = active_w & (sal_w < (rho_w / 2.0))
        if cfg.prune_fraction_cap is not None and candidate_w.any():
            active_scores = sal_w[active_w]
            thr = _fraction_threshold(active_scores, cfg.prune_fraction_cap)
            if thr is not None:
                candidate_w &= (sal_w <= thr)
        stats[f"{name}.weights_pruned"] = layer.prune_weights_(candidate_w)

        if layer.bias_mask is not None and rho_b is not None:
            eff_b = layer.effective_bias()
            sal_b = 0.5 * fisher[f"{name}.bias"] * (eff_b ** 2)
            active_b = layer.bias_mask.bool()
            candidate_b = active_b & (sal_b < (rho_b / 2.0))
            if cfg.prune_fraction_cap is not None and candidate_b.any():
                active_scores = sal_b[active_b]
                thr = _fraction_threshold(active_scores, cfg.prune_fraction_cap)
                if thr is not None:
                    candidate_b &= (sal_b <= thr)
            stats[f"{name}.biases_pruned"] = layer.prune_biases_(candidate_b)

    cleanup = model.cleanup_dead_neurons_()
    stats.update(cleanup)
    return stats


@torch.no_grad()
def sparsity_report(model: MaskedLeNet300100) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, layer in [("fc1", model.fc1), ("fc2", model.fc2), ("fc3", model.fc3)]:
        out[f"{name}.weight_sparsity"] = 1.0 - (layer.num_active_weights() / layer.num_total_weights())
        if layer.bias_mask is not None:
            out[f"{name}.bias_sparsity"] = 1.0 - (layer.num_active_biases() / layer.num_total_biases())
    total_active = sum(layer.num_active_weights() for layer in model.layers())
    total_weights = sum(layer.num_total_weights() for layer in model.layers())
    total_active_bias = model.fc1.num_active_biases() + model.fc2.num_active_biases()
    total_bias = model.fc1.num_total_biases() + model.fc2.num_total_biases()
    out["global_prunable_sparsity"] = 1.0 - ((total_active + total_active_bias) / (total_weights + total_bias))
    return out


def build_mnist_loaders(batch_size: int, root: str = "./data") -> Tuple[DataLoader, DataLoader]:
    if datasets is None or transforms is None:
        raise RuntimeError("torchvision is required for MNIST loading")
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


def run_pruning_experiment(cfg: PruningConfig) -> MaskedLeNet300100:
    set_seed(cfg.seed)
    train_loader, test_loader = build_mnist_loaders(cfg.batch_size)

    model = MaskedLeNet300100(activation=cfg.activation).to(cfg.device)

    print("Pretraining...")
    train_model(model, train_loader, cfg, epochs=max(1, cfg.train_epochs_per_round))
    test_loss, test_acc = evaluate(model, test_loader, cfg.device)
    print(f"Round 0 | test loss={test_loss:.4f} | test acc={test_acc:.4f} | sparsity={sparsity_report(model)['global_prunable_sparsity']:.4f}")

    for round_idx in range(1, cfg.max_pruning_rounds + 1):
        fisher = estimate_diag_fisher(model, train_loader, cfg)
        stats = prune_round(model, fisher, cfg)
        train_model(model, train_loader, cfg, epochs=cfg.finetune_epochs)

        report = sparsity_report(model)
        test_loss, test_acc = evaluate(model, test_loader, cfg.device)
        print(f"Round {round_idx} | test loss={test_loss:.4f} | test acc={test_acc:.4f} | sparsity={report['global_prunable_sparsity']:.4f}")
        print(f"  stats: {stats}")
        print(f"  layer sparsities: {report}")

        if cfg.target_global_sparsity is not None and report["global_prunable_sparsity"] >= cfg.target_global_sparsity:
            break
        if all(v == 0 for k, v in stats.items() if "pruned" in k):
            break

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = PruningConfig(device=device)
    run_pruning_experiment(cfg)
