import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from torchvision import datasets, transforms
    from torchvision.models import vgg16, VGG16_Weights
except Exception:
    datasets = None
    transforms = None
    vgg16 = None
    VGG16_Weights = None


# -----------------------------
# Masked layers
# -----------------------------
class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, prune_bias: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))
        if self.bias is not None and prune_bias:
            self.register_buffer("bias_mask", torch.ones_like(self.bias))
        else:
            self.bias_mask = None

    def effective_weight(self) -> torch.Tensor:
        return self.weight * self.weight_mask

    def effective_bias(self) -> Optional[torch.Tensor]:
        if self.bias is None:
            return None
        if self.bias_mask is None:
            return self.bias
        return self.bias * self.bias_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            x,
            self.effective_weight(),
            self.effective_bias(),
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    @torch.no_grad()
    def prune_weights_(self, prune_mask: torch.Tensor) -> int:
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
        if self.bias is not None and self.bias_mask is not None and self.bias.grad is not None:
            self.bias.grad.mul_(self.bias_mask)


class MaskedLinear(nn.Linear):
    def __init__(self, *args, prune_bias: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))
        if self.bias is not None and prune_bias:
            self.register_buffer("bias_mask", torch.ones_like(self.bias))
        else:
            self.bias_mask = None

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
        if self.bias is not None and self.bias_mask is not None and self.bias.grad is not None:
            self.bias.grad.mul_(self.bias_mask)


# -----------------------------
# Config
# -----------------------------
@dataclass
class VGGPruningConfig:
    device: str = "cpu"
    seed: int = 0
    num_classes: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    train_epochs_per_round: int = 1
    finetune_epochs: int = 1
    max_pruning_rounds: int = 25
    fisher_batches: int = 3
    eta_w: float = 1e-5
    eta_b: float = 1e-5
    rho_conv_w: float = 1e-8
    rho_conv_b: float = 1e-8
    rho_fc_w: Tuple[float, float, float] = (5e-8, 5e-8, 1e-8)
    rho_fc_b: Tuple[float, float] = (5e-8, 5e-8)  # output bias unpruned
    prune_fraction_cap: Optional[float] = 0.15
    target_global_sparsity: Optional[float] = 0.90
    data_root: str = "./data"
    use_pretrained: bool = True
    num_workers: int = 2
    image_size: int = 224


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pin_memory_for_device(device: str) -> bool:
    return device.startswith("cuda")


def build_cifar10_loaders(cfg: VGGPruningConfig) -> Tuple[DataLoader, DataLoader]:
    if datasets is None or transforms is None:
        raise RuntimeError("torchvision is required.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    train_ds = datasets.CIFAR10(root=cfg.data_root, train=True, download=True, transform=train_tf)
    test_ds = datasets.CIFAR10(root=cfg.data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=_pin_memory_for_device(cfg.device),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=_pin_memory_for_device(cfg.device),
    )
    return train_loader, test_loader


# -----------------------------
# Build masked VGG16
# -----------------------------
def make_masked_vgg16(cfg: VGGPruningConfig) -> nn.Module:
    if vgg16 is None:
        raise RuntimeError("torchvision.models.vgg16 is required.")

    weights = VGG16_Weights.DEFAULT if cfg.use_pretrained else None
    model = vgg16(weights=weights)

    # Replace feature convs
    new_features: List[nn.Module] = []
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            m = MaskedConv2d(
                layer.in_channels,
                layer.out_channels,
                kernel_size=layer.kernel_size,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
                bias=(layer.bias is not None),
                padding_mode=layer.padding_mode,
                prune_bias=True,
            )
            m.weight.data.copy_(layer.weight.data)
            if layer.bias is not None:
                m.bias.data.copy_(layer.bias.data)
            new_features.append(m)
        else:
            new_features.append(copy.deepcopy(layer))
    model.features = nn.Sequential(*new_features)

    # Replace classifier linears, keep last bias unpruned
    new_classifier: List[nn.Module] = []
    linears = [i for i, layer in enumerate(model.classifier) if isinstance(layer, nn.Linear)]
    last_linear_idx = linears[-1]
    for i, layer in enumerate(model.classifier):
        if isinstance(layer, nn.Linear):
            prune_bias = i != last_linear_idx
            m = MaskedLinear(
                layer.in_features,
                layer.out_features,
                bias=(layer.bias is not None),
                prune_bias=prune_bias,
            )
            m.weight.data.copy_(layer.weight.data)
            if layer.bias is not None:
                m.bias.data.copy_(layer.bias.data)
            new_classifier.append(m)
        else:
            new_classifier.append(copy.deepcopy(layer))
    model.classifier = nn.Sequential(*new_classifier)

    # Adapt final classifier to CIFAR-10 if requested.
    if cfg.num_classes != 1000:
        old_last = model.classifier[6]
        assert isinstance(old_last, MaskedLinear)
        new_last = MaskedLinear(old_last.in_features, cfg.num_classes, bias=True, prune_bias=False)
        nn.init.normal_(new_last.weight, 0, 0.01)
        nn.init.constant_(new_last.bias, 0)
        model.classifier[6] = new_last

    return model


def masked_modules(model: nn.Module) -> List[nn.Module]:
    mods = []
    for m in model.modules():
        if isinstance(m, (MaskedConv2d, MaskedLinear)):
            mods.append(m)
    return mods


def named_masked_modules(model: nn.Module):
    for name, m in model.named_modules():
        if isinstance(m, (MaskedConv2d, MaskedLinear)):
            yield name, m


# -----------------------------
# Training / eval
# -----------------------------
def masked_l2_penalty(model: nn.Module, cfg: VGGPruningConfig) -> torch.Tensor:
    device = next(model.parameters()).device
    penalty = torch.zeros((), device=device)
    for m in masked_modules(model):
        penalty = penalty + 0.5 * cfg.eta_w * (m.effective_weight() ** 2).sum()
        b = m.effective_bias()
        if b is not None:
            penalty = penalty + 0.5 * cfg.eta_b * (b ** 2).sum()
    return penalty


def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, cfg: VGGPruningConfig) -> torch.Tensor:
    logits = model(x)
    ce = F.cross_entropy(logits, y)
    reg = masked_l2_penalty(model, cfg)
    return ce + reg


def train_model(model: nn.Module, loader: DataLoader, cfg: VGGPruningConfig, epochs: int) -> None:
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    for _ in range(epochs):
        for x, y in loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model, x, y, cfg)
            loss.backward()
            for m in masked_modules(model):
                m.zero_masked_grads_()
            optimizer.step()
            with torch.no_grad():
                for m in masked_modules(model):
                    m.weight.data.mul_(m.weight_mask)
                    if m.bias is not None and m.bias_mask is not None:
                        m.bias.data.mul_(m.bias_mask)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[float, float]:
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


# -----------------------------
# Fisher / saliency
# -----------------------------
def estimate_diag_fisher(model: nn.Module, loader: DataLoader, cfg: VGGPruningConfig) -> Dict[str, torch.Tensor]:
    model.train()
    stats: Dict[str, torch.Tensor] = {}
    for name, m in named_masked_modules(model):
        stats[f"{name}.weight"] = torch.zeros_like(m.weight)
        if m.bias is not None:
            stats[f"{name}.bias"] = torch.zeros_like(m.bias)

    batches = 0
    for x, y in loader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        model.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        for name, m in named_masked_modules(model):
            stats[f"{name}.weight"] += m.weight.grad.detach() ** 2
            if m.bias is not None and m.bias.grad is not None:
                stats[f"{name}.bias"] += m.bias.grad.detach() ** 2
        batches += 1
        torch.cuda.empty_cache()
        if batches >= cfg.fisher_batches:
            break

    for k in stats:
        stats[k] /= max(1, batches)
    return stats


def _fraction_threshold(active_scores: torch.Tensor, fraction_cap: Optional[float]) -> Optional[torch.Tensor]:
    if fraction_cap is None or active_scores.numel() == 0:
        return None
    k = int(fraction_cap * active_scores.numel())
    if k <= 0:
        return None
    values, _ = torch.kthvalue(active_scores, k)
    return values


# -----------------------------
# Dead unit detection and cleanup
# -----------------------------
@torch.no_grad()
def alive_out_channels(conv: MaskedConv2d) -> torch.Tensor:
    dead = (conv.weight_mask.sum(dim=(1, 2, 3)) == 0)
    if conv.bias_mask is not None:
        dead &= (conv.bias_mask == 0)
    return (~dead).nonzero(as_tuple=False).squeeze(1)


@torch.no_grad()
def alive_linear_neurons(fc: MaskedLinear) -> torch.Tensor:
    dead = (fc.weight_mask.sum(dim=1) == 0)
    if fc.bias_mask is not None:
        dead &= (fc.bias_mask == 0)
    return (~dead).nonzero(as_tuple=False).squeeze(1)


@torch.no_grad()
def cleanup_dead_structure_(model: nn.Module) -> Dict[str, int]:
    stats = {
        "dead_conv_channels": 0,
        "zeroed_next_conv_inputs": 0,
        "zeroed_fc1_flat_inputs": 0,
        "dead_fc1": 0,
        "dead_fc2": 0,
        "zeroed_fc2_inputs": 0,
        "zeroed_fc3_inputs": 0,
    }

    convs: List[MaskedConv2d] = [m for m in model.features if isinstance(m, MaskedConv2d)]
    linears: List[MaskedLinear] = [m for m in model.classifier if isinstance(m, MaskedLinear)]
    fc1, fc2, fc3 = linears

    # Propagate dead conv outputs into next conv inputs.
    for i, conv in enumerate(convs):
        alive = alive_out_channels(conv)
        dead_mask = torch.ones(conv.out_channels, dtype=torch.bool, device=conv.weight.device)
        dead_mask[alive] = False
        stats["dead_conv_channels"] += int(dead_mask.sum().item())

        if i + 1 < len(convs):
            nxt = convs[i + 1]
            if dead_mask.any():
                prune_mask = dead_mask.view(1, -1, 1, 1).expand_as(nxt.weight_mask)
                stats["zeroed_next_conv_inputs"] += nxt.prune_weights_(prune_mask)
        else:
            # Last conv feeds fc1 through 7x7 flatten blocks.
            dead_channels = dead_mask.nonzero(as_tuple=False).squeeze(1)
            for c in dead_channels.tolist():
                cols = slice(c * 49, (c + 1) * 49)
                prune_mask = torch.zeros_like(fc1.weight_mask, dtype=torch.bool)
                prune_mask[:, cols] = True
                stats["zeroed_fc1_flat_inputs"] += fc1.prune_weights_(prune_mask)

    # Propagate dead fc neurons forward.
    alive1 = alive_linear_neurons(fc1)
    dead1 = torch.ones(fc1.out_features, dtype=torch.bool, device=fc1.weight.device)
    dead1[alive1] = False
    stats["dead_fc1"] = int(dead1.sum().item())
    if dead1.any():
        prune_mask = dead1.unsqueeze(0).expand_as(fc2.weight_mask)
        stats["zeroed_fc2_inputs"] += fc2.prune_weights_(prune_mask)

    alive2 = alive_linear_neurons(fc2)
    dead2 = torch.ones(fc2.out_features, dtype=torch.bool, device=fc2.weight.device)
    dead2[alive2] = False
    stats["dead_fc2"] = int(dead2.sum().item())
    if dead2.any():
        prune_mask = dead2.unsqueeze(0).expand_as(fc3.weight_mask)
        stats["zeroed_fc3_inputs"] += fc3.prune_weights_(prune_mask)

    return stats


# -----------------------------
# Pruning step
# -----------------------------
@torch.no_grad()
def prune_round(model: nn.Module, fisher: Dict[str, torch.Tensor], cfg: VGGPruningConfig) -> Dict[str, int]:
    stats: Dict[str, int] = {}

    conv_index = 0
    fc_index = 0
    for name, m in named_masked_modules(model):
        if isinstance(m, MaskedConv2d):
            rho_w = cfg.rho_conv_w
            rho_b = cfg.rho_conv_b
            eff_w = m.effective_weight()
            sal_w = 0.5 * fisher[f"{name}.weight"] * (eff_w ** 2)
            active_w = m.weight_mask.bool()
            cand_w = active_w & (sal_w < (rho_w / 2.0))
            if cfg.prune_fraction_cap is not None and cand_w.any():
                thr = _fraction_threshold(sal_w[active_w], cfg.prune_fraction_cap)
                if thr is not None:
                    cand_w &= (sal_w <= thr)
            stats[f"conv{conv_index}.weights_pruned"] = m.prune_weights_(cand_w)

            if m.bias_mask is not None:
                eff_b = m.effective_bias()
                sal_b = 0.5 * fisher[f"{name}.bias"] * (eff_b ** 2)
                active_b = m.bias_mask.bool()
                cand_b = active_b & (sal_b < (rho_b / 2.0))
                if cfg.prune_fraction_cap is not None and cand_b.any():
                    thr = _fraction_threshold(sal_b[active_b], cfg.prune_fraction_cap)
                    if thr is not None:
                        cand_b &= (sal_b <= thr)
                stats[f"conv{conv_index}.biases_pruned"] = m.prune_biases_(cand_b)
            conv_index += 1
        else:
            rho_w = cfg.rho_fc_w[fc_index]
            rho_b = cfg.rho_fc_b[fc_index] if fc_index < 2 else None
            eff_w = m.effective_weight()
            sal_w = 0.5 * fisher[f"{name}.weight"] * (eff_w ** 2)
            active_w = m.weight_mask.bool()
            cand_w = active_w & (sal_w < (rho_w / 2.0))
            if cfg.prune_fraction_cap is not None and cand_w.any():
                thr = _fraction_threshold(sal_w[active_w], cfg.prune_fraction_cap)
                if thr is not None:
                    cand_w &= (sal_w <= thr)
            stats[f"fc{fc_index + 1}.weights_pruned"] = m.prune_weights_(cand_w)

            if m.bias_mask is not None and rho_b is not None:
                eff_b = m.effective_bias()
                sal_b = 0.5 * fisher[f"{name}.bias"] * (eff_b ** 2)
                active_b = m.bias_mask.bool()
                cand_b = active_b & (sal_b < (rho_b / 2.0))
                if cfg.prune_fraction_cap is not None and cand_b.any():
                    thr = _fraction_threshold(sal_b[active_b], cfg.prune_fraction_cap)
                    if thr is not None:
                        cand_b &= (sal_b <= thr)
                stats[f"fc{fc_index + 1}.biases_pruned"] = m.prune_biases_(cand_b)
            fc_index += 1

    stats.update(cleanup_dead_structure_(model))
    return stats


# -----------------------------
# Reports
# -----------------------------
@torch.no_grad()
def sparsity_report(model: nn.Module) -> Dict[str, float]:
    total_active = 0
    total_prunable = 0
    report: Dict[str, float] = {}

    conv_index = 0
    fc_index = 0
    for _, m in named_masked_modules(model):
        active_w = int(m.weight_mask.sum().item())
        total_w = m.weight_mask.numel()
        total_active += active_w
        total_prunable += total_w
        if isinstance(m, MaskedConv2d):
            report[f"conv{conv_index}.weight_sparsity"] = 1.0 - active_w / total_w
            if m.bias_mask is not None:
                active_b = int(m.bias_mask.sum().item())
                total_b = m.bias_mask.numel()
                total_active += active_b
                total_prunable += total_b
                report[f"conv{conv_index}.bias_sparsity"] = 1.0 - active_b / total_b
            conv_index += 1
        else:
            report[f"fc{fc_index + 1}.weight_sparsity"] = 1.0 - active_w / total_w
            if m.bias_mask is not None:
                active_b = int(m.bias_mask.sum().item())
                total_b = m.bias_mask.numel()
                total_active += active_b
                total_prunable += total_b
                report[f"fc{fc_index + 1}.bias_sparsity"] = 1.0 - active_b / total_b
            fc_index += 1

    report["global_prunable_sparsity"] = 1.0 - total_active / total_prunable
    return report


# -----------------------------
# Compression
# -----------------------------
class CompressedVGG16(nn.Module):
    def __init__(self, features: nn.Sequential, classifier: nn.Sequential, num_classes: int):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = classifier
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@torch.no_grad()
def expand_channel_indices_to_flat(channel_idx: torch.Tensor, spatial: int = 49) -> torch.Tensor:
    cols = []
    for c in channel_idx.tolist():
        start = c * spatial
        cols.extend(range(start, start + spatial))
    return torch.tensor(cols, dtype=torch.long, device=channel_idx.device)


@torch.no_grad()
def compress_vgg16_features(masked_vgg: nn.Module):
    old_layers = list(masked_vgg.features)
    new_layers: List[nn.Module] = []

    prev_alive = torch.arange(3, device=next(masked_vgg.parameters()).device)
    conv_alive_lists: List[torch.Tensor] = []

    for layer in old_layers:
        if not isinstance(layer, MaskedConv2d):
            new_layers.append(copy.deepcopy(layer))
            continue

        alive_out = alive_out_channels(layer)
        conv_alive_lists.append(alive_out.clone())

        W = layer.effective_weight()[alive_out][:, prev_alive, :, :].clone()
        b = None
        if layer.bias is not None:
            b = layer.effective_bias()[alive_out].clone()

        new_conv = nn.Conv2d(
            in_channels=len(prev_alive),
            out_channels=len(alive_out),
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            groups=1,
            bias=(b is not None),
            padding_mode=layer.padding_mode,
        ).to(W.device)
        new_conv.weight.copy_(W)
        if b is not None:
            new_conv.bias.copy_(b)

        new_layers.append(new_conv)
        prev_alive = alive_out

    return nn.Sequential(*new_layers), conv_alive_lists


@torch.no_grad()
def compress_vgg16_classifier(masked_vgg: nn.Module, last_alive_channels: torch.Tensor):
    fc1_old = masked_vgg.classifier[0]
    relu1 = copy.deepcopy(masked_vgg.classifier[1])
    drop1 = copy.deepcopy(masked_vgg.classifier[2])
    fc2_old = masked_vgg.classifier[3]
    relu2 = copy.deepcopy(masked_vgg.classifier[4])
    drop2 = copy.deepcopy(masked_vgg.classifier[5])
    fc3_old = masked_vgg.classifier[6]

    assert isinstance(fc1_old, MaskedLinear)
    assert isinstance(fc2_old, MaskedLinear)
    assert isinstance(fc3_old, MaskedLinear)

    alive_fc1 = alive_linear_neurons(fc1_old)
    alive_fc2 = alive_linear_neurons(fc2_old)
    flat_cols = expand_channel_indices_to_flat(last_alive_channels, spatial=49)

    W1 = fc1_old.effective_weight()[alive_fc1][:, flat_cols].clone()
    b1 = fc1_old.effective_bias()[alive_fc1].clone()

    W2 = fc2_old.effective_weight()[alive_fc2][:, alive_fc1].clone()
    b2 = fc2_old.effective_bias()[alive_fc2].clone()

    W3 = fc3_old.effective_weight()[:, alive_fc2].clone()
    b3 = fc3_old.bias.clone()

    fc1_new = nn.Linear(W1.shape[1], W1.shape[0], bias=True).to(W1.device)
    fc2_new = nn.Linear(W2.shape[1], W2.shape[0], bias=True).to(W2.device)
    fc3_new = nn.Linear(W3.shape[1], W3.shape[0], bias=True).to(W3.device)
    fc1_new.weight.copy_(W1); fc1_new.bias.copy_(b1)
    fc2_new.weight.copy_(W2); fc2_new.bias.copy_(b2)
    fc3_new.weight.copy_(W3); fc3_new.bias.copy_(b3)

    classifier = nn.Sequential(fc1_new, relu1, drop1, fc2_new, relu2, drop2, fc3_new)
    return classifier, alive_fc1, alive_fc2


@torch.no_grad()
def compress_masked_vgg16(masked_vgg: nn.Module, num_classes: int):
    features, conv_alive_lists = compress_vgg16_features(masked_vgg)
    last_alive = conv_alive_lists[-1]
    classifier, alive_fc1, alive_fc2 = compress_vgg16_classifier(masked_vgg, last_alive)
    compact = CompressedVGG16(features, classifier, num_classes=num_classes).to(next(masked_vgg.parameters()).device)
    meta = {
        "conv_alive_lists": conv_alive_lists,
        "alive_fc1": alive_fc1,
        "alive_fc2": alive_fc2,
    }
    return compact, meta


# -----------------------------
# Main experiment
# -----------------------------
def run_pruning_experiment(cfg: VGGPruningConfig):
    set_seed(cfg.seed)
    train_loader, test_loader = build_cifar10_loaders(cfg)
    model = make_masked_vgg16(cfg).to(cfg.device)

    print("Pretraining...")
    train_model(model, train_loader, cfg, epochs=max(1, cfg.train_epochs_per_round))
    test_loss, test_acc = evaluate(model, test_loader, cfg.device)
    report = sparsity_report(model)
    print(f"Round 0 | test loss={test_loss:.4f} | test acc={test_acc:.4f} | sparsity={report['global_prunable_sparsity']:.4f}")

    for round_idx in range(1, cfg.max_pruning_rounds + 1):
        fisher = estimate_diag_fisher(model, train_loader, cfg)
        stats = prune_round(model, fisher, cfg)
        train_model(model, train_loader, cfg, epochs=cfg.finetune_epochs)
        test_loss, test_acc = evaluate(model, test_loader, cfg.device)
        report = sparsity_report(model)
        print(f"Round {round_idx} | test loss={test_loss:.4f} | test acc={test_acc:.4f} | sparsity={report['global_prunable_sparsity']:.4f}")
        print(f"  stats: {stats}")

        if cfg.target_global_sparsity is not None and report["global_prunable_sparsity"] >= cfg.target_global_sparsity:
            break
        if all(v == 0 for k, v in stats.items() if "pruned" in k):
            break

    compact, meta = compress_masked_vgg16(model, num_classes=cfg.num_classes)
    compact_loss, compact_acc = evaluate(compact, test_loader, cfg.device)
    print("Compressed model:")
    print(f"  test loss={compact_loss:.4f} | test acc={compact_acc:.4f}")
    print(f"  surviving channels in last conv block: {len(meta['conv_alive_lists'][-1])}")
    print(f"  surviving fc neurons: {len(meta['alive_fc1'])}, {len(meta['alive_fc2'])}")

    save_obj = {
        "masked_state_dict": model.state_dict(),
        "compact_state_dict": compact.state_dict(),
        "meta": {k: [t.cpu() for t in v] if isinstance(v, list) else v.cpu() for k, v in meta.items()},
        "config": cfg.__dict__,
    }
    out_path = "vgg16_pruned_and_compressed.pt"
    torch.save(save_obj, out_path)
    print(f"Saved checkpoint to {out_path}")
    return model, compact, meta


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    cfg = VGGPruningConfig(device=device)
    run_pruning_experiment(cfg)
