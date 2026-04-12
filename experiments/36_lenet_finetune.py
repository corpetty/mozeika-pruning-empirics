"""
Experiment 36 — Frozen-mask fine-tune of exp 33 sparse LeNet-300-100.

Loads the exp 33 final checkpoint (97.6% sparse, 784→122→35→10 with masks).
Freezes all masks (no pruning, no regrowth).
Runs SGD + Nesterov with cosine LR decay to squeeze out remaining accuracy.
"""

import os, json, math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = "/home/petty/.openclaw/workspace-ai-research/data"
RESULTS_DIR = "/home/petty/pruning-research/results"
CKPT_IN     = os.path.join(RESULTS_DIR, "33_compact_final.pt")
CKPT_OUT    = os.path.join(RESULTS_DIR, "36_lenet_finetuned.pt")
LOG_OUT     = os.path.join(RESULTS_DIR, "36_finetune_records.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Hyper-params ──────────────────────────────────────────────────────────────
LR_MAX      = 1e-3
LR_MIN      = 1e-5
MOMENTUM    = 0.9
WD          = 1e-4
EPOCHS      = 200
BATCH_TRAIN = 128
BATCH_EVAL  = 512
CKPT_EVERY  = 50   # save intermediate checkpoint every N epochs

# ── Model definition (must match exp 33) ─────────────────────────────────────
class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, prune_bias=True):
        super().__init__(in_features, out_features, bias=bias)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))
        if bias and prune_bias:
            self.register_buffer("bias_mask", torch.ones(out_features))
        else:
            self.bias_mask = None

    def effective_weight(self):
        return self.weight * self.weight_mask

    def effective_bias(self):
        if self.bias is None:
            return None
        if self.bias_mask is not None:
            return self.bias * self.bias_mask
        return self.bias

    def forward(self, x):
        return F.linear(x, self.effective_weight(), self.effective_bias())


class CompactMaskedNet(nn.Module):
    """Same architecture as exp 33 phase-2 compact network."""
    def __init__(self, k1: int, k2: int):
        super().__init__()
        self.fc1 = MaskedLinear(784, k1, prune_bias=True)
        self.fc2 = MaskedLinear(k1,  k2, prune_bias=True)
        self.fc3 = MaskedLinear(k2,  10, prune_bias=False)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ── Data ──────────────────────────────────────────────────────────────────────
def get_loaders():
    tfm = transforms.ToTensor()   # no normalization — matches exp 33 training
    train_ds = datasets.MNIST(DATA_ROOT, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)
    train_ldr = DataLoader(train_ds, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=1, pin_memory=True)
    test_ldr  = DataLoader(test_ds,  batch_size=BATCH_EVAL,  shuffle=False, num_workers=1, pin_memory=True)
    return train_ldr, test_ldr


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    ce_sum = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        ce_sum  += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        total   += y.size(0)
    return ce_sum / total, correct / total


# ── Freeze masks ──────────────────────────────────────────────────────────────
def freeze_masks(model):
    """Make masks non-differentiable buffers (they already are) and
    register a forward hook to re-apply them after each optimizer step."""
    for layer in [model.fc1, model.fc2, model.fc3]:
        layer.weight_mask.requires_grad_(False)
        if layer.bias_mask is not None:
            layer.bias_mask.requires_grad_(False)


def clamp_to_mask(model):
    """Zero out any weights that the mask has killed (shouldn't drift, but safe)."""
    with torch.no_grad():
        for layer in [model.fc1, model.fc2, model.fc3]:
            layer.weight.data.mul_(layer.weight_mask)
            if layer.bias_mask is not None:
                layer.bias.data.mul_(layer.bias_mask)


# ── Cosine LR ─────────────────────────────────────────────────────────────────
def cosine_lr(epoch, total_epochs, lr_max, lr_min):
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * epoch / total_epochs))


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    print(f"Loading checkpoint: {CKPT_IN}")

    ckpt = torch.load(CKPT_IN, map_location="cpu")
    k1 = ckpt["k1"]
    k2 = ckpt["k2"]
    init_sparsity = ckpt["phase2_weight_sparsity"]
    print(f"Architecture: 784 → {k1} → {k2} → 10")
    print(f"Weight sparsity: {init_sparsity*100:.2f}%")

    model = CompactMaskedNet(k1, k2)
    # Load with strict=False to handle any minor key mismatches
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing:
        print(f"  Missing keys: {missing}")
    if unexpected:
        print(f"  Unexpected keys: {unexpected}")

    model.to(DEVICE)
    freeze_masks(model)

    train_ldr, test_ldr = get_loaders()

    # Baseline before fine-tuning
    base_ce, base_acc = evaluate(model, test_ldr)
    print(f"\nBaseline (loaded checkpoint): acc={base_acc*100:.2f}%  ce={base_ce:.4f}")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR_MAX, momentum=MOMENTUM,
        weight_decay=WD, nesterov=True
    )

    records = []
    best_acc = base_acc
    best_epoch = 0
    best_sd = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nFine-tuning for {EPOCHS} epochs, LR {LR_MAX}→{LR_MIN} (cosine), WD={WD}")
    print(f"{'Epoch':>6} {'LR':>8} {'Train CE':>10} {'Test CE':>9} {'Test Acc':>9} {'Best':>6}")

    for epoch in range(1, EPOCHS + 1):
        lr = cosine_lr(epoch - 1, EPOCHS, LR_MAX, LR_MIN)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Train epoch
        model.train()
        train_ce_sum = train_n = 0
        for x, y in train_ldr:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
            clamp_to_mask(model)  # keep dead weights at 0
            train_ce_sum += loss.item() * y.size(0)
            train_n      += y.size(0)

        train_ce = train_ce_sum / train_n
        test_ce, test_acc = evaluate(model, test_ldr)

        is_best = test_acc > best_acc
        if is_best:
            best_acc   = test_acc
            best_epoch = epoch
            best_sd    = {k: v.clone() for k, v in model.state_dict().items()}

        records.append({
            "epoch": epoch, "lr": lr,
            "train_ce": train_ce, "test_ce": test_ce,
            "test_acc": test_acc, "is_best": is_best,
        })

        # Intermediate checkpoint every CKPT_EVERY epochs
        if epoch % CKPT_EVERY == 0:
            ckpt_mid = CKPT_OUT.replace(".pt", f"_ep{epoch}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "k1": k1, "k2": k2, "epoch": epoch,
                "best_acc_so_far": best_acc,
                "weight_sparsity": init_sparsity,
            }, ckpt_mid)

        if epoch % 10 == 0 or is_best or epoch <= 5:
            flag = " ★" if is_best else ""
            print(f"{epoch:6d} {lr:8.2e} {train_ce:10.4f} {test_ce:9.4f} {test_acc*100:9.2f}%{flag}")

    print(f"\nBest: epoch={best_epoch}, acc={best_acc*100:.2f}%")
    print(f"Gain over baseline: +{(best_acc - base_acc)*100:.3f} pp")

    # Save best
    torch.save({
        "model_state_dict": best_sd,
        "k1": k1, "k2": k2,
        "base_acc": base_acc,
        "best_acc": best_acc,
        "best_epoch": best_epoch,
        "weight_sparsity": init_sparsity,
        "epochs_trained": EPOCHS,
    }, CKPT_OUT)
    print(f"Checkpoint saved: {CKPT_OUT}")

    with open(LOG_OUT, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Records saved: {LOG_OUT}")


if __name__ == "__main__":
    main()
