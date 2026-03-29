"""
Fine-tune a frozen-mask sparse VGG16 checkpoint.
Imports model/module classes directly from vgg16_pruning_v3.py to ensure
exact architecture match.

Usage:
  python finetune_sparse.py \
    --checkpoint vgg16_pruned_and_compressed_v4_99pct.pt \
    --data-root /home/petty/.openclaw/workspace-ai-research/data \
    --epochs 100 --lr 1e-3 \
    --output vgg16_finetuned_99pct.pt
"""

import argparse, os, sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── import from pruning script (ensures identical architecture) ───────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vgg16_pruning_v3 import (
    MaskedConv2d, MaskedLinear, masked_modules,
    make_masked_vgg16, VGGPruningConfig,
)


# ── data ──────────────────────────────────────────────────────────────────────

def get_loaders(data_root, batch_size=128, image_size=224):
    # Use same normalization and resize as the pruning script (ImageNet stats, 224x224)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(image_size, padding=image_size // 8),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), norm,
    ])
    train_ds = datasets.CIFAR10(data_root, train=True,  download=False, transform=train_tf)
    test_ds  = datasets.CIFAR10(data_root, train=False, download=False, transform=test_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False,
                              num_workers=4, pin_memory=True)
    return train_loader, test_loader


# ── eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss    += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (logits.argmax(1) == y).sum().item()
        total         += y.size(0)
    return total_loss / total, total_correct / total


# ── train one epoch ───────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss    += F.cross_entropy(logits, y, reduction="sum").item()
        total_correct += (logits.argmax(1) == y).sum().item()
        total         += y.size(0)
        loss.backward()
        for m in masked_modules(model):
            m.zero_masked_grads_()
        optimizer.step()
        with torch.no_grad():
            for m in masked_modules(model):
                m.weight.data.mul_(m.weight_mask)
                if m.bias is not None and m.bias_mask is not None:
                    m.bias.data.mul_(m.bias_mask)
    return total_loss / total, total_correct / total


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root",  required=True)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=128)
    parser.add_argument("--output",      default="vgg16_finetuned_99pct.pt")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--save-every",  type=int, default=10,
                        help="Save best checkpoint to disk every N epochs (default: 10)")
    parser.add_argument("--resume",      default=None,
                        help="Resume from a previously saved fine-tune checkpoint (overrides --checkpoint for weights)")
    parser.add_argument("--start-epoch", type=int, default=1,
                        help="Epoch to start/resume from (default: 1)")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")

    # Build model using exact same factory as pruning scripts
    cfg = VGGPruningConfig(use_pretrained=False, num_classes=10, device=device,
                           data_root=args.data_root)
    model = make_masked_vgg16(cfg)

    # Load weights — resume checkpoint takes priority over base pruning checkpoint
    weight_source = args.resume if args.resume else args.checkpoint
    print(f"Loading weights: {weight_source}")
    ckpt = torch.load(weight_source, map_location="cpu")
    msd  = ckpt["masked_state_dict"]

    missing, unexpected = model.load_state_dict(msd, strict=False)
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    # Verify sparsity
    total = active = 0
    for m in masked_modules(model):
        total  += m.weight_mask.numel()
        active += int(m.weight_mask.sum().item())
    sparsity = 1 - active / total
    print(f"Sparsity: {sparsity:.4f}  active={active:,} / {total:,}")

    model = model.to(device)

    train_loader, test_loader = get_loaders(args.data_root, args.batch_size)

    # Baseline eval (test only — no training has happened yet)
    loss0, acc0 = evaluate(model, test_loader, device)
    print(f"Baseline | test loss={loss0:.4f} | test acc={acc0:.4f} (train loss not measured before training)")

    # Optimizer: SGD + cosine LR (better than Adam for fine-tuning sparse nets)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5)
    # Fast-forward scheduler to correct position when resuming
    for _ in range(args.start_epoch - 1):
        scheduler.step()

    # When resuming, best_acc comes from the saved checkpoint
    if args.resume and "best_acc" in ckpt:
        best_acc = ckpt["best_acc"]
        print(f"Resuming from epoch {args.start_epoch}, previous best: {best_acc:.4f}")
    else:
        best_acc = acc0
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        test_loss, test_acc = evaluate(model, test_loader, device)
        lr_now = scheduler.get_last_lr()[0]
        gap = test_loss - train_loss  # positive = generalizing well; large positive = possible underfit; negative = overfit
        flag = " ← best" if test_acc > best_acc else ""
        print(f"Epoch {epoch:3d}/{args.epochs} | lr={lr_now:.2e} | "
              f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"test loss={test_loss:.4f} acc={test_acc:.4f} | "
              f"gap={gap:+.4f}{flag}")
        # Alias for rest of loop
        loss, acc = test_loss, test_acc

        if acc > best_acc:
            best_acc   = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            # Save immediately on improvement so we never lose the best weights
            torch.save({
                "masked_state_dict": best_state,
                "sparsity": sparsity,
                "best_acc": best_acc,
                "baseline_acc": acc0,
                "epochs_completed": epoch,
                "lr": args.lr,
            }, args.output)
            print(f"  → checkpoint saved ({best_acc:.4f})")

        # Periodic save every N epochs regardless of improvement
        if epoch % args.save_every == 0:
            periodic_path = args.output.replace(".pt", f"_ep{epoch}.pt")
            torch.save({
                "masked_state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                "sparsity": sparsity,
                "current_acc": acc,
                "best_acc": best_acc,
                "baseline_acc": acc0,
                "epochs_completed": epoch,
                "lr": args.lr,
            }, periodic_path)
            print(f"  → periodic checkpoint: {periodic_path}")

    print(f"\nBest: {best_acc:.4f}  baseline: {acc0:.4f}  Δ={best_acc-acc0:+.4f}")

    save_obj = {
        "masked_state_dict": best_state,
        "sparsity": sparsity,
        "best_acc": best_acc,
        "baseline_acc": acc0,
        "epochs": args.epochs,
        "lr": args.lr,
    }
    torch.save(save_obj, args.output)
    print(f"Final save: {args.output}")


if __name__ == "__main__":
    main()
