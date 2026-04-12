#!/bin/bash
#SBATCH --job-name=ai-research/exp35-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --open-mode=append

mkdir -p /home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}
echo "Job ${SLURM_JOB_ID} started at $(date)"
nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | head -1

source /home/petty/torch-env/bin/activate
cd /home/petty/pruning-research

python3 - <<'PYEOF'
import os, sys, json, copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, "/home/petty/pruning-research")
sys.path.insert(0, "/home/petty/pruning-research/vgg16-fisher")

from vgg16_pruning_v4 import (
    make_masked_vgg16, VGGPruningConfig, evaluate,
    build_cifar10_loaders, sparsity_report
)

RESULTS_DIR = "/home/petty/pruning-research/results"
VGG_DIR     = "/home/petty/pruning-research/vgg16-fisher"
DATA_ROOT   = "/home/petty/.openclaw/workspace-ai-research/data"

EPOCHS     = 200
LR_START   = 1e-4
LR_END     = 1e-6
SAVE_EVERY = 25
BATCH_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── Load the exp35 final masked checkpoint ───────────────────────────────────
final_ckpt = os.path.join(VGG_DIR, "vgg16_exp35_final.pt")
print(f"Loading: {final_ckpt}")
ckpt = torch.load(final_ckpt, map_location=device)

cfg = VGGPruningConfig(device=device, data_root=DATA_ROOT, use_pretrained=False,
                       batch_size=BATCH_SIZE)
train_loader, test_loader = build_cifar10_loaders(cfg)

model = make_masked_vgg16(cfg).to(device)
model.load_state_dict(ckpt["masked_state_dict"], strict=False)
model.eval()

# Freeze all masks — only train the living weights
for name, param in model.named_parameters():
    if "weight_mask" in name or "bias_mask" in name:
        param.requires_grad_(False)

report = sparsity_report(model)
sparsity = report["global_prunable_sparsity"]
_, baseline_acc = evaluate(model, test_loader, device)
print(f"Baseline: sparsity={sparsity:.4f}, test_acc={baseline_acc:.4f}")

# ── Optimizer: SGD Nesterov + cosine LR (same recipe as OBD fine-tune) ───────
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR_START, momentum=0.9, weight_decay=1e-4, nesterov=True
)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_END)
criterion = nn.CrossEntropyLoss()

best_acc   = baseline_acc
best_epoch = 0
records    = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_correct = train_total = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        # Re-zero masked weights after grad step (masks frozen but grad may drift)
        optimizer.step()
        # Enforce masks
        with torch.no_grad():
            for n, m in model.named_modules():
                if hasattr(m, "weight_mask"):
                    m.weight.data.mul_(m.weight_mask)
        preds = out.argmax(1)
        train_correct += (preds == y).sum().item()
        train_total   += y.size(0)

    scheduler.step()
    train_acc = train_correct / train_total

    model.eval()
    _, test_acc = evaluate(model, test_loader, device)

    lr_now = scheduler.get_last_lr()[0]
    rec = {"epoch": epoch, "test_acc": test_acc, "train_acc": train_acc,
           "lr": lr_now}
    records.append(rec)

    if test_acc > best_acc:
        best_acc   = test_acc
        best_epoch = epoch
        best_path  = os.path.join(VGG_DIR, "vgg16_exp35_finetuned.pt")
        torch.save({
            "masked_state_dict": model.state_dict(),
            "epoch": epoch,
            "test_acc": test_acc,
            "sparsity": sparsity,
            "method": "glauber_perlayer_rho_finetuned",
        }, best_path)

    if epoch % SAVE_EVERY == 0:
        ckpt_path = os.path.join(VGG_DIR, f"vgg16_exp35_finetuned_ep{epoch}.pt")
        torch.save({
            "masked_state_dict": model.state_dict(),
            "epoch": epoch,
            "test_acc": test_acc,
            "sparsity": sparsity,
        }, ckpt_path)

    print(f"Ep{epoch:03d} | test={test_acc:.4f} | train={train_acc:.4f} | "
          f"lr={lr_now:.2e} | best={best_acc:.4f}@ep{best_epoch}")

    # Save rolling records
    with open(os.path.join(RESULTS_DIR, "35_finetune_records.json"), "w") as f:
        json.dump(records, f, indent=2)

print(f"\nFine-tune complete. Best: {best_acc:.4f} at epoch {best_epoch}")
print(f"Baseline was: {baseline_acc:.4f}  |  Delta: {(best_acc - baseline_acc)*100:+.2f}pp")
print(f"vs OBD post-finetune: 89.08%  |  Delta: {(best_acc - 0.8908)*100:+.2f}pp")
print(f"vs Dense baseline:    89.94%  |  Delta: {(best_acc - 0.8994)*100:+.2f}pp")

summary = {
    "experiment": "35_finetune",
    "method": "glauber_perlayer_rho_finetuned",
    "sparsity": float(sparsity),
    "baseline_acc": float(baseline_acc),
    "best_finetune_acc": float(best_acc),
    "best_epoch": best_epoch,
    "gain_pp": round((best_acc - baseline_acc) * 100, 3),
    "vs_obd_postfine_pp": round((best_acc - 0.8908) * 100, 3),
    "vs_dense_pp": round((best_acc - 0.8994) * 100, 3),
    "epochs": EPOCHS,
    "lr_start": LR_START,
    "lr_end": LR_END,
    "artifacts": {
        "best_pt": os.path.join(VGG_DIR, "vgg16_exp35_finetuned.pt"),
        "records_json": os.path.join(RESULTS_DIR, "35_finetune_records.json"),
    }
}
with open(os.path.join(RESULTS_DIR, "35_finetune_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Summary saved.")
PYEOF

echo "All done at $(date)"
