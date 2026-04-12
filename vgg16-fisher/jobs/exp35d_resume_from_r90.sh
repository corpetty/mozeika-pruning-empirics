#!/bin/bash
#SBATCH --job-name=ai-research/exp35d-resume-r90
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --error=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --open-mode=append

mkdir -p /home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}
echo "Job ${SLURM_JOB_ID} started at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

source /home/petty/torch-env/bin/activate
cd /home/petty/pruning-research

# Resume from r90 checkpoint (~98.1% sparsity) — push to 99%
python3 experiments/35_vgg16_perlayer_rho.py \
    --max-rounds 120 \
    --target-sparsity 0.99 \
    --train-epochs 3 \
    --fisher-batches 5 \
    --seed 0 \
    --resume /home/petty/pruning-research/vgg16-fisher/vgg16_exp35_r90.pt

echo "Pruning job finished at $(date)"

# ── Post-pruning: compact + ONNX + INT8 export ──
echo "Running final artifact export..."
python3 - <<'PYEOF'
import os, sys, json
import torch

sys.path.insert(0, "/home/petty/pruning-research")
sys.path.insert(0, "/home/petty/pruning-research/vgg16-fisher")

from vgg16_pruning_v4 import (
    make_masked_vgg16, VGGPruningConfig, evaluate, build_cifar10_loaders,
    compress_masked_vgg16, sparsity_report
)

RESULTS_DIR = "/home/petty/pruning-research/results"
VGG_DIR     = "/home/petty/pruning-research/vgg16-fisher"
DATA_ROOT   = "/home/petty/.openclaw/workspace-ai-research/data"

device = "cuda" if torch.cuda.is_available() else "cpu"
cfg = VGGPruningConfig(device=device, data_root=DATA_ROOT, use_pretrained=False)
_, test_loader = build_cifar10_loaders(cfg)

final_ckpt = os.path.join(VGG_DIR, "vgg16_exp35_final.pt")
model = make_masked_vgg16(cfg).to(device)
ckpt = torch.load(final_ckpt, map_location=device)
model.load_state_dict(ckpt["masked_state_dict"], strict=False)
model.eval()

_, acc_masked = evaluate(model, test_loader, device)
report = sparsity_report(model)
sparsity = report["global_prunable_sparsity"]
print(f"Masked model: sparsity={sparsity:.4f}, acc={acc_masked:.4f}")

masked_out = os.path.join(VGG_DIR, "vgg16_exp35_99pct.pt")
torch.save({
    "masked_state_dict": model.state_dict(),
    "sparsity": sparsity,
    "test_acc": acc_masked,
    "method": "glauber_perlayer_rho",
    "records": ckpt.get("records", []),
}, masked_out)
print(f"Saved masked: {masked_out}")

compact = compress_masked_vgg16(model)
compact.eval()
_, acc_compact = evaluate(compact, test_loader, device)
print(f"Compact model: acc={acc_compact:.4f}")

compact_out = os.path.join(VGG_DIR, "vgg16_exp35_99pct_compact.pt")
torch.save({
    "model_state_dict": compact.state_dict(),
    "sparsity": sparsity,
    "test_acc": acc_compact,
    "method": "glauber_perlayer_rho",
}, compact_out)
compact = compact.cpu()

onnx_fp32 = os.path.join(VGG_DIR, "vgg16_exp35_99pct_compact.onnx")
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(compact, dummy, onnx_fp32,
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17)
print(f"Saved ONNX FP32: {onnx_fp32} ({os.path.getsize(onnx_fp32)//1024//1024} MB)")

import numpy as np
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
from torchvision import datasets, transforms

tfm = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

class CIFAR10Calib(CalibrationDataReader):
    def __init__(self, n=200):
        ds = datasets.CIFAR10(DATA_ROOT, train=True, transform=tfm)
        idxs = torch.randperm(len(ds))[:n]
        self.data = [ds[i][0].unsqueeze(0).numpy() for i in idxs]
        self.idx = 0
    def get_next(self):
        if self.idx >= len(self.data): return None
        out = {"input": self.data[self.idx]}; self.idx += 1; return out

onnx_int8 = os.path.join(VGG_DIR, "vgg16_exp35_99pct_compact_int8_static.onnx")
print("Calibrating INT8...")
quantize_static(onnx_fp32, onnx_int8,
    calibration_data_reader=CIFAR10Calib(200),
    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
    per_channel=True, reduce_range=False)
print(f"Saved ONNX INT8: {onnx_int8} ({os.path.getsize(onnx_int8)//1024//1024} MB)")

sess = onnxruntime.InferenceSession(onnx_int8, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
correct = total = 0
for x, y in test_loader:
    logits = sess.run(None, {in_name: x.numpy()})[0]
    preds = np.argmax(logits, axis=1)
    correct += (preds == y.numpy()).sum(); total += y.size(0)
acc_int8 = correct / total
print(f"INT8 accuracy: {acc_int8:.4f}")

summary = {
    "experiment": "35d_resume_r90",
    "method": "glauber_perlayer_rho",
    "sparsity": float(sparsity),
    "acc_masked": float(acc_masked),
    "acc_compact_fp32": float(acc_compact),
    "acc_onnx_int8": float(acc_int8),
    "dense_baseline": 0.8994,
    "gap_to_dense_pp": round((float(acc_masked) - 0.8994) * 100, 3),
    "comparison_obd": {
        "obd_99pct_prefine": 0.8572,
        "obd_99pct_postfine": 0.8908,
        "glauber_99pct": float(acc_masked),
        "glauber_advantage_pp": round((float(acc_masked) - 0.8572) * 100, 2),
    },
    "artifacts": {
        "masked_pt": masked_out,
        "compact_pt": compact_out,
        "onnx_fp32": onnx_fp32,
        "onnx_int8": onnx_int8,
    }
}
out_summary = os.path.join(RESULTS_DIR, "35b_summary.json")
with open(out_summary, "w") as f:
    json.dump(summary, f, indent=2)
print("Summary:", json.dumps(summary, indent=2))
PYEOF

echo "All done at $(date)"
