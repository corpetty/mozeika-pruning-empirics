#!/bin/bash
#SBATCH --job-name=ai-research/exp35-export
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/home/petty/slurm-logs/%j-%x/job.log
#SBATCH --open-mode=append

mkdir -p /home/petty/slurm-logs/${SLURM_JOB_ID}-${SLURM_JOB_NAME}
source /home/petty/torch-env/bin/activate
cd /home/petty/pruning-research

python3 - <<'PYEOF'
import os, sys, json
import torch

sys.path.insert(0, "/home/petty/pruning-research/vgg16-fisher")
from vgg16_pruning_v4 import (
    make_masked_vgg16, VGGPruningConfig, evaluate,
    build_cifar10_loaders, compress_masked_vgg16, sparsity_report
)

VGG_DIR   = "/home/petty/pruning-research/vgg16-fisher"
DATA_ROOT = "/home/petty/.openclaw/workspace-ai-research/data"
RESULTS   = "/home/petty/pruning-research/results"
device    = "cuda" if torch.cuda.is_available() else "cpu"

# Load masked final checkpoint
ckpt_path = os.path.join(VGG_DIR, "vgg16_exp35_final.pt")
print(f"Loading {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)

cfg = VGGPruningConfig(device=device, data_root=DATA_ROOT, use_pretrained=False)
_, test_loader = build_cifar10_loaders(cfg)

model = make_masked_vgg16(cfg).to(device)
model.load_state_dict(ckpt["masked_state_dict"], strict=False)
model.eval()

report = sparsity_report(model)
sparsity = report["global_prunable_sparsity"]
_, acc = evaluate(model, test_loader, device)
print(f"Masked: sparsity={sparsity:.4f}, acc={acc:.4f}")

# Save canonical masked checkpoint
masked_out = os.path.join(VGG_DIR, "vgg16_exp35_99pct.pt")
torch.save({"masked_state_dict": model.state_dict(), "sparsity": sparsity,
            "test_acc": acc, "method": "glauber_perlayer_rho"}, masked_out)
print(f"Saved masked: {masked_out}")

# Compact
print("Compressing...")
compact, meta = compress_masked_vgg16(model, num_classes=10)
compact = compact.to(device)
compact.eval()
_, compact_acc = evaluate(compact, test_loader, device)
print(f"Compact: acc={compact_acc:.4f}")
print(f"Compact arch: {meta}")

compact_out = os.path.join(VGG_DIR, "vgg16_exp35_compact.pt")
torch.save({"state_dict": compact.state_dict(), "arch_meta": meta,
            "sparsity": sparsity, "test_acc": compact_acc}, compact_out)
print(f"Saved compact: {compact_out}")

# FP32 ONNX
import torch.onnx
dummy = torch.randn(1, 3, 224, 224, device=device)
onnx_fp32 = os.path.join(VGG_DIR, "vgg16_exp35_compact_fp32.onnx")
torch.onnx.export(compact, dummy, onnx_fp32, input_names=["input"],
                  output_names=["logits"], opset_version=17,
                  dynamic_axes={"input": {0: "batch_size"}})
print(f"Saved FP32 ONNX: {onnx_fp32}")

# INT8 static ONNX
try:
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
    import onnxruntime as ort
    import numpy as np
    from torchvision import datasets, transforms
    import onnx

    print("Running INT8 static quantization...")

    class CIFARCalibReader(CalibrationDataReader):
        def __init__(self, n=200):
            transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            ds = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=transform)
            self.data = [ds[i][0].unsqueeze(0).numpy() for i in range(n)]
            self.idx = 0
        def get_next(self):
            if self.idx >= len(self.data): return None
            x = {"input": self.data[self.idx]}; self.idx += 1; return x

    onnx_int8 = os.path.join(VGG_DIR, "vgg16_exp35_compact_int8_static.onnx")
    quantize_static(onnx_fp32, onnx_int8, CalibrationDataReader=CIFARCalibReader,
                    quant_type=QuantType.QInt8)

    sess = ort.InferenceSession(onnx_int8)
    correct = total = 0
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    ds = datasets.CIFAR10(DATA_ROOT, train=False, download=False, transform=transform)
    for i in range(0, min(2000, len(ds)), 1):
        x, y = ds[i]
        out = sess.run(None, {"input": x.unsqueeze(0).numpy()})[0]
        correct += (np.argmax(out) == y); total += 1
    int8_acc = correct / total
    print(f"INT8 static: acc={int8_acc:.4f} (on {total} samples)")
    print(f"Saved INT8: {onnx_int8}")
except Exception as e:
    print(f"INT8 quantization failed: {e}")
    int8_acc = None
    onnx_int8 = None

# Summary
summary = {
    "experiment": "35_export",
    "sparsity": float(sparsity),
    "masked_acc": float(acc),
    "compact_acc": float(compact_acc),
    "int8_acc": float(int8_acc) if int8_acc else None,
    "artifacts": {
        "masked_pt": masked_out,
        "compact_pt": compact_out,
        "onnx_fp32": onnx_fp32,
        "onnx_int8": onnx_int8,
    }
}
with open(os.path.join(RESULTS, "35_export_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Export complete.")
print(json.dumps(summary, indent=2))
PYEOF
