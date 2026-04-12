"""
Exp 33 — LeNet-300-100/MNIST: Pi5 deployment packager

Takes results/33_compact_final.pt (784→122→35→10 with weight+neuron masks)
and produces a deploy package with:
  1. truly_compact_net.onnx    — dense 784→active1→active2→10 FP32
  2. truly_compact_int8.onnx   — INT8 static quant (calibrated on MNIST test)
  3. infer.py                  — standalone inference script (onnxruntime)
  4. metadata.json             — architecture / accuracy summary
  5. README.md

Dead neurons (all-zero weight rows) are pruned physically, giving a genuinely
dense network — no sparse tensor ops required on the Pi5's Cortex-A76.
"""

import json
import os
import struct
import sys
import tarfile
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ONNX + quantization
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType
except ImportError:
    print("pip install onnx onnxruntime in torch-env first")
    sys.exit(1)

# MNIST loader (torchvision)
from torchvision import datasets, transforms

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "33_compact_final.pt",
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results",
)
DATA_ROOT = "/home/petty/.openclaw/workspace-ai-research/data"


# ---------------------------------------------------------------------------
# Model definition (masked net — matches training code)
# ---------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer("weight_mask", torch.ones_like(self.weight))
        if bias:
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


class MaskedLeNet300(nn.Module):
    def __init__(self, k1=300, k2=100):
        super().__init__()
        self.fc1 = MaskedLinear(784, k1)
        self.fc2 = MaskedLinear(k1, k2)
        self.fc3 = MaskedLinear(k2, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ---------------------------------------------------------------------------
# Dense compact model (no masks, no dead neurons)
# ---------------------------------------------------------------------------

class CompactLeNet(nn.Module):
    def __init__(self, w1, b1, w2, b2, w3, b3):
        """w1/b1/w2/b2/w3/b3 are dense tensors already stripped of dead rows."""
        super().__init__()
        self.fc1 = nn.Linear(w1.shape[1], w1.shape[0])
        self.fc2 = nn.Linear(w2.shape[1], w2.shape[0])
        self.fc3 = nn.Linear(w3.shape[1], w3.shape[0])
        with torch.no_grad():
            self.fc1.weight.copy_(w1)
            self.fc1.bias.copy_(b1)
            self.fc2.weight.copy_(w2)
            self.fc2.bias.copy_(b2)
            self.fc3.weight.copy_(w3)
            self.fc3.bias.copy_(b3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def build_compact_model(ckpt_path: str) -> tuple:
    """
    Load masked checkpoint, strip dead neurons, return (CompactLeNet, arch_info).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd   = ckpt["model_state_dict"]

    # ── FC1 ──────────────────────────────────────────────────────────────
    w1_full = (sd["fc1.weight"] * sd["fc1.weight_mask"])   # (122, 784)
    b1_full = (sd["fc1.bias"]   * sd["fc1.bias_mask"])
    # alive neurons: any weight OR bias non-zero
    alive1  = (w1_full.abs().sum(dim=1) + b1_full.abs()) > 0  # (122,)
    w1 = w1_full[alive1]    # (n1_alive, 784)
    b1 = b1_full[alive1]

    # ── FC2 (input cols correspond to alive1 outputs) ────────────────────
    w2_full = (sd["fc2.weight"] * sd["fc2.weight_mask"])   # (35, 122)
    b2_full = (sd["fc2.bias"]   * sd["fc2.bias_mask"])
    w2_pre  = w2_full[:, alive1]                           # keep alive1 cols
    alive2  = (w2_pre.abs().sum(dim=1) + b2_full.abs()) > 0
    w2 = w2_pre[alive2]
    b2 = b2_full[alive2]

    # ── FC3 (input cols correspond to alive2 outputs) ────────────────────
    w3_full = (sd["fc3.weight"] * sd["fc3.weight_mask"])   # (10, 35)
    b3      =  sd["fc3.bias"]
    w3      = w3_full[:, alive2]

    arch = {
        "input":      784,
        "fc1_alive":  int(alive1.sum().item()),
        "fc2_alive":  int(alive2.sum().item()),
        "output":     10,
        "fc1_dead":   int((~alive1).sum().item()),
        "fc2_dead":   int((~alive2).sum().item()),
        "total_params": (w1.numel() + b1.numel() +
                         w2.numel() + b2.numel() +
                         w3.numel() + b3.numel()),
        "phase2_weight_sparsity": float(ckpt["phase2_weight_sparsity"]),
        "phase2_final_acc":       float(ckpt["phase2_final_acc"]),
    }
    model = CompactLeNet(w1, b1, w2, b2, w3, b3)
    return model, arch


# ---------------------------------------------------------------------------
# Evaluate on MNIST
# ---------------------------------------------------------------------------

def get_mnist_loader(batch_size=1000, train=False):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds = datasets.MNIST(DATA_ROOT, train=train, download=True, transform=tfm)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)


@torch.no_grad()
def evaluate_pt(model, loader, device="cpu"):
    model.eval().to(device)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
    return correct / total


def evaluate_onnx(onnx_path, loader):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    inp_shape = [d.dim_value if hasattr(d, 'dim_value') else -1
                 for d in sess.get_inputs()[0].shape]
    # Determine whether model expects flat (batch, 784) or image (batch, 1, 28, 28)
    flat_input = len(inp_shape) == 2
    correct = total = 0
    for x, y in loader:
        if flat_input:
            x_np = x.numpy().reshape(x.shape[0], -1).astype(np.float32)
        else:
            x_np = x.numpy().astype(np.float32)  # keep (batch, 1, 28, 28)
        logits = sess.run(None, {inp_name: x_np})[0]
        preds = logits.argmax(axis=1)
        correct += (preds == y.numpy()).sum()
        total   += y.shape[0]
    return correct / total


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(model, out_path):
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Exported FP32 ONNX → {out_path}")


# ---------------------------------------------------------------------------
# INT8 static quantization
# ---------------------------------------------------------------------------

class MNISTCalibReader(CalibrationDataReader):
    def __init__(self, loader, n_batches=10, flat=False):
        self.data = []
        self.flat = flat
        for i, (x, _) in enumerate(loader):
            if i >= n_batches:
                break
            arr = x.numpy().astype(np.float32)
            if flat:
                arr = arr.reshape(arr.shape[0], -1)
            self.data.append(arr)
        self._iter = iter(self.data)

    def get_next(self):
        try:
            batch = next(self._iter)
            return {"input": batch}
        except StopIteration:
            return None


def quantize_int8(fp32_path, int8_path, calib_loader, flat=False):
    reader = MNISTCalibReader(calib_loader, n_batches=10, flat=flat)
    quantize_static(
        fp32_path, int8_path,
        calibration_data_reader=reader,
        quant_format=None,
        per_channel=False,
        weight_type=QuantType.QInt8,
    )
    print(f"  INT8 quantized ONNX → {int8_path}")


# ---------------------------------------------------------------------------
# Inference script (dropped into deploy package)
# ---------------------------------------------------------------------------

INFER_PY = '''\
#!/usr/bin/env python3
"""
LeNet-300-100 MNIST inference — Pi5 deploy (Exp 33)
Requires: onnxruntime, numpy, Pillow

Usage:
    python3 infer.py image.png
    python3 infer.py --benchmark   # 1000-sample throughput test (needs MNIST)
"""

import sys
import numpy as np
import onnxruntime as ort

LABELS = list(range(10))
MODEL  = "truly_compact_int8.onnx"   # or truly_compact_net.onnx for FP32


def preprocess(img_array: np.ndarray) -> np.ndarray:
    """img_array: (28, 28) uint8 or float, 0-255"""
    x = img_array.astype(np.float32) / 255.0
    x = (x - 0.1307) / 0.3081
    return x.reshape(1, 1, 28, 28)  # (batch=1, C=1, H=28, W=28)


def load_image(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("L").resize((28, 28))
    return np.array(img)


def run_inference(img_path: str):
    sess  = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    name  = sess.get_inputs()[0].name
    arr   = preprocess(load_image(img_path))
    out   = sess.run(None, {name: arr})[0]
    probs = np.exp(out[0]) / np.exp(out[0]).sum()
    pred  = int(probs.argmax())
    print(f"Prediction: {pred}  (confidence: {probs[pred]*100:.1f}%)")
    for i, p in enumerate(probs):
        bar = "#" * int(p * 40)
        print(f"  {i}: {bar:<40} {p*100:5.1f}%")


def benchmark():
    import time
    import torchvision.datasets as dsets
    import torchvision.transforms as T

    ds  = dsets.MNIST("/tmp/mnist_data", train=False, download=True,
                      transform=T.Compose([T.ToTensor(),
                                           T.Normalize((0.1307,), (0.3081,))]))
    ldr = __import__("torch").utils.data.DataLoader(ds, batch_size=1000)
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name

    correct = total = 0
    t0 = time.perf_counter()
    for x, y in ldr:
        x_np = x.numpy().astype(np.float32)  # keep (batch, 1, 28, 28)
        preds = sess.run(None, {name: x_np})[0].argmax(axis=1)
        correct += (preds == y.numpy()).sum()
        total   += y.shape[0]
    elapsed = time.perf_counter() - t0

    print(f"Accuracy:   {correct/total*100:.2f}%")
    print(f"Throughput: {total/elapsed:.0f} samples/sec")
    print(f"Latency:    {elapsed/total*1000:.3f} ms/sample")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    if sys.argv[1] == "--benchmark":
        benchmark()
    else:
        run_inference(sys.argv[1])
'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Exp 33 Pi5 Deploy Packager ===")

    # ── Build compact model ───────────────────────────────────────────────
    print("\n[1/5] Building compact model (stripping dead neurons)...")
    model, arch = build_compact_model(CHECKPOINT)
    print(
        f"  Architecture: 784 → {arch['fc1_alive']} → {arch['fc2_alive']} → 10"
        f"  (dead: fc1={arch['fc1_dead']}, fc2={arch['fc2_dead']})"
    )
    print(f"  Total parameters: {arch['total_params']:,}")

    # ── Evaluate FP32 ─────────────────────────────────────────────────────
    print("\n[2/5] Evaluating compact FP32 model on MNIST test...")
    test_loader = get_mnist_loader(batch_size=1000, train=False)
    acc_pt = evaluate_pt(model, test_loader)
    print(f"  FP32 accuracy: {acc_pt*100:.2f}%  (training reported {arch['phase2_final_acc']*100:.2f}%)")
    arch["compact_fp32_acc"] = float(acc_pt)

    # ── Export ONNX FP32 ──────────────────────────────────────────────────
    print("\n[3/5] Exporting FP32 ONNX...")
    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_path = os.path.join(tmpdir, "truly_compact_net.onnx")
        int8_path = os.path.join(tmpdir, "truly_compact_int8.onnx")

        export_onnx(model, fp32_path)
        acc_fp32_onnx = evaluate_onnx(fp32_path, test_loader)
        print(f"  FP32 ONNX accuracy: {acc_fp32_onnx*100:.2f}%")
        arch["onnx_fp32_acc"] = float(acc_fp32_onnx)

        fp32_size = os.path.getsize(fp32_path)
        print(f"  FP32 ONNX size: {fp32_size/1024:.1f} KB")
        arch["onnx_fp32_size_kb"] = fp32_size / 1024

        # ── INT8 static quantization ──────────────────────────────────────
        print("\n[4/5] INT8 static quantization...")
        calib_loader = get_mnist_loader(batch_size=512, train=True)
        # FP32 ONNX model was exported with 4D input (batch, 1, 28, 28)
        quantize_int8(fp32_path, int8_path, calib_loader, flat=False)
        acc_int8 = evaluate_onnx(int8_path, test_loader)
        print(f"  INT8 ONNX accuracy: {acc_int8*100:.2f}%")
        arch["onnx_int8_acc"] = float(acc_int8)

        int8_size = os.path.getsize(int8_path)
        print(f"  INT8 ONNX size: {int8_size/1024:.1f} KB")
        arch["onnx_int8_size_kb"] = int8_size / 1024
        arch["quantization_drop_pp"] = float((acc_fp32_onnx - acc_int8) * 100)

        # ── README ────────────────────────────────────────────────────────
        readme = f"""# LeNet-300-100 MNIST — Exp 33 Pi5 Deploy Package

## Architecture
Input 784 → FC({arch['fc1_alive']}) → FC({arch['fc2_alive']}) → Output 10

Derived from two-phase neuron+weight Glauber pruning (Mozeika & Pizzoferrato 2026):
- Phase 1: neuron Glauber — 60.7% neuron sparsity removed, acc improved 97.22%→98.21%
- Phase 2: OBD weight pruning — {arch['phase2_weight_sparsity']*100:.1f}% weight sparsity, acc {arch['phase2_final_acc']*100:.2f}%
- Dead neuron removal: fc1 -{arch['fc1_dead']} neurons, fc2 -{arch['fc2_dead']} neurons
- Effective architecture: 784→{arch['fc1_alive']}→{arch['fc2_alive']}→10

## Files
- `truly_compact_int8.onnx`  — INT8 static quantized, {int8_size/1024:.1f} KB  ← primary deploy artifact
- `truly_compact_net.onnx`   — FP32 baseline, {fp32_size/1024:.1f} KB
- `infer.py`                 — standalone inference + benchmark script

## Accuracy
| Model       | Accuracy |
|-------------|----------|
| FP32 dense  | {acc_fp32_onnx*100:.2f}%  |
| INT8 static | {acc_int8*100:.2f}%  |
| Quant drop  | {(acc_fp32_onnx-acc_int8)*100:.3f} pp |

## Install on Pi5
```bash
pip install onnxruntime numpy pillow
python3 infer.py --benchmark
```

## Single image inference
```bash
python3 infer.py digit.png
```
"""

        # ── Package ───────────────────────────────────────────────────────
        print("\n[5/5] Packaging...")
        pkg_path = os.path.join(OUT_DIR, "exp33_pi5_deploy.tar.gz")
        with tarfile.open(pkg_path, "w:gz") as tf:
            tf.add(fp32_path, arcname="exp33_pi5_deploy/truly_compact_net.onnx")
            tf.add(int8_path, arcname="exp33_pi5_deploy/truly_compact_int8.onnx")

            # infer.py
            infer_tmp = os.path.join(tmpdir, "infer.py")
            with open(infer_tmp, "w") as f:
                f.write(INFER_PY)
            tf.add(infer_tmp, arcname="exp33_pi5_deploy/infer.py")

            # metadata.json
            meta_tmp = os.path.join(tmpdir, "metadata.json")
            with open(meta_tmp, "w") as f:
                json.dump(arch, f, indent=2)
            tf.add(meta_tmp, arcname="exp33_pi5_deploy/metadata.json")

            # README.md
            readme_tmp = os.path.join(tmpdir, "README.md")
            with open(readme_tmp, "w") as f:
                f.write(readme)
            tf.add(readme_tmp, arcname="exp33_pi5_deploy/README.md")

        pkg_size = os.path.getsize(pkg_path)
        print(f"  Package: {pkg_path}")
        print(f"  Package size: {pkg_size/1024:.1f} KB")
        arch["package_size_kb"] = pkg_size / 1024

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    print(f"  Architecture:     784 → {arch['fc1_alive']} → {arch['fc2_alive']} → 10")
    print(f"  Parameters:       {arch['total_params']:,}")
    print(f"  FP32 accuracy:    {acc_fp32_onnx*100:.2f}%")
    print(f"  INT8 accuracy:    {acc_int8*100:.2f}%")
    print(f"  Quant drop:       {(acc_fp32_onnx-acc_int8)*100:.3f} pp")
    print(f"  FP32 ONNX size:   {arch['onnx_fp32_size_kb']:.1f} KB")
    print(f"  INT8 ONNX size:   {arch['onnx_int8_size_kb']:.1f} KB")
    print(f"  Package:          {arch['package_size_kb']:.1f} KB")

    meta_path = os.path.join(OUT_DIR, "exp33_deploy_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(arch, f, indent=2)
    print(f"  Metadata saved:   {meta_path}")


if __name__ == "__main__":
    main()
