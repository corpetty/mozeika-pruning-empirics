"""
Exp 36 Pi5 Deploy Script
Extracts the compact dense network from the exp36 fine-tuned masked checkpoint,
converts to ONNX + INT8 static quantization, validates accuracy, packages for Pi5.
"""

import os, json, tarfile, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms

DATA_ROOT   = "/home/petty/.openclaw/workspace-ai-research/data"
RESULTS_DIR = "/home/petty/pruning-research/results"
CKPT_IN     = os.path.join(RESULTS_DIR, "36_lenet_finetuned.pt")
DEPLOY_DIR  = os.path.join(RESULTS_DIR, "exp36_deploy")
TAR_OUT     = os.path.join(RESULTS_DIR, "exp36_pi5_deploy.tar.gz")

os.makedirs(DEPLOY_DIR, exist_ok=True)

# ── Model definitions ─────────────────────────────────────────────────────────
class MaskedLinear(nn.Linear):
    def __init__(self, in_f, out_f):
        super().__init__(in_f, out_f)
        self.register_buffer('weight_mask', torch.ones_like(self.weight))
        self.register_buffer('bias_mask', torch.ones(out_f))
    def forward(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias * self.bias_mask)

class CompactMaskedNet(nn.Module):
    def __init__(self, k1, k2):
        super().__init__()
        self.fc1 = MaskedLinear(784, k1)
        self.fc2 = MaskedLinear(k1, k2)
        self.fc3 = nn.Linear(k2, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

class DenseLeNet(nn.Module):
    """Compact extracted network — no masks, no dead neurons."""
    def __init__(self, a1, a2):
        super().__init__()
        self.fc1 = nn.Linear(784, a1)
        self.fc2 = nn.Linear(a1, a2)
        self.fc3 = nn.Linear(a2, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))

# ── Load & inspect masked model ───────────────────────────────────────────────
print("Loading exp36 checkpoint...")
ckpt = torch.load(CKPT_IN, map_location="cpu")
k1, k2 = ckpt["k1"], ckpt["k2"]
sd = ckpt["model_state_dict"]

masked_model = CompactMaskedNet(k1, k2)
masked_model.load_state_dict(sd, strict=False)
masked_model.eval()

# Identify live neurons
fc1_mask = sd["fc1.weight_mask"]  # (k1, 784)
fc2_mask = sd["fc2.weight_mask"]  # (k2, k1)

alive_fc1 = torch.where(fc1_mask.abs().sum(dim=1) > 0)[0]
alive_fc2 = torch.where(fc2_mask.abs().sum(dim=1) > 0)[0]
a1, a2 = len(alive_fc1), len(alive_fc2)

print(f"Architecture: 784 → {k1}({a1} alive) → {k2}({a2} alive) → 10")
print(f"Compact: 784 → {a1} → {a2} → 10")

# ── Extract compact weights ───────────────────────────────────────────────────
def extract_compact(sd, alive_fc1, alive_fc2):
    # fc1: rows = alive_fc1, cols = all 784
    fc1_w = (sd["fc1.weight"] * sd["fc1.weight_mask"])[alive_fc1]   # (a1, 784)
    fc1_b = (sd["fc1.bias"]   * sd["fc1.bias_mask"]  )[alive_fc1]   # (a1,)

    # fc2: rows = alive_fc2, cols = alive_fc1
    fc2_w_full = (sd["fc2.weight"] * sd["fc2.weight_mask"])           # (k2, k1)
    fc2_w = fc2_w_full[alive_fc2][:, alive_fc1]                       # (a2, a1)
    fc2_b = (sd["fc2.bias"]   * sd["fc2.bias_mask"]  )[alive_fc2]    # (a2,)

    # fc3: rows = all 10, cols = alive_fc2
    fc3_w = sd["fc3.weight"][:, alive_fc2]                             # (10, a2)
    fc3_b = sd["fc3.bias"]                                             # (10,)

    return fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b

fc1_w, fc1_b, fc2_w, fc2_b, fc3_w, fc3_b = extract_compact(sd, alive_fc1, alive_fc2)

compact = DenseLeNet(a1, a2)
compact.fc1.weight.data = fc1_w
compact.fc1.bias.data   = fc1_b
compact.fc2.weight.data = fc2_w
compact.fc2.bias.data   = fc2_b
compact.fc3.weight.data = fc3_w
compact.fc3.bias.data   = fc3_b
compact.eval()

# ── Accuracy validation ───────────────────────────────────────────────────────
tfm = transforms.ToTensor()  # no normalization — matches training
test_ds = datasets.MNIST(DATA_ROOT, train=False, download=True, transform=tfm)
test_ldr = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)

def eval_acc(model, loader, name):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            p = model(x).argmax(1)
            correct += (p==y).sum().item(); total += y.size(0)
    acc = correct / total
    print(f"  {name}: {acc*100:.2f}%  ({correct}/{total})")
    return acc

print("\nAccuracy check:")
acc_masked  = eval_acc(masked_model, test_ldr, "Sparse masked (exp36 best)")
acc_compact = eval_acc(compact,      test_ldr, "Compact dense (extracted) ")

# Disagreements
disagreements = 0
with torch.no_grad():
    for x, _ in test_ldr:
        p1 = masked_model(x).argmax(1)
        p2 = compact(x).argmax(1)
        disagreements += (p1 != p2).sum().item()
print(f"  Prediction disagreements: {disagreements}")
assert disagreements == 0, "Extraction mismatch!"
print("  ✓ Extraction verified — models are numerically equivalent")

# ── Save compact FP32 ─────────────────────────────────────────────────────────
compact_pt = os.path.join(DEPLOY_DIR, "lenet_compact_fp32.pt")
torch.save({"model_state_dict": compact.state_dict(), "a1": a1, "a2": a2, "acc": acc_compact}, compact_pt)
print(f"\nSaved compact FP32: {compact_pt}")

# ── Export ONNX FP32 ──────────────────────────────────────────────────────────
import onnx
onnx_fp32 = os.path.join(DEPLOY_DIR, "lenet_compact_fp32.onnx")
dummy = torch.randn(1, 1, 28, 28)
torch.onnx.export(
    compact, dummy, onnx_fp32,
    input_names=["input"], output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
print(f"Saved ONNX FP32:    {onnx_fp32}  ({os.path.getsize(onnx_fp32)//1024} KB)")

# ── INT8 static quantization ──────────────────────────────────────────────────
import onnxruntime
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

class MNISTCalib(CalibrationDataReader):
    def __init__(self, n=1000):
        calib_ds = datasets.MNIST(DATA_ROOT, train=True, transform=tfm)
        indices = torch.randperm(len(calib_ds))[:n]
        self.data = [calib_ds[i][0].unsqueeze(0).numpy() for i in indices]
        self.idx = 0
    def get_next(self):
        if self.idx >= len(self.data):
            return None
        batch = {"input": self.data[self.idx]}
        self.idx += 1
        return batch

onnx_int8 = os.path.join(DEPLOY_DIR, "lenet_compact_int8_static.onnx")
print("\nRunning INT8 static calibration (1000 samples)...")
quantize_static(
    onnx_fp32, onnx_int8,
    calibration_data_reader=MNISTCalib(1000),
    activation_type=QuantType.QInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
    reduce_range=False,
)
print(f"Saved ONNX INT8:    {onnx_int8}  ({os.path.getsize(onnx_int8)//1024} KB)")

# ── ONNX accuracy check ───────────────────────────────────────────────────────
def eval_onnx(path, loader, name):
    sess = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    correct = total = 0
    for x, y in loader:
        logits = sess.run(None, {in_name: x.numpy()})[0]
        preds = np.argmax(logits, axis=1)
        correct += (preds == y.numpy()).sum(); total += y.size(0)
    acc = correct / total
    print(f"  {name}: {acc*100:.2f}%")
    return acc

print("\nONNX accuracy check:")
acc_onnx_fp32 = eval_onnx(onnx_fp32, test_ldr, "ONNX FP32")
acc_onnx_int8 = eval_onnx(onnx_int8, test_ldr, "ONNX INT8")
print(f"  FP32→INT8 delta: {(acc_onnx_fp32 - acc_onnx_int8)*100:+.3f} pp")

# ── Write inference script ────────────────────────────────────────────────────
infer_py = os.path.join(DEPLOY_DIR, "infer.py")
with open(infer_py, "w") as f:
    f.write('''#!/usr/bin/env python3
"""
Pi5 MNIST inference — exp36 LeNet compact INT8
Usage: python3 infer.py <image.png> [image2.png ...]
       python3 infer.py --benchmark
"""
import sys, os, time
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL = os.path.join(os.path.dirname(__file__), "lenet_compact_int8_static.onnx")
LABELS = list("0123456789")

def preprocess(path):
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.float32) / 255.0
    # No normalization — model trained without it
    return arr.reshape(1, 1, 28, 28)

def main():
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    if "--benchmark" in sys.argv:
        x = np.random.rand(1, 1, 28, 28).astype(np.float32)
        # Warmup
        for _ in range(20):
            sess.run(None, {in_name: x})
        # Benchmark
        N = 1000
        t0 = time.perf_counter()
        for _ in range(N):
            sess.run(None, {in_name: x})
        elapsed = time.perf_counter() - t0
        print(f"Benchmark: {N} inferences in {elapsed*1000:.1f} ms")
        print(f"  Latency:    {elapsed/N*1000:.3f} ms/image")
        print(f"  Throughput: {N/elapsed:.0f} img/s")
        return

    for path in sys.argv[1:]:
        x = preprocess(path)
        logits = sess.run(None, {in_name: x})[0][0]
        pred = int(np.argmax(logits))
        conf = float(np.exp(logits[pred]) / np.exp(logits).sum())
        print(f"{path}: {pred}  (conf {conf*100:.1f}%)")

if __name__ == "__main__":
    main()
''')

# ── Write README ──────────────────────────────────────────────────────────────
readme = os.path.join(DEPLOY_DIR, "README.md")
with open(readme, "w") as f:
    f.write(f"""# Exp36 LeNet Pi5 Deploy Package

## Model
- Architecture: 784 → {a1} → {a2} → 10 (dense compact, extracted from 784→{k1}→{k2}→10)
- Original sparsity: 97.64%
- FP32 test accuracy: {acc_onnx_fp32*100:.2f}%
- INT8 test accuracy: {acc_onnx_int8*100:.2f}%
- Dense baseline: 98.24%
- Gap to dense: {(0.9824 - acc_onnx_int8)*100:.2f} pp

## Preprocessing
- Resize to 28×28, grayscale
- ToTensor (divide by 255) — **NO normalization**
- Shape: (1, 1, 28, 28) float32

## Files
- `lenet_compact_int8_static.onnx` — deploy model
- `lenet_compact_fp32.onnx` — reference FP32
- `lenet_compact_fp32.pt` — PyTorch checkpoint
- `infer.py` — inference + benchmark script

## Install on Pi5
```bash
pip install onnxruntime pillow numpy
```

## Run
```bash
python3 infer.py digit.png
python3 infer.py --benchmark
```
""")

# ── Package ───────────────────────────────────────────────────────────────────
with tarfile.open(TAR_OUT, "w:gz") as tar:
    tar.add(DEPLOY_DIR, arcname="exp36_pi5_deploy")
print(f"\nPackage: {TAR_OUT}  ({os.path.getsize(TAR_OUT)//1024} KB)")

# ── Summary ───────────────────────────────────────────────────────────────────
meta = {
    "exp": 36,
    "source_checkpoint": CKPT_IN,
    "architecture_masked": f"784->{k1}->{k2}->10",
    "architecture_compact": f"784->{a1}->{a2}->10",
    "weight_sparsity": float(ckpt["weight_sparsity"]),
    "acc_sparse_masked": float(acc_masked),
    "acc_compact_dense": float(acc_compact),
    "acc_onnx_fp32": float(acc_onnx_fp32),
    "acc_onnx_int8": float(acc_onnx_int8),
    "dense_baseline": 0.9824,
    "gap_to_dense_pp": round((0.9824 - acc_onnx_int8)*100, 3),
    "prediction_disagreements_masked_vs_compact": disagreements,
}
meta_path = os.path.join(RESULTS_DIR, "exp36_deploy_metadata.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata: {meta_path}")
print("\nDone.")
