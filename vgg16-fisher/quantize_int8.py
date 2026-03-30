#!/usr/bin/env python3
"""
quantize_int8.py

Post-training static INT8 quantization of a compact VGG16 ONNX model
using ONNX Runtime's quantization toolkit.

Usage:
  python3 quantize_int8.py \
    --onnx vgg16_finetuned_99pct_compact.onnx \
    --output vgg16_finetuned_99pct_compact_int8_static.onnx \
    --data-root /home/petty/.openclaw/workspace-ai-research/data \
    --n-calibration 512

Steps:
  1. Load calibration subset of CIFAR-10 (default 512 images)
  2. Run static INT8 quantization (per-channel weights, per-tensor activations)
  3. Save quantized ONNX
  4. Evaluate accuracy on full CIFAR-10 test set
  5. Print size comparison
"""

import sys, os, argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

# ── ONNX Runtime quantization ────────────────────────────────────────────────
try:
    from onnxruntime.quantization import (
        quantize_static, CalibrationDataReader,
        QuantFormat, QuantType
    )
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime and onnxruntime-tools required.")
    print("  pip install onnxruntime onnxruntime-tools")
    sys.exit(1)


# ── CIFAR-10 calibration data reader ─────────────────────────────────────────

class CIFAR10CalibrationReader(CalibrationDataReader):
    """Feeds calibration batches to the ONNX Runtime quantizer."""

    def __init__(self, data_root: str, n_images: int = 512, input_size: int = 224):
        transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=False, transform=transform
        )
        indices = list(range(min(n_images, len(dataset))))
        subset  = Subset(dataset, indices)
        loader  = DataLoader(subset, batch_size=32, shuffle=False, num_workers=2)

        # Pre-compute all batches as numpy arrays
        self._batches = []
        for x, _ in loader:
            self._batches.append({"input": x.numpy()})
        self._iter = iter(self._batches)

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._batches)


def evaluate_onnx(onnx_path: str, data_root: str, input_size: int = 224) -> float:
    """Evaluate ONNX model on full CIFAR-10 test set."""
    transform = T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=False, transform=transform
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        out   = sess.run(["logits"], {"input": x.numpy()})[0]
        preds = np.argmax(out, axis=1)
        correct += (preds == y.numpy()).sum()
        total   += len(y)
        if (i + 1) % 10 == 0:
            print(f"  eval {total}/{len(dataset)} — running acc {correct/total:.4f}", flush=True)

    return correct / total


def main():
    parser = argparse.ArgumentParser(description="Static INT8 quantization for compact VGG16 ONNX")
    parser.add_argument("--onnx",           default="vgg16_finetuned_99pct_compact.onnx")
    parser.add_argument("--output",         default="vgg16_finetuned_99pct_compact_int8_static.onnx")
    parser.add_argument("--data-root",      default="/home/petty/.openclaw/workspace-ai-research/data")
    parser.add_argument("--n-calibration",  type=int, default=512,
                        help="Number of calibration images (default 512)")
    parser.add_argument("--input-size",     type=int, default=224)
    parser.add_argument("--no-eval",        action="store_true",
                        help="Skip full accuracy evaluation (faster)")
    args = parser.parse_args()

    # Resolve relative paths to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.onnx):
        args.onnx = os.path.join(script_dir, args.onnx)
    if not os.path.isabs(args.output):
        args.output = os.path.join(script_dir, args.output)

    print(f"Input ONNX:  {args.onnx}  ({os.path.getsize(args.onnx)/1024**2:.1f} MB)")
    print(f"Output:      {args.output}")
    print(f"Calibration: {args.n_calibration} images from CIFAR-10 train set")

    # ── Calibration ─────────────────────────────────────────────────────────
    print("\nBuilding calibration data reader...")
    reader = CIFAR10CalibrationReader(
        data_root=args.data_root,
        n_images=args.n_calibration,
        input_size=args.input_size,
    )
    print(f"  {len(reader._batches)} calibration batches ready")

    # ── Quantize ─────────────────────────────────────────────────────────────
    print("\nRunning static INT8 quantization...")
    quantize_static(
        model_input=args.onnx,
        model_output=args.output,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    out_size_mb = os.path.getsize(args.output) / 1024**2
    in_size_mb  = os.path.getsize(args.onnx) / 1024**2
    print(f"\n✓ Quantized model saved → {args.output}")
    print(f"  FP32: {in_size_mb:.1f} MB  →  INT8: {out_size_mb:.1f} MB  "
          f"({in_size_mb/out_size_mb:.1f}× compression)")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    if not args.no_eval:
        print("\nEvaluating FP32 ONNX accuracy (full test set)...")
        fp32_acc = evaluate_onnx(args.onnx, args.data_root, args.input_size)
        print(f"  FP32 accuracy: {fp32_acc:.4f}")

        print("\nEvaluating INT8 ONNX accuracy (full test set)...")
        int8_acc = evaluate_onnx(args.output, args.data_root, args.input_size)
        print(f"  INT8 accuracy: {int8_acc:.4f}")

        delta = int8_acc - fp32_acc
        print(f"\n  Accuracy delta: {delta:+.4f} ({delta*100:+.2f}pp)")
        print(f"  FP32: {fp32_acc:.4f}  →  INT8: {int8_acc:.4f}")
    else:
        print("\n[skipped accuracy eval — use onnxruntime to verify manually]")

    print("\nDone.")


if __name__ == "__main__":
    main()
