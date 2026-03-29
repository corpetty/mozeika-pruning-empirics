#!/usr/bin/env python3
"""
extract_and_export.py

Convert a sparse masked VGG16 checkpoint → compact dense model → ONNX.

Usage:
  python3 extract_and_export.py --checkpoint <path.pt> [--output-dir <dir>] [--input-size 32|224] [--device cpu|cuda]

Steps:
  1. Load masked checkpoint
  2. Compress: physically remove dead channels/neurons → CompressedVGG16
  3. Verify accuracy is preserved
  4. Export to ONNX
  5. Print architecture summary (layer widths, MACs estimate, file size)
"""

import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np

from vgg16_pruning_v4 import (
    VGGPruningConfig, make_masked_vgg16, compress_masked_vgg16,
    build_cifar10_loaders, evaluate, named_masked_modules,
    alive_out_channels, alive_linear_neurons, MaskedConv2d, MaskedLinear,
)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_masked_model(ckpt_path: str, device: str) -> nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = VGGPruningConfig(use_pretrained=False, device=device, num_classes=10)

    # Support both raw state-dict saves and full checkpoint dicts
    if isinstance(ckpt, dict) and "masked_state_dict" in ckpt:
        state = ckpt["masked_state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt  # assume it's a raw state dict

    model = make_masked_vgg16(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)} (e.g. {missing[0]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)}")
    model.eval()
    return model, cfg


def print_sparse_summary(model: nn.Module):
    """Print per-layer sparsity and dead channel counts."""
    print("\n── Sparse model layer summary ──────────────────────────────────")
    convs = [m for m in model.features if isinstance(m, MaskedConv2d)]
    fcs   = [m for m in model.classifier if isinstance(m, MaskedLinear)]

    total_w = total_alive = 0
    for i, c in enumerate(convs):
        alive = alive_out_channels(c)
        dead  = c.out_channels - len(alive)
        w_total = c.weight_mask.numel()
        w_alive = int(c.weight_mask.sum().item())
        sp = 1 - w_alive / w_total
        total_w += w_total; total_alive += w_alive
        print(f"  conv{i:2d}: {c.in_channels:4d}→{c.out_channels:4d} ch  "
              f"dead_out={dead:4d}  w_sparsity={sp:.1%}")

    for i, fc in enumerate(fcs):
        alive = alive_linear_neurons(fc)
        dead  = fc.out_features - len(alive)
        w_total = fc.weight_mask.numel()
        w_alive = int(fc.weight_mask.sum().item())
        sp = 1 - w_alive / w_total
        total_w += w_total; total_alive += w_alive
        label = f"fc{i+1}"
        print(f"  {label:6s}: {fc.in_features:6d}→{fc.out_features:5d}  "
              f"dead_out={dead:4d}  w_sparsity={sp:.1%}")

    print(f"  TOTAL: {total_alive:,} / {total_w:,} active weights  "
          f"({1 - total_alive/total_w:.2%} sparse)")


def print_compact_summary(compact: nn.Module, input_size: int):
    """Print compact model architecture with MACs estimate."""
    print("\n── Compact model layer summary ─────────────────────────────────")
    total_params = sum(p.numel() for p in compact.parameters())
    total_macs   = 0

    h = w = input_size
    for layer in compact.features:
        if isinstance(layer, nn.Conv2d):
            # Output spatial size (assuming same padding)
            out_h = (h + 2*layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
            out_w = (w + 2*layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
            macs = layer.in_channels * layer.out_channels * layer.kernel_size[0] * layer.kernel_size[1] * out_h * out_w
            total_macs += macs
            print(f"  Conv2d({layer.in_channels:4d}→{layer.out_channels:4d})  "
                  f"{h}×{w} → {out_h}×{out_w}  {macs/1e6:.1f}M MACs")
            h, w = out_h, out_w
        elif isinstance(layer, nn.MaxPool2d):
            h //= 2; w //= 2

    # Avgpool to 7×7
    h = w = 7
    flat = compact.classifier[0].in_features
    for layer in compact.classifier:
        if isinstance(layer, nn.Linear):
            macs = layer.in_features * layer.out_features
            total_macs += macs
            print(f"  Linear({layer.in_features:6d}→{layer.out_features:5d})  "
                  f"{macs/1e6:.2f}M MACs")

    print(f"\n  Total params : {total_params:,}  ({total_params*4/1024**2:.1f} MB fp32)")
    print(f"  Total MACs   : {total_macs/1e6:.1f}M")

    # Compare to original VGG16
    if input_size == 224:
        orig_macs = 15_470_000_000
    else:  # 32×32
        orig_macs = 320_000_000   # rough estimate
    print(f"  Speedup vs dense VGG16: ~{orig_macs/total_macs:.1f}× fewer MACs")


def export_onnx(compact: nn.Module, output_path: str, input_size: int, device: str):
    compact.eval()
    dummy = torch.randn(1, 3, input_size, input_size, device=device)
    torch.onnx.export(
        compact,
        dummy,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        verbose=False,
    )
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"\n  ONNX exported → {output_path}  ({size_mb:.1f} MB)")


def verify_onnx(onnx_path: str, test_loader, device: str, n_batches: int = 10):
    """Quick sanity check: run a few batches through onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [skip] onnxruntime not installed — skipping ONNX verification")
        return None

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    correct = total = 0
    for i, (x, y) in enumerate(test_loader):
        if i >= n_batches:
            break
        out = sess.run(["logits"], {"input": x.numpy()})[0]
        preds = np.argmax(out, axis=1)
        correct += (preds == y.numpy()).sum()
        total   += len(y)
    acc = correct / total
    print(f"  ONNX accuracy ({n_batches} batches, CPU): {acc:.4f}")
    return acc


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract compact VGG16 and export to ONNX")
    parser.add_argument("--checkpoint",  default="/home/petty/.openclaw/workspace-ai-research/vgg16_pruned_and_compressed.pt",
                        help="Path to masked checkpoint (.pt)")
    parser.add_argument("--output-dir",  default="/home/petty/pruning-research/vgg16-fisher",
                        help="Directory to write outputs")
    parser.add_argument("--input-size",  type=int, default=224,
                        choices=[32, 224], help="Input image size (224 for ImageNet norm, 32 for CIFAR-10 native)")
    parser.add_argument("--data-root",   default="/home/petty/.openclaw/workspace-ai-research/data")
    parser.add_argument("--device",      default="cpu", help="cpu or cuda:N")
    parser.add_argument("--no-onnx",     action="store_true", help="Skip ONNX export")
    parser.add_argument("--verify",      action="store_true", help="Verify ONNX with onnxruntime")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(args.checkpoint))[0]

    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    model, cfg = load_masked_model(args.checkpoint, args.device)

    # ── Accuracy of sparse model ────────────────────────────────────────────────
    cfg.data_root = args.data_root
    cfg.image_size = args.input_size
    _, test_loader = build_cifar10_loaders(cfg)
    sparse_loss, sparse_acc = evaluate(model, test_loader, args.device)
    print(f"\nSparse model accuracy: {sparse_acc:.4f}  (loss={sparse_loss:.4f})")
    print_sparse_summary(model)

    # ── Compress ────────────────────────────────────────────────────────────────
    print("\nExtracting compact model...")
    compact, meta = compress_masked_vgg16(model, num_classes=10)
    compact.eval()

    compact_loss, compact_acc = evaluate(compact, test_loader, args.device)
    print(f"Compact model accuracy: {compact_acc:.4f}  (loss={compact_loss:.4f})")
    assert abs(compact_acc - sparse_acc) < 0.002, \
        f"Accuracy mismatch after compression! sparse={sparse_acc:.4f} compact={compact_acc:.4f}"
    print("✓ Accuracy preserved within tolerance")

    print_compact_summary(compact, args.input_size)

    # ── Save compact model ──────────────────────────────────────────────────────
    compact_pt = os.path.join(args.output_dir, f"{ckpt_stem}_compact.pt")
    torch.save({
        "compact_state_dict": compact.state_dict(),
        "meta": {k: [t.cpu() for t in v] if isinstance(v, list) else v.cpu()
                 for k, v in meta.items()},
        "accuracy": compact_acc,
        "input_size": args.input_size,
    }, compact_pt)
    size_mb = os.path.getsize(compact_pt) / 1024**2
    print(f"\n  PyTorch compact model saved → {compact_pt}  ({size_mb:.1f} MB)")

    # ── ONNX export ─────────────────────────────────────────────────────────────
    if not args.no_onnx:
        onnx_path = os.path.join(args.output_dir, f"{ckpt_stem}_compact.onnx")
        export_onnx(compact, onnx_path, args.input_size, args.device)

        if args.verify:
            verify_onnx(onnx_path, test_loader, args.device)

    # ── Pi5 deployment notes ────────────────────────────────────────────────────
    print("""
── Pi5 deployment notes ───────────────────────────────────────────
  Runtime options:
    1. ONNX Runtime (recommended): pip install onnxruntime
       sess = ort.InferenceSession("model.onnx")
       out  = sess.run(["logits"], {"input": img_array})[0]

    2. TFLite: convert ONNX → TFLite via onnx-tf + tf-lite-converter
       (adds ~30min conversion but runs on Pi5's Mali GPU if available)

  Preprocessing (must match training):
    - Resize to 224×224 (or 32×32 if using --input-size 32)
    - Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    - Layout: NCHW float32

  Pi5 RAM: 8GB max — compact model fits easily in RAM
  Pi5 CPU: Cortex-A76, 4 cores — use OMP_NUM_THREADS=4
""")


if __name__ == "__main__":
    main()
