# VGG16 CIFAR-10 — Pi5 Deployment Package

## Model Info

| Property | Value |
|---|---|
| Architecture | VGG16 (compact — dead filters removed) |
| Sparsity | 99.19% (OBD-pruned) |
| Accuracy | ~89.0% on CIFAR-10 test set |
| Dense baseline | 89.94% (gap: -0.94pp) |
| Format | ONNX INT8 (static quantized) |
| Size | 37 MB |
| Estimated inference | ~190 ms/image on Pi5 (Cortex-A76) |
| Params | 38.8M (compact), ~1.1M active |

## Files

- `vgg16_finetuned_99pct_compact_int8_static.onnx` — main model (use this)
- `vgg16_finetuned_99pct_compact.onnx` — FP32 fallback (148 MB, 89.08%)
- `infer.py` — inference script with benchmark mode

## Setup

```bash
pip install onnxruntime pillow numpy
```

## Usage

```bash
# Single image
python3 infer.py photo.jpg

# Specify model explicitly
python3 infer.py photo.jpg --model vgg16_finetuned_99pct_compact_int8_static.onnx

# Benchmark (100 runs, random input)
python3 infer.py --benchmark
python3 infer.py --benchmark --n 50
```

## CIFAR-10 Classes

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Preprocessing

Must match training:
- Resize to 224×224
- Normalize: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Layout: NCHW float32

## Performance Notes

- Set `OMP_NUM_THREADS=4` for best throughput on Pi5's 4 cores
- INT8 model uses `intra_op_num_threads=4` automatically
- FP32 fallback is ~3× slower but no accuracy loss from quantization
