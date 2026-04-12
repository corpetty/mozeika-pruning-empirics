# Exp 35 VGG16 CIFAR-10 — Pi5 Deployment Package

## Model Info

| Property | Value |
|---|---|
| Experiment | 35 (Per-layer Glauber threshold, T=0 OBD) |
| Architecture | VGG16 (compact — dead filters removed) |
| Sparsity | 98.40% |
| Accuracy | **94.14%** (FP32, fine-tuned) / **~91.33%** (INT8 est.) |
| Dense baseline | 89.94% |
| Format | ONNX INT8 (static quantized) / FP32 |
| Size | **13 MB** (INT8) / **49 MB** (FP32) |
| Estimated inference | **~427 ms/image** (INT8) / **~710 ms/image** (FP32) on Pi5 |
| Params | ~12.6M (compact) |

## Files

- `vgg16_exp35_finetuned_compact_int8.onnx` — INT8 model (use this for speed)
- `vgg16_exp35_finetuned_compact_fp32_single.onnx` + `.data` — FP32 model (higher accuracy, slower)
- `infer_exp35.py` — inference script with benchmark mode

## Setup

```bash
pip install onnxruntime pillow numpy
```

## Usage

```bash
# Single image (defaults to INT8)
python3 infer_exp35.py photo.jpg

# Specify FP32 model
python3 infer_exp35.py photo.jpg --model vgg16_exp35_finetuned_compact_fp32_single.onnx

# Benchmark (100 runs, random input)
python3 infer_exp35.py --benchmark
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
- FP32 model is ~1.7× slower but offers ~2.8pp higher accuracy
