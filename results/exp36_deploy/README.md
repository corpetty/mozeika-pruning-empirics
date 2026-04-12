# Exp36 LeNet Pi5 Deploy Package

## Model
- Architecture: 784 → 80 → 30 → 10 (dense compact, extracted from 784→122→35→10)
- Original sparsity: 97.64%
- FP32 test accuracy: 97.59%
- INT8 test accuracy: 97.39%
- Dense baseline: 98.24%
- Gap to dense: 0.85 pp

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
