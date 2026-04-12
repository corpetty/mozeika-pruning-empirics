#!/usr/bin/env python3
"""
Exp 35 VGG16 CIFAR-10 — Pi5 Inference
98.40% sparse (Glauber threshold), fine-tuned, INT8/FP32 ONNX
Accuracy: 94.14% (FP32) / 91.33% (INT8 est.) | Size: 49 MB (FP32) / 13 MB (INT8)

Requirements:
  pip install onnxruntime pillow numpy

Usage:
  python3 infer_exp35.py image.jpg
  python3 infer_exp35.py image.jpg --model vgg16_exp35_finetuned_compact_int8.onnx
  python3 infer_exp35.py --benchmark          # time 100 runs on a random input
"""

import sys
import os
import argparse
import time
import numpy as np

CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ImageNet normalization (used during training)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess(image_path: str) -> np.ndarray:
    """Load and preprocess an image for inference."""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # HWC, [0,1]
    arr = (arr - MEAN) / STD                          # normalize
    arr = arr.transpose(2, 0, 1)                      # HWC → CHW
    arr = arr[np.newaxis, ...]                         # → NCHW
    return arr


def load_session(model_path: str):
    import onnxruntime as ort
    # Use 4 threads — matches Pi5's 4 Cortex-A76 cores
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    return sess


def predict(sess, input_array: np.ndarray):
    logits = sess.run(["logits"], {"input": input_array})[0][0]
    probs  = softmax(logits)
    top5   = np.argsort(probs)[::-1][:5]
    return [(CLASSES[i], float(probs[i])) for i in top5]


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def benchmark(sess, n: int = 100):
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    # Warmup
    for _ in range(3):
        sess.run(["logits"], {"input": dummy})
    # Timed runs
    t0 = time.perf_counter()
    for _ in range(n):
        sess.run(["logits"], {"input": dummy})
    elapsed = time.perf_counter() - t0
    ms_per  = elapsed / n * 1000
    print(f"Benchmark: {n} runs in {elapsed:.2f}s → {ms_per:.1f} ms/image  ({1000/ms_per:.1f} img/s)")


def main():
    parser = argparse.ArgumentParser(description="Exp 35 VGG16 CIFAR-10 Pi5 inference")
    parser.add_argument("image",        nargs="?", help="Path to input image")
    parser.add_argument("--model",      default=os.path.join(
                            os.path.dirname(__file__),
                            "vgg16_exp35_finetuned_compact_int8.onnx"),
                        help="Path to ONNX model")
    parser.add_argument("--benchmark",  action="store_true", help="Run timing benchmark")
    parser.add_argument("--n",          type=int, default=100, help="Benchmark iterations")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}")
        print("Place vgg16_exp35_finetuned_compact_int8.onnx next to this script.")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    sess = load_session(args.model)
    print("Model loaded.")

    if args.benchmark:
        benchmark(sess, args.n)
        return

    if not args.image:
        parser.print_help()
        return

    arr = preprocess(args.image)
    t0  = time.perf_counter()
    results = predict(sess, arr)
    ms = (time.perf_counter() - t0) * 1000

    print(f"\nPrediction ({ms:.1f} ms):")
    for i, (cls, prob) in enumerate(results):
        bar = "█" * int(prob * 30)
        print(f"  {i+1}. {cls:<12}  {prob*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
