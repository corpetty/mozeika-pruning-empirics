#!/usr/bin/env python3
"""
Pi5 MNIST inference — exp36 LeNet compact INT8
Usage: python3 infer.py <image.png> [image2.png ...]
       python3 infer.py --invert <image.png> [image2.png ...]
       python3 infer.py --benchmark

Flags:
  --invert     Invert image colours before inference.
               Use when your digit is black-on-white (e.g. scanned paper).
               MNIST digits are white-on-black; skip this for native MNIST images.
  --benchmark  Run 1000 inference iterations and report latency/throughput.
"""
import sys, os, time
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MODEL = os.path.join(os.path.dirname(__file__), "lenet_compact_int8_static.onnx")

def preprocess(path, invert=False):
    img = Image.open(path).convert("L").resize((28, 28))
    if invert:
        img = ImageOps.invert(img)
    arr = np.array(img, dtype=np.float32) / 255.0
    # No normalization — model trained without it
    return arr.reshape(1, 1, 28, 28)

def main():
    args = sys.argv[1:]
    invert = "--invert" in args
    args = [a for a in args if a != "--invert"]

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name

    if "--benchmark" in args:
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

    for path in args:
        x = preprocess(path, invert=invert)
        logits = sess.run(None, {in_name: x})[0][0]
        pred = int(np.argmax(logits))
        conf = float(np.exp(logits[pred]) / np.exp(logits).sum())
        print(f"{path}: {pred}  (conf {conf*100:.1f}%)")

if __name__ == "__main__":
    main()
