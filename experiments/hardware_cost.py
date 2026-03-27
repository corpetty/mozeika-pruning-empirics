"""
hardware_cost.py — Microbenchmark: latency overhead of subspace projection + quantization.

Measures CUDA latency for:
  (a) Plain uniform quantization on (T, 128)
  (b) Project-then-quantize: matmul with (128, k) projection + quantize in k dims
  (c) PolarQuant: rotation with (128, 128) or (k, k) matrix + quantize

Runs at batch sizes T = 1, 8, 32, 128, 512, 1024.
Reports median over 200 iterations after warmup.

Usage:
    /home/petty/torch-env/bin/python3 experiments/hardware_cost.py
"""

import sys
import csv
import time
import numpy as np
import torch
from pathlib import Path

D_HEAD = 128
K_VALUES = [32, 64]
BATCH_SIZES = [1, 8, 32, 128, 512, 1024]
N_ITERS = 200
N_WARMUP = 50
N_BITS = 4  # bits for quantization


def quantize_uniform_torch(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Uniform scalar quantization (simulate quantize + dequantize)."""
    n_levels = 2 ** n_bits
    x_min = x.min(dim=0, keepdim=True).values
    x_max = x.max(dim=0, keepdim=True).values
    scale = (x_max - x_min) / (n_levels - 1)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    x_int = torch.round((x - x_min) / scale).clamp(0, n_levels - 1)
    return x_int * scale + x_min


def benchmark_plain_quantize(T: int, d: int, n_bits: int, device: str) -> float:
    """Benchmark plain uniform quantization on (T, d)."""
    x = torch.randn(T, d, device=device, dtype=torch.float16)
    times = []

    for i in range(N_WARMUP + N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = quantize_uniform_torch(x.float(), n_bits)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append(t1 - t0)

    return float(np.median(times))


def benchmark_project_quantize(T: int, d: int, k: int, n_bits: int, device: str) -> float:
    """Benchmark: project (T,d) to (T,k), then quantize (T,k)."""
    x = torch.randn(T, d, device=device, dtype=torch.float16)
    mean = torch.randn(d, device=device, dtype=torch.float16)
    U_k = torch.randn(d, k, device=device, dtype=torch.float16)
    # Orthogonalize
    U_k = torch.linalg.qr(U_k.float())[0][:, :k].half()
    times = []

    for i in range(N_WARMUP + N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Subtract mean, project, quantize, reconstruct
        xc = x - mean
        x_proj = xc @ U_k              # (T, k) — projection
        x_proj_q = quantize_uniform_torch(x_proj.float(), n_bits)  # quantize in k-dim
        x_recon = x_proj_q.half() @ U_k.T + mean  # reconstruct to d-dim
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append(t1 - t0)

    return float(np.median(times))


def benchmark_polarquant(T: int, d: int, n_bits: int, device: str) -> float:
    """Benchmark PolarQuant: rotation (d,d) + quantize + inverse rotation."""
    x = torch.randn(T, d, device=device, dtype=torch.float16)
    R = torch.randn(d, d, device=device, dtype=torch.float16)
    R = torch.linalg.qr(R.float())[0].half()
    times = []

    for i in range(N_WARMUP + N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xr = x @ R.T                    # rotate
        xr_q = quantize_uniform_torch(xr.float(), n_bits)  # quantize
        x_recon = xr_q.half() @ R       # inverse rotate
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append(t1 - t0)

    return float(np.median(times))


def benchmark_subspace_polarquant(T: int, d: int, k: int, n_bits: int, device: str) -> float:
    """Benchmark subspace PolarQuant: project to k-dim, rotation (k,k) + quantize, reconstruct."""
    x = torch.randn(T, d, device=device, dtype=torch.float16)
    mean = torch.randn(d, device=device, dtype=torch.float16)
    U_k = torch.randn(d, k, device=device, dtype=torch.float16)
    U_k = torch.linalg.qr(U_k.float())[0][:, :k].half()
    R_k = torch.randn(k, k, device=device, dtype=torch.float16)
    R_k = torch.linalg.qr(R_k.float())[0].half()
    times = []

    for i in range(N_WARMUP + N_ITERS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        xc = x - mean
        x_proj = xc @ U_k              # project to k-dim
        xr = x_proj @ R_k.T            # rotate in k-dim
        xr_q = quantize_uniform_torch(xr.float(), n_bits)  # quantize
        x_recon = (xr_q.half() @ R_k) @ U_k.T + mean  # rotate back + reconstruct
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= N_WARMUP:
            times.append(t1 - t0)

    return float(np.median(times))


def main():
    import os
    os.chdir(Path(__file__).resolve().parent.parent)
    Path("results").mkdir(exist_ok=True)

    device = "cuda"
    print("=" * 80)
    print("Hardware Cost Microbenchmark")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"d_head={D_HEAD}, k_values={K_VALUES}, batch_sizes={BATCH_SIZES}")
    print(f"n_bits={N_BITS}, warmup={N_WARMUP}, iters={N_ITERS}")
    print("=" * 80)

    rows = []

    for T in BATCH_SIZES:
        print(f"\n--- T={T} ---")

        # (a) Plain quantize
        lat = benchmark_plain_quantize(T, D_HEAD, N_BITS, device)
        print(f"  plain_quantize:        {lat*1e6:8.1f} us")
        rows.append({'batch_size': T, 'operation': 'plain_quantize', 'k': D_HEAD,
                     'latency_us': lat * 1e6})

        # (b) PolarQuant (full-dim rotation)
        lat = benchmark_polarquant(T, D_HEAD, N_BITS, device)
        print(f"  polarquant_128:        {lat*1e6:8.1f} us")
        rows.append({'batch_size': T, 'operation': 'polarquant_128', 'k': D_HEAD,
                     'latency_us': lat * 1e6})

        for k in K_VALUES:
            # (c) Project + quantize (no rotation in subspace)
            lat = benchmark_project_quantize(T, D_HEAD, k, N_BITS, device)
            print(f"  project_quantize_k{k}: {lat*1e6:8.1f} us")
            rows.append({'batch_size': T, 'operation': f'project_quantize_k{k}', 'k': k,
                         'latency_us': lat * 1e6})

            # (d) Full subspace PolarQuant (project + rotate in subspace + quantize + reconstruct)
            lat = benchmark_subspace_polarquant(T, D_HEAD, k, N_BITS, device)
            print(f"  subspace_polar_k{k}:   {lat*1e6:8.1f} us")
            rows.append({'batch_size': T, 'operation': f'subspace_polar_k{k}', 'k': k,
                         'latency_us': lat * 1e6})

    # Save CSV
    with open('results/hardware_cost.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to results/hardware_cost.csv")

    # Print summary table
    print("\n" + "=" * 80)
    print("LATENCY TABLE (microseconds)")
    print("=" * 80)

    operations = ['plain_quantize', 'polarquant_128']
    for k in K_VALUES:
        operations.extend([f'project_quantize_k{k}', f'subspace_polar_k{k}'])

    header = f"{'Operation':<26}  " + "  ".join(f"T={T:>5}" for T in BATCH_SIZES)
    print(header)
    print("-" * len(header))

    for op in operations:
        vals = []
        for T in BATCH_SIZES:
            r = [x for x in rows if x['batch_size'] == T and x['operation'] == op]
            if r:
                vals.append(f"{r[0]['latency_us']:7.1f}")
            else:
                vals.append("    N/A")
        print(f"{op:<26}  " + "  ".join(vals))

    # Overhead ratios
    print("\n" + "=" * 80)
    print("OVERHEAD RATIOS (relative to plain_quantize)")
    print("=" * 80)

    plain_latencies = {T: [x for x in rows if x['batch_size'] == T and x['operation'] == 'plain_quantize'][0]['latency_us']
                       for T in BATCH_SIZES}

    for op in operations[1:]:  # skip plain_quantize itself
        vals = []
        for T in BATCH_SIZES:
            r = [x for x in rows if x['batch_size'] == T and x['operation'] == op]
            if r:
                ratio = r[0]['latency_us'] / plain_latencies[T]
                vals.append(f"{ratio:6.2f}x")
            else:
                vals.append("   N/A")
        print(f"{op:<26}  " + "  ".join(f"{v:>7}" for v in vals))

    print("\nDone.")


if __name__ == "__main__":
    main()
