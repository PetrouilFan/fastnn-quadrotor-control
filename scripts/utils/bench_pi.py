#!/usr/bin/env python3
"""Benchmark FastNN vs PyTorch on Pi 5"""

import fastnn as fnn
import torch
import time

def bench_fn(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[iters // 2]  # median

sizes = [(256, 256), (512, 512), (1024, 1024), (4096, 4096)]

print("=" * 60)
print("FastNN (Rust) vs PyTorch (Python) - Pi 5 Benchmark")
print("=" * 60)
print()

print("=== FastNN (Rust, single-threaded) ===")
print(f"{'Size':<12} {'Time':>10} {'GFLOP/s':>12}")
for m, k in sizes:
    model = fnn.Sequential([fnn.Linear(k, m), fnn.ReLU()])
    x = fnn.randn([1, k])  # 2D tensor

    ms = bench_fn(lambda: model(x))
    gflops = (2 * m * k) / ms / 1e6
    print(f"{m}x{k:<7} {ms:>10.3f}ms {gflops:>11.2f}")

print()
print("=== PyTorch (Python/GIL, single-threaded) ===")
print(f"{'Size':<12} {'Time':>10} {'GFLOP/s':>12}")
for m, k in sizes:
    model = torch.nn.Linear(k, m)
    x = torch.randn(1, k)  # 2D tensor

    ms = bench_fn(lambda: model(x))
    gflops = (2 * m * k) / ms / 1e6
    print(f"{m}x{k:<7} {ms:>10.3f}ms {gflops:>11.2f}")

print()
print("=== Speedup: FastNN vs PyTorch ===")
print(f"{'Size':<12} {'Speedup':>10}")
for m, k in sizes:
    # FastNN
    fnn_model = fnn.Sequential([fnn.Linear(k, m), fnn.ReLU()])
    fnn_x = fnn.randn([1, k])
    fnn_ms = bench_fn(lambda: fnn_model(fnn_x))

    # PyTorch
    pt_model = torch.nn.Linear(k, m)
    pt_x = torch.randn(1, k)
    pt_ms = bench_fn(lambda: pt_model(pt_x))

    speedup = pt_ms / fnn_ms
    print(f"{m}x{k:<7} {speedup:>10.2f}x")

print()
print("=" * 60)
