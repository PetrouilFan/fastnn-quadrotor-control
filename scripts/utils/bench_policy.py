#!/usr/bin/env python3
"""
Benchmark FastNN vs PyTorch for quadrotor policy inference.

Architecture: 60 -> 256 -> 256 -> 128 -> 4 (with ReLU between layers)
"""

import sys
import time
import zipfile

# Add fastnn from llm/loader
sys.path.insert(0, '/home/petrouil/Projects/fastnn_projects/llm/loader')

import torch
import numpy as np
import fastnn as fnn


class PyTorchActor(torch.nn.Module):
    """Standard PyTorch actor network."""
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(60, 256)
        self.fc2 = torch.nn.Linear(256, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.fc4 = torch.nn.Linear(128, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


def load_actor_weights(policy_dict):
    """Load actor weights from SB3 policy OrderedDict.

    PyTorch stores weights as [out_features, in_features] (e.g., [256, 60]).
    FastNN expects [in_features, out_features] (e.g., [60, 256]).
    We transpose PyTorch weights for FastNN.
    """
    weights = {}
    # FC1: 60 -> 256
    weights['fc1_weight'] = policy_dict['actor.latent_pi.0.weight'].numpy().T.copy()
    weights['fc1_bias'] = policy_dict['actor.latent_pi.0.bias'].numpy()
    # FC2: 256 -> 256
    weights['fc2_weight'] = policy_dict['actor.latent_pi.2.weight'].numpy().T.copy()
    weights['fc2_bias'] = policy_dict['actor.latent_pi.2.bias'].numpy()
    # FC3: 256 -> 128
    weights['fc3_weight'] = policy_dict['actor.latent_pi.4.weight'].numpy().T.copy()
    weights['fc3_bias'] = policy_dict['actor.latent_pi.4.bias'].numpy()
    # FC4 (output): 128 -> 4
    weights['fc4_weight'] = policy_dict['actor.mu.weight'].numpy().T.copy()
    weights['fc4_bias'] = policy_dict['actor.mu.bias'].numpy()
    return weights


def benchmark_fn(fn, warmup=100, iters=1000):
    """Benchmark a function and return median latency in microseconds."""
    for _ in range(warmup):
        fn()
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - t0) * 1e6)
    latencies.sort()
    return latencies[iters // 2], latencies[int(iters * 0.99)]


def run_benchmarks(model_zip_path):
    """Run comprehensive benchmark comparing FastNN vs PyTorch."""

    # Load SB3 model
    print(f"Loading: {model_zip_path}")
    with zipfile.ZipFile(model_zip_path) as z:
        policy = torch.load(z.open('policy.pth'), map_location='cpu')
    print(f"Loaded policy with {len(policy)} tensors")

    weights = load_actor_weights(policy)

    # ============================================================
    # PyTorch Benchmark
    # ============================================================
    print("\n" + "=" * 60)
    print("PyTorch Actor Inference")
    print("=" * 60)

    pt_actor = PyTorchActor()
    # For PyTorch, use original (non-transposed) weights
    pt_actor.fc1.weight.data = policy['actor.latent_pi.0.weight']
    pt_actor.fc1.bias.data = policy['actor.latent_pi.0.bias']
    pt_actor.fc2.weight.data = policy['actor.latent_pi.2.weight']
    pt_actor.fc2.bias.data = policy['actor.latent_pi.2.bias']
    pt_actor.fc3.weight.data = policy['actor.latent_pi.4.weight']
    pt_actor.fc3.bias.data = policy['actor.latent_pi.4.bias']
    pt_actor.fc4.weight.data = policy['actor.mu.weight']
    pt_actor.fc4.bias.data = policy['actor.mu.bias']
    pt_actor.eval()

    x_pt = torch.randn(1, 60)

    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = pt_actor(x_pt)

    pt_median, pt_p99 = benchmark_fn(lambda: pt_actor(x_pt))
    print(f"PyTorch: median={pt_median:.2f}us, p99={pt_p99:.2f}us")

    gflops_pt = 114176 / pt_median / 1e6
    print(f"PyTorch GFLOP/s: {gflops_pt:.3f}")

    # ============================================================
    # FastNN Benchmark (Rust-based layers)
    # ============================================================
    print("\n" + "=" * 60)
    print("FastNN (Rust) Actor Inference")
    print("=" * 60)

    # Create FastNN layers with transposed weights
    def to_fnn(arr):
        return fnn.tensor_from_array(arr)

    fc1 = fnn.Linear.from_weights(to_fnn(weights['fc1_weight']), to_fnn(weights['fc1_bias']))
    fc2 = fnn.Linear.from_weights(to_fnn(weights['fc2_weight']), to_fnn(weights['fc2_bias']))
    fc3 = fnn.Linear.from_weights(to_fnn(weights['fc3_weight']), to_fnn(weights['fc3_bias']))
    fc4 = fnn.Linear.from_weights(to_fnn(weights['fc4_weight']), to_fnn(weights['fc4_bias']))

    # Create FastNN sequential model
    model = fnn.Sequential([
        fc1,
        fnn.ReLU(),
        fc2,
        fnn.ReLU(),
        fc3,
        fnn.ReLU(),
        fc4,
    ])

    x_fnn = fnn.randn([1, 60])

    # Warmup
    with fnn.no_grad():
        for _ in range(100):
            _ = model(x_fnn)

    fn_closure = lambda: model(x_fnn)
    fnn_median, fnn_p99 = benchmark_fn(fn_closure)
    print(f"FastNN: median={fnn_median:.2f}us, p99={fnn_p99:.2f}us")

    gflops_fnn = 114176 / fnn_median / 1e6
    print(f"FastNN GFLOP/s: {gflops_fnn:.3f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY: FastNN vs PyTorch (60->256->256->128->4)")
    print("=" * 60)
    print(f"{'Runtime':<15} {'Median':>12} {'P99':>12} {'Speedup':>10}")
    print("-" * 60)
    print(f"{'PyTorch':<15} {pt_median:>10.2f}us {pt_p99:>10.2f}us {'--':>10}")
    print(f"{'FastNN (Rust)':<15} {fnn_median:>10.2f}us {fnn_p99:>10.2f}us {pt_median/fnn_median:>9.2f}x")
    print("-" * 60)

    speedup = pt_median / fnn_median
    print(f"\nSpeedup: {speedup:.2f}x faster with FastNN")
    print(f"GFLOP/s improvement: {gflops_fnn:.2f} vs {gflops_pt:.2f}")

    return {
        'pytorch': {'median': pt_median, 'p99': pt_p99, 'gflops': gflops_pt},
        'fastnn': {'median': fnn_median, 'p99': fnn_p99, 'gflops': gflops_fnn},
        'speedup': speedup,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model", nargs="?",
                        default="archive/models/stage4_step140000.zip")
    args = parser.parse_args()

    results = run_benchmarks(args.model)
    print("\nBenchmark complete!")