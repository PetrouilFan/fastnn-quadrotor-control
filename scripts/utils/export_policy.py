#!/usr/bin/env python3
"""
Export SB3 SAC policy weights to FastNN format.
Extracts the actor network weights and exports as PyTorch tensors
for loading into FastNN's Rust-based inference.

With asymmetric architecture: Actor takes 52-dim deployable obs only.
"""

import torch
import zipfile
import json
import os
import numpy as np


def extract_sb3_policy(model_zip_path, output_dir=None, deployable_dim=52):
    """
    Extract actor weights from SB3 SAC zip checkpoint.
    Architecture: 52->256->256->128->4 (deployable obs only)
    """
    if output_dir is None:
        output_dir = "fastnn_exported"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading SB3 model: {model_zip_path}")

    with zipfile.ZipFile(model_zip_path) as z:
        policy = torch.load(z.open('policy.pth'), map_location='cpu', weights_only=False)

    state_dict = policy

    # Extract actor layers
    layers = {}
    for key, value in state_dict.items():
        if key.startswith('actor.'):
            clean_key = key.replace('actor.', '')
            layers[clean_key] = value.numpy()

    print(f"\nExtracted {len(layers)} actor layers:")
    for k, v in sorted(layers.items()):
        print(f"  {k}: {v.shape}")

    # The actor uses DeployableExtractor which slices obs to first 52 dims
    # The actor MLP (latent_pi) takes features_dim=52 as input
    weight_spec = {
        "input_dim": deployable_dim,
        "hidden_dims": [256, 256, 128],
        "output_dim": 4,
        "activation": "relu",
        "deployable_dim": deployable_dim,
        "full_obs_dim": 60,
        "note": "Actor takes first 52 dims (deployable) of 60-dim observation. "
                "Critic uses all 60 dims. At deployment, only 52-dim sensor data needed.",
        "layer_names": [
            "fc1_weight", "fc1_bias",
            "fc2_weight", "fc2_bias",
            "fc3_weight", "fc3_bias",
            "fc4_weight", "fc4_bias"
        ]
    }

    # Save weight spec
    spec_path = os.path.join(output_dir, "arch.json")
    with open(spec_path, "w") as f:
        json.dump(weight_spec, f, indent=2)
    print(f"\nSaved: {spec_path}")

    # FC1: 52 -> 256 (deployable input)
    np.save(os.path.join(output_dir, "fc1_weight.npy"), layers['latent_pi.0.weight'])
    np.save(os.path.join(output_dir, "fc1_bias.npy"), layers['latent_pi.0.bias'])

    # FC2: 256 -> 256
    np.save(os.path.join(output_dir, "fc2_weight.npy"), layers['latent_pi.2.weight'])
    np.save(os.path.join(output_dir, "fc2_bias.npy"), layers['latent_pi.2.bias'])

    # FC3: 256 -> 128
    np.save(os.path.join(output_dir, "fc3_weight.npy"), layers['latent_pi.4.weight'])
    np.save(os.path.join(output_dir, "fc3_bias.npy"), layers['latent_pi.4.bias'])

    # FC4 (output): 128 -> 4
    np.save(os.path.join(output_dir, "fc4_weight.npy"), layers['mu.weight'])
    np.save(os.path.join(output_dir, "fc4_bias.npy"), layers['mu.bias'])

    print(f"Saved weights to: {output_dir}/")

    # Also save a combined .pt file for easy loading
    combined = {
        'fc1_weight': torch.from_numpy(layers['latent_pi.0.weight']),
        'fc1_bias': torch.from_numpy(layers['latent_pi.0.bias']),
        'fc2_weight': torch.from_numpy(layers['latent_pi.2.weight']),
        'fc2_bias': torch.from_numpy(layers['latent_pi.2.bias']),
        'fc3_weight': torch.from_numpy(layers['latent_pi.4.weight']),
        'fc3_bias': torch.from_numpy(layers['latent_pi.4.bias']),
        'fc4_weight': torch.from_numpy(layers['mu.weight']),
        'fc4_bias': torch.from_numpy(layers['mu.bias']),
    }
    torch.save(combined, os.path.join(output_dir, "actor_weights.pt"))
    print(f"Saved combined: {output_dir}/actor_weights.pt")

    # Print shapes for verification
    print("\n--- Weight Summary ---")
    print(f"FC1: {layers['latent_pi.0.weight'].shape} -> ReLU -> {layers['latent_pi.2.weight'].shape} -> ReLU")
    print(f"    -> {layers['latent_pi.4.weight'].shape} -> ReLU -> {layers['mu.weight'].shape}")
    print(f"    Input: {deployable_dim} (deployable obs only)")
    print(f"    Total params: {sum(v.size for v in layers.values())}")

    # Verify input dimension matches deployable_dim
    fc1_in = layers['latent_pi.0.weight'].shape[1]
    if fc1_in != deployable_dim:
        print(f"\n  WARNING: FC1 input dim ({fc1_in}) != deployable_dim ({deployable_dim})")
        print(f"  The model may not have been trained with asymmetric architecture.")
        print(f"  Update deployable_dim to {fc1_in} or retrain with AsymmetricSACPolicy.")
    else:
        print(f"\n  ✓ FC1 input dimension matches deployable_dim ({deployable_dim})")

    return output_dir


def benchmark_pytorch_actor(model_path, deployable_dim=52, n_warmup=100, n_runs=1000):
    """Benchmark PyTorch actor inference latency."""
    import time

    print(f"\n{'='*60}")
    print("PyTorch Actor Inference Benchmark")
    print(f"{'='*60}")

    with zipfile.ZipFile(model_path) as z:
        policy = torch.load(z.open('policy.pth'), map_location='cpu', weights_only=False)

    # Build actor network matching SB3 structure (52-dim input)
    class Actor(torch.nn.Module):
        def __init__(self, input_dim=52):
            super().__init__()
            self.fc1 = torch.nn.Linear(input_dim, 256)
            self.fc2 = torch.nn.Linear(256, 256)
            self.fc3 = torch.nn.Linear(256, 128)
            self.fc4 = torch.nn.Linear(128, 4)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            return self.fc4(x)

    # Auto-detect input dim from weights
    fc1_weight = policy.get('actor.latent_pi.0.weight')
    if fc1_weight is not None:
        actual_input_dim = fc1_weight.shape[1]
    else:
        actual_input_dim = deployable_dim

    actor = Actor(input_dim=actual_input_dim)

    # Load weights
    sd = policy
    actor.fc1.weight.data = sd['actor.latent_pi.0.weight']
    actor.fc1.bias.data = sd['actor.latent_pi.0.bias']
    actor.fc2.weight.data = sd['actor.latent_pi.2.weight']
    actor.fc2.bias.data = sd['actor.latent_pi.2.bias']
    actor.fc3.weight.data = sd['actor.latent_pi.4.weight']
    actor.fc3.bias.data = sd['actor.latent_pi.4.bias']
    actor.fc4.weight.data = sd['actor.mu.weight']
    actor.fc4.bias.data = sd['actor.mu.bias']

    actor.eval()
    x = torch.randn(1, actual_input_dim)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = actor(x)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = actor(x)
            latencies.append((time.perf_counter() - t0) * 1e6)

    latencies.sort()
    mean_lat = latencies[n_runs // 2]
    p99_lat = latencies[int(n_runs * 0.99)]

    print(f"\nPyTorch Actor ({actual_input_dim}->256->256->128->4):")
    print(f"  Median latency: {mean_lat:.2f} us")
    print(f"  P99 latency: {p99_lat:.2f} us")

    # FLOPs: 52*256 + 256*256 + 256*128 + 128*4 = 13312 + 65536 + 32768 + 512 = 112128
    # (or 60*256 + ... if using full obs)
    gflops = (actual_input_dim * 256 + 256 * 256 + 256 * 128 + 128 * 4) / mean_lat / 1e6
    print(f"  GFLOP/s: {gflops:.2f}")

    return mean_lat, p99_lat


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export SB3 SAC weights to FastNN")
    parser.add_argument("model", nargs="?", default="models/seed_0/final.zip")
    parser.add_argument("--output", "-o", default="fastnn_exported")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    extract_sb3_policy(args.model, args.output)

    if args.benchmark:
        benchmark_pytorch_actor(args.model)