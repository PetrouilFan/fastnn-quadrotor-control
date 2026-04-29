#!/usr/bin/env python3
"""
FastNN Inference for Quadrotor Control

Loads the trained SAC actor and runs inference using FastNN for high-speed
CPU inference on Raspberry Pi 5.

Model architecture: 51 -> 256 -> 256 -> 4 (ReLU activation)
"""

import numpy as np
import time
import sys
import os
import json

# Add fastnn to path
sys.path.insert(0, os.path.expanduser('~/fastnn'))

import fastnn as fnn


def make_tensor(arr):
    """Create FastNN tensor from numpy array."""
    flat = arr.flatten()
    return fnn.tensor(flat, shape=arr.shape)


class FastNNQuadrotorController:
    """FastNN-based quadrotor controller for inference."""

    def __init__(self, weights_path):
        """Initialize controller with model weights.

        Args:
            weights_path: Path to JSON file with actor weights
        """
        # Load weights from JSON
        with open(weights_path, 'r') as f:
            state_dict = json.load(f)

        # PyTorch Linear stores weights as (out, in) and computes y = xW^T + b
        # FastNN matmul(x, w) expects w to be (in, out)
        # So we transpose the weight matrices

        # Layer 1: 51 -> 256 (PyTorch: (256, 51)) -> FastNN: (51, 256)
        self.w1 = make_tensor(np.array(state_dict['latent_pi.0.weight']).T)
        self.b1 = make_tensor(np.array(state_dict['latent_pi.0.bias']))

        # Layer 2: 256 -> 256 (PyTorch: (256, 256)) -> FastNN: (256, 256)
        self.w2 = make_tensor(np.array(state_dict['latent_pi.2.weight']).T)
        self.b2 = make_tensor(np.array(state_dict['latent_pi.2.bias']))

        # Output: 256 -> 4 (PyTorch: (4, 256)) -> FastNN: (256, 4)
        self.w_out = make_tensor(np.array(state_dict['mu.weight']).T)
        self.b_out = make_tensor(np.array(state_dict['mu.bias']))

        # Log std (learned parameter, for stochastic inference)
        self.log_std = np.array(state_dict['log_std.weight'])

        print(f'Loaded weights from {weights_path}')
        print(f'  W1: {self.w1.shape}, b1: {self.b1.shape}')
        print(f'  W2: {self.w2.shape}, b2: {self.b2.shape}')
        print(f'  W_out: {self.w_out.shape}, b_out: {self.b_out.shape}')

    def forward(self, x):
        """Run forward pass.

        Args:
            x: Input tensor (51 dims) or (batch, 51)

        Returns:
            action: Motor commands (4 dims)
        """
        x = fnn.tensor(x.flatten(), shape=x.shape if x.ndim > 1 else (1, 51))

        # Layer 1: Linear + ReLU
        x = fnn.add(fnn.matmul(x, self.w1), self.b1)
        x = fnn.relu(x)

        # Layer 2: Linear + ReLU
        x = fnn.add(fnn.matmul(x, self.w2), self.b2)
        x = fnn.relu(x)

        # Output: Linear (no activation - we output raw action)
        action = fnn.add(fnn.matmul(x, self.w_out), self.b_out)

        return action.numpy()

    def predict(self, state, deterministic=True):
        """Predict action from state.

        Args:
            state: 51-dim state vector
            deterministic: If True, use mean action; else add noise

        Returns:
            action: 4-dim motor command
        """
        state = np.array(state, dtype=np.float32)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        action = self.forward(state)

        if deterministic:
            return action[0]
        else:
            std = np.exp(self.log_std)
            noise = np.random.randn(4) * std
            return action[0] + noise

    def batch_predict(self, states):
        """Batch prediction for multiple states."""
        states = np.array(states, dtype=np.float32)
        return self.forward(states)


def benchmark(controller, n_iterations=1000, warmup=100):
    """Benchmark inference speed."""
    # Create random input
    state = np.random.randn(51).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        controller.predict(state)

    # Benchmark
    start = time.time()
    for _ in range(n_iterations):
        controller.predict(state)
    elapsed = time.time() - start

    latency = elapsed / n_iterations * 1000  # ms
    throughput = n_iterations / elapsed  # inferences/sec

    return latency, throughput


def main():
    import argparse
    parser = argparse.ArgumentParser(description='FastNN Quadrotor Inference')
    parser.add_argument('--weights', type=str,
                        default='~/fastnn/models/actor_weights.json',
                        help='Path to model weights')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark')
    parser.add_argument('--iterations', type=int, default=10000,
                        help='Benchmark iterations')
    args = parser.parse_args()

    # Expand path
    weights_path = os.path.expanduser(args.weights)

    if not os.path.exists(weights_path):
        print(f'Error: weights file not found at {weights_path}')
        return 1

    # Initialize controller
    print('Initializing FastNN controller...')
    controller = FastNNQuadrotorController(weights_path)

    if args.benchmark:
        print(f'\nRunning benchmark ({args.iterations} iterations)...')
        latency, throughput = benchmark(controller, n_iterations=args.iterations)

        print(f'\n=== Results ===')
        print(f'Latency: {latency:.4f} ms per inference')
        print(f'Throughput: {throughput:.1f} inferences/sec')
        print(f'\nFor 400 Hz control loop: {1000/400:.2f} ms available per step')
    else:
        # Quick test
        state = np.zeros(51, dtype=np.float32)
        action = controller.predict(state)
        print(f'\nTest prediction: {action}')

    return 0


if __name__ == '__main__':
    sys.exit(main())