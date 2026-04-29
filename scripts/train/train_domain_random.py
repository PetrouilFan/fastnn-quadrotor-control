#!/usr/bin/env python3
"""
Domain Randomization Training for Wind-Robust Quadrotor Control

Key insight: Instead of collecting data from fixed wind conditions, we
randomize wind during training so the network sees diverse wind patterns.

This helps because:
1. The network learns to handle diverse wind, not just one realization
2. Forces the network to learn adaptive behavior
3. Acts as a form of regularization against overfitting to specific wind

Architecture: Same GRU as best performer (GRUController)

Usage:
    python train_domain_random.py --epochs 50 --wind-scale 2.0
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
from fastnn_quadrotor.utils.transformer_bc import GRUController
from fastnn_quadrotor.training.train_transformer_bc import BCController, evaluate


DEPLOYABLE_DIM = 52
ACTION_DIM = 4


class RandomizedEnv:
    """Wrapper that adds random wind to any environment."""

    def __init__(self, base_env, wind_scale=1.0):
        self.base_env = base_env
        self.wind_scale = wind_scale

    def reset(self, seed=None):
        obs, info = self.base_env.reset(seed=seed)
        # Add randomization after reset
        self._randomize_wind()
        return obs, info

    def _randomize_wind(self):
        """Apply randomized wind force."""
        # Randomize wind parameters within the stage's range but potentially scaled
        if hasattr(self.base_env, 'wind_force'):
            scale = self.wind_scale
            self.base_env.wind_force = np.random.uniform(
                -0.5 * scale, 0.5 * scale, size=3
            ).astype(np.float32)

    def __getattr__(self, name):
        return getattr(self.base_env, name)


def collect_randomized_data(n_episodes=200, stage=3, wind_scale=1.0, max_steps=500):
    """Collect data with randomized wind conditions."""
    states, actions = [], []

    base_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=False)
    env = RandomizedEnv(base_env, wind_scale=wind_scale)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            pd_action = env._cascaded_controller()
            states.append(obs[:DEPLOYABLE_DIM].copy())
            actions.append(pd_action.copy())
            obs, _, terminated, truncated, _ = env.step(np.zeros(4))

        if (ep + 1) % 50 == 0:
            print(f"  Episode {ep+1}/{n_episodes}")

    env.close()
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)


def train_bc(model, states, actions, epochs=30, batch_size=256, lr=1e-3, device="cuda"):
    """Train BC model."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(actions))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            preds, _ = model(batch_states)
            loss = nn.MSELoss()(preds, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss / n_batches:.6f}")

    return model


def main():
    print("=== Domain Randomization Training ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Test different wind scales
    results = {}

    for wind_scale in [1.0, 1.5, 2.0]:
        print(f"\n{'='*50}")
        print(f"Wind Scale: {wind_scale}x")
        print(f"{'='*50}")

        # Collect data with randomization
        print(f"Collecting randomized data (scale={wind_scale})...")
        states, actions = collect_randomized_data(
            n_episodes=200, stage=3, wind_scale=wind_scale
        )
        print(f"Collected {len(states)} transitions")

        # Normalize
        state_mean = states.mean(axis=0)
        state_std = states.std(axis=0) + 1e-8
        states_norm = (states - state_mean) / state_std

        # Train GRU
        print(f"Training GRU...")
        model = GRUController(state_dim=52, action_dim=4, hidden_dim=128, num_layers=2).to(device)
        model = train_bc(model, states_norm, actions, epochs=50, batch_size=256, device=device)

        # Evaluate
        bc = BCController(model, state_mean, state_std, device, model_type='gru')

        print(f"Evaluation:")
        stage_results = {}
        for stage in [1, 3, 4]:
            env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
            metrics = evaluate(env, bc, n_episodes=100)
            print(f"  Stage {stage}: success={metrics['success_rate']:.1%}, dist={metrics['mean_final_dist']:.3f}m")
            stage_results[stage] = metrics
            env.close()

        results[wind_scale] = stage_results

    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Wind Scale':<12} {'Stage 1':<10} {'Stage 3':<10} {'Stage 4':<10}")
    for scale, res in results.items():
        print(f"{scale:.1f}x{'':<9} {res[1]['success_rate']:>8.0%}  {res[3]['success_rate']:>8.0%}  {res[4]['success_rate']:>8.0%}")


if __name__ == "__main__":
    main()