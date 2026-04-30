#!/usr/bin/env python3
"""
Evaluate Behavioral Cloning (BC) model on quadrotor control.

Loads a pretrained BC model and evaluates it against the PD baseline.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class SimpleBCController(nn.Module):
    """Simple MLP for behavioral cloning - outputs raw actions."""

    def __init__(self, state_dim=52, action_dim=4, hidden_dims=[256, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def evaluate_controller(env, controller_fn, n_episodes=50, description="Controller"):
    """Evaluate a controller (fn returns action given obs)."""
    successes = 0
    survivals = 0
    rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        while not (terminated or truncated) and steps < 500:
            action = controller_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        if steps >= 500:
            successes += 1
            survivals += 1
        else:
            survivals += 1 if not terminated else 0

        rewards.append(episode_reward)

    return {
        "description": description,
        "success_rate": successes / n_episodes,
        "survival_rate": survivals / n_episodes,
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BC model")
    parser.add_argument("--model", type=str, default="models/simple_bc_best.pt",
                        help="Path to BC model")
    parser.add_argument("--episodes", type=int, default=50, help="Evaluation episodes")
    parser.add_argument("--stages", type=str, default="1,2,3,4",
                        help="Comma-separated stages to evaluate")
    args = parser.parse_args()

    stages = [int(s) for s in args.stages.split(",")]

    print("=" * 60)
    print("Behavioral Cloning (BC) Evaluation")
    print("=" * 60)

    # Load BC model
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
        model = SimpleBCController(state_dim=52, action_dim=4, hidden_dims=[256, 256, 128])
        model.load_state_dict(checkpoint["model_state_dict"])
        state_mean = checkpoint["state_mean"]
        state_std = checkpoint["state_std"]
        print(f"Loaded BC model from {args.model}")
        print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    else:
        print(f"WARNING: Model not found at {args.model}")
        return

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Move normalization tensors to same device as model
    state_mean_t = torch.FloatTensor(state_mean).to(device)
    state_std_t = torch.FloatTensor(state_std).to(device)

    def bc_controller(obs):
        """BC controller that outputs direct control values.

        Model outputs PD-style actions (thrust ~10N, torques ~0).
        Env expects action in [-1, 1] for direct control mode.
        Normalize: thrust centered at 10N → [-1, 1], torques divided by max.
        """
        with torch.no_grad():
            # Normalize state (all tensors on same device)
            state_raw = torch.FloatTensor(obs[:52]).to(device)
            state_norm = (state_raw - state_mean_t) / (state_std_t + 1e-8)
            state_norm = state_norm.unsqueeze(0)

            # Get action from model
            action = model(state_norm).squeeze(0).cpu().numpy()
            # action is [thrust, roll, pitch, yaw] in PD units (N, Nm)

            # Normalize to [-1, 1] for env direct control mode
            # thrust: (thrust - 10) / 10 → roughly [-1, 1] for hover
            # torques: divide by max range
            normalized = np.zeros(4)
            normalized[0] = np.clip((action[0] - 10.0) / 10.0, -1.0, 1.0)
            normalized[1] = np.clip(action[1] / 3.0, -1.0, 1.0)
            normalized[2] = np.clip(action[2] / 3.0, -1.0, 1.0)
            normalized[3] = np.clip(action[3] / 2.0, -1.0, 1.0)

            return normalized

    for stage in stages:
        print(f"\n--- Stage {stage} ---")

        # Evaluate PD baseline
        env_pd = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=False)
        env_pd.reset(seed=42)

        def pd_controller(obs):
            return np.zeros(4)  # Zero residual = pure PD

        pd_results = evaluate_controller(env_pd, pd_controller, n_episodes=args.episodes,
                                         description=f"PD (Stage {stage})")
        env_pd.close()

        print(f"PD Baseline: success={pd_results['success_rate']:.1%}, "
              f"survival={pd_results['survival_rate']:.1%}, reward={pd_results['mean_reward']:.1f}±{pd_results['std_reward']:.1f}")

        # Evaluate BC model
        env_bc = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env_bc.reset(seed=42)

        bc_results = evaluate_controller(env_bc, bc_controller, n_episodes=args.episodes,
                                         description=f"BC (Stage {stage})")
        env_bc.close()

        print(f"BC (NN):     success={bc_results['success_rate']:.1%}, "
              f"survival={bc_results['survival_rate']:.1%}, reward={bc_results['mean_reward']:.1f}±{bc_results['std_reward']:.1f}")

        # Improvement
        improvement = bc_results['success_rate'] - pd_results['success_rate']
        print(f"Improvement: {improvement:+.1%}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()