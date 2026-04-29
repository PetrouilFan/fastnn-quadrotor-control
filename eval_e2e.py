#!/usr/bin/env python3
"""
Quick Evaluation: End-to-End NN Control vs PD baseline

Tests the trained end-to-end model against the PD controller baseline.
Properly compares NN (direct control) vs PD (residual RL with zero residual).
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO

from env_rma import RMAQuadrotorEnv


def evaluate_pd(env, n_episodes=50):
    """Evaluate PD controller (zero residual in residual RL mode)."""
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
            # PD baseline: zero residual (PD handles control)
            action = np.zeros(4)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        if steps >= 500:
            successes += 1
            survivals += 1
        elif terminated:
            survivals += 0
        else:
            survivals += 1

        rewards.append(episode_reward)

    return {
        "success_rate": successes / n_episodes,
        "survival_rate": survivals / n_episodes,
        "mean_reward": np.mean(rewards),
    }


def evaluate_nn(env, model, n_episodes=50):
    """Evaluate NN controller (direct control mode)."""
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
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        if steps >= 500:
            successes += 1
            survivals += 1
        elif terminated:
            survivals += 0
        else:
            survivals += 1

        rewards.append(episode_reward)

    return {
        "success_rate": successes / n_episodes,
        "survival_rate": survivals / n_episodes,
        "mean_reward": np.mean(rewards),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate end-to-end NN control")
    parser.add_argument("--model", type=str, default="models_e2e/seed_0/final.zip",
                        help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=50, help="Evaluation episodes")
    parser.add_argument("--stages", type=str, default="1,2,3,4",
                        help="Comma-separated stages to evaluate")
    args = parser.parse_args()

    stages = [int(s) for s in args.stages.split(",")]

    print("=" * 60)
    print("End-to-End NN Control Evaluation")
    print("=" * 60)

    # Load model if exists
    model = None
    if os.path.exists(args.model):
        print(f"Loading model from {args.model}")
        model = PPO.load(args.model)
    else:
        print("No model found, running PD baseline only")

    for stage in stages:
        print(f"\n--- Stage {stage} ---")

        # Evaluate PD baseline (residual RL mode with zero residual)
        # In this mode, PD controller handles all control
        env_pd = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=False)
        env_pd.reset(seed=42)
        pd_results = evaluate_pd(env_pd, n_episodes=args.episodes)
        env_pd.close()

        print(f"PD Baseline: success={pd_results['success_rate']:.1%}, "
              f"survival={pd_results['survival_rate']:.1%}, reward={pd_results['mean_reward']:.1f}")

        # Evaluate NN if available (direct control mode)
        if model is not None:
            env_nn = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
            env_nn.reset(seed=42)
            nn_results = evaluate_nn(env_nn, model, n_episodes=args.episodes)
            env_nn.close()

            print(f"NN (E2E):    success={nn_results['success_rate']:.1%}, "
                  f"survival={nn_results['survival_rate']:.1%}, reward={nn_results['mean_reward']:.1f}")

            # Improvement
            improvement = nn_results['success_rate'] - pd_results['success_rate']
            print(f"Improvement: {improvement:+.1%}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")


if __name__ == "__main__":
    main()