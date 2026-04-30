#!/usr/bin/env python3
"""
Stage 5 Fine-tuning: Take a trained model and refine it.

Strategy: Load the 5M checkpoint (100% success, 4.76m tracking error)
and fine-tune with:
- Lower learning rate (1e-4) for stable refinement
- Stronger attitude penalty to prevent crashes
- Full target speed from the start

This avoids the precision-stability tradeoff by starting from a stable
policy and gradually making it more precise.
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import argparse
import json

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


def make_env(stage, seed, rank):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        return env
    return _init


def evaluate(model, n_episodes=50):
    """Evaluate model on Stage 5 at full speed."""
    env = RMAQuadrotorEnv(curriculum_stage=5, use_direct_control=True)
    env.reset()
    env.set_target_speed(1.0)
    env.set_moving_target(True)

    successes = 0
    tracking_errors = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_errors = []

        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            ep_errors.append(np.linalg.norm(env.data.qpos[:3] - env.target_pos))
            steps += 1

        if steps >= 500:
            successes += 1
        tracking_errors.append(np.mean(ep_errors))

    env.close()
    return successes / n_episodes, np.mean(tracking_errors)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Stage 5 model")
    parser.add_argument("--base-model", type=str, default="models_stage5_curriculum/stage_5/seed_0/final.zip")
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = SAC.load(args.base_model)

    # Evaluate base model first
    base_success, base_error = evaluate(model, n_episodes=20)
    print(f"Base model: {base_success:.0%} success, {base_error:.3f}m tracking error")

    # Create fresh training envs at full speed
    env_fns = [make_env(stage=5, seed=args.seed, rank=i) for i in range(args.n_envs)]
    train_env = SubprocVecEnv(env_fns)

    # Set lower learning rate
    model.learning_rate = args.lr
    for param_group in model.actor.optimizer.param_groups:
        param_group['lr'] = args.lr
    for param_group in model.critic.optimizer.param_groups:
        param_group['lr'] = args.lr

    model_dir = "models_stage5_finetune"
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path=model_dir,
        name_prefix="stage5_finetune",
    )

    print(f"Fine-tuning for {args.steps} steps with lr={args.lr}...")
    model.set_env(train_env)
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save(os.path.join(model_dir, "final"))

    # Evaluate fine-tuned model
    ft_success, ft_error = evaluate(model, n_episodes=50)
    print(f"\nFine-tuned model: {ft_success:.0%} success, {ft_error:.3f}m tracking error")

    # Save results
    results = {
        'base_success': base_success,
        'base_error': base_error,
        'finetune_success': ft_success,
        'finetune_error': ft_error,
        'steps': args.steps,
        'lr': args.lr,
    }
    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    train_env.close()


if __name__ == "__main__":
    main()