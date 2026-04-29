#!/usr/bin/env python3
"""
Stage 8 Phase 2: Add altitude variation (±0.3m)
Transfer learning from Phase 1 model.
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import argparse
import json

from env_rma import RMAQuadrotorEnv


def make_env(stage, seed, rank, speed=1.0):
    def _():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        env.set_target_speed(speed)
        env.set_target_trajectory('extended')
        env.set_moving_target(True)
        return env
    return _


def train_phase2(stage=8, total_steps=10000000, n_envs=128, seeds=[0], fixed_speed=1.0):
    """Train Phase 2 with transfer learning from Phase 1."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"Stage 8 Phase 2: Altitude Variation ±0.3m")
        print(f"Transfer from Phase 1, {total_steps} steps")
        print("=" * 60)

        model_dir = f"models_stage8_extreme/stage_7_phase2/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create environment
        env_fns = [make_env(stage=stage, seed=seed, rank=i, speed=fixed_speed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Load Phase 1 model for transfer learning
        phase1_path = f"models_stage8_extreme/stage_7_extended/seed_{seed}/final.zip"
        print(f"Loading Phase 1 model from {phase1_path}")

        model = SAC.load(phase1_path, env=train_env, device="cuda" if torch.cuda.is_available() else "cpu")

        # Reset learning rate and optimizer for fine-tuning
        model.learning_rate = 3e-4

        checkpoint_callback = CheckpointCallback(
            save_freq=1000000,
            save_path=model_dir,
            name_prefix="stage8_phase2",
        )

        print(f"Training Phase 2 for {total_steps} steps...")
        model.learn(
            total_timesteps=total_steps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False,  # Continue from Phase 1
        )

        model.save(os.path.join(model_dir, "final"))
        print(f"Saved to {model_dir}/final")

        # Evaluate with XYZ tracking
        eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env.set_target_speed(fixed_speed)
        eval_env.set_target_trajectory('extended')
        eval_env.set_moving_target(True)

        successes = 0
        tracking_errors = []
        z_errors = []

        for ep in range(50):
            obs, _ = eval_env.reset()
            done = False
            steps = 0
            ep_errors = []
            ep_z_errors = []

            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = eval_env.step(action)
                pos = eval_env.data.qpos[:3]
                target = eval_env.target_pos
                xyz_error = np.linalg.norm(pos - target)
                z_error = abs(pos[2] - target[2])
                ep_errors.append(xyz_error)
                ep_z_errors.append(z_error)
                steps += 1

            if steps >= 500:
                successes += 1
            tracking_errors.append(np.mean(ep_errors))
            z_errors.append(np.mean(ep_z_errors))

        print(f"  Result: {successes}/50 success, {np.mean(tracking_errors):.3f}m XYZ error, {np.mean(z_errors):.3f}m Z error")
        results[seed] = {
            'success_rate': successes/50,
            'xyz_error': np.mean(tracking_errors),
            'z_error': np.mean(z_errors)
        }
        train_env.close()
        eval_env.close()

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    results = train_phase2(
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        fixed_speed=args.speed,
    )

    os.makedirs("results_stage8_extreme", exist_ok=True)
    with open(f"results_stage8_extreme/phase2_{args.speed}x.json", "w") as f:
        json.dump(results, f, indent=2)
