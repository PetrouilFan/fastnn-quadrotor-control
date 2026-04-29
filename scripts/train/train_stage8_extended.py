#!/usr/bin/env python3
"""
Stage 8 Extended: Longer training with better reward shaping
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import argparse
import json

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


def make_env(stage, seed, rank, speed=1.0):
    def _():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_trajectory("extended")
        env.set_moving_target(True)
        env.reset(seed=seed + rank)
        env.set_target_speed(speed)
        return env

    return _


def train_extended(
    stage=8, total_steps=10000000, n_envs=128, seeds=[0], fixed_speed=1.0
):
    """Train with extended time and larger batch size."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"Stage 8 Extended: {fixed_speed}x speed, {total_steps} steps")
        print("=" * 60)

        model_dir = f"models_stage8_extreme/stage_8_extended/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        env_fns = [
            make_env(stage=stage, seed=seed, rank=i, speed=fixed_speed)
            for i in range(n_envs)
        ]
        train_env = SubprocVecEnv(env_fns)

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=500_000,
            learning_starts=5000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256, 256, 128]),
            tensorboard_log=f"tb_logs_stage8_extended/speed_{fixed_speed}x/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=1000000,
            save_path=model_dir,
            name_prefix="stage8_extended",
        )

        model.learn(
            total_timesteps=total_steps,
            callback=checkpoint_callback,
            progress_bar=True,
        )

        model.save(os.path.join(model_dir, "final"))
        print(f"Saved to {model_dir}/final")

        # Evaluate
        eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env.reset(seed=seed)
        eval_env.set_target_speed(fixed_speed)
        eval_env.set_target_trajectory("extended")
        eval_env.set_moving_target(True)

        successes = 0
        tracking_errors = []

        for ep in range(50):
            obs, _ = eval_env.reset()
            done = False
            steps = 0
            ep_errors = []

            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = eval_env.step(action)
                pos = eval_env.data.qpos[:3]
                target = eval_env.target_pos
                xy_error = np.linalg.norm(pos[:2] - target[:2])
                ep_errors.append(xy_error)
                steps += 1

            if steps >= 500:
                successes += 1
            tracking_errors.append(np.mean(ep_errors))

        print(
            f"  Result: {successes}/50 success, {np.mean(tracking_errors):.3f}m error"
        )
        results[seed] = {
            "success_rate": successes / 50,
            "error": np.mean(tracking_errors),
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

    seeds = [int(s) for s in args.seeds.split(",")]
    results = train_extended(
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        fixed_speed=args.speed,
    )

    os.makedirs("results_stage8_extreme", exist_ok=True)
    with open(f"results_stage8_extreme/extended_{args.speed}x.json", "w") as f:
        json.dump(results, f, indent=2)
