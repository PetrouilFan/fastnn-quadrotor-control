#!/usr/bin/env python3
"""
Stage 8 Phase 1: Fixed-Speed Training (No Curriculum)
Train at fixed 1x speed (0.3 m/s) to master XY tracking first.
Only after 100% success do we increase speed.
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
        env.reset(seed=seed + rank)
        env.set_target_speed(speed)  # FIXED speed - no curriculum
        env.set_target_trajectory('extended')
        env.set_moving_target(True)
        return env
    return _()


def train_fixed_speed(stage=8, total_steps=5000000, n_envs=128, seeds=[0], fixed_speed=1.0):
    """Train at fixed speed until convergence."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"Stage 8 Phase 1: Fixed Speed {fixed_speed}x (0.3 m/s)")
        print(f"Training {total_steps} steps with {n_envs} parallel envs")
        print("=" * 60)

        model_dir = f"models_stage8_extreme/stage_7_fixed/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create training environment
        env_fns = [lambda rank=i: make_env(stage=stage, seed=seed, rank=rank, speed=fixed_speed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        print(f"Observation dimension: {train_env.observation_space.shape[0]}")

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=300_000,
            learning_starts=2000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            policy_kwargs=dict(
                net_arch=[256, 256, 256, 128],
            ),
            tensorboard_log=f"tb_logs_stage8_fixed/speed_{fixed_speed}x/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=500000,
            save_path=model_dir,
            name_prefix=f"stage8_fixed_{fixed_speed}x",
        )

        print(f"Training for {total_steps} steps at fixed {fixed_speed}x speed...")
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
        eval_env.set_target_trajectory('extended')
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

        print(f"  Speed {fixed_speed}x: {successes}/50 success, {np.mean(tracking_errors):.3f}m tracking error")
        
        results[seed] = {
            'model_dir': model_dir,
            'success_rate': successes / 50,
            'tracking_error': np.mean(tracking_errors)
        }
        
        train_env.close()
        eval_env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 8 Fixed Speed Training")
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--speed", type=float, default=1.0, help="Fixed speed multiplier")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_fixed_speed(
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        fixed_speed=args.speed,
    )

    os.makedirs("results_stage8_extreme", exist_ok=True)
    with open(f"results_stage8_extreme/fixed_speed_{args.speed}x.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: Fixed Speed Training Results")
    print("=" * 60)
    for seed, result in results.items():
        print(f"Seed {seed}: {result['success_rate']*100:.0f}% success, {result['tracking_error']:.3f}m error")


if __name__ == "__main__":
    main()
