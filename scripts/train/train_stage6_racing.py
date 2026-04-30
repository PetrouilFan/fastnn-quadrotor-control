#!/usr/bin/env python3
"""
Stage 6 Racing FPV Training: Extreme Speed with Aggressive Maneuvers

Stage 6 builds on Stage 5's breakthrough but focuses on RACING performance:
1. Racing circuit trajectory (oval with hairpin turns)
2. Speed matching rewards (not just direction alignment)
3. G-load penalties (limit to ~4-6G for safe racing)
4. Extreme speed curriculum: 0.5x → 5.0x
5. No payload drops (pure racing focus)

Key differences from Stage 5:
- Target: racing circuit, not figure-8
- Reward: speed matching + G-load limits
- Attitude: up to 120° allowed (FPV-style)
- Safety: wider boundary (3m) for high-speed tracking lag

Usage:
    python train_stage6_racing.py --steps 5000000 --n-envs 32
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


class SpeedCurriculumCallback(BaseCallback):
    """Callback that increases target speed during training."""

    def __init__(self, speed_schedule: dict, verbose=1):
        super().__init__(verbose)
        self.speed_schedule = sorted(speed_schedule.items())
        self.current_speed_idx = 0

    def _update_speed(self, speed):
        env = self.model.env
        if hasattr(env, 'envs'):
            for e in env.envs:
                if hasattr(e, 'env'):
                    e.env.set_target_speed(speed)
                elif hasattr(e, 'set_target_speed'):
                    e.set_target_speed(speed)
        elif hasattr(env, 'set_target_speed'):
            env.set_target_speed(speed)
        if self.verbose > 0:
            print(f"\n=== Speed curriculum: {speed:.1f}x ===")

    def _on_step(self):
        step = self.model.num_timesteps
        if self.current_speed_idx < len(self.speed_schedule) - 1:
            next_speed_step, next_speed = self.speed_schedule[self.current_speed_idx + 1]
            if step >= next_speed_step:
                self.current_speed_idx += 1
                _, speed = self.speed_schedule[self.current_speed_idx]
                self._update_speed(speed)
        return True


def make_env(stage, seed, rank, speed=0.5):
    def _():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        env.set_target_speed(speed)
        env.set_moving_target(True)
        return env
    return _


def train_stage6_racing(stage=6, total_steps=5000000, n_envs=32, seeds=[0], start_speed=0.5):
    """Train SAC on Stage 6 Racing FPV."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"Stage 6 Racing FPV - Seed {seed}")
        print(f"Speed curriculum: {start_speed}x -> 5.0x")
        print("=" * 60)

        model_dir = f"models_stage6_racing/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create training environment
        env_fns = [make_env(stage=stage, seed=seed, rank=i, speed=start_speed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Create eval environment
        eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env.reset(seed=seed)
        eval_env.set_target_speed(3.0)  # Evaluate at 3x speed
        eval_env.set_moving_target(True)

        print(f"Observation dimension: {train_env.observation_space.shape[0]}")

        # Extreme speed curriculum: 0.5x → 5.0x
        speed_schedule = {
            0: start_speed,
            total_steps // 10: 1.0,
            total_steps // 5: 2.0,
            total_steps // 3: 3.0,
            total_steps // 2: 4.0,
            int(total_steps * 0.8): 4.5,
            total_steps: 5.0,
        }

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=200_000,  # Larger buffer for diverse racing experiences
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            policy_kwargs=dict(
                net_arch=[256, 256, 128],  # Deeper network for racing
            ),
            tensorboard_log=f"tb_logs_stage6_racing/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=500000,
            save_path=model_dir,
            name_prefix="stage6_racing",
        )

        curriculum_callback = SpeedCurriculumCallback(speed_schedule)

        print(f"Training for {total_steps} steps...")
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, curriculum_callback],
            progress_bar=True,
        )

        model.save(os.path.join(model_dir, "final"))
        print(f"Saved to {model_dir}/final")

        # Final evaluation at multiple speeds
        for eval_speed in [1.0, 3.0, 5.0]:
            eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
            eval_env.reset(seed=seed)
            eval_env.set_target_speed(eval_speed)
            eval_env.set_moving_target(True)

            successes = 0
            tracking_errors = []
            max_gs = []

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
                    tracking_error = np.linalg.norm(pos - target)
                    ep_errors.append(tracking_error)
                    steps += 1

                if steps >= 500:
                    successes += 1
                tracking_errors.append(np.mean(ep_errors))

            print(f"  Speed {eval_speed:.1f}x: {successes}/50 success, {np.mean(tracking_errors):.3f}m tracking error")
            eval_env.close()

        results[seed] = {'model_dir': model_dir}
        train_env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 6 Racing FPV Training")
    parser.add_argument("--stage", type=int, default=6)
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--start-speed", type=float, default=0.5)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_stage6_racing(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        start_speed=args.start_speed,
    )

    os.makedirs("results_stage6_racing", exist_ok=True)
    with open(f"results_stage6_racing/stage_{args.stage}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: Stage 6 Racing FPV Results")
    print("=" * 60)
    for seed, result in results.items():
        print(f"Seed {seed}: {result['model_dir']}")


if __name__ == "__main__":
    main()
