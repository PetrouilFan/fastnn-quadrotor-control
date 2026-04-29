#!/usr/bin/env python3
"""
Stage 8 Extended Racing Training (Fixed)
Safe version that uses environment-compatible speeds (up to 5x)

Features:
1. Extended track: 29m lap (vs Stage 6's 22m)
2. Speed curriculum: 0.5x -> 5.0x (environment limit)
3. 3D vertical movement: 0.5m to 3.0m altitude
4. Extended figure-8 with larger amplitude

Usage:
    python train_stage8_final.py --steps 5000000 --n-envs 32 --seeds 0
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import argparse
import json

from env_rma import RMAQuadrotorEnv


class Stage8CurriculumCallback(BaseCallback):
    """Callback that progressively enables Stage 8 features."""

    def __init__(self, speed_schedule: dict, verbose=1):
        super().__init__(verbose)
        self.speed_schedule = sorted(speed_schedule.items())
        self.current_speed_idx = 0
        self.extended_enabled = False
        self.moving_enabled = False

    def _update_speed(self, speed):
        env = self.model.env
        if hasattr(env, "envs"):
            for e in env.envs:
                if hasattr(e, "env"):
                    e.env.set_target_speed(speed)
                elif hasattr(e, "set_target_speed"):
                    e.set_target_speed(speed)
        elif hasattr(env, "set_target_speed"):
            env.set_target_speed(speed)
        if self.verbose > 0:
            print(f"\n=== Speed curriculum: {speed:.1f}x ===")

    def _enable_extended_trajectory(self):
        """Enable extended trajectory after basic learning."""
        if not self.extended_enabled:
            env = self.model.env
            if hasattr(env, "envs"):
                for e in env.envs:
                    if hasattr(e, "env"):
                        e.env.set_target_trajectory("extended")
                    elif hasattr(e, "set_target_trajectory"):
                        e.set_target_trajectory("extended")
            elif hasattr(env, "set_target_trajectory"):
                env.set_target_trajectory("extended")
            self.extended_enabled = True
            if self.verbose > 0:
                print(f"\n=== Extended trajectory enabled ===")

    def _enable_moving_target(self):
        """Enable moving target after trajectory learning."""
        if not self.moving_enabled:
            env = self.model.env
            if hasattr(env, "envs"):
                for e in env.envs:
                    if hasattr(e, "env"):
                        e.env.set_moving_target(True)
                    elif hasattr(e, "set_moving_target"):
                        e.set_moving_target(True)
            elif hasattr(env, "set_moving_target"):
                env.set_moving_target(True)
            self.moving_enabled = True
            if self.verbose > 0:
                print(f"\n=== Moving target enabled ===")

    def _on_step(self):
        step = self.model.num_timesteps

        # Phase 1: Learn basic hovering (first 50k steps)
        if step >= 50000 and not self.extended_enabled:
            self._enable_extended_trajectory()

        # Phase 2: Enable moving target (after 150k steps)
        if step >= 150000 and not self.moving_enabled:
            self._enable_moving_target()

        # Phase 3: Speed curriculum
        while (
            self.current_speed_idx < len(self.speed_schedule)
            and step >= self.speed_schedule[self.current_speed_idx][0]
        ):
            _, speed = self.speed_schedule[self.current_speed_idx]
            self._update_speed(speed)
            self.current_speed_idx += 1


def make_env(stage, seed, rank):
    def _():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        # Start with basic trajectory, will be updated by callback
        env.reset(seed=seed + rank)
        env.set_target_speed(0.1)  # Start extremely slow
        return env

    return _


def train_stage8(stage=8, total_steps=5000000, n_envs=32, seeds=[0]):
    """Train Stage 8 with speed curriculum."""
    results = {}

    for seed in seeds:
        print("=" * 70)
        print("Stage 8 Extended Racing - Seed", seed)
        print("Extended track, speed curriculum 0.5x -> 5.0x")
        print(f"Training {total_steps} steps with {n_envs} parallel envs")
        print("=" * 70)

        # Create model directory
        model_dir = f"models_stage8_final/stage_{stage}/seed_{seed}"
        tb_log_dir = f"tb_logs_stage8_final/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tb_log_dir, exist_ok=True)

        # Create vectorized environment
        env_fns = [make_env(stage=stage, seed=seed, rank=i) for i in range(n_envs)]
        train_env = DummyVecEnv(env_fns)

        print(f"Observation dimension: {train_env.observation_space.shape[0]}")

        # SAC model
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=2000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            target_entropy=-2,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=tb_log_dir,
            verbose=1,
        )

        # Speed curriculum: start VERY slow for Stage 8 difficulty
        speed_schedule = {
            0: 0.1,  # Start extremely slow
            100000: 0.2,  # Very slow
            250000: 0.5,  # Slow
            500000: 0.8,  # Moderate
            1000000: 1.0,  # Standard
            1500000: 1.5,  # Fast
            2000000: 2.0,  # Very fast
            3000000: 3.0,  # Extreme
            4000000: 4.0,  # Maximum safe
        }

        curriculum_callback = Stage8CurriculumCallback(speed_schedule)

        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=500000,
            save_path=model_dir,
            name_prefix="stage8_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        # Train
        print("\nStarting training...")
        model.learn(
            total_timesteps=total_steps,
            callback=[curriculum_callback, checkpoint_callback],
            tb_log_name=f"SAC_stage{stage}",
            reset_num_timesteps=False,
        )

        # Save final model
        final_path = os.path.join(model_dir, "final.zip")
        model.save(final_path)
        print(f"\nFinal model saved to: {final_path}")

        # Save results
        results[f"seed_{seed}"] = {
            "model_dir": model_dir,
            "final_model": final_path,
            "total_steps": total_steps,
            "speed_curriculum": speed_schedule,
        }

        # Cleanup
        train_env.close()

    # Save training summary
    results_dir = "results_stage8_final"
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/stage_{stage}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: Stage 8 Extended Racing Results")
    print("=" * 60)
    for seed, result in results.items():
        print(f"Seed {seed}: {result['model_dir']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8 Extended Racing Training")
    parser.add_argument("--stage", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="0")

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    train_stage8(
        stage=args.stage, total_steps=args.steps, n_envs=args.n_envs, seeds=seeds
    )
