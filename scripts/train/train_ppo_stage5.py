#!/usr/bin/env python3
"""
Stage 5 with PPO: On-policy learning maintains temporal order.

PPO collects full rollouts before updating, preserving the temporal
sequence "target moved right → I need to move right" that SAC breaks
by sampling random timesteps from the replay buffer.

Key improvements over SAC:
1. Temporal order preserved in rollouts
2. Slower curriculum (0.05x start)
3. Longer training (2M steps)
4. Larger safety boundary for moving targets

Usage:
    python train_ppo_stage5.py --steps 2000000 --n-envs 512
"""

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import os
import argparse
import json

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class SpeedCurriculumCallback(BaseCallback):
    """Gradually increase target speed during training."""

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
            print(f"\n=== Speed curriculum: {speed:.2f}x ===")

    def _on_step(self):
        step = self.model.num_timesteps
        if self.current_speed_idx < len(self.speed_schedule) - 1:
            next_speed_step, next_speed = self.speed_schedule[self.current_speed_idx + 1]
            if step >= next_speed_step:
                self.current_speed_idx += 1
                _, speed = self.speed_schedule[self.current_speed_idx]
                self._update_speed(speed)
        return True


def make_env(stage, seed, rank, speed=0.05):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        env.set_target_speed(speed)
        env.set_moving_target(True)
        return env
    return _init


def train_ppo_stage5(stage=5, total_steps=2000000, n_envs=512, seeds=[0], start_speed=0.05):
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"PPO Stage 5 - Seed {seed}, Speed: {start_speed}x -> 1.0x")
        print("=" * 60)

        model_dir = f"models_ppo_stage5/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        env_fns = [make_env(stage=stage, seed=seed, rank=i, speed=start_speed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env.reset(seed=seed)
        eval_env.set_target_speed(1.0)
        eval_env.set_moving_target(True)

        obs_dim = train_env.observation_space.shape[0]
        print(f"Observation dim: {obs_dim}, n_envs: {n_envs}")

        # Slow curriculum: 0.05 -> 0.1 -> 0.2 -> 0.4 -> 0.7 -> 1.0
        speed_schedule = {
            0: start_speed,
            total_steps // 8: 0.1,
            total_steps // 4: 0.2,
            total_steps // 2: 0.4,
            int(total_steps * 0.7): 0.7,
            int(total_steps * 0.85): 0.9,
            total_steps: 1.0,
        }

        # With 512 envs, n_steps=128 gives 65K steps per rollout
        # This means ~30 rollouts for 2M steps = good training
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=128,         # steps per env per rollout (128 * 512 = 65K total)
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            tensorboard_log=f"tb_logs_ppo_stage5/stage_{stage}/seed_{seed}/",
            device="cpu",  # PPO is faster on CPU with MLP
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=model_dir,
            name_prefix="ppo_stage5",
        )

        curriculum_callback = SpeedCurriculumCallback(speed_schedule)

        print(f"Training PPO for {total_steps} steps...")
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, curriculum_callback],
            progress_bar=True,
        )

        model.save(os.path.join(model_dir, "final"))
        print(f"Saved to {model_dir}/final")

        # Evaluate at multiple speeds
        for eval_speed in [0.2, 0.5, 1.0]:
            eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
            eval_env.reset(seed=seed)
            eval_env.set_target_speed(eval_speed)
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
                    ep_errors.append(np.linalg.norm(eval_env.data.qpos[:3] - eval_env.target_pos))
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
    parser = argparse.ArgumentParser(description="PPO Stage 5 Training")
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--start-speed", type=float, default=0.05)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    results = train_ppo_stage5(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        start_speed=args.start_speed,
    )


if __name__ == "__main__":
    main()