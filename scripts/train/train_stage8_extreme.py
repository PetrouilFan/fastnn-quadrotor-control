#!/usr/bin/env python3
"""
Stage 8 Extreme Extended Racing Training

Stage 8 is the ultimate challenge:
1. Extended track: 24m straights, 4.5m curves, ~57m lap (3x Stage 6)
2. Extreme speed: 0.5x -> 15.0x (vs Stage 6's 5x)
3. Physics-based acceleration/deceleration (not constant speed)
4. 3D vertical movement: 0.5m to 3.0m altitude
5. Speed variation phases: random acceleration periods

Usage:
    python train_stage8_extreme.py --steps 10000000 --n-envs 32
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
        env.set_target_trajectory('extended')
        env.set_moving_target(True)
        return env
    return _


def train_stage8_extreme(stage=8, total_steps=10000000, n_envs=32, seeds=[0], start_speed=0.5):
    """Train SAC on Stage 8 Extreme Extended Racing."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"Stage 8 Extreme Extended Racing - Seed {seed}")
        print(f"Speed curriculum: {start_speed}x -> 15.0x")
        print("=" * 60)

        model_dir = f"models_stage8_extreme/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create training environment
        env_fns = [make_env(stage=stage, seed=seed, rank=i, speed=start_speed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Create eval environment
        eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env.reset(seed=seed)
        eval_env.set_target_speed(8.0)  # Evaluate at 8x speed
        eval_env.set_target_trajectory('extended')
        eval_env.set_moving_target(True)

        print(f"Observation dimension: {train_env.observation_space.shape[0]}")

        # Extreme speed curriculum: 0.5x -> 15.0x
        # Slower progression due to higher difficulty
        speed_schedule = {
            0: start_speed,
            total_steps // 10: 2.0,
            total_steps // 5: 4.0,
            total_steps // 3: 7.0,
            total_steps // 2: 10.0,
            int(total_steps * 0.75): 12.0,
            total_steps: 15.0,
        }

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=300_000,  # Larger buffer for diverse experiences
            learning_starts=2000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            policy_kwargs=dict(
                net_arch=[256, 256, 256, 128],  # Deeper for extreme challenges
            ),
            tensorboard_log=f"tb_logs_stage8_extreme/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=500000,
            save_path=model_dir,
            name_prefix="stage8_extreme",
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
        for eval_speed in [2.0, 5.0, 10.0, 15.0]:
            eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
            eval_env.reset(seed=seed)
            eval_env.set_target_speed(eval_speed)
            eval_env.set_target_trajectory('extended')
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
    parser = argparse.ArgumentParser(description="Stage 8 Extreme Extended Racing Training")
    parser.add_argument("--stage", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--start-speed", type=float, default=0.5)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_stage8_extreme(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        start_speed=args.start_speed,
    )

    os.makedirs("results_stage8_extreme", exist_ok=True)
    with open(f"results_stage8_extreme/stage_{args.stage}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: Stage 8 Extreme Extended Racing Results")
    print("=" * 60)
    for seed, result in results.items():
        print(f"Seed {seed}: {result['model_dir']}")


if __name__ == "__main__":
    main()
