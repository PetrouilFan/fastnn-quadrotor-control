#!/usr/bin/env python3
"""
Stage 7 Yaw-Only Training: Drone hovers at origin, learns to point at moving focal point.

The focal point traces a 3m figure-8 at the drone's hover position. The drone must learn
to rotate its heading to face the focal point as it moves. This isolates yaw control
from position tracking.

Curriculum:
1. Speed: 0.1x -> 1.5x (focal moves faster)
2. Yaw weight: 0.5 -> 3.0

Usage:
    python train_stage7_yaw_only.py --steps 10000000 --n-envs 128
    python train_stage7_yaw_only.py --steps 10000000 --seeds 0,1,2
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import argparse
import json

from env_rma import RMAQuadrotorEnv


class YawOnlyCurriculumCallback(BaseCallback):
    """Callback that increases speed and yaw reward weight during training."""

    def __init__(self, speed_schedule: dict, yaw_schedule: dict, verbose=1):
        super().__init__(verbose)
        self.speed_schedule = sorted(speed_schedule.items())
        self.yaw_schedule = sorted(yaw_schedule.items())
        self.current_speed_idx = 0
        self.current_yaw_idx = 0

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

    def _update_yaw_weight(self, weight):
        env = self.model.env
        if hasattr(env, 'envs'):
            for e in env.envs:
                if hasattr(e, 'env'):
                    e.env.set_yaw_reward_weight(weight)
                elif hasattr(e, 'set_yaw_reward_weight'):
                    e.set_yaw_reward_weight(weight)
        elif hasattr(env, 'set_yaw_reward_weight'):
            env.set_yaw_reward_weight(weight)
        if self.verbose > 0:
            print(f"\n=== Yaw reward weight: {weight:.2f} ===")

    def _on_step(self):
        step = self.model.num_timesteps

        # Speed curriculum
        if self.current_speed_idx < len(self.speed_schedule) - 1:
            next_step, next_speed = self.speed_schedule[self.current_speed_idx + 1]
            if step >= next_step:
                self.current_speed_idx += 1
                _, speed = self.speed_schedule[self.current_speed_idx]
                self._update_speed(speed)

        # Yaw reward weight curriculum
        if self.current_yaw_idx < len(self.yaw_schedule) - 1:
            next_step, next_weight = self.yaw_schedule[self.current_yaw_idx + 1]
            if step >= next_step:
                self.current_yaw_idx += 1
                _, weight = self.yaw_schedule[self.current_yaw_idx]
                self._update_yaw_weight(weight)

        return True


def make_env(stage, seed, rank, speed=0.1, yaw_weight=2.0, amplitude=3.0):
    def _():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=False)
        # Set parameters BEFORE reset so they persist into the first episode
        env.set_target_speed(speed)
        env.set_target_trajectory('figure8_yaw')
        env.set_moving_target(True)
        env.set_yaw_reward_weight(yaw_weight + np.random.uniform(-0.5, 0.5))
        env.set_figure8_amplitude(amplitude)
        env.set_yaw_only_mode(True)
        env.reset(seed=seed + rank)
        return env
    return _


def train_stage7_yaw_only(stage=7, total_steps=10000000, n_envs=128, seeds=[0],
                          start_speed=0.1, start_yaw_weight=2.0, start_amplitude=3.0):
    """Train SAC on Stage 7 Yaw-Only Control."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"Stage 7 Yaw-Only - Seed {seed}")
        print(f"Speed: {start_speed}x -> 1.5x, Yaw weight: {start_yaw_weight} -> 3.0")
        print(f"Amplitude: {start_amplitude}m")
        print(f"Mode: YAW-ONLY (drone hovers at origin, points at focal)")
        print("=" * 60)

        model_dir = f"models_stage7_yaw_only/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        env_fns = [make_env(stage=stage, seed=seed, rank=i, speed=start_speed,
                           yaw_weight=start_yaw_weight, amplitude=start_amplitude)
                   for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Speed curriculum: 0.1x -> 1.5x (slower than before since focal moves at drone)
        speed_schedule = {
            0: start_speed,
            total_steps // 4: 0.3,
            total_steps // 2: 0.6,
            total_steps * 3 // 4: 1.0,
            total_steps: 1.5,
        }

        # Yaw reward weight curriculum: 2.0 -> 3.0
        yaw_schedule = {
            0: start_yaw_weight,
            total_steps // 4: 2.5,
            total_steps // 2: 2.8,
            total_steps * 3 // 4: 3.0,
            total_steps: 3.0,
        }

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=5000,
            batch_size=512,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            policy_kwargs=dict(
                net_arch=[256, 256, 256, 128],
            ),
            tensorboard_log=f"tb_logs_stage7_yaw_only/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=1000000,
            save_path=model_dir,
            name_prefix="stage7_yaw_only",
        )

        curriculum_callback = YawOnlyCurriculumCallback(
            speed_schedule, yaw_schedule
        )

        print(f"Training for {total_steps} steps...")
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, curriculum_callback],
            progress_bar=True,
        )

        model.save(os.path.join(model_dir, "final"))
        print(f"Saved to {model_dir}/final")

        # Evaluation at multiple speeds
        for eval_speed in [0.3, 0.5, 1.0, 1.5]:
            eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=False)
            eval_env.reset(seed=seed)
            eval_env.set_target_speed(eval_speed)
            eval_env.set_target_trajectory('figure8_yaw')
            eval_env.set_moving_target(True)
            eval_env.set_yaw_reward_weight(3.0)
            eval_env.set_figure8_amplitude(3.0)
            eval_env.set_yaw_only_mode(True)

            successes = 0
            yaw_errors = []

            for ep in range(50):
                obs, _ = eval_env.reset(seed=seed + ep)

                done = False
                steps = 0
                ep_yaw_errors = []

                while not done and steps < 500:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated

                    quat = eval_env.data.qpos[3:7]
                    rpy = eval_env._quat_to_rpy(quat)
                    yaw_err = abs(eval_env._wrap_angle(rpy[2] - eval_env._current_target_yaw))
                    ep_yaw_errors.append(yaw_err)
                    steps += 1

                if steps >= 500:
                    successes += 1
                yaw_errors.append(np.mean(ep_yaw_errors) if ep_yaw_errors else float('nan'))

            print(f"  Speed {eval_speed:.1f}x: {successes}/50 success, "
                  f"yaw={np.nanmean(yaw_errors):.3f}rad ({np.degrees(np.nanmean(yaw_errors)):.1f}deg)")
            eval_env.close()

        results[seed] = {'model_dir': model_dir}
        train_env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 7 Yaw-Only Training")
    parser.add_argument("--stage", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--start-speed", type=float, default=0.1)
    parser.add_argument("--start-yaw-weight", type=float, default=2.0)
    parser.add_argument("--amplitude", type=float, default=3.0)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_stage7_yaw_only(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        start_speed=args.start_speed,
        start_yaw_weight=args.start_yaw_weight,
        start_amplitude=args.amplitude,
    )

    os.makedirs("results_stage7_yaw_only", exist_ok=True)
    with open(f"results_stage7_yaw_only/stage_{args.stage}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: Stage 7 Yaw-Only Results")
    print("=" * 60)
    for seed, result in results.items():
        print(f"Seed {seed}: {result['model_dir']}")


if __name__ == "__main__":
    main()
