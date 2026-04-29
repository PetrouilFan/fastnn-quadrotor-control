#!/usr/bin/env python3
"""
Stage 7 Yaw Control Training: Figure-8 with Gaze Targets

Curriculum:
1. Amplitude: 1.5m -> 3.0m (start easy, grow trajectory)
2. Speed: 0.2x -> 2.0x (moderate, since yaw is the focus)
3. Yaw reward weight: 0.0 -> 3.0 (gradually increase yaw emphasis)

Convergence-Predictive Tracking (CPT) reward catches "doomed states"
early — when the drone's trajectory diverges from the target's future
trajectory, even if current position error is small.

Usage:
    python train_stage7_yaw.py --steps 20000000 --n-envs 1024
    python train_stage7_yaw.py --steps 20000000 --seeds 0,1,2
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


class YawCurriculumCallback(BaseCallback):
    """Callback that increases amplitude, speed and yaw reward weight during training."""

    def __init__(self, speed_schedule: dict, yaw_schedule: dict,
                 amplitude_schedule: dict = None, verbose=1):
        super().__init__(verbose)
        self.speed_schedule = sorted(speed_schedule.items())
        self.yaw_schedule = sorted(yaw_schedule.items())
        self.amplitude_schedule = sorted(amplitude_schedule.items()) if amplitude_schedule else []
        self.current_speed_idx = 0
        self.current_yaw_idx = 0
        self.current_amp_idx = 0

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

    def _update_amplitude(self, amplitude):
        env = self.model.env
        if hasattr(env, 'envs'):
            for e in env.envs:
                if hasattr(e, 'env'):
                    e.env.set_figure8_amplitude(amplitude)
                elif hasattr(e, 'set_figure8_amplitude'):
                    e.set_figure8_amplitude(amplitude)
        elif hasattr(env, 'set_figure8_amplitude'):
            env.set_figure8_amplitude(amplitude)
        if self.verbose > 0:
            print(f"\n=== Amplitude curriculum: {amplitude:.2f}m ===")

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

        # Amplitude curriculum
        if self.current_amp_idx < len(self.amplitude_schedule) - 1:
            next_amp_step, next_amp = self.amplitude_schedule[self.current_amp_idx + 1]
            if step >= next_amp_step:
                self.current_amp_idx += 1
                _, amp = self.amplitude_schedule[self.current_amp_idx]
                self._update_amplitude(amp)

        # Speed curriculum
        if self.current_speed_idx < len(self.speed_schedule) - 1:
            next_speed_step, next_speed = self.speed_schedule[self.current_speed_idx + 1]
            if step >= next_speed_step:
                self.current_speed_idx += 1
                _, speed = self.speed_schedule[self.current_speed_idx]
                self._update_speed(speed)

        # Yaw reward weight curriculum
        if self.current_yaw_idx < len(self.yaw_schedule) - 1:
            next_yaw_step, next_yaw_weight = self.yaw_schedule[self.current_yaw_idx + 1]
            if step >= next_yaw_step:
                self.current_yaw_idx += 1
                _, yaw_weight = self.yaw_schedule[self.current_yaw_idx]
                self._update_yaw_weight(yaw_weight)

        return True


def make_env(stage, seed, rank, speed=0.1, yaw_weight=0.0, amplitude=3.0):
    def _():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        # Set parameters BEFORE reset so they persist into the first episode
        env.set_target_speed(speed)
        env.set_target_trajectory('figure8_yaw')
        env.set_moving_target(True)
        env.set_yaw_reward_weight(yaw_weight)
        env.set_figure8_amplitude(amplitude)
        env.reset(seed=seed + rank)
        return env
    return _


def train_stage7_yaw(stage=7, total_steps=20000000, n_envs=128, seeds=[0],
                     start_speed=0.1, start_yaw_weight=0.5, start_amplitude=3.0,
                     final_amplitude=3.0):
    """Train SAC on Stage 7 Yaw Control."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"Stage 7 Yaw Control - Seed {seed}")
        print(f"Speed: {start_speed}x -> 2.0x, Yaw weight: {start_yaw_weight} -> 3.0")
        print(f"Amplitude: {start_amplitude}m (fixed)")
        print(f"Attitude cliff: 50deg (raised from 30deg for 3m banking)")
        print("=" * 60)

        model_dir = f"models_stage7_yaw/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create training environment
        env_fns = [make_env(stage=stage, seed=seed, rank=i, speed=start_speed,
                           yaw_weight=start_yaw_weight, amplitude=start_amplitude)
                   for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Speed curriculum: 0.1x -> 2.0x (start slower, more time at inflection points)
        speed_schedule = {
            0: start_speed,  # 0.1x
            total_steps // 5: 0.5,
            total_steps * 2 // 5: 0.8,
            total_steps * 3 // 5: 1.2,
            total_steps * 4 // 5: 1.6,
            total_steps: 2.0,
        }

        # Yaw reward weight curriculum: 0.5 -> 3.0 (start with moderate yaw from beginning)
        yaw_schedule = {
            0: start_yaw_weight,
            total_steps // 5: 1.0,
            total_steps * 2 // 5: 1.5,
            total_steps * 3 // 5: 2.0,
            total_steps * 4 // 5: 2.5,
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
            tensorboard_log=f"tb_logs_stage7_yaw/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=1000000,
            save_path=model_dir,
            name_prefix="stage7_yaw",
        )

        curriculum_callback = YawCurriculumCallback(
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
        for eval_speed in [0.5, 1.0, 1.5, 2.0]:
            eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)

            successes = 0
            tracking_errors = []
            yaw_errors = []
            convergence_scores = []
            predictive_errors = []

            for ep in range(50):
                obs, _ = eval_env.reset(seed=seed + ep)
                eval_env.set_target_speed(eval_speed)
                eval_env.set_target_trajectory('figure8_yaw')
                eval_env.set_moving_target(True)
                eval_env.set_yaw_reward_weight(3.0)
                eval_env.set_figure8_amplitude(3.0)

                done = False
                steps = 0
                ep_track_errors = []
                ep_yaw_errors = []
                ep_convergence = []
                ep_pred_errors = []

                while not done and steps < 500:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = eval_env.step(action)
                    done = terminated or truncated

                    pos = eval_env.data.qpos[:3]
                    vel = eval_env.data.qvel[:3]
                    target = eval_env.target_pos
                    tracking_error = np.linalg.norm(pos - target)

                    # CPT metrics
                    if tracking_error > 0.01:
                        err_dir = (target - pos) / tracking_error
                        convergence = np.dot(vel, err_dir)
                    else:
                        convergence = 0.0
                    pred_pos = pos + vel * 0.3
                    pred_error = np.linalg.norm(pred_pos - eval_env._future_target_300ms)

                    quat = eval_env.data.qpos[3:7]
                    rpy = eval_env._quat_to_rpy(quat)
                    yaw_error = abs(eval_env._wrap_angle(rpy[2] - eval_env._current_target_yaw))

                    ep_track_errors.append(tracking_error)
                    ep_yaw_errors.append(yaw_error)
                    ep_convergence.append(convergence)
                    ep_pred_errors.append(pred_error)
                    steps += 1

                if steps >= 500:
                    successes += 1
                tracking_errors.append(np.mean(ep_track_errors) if ep_track_errors else 999)
                yaw_errors.append(np.mean(ep_yaw_errors) if ep_yaw_errors else float('nan'))
                convergence_scores.append(np.mean(ep_convergence) if ep_convergence else 0)
                predictive_errors.append(np.mean(ep_pred_errors) if ep_pred_errors else 999)

            print(f"  Speed {eval_speed:.1f}x: {successes}/50 success, "
                  f"track={np.nanmean(tracking_errors):.3f}m, "
                  f"yaw={np.nanmean(yaw_errors):.3f}rad, "
                  f"conv={np.mean(convergence_scores):.2f}, "
                  f"pred={np.mean(predictive_errors):.2f}m")
            eval_env.close()

        results[seed] = {'model_dir': model_dir}
        train_env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 7 Yaw Control Training")
    parser.add_argument("--stage", type=int, default=7)
    parser.add_argument("--steps", type=int, default=20000000)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--start-speed", type=float, default=0.1)
    parser.add_argument("--start-yaw-weight", type=float, default=0.5)
    parser.add_argument("--amplitude", type=float, default=3.0)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_stage7_yaw(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        start_speed=args.start_speed,
        start_yaw_weight=args.start_yaw_weight,
        start_amplitude=args.amplitude,
    )

    os.makedirs("results_stage7_yaw", exist_ok=True)
    with open(f"results_stage7_yaw/stage_{args.stage}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: Stage 7 Yaw Control Results")
    print("=" * 60)
    for seed, result in results.items():
        print(f"Seed {seed}: {result['model_dir']}")


if __name__ == "__main__":
    main()