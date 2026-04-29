#!/usr/bin/env python3
"""
Stage 5 Training WITHOUT mass_est (Sim-to-Real Deployment)

Uses NoMassEstEnvWrapper to remove mass_est from observations.
The actor learns to infer mass from action history and error integrals.
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
from fastnn_quadrotor.env_wrapper_stage5 import NoMassEstEnvWrapper


def make_env(stage, seed, rank, speed=0.05):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        env.set_target_speed(speed)
        env.set_moving_target(True)
        # Wrap to remove mass_est
        return NoMassEstEnvWrapper(env)
    return _init


def train_stage5_no_massest(stage=5, total_steps=5000000, n_envs=32, seeds=[0], start_speed=0.05):
    """Train SAC on Stage 5 WITHOUT mass_est for sim-to-real deployment."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"Stage 5 NO mass_est Training - Seed {seed}")
        print(f"Speed curriculum: {start_speed}x -> 1.0x")
        print("=" * 60)

        model_dir = f"models_stage5_no_massest/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create training environment
        env_fns = [make_env(stage=stage, seed=seed, rank=i, speed=start_speed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Create eval environment
        eval_env = NoMassEstEnvWrapper(RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True))
        eval_env.reset(seed=seed)
        eval_env.set_target_speed(1.0)
        eval_env.set_moving_target(True)

        obs_dim = train_env.observation_space.shape[0]
        print(f"Observation dimension: {obs_dim} (62 dims without mass_est)")

        # Speed curriculum schedule
        speed_schedule = {
            0: start_speed,
            total_steps // 8: 0.1,
            total_steps // 4: 0.2,
            total_steps // 2: 0.4,
            int(total_steps * 0.7): 0.7,
            int(total_steps * 0.85): 0.9,
            total_steps: 1.0,
        }

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            policy_kwargs=dict(
                net_arch=[256, 256],
            ),
            tensorboard_log=f"tb_logs_stage5_no_massest/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=500000,
            save_path=model_dir,
            name_prefix="stage5_no_massest",
        )

        # Speed curriculum callback
        from stable_baselines3.common.callbacks import BaseCallback
        class SpeedCurriculumCallback(BaseCallback):
            def __init__(self, schedule, verbose=1):
                super().__init__(verbose)
                self.schedule = sorted(schedule.items())
                self.idx = 0

            def _update_speed(self, speed):
                env = self.model.env
                if hasattr(env, 'envs'):
                    for e in env.envs:
                        if hasattr(e, 'env') and hasattr(e.env, 'set_target_speed'):
                            e.env.set_target_speed(speed)
                        elif hasattr(e, 'set_target_speed'):
                            e.set_target_speed(speed)
                if self.verbose > 0:
                    print(f"\n=== Speed curriculum: {speed:.2f}x ===")

            def _on_step(self):
                step = self.model.num_timesteps
                if self.idx < len(self.schedule) - 1:
                    next_step, next_speed = self.schedule[self.idx + 1]
                    if step >= next_step:
                        self.idx += 1
                        _, speed = self.schedule[self.idx]
                        self._update_speed(speed)
                return True

        curriculum_callback = SpeedCurriculumCallback(speed_schedule)

        print(f"Training for {total_steps} steps...")
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, curriculum_callback],
            progress_bar=True,
        )

        model.save(os.path.join(model_dir, "final"))
        print(f"Saved to {model_dir}/final")

        # Final evaluation
        successes = 0
        tracking_errors = []

        for ep in range(100):
            obs, _ = eval_env.reset()
            done = False
            steps = 0
            ep_tracking_error = []

            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = eval_env.step(action)

                pos = eval_env.data.qpos[:3]
                target = eval_env.target_pos
                tracking_error = np.linalg.norm(pos - target)
                ep_tracking_error.append(tracking_error)
                steps += 1

            if steps >= 500:
                successes += 1

            tracking_errors.append(np.mean(ep_tracking_error))

        success_rate = successes / 100
        mean_tracking_error = np.mean(tracking_errors)

        print(f"\nFinal Evaluation:")
        print(f"  Success rate: {success_rate:.0%}")
        print(f"  Mean tracking error: {mean_tracking_error:.3f}m")

        results[seed] = {
            'success_rate': success_rate,
            'mean_tracking_error': mean_tracking_error,
            'model_dir': model_dir,
        }

        train_env.close()
        eval_env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 5 Training WITHOUT mass_est")
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--start-speed", type=float, default=0.05)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_stage5_no_massest(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        start_speed=args.start_speed,
    )

    os.makedirs("results_stage5_no_massest", exist_ok=True)
    with open(f"results_stage5_no_massest/stage_5.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY: Stage 5 NO mass_est Training Results")
    print("=" * 60)
    for seed, result in results.items():
        print(f"Seed {seed}: {result['success_rate']:.0%} success, {result['mean_tracking_error']:.3f}m tracking error")


if __name__ == "__main__":
    main()
