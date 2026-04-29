#!/usr/bin/env python3
"""
Stage 5 Curriculum Training: Moving Target with Predictive Control

This script trains SAC on Stage 5 (moving target) using curriculum learning:
1. Start with slow target speed (0.2x)
2. Gradually increase to full speed (1.0x)
3. Predictive reward shaping (velocity matching)

Key changes from standard SAC:
- Target velocity included in state (54 deployable dims)
- Velocity matching reward term
- Speed curriculum for progressive difficulty

Usage:
    python train_stage5_curriculum.py --steps 1000000
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from tqdm import tqdm
import os
import argparse
import json

from env_rma import RMAQuadrotorEnv


ACTION_DIM = 4
STAGE = 5


from stable_baselines3.common.callbacks import BaseCallback

class SpeedCurriculumCallback(BaseCallback):
    """Callback that increases target speed during training."""

    def __init__(self, speed_schedule: dict, verbose=1):
        """
        Args:
            speed_schedule: Dict mapping steps to speed multiplier
                            e.g., {0: 0.2, 50000: 0.5, 100000: 1.0}
        """
        super().__init__(verbose)
        self.speed_schedule = sorted(speed_schedule.items())
        self.current_speed_idx = 0

    def _update_speed(self, speed):
        """Update target speed on all training environments."""
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
        """Called at each step of training."""
        step = self.model.num_timesteps
        if self.current_speed_idx < len(self.speed_schedule) - 1:
            next_speed_step, next_speed = self.speed_schedule[self.current_speed_idx + 1]
            if step >= next_speed_step:
                self.current_speed_idx += 1
                _, speed = self.speed_schedule[self.current_speed_idx]
                self._update_speed(speed)
        return True


def make_env(stage, seed, rank, speed=1.0):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        env.set_target_speed(speed)
        env.set_moving_target(True)
        return env
    return _init


def train_stage5_curriculum(stage=5, total_steps=1000000, n_envs=512, seeds=[0], start_speed=0.2, final_speed=1.0):
    """Train SAC on Stage 5 with speed curriculum."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"Stage 5 Curriculum Training - Seed {seed}")
        print(f"Speed curriculum: {start_speed}x -> {final_speed}x")
        print("=" * 60)

        model_dir = f"models_stage5_curriculum/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create training environment with slow initial speed
        env_fns = [make_env(stage=stage, seed=seed, rank=i, speed=start_speed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Create eval environment at full speed
        eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env.reset(seed=seed)
        eval_env.set_target_speed(final_speed)
        eval_env.set_moving_target(True)

        print(f"Observation dimension: {train_env.observation_space.shape[0]}")

        # Speed curriculum schedule
        speed_schedule = {
            0: start_speed,
            total_steps // 4: 0.4,
            total_steps // 2: 0.7,
            int(total_steps * 0.75): 0.9,
            total_steps: final_speed,
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
            tensorboard_log=f"tb_logs_stage5_curriculum/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=model_dir,
            name_prefix="stage5_curriculum",
        )

        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=50,
            eval_freq=20000,
            deterministic=True,
            render=False,
        )

        curriculum_callback = SpeedCurriculumCallback(speed_schedule)

        print(f"Training for {total_steps} steps...")
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, eval_callback, curriculum_callback],
            progress_bar=True,
        )

        model.save(os.path.join(model_dir, "final"))
        print(f"Saved to {model_dir}/final")

        # Final evaluation at full speed
        successes = 0
        final_dists = []
        tracking_errors = []

        for ep in range(100):
            obs, _ = eval_env.reset()
            done = False
            steps = 0
            ep_tracking_error = []

            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = eval_env.step(action)

                # Track distance to moving target
                pos = eval_env.data.qpos[:3]
                target = eval_env.target_pos
                tracking_error = np.linalg.norm(pos - target)
                ep_tracking_error.append(tracking_error)
                steps += 1

            if steps >= 500:
                successes += 1

            final_dist = np.linalg.norm(eval_env.data.qpos[:3] - eval_env.target_pos)
            final_dists.append(final_dist)
            tracking_errors.append(np.mean(ep_tracking_error))

        success_rate = successes / 100
        mean_tracking_error = np.mean(tracking_errors)
        mean_final_dist = np.mean(final_dists)

        print(f"\nFinal Evaluation (full speed):")
        print(f"  Success rate: {success_rate:.0%}")
        print(f"  Mean tracking error: {mean_tracking_error:.3f}m")
        print(f"  Mean final distance: {mean_final_dist:.3f}m")

        results[seed] = {
            'success_rate': success_rate,
            'mean_tracking_error': mean_tracking_error,
            'mean_final_dist': mean_final_dist,
            'model_dir': model_dir,
        }

        train_env.close()
        eval_env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage 5 Curriculum Training")
    parser.add_argument("--stage", type=int, default=5, help="Curriculum stage")
    parser.add_argument("--steps", type=int, default=1000000, help="Training steps")
    parser.add_argument("--n-envs", type=int, default=512, help="Parallel environments")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    parser.add_argument("--start-speed", type=float, default=0.2, help="Initial target speed")
    parser.add_argument("--final-speed", type=float, default=1.0, help="Final target speed")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_stage5_curriculum(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        start_speed=args.start_speed,
        final_speed=args.final_speed,
    )

    # Save results
    os.makedirs("results_stage5_curriculum", exist_ok=True)
    with open(f"results_stage5_curriculum/stage_{args.stage}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY: Stage 5 Curriculum Training Results")
    print("=" * 60)
    print(f"{'Seed':<10} {'Success':<12} {'Mean Track Err':<18} {'Mean Final Dist':<18}")
    print("-" * 60)
    for seed, result in results.items():
        print(f"{seed:<10} {result['success_rate']:>9.0%}  {result['mean_tracking_error']:>14.3f}m  {result['mean_final_dist']:>14.3f}m")


if __name__ == "__main__":
    main()