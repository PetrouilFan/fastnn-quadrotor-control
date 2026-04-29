#!/usr/bin/env python3
"""
Stage 5 with GRU SAC: Temporal awareness for moving target tracking

GRU's hidden state can maintain temporal context across timesteps,
which is critical for learning predictive tracking (target moved right → move right).

This script uses a custom GRU policy with Stable-Baselines3.
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
import os
import argparse
import json

from env_rma import RMAQuadrotorEnv


class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    GRU feature extractor that maintains hidden state across timesteps.
    Critical for moving target tracking where temporal context matters.
    """

    def __init__(self, observation_space, gru_hidden_dim=128, num_layers=2):
        super().__init__(observation_space, gru_hidden_dim)
        self.num_layers = num_layers
        self.hidden_state = None

        self.gru = nn.GRU(
            observation_space.shape[0],
            gru_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

    def forward(self, observations):
        """
        Forward pass with hidden state update.
        """
        batch_size = observations.shape[0]

        # Add sequence dimension if needed
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)

        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state.shape[1] != batch_size:
            self.hidden_state = torch.zeros(
                self.num_layers,
                batch_size,
                self.features_dim,
                device=observations.device,
            )

        # Forward through GRU
        output, hidden = self.gru(observations, self.hidden_state)

        # Store hidden state
        self.hidden_state = hidden.detach()

        return output[:, -1]

    def reset_hidden(self, batch_size=1):
        """Reset hidden state."""
        self.hidden_state = torch.zeros(
            self.num_layers,
            batch_size,
            self.features_dim,
            device=next(self.parameters()).device,
        )


class SpeedCurriculumCallback(BaseCallback):
    """Callback that increases target speed during training."""

    def __init__(self, speed_schedule: dict, verbose=1):
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


def train_gru_sac_stage5(stage=5, total_steps=1000000, n_envs=512, seeds=[0], start_speed=0.2):
    """Train SAC with GRU on Stage 5."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"GRU SAC Stage 5 Training - Seed {seed}")
        print(f"Speed curriculum: {start_speed}x -> 1.0x")
        print("=" * 60)

        model_dir = f"models_gru_sac_stage5/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create training environment
        env_fns = [make_env(stage=stage, seed=seed, rank=i, speed=start_speed) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        # Create eval environment
        eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env.reset(seed=seed)
        eval_env.set_target_speed(1.0)
        eval_env.set_moving_target(True)

        obs_dim = train_env.observation_space.shape[0]
        print(f"Observation dimension: {obs_dim}")

        # Speed curriculum schedule
        speed_schedule = {
            0: start_speed,
            total_steps // 4: 0.4,
            total_steps // 2: 0.7,
            int(total_steps * 0.75): 0.9,
            total_steps: 1.0,
        }

        # Custom GRU policy
        policy_kwargs = dict(
            net_arch=[256, 256],
            features_extractor_class=GRUFeatureExtractor,
            features_extractor_kwargs=dict(gru_hidden_dim=128, num_layers=2),
        )

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
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"tb_logs_gru_sac_stage5/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        from stable_baselines3.common.callbacks import CheckpointCallback
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=model_dir,
            name_prefix="gru_sac_stage5",
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

        # Final evaluation
        successes = 0
        tracking_errors = []

        for ep in range(100):
            obs, _ = eval_env.reset()
            eval_env.reset_hidden() if hasattr(eval_env, 'reset_hidden') else None
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
    parser = argparse.ArgumentParser(description="GRU SAC Stage 5 Training")
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--start-speed", type=float, default=0.2)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_gru_sac_stage5(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        start_speed=args.start_speed,
    )

    os.makedirs("results_gru_sac_stage5", exist_ok=True)
    with open(f"results_gru_sac_stage5/stage_{args.stage}.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for seed, result in results.items():
        print(f"Seed {seed}: {result['success_rate']:.0%} success, {result['mean_tracking_error']:.3f}m tracking error")


if __name__ == "__main__":
    main()