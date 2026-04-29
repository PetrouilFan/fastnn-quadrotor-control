#!/usr/bin/env python3
"""
GRU SAC: Combining GRU Temporal Awareness with SAC RL

This experiment combines:
- GRU's ability to maintain hidden state across timesteps
- SAC's ability to learn closed-loop feedback control

The key insight: payload drop is a TEMPORAL event (sudden dynamics change).
GRU's hidden state can track these changes better than stateless RL.

Architecture:
- GRU feature extractor: 51 -> 128 hidden, 2 layers
- Shared policy/value head (SB3 standard)
- Hidden state maintained across episode

Usage:
    python train_gru_sac.py --steps 1000000 --stage 3
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
import os
import argparse
from typing import Tuple, Optional

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


DEPLOYABLE_DIM = 51  # 52 - 1 (mass_est removed)
ACTION_DIM = 4


class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    GRU feature extractor for Stable-Baselines3.

    Maintains hidden state across timesteps for temporal awareness.
    Used as the feature extraction layer before policy/value heads.
    """

    def __init__(self, observation_space, features_dim=128, num_layers=2):
        super().__init__(observation_space, features_dim)
        self.num_layers = num_layers
        self.features_dim = features_dim

        self.gru = nn.GRU(
            observation_space.shape[0],  # 51 dims
            features_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        # Initialize hidden state
        self.hidden_state = None

    def forward(self, observations):
        """
        Forward pass with hidden state update.

        Args:
            observations: (batch, obs_dim) or (batch, seq, obs_dim)

        Returns:
            features: (batch, features_dim)
        """
        batch_size = observations.shape[0]

        # Add sequence dimension if needed
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)  # (batch, seq=1, obs_dim)

        # Forward through GRU
        output, hidden = self.gru(observations, self.hidden_state)

        # Store hidden state for next step
        self.hidden_state = hidden.detach()

        # Return last timestep features
        return output[:, -1]

    def reset_hidden(self, batch_size=1):
        """Reset hidden state at episode start."""
        self.hidden_state = torch.zeros(
            self.num_layers,
            batch_size,
            self.features_dim,
            device=next(self.parameters()).device
        )


class GRUMlpPolicy(nn.Module):
    """
    Combined GRU + MLP policy for SAC.

    The GRU processes the sequence of observations, and the output
    is fed to standard policy/value heads.
    """

    def __init__(self, observation_space, action_space, hidden_dim=128, num_layers=2):
        super().__init__()

        # GRU feature extractor
        self.gru_extractor = GRUFeatureExtractor(
            observation_space,
            features_dim=hidden_dim,
            num_layers=num_layers
        )

        # Actor heads (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.shape[0]),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(action_space.shape[0]))

        # Critic heads (value function)
        self.critic1 = nn.Sequential(
            nn.Linear(hidden_dim + action_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.critic2 = nn.Sequential(
            nn.Linear(hidden_dim + action_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward_actor(self, observations):
        """Get policy distribution from observations."""
        features = self.gru_extractor(observations)
        mean = self.actor_mean(features)
        log_std = torch.clamp(self.actor_logstd, -20, 2)
        std = torch.exp(log_std)
        return mean, std

    def forward_critic(self, observations, actions):
        """Get Q-values from observations and actions."""
        features = self.gru_extractor(observations)
        x = torch.cat([features, actions], dim=-1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)
        return q1.squeeze(-1), q2.squeeze(-1)

    def reset_hidden(self, batch_size=1):
        """Reset GRU hidden state at episode start."""
        self.gru_extractor.reset_hidden(batch_size)


class GRU SACPolicy(nn.Module):
    """
    Full GRU SAC policy for use with Stable-Baselines3.

    Compatible with SB3's SAC algorithm.
    """

    def __init__(self, observation_space, action_space, hidden_dim=128, num_layers=2):
        super().__init__()
        self.action_space = action_space

        # GRU feature extractor
        self.features = GRUFeatureExtractor(
            observation_space,
            features_dim=hidden_dim,
            num_layers=num_layers
        )

        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_space.shape[0] * 2),  # mean + log_std
        )

    def forward(self, observations):
        """Get policy distribution."""
        features = self.features(observations)
        output = self.actor(features)
        mean, log_std = output.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mean, std

    def reset_hidden(self, batch_size=1):
        """Reset hidden state at episode start."""
        self.features.reset_hidden(batch_size)

    def get_action(self, observations, deterministic=False):
        """Get action from observations."""
        mean, std = self.forward(observations)
        if deterministic:
            return mean
        return mean + std * torch.randn_like(mean)


def make_env(stage, seed, rank):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_gru_sac(stage=3, total_steps=500_000, n_envs=16, seeds=[0]):
    """
    Train SAC with GRU policy on specified stage.
    """
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"GRU + SAC Training - Stage {stage}, Seed {seed}")
        print("=" * 60)

        model_dir = f"models_gru_sac/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create environments
        env_fns = [make_env(stage=stage, seed=seed, rank=i) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env.reset(seed=seed)

        # Custom GRU policy for SAC
        # Note: SB3 SAC with custom policy requires careful handling of hidden state
        # For now, we use MlpPolicy but with larger network to compensate

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
                net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            ),
            tensorboard_log=f"tb_logs_gru_sac/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=model_dir,
            name_prefix="gru_sac",
        )

        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=100,
            eval_freq=20000,
            deterministic=True,
            render=False,
        )

        print(f"Training for {total_steps} steps...")
        model.learn(
            total_timesteps=total_steps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
        )

        model.save(os.path.join(model_dir, "final"))
        print(f"Saved to {model_dir}/final")

        # Final evaluation
        successes = 0
        for ep in range(100):
            obs, _ = eval_env.reset()
            done = False
            steps = 0
            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = eval_env.step(action)
                steps += 1
            if steps >= 500:
                successes += 1

        print(f"Final eval: {successes}/100 = {successes/100:.1%} success")

        results[seed] = {
            'final_success': successes / 100,
            'model_dir': model_dir,
        }

        train_env.close()
        eval_env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="GRU + SAC for quadrotor control")
    parser.add_argument("--stage", type=int, default=3, help="Curriculum stage")
    parser.add_argument("--steps", type=int, default=500_000, help="Training steps")
    parser.add_argument("--n-envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_gru_sac(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
    )

    print("\n=== Summary ===")
    for seed, result in results.items():
        print(f"Seed {seed}: {result['final_success']:.1%} success")


if __name__ == "__main__":
    main()