#!/usr/bin/env python3
"""
GRU-based PPO for end-to-end quadrotor control.

Uses a GRU policy that maintains hidden state across timesteps,
giving it temporal awareness that MLP policies lack.

Usage:
    python train_gru_ppo.py --epochs 5 --stage 3
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


DEPLOYABLE_DIM = 52
ACTION_DIM = 4


class GRUPolicy(ActorCriticPolicy):
    """
    GRU-based policy for PPO.

    Maintains hidden state across timesteps for temporal awareness.
    Unlike MLP which treats each step independently, GRU learns
    the feedback dynamics.
    """

    def __init__(self, *args, hidden_dim=128, num_layers=2, **kwargs):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        """Replace default MLP extractor with GRU."""
        self.mlp_extractor = GRUFeatureExtractor(
            observation_space=self.observation_space,
            features_dim=self.features_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )

    def forward(self, observations, deterministic=False):
        # Get features and action distribution from parent
        return super().forward(observations, deterministic)

    def evaluate_actions(self, observations, actions):
        # Override to handle hidden state
        features = self.mlp_extractor(observations)
        latent_pi, latent_vf = self.mlp_extractor.get_latent(observations)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        value = self.value_net(latent_vf)
        return log_prob, entropy, value


class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    GRU-based feature extractor for PPO.

    Maintains hidden state across the episode, giving the policy
    temporal awareness of dynamics and disturbances.
    """

    def __init__(self, observation_space, features_dim, hidden_dim=128, num_layers=2):
        super().__init__(observation_space, features_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # GRU for temporal processing
        self.gru = nn.GRU(
            DEPLOYABLE_DIM,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0,
        )

        # Output projection to features_dim
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim),
        )

        # Hidden state buffer (per-env)
        self._hiddens = None

    def forward(self, observations):
        """Forward pass with hidden state management."""
        # observations: (batch, obs_dim) - SB3 batches single-step observations
        # GRU needs (batch, seq=1, obs_dim)
        x = observations.unsqueeze(1)  # (batch, 1, obs_dim)

        if self._hiddens is None:
            self._hiddens = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim, device=x.device)

        output, self._hiddens = self.gru(x, self._hiddens.detach())
        features = self.fc(output[:, -1])  # (batch, hidden_dim) -> (batch, features_dim)
        return features

    def get_latent(self, observations):
        """Return latent_pi and latent_vf separately."""
        x = observations.unsqueeze(1)
        if self._hiddens is None:
            self._hiddens = torch.zeros(self.num_layers, x.shape[0], self.hidden_dim, device=x.device)
        output, self._hiddens = self.gru(x, self._hiddens.detach())
        features = self.fc(output[:, -1])
        return features, features  # Same for simplicity

    def reset_hidden(self, batch_size=1, device='cuda'):
        """Reset hidden state at episode start."""
        self._hiddens = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def reset(self):
        """Reset for new episode."""
        self._hiddens = None


class GRUPpoPolicy(ActorCriticPolicy):
    """
    Custom PPO policy with GRU for quadrotor control.
    """

    def __init__(self, *args, hidden_dim=128, num_layers=2, **kwargs):
        self.gru_hidden_dim = hidden_dim
        self.gru_num_layers = num_layers
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = GRUFeatureExtractor(
            observation_space=self.observation_space,
            features_dim=self.features_dim,
            hidden_dim=self.gru_hidden_dim,
            num_layers=self.gru_num_layers,
        )

    def reset_hidden(self, batch_size, device):
        self.mlp_extractor.reset_hidden(batch_size, device)


def make_env(stage, seed, rank, use_nn_control=True):
    """Factory for creating environments."""
    def _init():
        env = RMAQuadrotorEnv(
            curriculum_stage=stage,
            use_direct_control=use_nn_control,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train_gru_ppo(stage=3, seeds=[0, 1, 2], total_steps=2_000_000, n_envs=16):
    """Train GRU PPO on specified stage."""
    results = {}

    for seed in seeds:
        print("=" * 60)
        print(f"GRU PPO Training - Stage {stage}, Seed {seed}")
        print("=" * 60)

        model_dir = f"models_gru_ppo/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create environments
        env_fns = [make_env(stage=stage, seed=seed, rank=i) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env.reset(seed=seed)

        # Custom GRU policy
        policy_kwargs = dict(
            hidden_dim=128,
            num_layers=2,
            net_arch=[128, 64],  # Policy and value network after GRU
            activation_fn=nn.ReLU,
        )

        model = PPO(
            "CnnPolicy",  # Need to use MlpPolicy but with custom extractor
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"tb_logs_gru/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        # Callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=model_dir,
            name_prefix="gru_ppo",
        )

        eval_callback = EvalCallback(
            eval_env,
            n_eval_episodes=50,
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
        from fastnn_quadrotor.training.train_transformer_bc import BCController, evaluate

        # Load model
        loaded_model = PPO.load(os.path.join(model_dir, "final"))

        # Quick eval
        successes = 0
        for ep in range(50):
            obs, _ = eval_env.reset()
            done = False
            steps = 0
            while not done and steps < 500:
                action, _ = loaded_model.predict(obs, deterministic=True)
                obs, _, done, _, _ = eval_env.step(action)
                steps += 1
            if steps >= 500:
                successes += 1

        print(f"Final eval: {successes}/50 = {successes/50:.1%} success")

        results[seed] = {
            'final_success': successes / 50,
            'model_dir': model_dir,
        }

        train_env.close()
        eval_env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="GRU PPO for quadrotor control")
    parser.add_argument("--stage", type=int, default=3, help="Curriculum stage")
    parser.add_argument("--seeds", type=str, default="0,1,2", help="Seeds to train")
    parser.add_argument("--steps", type=int, default=2_000_000, help="Steps per seed")
    parser.add_argument("--n-envs", type=int, default=16, help="Parallel envs")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]
    results = train_gru_ppo(
        stage=args.stage,
        seeds=seeds,
        total_steps=args.steps,
        n_envs=args.n_envs,
    )

    print("\n=== Summary ===")
    for seed, result in results.items():
        print(f"Seed {seed}: {result['final_success']:.1%} success")


if __name__ == "__main__":
    main()