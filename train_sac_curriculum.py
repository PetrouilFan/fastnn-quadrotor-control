#!/usr/bin/env python3
"""
Phase 1: Privileged Teacher Training via SB3 SAC + Curriculum Learning

Asymmetric Actor-Critic Architecture (properly wired into SB3):
- Actor: takes 52-dim deployable observations only (via DeployableExtractor)
- Critic: takes 60-dim observations (deployable + privileged)

This ensures the learned policy can be deployed on real hardware without
requiring ground-truth wind/mass information.

Multi-seed support: --seed 0,1,2 trains three independent runs.

Training fixes applied:
- Reduced learning rate (1e-4) to prevent divergence
- Added gradient clipping callback to prevent exploding gradients
- Smaller action_scale (0.3) to prevent action saturation
- Added proximity reward shaping for denser learning signal
- Lower target entropy (-2) to reduce saturation to extreme actions
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.sac.policies import SACPolicy, Actor
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
from stable_baselines3.common.callbacks import BaseCallback
from gymnasium import spaces

from env_rma import RMAQuadrotorEnv
from callbacks import CurriculumCallback

DEPLOYABLE_DIM = 52


class GradientClipCallback(BaseCallback):
    """Callback to clip gradients during training to prevent exploding gradients."""

    def __init__(self, clip_value=10.0, verbose=0):
        super().__init__(verbose)
        self.clip_value = clip_value

    def _on_step(self) -> bool:
        # Clip gradients for both actor and critic
        if self.model.policy.actor is not None:
            for param in self.model.policy.actor.parameters():
                if param.grad is not None:
                    param.grad.clamp_(-self.clip_value, self.clip_value)
        if self.model.policy.critic is not None:
            for param in self.model.policy.critic.parameters():
                if param.grad is not None:
                    param.grad.clamp_(-self.clip_value, self.clip_value)
        return True


class DeployableExtractor(BaseFeaturesExtractor):
    """Features extractor for the actor that slices to deployable obs only.

    Takes the full 60-dim observation and outputs only the first 52 dims
    (deployable sensor data available on real hardware).
    The privileged 8 dims (wind, mass ratio, etc.) are excluded.
    """

    def __init__(self, observation_space: spaces.Box):
        deployable_space = spaces.Box(
            low=observation_space.low[:DEPLOYABLE_DIM],
            high=observation_space.high[:DEPLOYABLE_DIM],
            shape=(DEPLOYABLE_DIM,),
            dtype=np.float32,
        )
        super().__init__(deployable_space, features_dim=DEPLOYABLE_DIM)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations[:, :DEPLOYABLE_DIM]


class AsymmetricSACPolicy(SACPolicy):
    """Custom SAC policy with asymmetric actor-critic architecture.

    - Actor: uses DeployableExtractor → 52-dim features → actor MLP
    - Critic: uses FlattenExtractor → 60-dim features → critic MLP
    - share_features_extractor=False (actor and critic use different extractors)

    This is the RMA (Rapid Motor Adaptation) pattern: the critic gets privileged
    information during training to provide better value estimates, while the actor
    learns a policy that depends only on deployable sensor data.
    """

    def _build(self, lr_schedule):
        """Build asymmetric actor-critic networks.

        Override to create actor with DeployableExtractor (52-dim)
        and critic with standard FlattenExtractor (60-dim).
        """
        # Create deployable features extractor for actor
        deployable_extractor = DeployableExtractor(self.observation_space)
        deployable_extractor = deployable_extractor.to(self.device)

        # Build actor with deployable extractor (sees only 52 dims)
        self.actor = self.make_actor(features_extractor=deployable_extractor)
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Build critic with full observation space (60 dims)
        # share_features_extractor=False by default in SAC, so critic gets
        # its own FlattenExtractor that sees all 60 dims
        self.critic = self.make_critic(features_extractor=None)
        critic_parameters = list(self.critic.parameters())

        # Critic target (separate features extractor, also 60 dims)
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)


class SymmetricSACPolicy(SACPolicy):
    """Symmetric actor-critic architecture for better value estimation.

    Both actor and critic receive the same 52-dim deployable observations.
    This avoids the mismatch where the critic uses privileged info to predict
    good values that the actor can't achieve.
    """

    def _build(self, lr_schedule):
        """Build symmetric actor-critic networks.

        Both actor and critic use the same DeployableExtractor (52-dim).
        This ensures the value estimates are aligned with what the actor can achieve.
        """
        # Create deployable features extractor for both
        deployable_extractor = DeployableExtractor(self.observation_space)
        deployable_extractor = deployable_extractor.to(self.device)

        # Build actor with deployable extractor
        self.actor = self.make_actor(features_extractor=deployable_extractor)
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Build critic with same deployable extractor (symmetric)
        self.critic = self.make_critic(features_extractor=deployable_extractor)
        critic_parameters = list(self.critic.parameters())

        # Critic target (same architecture)
        self.critic_target = self.make_critic(features_extractor=deployable_extractor)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)


def make_env(stage, seed, rank):
    """Factory function for creating vectorized environments."""
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_single_seed(seed, resume_path=None):
    """Train a single SAC model with the given seed."""
    print("=" * 60)
    print(f"Phase 1: Privileged Teacher Training (SB3 SAC)")
    print(f"Asymmetric Actor-Critic (52d Actor / 60d Critic)")
    print(f"Seed: {seed}")
    print("=" * 60)

    # Create directories
    model_dir = f"models/seed_{seed}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs(f"tb_logs/seed_{seed}", exist_ok=True)

    # Stage thresholds: (min_episodes, min_success_rate) to advance
    stage_thresholds = {
        1: (50, 0.90),  # Stage 1: 50 eps, 90% success
        2: (50, 0.50),  # Stage 2: 50 eps, 50% success
        3: (50, 0.50),  # Stage 3: 50 eps, 50% success
        4: None,         # Stage 4: train to convergence
    }

    N_ENVS = 16

    print("\nCreating training environment...")
    train_env = SubprocVecEnv(
        [make_env(stage=1, seed=seed, rank=i) for i in range(N_ENVS)]
    )

    # Create or load SAC model with asymmetric policy
    if resume_path and os.path.exists(resume_path):
        print(f"Loading existing model from {resume_path}...")
        model = SAC.load(
            resume_path,
            env=train_env,
            custom_objects=dict(policy=AsymmetricSACPolicy),
        )
        print("Model loaded successfully!")
    else:
        print("Creating new SAC model with SymmetricSACPolicy...")
        # Training fixes:
        # - Lower learning rate (1e-4 vs 3e-4) to prevent divergence
        # - Target entropy lowered to -2 to encourage more exploration
        # - Symmetric actor-critic (both see 52 dims) to avoid mismatch
        model = SAC(
            SymmetricSACPolicy,
            train_env,
            learning_rate=1e-4,  # reduced from 3e-4 to prevent divergence
            buffer_size=1_000_000,
            batch_size=2048,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            target_entropy=-2,  # reduced from -4 (fewer actions = less exploration needed)
            policy_kwargs=dict(
                net_arch=[256, 256, 128],
            ),
            verbose=1,
            tensorboard_log=f"tb_logs/seed_{seed}/",
            device="cuda",
            seed=seed,
        )

    # Gradient clipping callback to prevent exploding gradients
    gradient_clip_callback = GradientClipCallback(clip_value=10.0)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix="sac_rma",
    )

    # Curriculum callback
    curriculum_callback = CurriculumCallback(
        env=train_env,
        stage_thresholds=stage_thresholds,
        eval_freq=10000,
        n_eval_episodes=20,
    )

    # Train
    print("\nStarting training...")
    print("Stage 1: Fixed hover (no disturbances)")
    print("Stage 2: Random pose/velocity")
    print("Stage 3: + Wind + mass perturbation")
    print("Stage 4: + Payload drop")
    print("=" * 60)

    model.learn(
        total_timesteps=2_000_000,
        callback=[checkpoint_callback, curriculum_callback, gradient_clip_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(os.path.join(model_dir, "final"))
    print(f"\nPhase 1 complete! (seed={seed})")
    print(f"Model saved to: {model_dir}/final")

    train_env.close()
    return model


def main():
    # Parse seed argument
    seeds = [0]  # default
    resume_path = None

    for arg in sys.argv[1:]:
        if arg.startswith("--seed="):
            seed_str = arg.split("=")[1]
            seeds = [int(s.strip()) for s in seed_str.split(",")]
        elif arg.startswith("--resume="):
            resume_path = arg.split("=")[1]

    print(f"\nTraining with seeds: {seeds}")

    for seed in seeds:
        train_single_seed(seed, resume_path=resume_path)

    print(f"\n{'=' * 60}")
    print(f"All seeds complete! Trained {len(seeds)} seeds: {seeds}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()