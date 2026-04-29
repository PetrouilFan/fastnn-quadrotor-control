#!/usr/bin/env python3
"""
Temporal SAC: Standard SAC with Temporal Context

This experiment tests whether temporal awareness specifically helps SAC,
by augmenting the state with past observations (action history + error integrals).

Two configurations:
1. Standard SAC: 51-dim state
2. Temporal SAC: 51-dim state + 4 previous obs (history) + action history (16) = large state

This tests the "GRU hypothesis": does temporal awareness help with payload drop?

Usage:
    python train_temporal_sac.py --steps 1000000 --stage 3 --temporal
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from tqdm import tqdm
import os
import argparse

from env_rma import RMAQuadrotorEnv


DEPLOYABLE_DIM = 51
ACTION_DIM = 4


from gymnasium.core import Env

class TemporalHistoryWrapper(Env):
    """
    Wraps environment to prepend historical observations to current state.

    Instead of obs_t, we provide [obs_t, obs_t-1, obs_t-2, obs_t-3] to give
    temporal context without changing the RL algorithm.
    """

    metadata = {"render_modes": []}

    def __init__(self, env, history_len=4):
        from gymnasium.spaces import Space
        self.env = env
        self.history_len = history_len
        self.obs_history = []
        self.render_mode = None
        self._step_count = 0

        # Determine base observation dimension
        base_obs, _ = self.env.reset()
        base_dim = len(base_obs)
        self._base_dim = base_dim

        # Compute new observation space
        # base_dim * (history_len + 1) for obs history + ACTION_DIM * history_len for action history
        new_dim = base_dim * (history_len + 1) + ACTION_DIM * history_len

        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(new_dim,), dtype=np.float32
        )
        self.action_space = env.action_space
        self.reward_range = env.reward_range if hasattr(env, 'reward_range') else (-float('inf'), float('inf'))
        self.spec = env.spec if hasattr(env, 'spec') else None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._step_count = 0
        # Initialize history with current observation
        self.obs_history = [obs.copy()] * self.history_len
        return self._get_augmented_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1

        # Update history
        self.obs_history.append(obs.copy())
        if len(self.obs_history) > self.history_len:
            self.obs_history.pop(0)

        return self._get_augmented_obs(), reward, terminated, truncated, info

    def _get_augmented_obs(self):
        """Get observation with history prepended."""
        # Current obs first, then history (most recent first)
        obs_with_history = []

        # Current observation (index 0)
        current_obs = self.obs_history[-1] if self.obs_history else self.env._get_obs()[:self._base_dim]
        obs_with_history.append(current_obs)

        # Historical observations
        for i in range(self.history_len - 1, -1, -1):
            if len(self.obs_history) > i + 1:
                obs_with_history.append(self.obs_history[-(i + 2)])
            elif len(self.obs_history) > 0:
                obs_with_history.append(self.obs_history[0])

        # Action history (4 previous actions)
        if hasattr(self.env, 'action_history'):
            action_hist = self.env.action_history.flatten()
        else:
            action_hist = np.zeros(ACTION_DIM * self.history_len, dtype=np.float32)

        return np.concatenate(obs_with_history + [action_hist]).astype(np.float32)

    @property
    def data(self):
        return self.env.data

    @property
    def target_pos(self):
        return self.env.target_pos

    @property
    def np_random(self):
        return self.env.np_random

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    def _cascaded_controller(self):
        return self.env._cascaded_controller()

    def get_privileged_info(self):
        return self.env.get_privileged_info()


def make_env(stage, seed, rank, use_temporal=False):
    def _init():
        base_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        base_env.reset(seed=seed + rank)
        if use_temporal:
            return TemporalHistoryWrapper(base_env, history_len=4)
        return base_env
    return _init


def train_temporal_sac(stage=3, total_steps=500_000, n_envs=16, seeds=[0], use_temporal=False):
    """Train SAC with or without temporal context."""
    results = {}

    suffix = "_temporal" if use_temporal else "_standard"
    prefix = "Temporal" if use_temporal else "Standard"

    for seed in seeds:
        print("=" * 60)
        print(f"{prefix} SAC - Stage {stage}, Seed {seed}")
        print("=" * 60)

        model_dir = f"models_temporal_sac{suffix}/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)

        # Create environments
        env_fns = [make_env(stage=stage, seed=seed, rank=i, use_temporal=use_temporal) for i in range(n_envs)]
        train_env = SubprocVecEnv(env_fns)

        eval_env_base = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        eval_env_base.reset(seed=seed)
        if use_temporal:
            eval_env = TemporalHistoryWrapper(eval_env_base, history_len=4)
        else:
            eval_env = eval_env_base

        # Determine observation dimension
        obs_dim = train_env.observation_space.shape[0]
        print(f"Observation dimension: {obs_dim}")

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
            tensorboard_log=f"tb_logs_temporal_sac{suffix}/stage_{stage}/seed_{seed}/",
            device="cuda" if torch.cuda.is_available() else "cpu",
            seed=seed,
            verbose=1,
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path=model_dir,
            name_prefix="temporal_sac",
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
    parser = argparse.ArgumentParser(description="Temporal SAC vs Standard SAC")
    parser.add_argument("--stage", type=int, default=3, help="Curriculum stage")
    parser.add_argument("--steps", type=int, default=500_000, help="Training steps")
    parser.add_argument("--n-envs", type=int, default=16, help="Parallel environments")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    parser.add_argument("--temporal", action="store_true", help="Use temporal context")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = train_temporal_sac(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seeds=seeds,
        use_temporal=args.temporal,
    )

    print("\n=== Summary ===")
    for seed, result in results.items():
        print(f"Seed {seed}: {result['final_success']:.1%} success")


if __name__ == "__main__":
    main()