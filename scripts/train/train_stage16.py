#!/usr/bin/env python3
"""
Stage 16: Full Obs + Action History Training

Uses full observation (63-dim) + action history (6-dim past 2 actions)
Total: 69-dim observation

This combines temporal memory with full state for IMU deployment.
"""

import numpy as np
import argparse
import torch
from gymnasium import spaces, ObservationWrapper
import os
from datetime import datetime

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv


class ActionHistoryWrapper(ObservationWrapper):
    """Simply adds action history to observation."""
    
    def __init__(self, env):
        super().__init__(env)
        self._action_buffer = []
        self._max_actions = 2
        
        obs_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, 
            shape=(obs_dim + 3 * self._max_actions,), 
            dtype=np.float32
        )
    
    def observation(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        
        while len(self._action_buffer) < self._max_actions:
            self._action_buffer.insert(0, np.zeros(3, dtype=np.float32))
        
        action_hist = np.concatenate(self._action_buffer)
        
        return np.concatenate([obs, action_hist]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        self._action_buffer = []
        obs, info = self.env.reset(seed=seed, options=options)
        return self.observation(obs), info
    
    def step(self, action):
        while len(self._action_buffer) < self._max_actions:
            self._action_buffer.append(np.zeros(3, dtype=np.float32))
        
        self._action_buffer.append(action.copy())
        if len(self._action_buffer) > self._max_actions:
            self._action_buffer.pop(0)
        
        result = self.env.step(action)
        
        if len(result) == 5:
            raw_next, reward, term, trunc, info = result
        else:
            raw_next, reward, term = result[:3]
            trunc, info = False, {}
        
        return self.observation(raw_next), reward, term, trunc, info


def make_env(stage, seed, rank):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=seed + rank)
        return ActionHistoryWrapper(env)
    return _init


def train_stage16(stage=5, total_steps=300000, n_envs=8, seed=0, save_dir="runs/stage16"):
    print(f"Stage 16: Full obs + action history = {63+6}=69 dimensions")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    env_fns = [make_env(stage, seed, i) for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)
    
    print(f"Obs space: {train_env.observation_space.shape}")
    
    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        verbose=1,
    )
    
    print(f"Training started at {timestamp}")
    
    model.learn(total_timesteps=total_steps, progress_bar=True)
    
    model.save(run_dir + "/final.zip")
    print(f"Saved to {run_dir}/final.zip")
    
    train_env.close()
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--steps", type=int, default=300000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    train_stage16(args.stage, args.steps, args.n_envs, args.seed)