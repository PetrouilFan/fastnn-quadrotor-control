#!/usr/bin/env python3
"""
State Estimation Approach:
- Maintain observation history
- Use learned state estimator to predict current state from history
- Feed predicted state to policy
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import ObservationWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from datetime import datetime
from collections import deque

from env_rma import RMAQuadrotorEnv


class StateEstimatorWrapper(ObservationWrapper):
    """
    Uses observation history to estimate current state.
    Simple approach: weighted average of recent observations
    Better approach: learn a predictor
    """
    
    def __init__(self, env, history_len=8):
        super().__init__(env)
        self.history_len = history_len
        self._obs_history = deque(maxlen=history_len)
        
        self.observation_space = env.observation_space
    
    def observation(self, obs):
        self._obs_history.append(obs.copy())
        
        if len(self._obs_history) < self.history_len:
            return obs.copy()
        
        # Simple state estimation: weighted average (recent = higher weight)
        # weights = linear from 0.2 to 1.0
        n = len(self._obs_history)
        weights = np.linspace(0.2, 1.0, n)
        weights = weights / weights.sum()
        
        estimated = np.zeros_like(obs)
        for i, o in enumerate(self._obs_history):
            estimated += weights[i] * o
        
        return estimated
    
    def reset(self, seed=None, options=None):
        self._obs_history.clear()
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self.history_len):
            self._obs_history.append(obs.copy())
        return obs, info


def make_env(stage, history_len, seed, rank):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=seed + rank)
        return StateEstimatorWrapper(env, history_len=history_len)
    return _init


def train_state_estimator(stage=5, history_len=8, total_steps=2000000, n_envs=8, seed=0, save_dir="runs/state_est_trained"):
    print(f"State Estimator Training: history={history_len}")
    print(f"Total steps: {total_steps}, N envs: {n_envs}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}/stage_{stage}_h{history_len}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    env_fns = [make_env(stage, history_len, seed, i) for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)
    
    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4, buffer_size=100000, learning_starts=1000,
        batch_size=256, tau=0.005, gamma=0.99, ent_coef='auto',
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed, verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=run_dir + "/checkpoints", name_prefix="sac")
    
    print(f"Training started at {timestamp}")
    
    model.learn(total_timesteps=total_steps, callback=checkpoint_callback, progress_bar=True)
    
    model.save(run_dir + "/final.zip")
    print(f"Saved to {run_dir}/final.zip")
    
    train_env.close()
    return run_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="runs/state_est_trained")
    args = parser.parse_args()
    
    train_state_estimator(stage=args.stage, history_len=args.history, total_steps=args.steps, n_envs=args.n_envs, seed=args.seed, save_dir=args.save_dir)