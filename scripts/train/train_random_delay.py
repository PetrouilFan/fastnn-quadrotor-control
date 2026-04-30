#!/usr/bin/env python3
"""Training with randomized delays."""

import numpy as np
import argparse
import torch
from gymnasium import ObservationWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from datetime import datetime

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class RandomDelayWrapper(ObservationWrapper):
    def __init__(self, env, max_delay_steps=10):
        super().__init__(env)
        self.max_delay_steps = max_delay_steps
        self._obs_buffer = []
        self._current_delay = 0
    
    def observation(self, obs):
        self._obs_buffer.append(obs.copy())
        if len(self._obs_buffer) > self.max_delay_steps + 1:
            self._obs_buffer.pop(0)
        if len(self._obs_buffer) > 0:
            return self._obs_buffer[0].copy()
        return obs.copy()
    
    def reset(self, seed=None, options=None):
        self._current_delay = np.random.randint(0, self.max_delay_steps + 1)
        self._obs_buffer = []
        obs, info = self.env.reset(seed=seed, options=options)
        for _ in range(self._current_delay):
            self._obs_buffer.append(obs.copy())
        return obs, info


def make_env(stage, max_delay, seed, rank):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=seed + rank)
        return RandomDelayWrapper(env, max_delay_steps=max_delay)
    return _init


def train_random_delay(stage=5, max_delay=10, total_steps=2000000, n_envs=8, seed=0, save_dir="runs/random_delay_trained"):
    print(f"Random Delay Training: max {max_delay*10}ms")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}/stage_{stage}_max{max_delay*10}ms_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    env_fns = [make_env(stage, max_delay, seed, i) for i in range(n_envs)]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--max-delay", type=int, default=10)
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="runs/random_delay_trained")
    args = parser.parse_args()
    
    train_random_delay(stage=args.stage, max_delay=args.max_delay, total_steps=args.steps, n_envs=args.n_envs, seed=args.seed, save_dir=args.save_dir)