#!/usr/bin/env python3
"""Stage 16 with 2-action history."""

import numpy as np
import argparse
from gymnasium import spaces, ObservationWrapper
import os

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv


class SimpleObsWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_dim = env.observation_space.shape[0]
        new_dim = old_dim + 6
        self.observation_space = spaces.Box(low=-20, high=20, shape=(new_dim,), dtype=np.float32)
        self._buffer = [np.zeros(3)] * 2
    
    def observation(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]
        flat = obs.flatten()[:63]
        
        self._buffer.append(flat[-3:].copy() if len(flat) >= 3 else flat.copy())
        self._buffer.pop(0)
        
        hist = np.concatenate([self._buffer[0], self._buffer[1]])
        
        return np.concatenate([flat, hist]).astype(np.float32)
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._buffer = [np.zeros(3)] * 2
        return self.observation(obs), info


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=300000)
    args = parser.parse_args()
    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/stage16_{ts}"
    os.makedirs(run_dir, exist_ok=True)
    
    def make_env():
        env = RMAQuadrotorEnv(curriculum_stage=5, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset()
        return SimpleObsWrapper(env)
    
    train_env = DummyVecEnv([make_env for _ in range(8)])
    
    print(f"Obs dim: {train_env.observation_space.shape}")
    
    model = SAC("MlpPolicy", train_env, learning_rate=3e-4, batch_size=256,
                buffer_size=150000, learning_starts=1000, verbose=1)
    
    print(f"Training {args.steps} steps...")
    model.learn(total_timesteps=args.steps, progress_bar=True)
    
    model.save(f"{run_dir}/final.zip")
    print(f"Saved {run_dir}/final.zip")
    train_env.close()