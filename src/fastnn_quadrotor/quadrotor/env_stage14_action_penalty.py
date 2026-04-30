#!/usr/bin/env python3
"""
Stage 14 with action-rate penalty: Reduces aggressive setpoint changes.

Action-rate penalty penalizes large changes between consecutive actions,
improving actuator realism and reducing jitter.
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict


from fastnn_quadrotor.quadrotor.env_stage12_hist import QuadrotorEnvStage12Hist


class QuadrotorEnvStage14ActionPenalty(gym.Env):
    """Stage 12Hist with action-rate penalty for smoother control."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, base_env: "QuadrotorEnvStage12Hist" = None, action_penalty_scale: float = 0.1):
        super().__init__()
        
        self.base_env = base_env
        self._action_penalty_scale = action_penalty_scale
        self._prev_action = np.zeros(4, dtype=np.float32)
        
        self.observation_space = base_env.observation_space if base_env else spaces.Dict({})
        self.action_space = base_env.action_space if base_env else spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._prev_action = np.zeros(4, dtype=np.float32)
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action_diff = np.linalg.norm(action - self._prev_action)
        action_penalty = -self._action_penalty_scale * action_diff
        
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        reward += action_penalty
        info["action_diff"] = action_diff
        info["action_penalty"] = action_penalty
        
        self._prev_action = action.copy()
        
        return obs, reward, terminated, truncated, info

    @property
    def render_mode(self):
        return self.base_env.render_mode

    def render(self):
        return self.base_env.render()


def make_env(s10, vecnorm, action_penalty_scale=0.1):
    def _init():
        from fastnn_quadrotor.quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
        base = QuadrotorEnvStage11Trajectory(s10, vecnorm)
        hist = QuadrotorEnvStage12Hist(base)
        return QuadrotorEnvStage14ActionPenalty(hist, action_penalty_scale)
    return _init


if __name__ == "__main__":
    from fastnn_quadrotor.quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    
    s10 = "runs/stage10_hierarchical/hierarchical_20260423_081224/best_model/best_model.zip"
    vecnorm = "runs/stage10_hierarchical/hierarchical_20260423_081224/best_model/vecnormalize.pkl"
    
    base = QuadrotorEnvStage11Trajectory(s10, vecnorm)
    hist = QuadrotorEnvStage12Hist(base)
    env = QuadrotorEnvStage14ActionPenalty(hist, action_penalty_scale=0.1)
    
    obs, info = env.reset(seed=42)
    print(f"Obs: {obs['observation'].shape}")
    
    total_penalty = 0
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_penalty += info.get("action_penalty", 0)
        if term or trunc:
            break
    
    print(f"Steps: {i+1}")
    print(f"Total action penalty: {total_penalty:.3f}")
    print("PASSED")