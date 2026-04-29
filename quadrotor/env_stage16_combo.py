#!/usr/bin/env python3
"""
Stage 16 Combo: Body-frame primitives + Action history.

Combines Stage 13 body-frame with Stage 14 action history for:
- IMU-compatible body-frame commands
- Delay compensation via action history
- Robustness to both world-frame drift and latency
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadrotor.env_stage13_bodyframe import QuadrotorEnvStage13Bodyframe


class QuadrotorEnvStage16Combo(gym.Env):
    """Body-frame + action history combination."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, base_env: "QuadrotorEnvStage13Bodyframe" = None):
        super().__init__()
        
        self.base_env = base_env
        
        # Action buffer for history
        self._action_buffer = [np.zeros(3, dtype=np.float32) for _ in range(3)]
        
        # Body-frame obs (14-dim) + action history (9-dim) = 23-dim
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-50.0, high=50.0, shape=(23,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
        })
        self.action_space = base_env.action_space if base_env else spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _augment_obs(self, obs: Dict) -> Dict:
        a_hist = np.concatenate([
            self._action_buffer[-1],
            self._action_buffer[-2],
            self._action_buffer[-3],
        ])
        
        obs23 = np.concatenate([obs["observation"], a_hist])
        obs23 = np.clip(obs23, -50.0, 50.0)
        
        return {
            "observation": obs23,
            "achieved_goal": obs["achieved_goal"],
            "desired_goal": obs["desired_goal"],
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        obs, info = self.base_env.reset(seed=seed, options=options)
        self._action_buffer = [np.zeros(3, dtype=np.float32) for _ in range(3)]
        return self._augment_obs(obs), info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        self._action_buffer.pop(0)
        self._action_buffer.append(action.copy())
        
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        return self._augment_obs(obs), reward, terminated, truncated, info

    @property
    def render_mode(self):
        return self.base_env.render_mode

    def render(self):
        return self.base_env.render()


def make_env(s10_path, vecnorm_path=None):
    def _init():
        from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
        
        if vecnorm_path:
            s10 = s10_path
            vecnorm = vecnorm_path
        else:
            s10 = s10_path
            vecnorm = s10_path.replace(".zip", "_vecnormalize.pkl")
            if not os.path.exists(vecnorm):
                vecnorm = s10_path.replace(".zip", "_vec_normalize.pkl")
        
        base = QuadrotorEnvStage11Trajectory(s10, vecnorm)
        bodyframe = QuadrotorEnvStage13Bodyframe(base)
        return QuadrotorEnvStage16Combo(bodyframe)
    return _init


if __name__ == "__main__":
    from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
    
    s10 = "runs/stage10_hierarchical/hierarchical_20260423_081224/best_model/best_model.zip"
    vecnorm = "runs/stage10_hierarchical/hierarchical_20260423_081224/best_model/vecnormalize.pkl"
    
    base = QuadrotorEnvStage11Trajectory(s10, vecnorm)
    bodyframe = QuadrotorEnvStage13Bodyframe(base)
    env = QuadrotorEnvStage16Combo(bodyframe)
    
    obs, info = env.reset(seed=42)
    print(f"Obs: {obs['observation'].shape}")
    
    for i in range(50):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            break
    
    print(f"Steps: {i+1}")
    print("PASSED")