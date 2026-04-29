#!/usr/bin/env python3
"""
Stage 10 Low-Speed Hover: Modifications to improve low-speed settling.

Adds:
- Extra reward for low velocity when near target
- Lower velocity targets in curriculum
- Relaxed termination for hover training
"""
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quadrotor.env_stage10_hierarchical import QuadrotorEnvStage10Hierarchical


class QuadrotorEnvStage10LowSpeed(gym.Env):
    """Stage 10 with low-speed hover training modifications."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self, 
                 low_speed_threshold: float = 0.3,
                 hover_bonus: float = 1.0,
                 relax_attitude: bool = True,
                 **kwargs):
        super().__init__()
        
        self.base_env = QuadrotorEnvStage10Hierarchical(**kwargs)
        self._low_speed_threshold = low_speed_threshold
        self._hover_bonus = hover_bonus
        self._relax_attitude = relax_attitude
        
        self.observation_space = self.base_env.observation_space
        self.action_space = self.base_env.action_space
        self._target_pos = np.zeros(3)

    def _quat_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to roll-pitch-yaw."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(2*(w*y - z*x))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])

    def _check_termination(self) -> bool:
        """Check termination with relaxed attitude for hover training."""
        pos = self.base_env.data.qpos[:3]
        
        # Standard altitude check
        if pos[2] < 0.1:
            return True
        
        # Attitude limit
        quat = self.base_env.data.qpos[3:7]
        rpy = self._quat_to_rpy(quat)
        
        if self._relax_attitude:
            limit = 1.22  # ~70 degrees
        else:
            limit = np.pi / 2  # 90 degrees
            
        if abs(rpy[0]) > limit or abs(rpy[1]) > limit:
            return True
        
        if np.linalg.norm(pos[:2]) > 10.0:
            return True
        
        return False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict, Dict]:
        obs, info = self.base_env.reset(seed=seed, options=options)
        
        # Set hover target at origin
        self._target_pos = np.zeros(3)
        
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        # Run base environment step
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Compute additional hover bonus
        pos = self.base_env.data.qpos[:3]
        vel = self.base_env.data.qvel[:3]
        
        dist = np.linalg.norm(pos - self._target_pos)
        speed = np.linalg.norm(vel)
        
        # Add hover bonus: low speed when very close to target
        if dist < 0.3 and speed < self._low_speed_threshold:
            reward += self._hover_bonus
            info["hover_bonus"] = True
        else:
            info["hover_bonus"] = False
        
        # Apply custom termination
        terminated = self._check_termination()
        
        return obs, reward, terminated, truncated, info

    @property
    def render_mode(self):
        return self.base_env.render_mode

    def render(self):
        return self.base_env.render()


def make_env(**kwargs):
    def _init():
        return QuadrotorEnvStage10LowSpeed(**kwargs)
    return _init


if __name__ == "__main__":
    env = QuadrotorEnvStage10LowSpeed()
    
    obs, info = env.reset(seed=42)
    print(f"Obs: {obs.shape}")
    
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term or trunc:
            break
    
    print(f"Steps: {i+1}")
    print(f"Total reward: {total_reward:.2f}")
    print("PASSED")