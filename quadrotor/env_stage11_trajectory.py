#!/usr/bin/env python3
"""
Stage 11: Trajectory tracking.

Key insight: Stage 10 is unstable at low speeds. Instead of hovering at a static waypoint,
we generate continuous trajectories that the pilot must track. The reference moves
continuously, keeping the vehicle in the dynamic regime where Stage 10 works.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from quadrotor.env_stage10_hierarchical import QuadrotorEnvStage10Hierarchical


class GoalEnv(gym.Env):
    """Simple GoalEnv base."""
    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError


class QuadrotorEnvStage11Trajectory(GoalEnv):
    """Trajectory tracking environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        stage10_model_path: str = "__PLACEHOLDER__",
        stage10_vecnormalize_path: Optional[str] = None,
        max_episode_steps: int = 5000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        import gymnasium as gym
        self.goal_env = gym
        
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        # Stage 10 env
        self.s10 = QuadrotorEnvStage10Hierarchical(
            max_episode_steps=max_episode_steps,
            motor_lag_tau=0.08,
            enable_domain_randomization=True,
            reward_mode="acro",
        )
        
        # Load Stage 10 model
        self.s10_model = None
        if stage10_model_path != "__PLACEHOLDER__":
            self.s10_model = SAC.load(stage10_model_path)

        # Load VecNormalize stats for Stage 10 inference
        self.s10_obs_mean = None
        self.s10_obs_std = None
        if stage10_vecnormalize_path and os.path.exists(stage10_vecnormalize_path):
            self._load_vecnormalize_stats(stage10_vecnormalize_path)
        elif stage10_model_path != "__PLACEHOLDER__":
            model_dir = os.path.dirname(stage10_model_path)
            candidates = [
                os.path.join(model_dir, "vecnormalize.pkl"),
                os.path.join(model_dir, "vecnormalize_best_model.pkl"),
            ]
            import glob
            checkpoints_dir = os.path.join(model_dir, "checkpoints")
            if os.path.isdir(checkpoints_dir):
                pkl_files = glob.glob(os.path.join(checkpoints_dir, "*vecnormalize*.pkl"))
                if pkl_files:
                    candidates.append(sorted(pkl_files)[-1])
            for c in candidates:
                if os.path.exists(c):
                    self._load_vecnormalize_stats(c)
                    break
        
        # Trajectory parameters
        self._min_velocity = 0.5  # m/s - keep in dynamic regime
        self._trajectory_speed = 1.0  # m/s
        self._trajectory_radius = 0.3  # radius for circular/spline paths
        
        # Spaces
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-50.0, high=50.0, shape=(21,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Tracking
        self._step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._ref_pos = np.zeros(3, dtype=np.float32)
        self._ref_velocity = np.zeros(3, dtype=np.float32)
        self._trajectory_type = "circle"
        self._trajectory_time = 0.0
        self._episode_start = np.zeros(3)
        self._success_count = 0
        
        # Action smoothing
        self._action_alpha = 0.3
        self._smoothed_action = np.zeros(4, dtype=np.float32)
        
        # Rendering
        self.viewer = None

    def _load_vecnormalize_stats(self, path: str):
        import pickle
        try:
            with open(path, "rb") as f:
                vecnorm = pickle.load(f)
            self.s10_obs_mean = vecnorm.obs_rms.mean.astype(np.float32)
            self.s10_obs_std = np.sqrt(vecnorm.obs_rms.var + vecnorm.epsilon).astype(np.float32)
            print(f"Loaded VecNormalize stats from: {path}")
        except Exception as e:
            print(f"WARNING: Failed to load VecNormalize stats: {e}")

    def _normalize_s10_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.s10_obs_mean is not None:
            return np.clip((obs - self.s10_obs_mean) / self.s10_obs_std, -10.0, 10.0)
        return obs

    def _generate_trajectory(self) -> Tuple[str, np.ndarray, np.ndarray]:
        """Generate a trajectory type and initial reference."""
        types = ["circle", "figure8", "line", "spiral"]
        self._trajectory_type = self.np_random.choice(types)
        
        start_pos = self.s10.data.qpos[:3].copy()
        self._episode_start = start_pos.copy()
        
        # Initialize reference at current position
        self._ref_pos = start_pos.copy()
        self._trajectory_time = 0.0
        
        # Initial reference velocity
        direction = self.np_random.uniform(-1, 1, size=3)
        direction[2] = 0  # mainly horizontal
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        self._ref_velocity = direction * self._trajectory_speed
        
        return self._trajectory_type, self._ref_pos.copy(), self._ref_velocity.copy()

    def _update_reference(self, dt: float = 0.01):
        """Update moving reference position."""
        self._trajectory_time += dt
        
        if self._trajectory_type == "circle":
            t = self._trajectory_time
            center = self._episode_start + np.array([self._trajectory_radius, 0, 0])
            self._ref_pos[0] = center[0] + self._trajectory_radius * np.cos(self._trajectory_speed * t / self._trajectory_radius)
            self._ref_pos[1] = center[1] + self._trajectory_radius * np.sin(self._trajectory_speed * t / self._trajectory_radius)
            self._ref_pos[2] = self._episode_start[2] + 0.1 * np.sin(t)
            
        elif self._trajectory_type == "figure8":
            t = self._trajectory_time
            self._ref_pos[0] = self._episode_start[0] + self._trajectory_radius * np.sin(t)
            self._ref_pos[1] = self._episode_start[1] + self._trajectory_radius * np.sin(t) * np.cos(t)
            self._ref_pos[2] = self._episode_start[2] + 0.1 * np.sin(2*t)
            
        elif self._trajectory_type == "line":
            direction = self._ref_velocity / (np.linalg.norm(self._ref_velocity) + 1e-8)
            self._ref_pos += direction * self._trajectory_speed * dt
            # Wrap back if too far
            if np.linalg.norm(self._ref_pos - self._episode_start) > 3.0:
                self._ref_pos = self._episode_start.copy()
                
        elif self._trajectory_type == "spiral":
            t = self._trajectory_time
            r = 0.5 + 0.3 * np.sin(t * 0.5)
            self._ref_pos[0] = self._episode_start[0] + r * np.cos(t)
            self._ref_pos[1] = self._episode_start[1] + r * np.sin(t)
            self._ref_pos[2] = self._episode_start[2] + 0.3 * np.sin(t * 0.3)
        
        # Compute reference velocity
        self._ref_velocity = (self._ref_pos - self._prev_ref_pos) / dt if hasattr(self, '_prev_ref_pos') else np.zeros(3)
        self._prev_ref_pos = self._ref_pos.copy()
        
        # Enforce minimum velocity
        speed = np.linalg.norm(self._ref_velocity)
        if speed < self._min_velocity and speed > 0:
            self._ref_velocity = self._ref_velocity / speed * self._min_velocity

    def _get_obs(self) -> Dict[str, np.ndarray]:
        pos = self.s10.data.qpos[:3].copy()
        vel = self.s10.data.qvel[:3].copy()
        quat = self.s10.data.qpos[3:7].copy()
        rpy = self.s10._quat_to_rpy(quat)
        
        tracking_error = self._ref_pos - pos
        ref_vel = self._ref_velocity
        
        obs = np.concatenate([
            pos,
            vel,
            rpy,
            tracking_error,
            ref_vel,
            self._prev_action,
            [np.linalg.norm(vel)],
            [np.linalg.norm(tracking_error)],
        ]).astype(np.float32)
        obs = np.clip(obs, -50.0, 50.0)
        
        return {
            "observation": obs,
            "achieved_goal": pos,
            "desired_goal": self._ref_pos.copy(),
        }

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict
    ) -> float:
        track_error = np.linalg.norm(desired_goal - achieved_goal)
        pos = achieved_goal
        
        r_tracking = -2.0 * track_error
        r_success = 50.0 if track_error < 0.3 else 0.0
        
        # Safety
        r_crash = -100.0 if pos[2] < 0.1 else 0.0
        
        return r_tracking + r_success + r_crash

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict, Dict]:
        super().reset(seed=seed)
        self.s10.np_random = self.np_random
        
        s10_obs, s10_info = self.s10.reset(seed=seed, options=options)
        self.s10.rc_input = np.zeros(4, dtype=np.float32)
        
        self._step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._generate_trajectory()
        self._prev_ref_pos = self._ref_pos.copy()
        self._success_count = 0
        self._smoothed_action = np.zeros(4, dtype=np.float32)
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        # Smooth action
        action = np.clip(action, -0.6, 0.6)
        self._smoothed_action = self._action_alpha * self._smoothed_action + (1 - self._action_alpha) * action
        action = self._smoothed_action.copy()
        
        self.s10.rc_input = action.copy()
        
        s10_obs = self.s10._get_obs()
        s10_obs_norm = self._normalize_s10_obs(s10_obs)
        
        s10_action, _ = self.s10_model.predict(s10_obs_norm, deterministic=False)
        if s10_action.ndim > 1:
            s10_action = s10_action[0]
        
        _, _, terminated, truncated, _ = self.s10.step(s10_action)
        self.s10.rc_input = action.copy()
        
        self._update_reference()
        
        pos = self.s10.data.qpos[:3]
        track_error = np.linalg.norm(self._ref_pos - pos)
        
        if track_error < 0.3:
            self._success_count += 1
        
        if track_error > 2.5:
            terminated = True
        
        self._step_count += 1
        if self._step_count >= self.max_episode_steps:
            truncated = True
        
        self._prev_action = action.copy()
        
        info = {
            "tracking_error": track_error,
            "ref_pos": self._ref_pos.copy(),
            "success_count": self._success_count,
        }
        
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = self.s10.viewer
        return self.s10.render()