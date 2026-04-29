#!/usr/bin/env python3
"""
Stage 11: Trajectory tracking with lookahead and progress reward.

Improvements:
1. Lookahead path points (5 future points)
2. Progress-based reward (path progress, cross-track error, heading)
3. Track curriculum (easy → hard)
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, List


from stable_baselines3 import SAC
from fastnn_quadrotor.quadrotor.env_stage10_hierarchical import QuadrotorEnvStage10Hierarchical


class GoalEnv(gym.Env):
    """Simple GoalEnv base."""
    def compute_reward(self, achieved_goal, desired_goal, info):
        raise NotImplementedError


class QuadrotorEnvStage11TrajectoryV2(GoalEnv):
    """Trajectory tracking environment v2 with lookahead and progress reward."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        stage10_model_path: str = "__PLACEHOLDER__",
        stage10_vecnormalize_path: Optional[str] = None,
        max_episode_steps: int = 5000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        
        self.s10 = QuadrotorEnvStage10Hierarchical(
            max_episode_steps=max_episode_steps,
            motor_lag_tau=0.08,
            enable_domain_randomization=True,
            reward_mode="acro",
        )
        
        self.s10_model = None
        if stage10_model_path != "__PLACEHOLDER__":
            self.s10_model = SAC.load(stage10_model_path)

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
                    self._ref_load_vecnormalize_stats(c)
                    break
        
        # Track curriculum (easy → hard) - GATED, not simultaneous
        self._track_curriculum = ["line", "oval", "large_circle", "small_circle", "figure8", "spline"]
        self._current_track_idx = 0  # Start at easiest
        
        # Curriculum gating
        self._ROLLING_WINDOW = 200  # episodes to check
        self._ADVANCE_THRESHOLD = 0.25  # mean CTE must be below this
        self._episode_cte_history = []  # rolling CTE history
        self._can_advance = True  # whether advancement is allowed
        
        # Parameters
        self._min_velocity = 0.8  # m/s (never stop)
        self._max_velocity = 1.2  # m/s (never too fast for Stage 10)
        self._trajectory_speed = 1.0  # m/s
        self._trajectory_radius = 1.2  # m (larger = less curvature)
        
        # Curvature constraint
        self._MAX_CURVATURE = 0.5  # rad/m max
        
        # Lookahead: 5 path points
        self._n_lookahead = 5
        self._lookahead_dt = 0.15  # seconds between points
        
        # Observation: pos(3) + vel(3) + rpy(3) + lookahead(15) + action_hist(4) + progress(1) + heading_error(1) + speed_ratio(1) = 31
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-50.0, high=50.0, shape=(31,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Tracking state
        self._step_count = 0
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._ref_pos = np.zeros(3, dtype=np.float32)
        self._trajectory_time = 0.0
        self._trajectory_type = "line"
        self._episode_start = np.zeros(3)
        self._total_distance = 0.0  # track progress
        self._success_count = 0
        
        # Action smoothing
        self._action_alpha = 0.3
        self._smoothed_action = np.zeros(4, dtype=np.float32)
        
        self.viewer = None

    def _ref_load_vecnormalize_stats(self, path: str):
        import pickle
        try:
            with open(path, "rb") as f:
                vecnorm = pickle.load(f)
            self.s10_obs_mean = vecnorm.obs_rms.mean.astype(np.float32)
            self.s10_obs_std = np.sqrt(vecnorm.obs_rms.var + vecnorm.epsilon).astype(np.float32)
        except:
            pass

    def _normalize_s10_obs(self, obs: np.ndarray) -> np.ndarray:
        if self.s10_obs_mean is not None:
            return np.clip((obs - self.s10_obs_mean) / self.s10_obs_std, -10.0, 10.0)
        return obs

    def _set_difficulty(self):
        """Set track difficulty based on curriculum gating."""
        # Check if we can advance
        if self._can_advance and len(self._episode_cte_history) >= self._ROLLING_WINDOW:
            recent_mean = np.mean(self._episode_cte_history[-self._ROLLING_WINDOW:])
            if recent_mean < self._ADVANCE_THRESHOLD:
                self._current_track_idx = min(
                    self._current_track_idx + 1, 
                    len(self._track_curriculum) - 1
                )
                self._can_advance = False  # Reset after advance
        
        # Current track type
        if self._current_track_idx < len(self._track_curriculum):
            self._trajectory_type = self._track_curriculum[self._current_track_idx]
        else:
            self._trajectory_type = self._np_random.choice(self._track_curriculum)
    
    def _record_episode_cte(self, mean_cte: float):
        """Record episode CTE for curriculum gating."""
        self._episode_cte_history.append(mean_cte)
        self._can_advance = True  # Allow advancement check next time
        # Keep history bounded
        if len(self._episode_cte_history) > 1000:
            self._episode_cte_history = self._episode_cte_history[-500:]

    def _generate_trajectory(self) -> Tuple[str, np.ndarray]:
        self._set_difficulty()
        
        start_pos = self.s10.data.qpos[:3].copy()
        self._episode_start = start_pos.copy()
        self._ref_pos = start_pos.copy()
        self._trajectory_time = 0.0
        self._total_distance = 0.0
        
        return self._trajectory_type, self._ref_pos.copy()

    def _get_lookahead_points(self) -> np.ndarray:
        """Get N future path points."""
        points = []
        dt = self._lookahead_dt
        
        for i in range(self._n_lookahead):
            t = self._trajectory_time + i * dt
            pos = self._compute_path_point(t)
            points.append(pos)
        
        return np.concatenate(points)  # 15 elements

    def _compute_path_point(self, t: float) -> np.ndarray:
        """Compute path position at time t."""
        start = self._episode_start
        R = self._trajectory_radius
        v = self._trajectory_speed
        
        if self._trajectory_type == "line":
            return start + np.array([v * t, 0, 0])
        
        elif self._trajectory_type == "oval":
            angle = v * t / R
            return start + np.array([R * np.cos(angle), 0.5 * R * np.sin(angle), 0])
        
        elif self._trajectory_type == "circle":
            angle = v * t / R
            return start + np.array([R * np.cos(angle), R * np.sin(angle), 0])
        
        elif self._trajectory_type == "figure8":
            return start + np.array([R * np.sin(v * t), R * np.sin(v * t) * np.cos(v * t), 0])
        
        elif self._trajectory_type == "spline":
            t_mod = t % 3.0
            if t_mod < 1.0:
                return start + np.array([t_mod * R, 0, 0])
            elif t_mod < 2.0:
                return start + np.array([R, (t_mod - 1) * R, 0])
            else:
                return start + np.array([R - (t_mod - 2) * R, R, 0])
        
        return start.copy()

    def _update_reference(self, dt: float = 0.01):
        self._trajectory_time += dt
        
        new_pos = self._compute_path_point(self._trajectory_time)
        delta = new_pos - self._ref_pos
        self._total_distance += np.linalg.norm(delta)
        self._ref_pos = new_pos.copy()

    def _get_obs(self) -> Dict[str, np.ndarray]:
        pos = self.s10.data.qpos[:3].copy()
        vel = self.s10.data.qvel[:3].copy()
        quat = self.s10.data.qpos[3:7].copy()
        rpy = self.s10._quat_to_rpy(quat)
        
        lookahead = self._get_lookahead_points()
        
        # Heading error: angle between velocity and path tangent
        heading_error = self._compute_heading_error()
        
        # Speed ratio: current speed / reference speed
        speed = np.linalg.norm(vel)
        speed_ratio = speed / (self._trajectory_speed + 1e-8)
        
        obs = np.concatenate([
            pos,
            vel,
            rpy,
            lookahead,
            self._prev_action,
            [self._total_distance],
            [heading_error],
            [speed_ratio],
        ]).astype(np.float32)
        obs = np.clip(obs, -50.0, 50.0)
        
        return {
            "observation": obs,
            "achieved_goal": pos,
            "desired_goal": self._ref_pos.copy(),
        }
    
    def _compute_heading_error(self) -> float:
        """Compute heading error between velocity and path direction."""
        vel = self.s10.data.qvel[:3]
        speed = np.linalg.norm(vel)
        if speed < 0.1:
            return 0.0
        
        # Path direction at current point
        path_dir = self._ref_pos - self._compute_path_point(self._trajectory_time - 0.05)
        if np.linalg.norm(path_dir) > 0:
            path_dir = path_dir / np.linalg.norm(path_dir)
            heading_error = 1.0 - np.dot(path_dir, vel / speed)
            return np.clip(heading_error, -1.0, 1.0)
        return 0.0

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict
    ) -> float:
        pos = achieved_goal
        
        # Cross-track error (distance from path)
        cte = info.get("cross_track_error", np.linalg.norm(desired_goal - pos))
        
        # Progress reward
        progress = info.get("progress", 0.0)
        
        # Heading alignment
        heading = info.get("heading_error", 0.0)
        
        # Reward components
        r_cte = -2.0 * cte
        r_progress = 10.0 * progress
        r_heading = -1.0 * heading
        
        # Success bonus
        r_success = 50.0 if cte < 0.3 else 0.0
        
        # Crash penalty
        r_crash = -100.0 if pos[2] < 0.1 else 0.0
        
        return r_cte + r_progress + r_heading + r_success + r_crash

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
        self._success_count = 0
        self._smoothed_action = np.zeros(4, dtype=np.float32)
        
        # Track current episode CTE for curriculum gating
        self._current_episode_ctes = []
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
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
        vel = self.s10.data.qvel[:3]
        
        cte = np.linalg.norm(self._ref_pos - pos)
        speed = np.linalg.norm(vel)
        
        if speed > 0:
            progress = np.dot(vel, (self._ref_pos - pos)) / (speed + 1e-8)
        else:
            progress = 0.0
        
        heading = 0.0
        if speed > 0.1:
            path_dir = self._ref_pos - self._compute_path_point(self._trajectory_time - 0.01)
            if np.linalg.norm(path_dir) > 0:
                path_dir = path_dir / np.linalg.norm(path_dir)
                heading = 1.0 - np.dot(path_dir, vel / speed)
        
        if cte < 0.3:
            self._success_count += 1
        
        if cte > 2.0:
            terminated = True
        
        # Track CTE for curriculum gating
        self._current_episode_ctes.append(cte)
        
        # Episode ended - record mean CTE
        if terminated or truncated:
            mean_cte = np.mean(self._current_episode_ctes) if self._current_episode_ctes else 1.0
            self._record_episode_cte(mean_cte)
        
        self._step_count += 1
        if self._step_count >= self.max_episode_steps:
            truncated = True
        
        self._prev_action = action.copy()
        
        info = {
            "cross_track_error": cte,
            "progress": progress,
            "heading_error": heading,
            "success_count": self._success_count,
        }
        
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = self.s10.viewer
        return self.s10.render()