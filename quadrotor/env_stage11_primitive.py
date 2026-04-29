#!/usr/bin/env python3
"""
Stage 11 V2: Body-Frame Primitive Architecture

Key insight: The RC-stick action space has no semantic alignment with path tracking.
The policy learns a position→RC mapping which is brittle and non-intuitive.

Body-frame primitives instead:
- forward_cmd: target forward velocity (m/s in body frame)
- turn_rate: yaw rate command (rad/s in body frame)  
- altitude_delta: target altitude change (m)
- duration: how many steps to execute (primitives are time-extended)

This architecture:
1. Generalizes better (semantic action space)
2. Is more robust to position drift (relative observations)
3. Has semantic alignment with the task
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


class PrimitiveExecutor:
    """Executes body-frame primitives by converting to RC inputs."""
    
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        self.max_roll = np.deg2rad(35)
        self.max_pitch = np.deg2rad(35)
        self.max_yaw_rate = np.deg2rad(180)
        self.max_thrust_ratio = 1.0
        self.min_thrust_ratio = 0.3
        
    def primitive_to_rc(
        self, 
        forward_cmd: float,  # m/s desired body-frame velocity
        turn_rate: float,   # rad/s yaw rate
        altitude_target: float,  # m target altitude
        current_altitude: float,   # m current altitude
        current_yaw: float = 0.0,
    ) -> np.ndarray:
        """Convert body-frame primitive to RC input [roll, pitch, yaw, thrust].
        
        Semantics:
        - forward_cmd: desired forward velocity (m/s). Converted to pitch angle:
            pitch ≈ arctan(forward_cmd / v_hover) where v_hover ≈ 3 m/s
        - turn_rate: yaw rate command (rad/s)
        - altitude_target: target altitude (m). P-controller to thrust.
        """
        # Forward velocity → pitch angle
        # At hover thrust ≈ 0.5, so v_hover ≈ 3 m/s
        # pitch angle needed to produce forward velocity:
        v_hover = 3.0  # m/s
        pitch_cmd = np.arctan(forward_cmd / (v_hover + 1e-6))
        pitch_cmd = np.clip(pitch_cmd, -self.max_pitch, self.max_pitch)
        
        # Turn rate → yaw command (rate mode)
        yaw_cmd = np.clip(turn_rate / self.max_yaw_rate, -1.0, 1.0)
        
        # Coordinated turn: roll into the turn
        roll_cmd = np.clip(-turn_rate * 0.3 / self.max_yaw_rate, -1.0, 1.0)
        
        # Altitude target → thrust P-controller
        alt_error = altitude_target - current_altitude
        # P-gain: 0.2 thrust per meter error
        thrust_cmd = 0.5 + np.clip(alt_error * 0.2, -0.2, 0.2)
        thrust_cmd = np.clip(thrust_cmd, self.min_thrust_ratio, self.max_thrust_ratio)
        
        return np.array([roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd], dtype=np.float32)
    
    def execute_primitive(
        self,
        forward_cmd: float,
        turn_rate: float,
        altitude_delta: float,
        env: 'QuadrotorEnvStage11Primitive',
    ) -> np.ndarray:
        """Execute one step of primitive."""
        return self.primitive_to_rc(
            forward_cmd,
            turn_rate,
            altitude_delta,
            env.s10.data.qpos[2],
            env._current_yaw,
        )


class QuadrotorEnvStage11Primitive(gym.Env):
    """Body-frame primitive tracking environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        stage10_model_path: str = "__PLACEHOLDER__",
        stage10_vecnormalize_path: Optional[str] = None,
        max_episode_steps: int = 5000,
        render_mode: Optional[str] = None,
        primitive_duration: int = 20,  # steps per primitive
    ):
        super().__init__()
        
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.primitive_duration = primitive_duration
        
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
        
        # Load VecNormalize stats
        self.s10_obs_mean = None
        self.s10_obs_std = None
        if stage10_vecnormalize_path and os.path.exists(stage10_vecnormalize_path):
            self._load_vecnormalize_stats(stage10_vecnormalize_path)
        
        # Primitive executor
        self.executor = PrimitiveExecutor()
        
        # Spaces: body-frame primitives
        # forward_cmd: [-1, 1] m/s
        # turn_rate: [-pi, pi] rad/s
        # altitude_delta: [-2, 2] m
        # duration: [1, 50] steps (discretized)
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=-10.0, high=10.0, shape=(16,), dtype=np.float32),
            "achieved_goal": spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
            "desired_goal": spaces.Box(low=-50.0, high=50.0, shape=(3,), dtype=np.float32),
        })
        
        # Primitive action space
        self.action_space = spaces.Box(
            low=np.array([-1.0, -np.pi, -2.0, 1], dtype=np.float32),
            high=np.array([1.0, np.pi, 2.0, 50], dtype=np.float32),
            dtype=np.float32
        )
        
        # Tracking state
        self._step_count = 0
        self._primitive_step = 0
        self._current_primitive = None
        self._goal_pos = np.zeros(3, dtype=np.float32)
        self._episode_start = np.zeros(3)
        self._current_yaw = 0.0
        self._prev_primitive = np.zeros(4, dtype=np.float32)
        self._success_count = 0
        self._altitude_target = 1.0
        
        # Internal state for execution
        self._rc_input = np.zeros(4, dtype=np.float32)
        
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
    
    def _quat_to_rpy(self, quat: np.ndarray) -> np.ndarray:
        """Convert quaternion to roll-pitch-yaw."""
        w, x, y, z = quat
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw], dtype=np.float32)
    
    def _world_to_body(self, world_pos: np.ndarray, yaw: float) -> np.ndarray:
        """Transform world frame to body frame."""
        dx = world_pos[0]
        dy = world_pos[1]
        
        # Rotate by -yaw
        body_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
        body_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)
        
        return np.array([body_x, body_y, world_pos[2]], dtype=np.float32)
    
    def _body_to_world(self, body_pos: np.ndarray, yaw: float) -> np.ndarray:
        """Transform body frame to world frame."""
        bx = body_pos[0]
        by = body_pos[1]
        
        # Rotate by +yaw
        world_x = bx * np.cos(yaw) - by * np.sin(yaw)
        world_y = bx * np.sin(yaw) + by * np.cos(yaw)
        
        return np.array([world_x, world_y, body_pos[2]], dtype=np.float32)
    
    def _generate_goal(self) -> np.ndarray:
        """Generate a new goal position."""
        types = ["circle", "figure8", "line", "random"]
        goal_type = self.np_random.choice(types)
        
        start_pos = self.s10.data.qpos[:3].copy()
        self._episode_start = start_pos.copy()
        
        if goal_type == "circle":
            angle = self.np_random.uniform(0, 2*np.pi)
            radius = self.np_random.uniform(1.0, 3.0)
            goal = start_pos + np.array([
                radius * np.cos(angle),
                radius * np.sin(angle),
                0
            ])
        elif goal_type == "figure8":
            t = self.np_random.uniform(0, 2*np.pi)
            goal = start_pos + np.array([
                2.0 * np.sin(t),
                2.0 * np.sin(t) * np.cos(t),
                0
            ])
        elif goal_type == "line":
            direction = self.np_random.uniform(-1, 1, size=2)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            goal = start_pos + np.concatenate([direction * 3.0, [0]])
        else:  # random
            goal = start_pos + self.np_random.uniform(-3, 3, size=3)
            goal[2] = max(0.5, goal[2])
        
        return goal.astype(np.float32)
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Body-frame relative observation."""
        pos = self.s10.data.qpos[:3].copy()
        vel = self.s10.data.qvel[:3].copy()
        quat = self.s10.data.qpos[3:7].copy()
        rpy = self._quat_to_rpy(quat)
        
        self._current_yaw = rpy[2]
        
        # World-frame state to body-frame
        rel_pos = self._world_to_body(self._goal_pos - pos, self._current_yaw)
        rel_vel = self._world_to_body(vel, self._current_yaw)
        
        # Current altitude
        altitude = pos[2]
        
        # Speed
        speed = np.linalg.norm(vel)
        
        # Previous primitive (for continuity)
        prev_primitive = self._prev_primitive.copy()
        
        # Time remaining in current primitive
        time_remaining = max(0, self.primitive_duration - self._primitive_step)
        
        obs = np.concatenate([
            rel_pos,          # 3: body-relative distance to goal
            rel_vel,          # 3: body-relative velocity
            [altitude],      # 1: current altitude
            [speed],          # 1: total speed
            [time_remaining], # 1: steps remaining in primitive
            prev_primitive,  # 4: previous primitive command
            rpy,              # 3: current attitude
        ]).astype(np.float32)
        obs = np.clip(obs, -10.0, 10.0)
        
        return {
            "observation": obs,
            "achieved_goal": pos,
            "desired_goal": self._goal_pos.copy(),
        }

    def compute_reward(
        self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict
    ) -> float:
        track_error = np.linalg.norm(desired_goal - achieved_goal)
        
        r_tracking = -2.0 * track_error
        r_success = 50.0 if track_error < 0.3 else 0.0
        
        pos = achieved_goal
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
        self._primitive_step = 0
        self._current_primitive = None
        self._goal_pos = self._generate_goal()
        self._current_yaw = 0.0
        self._prev_primitive = np.zeros(4, dtype=np.float32)
        self._success_count = 0
        self._rc_input = np.zeros(4, dtype=np.float32)
        
        start_alt = s10_obs[2]
        self._altitude_target = start_alt
        self._current_primitive = np.array([0.0, 0.0, start_alt, 10], dtype=np.float32)
        
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        action = action.copy()
        action[3] = int(np.clip(action[3], 1, self.primitive_duration))
        
        if self._primitive_step == 0:
            self._current_primitive = action.copy()
            self._altitude_target = action[2]
        
        forward_cmd, turn_rate, altitude_target, duration = self._current_primitive
        self._altitude_target = self._current_primitive[2]
        
        self._rc_input = self.executor.primitive_to_rc(
            forward_cmd,
            turn_rate,
            self._altitude_target,
            self.s10.data.qpos[2],
            self._current_yaw,
        )
        
        # Apply to Stage 10
        self.s10.rc_input = self._rc_input.copy()
        s10_obs = self.s10._get_obs()
        s10_obs_norm = self._normalize_s10_obs(s10_obs)
        
        s10_action, _ = self.s10_model.predict(s10_obs_norm, deterministic=False)
        if s10_action.ndim > 1:
            s10_action = s10_action[0]
        
        _, _, terminated, truncated, _ = self.s10.step(s10_action)
        
        # Update primitive execution
        self._primitive_step += 1
        self._step_count += 1
        
        # Check if primitive done
        if self._primitive_step >= int(duration):
            self._primitive_step = 0
            self._prev_primitive = self._current_primitive.copy()
        
        # Check goal distance
        pos = self.s10.data.qpos[:3].copy()
        track_error = np.linalg.norm(self._goal_pos - pos)
        
        if track_error < 0.3:
            self._success_count += 1
        
        if track_error > 5.0:
            terminated = True
        
        if self._step_count >= self.max_episode_steps:
            truncated = True
        
        info = {
            "tracking_error": track_error,
            "goal_pos": self._goal_pos.copy(),
            "success_count": self._success_count,
            "primitive_step": self._primitive_step,
        }
        
        obs = self._get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human" and self.viewer is None:
            self.viewer = self.s10.viewer
        return self.s10.render()


if __name__ == "__main__":
    env = QuadrotorEnvStage11Primitive(
        stage10_model_path="__PLACEHOLDER__",
        max_episode_steps=1000,
    )
    
    print("Stage 11 Primitive environment loaded")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")