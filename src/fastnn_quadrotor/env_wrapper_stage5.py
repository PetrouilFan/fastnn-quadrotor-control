#!/usr/bin/env python3
"""
Environment wrapper for Stage 5 that removes mass_est from privileged observations.

This is for sim-to-real deployment - the actor learns without direct access
to mass_est, forcing it to infer mass changes from action history and error integrals.
"""

import numpy as np
from gymnasium import spaces
from gymnasium.core import Env


class NoMassEstEnvWrapper(Env):
    """
    Wraps RMAQuadrotorEnv to remove mass_est from observations.

    For Stage 5: Original obs is 63 dims (54 deployable + 9 privileged).
    We remove mass_est (last element of privileged), making it 62 dims.

    The underlying env still computes mass_est internally, but the wrapped
    env's observation excludes it.
    """

    metadata = {"render_modes": []}

    def __init__(self, env):
        self.env = env
        # Override observation space to be 62 dims (63 - 1 mass_est)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(62,), dtype=np.float32
        )
        self.action_space = env.action_space
        self._step_count = 0
        self.render_mode = None

    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self._step_count = 0
        # Remove last element (mass_est from privileged section)
        return obs[:-1].copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        # Remove last element (mass_est)
        return obs[:-1].copy(), reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

    # Expose underlying env properties
    @property
    def data(self):
        return self.env.data

    @property
    def target_pos(self):
        return self.env.target_pos

    @property
    def curriculum_stage(self):
        return self.env.curriculum_stage

    @property
    def payload_mass(self):
        return self.env.payload_mass

    @property
    def wind_force(self):
        return self.env.wind_force

    @property
    def model(self):
        return self.env.model

    def set_target_speed(self, speed):
        return self.env.set_target_speed(speed)

    def set_moving_target(self, enabled):
        return self.env.set_moving_target(enabled)

    def _cascaded_controller(self):
        return self.env._cascaded_controller()

    def get_privileged_info(self):
        return self.env.get_privileged_info()

    def _quat_to_rpy(self, quat):
        return self.env._quat_to_rpy(quat)
