#!/usr/bin/env python3
"""
Environment wrapper that removes mass_est from observations for the actor.

This enforces actor isolation from mass_est by:
1. Keeping full observation internally (60 dims)
2. Providing only deployable obs (51 dims) to the policy
3. Mass_est is still computed internally for potential future critic use

The actor learns without direct access to mass_est, forcing it to infer
mass changes from action history and error integrals.
"""

import numpy as np
from gymnasium import spaces
from gymnasium.core import Env


class NoMassEstEnvWrapper(Env):
    """
    Wraps RMAQuadrotorEnv to remove mass_est from observations.

    The underlying env still computes mass_est internally, but:
    - The wrapped env's observation space is 51 dims (not 60)
    - Observations returned don't include mass_est
    - This forces the policy to learn without mass_est signal
    """

    metadata = {"render_modes": []}

    def __init__(self, env):
        self.env = env
        # Override observation space to be 51 dims (no mass_est)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(51,), dtype=np.float32
        )
        self.action_space = env.action_space
        self._step_count = 0
        self.render_mode = None

    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self._step_count = 0
        # Return only deployable portion (first 51 dims)
        # obs[:51] = deployable without mass_est
        return obs[:51].copy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        # Return only deployable portion
        return obs[:51].copy(), reward, terminated, truncated, info

    def close(self):
        return self.env.close()

    def render(self):
        return self.env.render()

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

    def _cascaded_controller(self):
        return self.env._cascaded_controller()

    def get_privileged_info(self):
        return self.env.get_privileged_info()