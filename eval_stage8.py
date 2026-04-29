#!/usr/bin/env python3
"""Quick evaluation of Stage 8 (Extended Racing) at multiple speeds."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env_rma import RMAQuadrotorEnv
from stable_baselines3 import SAC


def evaluate_stage8(model_path, speeds=[1.0, 2.0, 5.0], n_episodes=20, max_steps=500):
    """Evaluate Stage 8 model at multiple speeds."""
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path)
    model_obs_size = model.observation_space.shape[0]
    print(f"Model expects obs size: {model_obs_size}")

    for speed in speeds:
        env = RMAQuadrotorEnv(curriculum_stage=8, use_direct_control=True)

        successes = 0
        tracking_errors = []
        xy_errors = []
        z_errors = []

        for ep in range(n_episodes):
            obs, _ = env.reset(seed=42 + ep)
            # Set speed AFTER reset since Stage 8 reset sets _target_speed=0.5
            env.set_target_speed(speed)
            # Trim observation to match model's expected size
            if obs.shape[0] > model_obs_size:
                obs = obs[:model_obs_size]

            # Trim observation to match model's expected size
            if obs.shape[0] > model_obs_size:
                obs = obs[:model_obs_size]

            while not done and steps < max_steps:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if obs.shape[0] > model_obs_size:
                    obs = obs[:model_obs_size]
                done = terminated or truncated

                pos = env.data.qpos[:3]
                target = env.target_pos
                error = np.linalg.norm(pos - target)
                ep_errors.append(error)
                steps += 1

            if steps >= max_steps:
                successes += 1
            tracking_errors.append(np.mean(ep_errors))

            # Compute XY and Z errors at final step
            pos = env.data.qpos[:3]
            target = env.target_pos
            xy_errors.append(np.linalg.norm(pos[:2] - target[:2]))
            z_errors.append(abs(pos[2] - target[2]))

        print(f"  Speed {speed:.1f}x: {successes}/{n_episodes} success, "
              f"track={np.mean(tracking_errors):.3f}m, "
              f"xy={np.mean(xy_errors):.3f}m, "
              f"z={np.mean(z_errors):.3f}m")
        env.close()


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models_stage8_extreme/stage_8_extended/seed_0/final"
    speeds = [float(s) for s in sys.argv[2:]] if len(sys.argv) > 2 else [1.0, 2.0, 5.0]
    evaluate_stage8(model_path, speeds)