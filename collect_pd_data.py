#!/usr/bin/env python3
"""
Stage 1: Data Collection for Imitation Learning

Collects (state, PD_action) pairs from Stage 1 where PD achieves 100% success.
This data will be used to pretrain the NN to mimic PD behavior before RL fine-tuning.

Usage:
    python collect_pd_data.py --episodes 1000 --save data/pd_stage1_buffer.npz
"""

import os
import argparse
import numpy as np
from tqdm import tqdm

from env_rma import RMAQuadrotorEnv

# 52-dim deployable obs (same as what NN actor sees)
DEPLOYABLE_DIM = 52


def collect_pd_data(n_episodes=1000, save_path="data/pd_stage1_buffer.npz", stage=1):
    """Collect state-PD action pairs from Stage 1 with diverse initial conditions."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Use different random seed per episode for diversity
    env = RMAQuadrotorEnv(curriculum_stage=stage)
    # Stage 1 normally has small perturbations, but we need DIVERSE data
    # Override the random pose generation to get more variation
    env._reset_fixed_hover_orig = env._reset_fixed_hover

    def _reset_diverse(self):
        """Stage 1 with LARGER perturbations for diverse training data."""
        self.data.qpos[:] = 0
        self.data.qpos[2] = 1.0  # z = 1m

        # LARGER perturbations (5x normal) for BC training diversity
        self.data.qpos[:3] += self.np_random.uniform(-0.25, 0.25, size=3)
        self.data.qpos[2] += self.np_random.uniform(-0.5, 0.5)
        self.data.qvel[:3] = self.np_random.uniform(-0.5, 0.5, size=3)
        self.data.qvel[3:6] = self.np_random.uniform(-1.0, 1.0, size=3)
        # Larger random tilt (up to 25 degrees, 5x normal 5 degrees)
        angle_rad = np.deg2rad(25.0)
        roll = self.np_random.uniform(-angle_rad, angle_rad)
        pitch = self.np_random.uniform(-angle_rad, angle_rad)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = 1.0, 0.0  # yaw = 0
        self.data.qpos[3:7] = [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]

    env._reset_fixed_hover = _reset_diverse.__get__(env, type(env))

    states = []
    actions = []  # PD control outputs (not residual)
    rewards = []

    print(f"Collecting data from Stage {stage} ({n_episodes} episodes) with DIVERSE initial conditions...")

    for ep in tqdm(range(n_episodes)):
        obs, _ = env.reset()
        ep_states = []
        ep_actions = []
        ep_rewards = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Get PD controller output directly
            pd_action = env._cascaded_controller()

            # Scale PD output to NN action space for compatibility
            # PD output: [thrust_z, roll_torque, pitch_torque, yaw_torque]
            # NN action: [thrust_adj, roll_adj, pitch_adj, yaw_adj] in [-1, 1]
            # We collect the raw PD output for imitation learning
            ep_states.append(obs[:DEPLOYABLE_DIM].copy())
            ep_actions.append(pd_action.copy())

            # Step with zero residual (pure PD)
            obs, reward, terminated, truncated, info = env.step(np.zeros(4))
            ep_rewards.append(reward)

        states.extend(ep_states)
        actions.extend(ep_actions)
        rewards.extend(ep_rewards)

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}: reward={np.mean(ep_rewards):.1f}, steps={len(ep_states)}")

    states = np.array(states, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32)
    rewards = np.array(rewards, dtype=np.float32)

    print(f"\nCollected {len(states)} transitions")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")
    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Action stats: thrust={actions[:,0].mean():.2f}±{actions[:,0].std():.2f}, "
          f"roll={actions[:,1].mean():.2f}±{actions[:,1].std():.2f}")

    np.savez(save_path, states=states, actions=actions, rewards=rewards)
    print(f"Saved to {save_path}")

    env.close()
    return states, actions, rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect PD data for imitation learning")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes to collect")
    parser.add_argument("--save", type=str, default="data/pd_stage1_buffer.npz", help="Save path")
    parser.add_argument("--stage", type=int, default=1, help="Curriculum stage (1=hover, 2=random pose)")
    args = parser.parse_args()

    collect_pd_data(n_episodes=args.episodes, save_path=args.save, stage=args.stage)