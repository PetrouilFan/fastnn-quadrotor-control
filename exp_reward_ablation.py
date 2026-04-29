#!/usr/bin/env python3
"""
Reward Function Ablation Study

Tests which reward components are critical for learning good control policies.
Each ablation removes one reward term and measures the impact on performance.

Reward components:
- r_alive: 1.0 (time penalty)
- r_pos: -2.0 * tanh(pos_err)
- r_att: -1.5 * tanh(att_err)
- r_vel: -0.05 * tanh(vel_err)
- r_rate: -0.05 * tanh(rate_err)
- r_smooth: -0.01 * tanh(act_delta)
- r_success: 5.0 (sparse success bonus)
- r_proximity: 0.5 (proximity reward)
- r_alignment: 0.3 (alignment reward)
- r_recovery: 2.0 (post-drop recovery, stage 4 only)
- r_jerk: -0.05 * tanh(angular_accel/50.0)

Usage:
    python exp_reward_ablation.py --steps 500000 --stage 3
"""

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from tqdm import tqdm
import os
import argparse
import json
from typing import Dict, Tuple

from env_rma import RMAQuadrotorEnv


# Standard reward configuration
class StandardReward:
    """Standard reward function from RMAQuadrotorEnv."""
    def __call__(self, env, action):
        return env._compute_reward(action)


class NoProximityReward:
    """Ablation: Remove proximity reward."""
    def __call__(self, env, action):
        r = env._compute_reward(action)
        # Remove r_proximity = 0.5 and r_alignment = 0.3
        pos = env.data.qpos[:3]
        quat = env.data.qpos[3:7]
        rpy = env._quat_to_rpy(quat)
        pos_err = np.linalg.norm(env.target_pos - pos)
        att_err = np.linalg.norm(rpy[:2])
        r_prox = 0.5 if pos_err < 0.2 else 0.0
        r_align = 0.3 if pos_err < 0.3 and att_err < 0.1 else 0.0
        return r - r_prox - r_align


class NoRecoveryBonus:
    """Ablation: Remove recovery bonus."""
    def __call__(self, env, action):
        r = env._compute_reward(action)
        return r - 2.0  # Remove r_recovery


class NoSmoothnessPenalty:
    """Ablation: Remove action smoothness penalty."""
    def __call__(self, env, action):
        r = env._compute_reward(action)
        act_delta = np.linalg.norm(action - env.prev_action)
        r_smooth = -0.01 * np.tanh(act_delta)
        return r - r_smooth


class PositionOnlyReward:
    """Ablation: Only position reward + success (minimal reward)."""
    def __call__(self, env, action):
        pos = env.data.qpos[:3]
        vel = env.data.qvel[:3]
        quat = env.data.qpos[3:7]
        rpy = env._quat_to_rpy(quat)
        body_rates = env.data.qvel[3:6]

        pos_err = np.linalg.norm(env.target_pos - pos)
        att_err = np.linalg.norm(rpy[:2])

        r_alive = 1.0
        r_pos = -2.0 * np.tanh(pos_err)
        r_success = 5.0 if pos_err < 0.1 and att_err < 0.1 else 0.0

        return r_alive + r_pos + r_success


REWARD_CONFIGS = {
    'standard': StandardReward(),
    'no_proximity': NoProximityReward(),
    'no_recovery': NoRecoveryBonus(),
    'no_smoothness': NoSmoothnessPenalty(),
    'position_only': PositionOnlyReward(),
}


class RewardAblationEnv(RMAQuadrotorEnv):
    """Environment wrapper that uses custom reward function."""

    def __init__(self, *args, reward_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_reward_fn = reward_fn or StandardReward()

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        reward = self.custom_reward_fn(self, action)
        return obs, reward, terminated, truncated, info


def make_env(stage, seed, rank, reward_fn_key='standard'):
    def _init():
        reward_fn = REWARD_CONFIGS[reward_fn_key]
        env = RewardAblationEnv(curriculum_stage=stage, use_direct_control=True, reward_fn=reward_fn)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_and_eval(stage, config_name, total_steps, seed):
    """Train with a specific reward config and evaluate."""
    print(f"\n{'='*50}")
    print(f"Training with reward config: {config_name}")
    print(f"{'='*50}")

    model_dir = f"models_reward_ablation/{config_name}/stage_{stage}/seed_{seed}"
    os.makedirs(model_dir, exist_ok=True)

    # Create training environment
    env_fns = [make_env(stage=stage, seed=seed, rank=i, reward_fn_key=config_name) for i in range(16)]
    train_env = SubprocVecEnv(env_fns)

    # Create eval environment
    eval_env = RewardAblationEnv(
        curriculum_stage=stage,
        use_direct_control=True,
        reward_fn=REWARD_CONFIGS[config_name]
    )
    eval_env.reset(seed=seed)

    # Train SAC
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        policy_kwargs=dict(
            net_arch=[256, 256],
        ),
        tensorboard_log=f"tb_logs_reward_ablation/{config_name}/stage_{stage}/seed_{seed}/",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        verbose=0,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix=f"{config_name}_sac",
    )

    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=50,
        eval_freq=10000,
        deterministic=True,
        render=False,
    )

    print(f"Training for {total_steps} steps...")
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_callback],
        progress_bar=False,
    )

    # Final evaluation
    successes = 0
    final_dists = []
    for ep in range(100):
        obs, _ = eval_env.reset()
        done = False
        steps = 0
        dist = np.linalg.norm(eval_env.target_pos - eval_env.data.qpos[:3])
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = eval_env.step(action)
            steps += 1
            dist = np.linalg.norm(eval_env.target_pos - eval_env.data.qpos[:3])
        if steps >= 500:
            successes += 1
            final_dists.append(dist)
        else:
            final_dists.append(dist)

    success_rate = successes / 100
    mean_final_dist = np.mean(final_dists)

    print(f"Results: success={success_rate:.0%}, mean_dist={mean_final_dist:.3f}m")

    model.save(os.path.join(model_dir, "final"))

    train_env.close()
    eval_env.close()

    return {
        'success_rate': success_rate,
        'mean_final_dist': mean_final_dist,
        'model_dir': model_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Reward Function Ablation Study")
    parser.add_argument("--stage", type=int, default=3, help="Curriculum stage")
    parser.add_argument("--steps", type=int, default=500_000, help="Training steps")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated seeds")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    results = {}

    for config_name in REWARD_CONFIGS.keys():
        results[config_name] = {}
        for seed in seeds:
            results[config_name][seed] = train_and_eval(
                stage=args.stage,
                config_name=config_name,
                total_steps=args.steps,
                seed=seed,
            )

    # Save results
    os.makedirs("results_reward_ablation", exist_ok=True)
    with open(f"results_reward_ablation/stage_{args.stage}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Reward Function Ablation Results")
    print("="*60)
    print(f"{'Config':<20} {'Success':<12} {'Mean Dist':<12}")
    print("-" * 60)
    for config_name, config_results in results.items():
        avg_success = np.mean([r['success_rate'] for r in config_results.values()])
        avg_dist = np.mean([r['mean_final_dist'] for r in config_results.values()])
        print(f"{config_name:<20} {avg_success:>9.0%}  {avg_dist:>10.3f}m")


if __name__ == "__main__":
    main()