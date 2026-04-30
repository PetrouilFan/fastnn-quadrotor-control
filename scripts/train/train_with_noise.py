#!/usr/bin/env python3
"""
Stage 5 Training with Observation Noise

Train SAC with observation perturbations to make the policy
robust to sensor noise and delays.

Usage:
    python train_with_noise.py --steps 1000000 --noise 0.1 --n-envs 8
"""

import numpy as np
import argparse
import torch
from gymnasium import ObservationWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import json
from datetime import datetime

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class NoisyObsWrapper(ObservationWrapper):
    """Gymnasium wrapper that adds noise to observations."""
    
    def __init__(self, env, noise_std=0.1):
        super().__init__(env)
        self.noise_std = noise_std
    
    def observation(self, obs):
        return obs + np.random.randn(*obs.shape) * self.noise_std


def make_env(stage, noise_std, seed, rank):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=seed + rank)
        if noise_std > 0:
            env = NoisyObsWrapper(env, noise_std=noise_std)
        return env
    return _init


def train_with_noise(
    stage=5,
    total_steps=2000000,
    n_envs=8,
    noise_std=0.1,
    seed=0,
    save_dir="runs/noise_trained"
):
    """Train with observation noise."""
    
    print("=" * 60)
    print(f"Noise Training - Stage {stage}, Noise std={noise_std}")
    print(f"Total steps: {total_steps}, N envs: {n_envs}")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}/stage_{stage}_noise{noise_std}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    env_fns = [
        make_env(stage=stage, noise_std=noise_std, seed=seed, rank=i)
        for i in range(n_envs)
    ]
    train_env = DummyVecEnv(env_fns)
    
    eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_env.set_target_speed(1.0)
    eval_env.set_moving_target(True)
    eval_env.reset(seed=seed+1000)
    
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=run_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=run_dir + "/checkpoints",
        name_prefix="noise_sac",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir + "/best_model",
        log_path=run_dir + "/eval_logs",
        eval_freq=50000,
        n_eval_episodes=50,
        deterministic=True,
    )
    
    print(f"Training started at {timestamp}")
    print(f"Saving to: {run_dir}")
    
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )
    
    final_path = run_dir + "/final.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    
    # Final evaluation without noise
    print("\n" + "=" * 60)
    print("Final Evaluation (no noise)")
    print("=" * 60)
    
    eval_clean = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_clean.set_target_speed(1.0)
    eval_clean.set_moving_target(True)
    eval_clean.reset(seed=seed+2000)
    
    successes = 0
    for ep in range(100):
        obs, _ = eval_clean.reset()
        steps = 0
        done = False
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_clean.step(action)
            done = terminated or truncated
            steps += 1
        if steps >= 500:
            successes += 1
    
    print(f"Success rate (no noise): {successes}%")
    
    results = {
        "experiment": "noise_training",
        "stage": stage,
        "noise_std": noise_std,
        "total_steps": total_steps,
        "n_envs": n_envs,
        "success_rate": successes,
        "save_dir": run_dir,
    }
    
    with open(run_dir + "/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    train_env.close()
    eval_env.close()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="runs/noise_trained")
    
    args = parser.parse_args()
    
    train_with_noise(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        noise_std=args.noise,
        seed=args.seed,
        save_dir=args.save_dir,
    )