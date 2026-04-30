#!/usr/bin/env python3
"""
Stage 5 Training with True Delay Injection

Train SAC with simulated sensor delay to make the policy
robust to observation delays.

Usage:
    python train_with_delay.py --steps 2000000 --delay 50 --n-envs 8
"""

import numpy as np
import argparse
import torch
from gymnasium import ObservationWrapper, spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import json
from datetime import datetime

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class DelayObservationWrapper(ObservationWrapper):
    """
    Gymnasium-compliant wrapper that delays observations.
    
    Keeps a buffer of past observations and returns delayed ones.
    """
    
    def __init__(self, env, delay_steps=5):
        super().__init__(env)
        self.delay_steps = delay_steps
        self._obs_buffer = []
    
    def observation(self, obs):
        """Return delayed observation."""
        self._obs_buffer.append(obs.copy())
        
        # Keep buffer at correct size
        if len(self._obs_buffer) > self.delay_steps:
            self._obs_buffer.pop(0)
        
        # Return oldest in buffer (or current if not full)
        if len(self._obs_buffer) > 0:
            return self._obs_buffer[0].copy()
        return obs.copy()
    
    def reset(self, seed=None, options=None):
        """Reset the environment and buffer."""
        obs, info = self.env.reset(seed=seed, options=options)
        # Initialize buffer with initial observation
        self._obs_buffer = [obs.copy() for _ in range(self.delay_steps)]
        return obs, info


def make_env(stage, delay_steps, seed, rank):
    """Create environment with optional delay."""
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=seed + rank)
        
        if delay_steps > 0:
            env = DelayObservationWrapper(env, delay_steps=delay_steps)
        
        return env
    return _init


def train_with_delay(
    stage=5,
    total_steps=2000000,
    n_envs=8,
    delay_steps=5,
    seed=0,
    save_dir="runs/delay_trained"
):
    """Train with observation delay."""
    
    delay_ms = delay_steps * 10
    
    print("=" * 60)
    print(f"Delay Training - Stage {stage}, Delay {delay_ms}ms ({delay_steps} steps)")
    print(f"Total steps: {total_steps}, N envs: {n_envs}")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}/stage_{stage}_delay{delay_ms}ms_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create environments with delay
    env_fns = [
        make_env(stage=stage, delay_steps=delay_steps, seed=seed, rank=i)
        for i in range(n_envs)
    ]
    train_env = DummyVecEnv(env_fns)
    
    # Eval environment WITHOUT delay (for fair comparison)
    eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_env.set_target_speed(1.0)
    eval_env.set_moving_target(True)
    eval_env.reset(seed=seed+1000)
    
    print(f"Creating SAC model...")
    print(f"Observation space: {train_env.observation_space}")
    print(f"Action space: {train_env.action_space}")
    
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
        name_prefix="delay_sac",
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
    
    # Final evaluation WITHOUT delay
    print("\n" + "=" * 60)
    print("Final Evaluation (NO delay)")
    print("=" * 60)
    
    eval_clean = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_clean.set_target_speed(1.0)
    eval_clean.set_moving_target(True)
    eval_clean.reset(seed=seed+2000)
    
    successes = 0
    total_errors = 0
    
    for ep in range(100):
        obs, _ = eval_clean.reset()
        steps = 0
        done = False
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_clean.step(action)
            done = terminated or truncated
            steps += 1
        
        if steps >= 500:
            successes += 1
        if 'tracking_error' in info:
            total_errors += info['tracking_error']
    
    success_rate = successes
    avg_error = total_errors / 100
    
    print(f"Success rate (no delay): {success_rate}%")
    print(f"Mean tracking error: {avg_error:.3f}m")
    
    # Eval WITH delay (same as training)
    print("\n" + "=" * 60)
    print(f"Final Evaluation (WITH {delay_ms}ms delay)")
    print("=" * 60)
    
    eval_delay = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_delay.set_target_speed(1.0)
    eval_delay.set_moving_target(True)
    eval_delay.reset(seed=seed+2000)
    eval_delay = DelayObservationWrapper(eval_delay, delay_steps=delay_steps)
    
    successes = 0
    total_errors = 0
    
    for ep in range(100):
        obs, _ = eval_delay.reset()
        steps = 0
        done = False
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_delay.step(action)
            done = terminated or truncated
            steps += 1
        
        if steps >= 500:
            successes += 1
        if 'tracking_error' in info:
            total_errors += info['tracking_error']
    
    success_rate_delay = successes
    avg_error_delay = total_errors / 100
    
    print(f"Success rate (with delay): {success_rate_delay}%")
    print(f"Mean tracking error: {avg_error_delay:.3f}m")
    
    results = {
        "experiment": "delay_training",
        "stage": stage,
        "delay_steps": delay_steps,
        "delay_ms": delay_ms,
        "total_steps": total_steps,
        "n_envs": n_envs,
        "seed": seed,
        "success_no_delay": success_rate,
        "success_with_delay": success_rate_delay,
        "mean_error_no_delay": avg_error,
        "mean_error_with_delay": avg_error_delay,
        "save_dir": run_dir,
    }
    
    results_path = run_dir + "/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    train_env.close()
    eval_env.close()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with delay injection")
    parser.add_argument("--stage", type=int, default=5, help="Curriculum stage")
    parser.add_argument("--steps", type=int, default=2000000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel envs")
    parser.add_argument("--delay", type=int, default=50, help="Delay in milliseconds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="runs/delay_trained", help="Save directory")
    
    args = parser.parse_args()
    
    delay_steps = args.delay // 10  # Convert ms to steps (assuming 100Hz)
    
    train_with_delay(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        delay_steps=delay_steps,
        seed=args.seed,
        save_dir=args.save_dir,
    )