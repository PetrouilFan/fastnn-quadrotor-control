#!/usr/bin/env python3
"""
Delay Training for Temporal Robustness Testing

This script trains a policy with simulated sensor delays to test temporal
robustness. The delay is injected at the observation level.

Usage:
    python train_gru_delay.py --steps 500000 --n-envs 32 --delay-ms 50 --seed 0
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import argparse
import json
from datetime import datetime

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class DelayEnv:
    """Environment that introduces delay in observations."""
    
    def __init__(self, env, delay_steps=5):
        self.unwrapped = env
        self.delay_steps = delay_steps
        self._obs_buffer = []
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, 'metadata', {"render_modes": []})
        self.render_mode = None
    
    def reset(self, seed=None, **kwargs):
        obs, info = self.unwrapped.reset(seed=seed, **kwargs)
        self._obs_buffer = [obs.copy() for _ in range(self.delay_steps)]
        return self._obs_buffer[0].copy(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.unwrapped.step(action)
        self._obs_buffer.append(obs.copy())
        if len(self._obs_buffer) > self.delay_steps:
            self._obs_buffer.pop(0)
        delayed_obs = self._obs_buffer[0].copy()
        info['true_obs'] = obs.copy()
        info['delay_steps'] = self.delay_steps
        return delayed_obs, reward, terminated, truncated, info
    
    def close(self):
        return self.unwrapped.close()
    
    def render(self):
        return self.unwrapped.render()
    
    @property
    def env(self):
        return self.unwrapped
    
    @property
    def data(self):
        return self.unwrapped.data
    
    @property
    def target_pos(self):
        return self.unwrapped.target_pos
    
    @property
    def curriculum_stage(self):
        return self.unwrapped.curriculum_stage
    
    @property
    def payload_mass(self):
        return self.unwrapped.payload_mass
    
    @property
    def wind_force(self):
        return self.unwrapped.wind_force


def make_env(stage, delay_steps, seed, rank):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env = DelayEnv(env, delay_steps=delay_steps)
        env.reset(seed=seed + rank)
        return env
    return _init


def train_with_delay(
    stage=3,
    total_steps=500000,
    n_envs=32,
    delay_steps=5,
    seed=0,
    save_dir="runs/gru_delay"
):
    """Train with delay injection."""
    
    print("=" * 60)
    print(f"Delay Training - Stage {stage}, Delay {delay_steps} steps")
    print(f"Total steps: {total_steps}, N envs: {n_envs}")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}/stage_{stage}_delay{delay_steps}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    env_fns = [
        make_env(stage=stage, delay_steps=delay_steps, seed=seed, rank=i) 
        for i in range(n_envs)
    ]
    train_env = DummyVecEnv(env_fns)
    
    eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_env = DelayEnv(eval_env, delay_steps=delay_steps)
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
        save_freq=50000,
        save_path=run_dir + "/checkpoints",
        name_prefix="gru_delay",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir + "/best_model",
        log_path=run_dir + "/eval_logs",
        eval_freq=10000,
        n_eval_episodes=50,
        deterministic=True,
    )
    
    print(f"Training started at {timestamp}")
    print(f"Saving to: {run_dir}")
    
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_callback, eval_callback],
    )
    
    final_path = run_dir + "/final.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")
    
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    eval_env_single = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_env_single = DelayEnv(eval_env_single, delay_steps=delay_steps)
    eval_env_single.reset(seed=seed+2000)
    
    n_episodes = 100
    successes = 0
    total_episode_steps = 0
    
    for ep in range(n_episodes):
        obs, _ = eval_env_single.reset()
        episode_steps = 0
        done = False
        
        while not done and episode_steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_single.step(action)
            done = terminated or truncated
            episode_steps += 1
        
        if episode_steps >= 500:
            successes += 1
        total_episode_steps += episode_steps
    
    success_rate = successes / n_episodes * 100
    avg_steps = total_episode_steps / n_episodes
    
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Average steps: {avg_steps:.1f}")
    
    results = {
        "experiment": "gru_delay",
        "stage": stage,
        "delay_steps": delay_steps,
        "total_steps": total_steps,
        "n_envs": n_envs,
        "seed": seed,
        "success_rate": success_rate,
        "avg_steps": avg_steps,
        "save_dir": run_dir,
    }
    
    results_path = run_dir + "/results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")
    
    train_env.close()
    eval_env.close()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with delay")
    parser.add_argument("--stage", type=int, default=3, help="Curriculum stage")
    parser.add_argument("--steps", type=int, default=500000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel envs")
    parser.add_argument("--delay-ms", type=int, default=50, help="Delay in milliseconds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="runs/gru_delay", help="Save directory")
    
    args = parser.parse_args()
    
    delay_steps = args.delay_ms // 10
    
    train_with_delay(
        stage=args.stage,
        total_steps=args.steps,
        n_envs=args.n_envs,
        delay_steps=delay_steps,
        seed=args.seed,
        save_dir=args.save_dir,
    )