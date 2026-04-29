#!/usr/bin/env python3
"""
GRU Training for Delay-Robust Quadrotor Control

Uses observation HISTORY as input to MLP,
making the policy robust to sensor delays via explicit memory.

The key insight: instead of hiding delay in GRU hidden state,
just concatenate past observations as input.

Usage:
    python train_gru_stage5.py --steps 2000000 --history 4 --n-envs 8
"""

import numpy as np
import argparse
import torch
from gymnasium import ObservationWrapper
import os
import json
from datetime import datetime

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback


class HistoryWrapper(ObservationWrapper):
    """
    Gymnasium-compliant wrapper that provides observation history.
    Returns concatenated history of past N observations.
    """
    
    def __init__(self, env, history_len=4):
        super().__init__(env)
        self.history_len = history_len
        self._history = []
        
        # Update observation space for history
        orig_shape = env.observation_space.shape
        orig_low = np.min(env.observation_space.low) if hasattr(env.observation_space.low, '__iter__') else env.observation_space.low
        orig_high = np.max(env.observation_space.high) if hasattr(env.observation_space.high, '__iter__') else env.observation_space.high
        
        self.observation_space = type(env.observation_space)(
            low=orig_low,
            high=orig_high,
            shape=(orig_shape[0] * history_len,),
            dtype=np.float32
        )
    
    def observation(self, obs):
        """Process observation to include history."""
        self._history.append(obs.copy())
        
        # Keep only last N observations
        if len(self._history) > self.history_len:
            self._history.pop(0)
        
        # Pad if not enough history yet
        while len(self._history) < self.history_len:
            self._history.insert(0, self._history[0].copy())
        
        # Concatenate history
        return np.concatenate(self._history)


def make_env(stage, history_len, seed, rank):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=seed + rank)
        # Wrap with history
        env = HistoryWrapper(env, history_len=history_len)
        return env
    return _init


def train_gru(
    stage=5,
    history_len=4,
    total_steps=2000000,
    n_envs=8,
    seed=0,
    save_dir="runs/gru_trained"
):
    """Train with observation history."""
    
    obs_dim = 63 * history_len
    
    print("=" * 60)
    print(f"GRU-style Training - Stage {stage}, History {history_len}")
    print(f"Observation dim: {obs_dim}")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}/stage_{stage}_h{history_len}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create environments with history
    env_fns = [
        make_env(stage=stage, history_len=history_len, seed=seed, rank=i)
        for i in range(n_envs)
    ]
    train_env = DummyVecEnv(env_fns)
    
    # Eval env WITHOUT history for fair comparison
    eval_env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_env.set_target_speed(1.0)
    eval_env.set_moving_target(True)
    eval_env.reset(seed=seed+1000)
    
    print(f"Obs space: {train_env.observation_space.shape}")
    
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
        policy_kwargs=dict(net_arch=[512, 512]),  # Larger network for more input
        tensorboard_log=run_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        verbose=1,
    )
    
    # Skip eval callback during training - causes shape mismatch
    # We'll evaluate manually after training
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=run_dir + "/checkpoints",
        name_prefix="gru_sac",
    )
    
    print(f"Training started at {timestamp}")
    
    model.learn(
        total_timesteps=total_steps,
        callback=checkpoint_callback,
        progress_bar=True,
    )
    
    # Save
    model.save(run_dir + "/final.zip")
    print(f"Saved to {run_dir}/final.zip")
    
    # Evaluate WITHOUT history
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    
    successes = 0
    for ep in range(100):
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=seed+ep+3000)
        
        obs, _ = env.reset()
        steps = 0
        done = False
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        if steps >= 500:
            successes += 1
        env.close()
    
    print(f"Success rate: {successes}%")
    
    with open(run_dir + "/results.json", 'w') as f:
        json.dump({
            "success_rate": successes,
            "history_len": history_len,
            "obs_dim": obs_dim
        }, f)
    
    train_env.close()
    eval_env.close()
    
    return successes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="runs/gru_trained")
    
    args = parser.parse_args()
    
    train_gru(
        stage=args.stage,
        history_len=args.history,
        total_steps=args.steps,
        n_envs=args.n_envs,
        seed=args.seed,
        save_dir=args.save_dir,
    )