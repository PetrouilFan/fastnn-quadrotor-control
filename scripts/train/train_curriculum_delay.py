#!/usr/bin/env python3
"""Training with curriculum delay: starts at 0ms, increases to max over training."""

import numpy as np
import argparse
import torch
from gymnasium import ObservationWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
from datetime import datetime

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class CurriculumDelayWrapper(ObservationWrapper):
    """Wrapper with gradually increasing delay over training steps."""
    
    def __init__(self, env, max_delay_steps=10, curriculum_steps=1000000):
        super().__init__(env)
        self.max_delay_steps = max_delay_steps
        self.curriculum_steps = curriculum_steps
        self._obs_buffer = []
        self._current_delay = 0
        self._total_steps = 0
    
    def set_total_steps(self, steps):
        """Update current training progress."""
        self._total_steps = steps
        progress = min(1.0, steps / self.curriculum_steps)
        self._current_delay = int(progress * self.max_delay_steps)
    
    def observation(self, obs):
        self._obs_buffer.append(obs.copy())
        if len(self._obs_buffer) > self.max_delay_steps + 1:
            self._obs_buffer.pop(0)
        if len(self._obs_buffer) > self._current_delay and self._current_delay > 0:
            return self._obs_buffer[-self._current_delay - 1].copy()
        return obs.copy()
    
    def reset(self, seed=None, options=None):
        self._obs_buffer = []
        self._current_delay = 0
        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info
    
    def get_delay(self):
        return self._current_delay


class CurriculumTrainer:
    """Tracks curriculum progress and updates wrappers."""
    
    def __init__(self, max_delay, curriculum_steps):
        self.max_delay = max_delay
        self.curriculum_steps = curriculum_steps
        self.wrappers = []
    
    def register(self, wrapper):
        self.wrappers.append(wrapper)
    
    def update(self, total_steps):
        for w in self.wrappers:
            if hasattr(w, 'set_total_steps'):
                w.set_total_steps(total_steps)


def make_env(stage, max_delay, curriculum_steps, seed, rank, trainer):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=seed + rank)
        
        wrapper = CurriculumDelayWrapper(env, max_delay_steps=max_delay, curriculum_steps=curriculum_steps)
        trainer.register(wrapper)
        
        return wrapper
    return _init


def train_curriculum_delay(stage=5, max_delay=10, curriculum_steps=1500000, total_steps=2000000, n_envs=8, seed=0, save_dir="runs/curriculum_delay"):
    print(f"Curriculum Delay Training: 0ms -> {max_delay*10}ms over {curriculum_steps} steps")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}/stage_{stage}_curriculum_{max_delay*10}ms_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    trainer = CurriculumTrainer(max_delay, curriculum_steps)
    
    env_fns = [make_env(stage, max_delay, curriculum_steps, seed, i, trainer) for i in range(n_envs)]
    train_env = DummyVecEnv(env_fns)
    
    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4, buffer_size=100000, learning_starts=1000,
        batch_size=256, tau=0.005, gamma=0.99, ent_coef='auto',
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed, verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=run_dir + "/checkpoints", name_prefix="sac")
    
    print(f"Training started at {timestamp}")
    
    current_steps = 0
    update_interval = 10000
    
    while current_steps < total_steps:
        model.learn(total_timesteps=min(update_interval, total_steps - current_steps), reset_num_timesteps=False, progress_bar=False)
        current_steps += update_interval
        trainer.update(current_steps)
        
        if current_steps % 100000 == 0:
            print(f"Steps: {current_steps}/{total_steps}, max_delay: {trainer.max_delay * 10}ms")
    
    model.save(run_dir + "/final.zip")
    print(f"Saved to {run_dir}/final.zip")
    
    with open(run_dir + "/config.json", 'w') as f:
        import json
        json.dump({"max_delay": max_delay, "curriculum_steps": curriculum_steps}, f)
    
    train_env.close()
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--max-delay", type=int, default=10)
    parser.add_argument("--curriculum-steps", type=int, default=1500000)
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-dir", type=str, default="runs/curriculum_delay")
    args = parser.parse_args()
    
    train_curriculum_delay(stage=args.stage, max_delay=args.max_delay, curriculum_steps=args.curriculum_steps, total_steps=args.steps, n_envs=args.n_envs, seed=args.seed, save_dir=args.save_dir)