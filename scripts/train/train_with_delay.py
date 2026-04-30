#!/usr/bin/env python3
"""
Stage 5 Training with Random Delay

Train SAC with random sensor delay to make the policy
robust to observation delays.

Usage:
    python train_with_delay.py --steps 2000000 --delay-max 50 --n-envs 8
"""

import numpy as np
import argparse
from gymnasium import ObservationWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
import json
from datetime import datetime

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class DelayedObsWrapper(ObservationWrapper):
    """Wrapper that delays observations randomly."""
    
    def __init__(self, env, max_delay_steps=5):
        super().__init__(env)
        self.max_delay_steps = max_delay_steps
        self._buffer = []
    
    def observation(self, obs):
        # Add to buffer
        self._buffer.append(obs.copy())
        if len(self._buffer) > self.max_delay_steps:
            # Return oldest with some probability
            if len(self._buffer) > 0:
                return self._buffer[0].copy()
        # Return current if buffer not full
        return obs.copy()


class RobustEnv:
    """Simple wrapper that adds observation noise/perturbation."""
    
    def __init__(self, env, noise_std=0.0):
        self.env = env
        self.noise_std = noise_std
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add noise to observations
        if self.noise_std > 0:
            obs = obs + np.random.randn(*obs.shape) * self.noise_std
        return obs, reward, terminated, truncated, info
    
    def close(self):
        return self.env.close()
    
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


def make_env(stage, noise_std, seed, rank):
    """Create environment with observation noise."""
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=seed + rank)
        if noise_std > 0:
            env = RobustEnv(env, noise_std=noise_std)
        return env
    return _init


def train_with_delay(
    stage=5,
    total_steps=2000000,
    n_envs=16,
    delay_steps=5,
    seed=0,
    save_dir="runs/delay_trained"
):
    """Train with delay injection."""
    
    print("=" * 60)
    print(f"Delay Training - Stage {stage}, Delay {delay_steps} steps ({delay_steps*10}ms)")
    print(f"Total steps: {total_steps}, N envs: {n_envs}")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"{save_dir}/stage_{stage}_delay{delay_steps}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create environments
    env_fns = [
        make_env(stage=stage, delay_steps=delay_steps, seed=seed, rank=i)
        for i in range(n_envs)
    ]
    train_env = DummyVecEnv(env_fns)
    
    # Eval environment (no delay for fair comparison)
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
    
    # Final evaluation with delay
    print("\n" + "=" * 60)
    print("Final Evaluation (with delay)")
    print("=" * 60)
    
    eval_env_delay = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_env_delay.set_target_speed(1.0)
    eval_env_delay.set_moving_target(True)
    eval_env_delay.reset(seed=seed+2000)
    eval_env_delay = DelayEnv(eval_env_delay, delay_steps=delay_steps)
    
    successes = 0
    total_errors = 0
    
    for ep in range(100):
        obs, _ = eval_env_delay.reset()
        steps = 0
        done = False
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_delay.step(action)
            done = terminated or truncated
            steps += 1
        
        if steps >= 500:
            successes += 1
        if 'tracking_error' in info:
            total_errors += info['tracking_error']
    
    success_rate = successes / 100 * 100
    avg_error = total_errors / 100
    
    print(f"Success rate (with delay): {success_rate:.1f}%")
    print(f"Mean tracking error: {avg_error:.3f}m")
    
    # Eval without delay
    print("\n" + "=" * 60)
    print("Final Evaluation (no delay)")
    print("=" * 60)
    
    eval_env_clean = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
    eval_env_clean.set_target_speed(1.0)
    eval_env_clean.set_moving_target(True)
    eval_env_clean.reset(seed=seed+2000)
    
    successes = 0
    total_errors = 0
    
    for ep in range(100):
        obs, _ = eval_env_clean.reset()
        steps = 0
        done = False
        
        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env_clean.step(action)
            done = terminated or truncated
            steps += 1
        
        if steps >= 500:
            successes += 1
        if 'tracking_error' in info:
            total_errors += info['tracking_error']
    
    success_rate_no_delay = successes / 100 * 100
    avg_error_no_delay = total_errors / 100
    
    print(f"Success rate (no delay): {success_rate_no_delay:.1f}%")
    print(f"Mean tracking error: {avg_error_no_delay:.3f}m")
    
    results = {
        "experiment": "delay_training",
        "stage": stage,
        "delay_steps": delay_steps,
        "delay_ms": delay_steps * 10,
        "total_steps": total_steps,
        "n_envs": n_envs,
        "seed": seed,
        "success_with_delay": success_rate,
        "success_no_delay": success_rate_no_delay,
        "mean_error_with_delay": avg_error,
        "mean_error_no_delay": avg_error_no_delay,
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
    parser = argparse.ArgumentParser(description="Train with delay injection")
    parser.add_argument("--stage", type=int, default=5, help="Curriculum stage")
    parser.add_argument("--steps", type=int, default=2000000, help="Total training steps")
    parser.add_argument("--n-envs", type=int, default=16, help="Number of parallel envs")
    parser.add_argument("--delay-ms", type=int, default=50, help="Delay in milliseconds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save-dir", type=str, default="runs/delay_trained", help="Save directory")
    
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