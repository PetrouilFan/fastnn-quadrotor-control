#!/usr/bin/env python3
"""
Robustness Evaluation Script

Standardized evaluation for testing robustness across multiple dimensions:
- Delay: 0, 25, 50, 100 ms
- Noise: 0x, 0.5x, 1x baseline
- Mass: 0.6, 0.8, 1.0, 1.2, 1.4
- Wind: 0, ±0.5, ±1.0, ±2.0 N

Usage:
    python eval_robustness.py --model runs/gru_delay/final.zip --output results/
"""

import numpy as np
import torch
import argparse
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from gymnasium.core import Env
from stable_baselines3 import SAC

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class NoiseEnv(Env):
    """Environment that adds IMU noise to observations."""
    
    def __init__(self, env, noise_scale=0.5):
        super().__init__()
        self.env = env
        self.noise_scale = noise_scale
        self._step_count = 0
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = {"render_modes": env.metadata.get("render_modes", [])}
        self.render_mode = None
    
    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self._step_count = 0
        return obs + np.random.randn(*obs.shape) * self.noise_scale, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        noisy_obs = obs + np.random.randn(*obs.shape) * self.noise_scale
        self._step_count += 1
        return noisy_obs, reward, terminated, truncated, info
    
    def close(self):
        return self.env.close()
    
    @property
    def data(self):
        return self.env.data
    
    @property
    def target_pos(self):
        return self.env.target_pos


class DelayEnv(Env):
    """Environment that adds observation delay."""
    
    def __init__(self, env, delay_steps=5):
        super().__init__()
        self.env = env
        self.delay_steps = delay_steps
        self._buffer = []
        
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = {"render_modes": env.metadata.get("render_modes", [])}
        self.render_mode = None
    
    def reset(self, seed=None, **kwargs):
        obs, info = self.env.reset(seed=seed, **kwargs)
        self._buffer = [obs.copy() for _ in range(self.delay_steps)]
        return self._buffer[0].copy(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs.copy())
        if len(self._buffer) > self.delay_steps:
            self._buffer.pop(0)
        delayed_obs = self._buffer[0].copy()
        return delayed_obs, reward, terminated, truncated, info
    
    def close(self):
        return self.env.close()
        return self._buffer[0].copy(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs.copy())
        if len(self._buffer) > self.delay_steps:
            self._buffer.pop(0)
        return self._buffer[0].copy(), reward, terminated, truncated, info
    
    def close(self):
        return self.env.close()


def evaluate_model(
    model_path: str,
    stage: int = 3,
    n_episodes: int = 100,
    delay_ms: int = 0,
    noise_scale: float = 0.0,
    mass_ratio: float = 1.0,
    wind_force: float = 0.0,
) -> Dict:
    """Evaluate a single model under specific conditions."""
    
    # Create environment
    env = RMAQuadrotorEnv(
        curriculum_stage=stage, 
        use_direct_control=True,
    )
    
    # Apply mass ratio
    if mass_ratio != 1.0:
        env.payload_mass = env.payload_mass * mass_ratio
    
    # Apply wind
    if wind_force != 0.0:
        env.wind_force = np.array([wind_force, 0, 0])
    
    # Apply delay
    if delay_ms > 0:
        delay_steps = delay_ms // 10
        env = DelayEnv(env, delay_steps=delay_steps)
    
    # Apply noise
    if noise_scale > 0:
        env = NoiseEnv(env, noise_scale=noise_scale)
    
    # Load model
    model = SAC.load(model_path)
    
    # Run evaluation
    successes = 0
    total_rewards = 0
    tracking_errors = []
    crash_reasons = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_rewards = 0
        episode_steps = 0
        done = False
        
        while not done and episode_steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_rewards += reward
            episode_steps += 1
            
            if 'tracking_error' in info:
                tracking_errors.append(info['tracking_error'])
        
        if episode_steps >= 500:
            successes += 1
        else:
            crash_reasons.append(info.get('crash_reason', 'unknown'))
        
        total_rewards += episode_rewards
    
    env.close()
    
    return {
        'success_rate': successes / n_episodes * 100,
        'avg_reward': total_rewards / n_episodes,
        'avg_steps': n_episodes * 500 / (successes + 0.001),
        'mean_tracking_error': np.mean(tracking_errors) if tracking_errors else 0,
        'crash_reasons': crash_reasons,
    }


def run_robustness_sweep(
    model_path: str,
    output_dir: str,
    n_episodes: int = 100,
) -> Dict:
    """Run full robustness sweep."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Test conditions
    delays = [0, 25, 50, 100]
    noise_levels = [0.0, 0.5, 1.0]
    mass_ratios = [0.6, 0.8, 1.0, 1.2, 1.4]
    winds = [0.0, 0.5, 1.0, 2.0]
    
    results = {
        'model_path': model_path,
        'datetime': datetime.now().isoformat(),
        'n_episodes': n_episodes,
    }
    
    # Delay sweep
    print("\n" + "=" * 60)
    print("Delay Sweep")
    print("=" * 60)
    results['delay_sweep'] = {}
    
    for delay in delays:
        print(f"Testing delay: {delay}ms")
        result = evaluate_model(
            model_path, 
            delay_ms=delay, 
            n_episodes=n_episodes
        )
        results['delay_sweep'][f'{delay}ms'] = result
        print(f"  Success rate: {result['success_rate']:.1f}%")
    
    # Noise sweep
    print("\n" + "=" * 60)
    print("Noise Sweep")
    print("=" * 60)
    results['noise_sweep'] = {}
    
    for noise in noise_levels:
        print(f"Testing noise: {noise}x")
        result = evaluate_model(
            model_path,
            noise_scale=noise,
            n_episodes=n_episodes
        )
        results['noise_sweep'][f'{noise}x'] = result
        print(f"  Success rate: {result['success_rate']:.1f}%")
    
    # Mass sweep
    print("\n" + "=" * 60)
    print("Mass Variation Sweep")
    print("=" * 60)
    results['mass_sweep'] = {}
    
    for mass in mass_ratios:
        print(f"Testing mass ratio: {mass}")
        result = evaluate_model(
            model_path,
            mass_ratio=mass,
            n_episodes=n_episodes
        )
        results['mass_sweep'][f'{mass}x'] = result
        print(f"  Success rate: {result['success_rate']:.1f}%")
    
    # Wind sweep
    print("\n" + "=" * 60)
    print("Wind Sweep")
    print("=" * 60)
    results['wind_sweep'] = {}
    
    for wind in winds:
        print(f"Testing wind: {wind}N")
        result = evaluate_model(
            model_path,
            wind_force=wind,
            n_episodes=n_episodes
        )
        results['wind_sweep'][f'{wind}N'] = result
        print(f"  Success rate: {result['success_rate']:.1f}%")
    
    # Save results
    results_path = os.path.join(output_dir, 'robustness_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run robustness evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip file")
    parser.add_argument("--output", type=str, default="runs/robustness_eval", help="Output directory")
    parser.add_argument("--episodes", type=int, default=100, help="Episodes per condition")
    
    args = parser.parse_args()
    
    run_robustness_sweep(
        model_path=args.model,
        output_dir=args.output,
        n_episodes=args.episodes,
    )