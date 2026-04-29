#!/usr/bin/env python3
"""Test delay robustness with trained checkpoint."""

import numpy as np
from stable_baselines3 import SAC
from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
import sys

class DelayEnv:
    def __init__(self, env, delay_steps):
        self.env = env
        self.delay_steps = delay_steps
        self._buffer = []
        self.observation_space = env.observation_space
        self.action_space = env.action_space
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buffer = [obs.copy() for _ in range(self.delay_steps)]
        return self._buffer[0].copy(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs.copy())
        if len(self._buffer) > self.delay_steps:
            self._buffer.pop(0)
        return self._buffer[0].copy(), reward, terminated, truncated, info
    
    def close(self):
        return self.env.close()


def test_delay(model_path, delay_ms_vals=[0, 25, 50, 100], n_episodes=50):
    """Test model at different delay values."""
    print(f"Loading model: {model_path}")
    model = SAC.load(model_path)
    
    results = {}
    
    for delay_ms in delay_ms_vals:
        delay_steps = delay_ms // 10
        
        env = RMAQuadrotorEnv(curriculum_stage=5, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        if delay_steps > 0:
            env = DelayEnv(env, delay_steps)
        
        successes = 0
        total_err = 0
        
        for ep in range(n_episodes):
            obs, _ = env.reset()
            steps = 0
            done = False
            
            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            if steps >= 500:
                successes += 1
            if 'tracking_error' in info:
                total_err += info['tracking_error']
        
        success_rate = successes / n_episodes * 100
        avg_err = total_err / n_episodes
        
        results[delay_ms] = {
            'success_rate': success_rate,
            'mean_error': avg_err
        }
        print(f"  {delay_ms:3d}ms: {success_rate:5.1f}% success, {avg_err:.3f}m error")
        
        env.close()
    
    return results


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models_stage5_curriculum/stage_5/seed_0/final.zip"
    
    print("=" * 60)
    print("Delay Robustness Test")
    print("=" * 60)
    
    results = test_delay(model_path, [0, 25, 50, 100], n_episodes=50)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    for delay_ms, result in results.items():
        print(f"  {delay_ms}ms: {result['success_rate']:.1f}%")