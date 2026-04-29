#!/usr/bin/env python3
"""
Test delay robustness by using DelayObservationWrapper at evaluation time.
This simulates sensor delays without changing the model input size.
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from train_with_delay_fixed import DelayObservationWrapper
from train_gru_stage5 import HistoryWrapper
from env_rma import RMAQuadrotorEnv


def get_action_direct(model, obs):
    """Get action by calling actor directly."""
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(model.device).unsqueeze(0)
    with torch.no_grad():
        action, _ = model.actor.action_log_prob(obs_tensor)
    return action.cpu().numpy()[0]


def evaluate_gru_with_delay(model_path, stage=5, history_len=4, delay_steps=0, n_episodes=100):
    """
    Evaluate GRU model with simulated sensor delay.
    Uses DelayObservationWrapper to delay the observations.
    """
    delay_ms = delay_steps * 10
    print(f"GRU model + {delay_ms}ms delay ({delay_steps} steps)")
    print("-" * 40)
    
    model = SAC.load(model_path)
    
    successes = 0
    for ep in range(n_episodes):
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=ep)
        
        # Apply delay wrapper
        if delay_steps > 0:
            env = DelayObservationWrapper(env, delay_steps=delay_steps)
        
        # Then history wrapper (model expects 252-dim)
        env = HistoryWrapper(env, history_len=history_len)
        
        obs, _ = env.reset()
        steps = 0
        done = False
        
        while not done and steps < 500:
            action = get_action_direct(model, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        if steps >= 500:
            successes += 1
        
        env.close()
    
    rate = successes / n_episodes * 100
    print(f"Success rate: {rate}%\n")
    return rate


if __name__ == "__main__":
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "runs/gru_trained/stage_5_h4_20260428_180715/final.zip"
    
    print("=" * 60)
    print("GRU Model - Delay Robustness Test")
    print("=" * 60)
    
    s0 = evaluate_gru_with_delay(model_path, history_len=4, delay_steps=0)
    s3 = evaluate_gru_with_delay(model_path, history_len=4, delay_steps=3)   # 30ms
    s5 = evaluate_gru_with_delay(model_path, history_len=4, delay_steps=5)   # 50ms
    s10 = evaluate_gru_with_delay(model_path, history_len=4, delay_steps=10)  # 100ms
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"0ms delay:  {s0}%")
    print(f"30ms delay: {s3}%")
    print(f"50ms delay: {s5}%")
    print(f"100ms delay: {s10}%")