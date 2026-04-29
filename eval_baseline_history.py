#!/usr/bin/env python3
"""
Test baseline model with history wrapper at evaluation time.
Does having history help a baseline model?
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from train_gru_stage5 import HistoryWrapper
from env_rma import RMAQuadrotorEnv


def get_action_direct(model, obs):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(model.device).unsqueeze(0)
    with torch.no_grad():
        action, _ = model.actor.action_log_prob(obs_tensor)
    return action.cpu().numpy()[0]


def evaluate_baseline_without_history(model_path, stage=5, n_episodes=100):
    """Baseline without history."""
    print("Baseline without history wrapper")
    print("-" * 40)
    
    model = SAC.load(model_path)
    
    successes = 0
    for ep in range(n_episodes):
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=ep)
        
        # No wrapper - raw 63-dim obs
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
    
    print(f"Success rate: {successes/n_episodes*100}%\n")
    return successes / n_episodes * 100


def evaluate_baseline_with_history(model_path, stage=5, history_len=4, n_episodes=100):
    """Baseline WITH history wrapper at eval."""
    print(f"Baseline with history wrapper ({history_len} frames)")
    print("-" * 40)
    
    model = SAC.load(model_path)
    
    successes = 0
    for ep in range(n_episodes):
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=ep)
        
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
    
    print(f"Success rate: {successes/n_episodes*100}%\n")
    return successes / n_episodes * 100


if __name__ == "__main__":
    baseline_path = "models_stage5_curriculum/stage_5/seed_0/final.zip"
    
    print("=" * 60)
    print("Baseline Model + History Wrapper Test")
    print("=" * 60)
    
    s1 = evaluate_baseline_without_history(baseline_path)
    s2 = evaluate_baseline_with_history(baseline_path)
    
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline (no wrapper):     {s1}%")
    print(f"Baseline + history wrap: {s2}%")