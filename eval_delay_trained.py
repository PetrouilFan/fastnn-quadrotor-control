#!/usr/bin/env python3
"""Test delay robustness of model trained with 10ms delay."""

import numpy as np
import torch
from stable_baselines3 import SAC
from train_with_delay_fixed import DelayObservationWrapper
from env_rma import RMAQuadrotorEnv


def get_action(model, obs):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(model.device).unsqueeze(0)
    with torch.no_grad():
        action, _ = model.actor.action_log_prob(obs_tensor)
    return action.cpu().numpy()[0]


def evaluate(model_path, stage=5, delay_steps=0, n_episodes=100):
    delay_ms = delay_steps * 10
    print(f"Delay: {delay_ms}ms ({delay_steps} steps)")
    print("-" * 40)
    
    model = SAC.load(model_path)
    
    successes = 0
    for ep in range(n_episodes):
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=ep)
        
        if delay_steps > 0:
            env = DelayObservationWrapper(env, delay_steps=delay_steps)
        
        obs, _ = env.reset()
        steps = 0
        done = False
        
        while not done and steps < 500:
            action = get_action(model, obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        
        if steps >= 500:
            successes += 1
        env.close()
    
    rate = successes / n_episodes * 100
    print(f"Success: {rate}%\n")
    return rate


if __name__ == "__main__":
    model_path = "runs/delay_trained/stage_5_delay10ms_20260428_210127/final.zip"
    
    print("=" * 60)
    print("Model trained with 10ms delay - Delay Robustness Test")
    print("=" * 60)
    
    s0 = evaluate(model_path, delay_steps=0)
    s1 = evaluate(model_path, delay_steps=1)   # 10ms
    s2 = evaluate(model_path, delay_steps=3)   # 30ms
    s5 = evaluate(model_path, delay_steps=5)   # 50ms
    s10 = evaluate(model_path, delay_steps=10)  # 100ms
    
    print("=" * 60)
    print("Summary (trained with 10ms delay)")
    print("=" * 60)
    print(f"0ms:   {s0}%")
    print(f"10ms:  {s1}%")
    print(f"30ms:  {s2}%")
    print(f"50ms:  {s5}%")
    print(f"100ms: {s10}%")
    
    print("\n" + "=" * 60)
    print("Comparison: Baseline model")
    print("=" * 60)
    
    baseline_path = "models_stage5_curriculum/stage_5/seed_0/final.zip"
    b0 = evaluate(baseline_path, delay_steps=0)
    b10 = evaluate(baseline_path, delay_steps=10)
    
    print(f"Baseline 0ms:   {b0}%")
    print(f"Baseline 100ms: {b10}%")