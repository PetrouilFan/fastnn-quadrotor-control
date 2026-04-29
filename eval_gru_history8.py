#!/usr/bin/env python3
"""Evaluate GRU models with different history sizes."""

import numpy as np
import torch
from stable_baselines3 import SAC
from train_with_delay_fixed import DelayObservationWrapper
from train_gru_stage5 import HistoryWrapper
from env_rma import RMAQuadrotorEnv


def get_action_direct(model, obs):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(model.device).unsqueeze(0)
    with torch.no_grad():
        action, _ = model.actor.action_log_prob(obs_tensor)
    return action.cpu().numpy()[0]


def evaluate_gru_with_delay(model_path, stage=5, history_len=4, delay_steps=0, n_episodes=100):
    delay_ms = delay_steps * 10
    print(f"History={history_len},Delay={delay_ms}ms", end=" ")
    
    model = SAC.load(model_path)
    
    successes = 0
    for ep in range(n_episodes):
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=ep)
        
        if delay_steps > 0:
            env = DelayObservationWrapper(env, delay_steps=delay_steps)
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
    print(f"-> {rate}%")
    return rate


if __name__ == "__main__":
    print("=" * 60)
    print("GRU History 8 - Delay Robustness Test")
    print("=" * 60)
    
    model_path = "runs/gru_trained/stage_5_h8_20260429_155344/final.zip"
    history_len = 8
    
    s0 = evaluate_gru_with_delay(model_path, history_len=history_len, delay_steps=0)
    s3 = evaluate_gru_with_delay(model_path, history_len=history_len, delay_steps=3)
    s5 = evaluate_gru_with_delay(model_path, history_len=history_len, delay_steps=5)
    s10 = evaluate_gru_with_delay(model_path, history_len=history_len, delay_steps=10)
    
    print("=" * 60)
    print("Summary (History=8)")
    print("=" * 60)
    print(f"0ms:   {s0}%")
    print(f"30ms:  {s3}%")
    print(f"50ms:  {s5}%")
    print(f"100ms: {s10}%")