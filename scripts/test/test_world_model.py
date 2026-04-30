#!/usr/bin/env python3
"""Test world model delay compensation - simple version."""

import numpy as np
import torch
import torch.nn as nn
from collections import deque

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class SimpleWorldModel(nn.Module):
    def __init__(self, obs_dim=63, action_dim=4, hidden_dim=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.layers(x)


def predict_current_state(model, delayed_obs, action_history):
    """Roll forward to predict current state."""
    current_pred = torch.tensor(delayed_obs, dtype=torch.float32).unsqueeze(0)
    
    for action in action_history:
        act_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            current_pred = model(current_pred, act_tensor)
    
    return current_pred.numpy()[0]


def get_action_simple(policy_model, obs, act_dim=4):
    """Simple random action for testing (skip policy for now)."""
    return np.random.uniform(-1, 1, act_dim)


def test_world_model(delay_steps=5):
    """Test with world model compensation."""
    print(f"\nWith world model (delay={delay_steps*10}ms)")
    
    world_model = SimpleWorldModel()
    world_model.load_state_dict(torch.load("runs/world_model_20260429_174014.pt", map_location='cpu'))
    world_model.eval()
    
    successes = 0
    for ep in range(100):
        env = RMAQuadrotorEnv(curriculum_stage=5, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=ep)
        
        obs_buf = deque(maxlen=delay_steps + 1)
        act_buf = deque(maxlen=delay_steps)
        
        raw = env.reset()
        obs = raw[0] if isinstance(raw, tuple) else raw
        
        for _ in range(delay_steps):
            act = get_action_simple(None, obs)
            raw_next, _, _, _, _ = env.step(act)
            next_obs = raw_next[0] if isinstance(raw_next, tuple) else raw_next
            obs_buf.append(next_obs)
            act_buf.append(act)
        
        done = False
        steps = 0
        
        while not done and steps < 500:
            if len(obs_buf) > delay_steps:
                delayed = obs_buf[0]
                actions = list(act_buf)[:delay_steps]
                compensated = predict_current_state(world_model, delayed, actions)
                obs = compensated
            
            act = get_action_simple(None, obs)
            raw_next, _, terminated, truncated, _ = env.step(act)
            next_obs = raw_next[0] if isinstance(raw_next, tuple) else raw_next
            
            obs_buf.append(next_obs)
            act_buf.append(act)
            
            if len(obs_buf) > delay_steps:
                obs = obs_buf[-1]
            else:
                obs = next_obs
            
            done = terminated or truncated
            steps += 1
        
        if steps >= 500:
            successes += 1
        env.close()
    
    print(f"Success: {successes}%")


if __name__ == "__main__":
    print("=" * 50)
    print("World Model Delay Compensation Test")
    print("=" * 50)
    test_world_model(delay_steps=3)
    test_world_model(delay_steps=5)