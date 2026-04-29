#!/usr/bin/env python3
"""World Model Training - Simplified with single env."""

import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

from env_rma import RMAQuadrotorEnv


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


def collect_transitions(n_episodes=500):
    """Collect transition data from single env."""
    print("Collecting transitions...")
    
    obs_buf = []
    act_buf = []
    next_buf = []
    
    for ep in range(n_episodes):
        env = RMAQuadrotorEnv(curriculum_stage=5, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        env.reset(seed=ep * 100)
        
        raw_obs = env.reset()
        obs = raw_obs[0] if isinstance(raw_obs, tuple) else raw_obs
        
        done = False
        step = 0
        
        while not done and step < 500:
            action = env.action_space.sample()
            raw_next, _, terminated, truncated, _ = env.step(action)
            next_obs = raw_next[0] if isinstance(raw_next, tuple) else raw_next
            
            obs_buf.append(np.array(obs, dtype=np.float32))
            act_buf.append(np.array(action, dtype=np.float32))
            next_buf.append(np.array(next_obs, dtype=np.float32))
            
            obs = next_obs
            done = terminated or truncated
            step += 1
        
        env.close()
        
        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}/{n_episodes}")
    
    X = torch.tensor(np.array(obs_buf), dtype=torch.float32)
    A = torch.tensor(np.array(act_buf), dtype=torch.float32)
    Y = torch.tensor(np.array(next_buf), dtype=torch.float32)
    
    print(f"Collected {len(X)} transitions")
    return X, A, Y


def train_world_model(X, A, Y, epochs=30, batch_size=512, lr=1e-3):
    print("Training world model...")
    
    model = SimpleWorldModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    dataset = torch.utils.data.TensorDataset(X, A, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for obs, act, next_obs in loader:
            pred = model(obs, act)
            loss = loss_fn(pred, next_obs)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(obs)
        
        avg_loss = total_loss / len(X)
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, MSE: {avg_loss:.6f}")
    
    return model


def test_world_model(model, n_test=10):
    print(f"Testing world model predictions...")
    
    model.eval()
    errors = []
    
    for ep in range(n_test):
        env = RMAQuadrotorEnv(curriculum_stage=5, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        raw_obs = env.reset(seed=ep + 1000)
        obs = raw_obs[0] if isinstance(raw_obs, tuple) else raw_obs
        
        for step in range(100):
            action = env.action_space.sample()
            raw_next, _, terminated, truncated, _ = env.step(action)
            next_obs = raw_next[0] if isinstance(raw_next, tuple) else raw_next
            
            with torch.no_grad():
                pred = model(
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0),
                    torch.tensor(action, dtype=torch.float32).unsqueeze(0)
                )
            
            error = np.mean((pred.numpy()[0] - next_obs) ** 2)
            errors.append(error)
            
            obs = next_obs
            if terminated or truncated:
                break
    
    env.close()
    
    avg_error = np.mean(errors)
    print(f"  Prediction MSE: {avg_error:.4f}")
    return avg_error


if __name__ == "__main__":
    print("=" * 60)
    print("World Model Training")
    print("=" * 60)
    
    X, A, Y = collect_transitions(n_episodes=500)
    
    model = train_world_model(X, A, Y)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(model.state_dict(), f"runs/world_model_{timestamp}.pt")
    print(f"Saved to runs/world_model_{timestamp}.pt")
    
    test_world_model(model)