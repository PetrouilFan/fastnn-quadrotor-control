#!/usr/bin/env python3
"""
BC that predicts residuals from hover.

Instead of predicting absolute actions, BC predicts:
  residual = PD_action - hover_action
where hover_action = [10, 0, 0, 0]

At inference time: final_action = hover + residual

This should make BC more stable since it learns to make small corrections.
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class SimpleBCController(nn.Module):
    def __init__(self, state_dim=52, action_dim=4, hidden_dims=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BCController:
    def __init__(self, model, state_mean, state_std, device="cuda", hover_action=None):
        self.model = model
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device
        self.hover_action = hover_action if hover_action is not None else np.array([10.0, 0.0, 0.0, 0.0])

    def predict(self, obs):
        with torch.no_grad():
            state_raw = torch.FloatTensor(obs[:52]).to(self.device)
            state_norm = (state_raw - torch.FloatTensor(self.state_mean).to(self.device)) / \
                         (torch.FloatTensor(self.state_std).to(self.device) + 1e-8)
            residual = self.model(state_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
            # Add hover to get actual action
            action = self.hover_action + residual
            return action

    def to_env_action(self, raw_action):
        normalized = np.zeros(4)
        normalized[0] = np.clip((raw_action[0] - 10.0) / 10.0, -1.0, 1.0)
        normalized[1] = np.clip(raw_action[1] / 3.0, -1.0, 1.0)
        normalized[2] = np.clip(raw_action[2] / 3.0, -1.0, 1.0)
        normalized[3] = np.clip(raw_action[3] / 2.0, -1.0, 1.0)
        return normalized


def train_bc_residual(model, states, actions, hover_action, epochs=30, batch_size=256,
                      lr=1e-3, device="cuda"):
    """Train BC to predict residual from hover."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Compute residuals
    residuals = actions - hover_action

    dataset = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(residuals))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch_states, batch_residuals in loader:
            batch_states = batch_states.to(device)
            batch_residuals = batch_residuals.to(device)

            preds = model(batch_states)
            loss = nn.MSELoss()(preds, batch_residuals)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")

    return model


def evaluate(env, bc_controller, n_episodes=50):
    successes = 0
    survivals = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated) and steps < 500:
            bc_raw = bc_controller.predict(obs)
            action = bc_controller.to_env_action(bc_raw)
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

        if steps >= 500:
            successes += 1
            survivals += 1
        elif not terminated:
            survivals += 1

    return successes / n_episodes, survivals / n_episodes


def main():
    parser = argparse.ArgumentParser(description="BC predicting residuals from hover")
    parser.add_argument("--data", type=str, default="data/pd_stage1_diverse.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save", type=str, default="models/bc_residual.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    data = np.load(args.data)
    states = data["states"]
    actions = data["actions"]
    print(f"Loaded {len(states)} transitions")

    # Hover action
    hover_action = np.array([10.0, 0.0, 0.0, 0.0])

    # Normalize states
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8
    states_norm = (states - state_mean) / state_std

    # Train
    print(f"\n=== Training BC residual model ===")
    model = SimpleBCController(state_dim=52, action_dim=4, hidden_dims=[256, 256, 128]).to(device)
    model = train_bc_residual(model, states_norm, actions, hover_action,
                              epochs=args.epochs, device=device)

    bc = BCController(model, state_mean, state_std, device, hover_action)

    # Evaluate
    print(f"\n=== Evaluation ===")
    env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
    success, survival = evaluate(env, bc, n_episodes=50)
    print(f"Result: success={success:.1%}, survival={survival:.1%}")
    env.close()

    # Save
    if success > 0 or survival > 0.5:
        torch.save({
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "hover_action": hover_action,
        }, args.save)
        print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()