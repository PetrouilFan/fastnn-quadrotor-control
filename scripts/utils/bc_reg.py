#!/usr/bin/env python3
"""
BC with L2 regularization toward stable hover.

The idea: instead of just mimicking PD actions, we add a regularization term
that pushes predictions toward the hover baseline (thrust=10N, torques=0).
This makes the BC model more conservative and stable.
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
    def __init__(self, model, state_mean, state_std, device="cuda"):
        self.model = model
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device

    def predict(self, obs):
        with torch.no_grad():
            state_raw = torch.FloatTensor(obs[:52]).to(self.device)
            state_norm = (state_raw - torch.FloatTensor(self.state_mean).to(self.device)) / \
                         (torch.FloatTensor(self.state_std).to(self.device) + 1e-8)
            action = self.model(state_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
            return action

    def to_env_action(self, raw_action):
        normalized = np.zeros(4)
        normalized[0] = np.clip((raw_action[0] - 10.0) / 10.0, -1.0, 1.0)
        normalized[1] = np.clip(raw_action[1] / 3.0, -1.0, 1.0)
        normalized[2] = np.clip(raw_action[2] / 3.0, -1.0, 1.0)
        normalized[3] = np.clip(raw_action[3] / 2.0, -1.0, 1.0)
        return normalized


def train_bc_with_reg(model, states, actions, epochs=30, batch_size=256, lr=1e-3,
                      device="cuda", reg_lambda=0.1, hover_action=None):
    """Train BC with L2 regularization toward hover."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(actions))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Default hover action (thrust=10N, torques=0)
    if hover_action is None:
        hover_action = torch.FloatTensor([10.0, 0.0, 0.0, 0.0]).to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            preds = model(batch_states)

            # MSE loss
            mse_loss = nn.MSELoss()(preds, batch_actions)

            # L2 regularization toward hover
            reg_loss = torch.mean((preds - hover_action) ** 2)

            # Combined loss
            loss = mse_loss + reg_lambda * reg_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6f} (mse + {reg_lambda}*reg)")

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
    parser = argparse.ArgumentParser(description="BC with L2 regularization toward hover")
    parser.add_argument("--data", type=str, default="data/pd_stage1_diverse.npz")
    parser.add_argument("--reg-lambda", type=float, default=0.1, help="Regularization strength")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save", type=str, default="models/bc_reg.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    data = np.load(args.data)
    states = data["states"]
    actions = data["actions"]
    print(f"Loaded {len(states)} transitions")
    print(f"Action stats: thrust={actions[:,0].mean():.2f}±{actions[:,0].std():.2f}")

    # Normalize states
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8
    states_norm = (states - state_mean) / state_std

    # Train with regularization
    reg_lambda = args.reg_lambda
    print(f"\n=== Training with reg_lambda={reg_lambda} ===")
    model = SimpleBCController(state_dim=52, action_dim=4, hidden_dims=[256, 256, 128]).to(device)
    model = train_bc_with_reg(model, states_norm, actions, epochs=args.epochs,
                              reg_lambda=reg_lambda, device=device)

    bc = BCController(model, state_mean, state_std, device)

    env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
    success, survival = evaluate(env, bc, n_episodes=30)
    print(f"  Result: success={success:.1%}, survival={survival:.1%}")
    env.close()

    if success > 0 or survival > 0.5:
        torch.save({
            "model_state_dict": model.state_dict(),
            "state_mean": state_mean,
            "state_std": state_std,
            "reg_lambda": reg_lambda,
        }, args.save)
        print(f"  Saved to {args.save}")


if __name__ == "__main__":
    main()