#!/usr/bin/env python3
"""
Simple Behavioral Cloning for Quadrotor Control

Trains a neural network to mimic the PD controller using supervised learning.
The model outputs raw control values directly (no scaling).
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class SimpleBCController(nn.Module):
    """Simple MLP for behavioral cloning - outputs raw actions."""

    def __init__(self, state_dim=52, action_dim=4, hidden_dims=[256, 256, 128]):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Raw output - no scaling applied
        return self.net(x)


class ImitationTrainer:
    def __init__(self, model, lr=1e-3, batch_size=256, device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        self.batch_size = batch_size
        self.loss_history = []

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for states, actions in dataloader:
            states = states.to(self.device)
            actions = actions.to(self.device)

            preds = self.model(states)
            loss = nn.MSELoss()(preds, actions)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for states, actions in dataloader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                preds = self.model(states)
                loss = nn.MSELoss()(preds, actions)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "loss_history": self.loss_history,
        }, path)
        print(f"Saved to {path}")


def load_data(path):
    data = np.load(path)
    states = data["states"]
    actions = data["actions"]
    print(f"Loaded {len(states)} transitions")
    print(f"State shape: {states.shape}, Action shape: {actions.shape}")
    print(f"Action stats: thrust={actions[:,0].mean():.2f}±{actions[:,0].std():.2f}, "
          f"roll={actions[:,1].mean():.4f}±{actions[:,1].std():.4f}")
    return states, actions


def main():
    parser = argparse.ArgumentParser(description="Behavioral cloning for quadrotor")
    parser.add_argument("--data", type=str, default="data/pd_stage1_500ep.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save", type=str, default="models/simple_bc.pt")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    states, actions = load_data(args.data)

    # Normalize states only (not actions)
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8
    states_norm = (states - state_mean) / state_std

    print(f"State normalization: mean range [{state_mean[:3].min():.4f}, {state_mean[:3].max():.4f}]")
    print(f"                   std range [{state_std[:3].min():.4f}, {state_std[:3].max():.4f}]")

    # Create model - simple MLP that outputs raw actions
    model = SimpleBCController(state_dim=52, action_dim=4, hidden_dims=[256, 256, 128])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Datasets
    dataset = TensorDataset(
        torch.FloatTensor(states_norm),
        torch.FloatTensor(actions)
    )

    # Train/val split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Train
    trainer = ImitationTrainer(model, lr=args.lr, batch_size=args.batch_size, device=device)

    print(f"\nTraining for {args.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in tqdm(range(args.epochs)):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        trainer.scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "state_mean": state_mean,
                "state_std": state_std,
                "epoch": epoch,
                "val_loss": val_loss,
            }, args.save.replace(".pt", "_best.pt"))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "epoch": args.epochs,
        "val_loss": best_val_loss,
        "loss_history": trainer.loss_history,
    }, args.save)

    print(f"\nFinal model saved to {args.save}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Test with first sample
    model.eval()
    with torch.no_grad():
        sample = torch.FloatTensor(states_norm[:1]).to(device)
        pred = model(sample).cpu().numpy()[0]
        target = actions[0]
        print(f"\nSample prediction: {pred.round(3)}")
        print(f"Target:           {target.round(3)}")


if __name__ == "__main__":
    main()