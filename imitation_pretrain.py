#!/usr/bin/env python3
"""
Phase 0: Imitation Learning from PD Controller (Stage 1 Data)

Pretrains NN to mimic PD controller behavior using collected Stage 1 data.
This gives the NN a good initialization for stable hover before RL fine-tuning.

Usage:
    python imitation_pretrain.py --data data/pd_stage1_buffer.npz --epochs 50 --save models/nn_pretrained.pt
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from controllers import NNOnlyController

DEPLOYABLE_DIM = 52
ACTION_DIM = 4


class ImitationTrainer:
    """Trainer for imitation learning from PD data."""

    def __init__(self, model, lr=1e-3, batch_size=256, device="cuda"):
        self.device = device
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.batch_size = batch_size
        self.loss_history = []

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for states, actions in dataloader:
            states = states.to(self.device)
            actions = actions.to(self.device)

            # Forward pass
            preds = self.model(states)

            # MSE loss on all 4 outputs
            loss = nn.MSELoss()(preds, actions)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        self.loss_history.append(avg_loss)
        return avg_loss

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
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
        }, path)
        print(f"Saved pretrained model to {path}")


def load_data(path):
    """Load PD data from npz file."""
    data = np.load(path)
    states = data["states"]
    actions = data["actions"]
    rewards = data["rewards"]

    print(f"Loaded data: {states.shape[0]} transitions")
    print(f"  State shape: {states.shape}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Mean reward: {rewards.mean():.2f}")

    # Normalize states (standardize)
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8
    states = (states - state_mean) / state_std

    # Normalize actions (standardize for better training)
    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0) + 1e-8
    actions = (actions - action_mean) / action_std

    return states, actions, (state_mean, state_std), (action_mean, action_std)


def main():
    parser = argparse.ArgumentParser(description="Imitation learning from PD data")
    parser.add_argument("--data", type=str, default="data/pd_stage1_buffer.npz",
                        help="Path to PD data")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save", type=str, default="models/nn_pretrained.pt",
                        help="Save path for pretrained model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    states, actions, state_norm, action_norm = load_data(args.data)

    # Create model
    model = NNOnlyController(state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dataloader
    dataset = TensorDataset(
        torch.FloatTensor(states),
        torch.FloatTensor(actions)
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Split into train/val (90/10)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Train
    trainer = ImitationTrainer(model, lr=args.lr, batch_size=args.batch_size, device=device)

    print(f"\nTraining imitation learning for {args.epochs} epochs...")
    best_val_loss = float("inf")

    for epoch in tqdm(range(args.epochs)):
        train_loss = trainer.train_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        trainer.scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            torch.save({
                "model_state_dict": model.state_dict(),
                "state_norm": state_norm,
                "action_norm": action_norm,
                "epoch": epoch,
                "val_loss": val_loss,
            }, args.save.replace(".pt", "_best.pt"))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

    # Save final model
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_norm": state_norm,
        "action_norm": action_norm,
        "epoch": args.epochs,
        "val_loss": best_val_loss,
        "loss_history": trainer.loss_history,
    }, args.save)
    print(f"\nFinal model saved to {args.save}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Quick evaluation: compare PD vs NN actions on a few samples
    model.eval()
    with torch.no_grad():
        sample_states = torch.FloatTensor(states[:10]).to(device)
        nn_actions = model(sample_states).cpu().numpy()
        pd_actions = actions[:10]

        print(f"\nSample comparison (first 5 transitions):")
        print(f"  NN actions:  {nn_actions[:5].round(2)}")
        print(f"  PD actions:  {pd_actions[:5].round(2)}")


if __name__ == "__main__":
    main()