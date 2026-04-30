#!/usr/bin/env python3
"""
Transformer Data Scaling Study

Tests whether Transformer BC's underperformance is data-limited by training
with increasing amounts of data.

Hypothesis: Transformer (613K params) needs more data than GRU (177K params)
to outperform due to higher model capacity.

Usage:
    python exp_transformer_scaling.py --data data/pd_stage1_diverse.npz
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
import json

from fastnn_quadrotor.utils.transformer_bc import GRUController, TransformerController
from fastnn_quadrotor.training.train_transformer_bc import BCController, train_bc, evaluate


DEPLOYABLE_DIM = 51  # Use 51 dims (no mass_est)
ACTION_DIM = 4


def train_and_evaluate(model, model_type, train_states, train_actions, epochs=30, batch_size=256, lr=1e-3):
    """Train a model and evaluate it."""
    print(f"\n{'='*50}")
    print(f"Training {model_type} with {len(train_states)} samples")
    print(f"{'='*50}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Normalize
    state_mean = train_states.mean(axis=0)
    state_std = train_states.std(axis=0) + 1e-8
    train_states_norm = (train_states - state_mean) / state_std

    # Train with appropriate method for each model type
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_states_norm),
        torch.FloatTensor(train_actions)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    for epoch in range(epochs):
        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            if model_type == 'gru':
                preds, _ = model(batch_states)
            elif model_type == 'transformer':
                # Transformer needs sequence dimension
                preds = model.forward_sequence(batch_states.unsqueeze(1))
                preds = preds[:, -1]  # Take last timestep
            else:
                preds = model(batch_states)

            loss = nn.MSELoss()(preds, batch_actions)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()

    model.eval()

    # Wrap controller
    bc = BCController(model, state_mean, state_std, device, model_type=model_type)

    # Evaluate
    results = {}
    for stage in [1, 3, 4]:
        from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        metrics = evaluate(env, bc, n_episodes=50)
        print(f"  Stage {stage}: success={metrics['success_rate']:.0%}, dist={metrics['mean_final_dist']:.3f}m")
        results[stage] = metrics['success_rate']
        env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Transformer Data Scaling Study")
    parser.add_argument("--data", type=str, default="data/pd_stage1_diverse.npz",
                        help="Path to training data")
    parser.add_argument("--max-data", type=int, default=166781,
                        help="Maximum data points to use")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    data = np.load(args.data)
    all_states = data["states"][:args.max_data, :DEPLOYABLE_DIM]  # Use only 51 dims
    all_actions = data["actions"][:args.max_data]

    print(f"Total data available: {len(all_states)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Data fractions to test
    fractions = [0.1, 0.25, 0.5, 1.0]

    results = {
        "gru": {},
        "transformer": {}
    }

    for frac in fractions:
        n_samples = int(len(all_states) * frac)
        indices = np.random.permutation(len(all_states))[:n_samples]
        train_states = all_states[indices]
        train_actions = all_actions[indices]

        print(f"\n{'='*60}")
        print(f"Testing with {frac*100:.0f}% of data ({n_samples} samples)")
        print(f"{'='*60}")

        # GRU model
        gru_model = GRUController(state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM, hidden_dim=128, num_layers=2).to(device)
        gru_results = train_and_evaluate(gru_model, "gru", train_states, train_actions, epochs=30)
        results["gru"][frac] = gru_results

        # Transformer model
        transformer_model = TransformerController(state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM, d_model=128, n_layers=3).to(device)
        transformer_results = train_and_evaluate(transformer_model, "transformer", train_states, train_actions, epochs=30)
        results["transformer"][frac] = transformer_results

    # Save results
    with open("transformer_scaling_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY: Transformer vs GRU Data Scaling")
    print("="*60)
    print(f"{'Fraction':<12} {'GRU S3':<12} {'GRU S4':<12} {'Trans S3':<12} {'Trans S4':<12}")
    print("-" * 60)
    for frac in fractions:
        gru_s3 = results["gru"][frac][3]
        gru_s4 = results["gru"][frac][4]
        trans_s3 = results["transformer"][frac][3]
        trans_s4 = results["transformer"][frac][4]
        print(f"{frac*100:.0f}%{'':<9} {gru_s3:>9.0%}  {gru_s4:>9.0%}  {trans_s3:>9.0%}  {trans_s4:>9.0%}")


if __name__ == "__main__":
    main()