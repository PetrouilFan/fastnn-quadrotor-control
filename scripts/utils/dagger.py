#!/usr/bin/env python3
"""
DAgger-style Iterative Refinement for Behavioral Cloning

Algorithm:
1. Train initial BC from PD data
2. Run BC policy in environment
3. When BC deviates from PD by > threshold, use PD action instead
4. Collect (state, PD_action) pairs where BC failed
5. Retrain BC on combined data
6. Repeat until convergence

Usage:
    python dagger.py --iterations 5 --episodes-per-iter 100
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
from fastnn_quadrotor.utils.simple_bc import SimpleBCController

DEPLOYABLE_DIM = 52
ACTION_DIM = 4


class BCController:
    """Wraps BC model with environment interaction."""

    def __init__(self, model, state_mean, state_std, device="cuda"):
        self.model = model
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device

    def predict(self, obs):
        with torch.no_grad():
            state_raw = torch.FloatTensor(obs[:DEPLOYABLE_DIM]).to(self.device)
            state_norm = (state_raw - torch.FloatTensor(self.state_mean).to(self.device)) / \
                         (torch.FloatTensor(self.state_std).to(self.device) + 1e-8)
            action = self.model(state_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
            return action

    def to_env_action(self, raw_action):
        """Convert BC raw output to env action space."""
        normalized = np.zeros(4)
        normalized[0] = np.clip((raw_action[0] - 10.0) / 10.0, -1.0, 1.0)
        normalized[1] = np.clip(raw_action[1] / 3.0, -1.0, 1.0)
        normalized[2] = np.clip(raw_action[2] / 3.0, -1.0, 1.0)
        normalized[3] = np.clip(raw_action[3] / 2.0, -1.0, 1.0)
        return normalized


def run_episode(env, bc_controller, use_bc=True, threshold=1.0, max_steps=500):
    """Run one episode with BC policy.

    Returns:
        states: list of states visited
        actions: list of PD actions (what BC should have done)
        bc_actions: list of BC actions taken
        deviations: list of deviation magnitudes
    """
    states = []
    actions = []
    bc_actions_taken = []

    obs, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        # Get BC prediction
        bc_raw = bc_controller.predict(obs)

        # Get PD action for comparison/correction
        pd_action = env._cascaded_controller()

        # If BC deviates too much from PD, use PD instead
        deviation = np.linalg.norm(bc_raw - pd_action)
        use_pd = deviation > threshold

        # Record state and what BC should have done (PD action)
        states.append(obs[:DEPLOYABLE_DIM].copy())
        actions.append(pd_action.copy())

        # Use BC action or PD action based on threshold
        if use_pd:
            bc_actions_taken.append(pd_action.copy())  # Recording what we actually did
        else:
            bc_actions_taken.append(bc_raw.copy())

        # Take action (use PD if deviation too high)
        if use_pd:
            action_for_env = bc_controller.to_env_action(pd_action)
        else:
            action_for_env = bc_controller.to_env_action(bc_raw)

        obs, reward, terminated, truncated, info = env.step(action_for_env)
        steps += 1

    return {
        'states': states,
        'actions': actions,
        'bc_actions': bc_actions_taken,
        'steps': steps,
        'terminated': terminated,
        'truncated': truncated
    }


def collect_dagger_data(env, bc_controller, n_episodes, threshold=1.0):
    """Collect correction data using DAgger approach."""
    all_states = []
    all_actions = []
    corrections = 0

    for ep in range(n_episodes):
        result = run_episode(env, bc_controller, threshold=threshold)
        all_states.extend(result['states'])
        all_actions.extend(result['actions'])

        # Count how many times we used PD instead of BC
        for bc_act, pd_act in zip(result['bc_actions'], result['actions']):
            if not np.allclose(bc_act, pd_act, atol=0.1):
                corrections += 1

    print(f"  Collected {len(all_states)} transitions, {corrections} corrections "
          f"({100*corrections/len(all_states):.1f}%)")

    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)


def train_bc(model, states, actions, epochs=20, batch_size=256, lr=1e-3, device="cuda"):
    """Train BC model on given data."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(torch.FloatTensor(states), torch.FloatTensor(actions))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            preds = model(batch_states)
            loss = nn.MSELoss()(preds, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch+1}: loss={avg_loss:.6f}")

    return model


def evaluate_bc(env, bc_controller, n_episodes=50):
    """Evaluate BC policy."""
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
    parser = argparse.ArgumentParser(description="DAgger-style BC refinement")
    parser.add_argument("--iterations", type=int, default=5, help="DAgger iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=100, help="Episodes per iteration")
    parser.add_argument("--threshold", type=float, default=2.0, help="Deviation threshold for correction")
    parser.add_argument("--init-data", type=str, default="data/pd_stage1_diverse.npz",
                        help="Initial training data")
    parser.add_argument("--save", type=str, default="models/dagger_bc.pt", help="Save path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load initial data
    init_data = np.load(args.init_data)
    init_states = init_data["states"]
    init_actions = init_data["actions"]
    print(f"Loaded initial data: {len(init_states)} transitions")

    # Initialize model
    model = SimpleBCController(state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM,
                               hidden_dims=[256, 256, 128]).to(device)

    # Compute normalization from initial data
    state_mean = init_states.mean(axis=0)
    state_std = init_states.std(axis=0) + 1e-8
    action_mean = init_actions.mean(axis=0)
    action_std = init_actions.std(axis=0) + 1e-8

    # Normalize states
    init_states_norm = (init_states - state_mean) / state_std

    # Train initial model
    print("\n=== Training initial BC ===")
    model = train_bc(model, init_states_norm, init_actions, epochs=30, device=device)

    # Initialize BC controller
    bc = BCController(model, state_mean, state_std, device)

    # Evaluate initial
    env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
    success, survival = evaluate_bc(env, bc, n_episodes=20)
    print(f"Initial BC: success={success:.1%}, survival={survival:.1%}")
    env.close()

    # DAgger iterations
    cumulative_states = [init_states_norm]
    cumulative_actions = [init_actions]

    for iteration in range(args.iterations):
        print(f"\n=== DAgger Iteration {iteration + 1}/{args.iterations} ===")

        # Collect corrections with current BC
        env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
        new_states, new_actions = collect_dagger_data(
            env, bc, n_episodes=args.episodes_per_iter, threshold=args.threshold
        )
        env.close()

        # Normalize new states
        new_states_norm = (new_states - state_mean) / state_std

        # Add to cumulative data
        cumulative_states.append(new_states_norm)
        cumulative_actions.append(new_actions)
        all_states = np.concatenate(cumulative_states)
        all_actions = np.concatenate(cumulative_actions)

        print(f"  Total training data: {len(all_states)} transitions")

        # Retrain on combined data
        model = SimpleBCController(state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM,
                                   hidden_dims=[256, 256, 128]).to(device)
        print("  Retraining BC...")
        model = train_bc(model, all_states, all_actions, epochs=20, device=device)
        bc = BCController(model, state_mean, state_std, device)

        # Evaluate
        env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
        success, survival = evaluate_bc(env, bc, n_episodes=20)
        print(f"  After iteration {iteration + 1}: success={success:.1%}, survival={survival:.1%}")
        env.close()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
    success, survival = evaluate_bc(env, bc, n_episodes=50)
    print(f"Final BC: success={success:.1%}, survival={survival:.1%}")
    env.close()

    # Save final model
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
    }, args.save)
    print(f"Saved to {args.save}")


if __name__ == "__main__":
    main()