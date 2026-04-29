#!/usr/bin/env python3
"""
Training script for Transformer/GRU Behavioral Cloning with DAgger refinement.

This builds on the DAgger approach but uses temporal models (Transformer or GRU)
instead of MLPs. The key improvement: temporal models learn the feedback loop
that BC drifts without.

Usage:
    # Train transformer BC
    python train_transformer_bc.py --model transformer --data data/pd_stage1_diverse.npz

    # Train GRU BC
    python train_transformer_bc.py --model gru --data data/pd_stage1_diverse.npz

    # Run DAgger refinement
    python train_transformer_bc.py --model transformer --dagger --iterations 3

    # Evaluate on multiple stages
    python train_transformer_bc.py --model transformer --evaluate --stages 1,2,3,4
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json
from datetime import datetime

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
from fastnn_quadrotor.utils.transformer_bc import TransformerController, GRUController, EnsembleController, count_parameters


DEPLOYABLE_DIM = 51  # 52 - 1 (mass_est removed for sim-to-real safe deployment)
ACTION_DIM = 4


class BCController:
    """Wraps BC model for environment interaction."""

    def __init__(self, model, state_mean, state_std, device="cuda", model_type="transformer"):
        self.model = model
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device
        self.model_type = model_type

    def reset(self, batch_size=1):
        """Reset model state at episode start."""
        if self.model_type in ['transformer', 'ensemble']:
            self.model.reset(batch_size=batch_size, device=torch.device(self.device))
        elif self.model_type == 'gru':
            self.hidden = self.model.reset(batch_size=batch_size, device=torch.device(self.device))
        else:
            self.hidden = None

    def predict(self, obs):
        with torch.no_grad():
            state_raw = torch.FloatTensor(obs[:DEPLOYABLE_DIM]).to(self.device)
            state_norm = (state_raw - torch.FloatTensor(self.state_mean).to(self.device)) / \
                         (torch.FloatTensor(self.state_std).to(self.device) + 1e-8)

            if self.model_type == 'gru':
                action, self.hidden = self.model(state_norm.unsqueeze(0), self.hidden)
            elif self.model_type == 'transformer':
                action, _ = self.model(state_norm.unsqueeze(0))
            else:
                # MLP returns just action
                action = self.model(state_norm.unsqueeze(0))
                if isinstance(action, tuple):
                    action = action[0]

            return action.squeeze(0).cpu().numpy()

    def to_env_action(self, raw_action):
        """Convert raw model output to normalized action space."""
        normalized = np.zeros(4)
        normalized[0] = np.clip((raw_action[0] - 10.0) / 10.0, -1.0, 1.0)
        normalized[1] = np.clip(raw_action[1] / 3.0, -1.0, 1.0)
        normalized[2] = np.clip(raw_action[2] / 3.0, -1.0, 1.0)
        normalized[3] = np.clip(raw_action[3] / 2.0, -1.0, 1.0)
        return normalized


def run_episode(env, bc_controller, use_bc=True, threshold=1.0, max_steps=500, record_trajectory=False):
    """Run one episode with BC policy.

    Returns trajectory data for DAgger and evaluation metrics.
    """
    bc_controller.reset()
    states = []
    actions = []
    bc_actions_taken = []
    rewards = []

    obs, _ = env.reset()
    terminated = False
    truncated = False
    steps = 0

    while not (terminated or truncated) and steps < max_steps:
        bc_raw = bc_controller.predict(obs)
        pd_action = env._cascaded_controller()

        deviation = np.linalg.norm(bc_raw - pd_action)
        use_pd = deviation > threshold

        states.append(obs[:DEPLOYABLE_DIM].copy())
        actions.append(pd_action.copy())

        if use_pd:
            bc_actions_taken.append(pd_action.copy())
            action_for_env = bc_controller.to_env_action(pd_action)
        else:
            bc_actions_taken.append(bc_raw.copy())
            action_for_env = bc_controller.to_env_action(bc_raw)

        obs, reward, terminated, truncated, info = env.step(action_for_env)
        rewards.append(reward)
        steps += 1

    return {
        'states': states,
        'actions': actions,
        'bc_actions': bc_actions_taken,
        'rewards': rewards,
        'steps': steps,
        'terminated': terminated,
        'truncated': truncated,
        'final_dist': np.linalg.norm(env.data.qpos[:3] - env.target_pos) if hasattr(env, 'target_pos') else 0,
    }


def collect_dagger_data(env, bc_controller, n_episodes, threshold=1.0):
    """Collect correction data using DAgger."""
    all_states = []
    all_actions = []
    corrections = 0

    for ep in range(n_episodes):
        result = run_episode(env, bc_controller, threshold=threshold)
        all_states.extend(result['states'])
        all_actions.extend(result['actions'])

        for bc_act, pd_act in zip(result['bc_actions'], result['actions']):
            if not np.allclose(bc_act, pd_act, atol=0.1):
                corrections += 1

    print(f"  Collected {len(all_states)} transitions, {corrections} corrections "
          f"({100*corrections/len(all_states):.1f}%)")

    return np.array(all_states, dtype=np.float32), np.array(all_actions, dtype=np.float32)


def train_bc(model, states, actions, epochs=20, batch_size=256, lr=1e-3, device="cuda",
             model_type="transformer", verbose=True):
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

            if model_type == 'transformer':
                # Reshape to (batch, seq=1, features) for single-step training
                preds = model.forward_sequence(batch_states.unsqueeze(1))
                # Predict action at each step, target is the action
                loss = nn.MSELoss()(preds[:, -1], batch_actions)  # Only last step prediction
            elif model_type == 'gru':
                preds, _ = model(batch_states)
                loss = nn.MSELoss()(preds, batch_actions)
            else:
                preds = model(batch_states)
                loss = nn.MSELoss()(preds, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if verbose and (epoch + 1) % 5 == 0:
            avg_loss = total_loss / n_batches
            print(f"    Epoch {epoch+1}: loss={avg_loss:.6f}")

    return model


def evaluate(env, bc_controller, n_episodes=50, max_steps=500, verbose=False):
    """Evaluate BC policy with detailed metrics."""
    successes = 0
    survivals = 0
    total_steps = 0
    total_rewards = []
    final_distances = []

    for ep in range(n_episodes):
        result = run_episode(env, bc_controller, max_steps=max_steps)
        total_steps += result['steps']
        total_rewards.append(sum(result['rewards']))
        final_distances.append(result['final_dist'])

        if result['steps'] >= max_steps:
            successes += 1
            survivals += 1
        elif not result['terminated']:
            survivals += 1

        if verbose and (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{n_episodes}: steps={result['steps']}, "
                  f"return={sum(result['rewards']):.1f}, dist={result['final_dist']:.3f}")

    metrics = {
        'success_rate': successes / n_episodes,
        'survival_rate': survivals / n_episodes,
        'mean_steps': total_steps / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_final_dist': np.mean(final_distances),
        'std_final_dist': np.std(final_distances),
    }

    return metrics


def evaluate_multi_stage(controller_class, model_path, state_mean, state_std,
                         stages=[1, 2, 3, 4], n_episodes=50, device='cuda', model_type='transformer'):
    """Evaluate on multiple curriculum stages."""
    results = {}

    for stage in stages:
        print(f"\n=== Stage {stage} Evaluation ===")
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)

        # Create fresh model and load weights
        if model_type == 'transformer':
            model = TransformerController(state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM,
                                        d_model=128, n_layers=3)
        elif model_type == 'gru':
            model = GRUController(state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM, hidden_dim=128)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)

        bc = BCController(model, state_mean, state_std, device, model_type=model_type)

        metrics = evaluate(env, bc, n_episodes=n_episodes)

        print(f"  Success: {metrics['success_rate']:.1%}, Survival: {metrics['survival_rate']:.1%}, "
              f"Mean Steps: {metrics['mean_steps']:.0f}, Mean Dist: {metrics['mean_final_dist']:.3f}m")

        results[f'stage_{stage}'] = metrics
        env.close()

    return results


def main():
    parser = argparse.ArgumentParser(description="Train Transformer/GRU BC with DAgger")
    parser.add_argument("--model", type=str, default="transformer",
                        choices=["transformer", "gru", "ensemble"],
                        help="Model architecture")
    parser.add_argument("--data", type=str, default="data/pd_stage1_diverse.npz",
                        help="Initial training data")
    parser.add_argument("--save", type=str, default="models/transformer_bc.pt",
                        help="Save path")
    parser.add_argument("--dagger", action="store_true", help="Run DAgger refinement")
    parser.add_argument("--iterations", type=int, default=5, help="DAgger iterations")
    parser.add_argument("--episodes-per-iter", type=int, default=100, help="Episodes per DAgger iteration")
    parser.add_argument("--threshold", type=float, default=2.0, help="Deviation threshold for correction")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--evaluate", action="store_true", help="Only run evaluation")
    parser.add_argument("--stages", type=str, default="1,2,3,4", help="Stages to evaluate")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Episodes per evaluation")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    # Create model
    if args.model == 'transformer':
        model = TransformerController(
            state_dim=DEPLOYABLE_DIM,
            action_dim=ACTION_DIM,
            d_model=128,
            n_layers=3,
            n_heads=1,
            ff_dim=512,
            context_len=20,
        )
    elif args.model == 'gru':
        model = GRUController(
            state_dim=DEPLOYABLE_DIM,
            action_dim=ACTION_DIM,
            hidden_dim=128,
            num_layers=2,
        )
    elif args.model == 'ensemble':
        model = EnsembleController(
            n_models=3,
            model_type='transformer',
            state_dim=DEPLOYABLE_DIM,
            action_dim=ACTION_DIM,
            d_model=128,
            n_layers=3,
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # Evaluation mode
    if args.evaluate:
        if not os.path.exists(args.save):
            print(f"Error: Model not found at {args.save}")
            return

        checkpoint = torch.load(args.save, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        state_mean = checkpoint['state_mean']
        state_std = checkpoint['state_std']

        bc = BCController(model, state_mean, state_std, device, model_type=args.model)

        stages = [int(s) for s in args.stages.split(',')]
        results = evaluate_multi_stage(
            type(model), args.save, state_mean, state_std,
            stages=stages, n_episodes=args.eval_episodes, device=device, model_type=args.model
        )

        # Save results
        results_path = args.save.replace('.pt', '_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        return

    # Training mode
    if not os.path.exists(args.data):
        print(f"Error: Data not found at {args.data}")
        print("Run 'python collect_pd_data.py --episodes 500 --save data/pd_stage1_diverse.npz' first")
        return

    # Load data
    data = np.load(args.data)
    init_states = data["states"]
    init_actions = data["actions"]
    print(f"Loaded data: {len(init_states)} transitions")
    print(f"State shape: {init_states.shape}, Action shape: {init_actions.shape}")

    # Normalization
    state_mean = init_states.mean(axis=0)
    state_std = init_states.std(axis=0) + 1e-8
    init_states_norm = (init_states - state_mean) / state_std

    # Initial evaluation
    print("\n=== Initial Evaluation (random weights) ===")
    env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
    bc = BCController(model, state_mean, state_std, device, model_type=args.model)
    metrics = evaluate(env, bc, n_episodes=20)
    print(f"Random init: success={metrics['success_rate']:.1%}, survival={metrics['survival_rate']:.1%}")
    env.close()

    # Initial BC training
    print(f"\n=== Initial BC Training ({args.epochs} epochs) ===")
    model = train_bc(
        model, init_states_norm, init_actions,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=device, model_type=args.model
    )

    # Evaluate after initial training
    bc = BCController(model, state_mean, state_std, device, model_type=args.model)
    env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
    metrics = evaluate(env, bc, n_episodes=50)
    print(f"After initial BC: success={metrics['success_rate']:.1%}, survival={metrics['survival_rate']:.1%}, "
          f"mean_dist={metrics['mean_final_dist']:.3f}m")
    env.close()

    # DAgger refinement
    if args.dagger:
        print(f"\n=== DAgger Refinement ({args.iterations} iterations) ===")
        cumulative_states = [init_states_norm]
        cumulative_actions = [init_actions]

        for iteration in range(args.iterations):
            print(f"\n--- DAgger Iteration {iteration + 1}/{args.iterations} ---")

            # Collect corrections
            env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
            new_states, new_actions = collect_dagger_data(
                env, bc, n_episodes=args.episodes_per_iter, threshold=args.threshold
            )
            env.close()

            if len(new_states) == 0:
                print("  No new data collected, stopping DAgger")
                break

            # Normalize new states
            new_states_norm = (new_states - state_mean) / state_std

            # Add to cumulative
            cumulative_states.append(new_states_norm)
            cumulative_actions.append(new_actions)
            all_states = np.concatenate(cumulative_states)
            all_actions = np.concatenate(cumulative_actions)
            print(f"  Total data: {len(all_states)} transitions")

            # Retrain
            if args.model == 'transformer':
                model = TransformerController(state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM,
                                             d_model=128, n_layers=3)
            elif args.model == 'gru':
                model = GRUController(state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM, hidden_dim=128)
            else:
                model = EnsembleController(n_models=3, model_type='transformer',
                                          state_dim=DEPLOYABLE_DIM, action_dim=ACTION_DIM)
            model = model.to(device)

            model = train_bc(model, all_states, all_actions, epochs=20,
                           batch_size=args.batch_size, lr=args.lr, device=device, model_type=args.model)
            bc = BCController(model, state_mean, state_std, device, model_type=args.model)

            # Evaluate
            env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
            metrics = evaluate(env, bc, n_episodes=50)
            print(f"  After iter {iteration+1}: success={metrics['success_rate']:.1%}, "
                  f"survival={metrics['survival_rate']:.1%}, mean_dist={metrics['mean_final_dist']:.3f}m")
            env.close()

    # Final evaluation on all stages
    print("\n=== Final Evaluation ===")
    stages = [int(s) for s in args.stages.split(',')]
    final_results = evaluate_multi_stage(
        type(model), None, state_mean, state_std,
        stages=stages, n_episodes=args.eval_episodes, device=device, model_type=args.model
    )

    # Save model
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "state_mean": state_mean,
        "state_std": state_std,
        "model_type": args.model,
        "config": {
            "state_dim": DEPLOYABLE_DIM,
            "action_dim": ACTION_DIM,
            "d_model": 128,
            "n_layers": 3,
        }
    }, args.save)
    print(f"\nModel saved to {args.save}")

    # Save results
    results_path = args.save.replace('.pt', '_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
