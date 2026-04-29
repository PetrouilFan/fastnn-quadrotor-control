#!/usr/bin/env python3
"""
Error-Dynamics-Aware BC for Quadrotor Control

Key insight: The BC drifts because it doesn't know if tracking is improving or degrading.
This adds explicit error dynamics features:
- Error derivative (d_error/dt) - is error growing or shrinking?
- Error second derivative (d²_error/dt²) - is the rate of change accelerating?
- Cumulative error - integral of error over time

These features explicitly tell the network "I'm drifting" vs "I'm stable",
so it can learn more aggressive corrections when needed.

Usage:
    python train_error_dynamics.py --epochs 30
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
from fastnn_quadrotor.utils.simple_bc import SimpleBCController
from fastnn_quadrotor.training.train_transformer_bc import BCController, train_bc, evaluate


DEPLOYABLE_DIM = 52
ACTION_DIM = 4


class ErrorDynamicsBC(nn.Module):
    """
    BC with explicit error dynamics features.

    Input: 52-dim state + 9 additional error dynamics features
    Total: 61-dim input

    Additional features:
    - pos_error_magnitude: ||pos_err||
    - vel_error_magnitude: ||vel_err||
    - error_change_rate: d(pos_error_magnitude)/dt
    - cumulative_pos_error: integral of pos_error over time
    - lin_accel_magnitude: ||lin_accel|| (indicates disturbance)
    - body_rate_magnitude: ||ang_vel||
    """

    def __init__(self, state_dim=62, action_dim=4, hidden_dims=[256, 256, 128]):
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
        return self.net(x)


class ErrorDynamicsExtractor:
    """Extract error dynamics features from state."""

    def __init__(self):
        self.prev_pos_err = None
        self.prev_vel_err = None
        self.cumulative_pos_err = np.zeros(3)
        self.step_count = 0

    def reset(self):
        self.prev_pos_err = None
        self.prev_vel_err = None
        self.cumulative_pos_err = np.zeros(3)
        self.step_count = 0

    def extract(self, state):
        """
        Extract error dynamics from 52-dim state.

        State layout:
        - 0:3 pos_err
        - 3:6 vel_err
        - 6:9 att_err
        - 9:12 rate_err
        - 12:15 lin_accel
        - 15:24 rotmat
        - 24:27 ang_vel
        - 27:43 action_hist (16)
        - 43:47 error_integrals
        - 47:51 rotor_thrust
        - 51 mass_est
        """
        pos_err = state[0:3]
        vel_err = state[3:6]
        lin_accel = state[12:15]
        ang_vel = state[24:27]

        # Basic magnitudes
        pos_err_mag = np.linalg.norm(pos_err)
        vel_err_mag = np.linalg.norm(vel_err)
        lin_accel_mag = np.linalg.norm(lin_accel)
        ang_vel_mag = np.linalg.norm(ang_vel)

        # Error dynamics
        if self.prev_pos_err is not None:
            pos_err_change = pos_err - self.prev_pos_err
            pos_err_change_rate = np.linalg.norm(pos_err_change)

            if self.prev_vel_err is not None:
                vel_err_change = vel_err - self.prev_vel_err
                vel_err_change_rate = np.linalg.norm(vel_err_change)
            else:
                vel_err_change_rate = 0.0
        else:
            pos_err_change_rate = 0.0
            vel_err_change_rate = 0.0

        # Update cumulative error (integral)
        self.cumulative_pos_err += pos_err * 0.01  # scaled dt
        cumulative_pos_mag = np.linalg.norm(self.cumulative_pos_err)

        # Update for next step
        self.prev_pos_err = pos_err.copy()
        self.prev_vel_err = vel_err.copy()
        self.step_count += 1

        # Additional features
        additional = np.array([
            pos_err_mag,                      # 1
            vel_err_mag,                      # 2
            pos_err_change_rate,               # 3 - d_error/dt
            vel_err_change_rate,               # 4 - d_velocity_error/dt
            cumulative_pos_mag,               # 5 - integral of error
            lin_accel_mag,                    # 6 - disturbance indicator
            ang_vel_mag,                      # 7 - angular instability
            pos_err[0], pos_err[1], pos_err[2],  # 8-10 - direction
        ], dtype=np.float32)

        return np.concatenate([state, additional])


class EDController(BCController):
    """Error Dynamics BC Controller with state augmentation."""

    def __init__(self, model, state_mean, state_std, device="cuda"):
        self.model = model
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device
        self.ed_extractor = ErrorDynamicsExtractor()

        # Adjust normalization to account for additional features
        # We'll use learned LayerNorm instead of running stats
        self.layer_norm = nn.LayerNorm(62).to(device)

    def reset(self):
        self.ed_extractor.reset()

    def predict(self, obs):
        with torch.no_grad():
            state = obs[:DEPLOYABLE_DIM]

            # Extract error dynamics features
            ed_state = self.ed_extractor.extract(state)
            state_raw = torch.FloatTensor(ed_state).to(self.device)

            # Normalize with LayerNorm
            state_norm = self.layer_norm(state_raw.unsqueeze(0)).squeeze(0)

            action = self.model(state_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
            return action

    def to_env_action(self, raw_action):
        """Convert raw model output to normalized action space."""
        normalized = np.zeros(4)
        normalized[0] = np.clip((raw_action[0] - 10.0) / 10.0, -1.0, 1.0)
        normalized[1] = np.clip(raw_action[1] / 3.0, -1.0, 1.0)
        normalized[2] = np.clip(raw_action[2] / 3.0, -1.0, 1.0)
        normalized[3] = np.clip(raw_action[3] / 2.0, -1.0, 1.0)
        return normalized


def collect_ed_data(n_episodes=100, stage=3):
    """Collect data with error dynamics features."""
    ed_extractor = ErrorDynamicsExtractor()
    states, actions = [], []

    env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=False)

    for ep in range(n_episodes):
        ed_extractor.reset()
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            pd_action = env._cascaded_controller()
            state_with_ed = ed_extractor.extract(obs[:DEPLOYABLE_DIM])

            states.append(state_with_ed)
            actions.append(pd_action.copy())

            obs, _, terminated, truncated, _ = env.step(np.zeros(4))

    env.close()
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32)


def train_ed_bc(model, states, actions, epochs=30, batch_size=256, lr=1e-3, device="cuda"):
    """Train error dynamics BC."""
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

            # Apply layer norm
            preds = model(batch_states)
            loss = nn.MSELoss()(preds, batch_actions)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch+1}: loss={avg_loss:.6f}")

    return model


def main():
    print("=== Error Dynamics BC Experiments ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Collect data from multiple stages
    print("Collecting data...")
    all_states, all_actions = [], []

    for stage in [1, 2, 3]:
        states, actions = collect_ed_data(n_episodes=100, stage=stage)
        all_states.append(states)
        all_actions.append(actions)
        print(f"  Stage {stage}: {len(states)} transitions, state shape {states.shape}")

    all_states = np.concatenate(all_states)
    all_actions = np.concatenate(all_actions)
    print(f"Total: {len(all_states)} transitions, state shape {all_states.shape}")

    # Compare with baseline (no error dynamics)
    print("\n=== Baseline (no error dynamics) ===")
    baseline_data = np.load('data/pd_stage1_diverse.npz')
    baseline_states = baseline_data['states'][:len(all_states)]
    baseline_actions = baseline_data['actions'][:len(all_states)]

    baseline_model = SimpleBCController(state_dim=52, action_dim=4, hidden_dims=[256, 256, 128]).to(device)
    baseline_model = train_ed_bc(baseline_model, baseline_states, baseline_actions, epochs=30, device=device, lr=1e-3)

    # Evaluate baseline
    class BaselineBC(BCController):
        def __init__(self, model, state_mean, state_std, device):
            super().__init__(model, state_mean, state_std, device, model_type='mlp')

    baseline_mean = baseline_states.mean(axis=0)
    baseline_std = baseline_states.std(axis=0) + 1e-8
    baseline_bc = BaselineBC(baseline_model, baseline_mean, baseline_std, device)

    print("\nBaseline evaluation:")
    for stage in [1, 3, 4]:
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        metrics = evaluate(env, baseline_bc, n_episodes=50)
        print(f"  Stage {stage}: success={metrics['success_rate']:.1%}, dist={metrics['mean_final_dist']:.3f}m")
        env.close()

    # Error Dynamics model
    print("\n=== Error Dynamics BC ===")

    ed_model = ErrorDynamicsBC(state_dim=62, action_dim=4, hidden_dims=[256, 256, 128]).to(device)
    print(f"Parameters: {sum(p.numel() for p in ed_model.parameters()):,}")

    ed_model = train_ed_bc(ed_model, all_states, all_actions, epochs=30, device=device)

    # Evaluate ED model
    ed_bc = EDController(ed_model, np.zeros(61), np.ones(61), device)

    print("\nError Dynamics evaluation:")
    for stage in [1, 3, 4]:
        ed_bc.reset()
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        metrics = evaluate(env, ed_bc, n_episodes=50)
        print(f"  Stage {stage}: success={metrics['success_rate']:.1%}, dist={metrics['mean_final_dist']:.3f}m")
        env.close()

    # Save model
    torch.save({
        'model_state_dict': ed_model.state_dict(),
        'state_dim': 62,
        'config': 'error_dynamics_bc',
    }, 'models/ed_bc.pt')
    print("\nSaved to models/ed_bc.pt")


if __name__ == "__main__":
    main()