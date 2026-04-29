#!/usr/bin/env python3
"""
Force Observer BC for Quadrotor Wind Rejection

Key insight: Wind is a force disturbance that we can estimate from IMU data.
Instead of trying to predict wind from position/velocity errors (which are
effects, not causes), we estimate the wind force directly.

Force estimation:
- Newton's 2nd law: F_net = m * a
- F_net = F_thrust + F_gravity + F_wind
- F_wind = m * a - F_thrust + F_gravity

With F_wind estimated, we can:
1. Use it as additional input to the controller
2. Apply explicit wind compensation: correction = -F_wind

Architecture:
- MLP that takes 52-dim state + 3-dim estimated wind force = 55-dim input
- Output: motor command

Usage:
    python train_force_obs.py --epochs 50
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from env_rma import RMAQuadrotorEnv
from simple_bc import SimpleBCController
from train_transformer_bc import BCController, train_bc, evaluate


DEPLOYABLE_DIM = 52
ACTION_DIM = 4


class ForceObserver:
    """
    Estimates external force on the quadrotor from IMU + dynamics.

    F_wind = m * a_imu - F_thrust + F_gravity

    Uses filtered estimates to reduce noise.
    """

    def __init__(self, nominal_mass=0.032, gravity=9.81):
        self.nominal_mass = nominal_mass
        self.gravity = gravity
        self.mass_estimate = nominal_mass

        # Filtered estimates
        self.filtered_wind = np.zeros(3)
        self.filter_alpha = 0.3  # Low-pass filter for noise reduction

        # Previous state for differentiation
        self.prev_lin_accel = None
        self.prev_time = None

    def reset(self):
        self.filtered_wind = np.zeros(3)
        self.prev_lin_accel = None
        self.prev_time = None

    def estimate(self, lin_accel, thrust_per_rotor, rotor_directions, dt=0.01):
        """
        Estimate external wind force.

        Args:
            lin_accel: IMU linear acceleration (m/s^2)
            thrust_per_rotor: thrust per rotor (N)
            rotor_directions: unit vectors for thrust directions
            dt: time step

        Returns:
            estimated_wind_force (3D)
        """
        # Total thrust in body frame
        total_thrust = thrust_per_rotor * np.sum(rotor_directions, axis=0)

        # Gravity in world frame (we need to account for rotation)
        # For simplicity, approximate as -z direction
        gravity_force = np.array([0, 0, -self.gravity * self.mass_estimate])

        # Net force from acceleration
        # Note: lin_accel from IMU is in body frame
        # For wind estimation, we need to account for body rotation

        # Simplified: use vertical force balance for thrust estimation
        # Then estimate horizontal wind from horizontal acceleration

        # Vertical: T ≈ m * g + m * a_z  (approximately)
        # Horizontal: F_wind_x ≈ m * a_x (in world frame, but IMU gives body frame)

        # Use previous estimate as prior, then update with measurement
        measured_wind = np.zeros(3)

        # The IMU lin_accel includes gravity, so we subtract it
        # This is a simplified estimation
        measured_wind = self.mass_estimate * lin_accel - gravity_force

        # Low-pass filter to reduce noise
        self.filtered_wind = (
            self.filter_alpha * measured_wind +
            (1 - self.filter_alpha) * self.filtered_wind
        )

        return self.filtered_wind


class ForceObsBC(nn.Module):
    """BC with force observer input."""

    def __init__(self, state_dim=55, action_dim=4, hidden_dims=[256, 256, 128]):
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


class ForceObsController(BCController):
    """BC with force observer."""

    def __init__(self, model, state_mean, state_std, device="cuda"):
        super().__init__(model, state_mean, state_std, device, model_type="mlp")
        self.force_obs = ForceObserver()

    def reset(self):
        self.force_obs.reset()

    def predict(self, obs):
        with torch.no_grad():
            state = obs[:DEPLOYABLE_DIM]

            # Estimate wind force
            lin_accel = state[12:15]  # IMU linear acceleration
            rotor_thrust = state[47:51]  # rotor thrust estimates
            # Simplified rotor directions (assuming standard quad config)
            rotor_dirs = np.array([
                [0.707, 0.707, 0],   # front-right
                [-0.707, 0.707, 0],  # front-left
                [0.707, -0.707, 0],  # rear-right
                [-0.707, -0.707, 0], # rear-left
            ])
            avg_thrust = rotor_thrust.mean()
            wind_force = self.force_obs.estimate(lin_accel, avg_thrust, rotor_dirs)

            # Concatenate wind force to state
            state_with_wind = np.concatenate([state, wind_force])

            state_raw = torch.FloatTensor(state_with_wind).to(self.device)
            state_norm = (state_raw - torch.FloatTensor(self.state_mean).to(self.device)) / \
                         (torch.FloatTensor(self.state_std).to(self.device) + 1e-8)

            action = self.model(state_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
            return action


def collect_force_obs_data(n_episodes=100, stage=3):
    """Collect data with force observer estimates."""
    force_obs = ForceObserver()
    states, actions, wind_forces = [], [], []

    env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=False)

    for ep in range(n_episodes):
        force_obs.reset()
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            pd_action = env._cascaded_controller()

            # Get state components
            lin_accel = obs[12:15]
            rotor_thrust = obs[47:51]
            rotor_dirs = np.array([
                [0.707, 0.707, 0],
                [-0.707, 0.707, 0],
                [0.707, -0.707, 0],
                [-0.707, -0.707, 0],
            ])
            avg_thrust = rotor_thrust.mean()
            wind_force = force_obs.estimate(lin_accel, avg_thrust, rotor_dirs)

            states.append(obs[:DEPLOYABLE_DIM].copy())
            actions.append(pd_action.copy())
            wind_forces.append(wind_force.copy())

            obs, _, terminated, truncated, _ = env.step(np.zeros(4))

    env.close()
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(wind_forces, dtype=np.float32)


def train_force_obs_bc(states, wind_forces, actions, epochs=30, batch_size=256, lr=1e-3, device="cuda"):
    """Train BC with force observer input."""
    # Combine state + wind force
    combined_states = np.concatenate([states, wind_forces], axis=1)

    # Normalize
    state_mean = combined_states.mean(axis=0)
    state_std = combined_states.std(axis=0) + 1e-8
    combined_norm = (combined_states - state_mean) / state_std

    # Model
    model = ForceObsBC(state_dim=55, action_dim=4, hidden_dims=[256, 256, 128]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(torch.FloatTensor(combined_norm), torch.FloatTensor(actions))
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

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss / n_batches:.6f}")

    return model, state_mean, state_std


def main():
    print("=== Force Observer BC Experiments ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Collect data from multiple stages
    print("Collecting data...")
    all_states, all_actions, all_wind = [], [], []

    for stage in [1, 2, 3]:
        states, actions, wind = collect_force_obs_data(n_episodes=100, stage=stage)
        all_states.append(states)
        all_actions.append(actions)
        all_wind.append(wind)
        print(f"  Stage {stage}: {len(states)} transitions, wind mean={wind.mean(axis=0)}")

    all_states = np.concatenate(all_states)
    all_actions = np.concatenate(all_actions)
    all_wind = np.concatenate(all_wind)
    print(f"Total: {len(all_states)} transitions")

    # Baseline (no force observer)
    print("\n=== Baseline (no force observer) ===")
    baseline_data = np.load('data/pd_stage1_diverse.npz')
    baseline_states = baseline_data['states'][:len(all_states)]
    baseline_actions = baseline_data['actions'][:len(all_states)]

    baseline_mean = baseline_states.mean(axis=0)
    baseline_std = baseline_states.std(axis=0) + 1e-8
    baseline_states_norm = (baseline_states - baseline_mean) / baseline_std

    baseline_model = SimpleBCController(state_dim=52, action_dim=4, hidden_dims=[256, 256, 128]).to(device)
    baseline_model = train_bc(baseline_model, baseline_states_norm, baseline_actions, epochs=30,
                              device=device, model_type="mlp", verbose=False)

    baseline_bc = BCController(baseline_model, baseline_mean, baseline_std, device, model_type="mlp")

    print("Baseline evaluation:")
    for stage in [1, 3, 4]:
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        metrics = evaluate(env, baseline_bc, n_episodes=50)
        print(f"  Stage {stage}: success={metrics['success_rate']:.1%}, dist={metrics['mean_final_dist']:.3f}m")
        env.close()

    # Force Observer BC
    print("\n=== Force Observer BC ===")

    model, state_mean, state_std = train_force_obs_bc(
        all_states, all_wind, all_actions, epochs=30, device=device
    )

    # Evaluate
    fo_bc = ForceObsController(model, state_mean, state_std, device)

    print("\nForce Observer evaluation:")
    for stage in [1, 3, 4]:
        fo_bc.reset()
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        metrics = evaluate(env, fo_bc, n_episodes=50)
        print(f"  Stage {stage}: success={metrics['success_rate']:.1%}, dist={metrics['mean_final_dist']:.3f}m")
        env.close()

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'state_mean': state_mean,
        'state_std': state_std,
        'config': 'force_obs_bc',
    }, 'models/force_obs_bc.pt')
    print("\nSaved to models/force_obs_bc.pt")


if __name__ == "__main__":
    main()