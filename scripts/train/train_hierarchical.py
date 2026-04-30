#!/usr/bin/env python3
"""
Hierarchical Architecture: DRL Outer Loop + PD Inner Loop

Key insight from GustPilot paper:
- Inner loop (PD/INDI): handles fast attitude stabilization
- Outer loop (DRL): provides velocity/position setpoints to inner loop

The DRL learns to output velocity corrections that help the PD
track the target despite disturbances.

Implementation:
1. DRL outputs: [vx_correction, vy_correction, vz_correction, vyaw]
2. These are added to the PD's velocity targets
3. PD inner loop tracks the modified targets

This is different from residual RL because:
- Residual: motor_cmd = PD_motor_cmd + DRL_residual
- Hierarchical: velocity_target = PD_velocity_target + DRL_correction → PD tracks new target

Usage:
    python train_hierarchical.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
from fastnn_quadrotor.utils.transformer_bc import GRUController
from fastnn_quadrotor.training.train_transformer_bc import evaluate


DEPLOYABLE_DIM = 52
ACTION_DIM = 4


class HierarchicalController:
    """
    Hierarchical controller: DRL outer + PD inner.

    DRL outputs velocity corrections that are added to the PD's velocity targets.
    PD inner loop tracks the modified targets.
    """

    def __init__(self, model, state_mean, state_std, device='cuda'):
        self.model = model
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device
        self.hidden = None

        # PD gains from env
        self.outer_gains = np.array([3.0, 3.0, 4.0, 1.5])
        self.inner_gains = np.array([10.0, 10.0, 2.5])
        self.rate_damping = 6.0
        self.mass_estimate = 0.032
        self.gravity = 9.81

    def reset(self):
        self.hidden = None
        self.pos_integral = np.zeros(4)

    def predict(self, obs):
        """Predict velocity corrections from state."""
        with torch.no_grad():
            state_raw = torch.FloatTensor(obs[:DEPLOYABLE_DIM]).to(self.device)
            state_norm = (state_raw - torch.FloatTensor(self.state_mean).to(self.device)) / \
                         (torch.FloatTensor(self.state_std).to(self.device) + 1e-8)

            vel_correction, self.hidden = self.model(state_norm.unsqueeze(0), self.hidden)
            return vel_correction.squeeze(0).cpu().numpy()

    def pd_controller(self, obs, vel_targets):
        """
        PD inner loop that tracks velocity targets.

        Args:
            obs: environment observation
            vel_targets: [vx, vy, vz, vyaw] desired velocities

        Returns:
            motor_commands: [thrust, roll_torque, pitch_torque, yaw_torque]
        """
        pos = obs[:3]
        vel = obs[3:6]
        ang_vel = obs[24:27]

        # Hover thrust
        hover = self.mass_estimate * self.gravity

        # Current yaw
        quat = obs[15:19]  # rotation quaternion
        rpy = self._quat_to_rpy(quat)
        current_yaw = rpy[2]

        # Desired attitude from velocity targets
        # X-velocity → pitch angle
        # Y-velocity → roll angle
        desired_pitch = np.arctan2(vel_targets[0], hover) if abs(hover) > 0.1 else 0
        desired_roll = np.arctan2(-vel_targets[1], hover) if abs(hover) > 0.1 else 0
        desired_yaw = current_yaw + vel_targets[3] * 0.01  # integrate yaw rate

        # Attitude errors
        current_roll = rpy[0]
        current_pitch = rpy[1]
        roll_error = desired_roll - current_roll
        pitch_error = desired_pitch - current_pitch
        yaw_error = desired_yaw - current_yaw

        # Rate commands from attitude errors
        desired_roll_rate = self.inner_gains[0] * roll_error - self.rate_damping * ang_vel[0]
        desired_pitch_rate = self.inner_gains[1] * pitch_error - self.rate_damping * ang_vel[1]
        desired_yaw_rate = self.inner_gains[2] * yaw_error - self.rate_damping * ang_vel[2]

        # Thrust from velocity error
        vz_error = vel_targets[2] - vel[2]
        desired_thrust_z = hover + self.outer_gains[2] * vz_error

        # Clip
        desired_thrust_z = np.clip(desired_thrust_z, 0, 20)
        roll_torque = np.clip(desired_roll_rate, -5, 5)
        pitch_torque = np.clip(desired_pitch_rate, -5, 5)
        yaw_torque = np.clip(desired_yaw_rate, -2, 2)

        return np.array([desired_thrust_z, roll_torque, pitch_torque, yaw_torque])

    def _quat_to_rpy(self, quat):
        """Convert quaternion to roll, pitch, yaw."""
        w, x, y, z = quat
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def step(self, obs):
        """
        Single step: get DRL correction, apply PD, return motor commands.

        Returns:
            action: normalized action for environment
        """
        # Get velocity correction from DRL
        vel_correction = self.predict(obs)

        # PD velocity targets (hover = zero velocity)
        vel_targets = vel_correction  # DRL learns to output velocity commands

        # PD inner loop
        motor_cmd = self.pd_controller(obs, vel_targets)

        # Normalize to environment action space
        normalized = np.zeros(4)
        normalized[0] = np.clip((motor_cmd[0] - 10.0) / 10.0, -1, 1)
        normalized[1] = np.clip(motor_cmd[1] / 3.0, -1, 1)
        normalized[2] = np.clip(motor_cmd[2] / 3.0, -1, 1)
        normalized[3] = np.clip(motor_cmd[3] / 2.0, -1, 1)

        return normalized


class HierarchicalBC(nn.Module):
    """
    Hierarchical BC: DRL outer loop.

    Outputs velocity corrections that help the PD track targets despite disturbances.
    """

    def __init__(self, state_dim=52, hidden_dim=128, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            state_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # vx, vy, vz, vyaw
            nn.Tanh(),
        )
        self.velocity_scale = 1.0  # max correction in m/s

    def forward(self, state, hidden=None):
        if state.dim() == 2:
            state = state.unsqueeze(1)
        output, hidden = self.gru(state, hidden)
        velocity_correction = self.velocity_head(output[:, -1]) * self.velocity_scale
        return velocity_correction, hidden


def collect_hierarchical_data(n_episodes=100, stage=3):
    """
    Collect data for hierarchical BC.

    For each state, we compute:
    - The velocity correction the DRL should output
    - This is learned by having DRL predict what the PD should be tracking
    """
    states, vel_commands = [], []

    env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=False)

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated = truncated = False

        while not (terminated or truncated):
            # Get PD output (motor commands)
            pd_output = env._cascaded_controller()

            # Extract desired velocity from PD output
            # PD outputs attitude commands, not velocities
            # But we can estimate what velocity the PD was trying to achieve

            # For now, just collect state and use PD output as "what DRL should learn"
            states.append(obs[:DEPLOYABLE_DIM].copy())
            vel_commands.append(np.zeros(4))  # placeholder

            obs, _, terminated, truncated, _ = env.step(np.zeros(4))

    env.close()
    return np.array(states, dtype=np.float32), np.array(vel_commands, dtype=np.float32)


def train_hierarchical_bc(states, vel_commands, epochs=30, batch_size=256, lr=1e-3, device='cuda'):
    """Train hierarchical BC."""
    model = HierarchicalBC(state_dim=52, hidden_dim=128, num_layers=2).to(device)

    # Normalize states
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8
    states_norm = (states - state_mean) / state_std

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(torch.FloatTensor(states_norm), torch.FloatTensor(vel_commands))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for batch_states, batch_vel in loader:
            batch_states = batch_states.to(device)
            batch_vel = batch_vel.to(device)

            vel_pred, _ = model(batch_states)
            loss = nn.MSELoss()(vel_pred, batch_vel)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss / n_batches:.6f}")

    return model, state_mean, state_std


def evaluate_hierarchical(env, controller, n_episodes=50):
    """Evaluate hierarchical controller."""
    successes = 0
    survivals = 0
    final_distances = []

    for ep in range(n_episodes):
        controller.reset()
        obs, _ = env.reset()
        terminated = truncated = False
        steps = 0

        while not (terminated or truncated) and steps < 500:
            action = controller.step(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1

        final_dist = np.linalg.norm(env.data.qpos[:3] - env.target_pos)
        final_distances.append(final_dist)

        if steps >= 500:
            successes += 1
            survivals += 1
        elif not terminated:
            survivals += 1

    return {
        'success_rate': successes / n_episodes,
        'survival_rate': survivals / n_episodes,
        'mean_final_dist': np.mean(final_distances),
    }


def main():
    print("=== Hierarchical Architecture: DRL + PD ===\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Collect data
    print("Collecting data...")
    all_states, all_vel_commands = [], []

    for stage in [1, 2, 3]:
        states, vel_commands = collect_hierarchical_data(n_episodes=100, stage=stage)
        all_states.append(states)
        all_vel_commands.append(vel_commands)
        print(f"  Stage {stage}: {len(states)} transitions")

    all_states = np.concatenate(all_states)
    all_vel_commands = np.concatenate(all_vel_commands)
    print(f"Total: {len(all_states)} transitions")

    # Train
    print("\n=== Training Hierarchical BC ===")
    model, state_mean, state_std = train_hierarchical_bc(
        all_states, all_vel_commands, epochs=30, batch_size=256, device=device
    )

    # Wrap in controller
    controller = HierarchicalController(model, state_mean, state_std, device)

    # Evaluate
    print("\nEvaluation:")
    for stage in [1, 3, 4]:
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        metrics = evaluate_hierarchical(env, controller, n_episodes=50)
        print(f"  Stage {stage}: success={metrics['success_rate']:.1%}, dist={metrics['mean_final_dist']:.3f}m")
        env.close()


if __name__ == "__main__":
    main()