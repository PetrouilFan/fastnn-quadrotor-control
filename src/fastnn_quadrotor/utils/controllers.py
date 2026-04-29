#!/usr/bin/env python3
"""
Universal Neural Network Controller for Recovery + Control
A single neural network that handles both nominal control and recovery tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UniversalController(nn.Module):
    """
    Universal NN that handles:
    - Nominal hover control
    - Recovery from extreme conditions (>15° tilt, >10°/s)
    - Wind disturbance rejection
    - Payload adaptation

    Updated for new state space (52 dims):
    - Tracking errors (12) + Linear accel (3) + Body rates (3) +
    - Rotor thrusts (4) + Error integrals (4) + Dilated history (16) + Rotmat (9) + Mass (1) = 52
    """

    def __init__(self, state_dim=52, action_dim=4, hidden_dims=[256, 256, 128]):
        super().__init__()

        # Feature extraction layers
        self.feature_layers = nn.ModuleList()
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            self.feature_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.feature_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        # Multi-head output for different tasks
        # Head 1: Thrust (1D)
        self.thrust_head = nn.Linear(prev_dim, 1)

        # Head 2: Roll/Pitch/Yaw torques (3D)
        self.torque_head = nn.Linear(prev_dim, 3)

        # Task-aware embedding (recovers mode from state)
        self.task_embedding = nn.Sequential(
            nn.Linear(prev_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # 4 modes: hover, recovery, wind, payload
        )

        # Use LayerNorm instead of BatchNorm for single-sample inference
        self.state_normalizer = nn.LayerNorm(state_dim)

    def forward(self, state):
        # Handle both numpy arrays and torch tensors
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Feature extraction
        x = state
        for layer in self.feature_layers:
            x = layer(x)

        # Task detection
        task_logits = self.task_embedding(x)
        task_mode = torch.argmax(task_logits, dim=-1)

        # Action outputs with proper bounds
        # Thrust: hover compensation adjustment [-5, +5] relative to baseline
        thrust = self.thrust_head(x)
        thrust = (
            torch.tanh(thrust) * 5.0 + 10.0
        )  # Range: [5, 15] (baseline hover + adjustment)

        # Torque: small adjustments [-1, +1] to PD baseline
        torque = self.torque_head(x)
        torque = torch.tanh(torque)  # Range: [-1, 1]

        return torch.cat([thrust, torque], dim=-1), task_mode

    def predict(self, state):
        """Inference mode - no gradient"""
        self.eval()
        with torch.no_grad():
            action, mode = self.forward(state)
        return action.numpy(), mode.numpy()


class RecoveryController(nn.Module):
    """
    Specialized recovery controller for extreme conditions.
    Uses deeper network with residual connections.
    """

    def __init__(self, state_dim=30, action_dim=4):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(state_dim, 128)

        # Residual blocks
        self.res_block1 = ResidualBlock(128, 128)
        self.res_block2 = ResidualBlock(128, 128)
        self.res_block3 = ResidualBlock(128, 64)

        # Output heads - more aggressive for recovery
        self.thrust_head = nn.Linear(64, 1)
        self.torque_head = nn.Linear(64, 3)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = F.relu(self.input_proj(state))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)

        # Recovery mode: more aggressive but bounded
        thrust = torch.tanh(self.thrust_head(x)) * 8.0 + 12.0  # Range: [4, 20]
        torque = torch.tanh(self.torque_head(x)) * 5.0  # Range: [-5, 5]

        return torch.cat([thrust, torque], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.ln1 = nn.LayerNorm(out_dim)
        self.ln2 = nn.LayerNorm(out_dim)

        # Skip connection projection
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        return F.relu(out + identity)


class HybridPIDNN(nn.Module):
    """
    Hybrid PID + Neural Network Controller
    - PID provides baseline stable control
    - NN learns residual correction for improved performance
    Updated for new state space (51 dims)
    """

    def __init__(self, state_dim=30, action_dim=4):
        super().__init__()

        # NN residual learner
        self.residual_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),  # Bound residual to [-1, 1]
        )

        # Residual scaling (learnable)
        self.residual_scale = nn.Parameter(torch.ones(action_dim))

    def forward(self, state, pid_output):
        """
        Args:
            state: current state
            pid_output: baseline PID control output
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if isinstance(pid_output, np.ndarray):
            pid_output = torch.FloatTensor(pid_output)
        if pid_output.dim() == 1:
            pid_output = pid_output.unsqueeze(0)

        # Get residual
        residual = self.residual_net(state)
        residual = residual * self.residual_scale  # Learnable scaling

        # Combine: PID + NN residual
        hybrid_output = pid_output + residual

        return hybrid_output


class NNOnlyController(nn.Module):
    """
    Pure Neural Network controller (no PID baseline)
    End-to-end learning from state to action
    Updated for new state space (52 dims)
    """

    def __init__(self, state_dim=52, action_dim=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )

        # Output scaling - LARGER bounds for drop survival
        # [thrust, roll_torque, pitch_torque, yaw_torque]
        self.action_scale = nn.Parameter(torch.tensor([15.0, 5.0, 5.0, 2.0]))

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        if state.dim() == 1:
            state = state.unsqueeze(0)

        raw = self.net(state)
        # Output: [thrust_adj, roll_adj, pitch_adj, yaw_adj]
        # Final: thrust = 10 + adj, torques = adj * scale
        # Range: thrust [-5, 25], torques [-5, 5]
        thrust = 10.0 + raw[:, 0] * 15.0  # Range: [-5, 25]
        torque = raw[:, 1:] * 5.0  # Range: [-5, 5]

        return torch.cat([thrust.unsqueeze(1), torque], dim=1)


def test_controllers():
    """Quick sanity check"""
    state_dim = 30
    batch_size = 4

    # Test Universal Controller
    universal = UniversalController(state_dim, 4)
    dummy_state = torch.randn(batch_size, state_dim)
    action, mode = universal(dummy_state)
    print(f"Universal: action shape={action.shape}, mode shape={mode.shape}")

    # Test Recovery Controller
    recovery = RecoveryController(state_dim, 4)
    action = recovery(dummy_state)
    print(f"Recovery: action shape={action.shape}")

    # Test Hybrid PID+NN
    hybrid = HybridPIDNN(state_dim, 4)
    pid_out = torch.randn(batch_size, 4)
    action = hybrid(dummy_state, pid_out)
    print(f"Hybrid: action shape={action.shape}")

    # Test NN-only
    nn_only = NNOnlyController(state_dim, 4)
    action = nn_only(dummy_state)
    print(f"NN-only: action shape={action.shape}")

    print("\nAll controllers initialized successfully!")


if __name__ == "__main__":
    test_controllers()
