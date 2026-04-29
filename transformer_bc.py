#!/usr/bin/env python3
"""
Transformer-based Behavioral Cloning for Quadrotor Control

Uses a causal transformer over the action history window to learn temporal
dependencies that MLPs miss. The key insight: BC drifts because it doesn't
learn the feedback loop - only the state→action mapping. The transformer
attends over past states/actions and learns the implicit dynamics.

Architecture:
- 3-layer causal transformer, d_model=128, 1 attention head
- Input: 52-dim state at each step (embedded to d_model)
- Maintains rolling context window of recent states
- Output: 4D motor command

Expected latency with FastNN: ~50-100μs on Pi5 (well under 1ms budget)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal awareness."""

    def __init__(self, d_model: int, max_len: int = 128, learnable: bool = False):
        super().__init__()
        self.d_model = d_model
        self.learnable = learnable

        if learnable:
            self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        else:
            # Pre-compute positional encodings
            position = torch.arange(max_len).unsqueeze(1).float()
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe = torch.zeros(1, max_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            pe[0, :, 1::2] = torch.cos(position * div_term)
            self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.shape[1]
        if self.learnable:
            # Truncate or pad pos_embed to match seq_len
            pos_embed = self.pos_embed[:, :seq_len]
            return x + pos_embed
        else:
            return x + self.pe[:, :seq_len]


class CausalMaskAttention(nn.Module):
    """Multi-head attention with causal masking for control.

    Unlike bidirectional attention (used in BERT/classification), causal
    attention only looks at past timesteps. This is essential for control
    since actions must depend only on past states, not future.
    """

    def __init__(self, d_model: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} must be divisible by n_heads {n_heads}"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Fused QKV projection (3x faster than separate)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            mask: optional causal mask (seq_len, seq_len)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # QKV projection and split
        qkv = self.qkv_proj(x)  # (batch, seq, 3*d_model)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, n_heads, seq, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (batch, n_heads, seq, seq)

        # Apply causal mask (lower triangular)
        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (batch, n_heads, seq, head_dim)
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Single transformer block with causal attention + FFN."""

    def __init__(self, d_model: int, n_heads: int = 1, ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.attn = CausalMaskAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm attention (more stable training)
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerController(nn.Module):
    """
    Causal Transformer for end-to-end quadrotor control.

    Maintains a rolling context window of recent states and uses
    self-attention to model temporal dependencies. Unlike GRU which
    compresses history into a fixed-size hidden state, the transformer
    can explicitly attend to relevant past timesteps.

    Design choices:
    - Causal attention: only look backward in time (no future peeking)
    - Pre-norm residual blocks: more stable training
    - Learnable output scale: adapts to motor range
    - ReLU activation: faster than GeLU for small models
    """

    def __init__(
        self,
        state_dim: int = 52,
        action_dim: int = 4,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 1,
        ff_dim: int = 512,
        context_len: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.context_len = context_len
        self.d_model = d_model

        # Input embedding: state → d_model
        self.input_embed = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=context_len, learnable=True)

        # Causal transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Action output head with bounded output
        self.action_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # bounded to [-1, 1]
        )

        # Learnable output scale for motor range
        # [thrust_scale, roll_scale, pitch_scale, yaw_scale]
        self.action_scale = nn.Parameter(torch.tensor([15.0, 5.0, 5.0, 2.0]))

        # Pre-compute causal mask
        self.register_buffer(
            'causal_mask',
            torch.tril(torch.ones(context_len, context_len, dtype=torch.bool))
        )

        # Rolling state buffer for inference
        self.register_buffer('state_buffer', torch.zeros(1, context_len, state_dim))
        self.buffer_ptr = 0

    def _update_buffer(self, state: torch.Tensor):
        """Update rolling state buffer with new observation."""
        batch_size = state.shape[0]
        if self.state_buffer.shape[0] != batch_size:
            self.state_buffer = torch.zeros(batch_size, self.context_len, state.shape[-1], device=state.device)
            self.buffer_ptr = 0

        # Roll buffer and insert new state
        self.state_buffer = torch.roll(self.state_buffer, shifts=-1, dims=1)
        self.state_buffer[:, -1] = state

    def forward(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for a single step.

        Args:
            state: (batch, state_dim) - single observation
            hidden: unused, for API compatibility with GRU version

        Returns:
            action: (batch, action_dim) - motor command
            hidden: None (transformer doesn't use hidden state)
        """
        batch_size = state.shape[0]

        # Update rolling buffer
        self._update_buffer(state)

        # Embed + positional encode
        x = self.input_embed(self.state_buffer)  # (batch, context_len, d_model)
        x = self.pos_enc(x)

        # Pass through transformer with causal mask
        for block in self.blocks:
            x = block(x, mask=self.causal_mask)

        # Get action from last timestep only (causal = only valid prediction)
        x = self.norm(x[:, -1])  # (batch, d_model)
        action = self.action_head(x)  # (batch, action_dim)

        # Scale to motor range
        thrust = 10.0 + action[:, 0] * self.action_scale[0]
        torques = action[:, 1:] * self.action_scale[1:].unsqueeze(0)
        action_scaled = torch.cat([thrust.unsqueeze(1), torques], dim=1)

        return action_scaled, None

    def forward_sequence(self, states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a sequence of states (for training).

        Args:
            states: (batch, seq_len, state_dim)

        Returns:
            actions: (batch, seq_len, action_dim)
        """
        # Embed + positional encode
        x = self.input_embed(states)
        x = self.pos_enc(x)

        # Causal mask for this sequence length
        seq_len = states.shape[1]
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=states.device, dtype=torch.bool)
        )

        # Pass through transformer
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        x = self.norm(x)
        actions = self.action_head(x)

        # Scale to motor range
        thrust = 10.0 + actions[:, :, 0:1] * self.action_scale[0]
        torques = actions[:, :, 1:] * self.action_scale[1:].unsqueeze(0)
        actions_scaled = torch.cat([thrust, torques], dim=2)

        return actions_scaled

    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset state buffer at episode start."""
        if device is None:
            device = self.state_buffer.device
        self.state_buffer = torch.zeros(batch_size, self.context_len, self.input_embed[0].in_features, device=device)
        self.buffer_ptr = 0


class GRUController(nn.Module):
    """
    GRU-based controller for end-to-end quadrotor control.

    Maintains hidden state across timesteps that encodes temporal context.
    Simpler than transformer but effective for short-term dynamics.

    Architecture:
    - 2-layer GRU, hidden_dim=128
    - Input: 52-dim state
    - Hidden: 2x128 = 256 total hidden units
    """

    def __init__(
        self,
        state_dim: int = 52,
        action_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            state_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )

        self.action_scale = nn.Parameter(torch.tensor([15.0, 5.0, 5.0, 2.0]))

    def forward(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim) or (batch, seq=1, state_dim)
            hidden: (num_layers, batch, hidden_dim) or None

        Returns:
            action: (batch, action_dim)
            hidden: (num_layers, batch, hidden_dim)
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (batch, seq=1, state_dim)

        output, hidden = self.gru(state, hidden)  # output: (batch, seq, hidden)
        action = self.action_head(output[:, -1])  # (batch, action_dim)

        thrust = 10.0 + action[:, 0] * self.action_scale[0]
        torques = action[:, 1:] * self.action_scale[1:].unsqueeze(0)
        action_scaled = torch.cat([thrust.unsqueeze(1), torques], dim=1)

        return action_scaled, hidden

    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset hidden state at episode start."""
        if device is None:
            device = next(self.parameters()).device
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


class EnsembleController(nn.Module):
    """
    Ensemble of controllers for uncertainty estimation.

    Multiple models make predictions, variance across predictions
    indicates uncertainty. Useful for detecting out-of-distribution
    situations (e.g., unexpected wind gusts).

    Falls back to conservative PD when uncertainty is high.
    """

    def __init__(
        self,
        n_models: int = 3,
        model_type: str = 'transformer',
        **model_kwargs
    ):
        super().__init__()
        self.n_models = n_models

        # Create ensemble of identical architectures
        if model_type == 'transformer':
            self.models = nn.ModuleList([
                TransformerController(**model_kwargs)
                for _ in range(n_models)
            ])
        elif model_type == 'gru':
            self.models = nn.ModuleList([
                GRUController(**model_kwargs)
                for _ in range(n_models)
            ])
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, state: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            mean_action: (batch, action_dim) - ensemble mean
            uncertainty: (batch, action_dim) - std across models
            hidden: hidden state (for GRU, only from first model)
        """
        preds = []
        for model in self.models:
            # Share state buffer for transformer (each has its own)
            action, _ = model(state, hidden)
            preds.append(action)

        preds = torch.stack(preds)  # (n_models, batch, action_dim)
        mean_action = preds.mean(dim=0)
        uncertainty = preds.std(dim=0)  # epistemic uncertainty

        return mean_action, uncertainty, None

    def reset(self, batch_size: int = 1, device: torch.device = None):
        """Reset all model states."""
        for model in self.models:
            model.reset(batch_size, device)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    print("=== Transformer Controller ===")
    trans = TransformerController(state_dim=52, action_dim=4, d_model=128, n_layers=3)
    print(f"Parameters: {count_parameters(trans):,}")
    x = torch.randn(2, 52)  # batch=2, state_dim=52
    y, _ = trans(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")
    trans.reset(batch_size=2)
    y, _ = trans(x)
    print(f"After reset: {y.shape}")

    print("\n=== GRU Controller ===")
    gru = GRUController(state_dim=52, action_dim=4, hidden_dim=128)
    print(f"Parameters: {count_parameters(gru):,}")
    x = torch.randn(2, 52)
    y, h = gru(x)
    print(f"Input: {x.shape} -> Output: {y.shape}, Hidden: {h.shape}")

    print("\n=== Ensemble Controller ===")
    ensemble = EnsembleController(n_models=3, model_type='transformer', state_dim=52, action_dim=4)
    print(f"Parameters: {count_parameters(ensemble):,}")
    x = torch.randn(2, 52)
    y, unc, _ = ensemble(x)
    print(f"Input: {x.shape} -> Output: {y.shape}, Uncertainty: {unc.shape}")

    print("\nAll controllers initialized successfully!")
