#!/usr/bin/env python3
"""
IQL (Implicit Q-Learning) for Quadrotor Control

IQL is an offline RL method that:
1. Extracts confident Q-function from data
2. Uses expectile regression to estimate value function
3. Policy extraction via advantage-weighted regression

This is simpler than PPO/SAC for offline setting and works well
when you have BC data but want to improve beyond BC.

Usage:
    python train_iql.py --data data/pd_stage3.npz --stage 3
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import json

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
from fastnn_quadrotor.utils.transformer_bc import GRUController, TransformerController
from fastnn_quadrotor.training.train_transformer_bc import BCController, evaluate


DEPLOYABLE_DIM = 52
ACTION_DIM = 4


class IQL:
    """
    Implicit Q-Learning for offline RL.

    Reference: "IQL: Implicit Q-Learning" (Kostrikov et al., 2022)
    https://arxiv.org/abs/2110.06169

    Key ideas:
    - Learn Q-function with TD learning (like SAC but offline)
    - Learn V-function via expectile regression
    - Extract policy via advantage-weighted regression
    - No need for OOD action penalty (unlike CQL)
    """

    def __init__(
        self,
        state_dim=52,
        action_dim=4,
        hidden_dim=256,
        actor_dim=256,
        quantile_embedding_dim=64,
        device='cuda',
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q-function (critic)
        self.q1 = self._build_critic(state_dim, action_dim, hidden_dim)
        self.q2 = self._build_critic(state_dim, action_dim, hidden_dim)

        # V-function (value)
        self.v = self._build_critic(state_dim, 0, hidden_dim)  # state-only

        # Actor (policy)
        self.actor = self._build_actor(state_dim, action_dim, actor_dim)

        # Target networks
        self.q1_target = self._build_critic(state_dim, action_dim, hidden_dim)
        self.q2_target = self._build_critic(state_dim, action_dim, hidden_dim)
        self.v_target = self._build_critic(state_dim, 0, hidden_dim)
        self.hard_update_target()

        # Optimizers
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=3e-4
        )
        self.v_opt = torch.optim.Adam(self.v.parameters(), lr=3e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        # IQL specific
        self.beta = 0.5  # Expectile parameter for V
        self.expectile_weight = 0.0  # Weight for expectile loss (use 0 for now)
        self.actor_lr = 3e-4

    def _build_critic(self, state_dim, action_dim, hidden_dim):
        """Build Q or V network."""
        layers = []
        prev_dim = state_dim + action_dim
        for dim in [hidden_dim, hidden_dim, hidden_dim // 2]:
            layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ReLU()])
            prev_dim = dim

        if action_dim > 0:
            layers.append(nn.Linear(prev_dim, 1))  # Q-value
        else:
            layers.append(nn.Linear(prev_dim, 1))  # V-value

        return nn.Sequential(*layers).to(self.device)

    def _build_actor(self, state_dim, action_dim, hidden_dim):
        """Build policy network."""
        layers = []
        prev_dim = state_dim
        for dim in [hidden_dim, hidden_dim, hidden_dim // 2]:
            layers.extend([nn.Linear(prev_dim, dim), nn.LayerNorm(dim), nn.ReLU()])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, action_dim))
        layers.append(nn.Tanh())  # Bounded actions

        return nn.Sequential(*layers).to(self.device)

    def hard_update_target(self):
        """Copy parameters to target networks."""
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.v_target.load_state_dict(self.v.state_dict())

    def soft_update_target(self, tau=0.005):
        """Soft update target networks."""
        with torch.no_grad():
            for target, source in [
                (self.q1_target, self.q1),
                (self.q2_target, self.q2),
                (self.v_target, self.v),
            ]:
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    def train_step(self, states, actions, next_states, dones, rewards, gamma=0.99):
        """Single training step."""
        # Compute target Q
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_q = torch.min(
                self.q1_target(torch.cat([next_states, next_actions], dim=-1)),
                self.q2_target(torch.cat([next_states, next_actions], dim=-1)),
            )
            next_v = self.v_target(next_states)
            target_q = rewards + gamma * (1 - dones) * next_q
            target_v = next_v  # For now, use next V as target

        # Compute current Q
        current_q1 = self.q1(torch.cat([states, actions], dim=-1))
        current_q2 = self.q2(torch.cat([states, actions], dim=-1))

        # Q loss
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        # V loss (expectile regression)
        current_v = self.v(states)
        with torch.no_grad():
            v_target = target_v  # Simplified

        # Expectile regression for V
        diff = v_target - current_v
        weight = torch.where(
            diff > 0,
            torch.tensor(self.beta),
            torch.tensor(1 - self.beta)
        )
        v_loss = weight * (diff ** 2)

        # Actor loss (policy extraction via advantage)
        with torch.no_grad():
            q_val = torch.min(
                self.q1(torch.cat([states, actions], dim=-1)),
                self.q2(torch.cat([states, actions], dim=-1)),
            )
            v_val = self.v(states)
            advantage = q_val - v_val

        # Sample actions from policy
        policy_actions = self.actor(states)
        log_probs = self.actor.log_prob(states, policy_actions)

        # Weighted policy loss (like BC but weighted by advantage)
        actor_loss = -(log_probs * torch.exp(advantage / 10)).mean()

        # Update networks
        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        self.v_opt.zero_grad()
        v_loss.mean().backward()
        self.v_opt.step()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update target
        self.soft_update_target()

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'v_loss': v_loss.mean().item(),
            'actor_loss': actor_loss.item(),
        }

    def save(self, path):
        torch.save({
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'v': self.v.state_dict(),
            'actor': self.actor.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'v_target': self.v_target.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.v.load_state_dict(checkpoint['v'])
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        self.v_target.load_state_dict(checkpoint['v_target'])


class IQLController:
    """Wraps IQL actor for environment interaction."""

    def __init__(self, iql, state_mean, state_std, device='cuda'):
        self.iql = iql
        self.state_mean = state_mean
        self.state_std = state_std
        self.device = device

    def predict(self, obs):
        with torch.no_grad():
            state_raw = torch.FloatTensor(obs[:DEPLOYABLE_DIM]).to(self.device)
            state_norm = (state_raw - torch.FloatTensor(self.state_mean).to(self.device)) / \
                         (torch.FloatTensor(self.state_std).to(self.device) + 1e-8)

            action = self.iql.actor(state_norm.unsqueeze(0)).squeeze(0).cpu().numpy()
            return action

    def to_env_action(self, raw_action):
        """Convert raw output to normalized action space."""
        normalized = np.zeros(4)
        normalized[0] = np.clip((raw_action[0] - 10.0) / 10.0, -1.0, 1.0)
        normalized[1] = np.clip(raw_action[1] / 3.0, -1.0, 1.0)
        normalized[2] = np.clip(raw_action[2] / 3.0, -1.0, 1.0)
        normalized[3] = np.clip(raw_action[3] / 2.0, -1.0, 1.0)
        return normalized


def collect_transitions(env, n_episodes=100, max_steps=500):
    """Collect (s, a, r, s', done) transitions using PD controller."""
    states, actions, rewards, next_states, dones = [], [], [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        steps = 0

        while not (terminated or truncated) and steps < max_steps:
            pd_action = env._cascaded_controller()
            next_obs, reward, terminated, truncated, _ = env.step(np.zeros(4))

            states.append(obs[:DEPLOYABLE_DIM].copy())
            actions.append(pd_action.copy())
            rewards.append(reward)
            next_states.append(next_obs[:DEPLOYABLE_DIM].copy())
            dones.append(terminated or truncated)

            obs = next_obs
            steps += 1

    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(next_states, dtype=np.float32),
        np.array(dones, dtype=np.float32),
    )


def train_iql(states, actions, rewards, next_states, dones, epochs=100, batch_size=256, device='cuda'):
    """Train IQL on collected transitions."""
    iql = IQL(state_dim=52, action_dim=4, hidden_dim=256, device=device)

    dataset = TensorDataset(
        torch.FloatTensor(states),
        torch.FloatTensor(actions),
        torch.FloatTensor(rewards),
        torch.FloatTensor(next_states),
        torch.FloatTensor(dones),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    losses = []
    for epoch in tqdm(range(epochs), desc="IQL Training"):
        epoch_losses = {'q1_loss': 0, 'q2_loss': 0, 'v_loss': 0, 'actor_loss': 0}
        n_batches = 0

        for states_b, actions_b, rewards_b, next_states_b, dones_b in loader:
            states_b = states_b.to(device)
            actions_b = actions_b.to(device)
            rewards_b = rewards_b.to(device)
            next_states_b = next_states_b.to(device)
            dones_b = dones_b.to(device)

            loss_dict = iql.train_step(states_b, actions_b, next_states_b, dones_b, rewards_b)

            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

        for k in epoch_losses:
            epoch_losses[k] /= n_batches
        losses.append(epoch_losses)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: "
                  f"Q1={epoch_losses['q1_loss']:.4f}, "
                  f"V={epoch_losses['v_loss']:.4f}, "
                  f"Actor={epoch_losses['actor_loss']:.4f}")

    return iql, losses


def main():
    parser = argparse.ArgumentParser(description="IQL for quadrotor control")
    parser.add_argument("--stage", type=int, default=3, help="Curriculum stage for data collection")
    parser.add_argument("--episodes", type=int, default=200, help="Episodes for data collection")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--save", type=str, default="models/iql_quadrotor.pt", help="Save path")
    parser.add_argument("--evaluate", action="store_true", help="Only evaluate")
    parser.add_argument("--data-save", type=str, default=None, help="Save collected data")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.evaluate:
        if not os.path.exists(args.save):
            print(f"Model not found: {args.save}")
            return

        print(f"Loading IQL model from {args.save}")
        iql = IQL(device=device)
        iql.load(args.save)

        # Need normalization stats - load from actor checkpoint if available
        state_mean = np.zeros(52)
        state_std = np.ones(52)

        controller = IQLController(iql, state_mean, state_std, device)

        print("Evaluating on all stages...")
        for stage in [1, 2, 3, 4]:
            env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
            metrics = evaluate(env, controller, n_episodes=100)
            print(f"Stage {stage}: success={metrics['success_rate']:.1%}, "
                  f"survival={metrics['survival_rate']:.1%}, "
                  f"dist={metrics['mean_final_dist']:.3f}m")
            env.close()
        return

    # Collect data
    print(f"Collecting transitions from Stage {args.stage}...")
    env = RMAQuadrotorEnv(curriculum_stage=args.stage, use_direct_control=False)
    states, actions, rewards, next_states, dones = collect_transitions(env, n_episodes=args.episodes)
    env.close()

    print(f"Collected {len(states)} transitions")
    print(f"Reward: {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"Done rate: {dones.mean():.1%}")

    if args.data_save:
        os.makedirs(os.path.dirname(args.data_save), exist_ok=True)
        np.savez(args.data_save, states=states, actions=actions, rewards=rewards,
                 next_states=next_states, dones=dones)
        print(f"Data saved to {args.data_save}")

    # Normalize states
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0) + 1e-8
    states_norm = (states - state_mean) / state_std
    next_states_norm = (next_states - state_mean) / state_std

    # Train IQL
    print(f"\nTraining IQL for {args.epochs} epochs...")
    iql, losses = train_iql(
        states_norm, actions, rewards, next_states_norm, dones,
        epochs=args.epochs, batch_size=args.batch_size, device=device
    )

    # Save
    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    iql.save(args.save)
    print(f"Model saved to {args.save}")

    # Save losses
    loss_path = args.save.replace('.pt', '_losses.json')
    with open(loss_path, 'w') as f:
        json.dump(losses, f, indent=2)

    # Evaluate
    print("\n=== Evaluation ===")
    controller = IQLController(iql, state_mean, state_std, device)

    for stage in [1, 2, 3, 4]:
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        metrics = evaluate(env, controller, n_episodes=100)
        print(f"Stage {stage}: success={metrics['success_rate']:.1%}, "
              f"survival={metrics['survival_rate']:.1%}, "
              f"dist={metrics['mean_final_dist']:.3f}m")
        env.close()


if __name__ == "__main__":
    main()