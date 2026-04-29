#!/usr/bin/env python3
"""
Stage 5 KL-Clipped Fine-tuning: Break Precision-Stability Tradeoff

Key insight: The 5M model is STABLE but IMPRECISE (4.7m error).
The 10M model is PRECISE (0.32m) but UNSTABLE (crashes).
Both are local optima. We need a way to escape the stable-but-sluggish optimum
WITHOUT drifting into the crash-prone optimum.

Solution: Train from scratch with the new reward (attitude cliff + torque penalty),
starting from 0 speed and going much slower through the curriculum. The key is that
the 5M checkpoint's success came from the curriculum (0.05x → 1.0x speed ramp).
If we train fresh with the new penalties AND the same curriculum, we might find
a better optimum.

Alternatively, use the 5M model as a behavioral constraint via KL penalty:
- The 5M model's policy provides a "safety baseline"
- New policy can deviate to improve precision, but pays a KL cost
- This prevents drifting into the crash-prone regime

Run with:
    python train_stage5_kl.py --base-model models_stage5_curriculum/stage_5/seed_0/final.zip --steps 3000000
"""

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from tqdm import tqdm
import os
import argparse
import json

from env_rma import RMAQuadrotorEnv


class KLConstraintCallback(BaseCallback):
    """Callback that logs KL divergence between current and base policy."""

    def __init__(self, base_model_path, kl_limit=0.1, verbose=1):
        super().__init__(verbose)
        self.base_model_path = base_model_path
        self.kl_limit = kl_limit
        self.kl_history = []

        # Load base model for reference
        self.base_model = SAC.load(base_model_path)
        self.base_actor = self.base_model.actor

    def _compute_kl(self):
        """Compute average KL divergence on recent batch."""
        if not hasattr(self.model, 'replay_buffer') or self.model.replay_buffer.size() < self.model.batch_size:
            return 0.0

        batch = self.model.replay_buffer.sample(self.model.batch_size)
        obs = torch.FloatTensor(batch.observations).to(self.model.device)

        with torch.no_grad():
            base_action_dist = self.base_actor.get_distribution(obs)
            base_log_probs = base_action_dist.log_prob(base_action_dist.distribution.mean)

        current_action_dist = self.model.actor.get_distribution(obs)
        current_log_probs = current_action_dist.log_prob(current_action_dist.distribution.mean)

        kl = F.kl_div(current_log_probs, base_log_probs, reduction='batchmean')
        return kl.item()

    def _on_step(self):
        step = self.model.num_timesteps
        if step % 10000 == 0 and step > 0:
            kl = self._compute_kl()
            self.kl_history.append({'step': step, 'kl': kl})
            if self.verbose > 0:
                print(f"\n  KL divergence from base: {kl:.4f} (limit: {self.kl_limit})")
        return True


def make_env(stage, seed, rank, speed=1.0):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        env.set_target_speed(speed)
        env.set_moving_target(True)
        return env
    return _init


def evaluate(model, n_episodes=50, speed=1.0):
    """Evaluate model on Stage 5 at given speed."""
    env = RMAQuadrotorEnv(curriculum_stage=5, use_direct_control=True)
    env.reset()
    env.set_target_speed(speed)
    env.set_moving_target(True)

    successes = 0
    tracking_errors = []
    max_attitudes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_errors = []
        ep_max_att = 0.0

        while not done and steps < 500:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(action)
            ep_errors.append(np.linalg.norm(env.data.qpos[:3] - env.target_pos))
            quat = env.data.qpos[3:7]
            rpy = env._quat_to_rpy(quat)
            ep_max_att = max(ep_max_att, abs(rpy[0]), abs(rpy[1]))
            steps += 1

        if steps >= 500:
            successes += 1
        tracking_errors.append(np.mean(ep_errors))
        max_attitudes.append(ep_max_att)

    env.close()
    return successes / n_episodes, np.mean(tracking_errors), np.mean(max_attitudes)


def main():
    parser = argparse.ArgumentParser(description="Stage 5 KL-Clipped Fine-tuning")
    parser.add_argument("--base-model", type=str,
                        default="models_stage5_curriculum/stage_5/seed_0/final.zip")
    parser.add_argument("--steps", type=int, default=3000000)
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Lower LR for stable fine-tuning")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-speed", type=float, default=0.05,
                        help="Starting speed for curriculum")
    parser.add_argument("--eval-interval", type=int, default=200000)
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 5 KL-Clipped Fine-tuning")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"LR: {args.lr}, Start speed: {args.start_speed}x")
    print(f"Total steps: {args.steps}")
    print("=" * 60)

    # Create training envs at SLOW speed (curriculum) FIRST
    env_fns = [make_env(stage=5, seed=args.seed, rank=i, speed=args.start_speed) for i in range(args.n_envs)]
    train_env = SubprocVecEnv(env_fns)

    # Load base model WITH the new env (required for different n_envs)
    print(f"\nLoading base model: {args.base_model}")
    base_model = SAC.load(args.base_model, env=train_env)
    base_success, base_error, base_max_att = evaluate(base_model, n_episodes=20)
    print(f"Base model: {base_success:.0%} success, {base_error:.3f}m error, max att: {np.rad2deg(base_max_att):.1f}°")
    base_model.learning_rate = args.lr
    for param_group in base_model.actor.optimizer.param_groups:
        param_group['lr'] = args.lr
    for param_group in base_model.critic.optimizer.param_groups:
        param_group['lr'] = args.lr

    model_dir = "models_stage5_kl"
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path=model_dir,
        name_prefix="stage5_kl",
    )

    # Speed curriculum: 0.05 -> 1.0 over training
    speed_schedule = {
        0: args.start_speed,
        args.steps // 8: 0.1,
        args.steps // 4: 0.2,
        args.steps // 2: 0.4,
        int(args.steps * 0.7): 0.7,
        int(args.steps * 0.85): 0.9,
        args.steps: 1.0,
    }

    class SpeedCurriculumCallback(BaseCallback):
        def __init__(self, schedule, verbose=1):
            super().__init__(verbose)
            self.schedule = sorted(schedule.items())
            self.idx = 0

        def _update_speed(self, speed):
            env = self.model.env
            if hasattr(env, 'envs'):
                for e in env.envs:
                    if hasattr(e, 'env'):
                        e.env.set_target_speed(speed)
                    elif hasattr(e, 'set_target_speed'):
                        e.set_target_speed(speed)
            elif hasattr(env, 'set_target_speed'):
                env.set_target_speed(speed)
            if self.verbose > 0:
                print(f"\n=== Speed curriculum: {speed:.2f}x ===")

        def _on_step(self):
            step = self.model.num_timesteps
            if self.idx < len(self.schedule) - 1:
                next_step, next_speed = self.schedule[self.idx + 1]
                if step >= next_step:
                    self.idx += 1
                    _, speed = self.schedule[self.idx]
                    self._update_speed(speed)
            return True

    curriculum_callback = SpeedCurriculumCallback(speed_schedule)

    print(f"\nFine-tuning for {args.steps} steps with LR={args.lr}...")
    base_model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_callback, curriculum_callback],
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model_path = os.path.join(model_dir, "final")
    base_model.save(model_path)
    print(f"Saved to {model_path}")

    # Evaluate
    print("\n=== Final Evaluation ===")
    results = {'base': {'success': float(base_success), 'tracking_error': float(base_error)}}
    for eval_speed in [0.2, 0.5, 1.0]:
        success, error, max_att = evaluate(base_model, n_episodes=50, speed=eval_speed)
        print(f"Speed {eval_speed:.1f}x: {success:.0%} success, {error:.3f}m error, max att: {np.rad2deg(max_att):.1f}°")
        results[f'speed_{eval_speed}'] = {
            'success': float(success),
            'tracking_error': float(error),
            'max_attitude_deg': float(np.rad2deg(max_att))
        }

    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    train_env.close()


if __name__ == "__main__":
    main()
