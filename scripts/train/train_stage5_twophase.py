#!/usr/bin/env python3
"""
Stage 5 Two-Phase Training: Break the Precision-Stability Tradeoff

Phase 1: Load the best 5M checkpoint (100% success, 4.76m error)
Phase 2: Fine-tune at full speed with:
  - Very low learning rate (5e-5) for stable refinement
  - High entropy coefficient to prevent aggressive deterministic policy
  - Larger replay buffer (500K) to retain diverse experiences
  - Attitude cliff penalty + roll/pitch torque penalty in reward

Key insight: The 5M model is stable but imprecise (4.7m error).
The 10M model is precise (0.32m) but crashes. We need to bridge the gap
by keeping the stable policy but learning NOT to be sluggish.
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import argparse
import json

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


def make_env(stage, seed, rank):
    def _init():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        env.set_target_speed(1.0)
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
    max_attitudes = []  # Track max roll/pitch to detect near-crashes

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


class EntropyBonusCallback(BaseCallback):
    """Monitor entropy during training."""

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.entropy_history = []

    def _on_step(self):
        if hasattr(self.model, 'actor'):
            # Log actor entropy if available
            pass
        return True


def main():
    parser = argparse.ArgumentParser(description="Stage 5 Two-Phase Training")
    parser.add_argument("--base-model", type=str,
                        default="models_stage5_curriculum/stage_5/seed_0/final.zip",
                        help="Path to base 5M checkpoint")
    parser.add_argument("--phase1-steps", type=int, default=2000000,
                        help="Phase 1 fine-tuning steps")
    parser.add_argument("--phase2-steps", type=int, default=0,
                        help="Phase 2 additional steps (0 = skip)")
    parser.add_argument("--n-envs", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--entropy-coef", type=float, default=0.05,
                        help="Entropy coefficient for exploration")
    parser.add_argument("--buffer-size", type=int, default=500000,
                        help="Replay buffer size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-interval", type=int, default=100000,
                        help="Evaluation interval in steps")
    args = parser.parse_args()

    print("=" * 60)
    print("Stage 5 Two-Phase Training")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"LR: {args.lr}, Entropy coef: {args.entropy_coef}")
    print(f"Buffer: {args.buffer_size}, Envs: {args.n_envs}")
    print("=" * 60)

    # Evaluate base model first
    print("\nEvaluating base model (5M checkpoint)...")
    base_model_tmp = SAC.load(args.base_model)
    base_success, base_error, base_max_att = evaluate(base_model_tmp, n_episodes=20)
    del base_model_tmp
    print(f"Base model: {base_success:.0%} success, {base_error:.3f}m error, max att: {np.rad2deg(base_max_att):.1f}°")

    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    model = SAC.load(args.base_model)

    # Re-evaluate after loading
    print("Evaluating loaded model...")
    loaded_success, loaded_error, loaded_max_att = evaluate(model, n_episodes=20)
    print(f"Base model ({args.base_model}):")
    print(f"  Success: {loaded_success:.0%}, Tracking error: {loaded_error:.3f}m, Max attitude: {np.rad2deg(loaded_max_att):.1f}°")

    # Create fresh training envs at full speed
    env_fns = [make_env(stage=5, seed=args.seed, rank=i) for i in range(args.n_envs)]
    train_env = SubprocVecEnv(env_fns)

    # Set lower learning rate + higher entropy
    model.learning_rate = args.lr
    for param_group in model.actor.optimizer.param_groups:
        param_group['lr'] = args.lr
    for param_group in model.critic.optimizer.param_groups:
        param_group['lr'] = args.lr

    # Update entropy coefficient
    if args.entropy_coef > 0:
        model.ent_coef = args.entropy_coef

    # Update buffer size by creating new buffer
    model.buffer_size = args.buffer_size
    # Note: This won't resize existing buffer, but new samples will use new size

    model_dir = "models_stage5_twophase"
    os.makedirs(model_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path=model_dir,
        name_prefix="stage5_twophase",
    )

    # Phase 1: Fine-tune at full speed
    print(f"\n=== Phase 1: Fine-tuning for {args.phase1_steps} steps ===")
    print(f"LR={args.lr}, entropy_coef={model.ent_coef}")

    model.set_env(train_env)
    model.learn(
        total_timesteps=args.phase1_steps,
        callback=[checkpoint_callback],
        progress_bar=True,
        reset_num_timesteps=False,
    )

    model.save(os.path.join(model_dir, "phase1_final"))

    # Evaluate phase 1
    print("\n=== Phase 1 Evaluation ===")
    for eval_speed in [0.5, 1.0]:
        success, error, max_att = evaluate(model, n_episodes=50, speed=eval_speed)
        print(f"Speed {eval_speed:.1f}x: {success:.0%} success, {error:.3f}m error, max att: {np.rad2deg(max_att):.1f}°")

    # Phase 2: Even lower LR + more entropy (if requested)
    if args.phase2_steps > 0:
        print(f"\n=== Phase 2: Further refinement for {args.phase2_steps} steps ===")
        model.learning_rate = args.lr / 2
        for param_group in model.actor.optimizer.param_groups:
            param_group['lr'] = args.lr / 2
        for param_group in model.critic.optimizer.param_groups:
            param_group['lr'] = args.lr / 2
        model.ent_coef = args.entropy_coef * 2

        model.learn(
            total_timesteps=args.phase2_steps,
            callback=[checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=False,
        )

        model.save(os.path.join(model_dir, "phase2_final"))

    # Final evaluation
    print("\n=== Final Evaluation ===")
    results = {}
    for eval_speed in [0.2, 0.5, 1.0]:
        success, error, max_att = evaluate(model, n_episodes=50, speed=eval_speed)
        print(f"Speed {eval_speed:.1f}x: {success:.0%} success, {error:.3f}m error, max att: {np.rad2deg(max_att):.1f}°")
        results[f'speed_{eval_speed}'] = {
            'success': success,
            'tracking_error': error,
            'max_attitude_deg': float(np.rad2deg(max_att))
        }

    model.save(os.path.join(model_dir, "final"))

    # Save results
    results['base'] = {
        'success': float(base_success),
        'tracking_error': float(base_error),
        'max_attitude_deg': float(np.rad2deg(base_max_att))
    }
    results['config'] = {
        'lr': args.lr,
        'entropy_coef': args.entropy_coef,
        'buffer_size': args.buffer_size,
        'phase1_steps': args.phase1_steps,
        'phase2_steps': args.phase2_steps,
    }

    with open(os.path.join(model_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    train_env.close()

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
