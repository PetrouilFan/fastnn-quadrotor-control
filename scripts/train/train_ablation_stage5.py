#!/usr/bin/env python3
"""
Ablation Study for Stage 5 Reward Function

Tests three configurations:
A) Attitude cliff only (no torque penalty)
B) Torque penalty only (no attitude cliff)
C) Both (baseline - already trained)

Run with: python train_ablation_stage5.py --config [cliff_only|torque_only|both]
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import argparse
import json
import shutil


def make_env(stage, seed, rank, speed=0.05):
    def _init():
        # Import here to get fresh module
        import importlib
        import env_rma
        importlib.reload(env_rma)
        from fastnn_quadrotor.env_rma import RMAQuadrotorEnv

        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.reset(seed=seed + rank)
        env.set_target_speed(speed)
        env.set_moving_target(True)
        return env
    return _init


def setup_reward_config(config_type):
    """Modify env_rma.py for different ablation configs."""
    with open('env_rma.py', 'r') as f:
        content = f.read()

    if config_type == 'cliff_only':
        # Remove torque penalty but keep attitude cliff
        content = content.replace(
            '# Roll/pitch torque penalty: penalize extreme angular torques that cause flips\n        # action[1] = roll torque, action[2] = pitch torque (scaled to [-1,1])\n        r_torque = -0.2 * (action[1]**2 + action[2]**2)',
            '# Roll/pitch torque penalty: DISABLED for ablation\n        r_torque = 0.0'
        )
        print("Configuration: Attitude cliff ONLY (no torque penalty)")

    elif config_type == 'torque_only':
        # Remove attitude cliff but keep torque penalty
        content = content.replace(
            '# Attitude cliff: steep penalty for tilting past 30° to prevent crash spiral\n        # Below 30° (0.52 rad): penalty is zero. Above: quadratic cliff grows fast.\n        att_cliff_threshold = 0.52  # ~30 degrees\n        if att_err > att_cliff_threshold:\n            excess = att_err - att_cliff_threshold\n            r_att -= 5.0 * excess * excess  # up to -5.0 at 60° tilt',
            '# Attitude cliff: DISABLED for ablation\n        # att_cliff_threshold = 0.52\n        # if att_err > att_cliff_threshold:\n        #     excess = att_err - att_cliff_threshold\n        #     r_att -= 5.0 * excess * excess\n        pass  # No attitude cliff'
        )
        print("Configuration: Torque penalty ONLY (no attitude cliff)")

    elif config_type == 'both':
        print("Configuration: Both penalties (baseline)")
        return  # No changes needed

    with open('env_rma.py', 'w') as f:
        f.write(content)


def train_ablation(config_type, steps=2000000, n_envs=32, seed=0):
    """Train with specific ablation configuration."""
    print("=" * 60)
    print(f"Stage 5 Ablation Study: {config_type}")
    print("=" * 60)

    model_dir = f"models_stage5_ablation/{config_type}/seed_{seed}"
    os.makedirs(model_dir, exist_ok=True)

    env_fns = [make_env(stage=5, seed=seed, rank=i) for i in range(n_envs)]
    train_env = SubprocVecEnv(env_fns)

    speed_schedule = {
        0: 0.05,
        steps // 8: 0.1,
        steps // 4: 0.2,
        steps // 2: 0.4,
        int(steps * 0.7): 0.7,
        int(steps * 0.85): 0.9,
        steps: 1.0,
    }

    class SpeedCurriculumCallback:
        def __init__(self):
            self.schedule = sorted(speed_schedule.items())
            self.idx = 0

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        ent_coef='auto',
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=f"tb_logs_stage5_ablation/{config_type}/seed_{seed}/",
        device="cuda" if torch.cuda.is_available() else "cpu",
        seed=seed,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path=model_dir,
        name_prefix=f"stage5_{config_type}",
    )

    print(f"Training for {steps} steps...")
    model.learn(
        total_timesteps=steps,
        callback=[checkpoint_callback],
        progress_bar=True,
    )

    model.save(os.path.join(model_dir, "final"))
    print(f"Saved to {model_dir}/final")
    train_env.close()


def main():
    parser = argparse.ArgumentParser(description="Stage 5 Ablation Study")
    parser.add_argument("--config", type=str, required=True,
                        choices=['cliff_only', 'torque_only', 'both'],
                        help="Ablation configuration")
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Backup original
    shutil.copy('env_rma.py', 'env_rma_original.py')

    try:
        # Setup reward config
        setup_reward_config(args.config)

        # Train
        train_ablation(args.config, args.steps, args.n_envs, args.seed)

    finally:
        # Restore original
        shutil.move('env_rma_original.py', 'env_rma.py')
        print("\nRestored original env_rma.py")


if __name__ == "__main__":
    main()
