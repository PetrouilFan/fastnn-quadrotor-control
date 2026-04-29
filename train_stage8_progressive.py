#!/usr/bin/env python3
"""
Stage 8 Progressive Curriculum Training
========================================

Four-phase curriculum to master extreme racing:

Phase 1 (0-50k): Static target hover
  - Target fixed at origin
  - Learn basic stabilization
  - 100% survival expected

Phase 2 (50k-150k): Short linear trajectory
  - 5m straight line back-and-forth
  - Moving target at 0.1x speed
  - Learn to follow simple path

Phase 3 (150k-350k): Extended trajectory, slow speed
  - Full 29m extended figure-8
  - Speed: 0.1x → 0.3x → 0.5x
  - Master the full track geometry

Phase 4 (350k-5M): Speed curriculum to 5x
  - Full extended track
  - Speed: 0.5x → 1.0x → 2.0x → 3.0x → 5.0x
  - Extreme racing mastery

Reward Enhancements:
  - Waypoint bonuses for hitting trajectory milestones
  - Velocity matching reward
  - Smoother penalty for long-horizon stability
"""

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    BaseCallback,
    EvalCallback,
    CallbackList,
)
import os
import argparse
import json
from datetime import datetime

from env_rma import RMAQuadrotorEnv


class Stage8CurriculumCallback(BaseCallback):
    """Manages curriculum and saves checkpoints at phase transitions."""

    def __init__(self, phase_schedule: dict, model_dir: str, verbose=1):
        super().__init__(verbose)
        self.phase_schedule = sorted(phase_schedule.items())
        self.current_phase_idx = 0
        self.current_phase = 0
        self.phase_history = []
        self.model_dir = model_dir

    def _on_training_start(self):
        """Initialize phase 1."""
        self._enable_phase(1)

    def _enable_phase(self, phase_num):
        """Configure environment for specific phase with progressive difficulty."""
        env = self.model.env

        # Define phase parameters - 6-phase progressive curriculum
        phase_config = {
            1: {
                "trajectory": "static",
                "moving": False,
                "wind": 0.0,
                "pos_range": 0.05,
                "angle_range": 3.0,
                "speed": 0.1,
                "name": "Static Target Hover (minimal perturbations)",
            },
            2: {
                "trajectory": "linear_short",
                "moving": True,
                "wind": 0.0,
                "pos_range": 0.1,
                "angle_range": 5.0,
                "speed": 0.1,
                "name": "Short Linear Trajectory (no wind)",
            },
            2.5: {
                "trajectory": "figure8_medium",
                "moving": True,
                "wind": 0.2,
                "pos_range": 0.15,
                "angle_range": 8.0,
                "speed": 0.15,
                "name": "Medium Figure-8 (1.5m, gentle turns)",
            },
            3: {
                "trajectory": "figure8_large",
                "moving": True,
                "wind": 0.3,
                "pos_range": 0.2,
                "angle_range": 10.0,
                "speed": 0.3,
                "name": "Large Figure-8 (2.0m + light vertical)",
            },
            3.5: {
                "trajectory": "figure8_large2",
                "moving": True,
                "wind": 0.4,
                "pos_range": 0.22,
                "angle_range": 11.0,
                "speed": 0.4,
                "name": "Intermediate Extended (2.5m + moderate vertical)",
            },
            4: {
                "trajectory": "extended",
                "moving": True,
                "wind": 0.5,
                "pos_range": 0.25,
                "angle_range": 12.0,
                "speed": 0.5,
                "name": "Full Extended Racing (speed curriculum 0.5x→5.0x)",
            },
        }

        config = phase_config[phase_num]

        def config_env(e):
            if hasattr(e, "env"):
                e = e.env
            e.set_target_trajectory(config["trajectory"])
            e.set_moving_target(config["moving"])
            e.wind_force[:] = config["wind"]
            e.set_target_speed(config["speed"])
            e._phase_pos_range = config["pos_range"]
            e._phase_angle_range = config["angle_range"]
            if self.verbose > 0:
                print(f"\n=== PHASE {phase_num}: {config['name']} ===")

        if hasattr(env, "envs"):
            for e in env.envs:
                config_env(e)
        else:
            config_env(env)

        self.current_phase = phase_num
        self.phase_history.append((self.num_timesteps, phase_num))

        # Save phase transition checkpoint
        phase_path = os.path.join(
            self.model_dir, f"phase{phase_num}_steps_{self.num_timesteps}.zip"
        )
        self.model.save(phase_path)
        print(f"✓ Saved Phase {phase_num} checkpoint: {phase_path}")

    def _update_speed_for_phase4(self, step):
        """Speed curriculum for Phase 4 (starts at 1.5M steps)."""
        # Phase 4: Full extended racing, speed 0.5x → 5.0x
        if step < 1400000:
            return

        # Gradual progression to 5.0x over 2.5M steps
        if step >= 3200000:
            speed = 5.0
        elif step >= 2700000:
            speed = 4.0
        elif step >= 2200000:
            speed = 3.0
        elif step >= 1800000:
            speed = 2.0
        elif step >= 1600000:
            speed = 1.0
        else:  # 1400000 - 1600000
            speed = 0.5

        # Apply to all environments
        env = self.model.env
        if hasattr(env, "envs"):
            for e in env.envs:
                if hasattr(e, "env"):
                    e.env.set_target_speed(speed)
                elif hasattr(e, "set_target_speed"):
                    e.set_target_speed(speed)
        elif hasattr(env, "set_target_speed"):
            env.set_target_speed(speed)

    def _on_step(self):
        step = self.model.num_timesteps

        # Phase transitions
        while (
            self.current_phase_idx < len(self.phase_schedule)
            and step >= self.phase_schedule[self.current_phase_idx][0]
        ):
            _, next_phase = self.phase_schedule[self.current_phase_idx]
            if next_phase > self.current_phase:
                self._enable_phase(next_phase)
            self.current_phase_idx += 1

        # Phase 4 speed curriculum
        if self.current_phase == 4:
            self._update_speed_for_phase4(step)

        return True


def make_env(stage, seed, rank):
    """Create a single environment instance."""

    def _():
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        # Initial config will be overridden by curriculum callback
        env.reset(seed=seed + rank)
        return env

    return _


def train_stage8_progressive(stage=8, total_steps=4000000, n_envs=32, seeds=[0]):
    """Train Stage 8 with progressive 6-phase curriculum (including intermediate bridge)."""

    for seed in seeds:
        print("=" * 70)
        print(f"Stage 8 Progressive Curriculum - Seed {seed}")
        print(f"Total training: {total_steps:,} steps with {n_envs} parallel envs")
        print("=" * 70)

        # Setup directories
        model_dir = f"models_stage8_progressive/stage_{stage}/seed_{seed}"
        tb_log_dir = f"tb_logs_stage8_progressive/stage_{stage}/seed_{seed}"
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(tb_log_dir, exist_ok=True)

        # Create environments
        env_fns = [make_env(stage=stage, seed=seed, rank=i) for i in range(n_envs)]
        train_env = DummyVecEnv(env_fns)

        print(f"Observation dimension: {train_env.observation_space.shape[0]}")

        # SAC model
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            learning_starts=2000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            ent_coef="auto",
            target_entropy=-2,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=tb_log_dir,
            verbose=1,
        )

        # Curriculum phases: 6-phase progressive curriculum with intermediate bridge
        phase_schedule = {
            0: 1,  # Phase 1: Static hover (0-200k)
            200000: 2,  # Phase 2: Linear short (200k-500k)
            500000: 2.5,  # Phase 2.5: Medium figure-8 (500k-800k)
            800000: 3,  # Phase 3: Large figure-8 (800k-1.1M)
            1100000: 3.5,  # Phase 3.5: Intermediate extended, 2.5m (1.1M-1.5M)
            1400000: 4,  # Phase 4: Full extended + speed curriculum (1.5M+)
        }

        curriculum_callback = Stage8CurriculumCallback(phase_schedule, model_dir)

        # Checkpoint callback - save every 100k steps to capture phase transitions
        checkpoint_callback = CheckpointCallback(
            save_freq=100000,
            save_path=model_dir,
            name_prefix="stage8_checkpoint",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )

        # Training
        print("\nPhase Schedule:")
        print("  0-200k:   Phase 1 - Static hover (minimal perturbations)")
        print("  200k-500k: Phase 2 - Linear short trajectory, 0.1x speed")
        print("  500k-800k: Phase 2.5 - Medium figure-8 (1.5m), 0.15x speed")
        print(
            "  800k-1.1M: Phase 3 - Large figure-8 (2.0m + light vertical), 0.3x speed"
        )
        print(
            "  1.1M-1.5M: Phase 3.5 - Intermediate extended (2.5m + moderate vertical), 0.4x speed"
        )
        print("  1.5M+:    Phase 4 - Full extended (3.0m) + speed curriculum 0.5x→5.0x")
        print("\nStarting training...\n")

        model.learn(
            total_timesteps=total_steps,
            callback=[curriculum_callback, checkpoint_callback],
            tb_log_name=f"SAC_stage{stage}",
            reset_num_timesteps=False,
        )

        # Save final model
        final_path = os.path.join(model_dir, "final.zip")
        model.save(final_path)
        print(f"\n✓ Final model saved to: {final_path}")
        print(f"✓ Total timesteps: {model.num_timesteps:,}")

        # Save curriculum history
        with open(os.path.join(model_dir, "curriculum_history.json"), "w") as f:
            json.dump(
                {
                    "phase_schedule": phase_schedule,
                    "phase_history": curriculum_callback.phase_history,
                    "total_steps": model.num_timesteps,
                },
                f,
                indent=2,
            )

        # Cleanup
        train_env.close()

    return model_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stage 8 Progressive Curriculum Training"
    )
    parser.add_argument("--stage", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000000)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="0")

    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    print("\n" + "=" * 70)
    print("STAGE 8: EXTREME RACING - PROGRESSIVE CURRICULUM")
    print("=" * 70)
    print("Strategy: 4-phase curriculum from static hover to full racing")
    print("Expected: Gradual improvement, 100% survival by Phase 3")
    print("=" * 70 + "\n")

    train_stage8_progressive(
        stage=args.stage, total_steps=args.steps, n_envs=args.n_envs, seeds=seeds
    )
