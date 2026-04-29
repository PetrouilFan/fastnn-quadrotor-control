#!/usr/bin/env python3
"""Resume Stage 8 training from 800k checkpoint to 1.5M steps."""

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import os
import json

from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class Stage8CurriculumCallback(BaseCallback):
    """Curriculum callback for Stage 8 (same as training script)."""

    def __init__(self, phase_schedule: dict, verbose=1):
        super().__init__(verbose)
        self.phase_schedule = sorted(phase_schedule.items())
        self.current_phase_idx = 0
        self.current_phase = 0
        self.phase_history = []

    def _on_training_start(self):
        self._enable_phase(1)

    def _enable_phase(self, phase_num):
        env = self.model.env
        phase_config = {
            1: {
                "trajectory": "static",
                "moving": False,
                "wind": 0.0,
                "pos_range": 0.05,
                "angle_range": 3.0,
                "speed": 0.1,
                "name": "Static Hover",
            },
            2: {
                "trajectory": "linear_short",
                "moving": True,
                "wind": 0.0,
                "pos_range": 0.1,
                "angle_range": 5.0,
                "speed": 0.1,
                "name": "Linear Short",
            },
            2.5: {
                "trajectory": "figure8_medium",
                "moving": True,
                "wind": 0.2,
                "pos_range": 0.15,
                "angle_range": 8.0,
                "speed": 0.15,
                "name": "Medium Figure-8",
            },
            3: {
                "trajectory": "figure8_large",
                "moving": True,
                "wind": 0.3,
                "pos_range": 0.2,
                "angle_range": 10.0,
                "speed": 0.3,
                "name": "Large Figure-8",
            },
            4: {
                "trajectory": "extended",
                "moving": True,
                "wind": 0.5,
                "pos_range": 0.25,
                "angle_range": 12.0,
                "speed": 0.5,
                "name": "Full Extended Racing",
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

    def _update_speed_phase4(self, step):
        if step < 1100000:
            return
        if step >= 2500000:
            speed = 5.0
        elif step >= 2000000:
            speed = 3.0
        elif step >= 1500000:
            speed = 2.0
        elif step >= 1300000:
            speed = 1.0
        else:
            speed = 0.5
        env = self.model.env
        setter = lambda e: e.set_target_speed(speed)
        if hasattr(env, "envs"):
            for e in env.envs:
                if hasattr(e, "env"):
                    e.env.set_target_speed(speed)
                else:
                    e.set_target_speed(speed)
        else:
            env.set_target_speed(speed)

    def _on_step(self):
        step = self.model.num_timesteps
        while (
            self.current_phase_idx < len(self.phase_schedule)
            and step >= self.phase_schedule[self.current_phase_idx][0]
        ):
            _, next_phase = self.phase_schedule[self.current_phase_idx]
            if next_phase > self.current_phase:
                self._enable_phase(next_phase)
            self.current_phase_idx += 1
        if self.current_phase == 4:
            self._update_speed_phase4(step)
        return True


def main():
    stage = 8
    seed = 0
    total_target = 1500000

    model_dir = f"models_stage8_progressive/stage_{stage}/seed_{seed}"
    tb_log_dir = f"tb_logs_stage8_progressive/stage_{stage}/seed_{seed}"

    print("Resuming Stage 8 training...")
    print(f"Loading model from {model_dir}/final.zip")
    model = SAC.load(f"{model_dir}/final.zip", device="cpu")
    print(f"Loaded model with {model.num_timesteps} steps")

    # Create training environments
    def make_env(seed, rank):
        def _():
            env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
            env.reset(seed=seed + rank)
            return env

        return _

    train_env = DummyVecEnv([make_env(seed, i) for i in range(8)])

    # Attach environment to model
    model.set_env(train_env)

    # Update tensorboard log
    model.tensorboard_log = tb_log_dir

    # Curriculum schedule
    phase_schedule = {
        0: 1,
        200000: 2,
        500000: 2.5,
        800000: 3,
        1100000: 4,
    }
    curriculum_callback = Stage8CurriculumCallback(phase_schedule)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path=model_dir,
        name_prefix="stage8_checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # Determine remaining steps
    remaining = total_target - model.num_timesteps
    print(f"Training for {remaining:,} more steps (total {total_target:,})")

    model.learn(
        total_timesteps=total_target,
        callback=[curriculum_callback, checkpoint_callback],
        tb_log_name=f"SAC_stage{stage}_resumed",
        reset_num_timesteps=False,
    )

    final_path = f"{model_dir}/final.zip"
    model.save(final_path)
    print(f"Saved final model to {final_path}")

    with open(f"{model_dir}/curriculum_history_resumed.json", "w") as f:
        json.dump(
            {
                "phase_schedule": phase_schedule,
                "phase_history": curriculum_callback.phase_history,
                "total_steps": model.num_timesteps,
            },
            f,
            indent=2,
        )

    train_env.close()


if __name__ == "__main__":
    main()
