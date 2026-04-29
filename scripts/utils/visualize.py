#!/usr/bin/env python3
"""
Visualization script for trained quadrotor policy with live terminal HUD.

Usage:
    python visualize.py --stage 7          # Stage 7 yaw-only
    python visualize.py --stage 8          # Stage 8 racing
    python visualize.py --stage 5 --episodes 3
"""

import argparse
import numpy as np
import mujoco
from mujoco import viewer
import sys
import os
import time

# Force line-buffered stdout for real-time display
sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


class TerminalHUD:
    """Live single-line HUD for terminal display."""

    def __init__(self):
        self.last_len = 0

    def update(
        self, step, raw_action, pos, target=None, reward=0.0, use_direct_control=False
    ):
        """Update HUD with properly interpreted motor values.

        For direct control (use_direct_control=True):
          raw_action = [thrust_cmd (-1 to 1), roll_torque_cmd, pitch_torque_cmd, yaw_torque_cmd]
          Physical: thrust [0,20]N, torques [-3,3]Nm, [-3,3]Nm, [-2,2]Nm
          Display: M1 = normalized thrust [0,1], M2-M4 = raw torque commands [-1,1]
        For residual mode: show raw actions as-is (these are residuals)
        """
        if use_direct_control:
            # Convert raw action to displayable motor values
            thrust_norm = (
                raw_action[0] + 1.0
            ) / 2.0  # [-1,1] -> [0,1] normalized thrust
            motor_vals = [thrust_norm, raw_action[1], raw_action[2], raw_action[3]]
        else:
            motor_vals = raw_action

        motor_str = " ".join([f"M{i + 1}:{v:+.3f}" for i, v in enumerate(motor_vals)])
        pos_str = f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]"
        target_str = (
            f" | Target: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]"
            if target is not None
            else ""
        )

        line = f"\rStep {step:04d} | {motor_str} | {pos_str}{target_str} | r: {reward:+.2f}"

        # Clear previous line if needed
        if len(line) < self.last_len:
            line += " " * (self.last_len - len(line))
        self.last_len = len(line)

        print(line, end="", flush=True)


def run_visualization(
    model_path=None,
    stage=5,
    pd_only=False,
    num_episodes=5,
    max_steps=500,
    deterministic=True,
):
    # Load RL model
    model = None
    if not pd_only:
        if model_path is None:
            model_paths = {
                1: "models_stage1/stage_1/seed_0/final.zip",
                2: "models_stage2/stage_2/seed_0/final.zip",
                3: "models_stage3/stage_3/seed_0/final.zip",
                4: "models_stage4/stage_4/seed_0/final.zip",
                5: "models_stage5_curriculum/stage_5/seed_0/final.zip",
                6: "models_stage6_racing/stage_6/seed_0/final.zip",
                7: "models_stage7_yaw_only/stage_7/seed_0/final.zip",
                8: "models_stage8_progressive/stage_8/seed_0/final.zip",
            }
            model_path = model_paths.get(stage)
            if model_path is None:
                print(f"No default model for stage {stage}. Please specify --model.")
                return

        print(f"Loading model from: {model_path}")
        from stable_baselines3 import SAC

        try:
            custom_objects = {}
            if stage in [5, 6]:
                try:
                    from fastnn_quadrotor.training.train_sac_curriculum import SymmetricSACPolicy

                    custom_objects = {"policy": SymmetricSACPolicy}
                except ImportError:
                    pass
            model = SAC.load(
                model_path, custom_objects=custom_objects if custom_objects else None
            )
            print(f"Model loaded! Obs: {model.observation_space.shape}")
        except Exception as e:
            print(f"Failed: {e}")
            pd_only = True

    # Create environment
    env = RMAQuadrotorEnv(curriculum_stage=stage, max_episode_steps=max_steps)
    env.use_direct_control = pd_only

    if stage in [5, 6, 7, 8]:
        env.set_moving_target(True)
        env.set_target_speed(1.0)
        # Stage 7: yaw-only specific configuration
        if stage == 7:
            env.set_target_trajectory("figure8_yaw")
            env.set_yaw_only_mode(True)
            env.set_figure8_amplitude(3.0)
            print(f"Stage {stage}: Yaw-only mode with figure8_yaw trajectory")
        # Stage 7: yaw-only specific configuration
        if stage == 7:
            env.set_target_trajectory("figure8_yaw")
            env.set_yaw_only_mode(True)
            env.set_figure8_amplitude(3.0)
            print(f"Stage {stage}: Yaw-only mode with figure8_yaw trajectory")

    # Initialize terminal HUD
    hud = TerminalHUD()

    # Launch MuJoCo passive viewer
    handle = viewer.launch_passive(env.model, env.data)
    print("Viewer launched - 3D visualization active")
    handle.cam.azimuth = 135
    handle.cam.elevation = -20
    handle.cam.distance = 3.0
    handle.cam.lookat[:] = [0.0, 0.0, 1.0]

    stage_desc = {
        1: "Fixed hover",
        2: "Random pose",
        3: "Wind + mass",
        4: "Payload drop",
        5: "Moving target",
        6: "Racing circuit",
        7: "Yaw control",
        8: "Extended racing",
    }

    try:
        for ep in range(num_episodes):
            obs, _ = env.reset()
            step_count = 0
            episode_reward = 0

            print(f"\n{'=' * 50}")
            print(
                f"Episode {ep + 1} | Stage {stage} | {'PD Only' if pd_only else 'RL Policy'}"
            )
            print(f"  {stage_desc.get(stage, '?')} | Wind: {env.wind_force.round(2)}")

            while step_count < env.max_episode_steps:
                # Get action
                if pd_only:
                    action = np.zeros(4)
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                # Update terminal HUD (single line)
                pos = info["pos"]
                target_pos = env.target_pos if stage >= 5 else None
                hud.update(
                    step_count,
                    action,
                    pos,
                    target=target_pos,
                    reward=reward,
                    use_direct_control=env.use_direct_control,
                )

                # Render
                handle.sync()
                time.sleep(0.005)

                if terminated or truncated:
                    print(f"\n  --> Terminated: {terminated}, Truncated: {truncated}")
                    break

            print(f"\n  Episode reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted")

    handle.close()
    print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(description="Visualize with live terminal HUD")
    parser.add_argument("--stage", type=int, default=5)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--pd-only", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()

    run_visualization(
        model_path=args.model,
        stage=args.stage,
        pd_only=args.pd_only,
        num_episodes=args.episodes,
        max_steps=args.steps,
        deterministic=not args.stochastic,
    )


if __name__ == "__main__":
    main()
