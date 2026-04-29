#!/usr/bin/env python3
"""
Visualization with live motor HUD in separate matplotlib window
Shows motor commands updating in real-time while 3D viewer runs
"""

import argparse
import numpy as np
import mujoco
from mujoco import viewer
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_rma import RMAQuadrotorEnv

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("Warning: matplotlib not available - HUD disabled")


class MotorHUD:
    """Live motor values display using matplotlib."""

    def __init__(self):
        if not HAS_PLT:
            return
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.bars = self.ax.bar(
            ["M1", "M2", "M3", "M4"],
            [0, 0, 0, 0],
            color=["#e74c3c", "#2ecc71", "#3498db", "#f1c40f"],
        )
        self.ax.set_ylim(-1.5, 1.5)
        self.ax.set_ylabel("Command Value")
        self.ax.set_title("Motor Commands (Live)")
        self.ax.axhline(0, color="k", linestyle="-", linewidth=0.8)
        self.ax.grid(True, alpha=0.3)
        plt.ion()
        self.fig.canvas.draw_idle()
        self.fig.canvas.start_event_loop(0.001)

    def update(self, motor_vals):
        if not HAS_PLT:
            return
        for bar, val in zip(self.bars, motor_vals):
            bar.set_height(val)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


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
                    from train_sac_curriculum import SymmetricSACPolicy

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

    # Initialize HUD
    hud = MotorHUD() if HAS_PLT else None
    if hud:
        print("Motor HUD window opened - shows live motor commands")

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

                # Update HUD
                if hud:
                    hud.update(action)

                # Render
                handle.sync()
                time.sleep(0.005)

                # Terminal output every step
                if step_count % 1 == 0:
                    pos = info["pos"]
                    target_str = (
                        f" | target: [{env.target_pos[0]:.2f}, {env.target_pos[1]:.2f}, {env.target_pos[2]:.2f}]"
                        if stage >= 5
                        else ""
                    )
                    print(
                        f"  Step {step_count:3d} | pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]{target_str} | motors: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}] | r: {reward:.2f}",
                        flush=True,
                    )

                if terminated or truncated:
                    print(f"  --> Terminated: {terminated}, Truncated: {truncated}")
                    break

            print(f"  Episode reward: {episode_reward:.2f}")

    except KeyboardInterrupt:
        print("\nInterrupted")

    handle.close()
    if hud:
        plt.close(hud.fig)
    print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(description="Visualize with live motor HUD")
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
