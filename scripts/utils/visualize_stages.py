#!/usr/bin/env python3
"""
Comprehensive Stage Visualization Tool
Visualize performance of trained models across all stages and conditions.

Usage:
    python visualize_stages.py --stage 7          # Visualize Stage 7 yaw-only
    python visualize_stages.py --stage 8          # Visualize Stage 8 progressive
    python visualize_stages.py --stage 8 --phase 2.5  # Specific phase
    python visualize_stages.py --compare 7 8       # Compare stages side-by-side
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from fastnn_quadrotor.env_rma import RMAQuadrotorEnv
import os


def get_model_path(stage, seed=0):
    """Get model path for given stage."""
    paths = {
        1: f"models_stage1/stage_{stage}/seed_{seed}/final.zip",
        2: f"models_stage2/stage_{stage}/seed_{seed}/final.zip",
        3: f"models_stage3/stage_{stage}/seed_{seed}/final.zip",
        4: f"models_stage4/stage_{stage}/seed_{seed}/final.zip",
        5: f"models_stage5/stage_{stage}/seed_{seed}/final.zip",
        6: f"models_stage6/stage_{stage}/seed_{seed}/final.zip",
        7: f"models_stage7_yaw_only/stage_{stage}/seed_{seed}/final.zip",
        8: f"models_stage8_progressive/stage_{stage}/seed_{seed}/final.zip",
    }
    return paths.get(stage)


def evaluate_stage(model, stage, configs, n_episodes=10):
    """Evaluate model on given configurations."""
    results = {}
    for config in configs:
        traj = config["trajectory"]
        speed = config.get("speed", 0.1)
        name = config["name"]

        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        env.set_target_trajectory(traj)
        env.set_moving_target(traj != "static")
        env.set_target_speed(speed)

        successes = 0
        errors = []
        lengths = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            steps = 0
            ep_errors = []
            while not done and steps < 500:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                err = np.linalg.norm(env.data.qpos[:3] - env.target_pos)
                ep_errors.append(err)
            errors.extend(ep_errors)
            lengths.append(steps)
            if steps >= 500:
                successes += 1

        results[name] = {
            "success_rate": successes / n_episodes * 100,
            "mean_error": np.mean(errors),
            "mean_length": np.mean(lengths),
        }
    return results


def plot_stage_results(results, stage, ax=None):
    """Plot bar chart for stage results."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    names = list(results.keys())
    success_rates = [results[n]["success_rate"] for n in names]
    errors = [results[n]["mean_error"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width / 2, success_rates, width, label="Success %", alpha=0.8)
    ax2 = ax.twinx()
    bars2 = ax2.bar(
        x + width / 2, errors, width, label="Error (m)", alpha=0.8, color="orange"
    )

    ax.set_xlabel("Condition")
    ax.set_ylabel("Success Rate (%)", color="blue")
    ax2.set_ylabel("Tracking Error (m)", color="orange")
    ax.set_title(f"Stage {stage} Performance")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            f"{height:.0f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}m",
            ha="center",
            va="bottom",
            fontsize=8,
            color="orange",
        )

    return ax


def main():
    parser = argparse.ArgumentParser(description="Visualize Stage Training Results")
    parser.add_argument(
        "--stage", type=int, default=None, help="Stage number to visualize"
    )
    parser.add_argument(
        "--compare", type=int, nargs=2, default=None, help="Compare two stages"
    )
    parser.add_argument("--all", action="store_true", help="Show all stages")
    parser.add_argument("--seed", type=int, default=0, help="Seed number")
    args = parser.parse_args()

    if args.compare:
        # Compare two stages side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        for ax, stage in zip([ax1, ax2], args.compare):
            model_path = get_model_path(stage, args.seed)
            if not os.path.exists(model_path):
                ax.text(
                    0.5,
                    0.5,
                    f"Stage {stage} model\nnot found",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(f"Stage {stage}")
                continue
            model = SAC.load(model_path, device="cpu")
            print(f"Loaded Stage {stage} model ({model.num_timesteps} steps)")

            # Define test configs based on stage
            if stage == 7:
                configs = [
                    {
                        "trajectory": "figure8_yaw",
                        "speed": 0.1,
                        "name": "Yaw Fig8 0.1x",
                    },
                    {
                        "trajectory": "figure8_yaw",
                        "speed": 1.0,
                        "name": "Yaw Fig8 1.0x",
                    },
                    {
                        "trajectory": "figure8_yaw",
                        "speed": 5.0,
                        "name": "Yaw Fig8 5.0x",
                    },
                ]
            elif stage == 8:
                # Get curriculum history to determine which phases were trained
                import json

                hist_path = f"models_stage8_progressive/stage_{stage}/seed_{args.seed}/curriculum_history.json"
                if os.path.exists(hist_path):
                    with open(hist_path) as f:
                        hist = json.load(f)
                    phase_schedule = hist.get("phase_schedule", {})
                    print(f"  Phase schedule: {phase_schedule}")
                # Test key conditions
                configs = [
                    {"trajectory": "static", "speed": 0.1, "name": "Static"},
                    {"trajectory": "linear_short", "speed": 0.5, "name": "Linear 0.5x"},
                    {
                        "trajectory": "figure8_medium",
                        "speed": 0.15,
                        "name": "Med Fig8 0.15x",
                    },
                    {
                        "trajectory": "figure8_medium",
                        "speed": 0.5,
                        "name": "Med Fig8 0.5x",
                    },
                    {
                        "trajectory": "figure8_large",
                        "speed": 0.3,
                        "name": "Large Fig8 0.3x",
                    },
                    {
                        "trajectory": "figure8_large",
                        "speed": 0.5,
                        "name": "Large Fig8 0.5x",
                    },
                ]
            else:
                # Stages 1-6 use default trajectory and speed
                configs = [
                    {"trajectory": "figure8", "speed": 0.1, "name": "Fig8 0.1x"},
                    {"trajectory": "figure8", "speed": 1.0, "name": "Fig8 1.0x"},
                    {"trajectory": "racing", "speed": 1.0, "name": "Racing 1.0x"},
                ]

            results = evaluate_stage(model, stage, configs, n_episodes=10)
            plot_stage_results(results, stage, ax)
            ax.set_title(f"Stage {stage} Performance")

        plt.tight_layout()
        plt.savefig(
            f"stage{args.compare[0]}_vs_stage{args.compare[1]}_comparison.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.show()

    elif args.stage:
        # Single stage visualization
        model_path = get_model_path(args.stage, args.seed)
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            return

        model = SAC.load(model_path, device="cpu")
        print(f"Loaded Stage {args.stage} model: {model.num_timesteps} steps")

        # Define test configs
        if args.stage == 7:
            configs = [
                {"trajectory": "figure8_yaw", "speed": 0.1, "name": "Yaw 0.1x"},
                {"trajectory": "figure8_yaw", "speed": 0.5, "name": "Yaw 0.5x"},
                {"trajectory": "figure8_yaw", "speed": 1.0, "name": "Yaw 1.0x"},
                {"trajectory": "figure8_yaw", "speed": 2.0, "name": "Yaw 2.0x"},
                {"trajectory": "figure8_yaw", "speed": 5.0, "name": "Yaw 5.0x"},
            ]
        elif args.stage == 8:
            configs = [
                {"trajectory": "static", "speed": 0.1, "name": "Static Hover"},
                {"trajectory": "linear_short", "speed": 0.1, "name": "Linear 0.1x"},
                {"trajectory": "linear_short", "speed": 0.5, "name": "Linear 0.5x"},
                {
                    "trajectory": "figure8_medium",
                    "speed": 0.15,
                    "name": "Med Fig8 0.15x",
                },
                {"trajectory": "figure8_medium", "speed": 0.3, "name": "Med Fig8 0.3x"},
                {"trajectory": "figure8_medium", "speed": 0.5, "name": "Med Fig8 0.5x"},
                {
                    "trajectory": "figure8_large",
                    "speed": 0.3,
                    "name": "Large Fig8 0.3x",
                },
                {
                    "trajectory": "figure8_large",
                    "speed": 0.5,
                    "name": "Large Fig8 0.5x",
                },
                {"trajectory": "extended", "speed": 0.5, "name": "Extended 0.5x"},
                {"trajectory": "extended", "speed": 1.0, "name": "Extended 1.0x"},
                {"trajectory": "extended", "speed": 2.0, "name": "Extended 2.0x"},
            ]
        else:
            configs = [
                {"trajectory": "figure8", "speed": 0.1, "name": "Fig8 0.1x"},
                {"trajectory": "figure8", "speed": 0.5, "name": "Fig8 0.5x"},
                {"trajectory": "figure8", "speed": 1.0, "name": "Fig8 1.0x"},
            ]

        results = evaluate_stage(model, args.stage, configs, n_episodes=10)

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_stage_results(results, args.stage, ax)
        plt.tight_layout()
        plt.savefig(f"stage{args.stage}_performance.png", dpi=150, bbox_inches="tight")
        plt.show()

        print(f"\nStage {args.stage} Results:")
        for name, data in results.items():
            print(
                f"  {name}: {data['success_rate']:.1f}% success, {data['mean_error']:.3f}m error"
            )

    else:
        print("Please specify --stage or --compare")
        parser.print_help()


if __name__ == "__main__":
    main()
