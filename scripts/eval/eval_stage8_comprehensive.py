#!/usr/bin/env python3
"""
Comprehensive Evaluation of Stage 8 Model (250k steps)
Tests across all phases and speed conditions
"""

import numpy as np
from stable_baselines3 import SAC
from fastnn_quadrotor.env_rma import RMAQuadrotorEnv


def evaluate_model(model, env_config, n_episodes=20, max_steps=500, description=""):
    """Evaluate model on given environment configuration."""
    print(f"\n{'=' * 60}")
    print(f"{description}")
    print(f"{'=' * 60}")

    env = RMAQuadrotorEnv(curriculum_stage=8, use_direct_control=True)
    env.set_target_trajectory(env_config["trajectory"])
    env.set_moving_target(env_config.get("moving", False))
    if "speed" in env_config:
        env.set_target_speed(env_config["speed"])

    successes = 0
    lengths = []
    all_errors = []
    ep_errors_mean = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_errors = []

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            pos = env.data.qpos[:3]
            target = env.target_pos
            error = np.linalg.norm(pos - target)
            ep_errors.append(error)

        lengths.append(steps)
        all_errors.extend(ep_errors)
        ep_errors_mean.append(np.mean(ep_errors))

        if steps >= max_steps:
            successes += 1

    success_rate = successes / n_episodes * 100
    mean_error = np.mean(all_errors)
    std_error = np.std(all_errors)
    mean_episode_len = np.mean(lengths)

    print(f"Success Rate: {success_rate:.1f}% ({successes}/{n_episodes})")
    print(f"Mean Episode Length: {mean_episode_len:.1f} steps")
    print(f"Mean Tracking Error: {mean_error:.3f} ± {std_error:.3f} m")
    print(
        f"Per-episode errors: min={np.min(ep_errors_mean):.3f}, max={np.max(ep_errors_mean):.3f}"
    )

    return {
        "success_rate": success_rate,
        "mean_error": mean_error,
        "std_error": std_error,
        "mean_length": mean_episode_len,
        "all_errors": all_errors,
    }


def pd_baseline(env_config, n_episodes=10, max_steps=500):
    """Run PD baseline (zero action)."""
    env = RMAQuadrotorEnv(curriculum_stage=8, use_direct_control=True)
    env.set_target_trajectory(env_config["trajectory"])
    env.set_moving_target(env_config.get("moving", False))
    if "speed" in env_config:
        env.set_target_speed(env_config["speed"])

    lengths = []
    errors = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_errors = []

        while not done and steps < max_steps:
            action = np.zeros(4)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            pos = env.data.qpos[:3]
            target = env.target_pos
            error = np.linalg.norm(pos - target)
            ep_errors.append(error)

        lengths.append(steps)
        errors.extend(ep_errors)

    return {
        "success_rate": np.mean([l >= max_steps for l in lengths]) * 100,
        "mean_error": np.mean(errors),
        "mean_length": np.mean(lengths),
    }


def main():
    print("=" * 70)
    print("STAGE 8 MODEL EVALUATION (250k steps)")
    print("=" * 70)

    # Load model
    model = SAC.load("models_stage8_progressive/stage_8/seed_0/final.zip", device="cpu")
    print(f"Model trained for: {model.num_timesteps} steps")

    # Test configurations
    configs = [
        {
            "name": "Phase 1: Static Hover",
            "config": {"trajectory": "static", "moving": False},
        },
        {
            "name": "Phase 2: Linear Short (0.1x)",
            "config": {"trajectory": "linear_short", "moving": True, "speed": 0.1},
        },
        {
            "name": "Phase 2: Linear Short (0.2x)",
            "config": {"trajectory": "linear_short", "moving": True, "speed": 0.2},
        },
        {
            "name": "Phase 2: Linear Short (0.5x)",
            "config": {"trajectory": "linear_short", "moving": True, "speed": 0.5},
        },
        {
            "name": "Phase 3: Extended (0.1x) - NOT YET TRAINED",
            "config": {"trajectory": "extended", "moving": True, "speed": 0.1},
        },
        {
            "name": "Phase 3: Extended (0.2x) - NOT YET TRAINED",
            "config": {"trajectory": "extended", "moving": True, "speed": 0.2},
        },
    ]

    results = {}
    for test in configs:
        try:
            result = evaluate_model(
                model, test["config"], n_episodes=20, description=test["name"]
            )
            results[test["name"]] = result

            # Compare to PD baseline
            baseline = pd_baseline(test["config"], n_episodes=10)
            print(
                f"\nPD Baseline: {baseline['success_rate']:.1f}% success, {baseline['mean_error']:.3f}m error"
            )
            improvement = (
                baseline["mean_error"] / result["mean_error"]
                if result["mean_error"] > 0
                else float("inf")
            )
            print(f"Model improvement over PD: {improvement:.2f}x\n")

        except Exception as e:
            print(f"ERROR in {test['name']}: {e}")
            results[test["name"]] = None

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Condition':<35} {'Success':>8} {'Error':>10} {'Len':>6}")
    print("-" * 70)

    for test_name, result in results.items():
        if result:
            print(
                f"{test_name:<35} {result['success_rate']:>6.1f}% "
                f"{result['mean_error']:>10.3f}m {result['mean_length']:>6.1f}"
            )
        else:
            print(f"{test_name:<35} {'FAILED':>8}")

    return results


if __name__ == "__main__":
    main()
