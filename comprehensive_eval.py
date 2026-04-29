#!/usr/bin/env python3
"""
Comprehensive Evaluation Script

Fixes applied:
- PD baseline uses env's internal state directly (not parsed from obs)
- PD baseline evaluated via direct control mode (not residual)
- Correct observation layout for 60-dim obs
- Multi-seed evaluation support
- Failure bound now measures distance from target (not origin)
"""

import numpy as np
import torch
import json
import os
from datetime import datetime
from stable_baselines3 import SAC

from train_sac_curriculum import AsymmetricSACPolicy
from env_rma import RMAQuadrotorEnv
from baseline_controllers import PDController


def evaluate_model(env, model, n_episodes=50, deterministic=True, track_drop_time=False):
    """Evaluate an RL model on the environment."""
    rewards = []
    successes = 0
    max_tilts = []
    episode_lengths = []
    drop_times = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        max_tilt = 0.0
        steps = 0
        drop_time_recorded = None

        while steps < env.max_episode_steps:
            # Pass full 60-dim obs; DeployableExtractor slices to 52 internally
            action, _ = model.predict(obs[np.newaxis], deterministic=deterministic)
            action = action[0]

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1

            if "rpy" in info:
                tilt = np.rad2deg(max(abs(info["rpy"][0]), abs(info["rpy"][1])))
                max_tilt = max(max_tilt, tilt)

            if track_drop_time and info.get("drop_occurred", False) and drop_time_recorded is None:
                drop_time_recorded = steps * 0.01

            if terminated or truncated:
                break

        successes += (1 if steps >= env.max_episode_steps else 0)
        rewards.append(episode_reward)
        max_tilts.append(max_tilt)
        episode_lengths.append(steps)
        if track_drop_time:
            drop_times.append(drop_time_recorded)

        status = "SUCCESS" if steps >= env.max_episode_steps else f"FAIL@{steps}"
        print(f"  Ep {ep+1:2d}: {status:12s} | reward={episode_reward:8.1f} | tilt={max_tilt:5.1f}° | len={steps}")

    return {
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": 100 * successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_max_tilt": float(np.mean(max_tilts)),
        "max_tilt_recorded": float(np.max(max_tilts)),
        "mean_ep_length": float(np.mean(episode_lengths)),
        "rewards": rewards,
        "max_tilts": max_tilts,
        "episode_lengths": episode_lengths,
        "drop_times": drop_times if track_drop_time else None,
    }


def evaluate_pd_standalone(env, pd_ctrl, n_episodes=30):
    """Evaluate a standalone PD controller using direct control mode.

    Uses env's internal state directly (not parsed from observation),
    and applies PD output directly (not as residual).
    """
    rewards = []
    successes = 0
    max_tilts = []
    episode_lengths = []

    for ep in range(n_episodes):
        # Reset the env and the PD controller
        obs, _ = env.reset()
        pd_ctrl.reset()
        episode_reward = 0.0
        max_tilt = 0.0
        steps = 0

        while steps < env.max_episode_steps:
            # Read state directly from MuJoCo (not from obs vector)
            pos = env.data.qpos[:3].copy()
            vel = env.data.qvel[:3].copy()
            quat = env.data.qpos[3:7].copy()
            ang_vel = env.data.qvel[3:6].copy()

            # PD computes: [thrust, roll_torque, pitch_torque, yaw_torque]
            ctrl = pd_ctrl.compute(pos, vel, quat, ang_vel, mass=env._mass_hat)

            # Convert PD output to action space [-1, 1] for direct control mode
            # Direct control scales: thrust (a+1)*10, roll/pitch a*3, yaw a*2
            action = np.array([
                ctrl[0] / 10.0 - 1.0,  # thrust: [0,20] → [-1,1]
                ctrl[1] / 3.0,          # roll: [-3,3] → [-1,1]
                ctrl[2] / 3.0,          # pitch: [-3,3] → [-1,1]
                ctrl[3] / 2.0,          # yaw: [-2,2] → [-1,1]
            ])
            action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1

            if "rpy" in info:
                tilt = np.rad2deg(max(abs(info["rpy"][0]), abs(info["rpy"][1])))
                max_tilt = max(max_tilt, tilt)

            if terminated or truncated:
                break

        successes += (1 if steps >= env.max_episode_steps else 0)
        rewards.append(episode_reward)
        max_tilts.append(max_tilt)
        episode_lengths.append(steps)

        status = "SUCCESS" if steps >= env.max_episode_steps else f"FAIL@{steps}"
        print(f"  Ep {ep+1:2d}: {status:12s} | reward={episode_reward:8.1f} | tilt={max_tilt:5.1f}° | len={steps}")

    return {
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": 100 * successes / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_max_tilt": float(np.mean(max_tilts)),
        "max_tilt_recorded": float(np.max(max_tilts)),
        "mean_ep_length": float(np.mean(episode_lengths)),
    }


def run_generalization_tests(model_path=None):
    """Test generalization to out-of-distribution perturbations."""
    print("\n" + "=" * 70)
    print("GENERALIZATION TESTS")
    print("=" * 70)

    if model_path is None:
        model_path = "models/seed_0/final.zip"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, skipping...")
        return {}

    model = SAC.load(model_path, device="cpu", custom_objects=dict(policy=AsymmetricSACPolicy))
    results = {}

    # Test 1: 2x wind
    print("\n--- Test 1: 2x Wind Magnitude (+/-1.0 N) ---")
    env = RMAQuadrotorEnv(curriculum_stage=3)

    orig_reset = env.reset
    def double_wind_reset():
        obs, info = orig_reset()
        env.wind_force = np.random.uniform(-1.0, 1.0, size=3)
        return obs, info
    env.reset = double_wind_reset

    result = evaluate_model(env, model, n_episodes=30)
    results["2x_wind"] = result
    print(f"  2x Wind: {result['success_rate']:.1f}% success, tilt={result['mean_max_tilt']:.1f}°")

    # Test 2: 50% mass drop
    print("\n--- Test 2: 50% Mass Drop ---")
    result = evaluate_50pct_drop(model, n_episodes=30)
    results["50pct_drop"] = result
    print(f"  50% Drop: {result['success_rate']:.1f}% success, tilt={result['mean_max_tilt']:.1f}°")

    # Test 3: Combined extreme
    print("\n--- Test 3: 2x Wind + 50% Drop ---")
    result = evaluate_extreme(model, n_episodes=30)
    results["extreme_combined"] = result
    print(f"  Extreme: {result['success_rate']:.1f}% success, tilt={result['mean_max_tilt']:.1f}°")

    # Test 4: No perturbations
    print("\n--- Test 4: No Perturbations Baseline ---")
    env = RMAQuadrotorEnv(curriculum_stage=1)
    result = evaluate_model(env, model, n_episodes=30)
    results["no_perturb"] = result
    print(f"  No Perturb: {result['success_rate']:.1f}% success, tilt={result['mean_max_tilt']:.1f}°")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("benchmark_results", f"generalization_{timestamp}.json")
    save_data = {k: {kk: vv if not isinstance(vv, (np.ndarray, list)) else vv
                     for kk, vv in v.items() if kk not in ['rewards', 'max_tilts', 'drop_times']}
                 for k, v in results.items()}
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")

    return results


def evaluate_50pct_drop(model, n_episodes=30):
    """Evaluate with forced 50% mass drop."""
    env = RMAQuadrotorEnv(curriculum_stage=4)

    successes = 0
    rewards_list = []
    max_tilts = []
    episode_lengths = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        env.drop_time = 3.0
        env.drop_magnitude = 0.50
        env.com_shift = np.zeros(3)
        env.drop_triggered = False
        env.drop_occurred = False
        env.model.body_mass[1] = env.nominal_mass  # start at nominal

        episode_reward = 0.0
        max_tilt = 0.0
        steps = 0

        while steps < env.max_episode_steps:
            action, _ = model.predict(obs[np.newaxis], deterministic=True)
            action = action[0]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1

            if "rpy" in info:
                tilt = np.rad2deg(max(abs(info["rpy"][0]), abs(info["rpy"][1])))
                max_tilt = max(max_tilt, tilt)

            if terminated or truncated:
                break

        successes += (1 if steps >= env.max_episode_steps else 0)
        rewards_list.append(episode_reward)
        max_tilts.append(max_tilt)
        episode_lengths.append(steps)

        status = "SUCCESS" if steps >= env.max_episode_steps else f"FAIL@{steps}"
        print(f"  Ep {ep+1:2d}: {status:12s} | reward={episode_reward:8.1f} | tilt={max_tilt:5.1f}° | len={steps}")

    env.close()

    return {
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": 100 * successes / n_episodes,
        "mean_reward": float(np.mean(rewards_list)),
        "std_reward": float(np.std(rewards_list)),
        "mean_max_tilt": float(np.mean(max_tilts)),
        "max_tilt_recorded": float(np.max(max_tilts)),
        "mean_ep_length": float(np.mean(episode_lengths)),
    }


def evaluate_extreme(model, n_episodes=30):
    """Evaluate with 2x wind + 50% drop."""
    successes = 0
    rewards_list = []
    max_tilts = []
    episode_lengths = []

    for ep in range(n_episodes):
        env = RMAQuadrotorEnv(curriculum_stage=4)
        obs, _ = env.reset()

        # Apply extreme conditions
        env.wind_force = np.random.uniform(-1.0, 1.0, size=3)
        env.drop_time = 3.0
        env.drop_magnitude = 0.50
        env.com_shift = np.zeros(3)
        env.drop_triggered = False
        env.drop_occurred = False
        env.model.body_mass[1] = env.nominal_mass

        episode_reward = 0.0
        max_tilt = 0.0
        steps = 0

        while steps < env.max_episode_steps:
            action, _ = model.predict(obs[np.newaxis], deterministic=True)
            action = action[0]
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            steps += 1

            if "rpy" in info:
                tilt = np.rad2deg(max(abs(info["rpy"][0]), abs(info["rpy"][1])))
                max_tilt = max(max_tilt, tilt)

            if terminated or truncated:
                break

        successes += (1 if steps >= env.max_episode_steps else 0)
        rewards_list.append(episode_reward)
        max_tilts.append(max_tilt)
        episode_lengths.append(steps)

        status = "SUCCESS" if steps >= env.max_episode_steps else f"FAIL@{steps}"
        print(f"  Ep {ep+1:2d}: {status:12s} | reward={episode_reward:8.1f} | tilt={max_tilt:5.1f}° | len={steps}")

        env.close()

    return {
        "n_episodes": n_episodes,
        "successes": successes,
        "success_rate": 100 * successes / n_episodes,
        "mean_reward": float(np.mean(rewards_list)),
        "std_reward": float(np.std(rewards_list)),
        "mean_max_tilt": float(np.mean(max_tilts)),
        "max_tilt_recorded": float(np.max(max_tilts)),
        "mean_ep_length": float(np.mean(episode_lengths)),
    }


def run_pd_analysis():
    """Analyze PD controller behavior using direct control mode."""
    print("\n" + "=" * 70)
    print("PD CONTROLLER ANALYSIS (Direct Control Mode)")
    print("=" * 70)

    results_dir = "benchmark_results"
    os.makedirs(results_dir, exist_ok=True)

    all_results = {}

    for stage in [1, 2, 3, 4]:
        stage_names = {1: "Fixed Hover", 2: "Random Pose", 3: "Wind+Mass", 4: "Payload Drop"}
        print(f"\n--- Stage {stage}: {stage_names[stage]} ---")

        # Use direct control mode for standalone PD evaluation
        env = RMAQuadrotorEnv(curriculum_stage=stage, use_direct_control=True)
        pd_ctrl = PDController()
        result = evaluate_pd_standalone(env, pd_ctrl, n_episodes=30)
        all_results[f"stage_{stage}"] = result

        print(f"  Success: {result['success_rate']:.1f}%, Tilt: {result['mean_max_tilt']:.1f}° "
              f"(max: {result['max_tilt_recorded']:.1f}°)")
        print(f"  Reward: {result['mean_reward']:.1f} +/- {result['std_reward']:.1f}")
        env.close()

    # Detailed trajectory
    print("\n--- Detailed PD Trajectory Analysis (Stage 1) ---")
    env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
    obs, _ = env.reset()
    pd_ctrl = PDController(kp_pos=8.0, kd_pos=4.0, kp_att=5.0, kd_att=2.0)
    pd_ctrl.reset()

    rolls, pitches, actions_hist = [], [], []
    steps = 0

    while steps < 500:
        pos = env.data.qpos[:3].copy()
        vel = env.data.qvel[:3].copy()
        quat = env.data.qpos[3:7].copy()
        ang_vel = env.data.qvel[3:6].copy()

        ctrl = pd_ctrl.compute(pos, vel, quat, ang_vel, mass=env._mass_hat)
        action = np.array([
            ctrl[0] / 10.0 - 1.0,
            ctrl[1] / 3.0,
            ctrl[2] / 3.0,
            ctrl[3] / 2.0,
        ])
        action = np.clip(action, -1.0, 1.0)

        obs, reward, terminated, truncated, info = env.step(action)
        rpy = info.get("rpy", [0, 0, 0])
        rolls.append(rpy[0])
        pitches.append(rpy[1])
        actions_hist.append(ctrl)
        steps += 1
        if terminated or truncated:
            break

    rolls, pitches = np.array(rolls), np.array(pitches)
    actions_hist = np.array(actions_hist)

    print(f"\n  Episode length: {steps}")
    print(f"  Roll:  mean={np.rad2deg(rolls).mean():.2f}°, std={np.rad2deg(rolls).std():.2f}°, "
          f"max_abs={np.rad2deg(np.abs(rolls)).max():.2f}°")
    print(f"  Pitch: mean={np.rad2deg(pitches).mean():.2f}°, std={np.rad2deg(pitches).std():.2f}°, "
          f"max_abs={np.rad2deg(np.abs(pitches)).max():.2f}°")
    if len(actions_hist) > 0:
        print(f"  Thrust: mean={actions_hist[:, 0].mean():.2f}, std={actions_hist[:, 0].std():.2f}")
        print(f"  Roll torque: mean={actions_hist[:, 1].mean():.4f}, std={actions_hist[:, 1].std():.4f}")

    env.close()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(results_dir, f"pd_analysis_{timestamp}.json")
    save_data = {k: {kk: vv if not isinstance(vv, (np.ndarray, list)) else vv
                     for kk, vv in v.items() if kk not in ['rewards', 'max_tilts']}
                 for k, v in all_results.items()}
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved: {save_path}")

    return all_results


def run_drop_timing_analysis(model_path=None):
    """Analyze drop timing correlation."""
    print("\n" + "=" * 70)
    print("DROP TIMING ANALYSIS")
    print("=" * 70)

    if model_path is None:
        model_path = "models/seed_0/final.zip"

    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}, skipping...")
        return {}

    model = SAC.load(model_path, device="cpu", custom_objects=dict(policy=AsymmetricSACPolicy))

    drop_times_to_test = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    results = {}

    for drop_time in drop_times_to_test:
        print(f"\n--- Drop at t={drop_time}s ---")

        successes = 0
        max_tilts = []

        for ep in range(20):
            env = RMAQuadrotorEnv(curriculum_stage=4)
            obs, _ = env.reset()

            env.drop_time = drop_time
            env.drop_magnitude = 0.30
            env.drop_triggered = False
            env.drop_occurred = False
            env.model.body_mass[1] = env.nominal_mass

            max_tilt = 0.0
            steps = 0

            while steps < env.max_episode_steps:
                action, _ = model.predict(obs[np.newaxis], deterministic=True)
                action = action[0]
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1

                if "rpy" in info:
                    tilt = np.rad2deg(max(abs(info["rpy"][0]), abs(info["rpy"][1])))
                    max_tilt = max(max_tilt, tilt)

                if terminated or truncated:
                    break

            successes += (1 if steps >= env.max_episode_steps else 0)
            max_tilts.append(max_tilt)

            status = "SUCCESS" if steps >= env.max_episode_steps else f"FAIL@{steps}"
            print(f"  Ep {ep+1:2d}: {status} | tilt={max_tilt:5.1f}° | len={steps}")

            env.close()

        results[f"drop_at_{drop_time}s"] = {
            "success_rate": 100 * successes / 20,
            "mean_max_tilt": float(np.mean(max_tilts)),
        }
        print(f"  ► Success: {results[f'drop_at_{drop_time}s']['success_rate']:.1f}%, "
              f"tilt={results[f'drop_at_{drop_time}s']['mean_max_tilt']:.1f}°")

    # Correlation
    print("\n--- Drop Timing Correlation ---")
    times = []
    success_rates = []
    for key, res in sorted(results.items()):
        t = float(key.split("_")[2][:-1])
        times.append(t)
        success_rates.append(res['success_rate'])

    if len(times) > 1:
        correlation = np.corrcoef(times, success_rates)[0, 1]
        print(f"  Correlation (drop_time vs success): {correlation:.3f}")
        print(f"  Interpretation: {'Later drops are easier' if correlation > 0 else 'Earlier drops are harder'}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join("benchmark_results", f"drop_timing_{timestamp}.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {save_path}")

    return results


def run_multiseed_eval(seeds=None, stage=4):
    """Evaluate models trained with different seeds."""
    print("\n" + "=" * 70)
    print("MULTI-SEED EVALUATION")
    print("=" * 70)

    if seeds is None:
        seeds = [0, 1, 2]

    all_results = {}

    for seed in seeds:
        model_path = f"models/seed_{seed}/final.zip"
        if not os.path.exists(model_path):
            print(f"  Seed {seed}: model not found at {model_path}, skipping...")
            continue

        print(f"\n--- Seed {seed} ---")
        model = SAC.load(model_path, device="cpu", custom_objects=dict(policy=AsymmetricSACPolicy))
        env = RMAQuadrotorEnv(curriculum_stage=stage)
        result = evaluate_model(env, model, n_episodes=50)
        all_results[f"seed_{seed}"] = result
        print(f"  Success: {result['success_rate']:.1f}%, Tilt: {result['mean_max_tilt']:.1f}°")

    # Compute mean ± std across seeds
    if len(all_results) >= 2:
        success_rates = [r["success_rate"] for r in all_results.values()]
        mean_success = np.mean(success_rates)
        std_success = np.std(success_rates)
        print(f"\n  Multi-seed: {mean_success:.1f}% ± {std_success:.1f}% success rate")

    return all_results


def print_summary(multiseed, gen, pd_res, drop_timing):
    print("\n" + "=" * 70)
    print("SUMMARY: MULTI-MODEL EVALUATION (Stage 4)")
    print("=" * 70)
    print(f"{'Model':<15} {'Success':>10} {'Tilt':>10} {'Reward':>12}")
    print("-" * 70)
    for name, res in sorted(multiseed.items()):
        print(f"{name:<15} {res['success_rate']:>9.1f}% {res['mean_max_tilt']:>9.1f}° {res['mean_reward']:>11.0f}")

    # Compute mean ± std if multiple seeds
    if len(multiseed) >= 2:
        success_rates = [r["success_rate"] for r in multiseed.values()]
        print(f"{'MEAN ± STD':<15} {np.mean(success_rates):>8.1f}±{np.std(success_rates):.1f}%")

    if gen:
        print("\n" + "=" * 70)
        print("SUMMARY: GENERALIZATION TESTS")
        print("=" * 70)
        print(f"{'Test':<25} {'Success':>10} {'Tilt':>10} {'Reward':>12}")
        print("-" * 70)
        test_names = {"2x_wind": "2x Wind (+/-1.0 N)", "50pct_drop": "50% Mass Drop",
                      "extreme_combined": "2x Wind + 50% Drop", "no_perturb": "No Perturbations"}
        for name, res in sorted(gen.items()):
            print(f"{test_names.get(name, name):<25} {res['success_rate']:>9.1f}% "
                  f"{res['mean_max_tilt']:>9.1f}° {res.get('mean_reward', 0):>11.0f}")

    if pd_res:
        print("\n" + "=" * 70)
        print("SUMMARY: PD CONTROLLER ANALYSIS (Direct Control)")
        print("=" * 70)
        print(f"{'Stage':<15} {'Success':>10} {'Tilt':>10}")
        print("-" * 70)
        for name, res in sorted(pd_res.items()):
            print(f"{name:<15} {res['success_rate']:>9.1f}% {res['mean_max_tilt']:>9.1f}°")

    if drop_timing:
        print("\n" + "=" * 70)
        print("SUMMARY: DROP TIMING ANALYSIS")
        print("=" * 70)
        print(f"{'Drop Time':>10} {'Success':>10} {'Tilt':>10}")
        print("-" * 70)
        for name, res in sorted(drop_timing.items()):
            t = name.split("_")[2][:-1]
            print(f"{t:>9}s {res['success_rate']:>9.1f}% {res['mean_max_tilt']:>9.1f}°")


def main():
    print("=" * 70)
    print("COMPREHENSIVE QUADROTOR EVALUATION")
    print("=" * 70)

    # Multi-seed evaluation
    print("\n[1/4] Running multi-seed evaluation...")
    multiseed = run_multiseed_eval(seeds=[0, 1, 2])

    print("\n[2/4] Running generalization tests...")
    gen = run_generalization_tests()

    print("\n[3/4] Running PD analysis (direct control mode)...")
    pd_res = run_pd_analysis()

    print("\n[4/4] Running drop timing analysis...")
    drop_timing = run_drop_timing_analysis()

    print_summary(multiseed, gen, pd_res, drop_timing)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()