#!/usr/bin/env python3
"""Final evaluation of 1.5M-step Stage 8 model."""

import numpy as np
from stable_baselines3 import SAC
from env_rma import RMAQuadrotorEnv


def evaluate(model, trajectory, speed, n_episodes=10, max_steps=500):
    env = RMAQuadrotorEnv(curriculum_stage=8, use_direct_control=True)
    env.set_target_trajectory(trajectory)
    env.set_moving_target(trajectory != "static")
    env.set_target_speed(speed)

    succ = 0
    errors = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        ep_err = []
        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            err = np.linalg.norm(env.data.qpos[:3] - env.target_pos)
            ep_err.append(err)
        errors.extend(ep_err)
        if steps >= max_steps:
            succ += 1
    return succ / n_episodes * 100, np.mean(errors)


def main():
    model = SAC.load("models_stage8_progressive/stage_8/seed_0/final.zip", device="cpu")
    print(f"Model steps: {model.num_timesteps}")
    tests = [
        ("static", 0.1, "Phase1: Static", 10),
        ("linear_short", 0.1, "Phase2: Lin0.1x", 10),
        ("linear_short", 0.5, "Phase2: Lin0.5x", 10),
        ("figure8_medium", 0.15, "Phase2.5: MedF8 0.15x", 10),
        ("figure8_medium", 0.3, "Phase2.5: MedF8 0.3x", 10),
        ("figure8_medium", 0.5, "Phase2.5: MedF8 0.5x", 10),
        ("figure8_large", 0.3, "Phase3: LargeF8 0.3x", 10),
        ("figure8_large", 0.5, "Phase3: LargeF8 0.5x", 10),
        # Phase 4: extended trajectory at various speeds
        ("extended", 0.5, "Phase4: Ext 0.5x", 5),
        ("extended", 1.0, "Phase4: Ext 1.0x", 5),
        ("extended", 2.0, "Phase4: Ext 2.0x", 5),
        ("extended", 3.0, "Phase4: Ext 3.0x", 5),
    ]

    print(f"{'Condition':<30} {'Success':>8} {'Error':>10}")
    print("-" * 50)
    for traj, speed, name, n in tests:
        try:
            s, e = evaluate(model, traj, speed, n_episodes=n)
            print(f"{name:<30} {s:>6.1f}% {e:>10.3f}m")
        except Exception as ex:
            print(f"{name:<30} ERROR: {ex}")


if __name__ == "__main__":
    main()
