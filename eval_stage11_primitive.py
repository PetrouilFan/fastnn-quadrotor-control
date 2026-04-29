#!/usr/bin/env python3
"""
Evaluate Stage 11 Primitive model with distance-based bucketing.
"""

import os
import sys
import argparse
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC

from quadrotor.env_stage11_primitive import QuadrotorEnvStage11Primitive


DISTANCE_BUCKETS = {
    "easy": (0, 2.0),      # 0-2m from start
    "medium": (2.0, 4.0),  # 2-4m
    "hard": (4.0, 8.0),     # 4-8m
}


def run_evaluation(model_path, stage10_path, vecnorm_path, episodes=30):
    """Run evaluation with distance bucketing."""
    
    model = SAC.load(model_path)
    
    results = {bucket: {"ctes": [], "bucket": bucket} for bucket in DISTANCE_BUCKETS}
    
    print(f"\n=== EVALUATING {episodes} EPISODES ===")
    
    all_ctes = []
    
    for ep in range(episodes):
        env = QuadrotorEnvStage11Primitive(
            stage10_model_path=stage10_path,
            stage10_vecnormalize_path=vecnorm_path,
            max_episode_steps=5000,
        )
        
        env.reset()
        done = False
        episode_ctes = []
        
        while not done:
            action, _ = model.predict(env._get_obs(), deterministic=False)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            cte = info.get("tracking_error", 0.0)
            episode_ctes.append(cte)
        
        mean_cte = np.mean(episode_ctes)
        p95_cte = np.percentile(episode_ctes, 95)
        all_ctes.extend(episode_ctes)
        
        # Bucket by goal distance
        goal_dist = np.linalg.norm(env._goal_pos - env._episode_start)
        for bucket_name, (low, high) in DISTANCE_BUCKETS.items():
            if low <= goal_dist < high:
                results[bucket_name]["ctes"].extend(episode_ctes)
                break
        
        if (ep + 1) % 10 == 0:
            print(f"Ep {ep+1}: mean={mean_cte:.3f}m, p95={p95_cte:.3f}m")
    
    print("\n=== DISTANCE BREAKDOWN ===")
    for bucket_name, (low, high) in DISTANCE_BUCKETS.items():
        ctes = results[bucket_name]["ctes"]
        if ctes:
            ctes = np.array(ctes)
            print(f"{bucket_name} ({low}-{high}m): mean={np.mean(ctes):.3f}m, p95={np.percentile(ctes, 95):.3f}m")
    
    all_ctes = np.array(all_ctes)
    print(f"\nOVERALL: mean={np.mean(all_ctes):.3f}m, p95={np.percentile(all_ctes, 95):.3f}m")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--stage10-path", type=str, required=True)
    parser.add_argument("--vecnorm-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=30)
    args = parser.parse_args()
    
    vecnorm_path = args.vecnorm_path or args.stage10_path.replace(".zip", "_vecnormalize.pkl")
    
    run_evaluation(
        args.model_path,
        args.stage10_path,
        vecnorm_path,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()