#!/usr/bin/env python3
"""
Train Stage 15: Adaptive SAC with Latent Adaptation Encoder.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env_stage15_adaptive import QuadrotorEnvStage15Adaptive
from quadrotor.env_stage11_trajectory import QuadrotorEnvStage11Trajectory
from quadrotor.env_stage12_hist import QuadrotorEnvStage12Hist


def make_env(s10, vecnorm, encoder_path):
    def _init():
        base = QuadrotorEnvStage11Trajectory(s10, vecnorm)
        hist = QuadrotorEnvStage12Hist(base)
        return QuadrotorEnvStage15Adaptive(hist, encoder_path)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-path", type=str, required=True)
    parser.add_argument("--encoder-path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--continue-from", type=str, default=None, help="Continue training from checkpoint")
    args = parser.parse_args()

    s10 = args.stage10_path
    vecnorm = s10.replace(".zip", "_vecnormalize.pkl")
    if not os.path.exists(vecnorm):
        vecnorm = s10.replace(".zip", "_vec_normalize.pkl")
    os.makedirs(args.save_path, exist_ok=True)

    print("=" * 80)
    print("STAGE 15 ADAPTIVE SAC TRAINING")
    print("=" * 80)
    print(f"  - Latent dynamics token (32-dim)")
    print(f"  - Frozen adaptation encoder")
    print(f"  - Off-policy SAC")
    print("=" * 80)

    train_env = DummyVecEnv([
        make_env(s10, vecnorm, args.encoder_path) for _ in range(4)
    ])
    eval_env = DummyVecEnv([
        make_env(s10, vecnorm, args.encoder_path) for _ in range(2)
    ])

    for e in train_env.envs:
        e.reset(seed=0)
    for e in eval_env.envs:
        e.reset(seed=100)

    dummy_env = make_env(s10, vecnorm, args.encoder_path)()
    obs_space = dummy_env.observation_space
    print(f"Observation space: {obs_space}")
    print(f"  - Keys: {list(obs_space.keys())}")

    # Continue from checkpoint or create new model
    if args.continue_from and os.path.exists(args.continue_from):
        print(f"Continuing from: {args.continue_from}")
        model = SAC.load(args.continue_from, env=train_env, verbose=1)
    else:
        model = SAC("MultiInputPolicy", train_env, verbose=1,
                  learning_rate=3e-4, buffer_size=100000,
                  learning_starts=1000, batch_size=256,
                  tau=0.005, gamma=0.99, ent_coef='auto')

    ckpt = CheckpointCallback(save_freq=50000, save_path=args.save_path, name_prefix="stage15")
    eval_cb = EvalCallback(eval_env, best_model_save_path=args.save_path,
                      eval_freq=10000, n_eval_episodes=10)

    print(f"Training for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, callback=[ckpt, eval_cb], log_interval=50)
    model.save(os.path.join(args.save_path, "stage15_final.zip"))

    train_env.close()
    eval_env.close()

    print(f"\nSaved to {args.save_path}")


if __name__ == "__main__":
    main()