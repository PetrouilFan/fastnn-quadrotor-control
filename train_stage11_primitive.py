#!/usr/bin/env python3
"""
Train Stage 11 Primitive (body-frame primitives).
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from quadrotor.env_stage11_primitive import QuadrotorEnvStage11Primitive


def make_env(stage10_path: str, vecnorm_path: str, seed: int):
    def _init():
        env = QuadrotorEnvStage11Primitive(
            stage10_model_path=stage10_path,
            stage10_vecnormalize_path=vecnorm_path,
            max_episode_steps=5000,
            primitive_duration=20,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage10-path", type=str, required=True, help="Path to Stage 10 model")
    parser.add_argument("--vecnorm-path", type=str, default=None, help="Path to VecNormalize stats")
    parser.add_argument("--steps", type=int, default=500000, help="Training steps")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Eval frequency")
    parser.add_argument("--save-path", type=str, default=None, help="Save path")
    args = parser.parse_args()
    
    stage10_path = args.stage10_path
    vecnorm_path = args.vecnorm_path or stage10_path.replace(".zip", "_vecnormalize.pkl")
    
    save_path = args.save_path or f"stage11_primitive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(save_path, exist_ok=True)
    
    print(f"Stage 10: {stage10_path}")
    print(f"VecNormalize: {vecnorm_path}")
    print(f"Save path: {save_path}")
    
    env_fns = [make_env(stage10_path, vecnorm_path, i) for i in range(4)]
    env = DummyVecEnv(env_fns)
    
    eval_env_fns = [make_env(stage10_path, vecnorm_path, 100+i) for i in range(2)]
    eval_env = DummyVecEnv(eval_env_fns)
    
    model = SAC(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_sigma=0.1,
        policy_kwargs=dict(
            net_arch=[256, 256],
            use_sde=True,
            use_sde_at_exploration=True,
        ),
        verbose=1,
        device="cuda",
        tensorboard_log=save_path,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=save_path,
        name_prefix="stage11_primitive",
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        eval_freq=args.eval_freq,
        n_eval_episodes=5,
        deterministic=False,
    )
    
    print(f"Training for {args.steps} steps...")
    model.learn(
        total_timesteps=args.steps,
        callback=[checkpoint_callback, eval_callback],
        log_interval=100,
    )
    
    final_path = os.path.join(save_path, "stage11_primitive_final.zip")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")


if __name__ == "__main__":
    main()