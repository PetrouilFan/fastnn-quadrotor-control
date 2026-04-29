#!/usr/bin/env python3
"""Test action delay - using policy."""

import numpy as np
import torch

from env_rma import RMAQuadrotorEnv
from stable_baselines3 import SAC


class ActionDelayWrapper:
    def __init__(self, env, delay_steps=3):
        self.env = env
        self.delay_steps = delay_steps
        self._pending_actions = []
        self._action_space = env.action_space
    
    def reset(self, **kwargs):
        self._pending_actions = []
        return self.env.reset(**kwargs)
    
    def step(self, action):
        self._pending_actions.append(action.copy())
        
        if len(self._pending_actions) > self.delay_steps:
            delayed = self._pending_actions.pop(0)
            return self.env.step(delayed)
        
        return self.env.step(action)
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    def close(self):
        self.env.close()


def get_action(model, obs):
    device = model.device
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        action, _ = model.actor.action_log_prob(obs_t)
    return action.cpu().numpy()[0]


def evaluate_action_delay(model_path, delay_steps=3, n_episodes=100):
    print(f"Action delay: {delay_steps * 10}ms", end=" ")
    
    model = SAC.load(model_path)
    
    successes = 0
    for ep in range(n_episodes):
        env = RMAQuadrotorEnv(curriculum_stage=5, use_direct_control=True)
        env.set_target_speed(1.0)
        env.set_moving_target(True)
        
        env = ActionDelayWrapper(env, delay_steps=delay_steps)
        raw = env.reset()
        obs = raw[0] if isinstance(raw, tuple) else raw
        
        done = False
        steps = 0
        
        while not done and steps < 500:
            action = get_action(model, obs)
            result = env.step(action)
            
            if len(result) >= 4:
                obs = result[0]
                done = result[2] or result[3]
            
            obs = obs[0] if isinstance(obs, tuple) else obs
            steps += 1
        
        if steps >= 500:
            successes += 1
        env.close()
    
    rate = successes / n_episodes * 100
    print(f"-> {rate}%")
    return rate


if __name__ == "__main__":
    print("=" * 40)
    baseline = "models_stage5_curriculum/stage_5/seed_0/final.zip"
    
    print("Action Delay Test")
    print("=" * 40)
    
    s0 = evaluate_action_delay(baseline, 0)
    s3 = evaluate_action_delay(baseline, 3)
    s5 = evaluate_action_delay(baseline, 5)
    s10 = evaluate_action_delay(baseline, 10)
    
    print(f"0ms: {s0}%, 30ms: {s3}%, 50ms: {s5}%, 100ms: {s10}%")