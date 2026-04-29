#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from fastnn_quadrotor.env_rma import RMAQuadrotorEnv

class SimpleBCController(nn.Module):
    def __init__(self, state_dim=52, action_dim=4, hidden_dims=[256, 256, 128]):
        super().__init__()
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Load model
checkpoint = torch.load('models/simple_bc_best.pt', map_location='cpu', weights_only=False)
model = SimpleBCController(state_dim=52, action_dim=4, hidden_dims=[256, 256, 128])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

state_mean = checkpoint['state_mean']
state_std = checkpoint['state_std']

# Run one episode and trace what happens
env = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=True)
obs, _ = env.reset()

print('Tracing BC episode...')
for step in range(50):
    # Get BC action
    state_norm = (torch.FloatTensor(obs[:52]) - torch.FloatTensor(state_mean)) / (torch.FloatTensor(state_std) + 1e-8)
    with torch.no_grad():
        bc_action = model(state_norm.unsqueeze(0)).squeeze(0).numpy()

    normalized = np.zeros(4)
    normalized[0] = np.clip((bc_action[0] - 10.0) / 10.0, -1.0, 1.0)
    normalized[1] = np.clip(bc_action[1] / 3.0, -1.0, 1.0)
    normalized[2] = np.clip(bc_action[2] / 3.0, -1.0, 1.0)
    normalized[3] = np.clip(bc_action[3] / 2.0, -1.0, 1.0)

    # Print every 10 steps
    if step % 10 == 0:
        rpy = env._quat_to_rpy(env.data.qpos[3:7])
        print(f'Step {step:3d}: rpy={np.rad2deg(rpy).round(1)}, action={bc_action.round(2)}, normalized={normalized.round(2)}')

    obs, reward, terminated, truncated, info = env.step(normalized)

    if terminated:
        print(f'CRASH at step {step}!')
        rpy = env._quat_to_rpy(env.data.qpos[3:7])
        print(f'  rpy(deg)={np.rad2deg(rpy).round(1)}')
        break

print('\nWhat PD would do at same state:')
env_pd = RMAQuadrotorEnv(curriculum_stage=1, use_direct_control=False)
obs_pd, _ = env_pd.reset()
for step in range(50):
    if step % 10 == 0:
        rpy = env_pd._quat_to_rpy(env_pd.data.qpos[3:7])
        pd_action = env_pd._cascaded_controller()
        print(f'Step {step:3d}: rpy={np.rad2deg(rpy).round(1)}, pd_action={pd_action.round(2)}')
    obs_pd, reward, terminated, truncated, info = env_pd.step(np.zeros(4))
    if terminated:
        print(f'PD CRASH at step {step}!')
        break

env.close()
env_pd.close()