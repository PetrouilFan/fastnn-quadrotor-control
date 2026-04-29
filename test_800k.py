from stable_baselines3 import SAC
from env_rma import RMAQuadrotorEnv
import numpy as np

model = SAC.load(
    "models_stage8_progressive/stage_8/seed_0/stage8_checkpoint_800000_steps.zip",
    device="cpu",
)
env = RMAQuadrotorEnv(curriculum_stage=8, use_direct_control=True)
env.set_target_trajectory("figure8_medium")
env.set_moving_target(True)
env.set_target_speed(0.15)

obs, _ = env.reset()
for i in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(f"Crash at step {i}")
        break
else:
    print("100 steps completed successfully")

print(
    f"Final state: pos={env.data.qpos[:3]}, target={env.target_pos}, error={np.linalg.norm(env.data.qpos[:3] - env.target_pos):.3f}"
)
