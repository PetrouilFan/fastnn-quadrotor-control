# Methods Details

## Environment Implementation

### MuJoCo Configuration

```python
# Environment setup in env_rma.py
self.model = mujoco.MjModel.from_xml_path(xml_path)
self.data = mujoco.MjData(self.model)
self.renderer = mujoco.Renderer(self.model)

# Physics parameters
self.model.opt.timestep = 0.01  # 100 Hz
self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
self.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
self.model.opt.iterations = 50
self.model.opt.tolerance = 1e-6
```

### Quadrotor Dynamics

**Nominal Parameters**:
- Mass: 1.0 kg
- Hover thrust: ~10 N (gravity compensation)
- Max thrust: 20.0 N per motor
- Motor time constant: 57 ms (first-order lag)
- Rotor inertia: 1e-5 kg·m²
- Arm length: 0.17 m (diagonal 0.34 m)

**Disturbance Models**:
- Wind: Constant force applied via `xfrc_applied` (+x, +y directions)
- Mass variation: ±25% via `body_mass` and `body_inertia`
- Motor degradation: ±15% thrust coefficient scaling
- Center of mass shift: ±5 cm offset

### Observation Space Details

**Deployable Observations (51-dim)**:

| Index | Component | Shape | Description | Units |
|-------|-----------|-------|-------------|-------|
| 0-2 | position_error | (3,) | target_pos - current_pos | m |
| 3-5 | velocity_error | (3,) | -current_velocity | m/s |
| 6-8 | attitude_error | (3,) | roll, pitch, yaw error | rad |
| 9-11 | rate_error | (3,) | -angular_rates | rad/s |
| 12-14 | acceleration | (3,) | IMU linear acceleration (body frame) | m/s² |
| 15-23 | rotation_matrix | (9,) | SO(3) rotation matrix (flattened) | - |
| 24-26 | body_rates | (3,) | angular velocities | rad/s |
| 27-42 | action_history | (16,) | 4-step action buffer (4×4 dims) | - |
| 43-46 | error_integrals | (4,) | ∫position_error + ∫yaw_error | - |
| 47-50 | rotor_thrust | (4,) | Filtered motor thrust estimates | N |
| 51-53 | target_velocity | (3,) | Target velocity (stages 5-6) | m/s |

**Privileged Observations (9-dim, critic-only)**:
- mass_ratio: Current mass / nominal mass
- com_shift: Center of mass offset (3 values)
- wind: Wind force vector (2 values, x/y)
- motor_deg: Motor degradation factors (4 values)
- mass_est: Online mass estimate

**Total**: 51 deployable + 9 privileged = 60-dim

### Action Space Details

**Residual Actions (4-dim)**:
```python
action_space = gym.spaces.Box(
    low=np.array([-1.0, -1.0, -1.0, -1.0]),
    high=np.array([1.0, 1.0, 1.0, 1.0]),
    dtype=np.float32
)
```

**Scaling to Physical Units**:
```python
action_scale = np.array([1.0, 1.0, 1.0, 1.0])  # N, Nm, Nm, Nm
total_control = pd_output + action * action_scale
```

**PD Controller Base**:
- Position control: P=2.0, I=0.5, D=1.0
- Attitude control: P=5.0, D=1.0
- Rate control: P=0.5, D=0.1

## Reward Function Specification

### Base Reward Components

```python
def compute_reward(self, obs, action, next_obs):
    # Position tracking
    r_pos = -np.linalg.norm(obs[0:3])  # Position error penalty

    # Attitude stability
    r_att = -0.1 * np.linalg.norm(obs[6:9])  # Attitude error penalty

    # Velocity matching
    r_vel = -0.01 * np.linalg.norm(obs[3:6])  # Velocity error penalty

    # Rate damping
    r_rate = -0.005 * np.linalg.norm(obs[9:12])  # Angular rate penalty

    # Action smoothness
    r_smooth = -0.001 * np.linalg.norm(action)  # Action magnitude penalty

    # Alive bonus
    r_alive = 0.1  # Per timestep survival reward

    # Success bonus (stage-dependent)
    r_success = 10.0 if self.is_success() else 0.0

    return r_pos + r_att + r_vel + r_rate + r_smooth + r_alive + r_success
```

### Stage 5 Precision Additions

**Attitude Cliff Barrier**:
```python
def attitude_cliff_penalty(attitude_error):
    """Quadratic penalty beyond 30° threshold"""
    threshold = 0.52  # ~30 degrees in radians
    if attitude_error > threshold:
        return -5.0 * (attitude_error - threshold)**2
    return 0.0

r_att_cliff = attitude_cliff_penalty(np.linalg.norm(obs[6:8]))  # Roll/pitch only
```

**Torque Penalty**:
```python
r_torque = -0.2 * (action[1]**2 + action[2]**2)  # Roll/pitch torque magnitude
```

### Stage 4 Payload Drop Recovery

**Recovery Bonus**:
```python
def recovery_reward(self):
    """Bonus for returning to target after payload drop"""
    if self.payload_dropped and self.steps_since_drop > 50:
        pos_error = np.linalg.norm(obs[0:3])
        att_error = np.linalg.norm(obs[6:9])
        if pos_error < 0.15 and att_error < 0.15:  # Within 15cm, 15°
            return 5.0
    return 0.0
```

## Training Curriculum Implementation

### Stage Definitions

**Stage 1: Fixed Hover**
- Target: Fixed position at origin
- Initial conditions: Small random perturbations
- Success: Maintain position for 500 steps

**Stage 2: Random Pose + Velocity**
- Target: Fixed position
- Initial: Random position (±0.2m), velocity (±0.5m/s)
- Disturbances: None

**Stage 3: Wind + Mass**
- Target: Fixed position
- Disturbances: Wind (±0.5N), mass (±10%)
- Initial: Random pose/velocity

**Stage 4: Payload Drop**
- Target: Fixed position
- Event: Mass drop at random timestep (50% probability)
- Magnitude: -15% to -40% mass reduction
- Recovery required

**Stage 5: Moving Target**
- Target: Figure-8 trajectory (Lemniscate)
- Speed curriculum: 0.05x → 1.0x over training
- Safety boundary: 1.5m
- Success: Track within 0.2m average error

**Stage 6: Racing FPV**
- Target: Oval circuit (22m lap)
- Speed: Up to 5x baseline
- Safety boundary: 3.0m
- Attitude limit: 120°

### Curriculum Progression Logic

```python
def should_advance_stage(self, eval_results):
    """Stage advancement criteria"""
    success_rate = eval_results['success_rate']

    if self.current_stage == 1:
        return success_rate >= 0.9  # 90% success
    elif self.current_stage in [2, 3]:
        return success_rate >= 0.5  # 50% success
    elif self.current_stage == 4:
        return success_rate >= 0.95  # Near convergence
    elif self.current_stage == 5:
        return success_rate == 1.0 and eval_results['mean_error'] < 0.2
    return False  # Stage 6: Manual evaluation
```

## SAC Training Configuration

### Stable-Baselines3 Setup

```python
from stable_baselines3 import SAC

model = SAC(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=1_000_000,
    learning_starts=1000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef='auto',
    target_entropy=-2,
    policy_kwargs=dict(
        net_arch=[256, 256],
        activation_fn=nn.ReLU
    ),
    verbose=1,
)
```

### Network Architecture

**Actor Network**:
```
Input (51/60-dim) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(4) → Tanh
```

**Critic Network**:
```
Input (51/60-dim + 4 actions) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1)
```

**Value Network** (for SAC):
```
Input (51/60-dim) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(1)
```

### Training Hyperparameters by Stage

| Parameter | Stages 1-4 | Stage 5 | Stage 6 |
|-----------|------------|---------|---------|
| Total steps | 1,000,000 | 5,000,000 | 5,000,000 |
| Parallel envs | 32 | 64 | 128 |
| Eval frequency | 50,000 | 100,000 | 100,000 |
| Checkpoint freq | 100,000 | 500,000 | 500,000 |

## Evaluation Protocol

### Metrics Computation

```python
def evaluate_policy(model, env, n_eval_episodes=100):
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    successes = 0

    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_errors = []

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            episode_length += 1
            episode_errors.append(np.linalg.norm(obs[0:3]))  # Position error

            done = terminated or truncated

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        tracking_errors.append(np.mean(episode_errors))

        if episode_length >= 500:  # Success criterion
            successes += 1

    return {
        'mean_reward': np.mean(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'success_rate': successes / n_eval_episodes,
        'mean_tracking_error': np.mean(tracking_errors),
        'std_tracking_error': np.std(tracking_errors)
    }
```

### Single-Seed Qualification

All reported results use seed 0 for training and evaluation unless explicitly noted as multi-seed. Multi-seed validation (seeds 0,1,2) confirms reproducibility but is not required for publication claims.

## Code Architecture

### Core Files

- `env_rma.py`: Main environment (1603 lines)
- `env_wrapper.py`: Base environment wrappers
- `env_wrapper_stage5.py`: Stage 5 specific modifications
- `train_stage5_curriculum.py`: Curriculum training script
- `visualize.py`: Interactive visualization
- `eval_e2e.py`: End-to-end evaluation

### Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = "^3.9"
mujoco = "^3.1.0"
gymnasium = "^0.29.0"
stable-baselines3 = "^2.1.0"
torch = "^2.0.0"
numpy = "^1.24.0"
scipy = "^1.11.0"
matplotlib = "^3.7.0"
```

## FastNN Deployment

### Rust Implementation

**Model Export**:
```python
# Convert PyTorch to ONNX
torch.onnx.export(
    model.policy,
    sample_input,
    "model.onnx",
    opset_version=11,
    input_names=['obs'],
    output_names=['action']
)
```

**FastNN Inference**:
- Static computation graph
- ARM NEON optimizations
- Memory pooling
- Quantization support (future)

### Performance Benchmarks

**Raspberry Pi 5**:
- Model size: ~200KB (FP32)
- Memory usage: ~50MB
- Warmup time: <100ms
- Steady-state: 114μs median latency