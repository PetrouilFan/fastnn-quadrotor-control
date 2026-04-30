# Training Guide

This guide covers advanced training techniques, hyperparameter tuning, and curriculum customization for the FastNN Quadrotor Control framework.

## Training Scripts Overview

### Core Training Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `train_stage5_curriculum.py` | Main curriculum training | Speed progression, reward shaping |
| `train_stage6_racing.py` | Racing circuit training | High-speed maneuvers |
| `train_ablation_stage5.py` | Ablation studies | Component analysis |
| `train_with_delay_fixed.py` | Delay robustness | Fixed delay training |
| `train_stage5_no_massest.py` | Deployable training | No privileged mass_est |

### Training Modes

**Curriculum Mode** (Default):
```bash
python train_stage5_curriculum.py --curriculum-speed
```

**Fixed Speed Mode**:
```bash
python train_stage5_curriculum.py --fixed-speed 1.0
```

**Ablation Mode**:
```bash
python train_ablation_stage5.py --ablate reward_att_cliff
```

## Hyperparameter Tuning

### Critical Parameters

#### Learning Configuration

```python
# Stable-Baselines3 SAC defaults (recommended)
learning_rate = 1e-4      # Critic/actor learning rate
batch_size = 256          # Replay batch size
buffer_size = 1_000_000   # Experience replay size
tau = 0.005              # Soft update coefficient
gamma = 0.99             # Discount factor
ent_coef = 'auto'        # Entropy coefficient (auto-tuned)
```

#### Network Architecture

```python
policy_kwargs = {
    'net_arch': [256, 256],  # Hidden layer sizes
    'activation_fn': nn.ReLU  # Activation function
}
```

#### Training Setup

```python
n_envs = 32              # Parallel environments
total_timesteps = 5_000_000  # Total training steps
eval_freq = 50_000       # Evaluation frequency
save_freq = 100_000      # Checkpoint frequency
```

### Advanced Tuning

#### Learning Rate Schedule

```python
# Exponential decay (custom callback)
def lr_schedule(progress):
    initial_lr = 1e-4
    final_lr = 1e-5
    return initial_lr * (final_lr/initial_lr)**progress
```

#### Entropy Tuning

```python
# Manual entropy coefficient
ent_coef = 0.01          # Fixed entropy bonus
# Or auto-tuning
ent_coef = 'auto'        # Adaptive entropy
target_entropy = -2      # Target entropy (2 = 4 actions)
```

#### Batch Size Effects

| Batch Size | Memory Usage | Training Stability | Convergence Speed |
|------------|--------------|-------------------|-------------------|
| 128 | Low | Less stable | Faster |
| 256 | Medium | Stable | Good |
| 512 | High | Very stable | Slower |

## Curriculum Design

### Speed Curriculum (Stage 5)

**Automatic Progression**:
```python
# Speed increases over training
speed_stages = [
    (0, 1_000_000, 0.05),    # 0-1M: 0.05x speed
    (1_000_000, 2_000_000, 0.2),   # 1-2M: 0.2x speed
    (2_000_000, 3_000_000, 0.4),   # 2-3M: 0.4x speed
    (3_000_000, 4_000_000, 0.7),   # 3-4M: 0.7x speed
    (4_000_000, 5_000_000, 1.0),   # 4-5M: 1.0x speed
]
```

**Custom Curriculum**:
```python
# Modify speed progression
python train_stage5_curriculum.py \
  --speed-stages 0.1,0.3,0.6,0.9 \
  --stage-steps 1000000,1000000,1500000,1500000
```

### Reward Shaping

#### Precision-Stability Tradeoff

**Attitude Cliff**:
```python
# Prevents excessive tilting
cliff_threshold = 0.52  # 30 degrees
cliff_penalty = 5.0     # Quadratic penalty weight
```

**Torque Penalty**:
```python
# Discourages aggressive maneuvers
torque_weight = 0.2     # Roll/pitch torque penalty
```

#### Custom Reward Functions

```python
def custom_reward(obs, action, next_obs):
    # Base reward components
    r_pos = -np.linalg.norm(obs[0:3])  # Position error
    r_att = -0.1 * np.linalg.norm(obs[6:9])  # Attitude error

    # Custom additions
    r_energy = -0.001 * np.sum(action**2)  # Energy penalty
    r_smooth = -0.01 * np.linalg.norm(action - prev_action)  # Jerk penalty

    return r_pos + r_att + r_energy + r_smooth
```

## Ablation Studies

### Reward Component Analysis

```bash
# Test individual reward components
python train_ablation_stage5.py --ablate attitude_cliff
python train_ablation_stage5.py --ablate torque_penalty
python train_ablation_stage5.py --ablate velocity_matching
```

### Architecture Variations

```bash
# Network size ablation
python train_ablation_stage5.py --net-arch 128,128
python train_ablation_stage5.py --net-arch 512,512

# Asymmetric actor-critic
python train_ablation_stage5.py --asymmetric
```

### Observation Space

```bash
# Test observation components
python train_ablation_stage5.py --no-mass-est
python train_ablation_stage5.py --no-action-history
python train_ablation_stage5.py --no-target-velocity
```

## Multi-Seed Training

### Reproducibility Testing

```bash
# Train multiple seeds in parallel
for seed in 0 1 2 3 4; do
    python train_stage5_curriculum.py --seeds $seed &
done
```

### Statistical Analysis

```python
# Collect results across seeds
seeds = [0, 1, 2, 3, 4]
results = []
for seed in seeds:
    model_path = f"models_stage5_curriculum/stage_5/seed_{seed}/final.zip"
    result = evaluate_policy(model_path)
    results.append(result)

# Compute statistics
mean_success = np.mean([r['success_rate'] for r in results])
std_success = np.std([r['success_rate'] for r in results])
```

## Delay Robustness Training

### Fixed Delay Training

```bash
# Train with expected delay
python train_with_delay_fixed.py \
  --delay 30 \          # 30ms observation delay
  --steps 2000000 \
  --n-envs 8
```

### Curriculum Delay

```bash
# Progressive delay increase
python train_curriculum_delay.py \
  --delay-start 0 \
  --delay-end 100 \
  --steps 5000000
```

### State Estimation

```bash
# Train with state estimation wrapper
python train_state_est.py \
  --estimator kalman \
  --delay 50
```

## Performance Monitoring

### Real-time Metrics

```bash
# Terminal HUD monitoring
python terminal_hud.py \
  --log-dir logs/ \
  --refresh-rate 1
```

### Key Metrics to Monitor

- **Entropy coefficient**: Should decrease from 1.0 to ~0.01
- **Episode reward**: Should increase steadily
- **Success rate**: Should reach 100% by stage completion
- **Tracking error**: Should decrease to <0.1m
- **Gradient norms**: Should remain <1.0

### Early Stopping Criteria

```python
def should_stop_training(eval_results, patience=10):
    """Stop if no improvement for patience evaluations"""
    recent_success = eval_results[-patience:]['success_rate']
    return all(s >= 0.95 for s in recent_success)  # 95% success threshold
```

## Custom Environment Setup

### Domain Randomization

```python
# Custom randomization ranges
env_config = {
    'mass_range': [0.75, 1.25],      # ±25% mass
    'wind_range': [-0.8, 0.8],       # Wind force
    'motor_range': [0.85, 1.15],     # Motor efficiency
    'delay_range': [0, 3],           # Step delay
}
```

### Custom Stages

```python
# Define new training stage
custom_stage = {
    'trajectory': 'custom_path',      # Trajectory type
    'disturbances': ['wind', 'mass'], # Active disturbances
    'safety_boundary': 2.0,           # Episode boundary
    'max_attitude': 90,               # Crash threshold
    'success_steps': 500,             # Episode length
}
```

## Troubleshooting Training

### Common Issues

**Divergence**:
- Reduce learning rate: `--learning-rate 5e-5`
- Increase batch size: `--batch-size 512`
- Add gradient clipping: `--clip-grad 0.5`

**Poor Sample Efficiency**:
- Increase parallel envs: `--n-envs 64`
- Larger replay buffer: `--buffer-size 2000000`
- More frequent updates

**Overfitting**:
- Increase exploration: `--ent-coef 0.1`
- Domain randomization: `--randomize all`
- Regularization: `--weight-decay 1e-4`

### Performance Optimization

**GPU Training**:
```bash
# Use GPU for policy
export CUDA_VISIBLE_DEVICES=0
python train_stage5_curriculum.py --device cuda
```

**Multi-GPU**:
```bash
# Parallel training across GPUs
CUDA_VISIBLE_DEVICES=0 python train_stage5_curriculum.py --seeds 0 &
CUDA_VISIBLE_DEVICES=1 python train_stage5_curriculum.py --seeds 1 &
```

**Memory Optimization**:
```bash
# Reduce memory usage
python train_stage5_curriculum.py \
  --n-envs 16 \
  --buffer-size 500000 \
  --batch-size 128
```