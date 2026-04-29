# FastNN Quadrotor Control with Curriculum Learning

A residual reinforcement learning approach for robust quadrotor control under wind, mass perturbations, and dynamic target tracking.

## 🎯 Key Results

### Stage 5: Moving Target Tracking (Figure-8)

**BREAKTHROUGH**: We solved the precision-stability tradeoff!

| Metric | Before | After (New) | Improvement |
|--------|--------|-------------|-------------|
| **Success Rate** | 90% (avg) | **100%** | +10% |
| **Tracking Error** | 4.76m | **0.10m** | **47x better** |
| **Max Attitude** | 30° | 33° | Stable |

The drone now achieves **centimeter-precision tracking** on a dynamic figure-8 trajectory while handling wind disturbances and payload drops.

**Visual Demo**:
```bash
python visualize.py  # Launches figure-8 tracking demo
```

## 🔧 Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt  # stable-baselines3, mujoco, torch, numpy

# Verify installation
python -c "from env_rma import RMAQuadrotorEnv; print('✓ Environment loaded')"
```

## 🚀 Quick Start

### Visualize the Best Model (Figure-8)

```bash
# Default: Stage 5 with moving target
python visualize.py

# Options
python visualize.py --stage 4          # Test Stage 4 (payload drop)
python visualize.py --model PATH     # Use specific model
python visualize.py --pd-only         # Pure PD controller (no RL)
```

### Train Stage 5 (New Reward Function)

```bash
# Full training (5M steps, ~1-2 hours on 32 envs)
python train_stage5_curriculum.py --steps 5000000 --n-envs 32 --start-speed 0.05 --seeds 0

# Quick test (2M steps)
python train_stage5_curriculum.py --steps 2000000 --n-envs 32 --start-speed 0.05 --seeds 0
```

## 📊 Curriculum Stages

| Stage | Description | Challenge | Best Result |
|-------|-------------|-----------|-------------|
| 1 | Fixed hover | Basic stability | 100% success |
| 2 | Random pose + velocity | Initial conditions | 100% success |
| 3 | Wind + mass | External disturbances | 100% success |
| 4 | Payload drop | Sudden mass change | 100% success |
| **5** | **Moving target (figure-8)** | **Predictive tracking** | **100%, 0.10m error** |
| **6** | **Racing FPV (circuit)** | **Extreme speed** | Training (~5M steps) |

### Train Stage 6 Racing

```bash
# Train Stage 6 Racing FPV (5M steps, ~30 min on 32 envs)
python train_stage6_racing.py --steps 5000000 --n-envs 32 --start-speed 0.5

# Quick test (2M steps)
python train_stage6_racing.py --steps 2000000 --n-envs 32 --start-speed 0.5
```

Stage 6 features:
- Racing circuit with hairpin turns
- Extreme speed curriculum (0.5x → 5.0x)
- G-load penalties and angular rate limits
- No payload drops (pure racing focus)

## 🧠 Key Innovations

### 1. Precision-Stability Tradeoff SOLVED

**Problem**: Training longer improved tracking but caused crashes.

**Solution**: New reward terms:

```python
# Attitude Cliff: Steep penalty for tilting past 30°
if att_err > 0.52:  # ~30 degrees
    r_att -= 5.0 * (att_err - 0.52)**2

# Torque Penalty: Prevent aggressive roll/pitch
r_torque = -0.2 * (action[1]**2 + action[2]**2)
```

**Training Breakthrough**:
- 800K steps: 100% success, 41.7m error (undertrained)
- **1.6M steps: 100% success, 0.19m error (BREAKTHROUGH!)**
- 5M steps: 100% success, 0.10m error (optimal)

### 2. Reward Function

**Standard** (Stages 1-4):
```
r_total = r_alive + r_pos + r_att + r_vel + r_rate + r_smooth + 
          r_proximity + r_alignment + r_success + r_recovery + r_jerk
```

**Stage 5 Enhanced**:
```
r_total = ... + r_track + r_torque + r_att_cliff

# r_track: Direction alignment for moving target
r_track = 0.5 * cos_sim(drone_vel, target_vel)

# r_torque: Penalize aggressive roll/pitch
r_torque = -0.2 * (action[1]**2 + action[2]**2)

# r_att_cliff: Quadratic penalty past 30°
r_att_cliff = -5.0 * max(0, att_err - 0.52)**2
```

## 🧪 Experiments

### Ablation Study

```bash
# Test individual components
python train_ablation_stage5.py --config cliff_only    # Attitude cliff only
python train_ablation_stage5.py --config torque_only   # Torque penalty only
python train_ablation_stage5.py --config both          # Both (baseline)
```

### Generalization Testing

The model generalizes to different speeds:

| Speed | Success | Tracking Error |
|-------|---------|----------------|
| 0.5x | 98% | 0.101m |
| **1.0x** | **100%** | **0.096m** |
| 1.5x | 100% | 0.097m |
| 2.0x | 100% | 0.093m |

### Sim-to-Real Training

Train without `mass_est` for hardware deployment:

```bash
python train_stage5_no_massest.py --steps 5000000 --n-envs 32
```

The network learns to infer mass from action history and error integrals.

## 📁 Repository Structure

```
├── env_rma.py              # Main environment (MuJoCo-based)
├── visualize.py            # Interactive visualization
├── train_stage5_curriculum.py  # Stage 5 training script
├── train_stage5_no_massest.py   # Sim-to-real training
├── train_ablation_stage5.py     # Ablation study
├── env_wrapper_stage5.py   # Wrapper for no-mass-est training
├── fastnn_quadrotor_paper.md   # Full research paper
├── models_stage5_curriculum/   # Trained models
│   └── stage_5/
│       └── seed_0/
│           └── final.zip       # Best model (0.10m error)
└── results_stage5_curriculum/  # Evaluation results
```

## 🎓 Citation

If you use this code, please cite:

```bibtex
@article{fastnn_quadrotor_2026,
  title={FastNN Quadrotor Control: Solving the Precision-Stability Tradeoff},
  author={[Your Name]},
  year={2026}
}
```

## 📚 Full Documentation

See [`fastnn_quadrotor_paper.md`](fastnn_quadrotor_paper.md) for:
- Complete experimental results
- Architecture details
- Training methodology
- Hardware benchmarks (Raspberry Pi 5)
- Sim-to-real considerations

## 🤝 License

MIT License - See LICENSE file

## 🔗 Links

- Paper: [`fastnn_quadrotor_paper.md`](fastnn_quadrotor_paper.md)
- Issues: [GitHub Issues](https://github.com/yourusername/fastnn-quadrotor/issues)
- FastNN: [fastnn_inference.py](fastnn_inference.py) for Rust deployment
