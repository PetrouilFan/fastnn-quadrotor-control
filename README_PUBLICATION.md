# FastNN Quadrotor Control: Residual RL with Curriculum Learning

A comprehensive research project on robust quadrotor control using residual reinforcement learning with curriculum adaptation. This work addresses the precision-stability tradeoff in trajectory tracking through novel reward shaping and hierarchical control, achieving centimeter-precision on dynamic figure-8 trajectories.

## 🎯 Key Results

### Stage 5: Moving Target Tracking (Figure-8)

**BREAKTHROUGH**: Solved the precision-stability tradeoff in residual RL control!

| Metric | Baseline | This Work | Improvement |
|--------|----------|-----------|-------------|
| **Success Rate** | 90% (avg) | **100%** | +10% |
| **Tracking Error** | 4.76m | **0.10m** | **47× better** |
| **Max Attitude** | 30° | 33° | Stable |

The drone achieves **centimeter-precision tracking** on dynamic figure-8 trajectories while handling wind disturbances (±0.8N) and payload drops (up to 40% mass reduction).

**Training Breakthrough:**
- 800K steps: 100% success, 41.7m error (undertrained)
- **1.6M steps: 100% success, 0.19m error (BREAKTHROUGH!)**
- 5M steps: 100% success, 0.10m error (optimal)

### Delay Robustness Study

Comprehensive investigation of sensor delay tolerance (0-100ms):
- **Key Finding**: No generalization beyond trained delay without explicit adaptation
- **Best Practice**: Train with expected delay (e.g., 30ms → 96% success)
- Methods tested: GRU history, fixed delay, random delay, curriculum delay, world model compensation
- Full results in [`DELAY_RESULTS.md`](DELAY_RESULTS.md)

### Stage 11: Hierarchical Trajectory Tracking

High-level pilot policy (SAC) + frozen low-level rate controller (Stage 10):
- V3 (curvature clamping): **36% success**, 0.331m mean CTE, 0.687m p95 CTE
- V2 (no clamping): 5% success, 0.753m mean CTE, **1.806m p95 CTE**
- **Curvature clamping is critical** — keeps vehicle in Stage 10's stable regime

## 📁 Repository Structure

```
fastnn_quadrotor/
├── README_PUBLICATION.md            # This file (publication overview)
├── README.md                        # Original project README
├── CONTRIBUTING.md                  # Development guidelines
├── DELAY_RESULTS.md                 # Comprehensive delay robustness study
├── fastnn_quadrotor_paper.md        # Full research paper (methods, results)
├── quadrotor_best_path_forward.md   # Technical analysis & future directions
├── quadrotor_research_paper.md      # Complete experimental history
├── pyproject.toml                   # Project dependencies
├── requirements.txt                 # Python dependencies
│
├── canonical/                       # Core publication artifacts
│   ├── core/                        # Essential algorithms
│   ├── models/                      # Best trained models
│   └── results/                     # Key evaluation results
│
├── docs/                            # Documentation
│   ├── guides/                      # User guides
│   │   ├── getting_started.md
│   │   ├── training_guide.md
│   │   └── deployment_guide.md
│   └── archived/                    # Historical experiments
│       ├── 2026_archive/
│       ├── training_scripts/
│       ├── evaluation_scripts/
│       └── experiments/
│
├── env_rma.py                       # Main MuJoCo environment (1603 lines)
├── env_wrapper.py                   # Base wrappers
├── env_wrapper_stage5.py            # Stage 5 specific wrapper
│
├── # Core Training Scripts (Canonical)
├── train_stage5_curriculum.py       # Stage 5: Moving target (CURRICULUM)
├── train_stage5_no_massest.py       # Sim-to-real (no mass estimation)
├── train_stage6_racing.py           # Stage 6: Racing FPV
├── train_ablation_stage5.py         # Ablation study
│
├── # Delay Robustness Experiments
├── train_gru_stage5.py              # GRU with observation history
├── train_with_delay_fixed.py        # Fixed delay training
├── train_random_delay.py            # Random delay training
├── train_curriculum_delay.py        # Curriculum delay
├── train_state_est.py               # State estimation
├── train_world_model.py             # World model compensation
│
├── # Hierarchical Control (Stage 11)
├── train_stage11_primitive.py       # Body-frame primitives
├── train_stage16.py                 # Full obs + action history
├── train_stage16_simple.py          # Simplified version
│
├── # Evaluation & Testing
├── visualize.py                     # Interactive visualization
├── eval_bc.py                       # Behavior cloning evaluation
├── eval_e2e.py                      # End-to-end evaluation
├── eval_stage8_final.py             # Stage 8 comprehensive eval
├── eval_gru_history8.py             # GRU delay robustness
├── eval_delay_trained.py            # Evaluate delay-trained models
├── test_delay.py                    # Delay robustness testing
├── test_action_delay.py             # Action delay testing
├── test_world_model.py              # World model evaluation
│
├── # Baseline Controllers
├── baseline_controllers.py          # PD, PID, LQR baselines
├── bc_reg.py                        # Behavior cloning regression
├── bc_residual.py                   # BC on residuals
├── controllers.py                   # Controller implementations
│
├── # Utilities
├── callbacks.py                     # Training callbacks
├── terminal_hud.py                  # Real-time terminal display
├── render_episode.py                # Episode rendering
├── visualize_stages.py              # Multi-stage visualization
├── visualize_with_hud.py            # Visualization with HUD
└── fastnn_inference.py              # FastNN Rust inference
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from env_rma import RMAQuadrotorEnv; print('✓ Environment loaded')"
```

### Visualize Trained Model (Figure-8 Tracking)

```bash
# Default: Stage 5 with moving target
python visualize.py

# Options
python visualize.py --stage 4          # Test Stage 4 (payload drop)
python visualize.py --model PATH       # Use specific model
python visualize.py --pd-only          # Pure PD controller (no RL)
```

### Train Stage 5 (Moving Target Tracking)

```bash
# Full training (5M steps, ~1-2 hours on 32 envs)
python train_stage5_curriculum.py --steps 5000000 --n-envs 32 --start-speed 0.05 --seeds 0

# Quick test (2M steps)
python train_stage5_curriculum.py --steps 2000000 --n-envs 32 --start-speed 0.05 --seeds 0
```

### Test Delay Robustness

```bash
# Evaluate baseline model at different delays
python test_delay.py models_stage5_curriculum/stage_5/seed_0/final.zip

# Train with fixed delay (30ms)
python train_with_delay_fixed.py --delay 30 --steps 2000000
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
| 7 | Yaw-controlled figure-8 | Heading control | In progress |
| 8 | Progressive curriculum | Adaptive difficulty | In progress |

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

## 🔬 Delay Robustness Study

### Problem Statement
- **Target**: Make policy robust to sensor delays (0-100ms)
- **Baseline**: 39% success at 0ms, 0% at 50ms+
- **Goal**: Achieve >80% across all delay levels

### Approaches Tested

1. **Baseline (No Delay Training)**: 39% at 0ms, 0% at 50ms+
2. **GRU with Observation History**: 99% at train, 0% at delay
3. **Fixed Delay Training**: Works only for trained delay
   - 10ms: 100% at 0/10ms, 0% at 30/100ms
   - 30ms: 98% at 0/30ms, 0% at 50/100ms
4. **Random Delay Training**: 0% everywhere (too hard)
5. **Curriculum Delay**: 27% at 0ms, 0% elsewhere
6. **State Estimator**: 92% at train, 0% at delay
7. **World Model**: 0% (compounding prediction error)
8. **Action Delay**: 44% at 0ms, 0% at delay

### Key Insights

1. **Credit Assignment Problem**: At 30ms+ delay, reward based on current state but policy sees old state — fundamentally misaligned
2. **No Free Lunch**: Cannot generalize beyond training conditions
3. **Reward Delay is the Key**: Delay >30ms makes learning impossible because reward doesn't reflect recent actions
4. **World Model Errors Compound**: Even accurate models (MSE=0.0066) compound errors over multiple steps

### Practical Recommendation

**For specific known delay (e.g., 30ms):**
```bash
python train_with_delay_fixed.py --delay 30 --steps 2000000
# Result: 96% at 30ms
```

**For unknown delays:**
1. Use multiple parallel environments with different delays
2. Train curriculum: start low, increase gradually
3. Combine with explicit state estimator

**Full results**: See [`DELAY_RESULTS.md`](DELAY_RESULTS.md)

## 🎓 Citation

If you use this code, please cite:

```bibtex
@article{fastnn_quadrotor_2026,
  title={FastNN Quadrotor Control: Solving the Precision-Stability Tradeoff},
  author={[Your Name]},
  year={2026}
}
```

## 📚 Documentation

### Core Papers

- **[`fastnn_quadrotor_paper.md`](fastnn_quadrotor_paper.md)**: Complete research paper with methods, results, and analysis
- **[`quadrotor_best_path_forward.md`](quadrotor_best_path_forward.md)**: Technical analysis and future directions
- **[`quadrotor_research_paper.md`](quadrotor_research_paper.md)**: Complete experimental history
- **[`DELAY_RESULTS.md`](DELAY_RESULTS.md)**: Comprehensive delay robustness study

### User Guides

- **[Getting Started](docs/guides/getting_started.md)**: Installation and first run
- **[Training Guide](docs/guides/training_guide.md)**: How to train your own models
- **[Deployment Guide](docs/guides/deployment_guide.md)**: Deploying to Raspberry Pi 5

### API Documentation

- **Environment**: [`env_rma.py`](env_rma.py) — MuJoCo-based quadrotor with curriculum
- **Training**: [`train_stage5_curriculum.py`](train_stage5_curriculum.py) — Stage 5 training
- **Evaluation**: [`visualize.py`](visualize.py) — Interactive visualization

## 🏗️ Architecture

### Hierarchical Control

```
Stage 11 (Pilot SAC)                    High-level planning
  ↓ (RC sticks / body rates)
Stage 10 (Rate SAC, frozen)             Low-level rate control
  ↓ (Motor PWM)
MuJoCo Quadrotor                        Physics simulation
```

### Observation Space (54-dim for Stage 5)

- Position error (3)
- Velocity error (3)
- Attitude error (3)
- Rate error (3)
- Linear acceleration (3)
- Rotation matrix (9)
- Body rates (3)
- Action history (16)
- Error integrals (4)
- Rotor thrust estimate (4)
- **Target velocity (3)** ← Stage 5 addition

Privileged (critic-only): mass_ratio, com_shift, wind, motor_deg, mass_est

### Action Space (4-dim)

- Residual thrust: [-1, 1] → ±1.0 N (≈ ±10% of hover)
- Residual roll torque: [-1, 1] → ±1.0 Nm
- Residual pitch torque: [-1, 1] → ±1.0 Nm
- Residual yaw torque: [-1, 1] → ±1.0 Nm

## 🔧 Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_env.py -v
```

### Code Style

```bash
# Format code
black .

# Lint code
ruff check .

# Type check
mypy .
```

### Building Documentation

```bash
# Generate API docs
pdoc --html --output-dir docs/api/ env_rma
```

## 🌐 Links

- **Paper**: [`fastnn_quadrotor_paper.md`](fastnn_quadrotor_paper.md)
- **Delay Study**: [`DELAY_RESULTS.md`](DELAY_RESULTS.md)
- **FastNN**: [`fastnn_inference.py`](fastnn_inference.py) — Rust deployment
- **Issues**: [GitHub Issues](https://github.com/yourusername/fastnn-quadrotor/issues)

## 🤝 Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - See [`LICENSE`](LICENSE) file

---

**Note**: This is a research project. Results may vary based on hardware, random seeds, and training conditions. Always validate in simulation before hardware deployment.
