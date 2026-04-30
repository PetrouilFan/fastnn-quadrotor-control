# Getting Started

This guide covers installation, basic usage, and first experiments with the FastNN Quadrotor Control framework.

## Installation

### System Requirements

- **OS**: Linux (Ubuntu 20.04+), macOS (12+), or Windows (WSL2)
- **Python**: 3.9 or later
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: NVIDIA GPU recommended for training (optional for inference)
- **Storage**: 10GB free space

### Dependencies Installation

```bash
# Clone the repository
git clone https://github.com/username/fastnn_quadrotor.git
cd fastnn_quadrotor

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install Python dependencies
pip install -r requirements.txt

# Verify MuJoCo installation
python -c "from env_rma import RMAQuadrotorEnv; print('✓ Environment loaded successfully')"
```

### Optional: GPU Support

```bash
# Install PyTorch with CUDA (adjust version for your CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## First Run: Visualize Trained Model

### Download Pre-trained Models

```bash
# Download model checkpoints (if available)
# Models are typically stored in models_stage5_curriculum/
# This step may be automated in future releases
```

### Interactive Visualization

```bash
# Launch visualization for Stage 5 (figure-8 tracking)
python visualize.py

# Options:
python visualize.py --stage 4          # Stage 4: Payload drop
python visualize.py --model PATH       # Custom model path
python visualize.py --pd-only          # PD controller baseline
python visualize.py --no-gui           # Headless mode
```

**Controls**:
- **Space**: Pause/unpause
- **R**: Reset episode
- **Q/W**: Speed control
- **Mouse**: Camera control

### Expected Output

The visualization shows:
- **Drone trajectory**: Blue line (target), red line (actual)
- **Attitude indicator**: Real-time roll/pitch/yaw
- **Performance metrics**: Tracking error, success rate
- **Terminal HUD**: Detailed telemetry

## Basic Training: Stage 5 Curriculum

### Quick Test (2M steps, ~1 hour)

```bash
# Fast training for testing
python train_stage5_curriculum.py \
  --steps 2000000 \
  --n-envs 32 \
  --start-speed 0.05 \
  --seeds 0
```

### Full Training (5M steps, ~2.5 hours)

```bash
# Complete curriculum training
python train_stage5_curriculum.py \
  --steps 5000000 \
  --n-envs 32 \
  --start-speed 0.05 \
  --seeds 0
```

### Training Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--steps` | 5000000 | Total training steps |
| `--n-envs` | 32 | Parallel environments |
| `--start-speed` | 0.05 | Initial trajectory speed |
| `--seeds` | 0 | Random seed |
| `--eval-freq` | 50000 | Evaluation frequency |
| `--save-freq` | 100000 | Checkpoint frequency |

### Monitoring Training

```bash
# Training outputs real-time metrics:
# Episode reward, success rate, tracking error
# Use terminal_hud.py for detailed monitoring
python terminal_hud.py --log-dir logs/
```

## Understanding the Curriculum

### Stage Progression

The training follows a 6-stage curriculum from basic hover to advanced tracking:

1. **Stage 1**: Fixed hover (stability)
2. **Stage 2**: Random pose/velocity (robustness)
3. **Stage 3**: Wind + mass disturbances
4. **Stage 4**: Payload drop recovery
5. **Stage 5**: Figure-8 tracking (precision)
6. **Stage 6**: Racing circuit (speed)

### Key Concepts

- **Residual Control**: Policy outputs small corrections to PD controller
- **Curriculum Learning**: Progressive difficulty increase
- **Reward Shaping**: Precision-stability balance through attitude cliffs
- **Domain Randomization**: Robustness to real-world variations

## Troubleshooting

### Common Issues

**MuJoCo Import Error**:
```bash
# Install MuJoCo properly
# Linux
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
# macOS
brew install mesa-glfw-osmesa
```

**GPU Memory Error**:
```bash
# Reduce parallel environments
python train_stage5_curriculum.py --n-envs 16
```

**Visualization Won't Start**:
```bash
# Check display (Linux)
export DISPLAY=:0
# Or use headless mode
python visualize.py --no-gui
```

### Performance Benchmarks

**Training Time** (RTX 3090):
- 1M steps: ~30 minutes
- 5M steps: ~2.5 hours

**System Resources**:
- CPU: 4-8 cores recommended
- RAM: 16GB for parallel training
- GPU: 8GB VRAM minimum

## Next Steps

- **Training Guide**: Advanced training techniques
- **Deployment Guide**: Hardware implementation
- **API Reference**: Code documentation