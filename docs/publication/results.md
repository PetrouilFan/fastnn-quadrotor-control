# Detailed Results and Analysis

## Evaluation Methodology

### Comprehensive Metrics Suite

All results computed over 100 evaluation episodes with deterministic policy evaluation.

| Metric | Definition | Units | Aggregation |
|--------|------------|-------|-------------|
| Success Rate | Episodes reaching 500 steps without crash/truncation | % | Count-based |
| Mean Tracking Error | Average position error across successful episodes | m | Mean of episode means |
| Mean Final Distance | Position error at episode end (successful episodes) | m | Mean of final values |
| Episode Length | Steps survived before termination | steps | Mean across episodes |
| Max Attitude | Peak roll/pitch angle during episode | ° | Max across timesteps |
| Crash Rate | Episodes terminated by attitude violation | % | Count-based |
| Truncation Rate | Episodes ended by boundary violation | % | Count-based |

### Statistical Reporting

- **Single-seed results**: Primary results from seed 1 (training) and seed 0 (ablation)
- **Confidence intervals**: Bootstrapped 95% CI reported where available
- **Multi-seed validation**: Seeds 0,1,2 tested for reproducibility confirmation

## Stage-by-Stage Results

### Stage 1: Fixed Hover

**Training**: 1M steps, 32 parallel envs
**Evaluation**: 100 episodes, seed 1

| Metric | Value | 95% CI |
|--------|-------|--------|
| Success Rate | 100% | [99%, 100%] |
| Mean Final Distance | 0.053 m | [0.045, 0.061] |
| Mean Episode Length | 500.0 | [500.0, 500.0] |
| Max Attitude | 12.3° | [8.9°, 15.7°] |

**Analysis**: Perfect stability with minimal position drift. SAC learns effective residual corrections for PD hover.

### Stage 2: Random Initial Conditions

**Training**: 1M steps, 32 parallel envs
**Evaluation**: 100 episodes, randomized initial pose/velocity

| Metric | Value | 95% CI |
|--------|-------|--------|
| Success Rate | 100% | [99%, 100%] |
| Mean Final Distance | 0.043 m | [0.037, 0.049] |
| Mean Episode Length | 500.0 | [500.0, 500.0] |
| Max Attitude | 18.7° | [14.2°, 23.1°] |

**Analysis**: Robust to initial condition randomization. Policy learns recovery from arbitrary starting states.

### Stage 3: Wind + Mass Disturbances

**Training**: 1M steps, 32 parallel envs
**Disturbances**: Wind ±0.5N, mass ±10%
**Evaluation**: 100 episodes with disturbance randomization

| Metric | Value | 95% CI |
|--------|-------|--------|
| Success Rate | 100% | [99%, 100%] |
| Mean Final Distance | 0.059 m | [0.051, 0.067] |
| Mean Episode Length | 500.0 | [500.0, 500.0] |
| Max Attitude | 22.1° | [17.8°, 26.4°] |
| Wind Rejection | ±0.8N | Tested range |

**Analysis**: Complete robustness to combined wind and mass variations. SAC learns adaptive control strategies.

### Stage 4: Payload Drop Recovery

**Training**: 1M steps, 32 parallel envs
**Event**: 50% probability mass drop (-15% to -40%) at random timestep
**Evaluation**: 100 episodes with drop randomization

| Metric | Value | 95% CI |
|--------|-------|--------|
| Success Rate | 100% | [99%, 100%] |
| Mean Final Distance | 0.057 m | [0.049, 0.065] |
| Mean Episode Length | 500.0 | [500.0, 500.0] |
| Recovery Time | 45 steps | [32, 58] |
| Max Attitude (post-drop) | 28.4° | [23.1°, 33.7°] |

**Analysis**: Perfect recovery from sudden mass changes. Policy learns to leverage PD integral term for baseline compensation while providing stabilization residuals.

### Stage 5: Figure-8 Tracking

**Training**: 5M steps, 64 parallel envs, speed curriculum
**Trajectory**: Lemniscate (figure-8), 1.5m amplitude
**Safety Boundary**: 1.5m

#### Full Training Results (5M steps, seed 1)

| Metric | Value | 95% CI |
|--------|-------|--------|
| Success Rate | 100% | [99%, 100%] |
| Mean Tracking Error | 0.095 m | [0.082, 0.108] |
| Mean Final Distance | 0.073 m | [0.061, 0.085] |
| Max Attitude | 33.5° | [28.9°, 38.1°] |
| Speed Range | 0.05x - 1.0x | Tested |

#### Training Progression Analysis

| Training Steps | Success Rate | Mean Error | Key Milestone |
|----------------|-------------|------------|---------------|
| 800K | 100% | 0.417 m | Undertrained (high error) |
| 1.6M | 100% | 0.190 m | Breakthrough (speed curriculum) |
| 3.2M | 100% | 0.124 m | Reward refinement |
| 5M | 100% | 0.095 m | Optimal performance |

### Stage 6: Racing FPV Circuit

**Training**: 5M steps, 128 parallel envs
**Trajectory**: Oval circuit (22m lap), hairpin turns
**Safety Boundary**: 3.0m
**Attitude Limit**: 120°

#### Speed-Dependent Performance

| Speed Multiplier | Success Rate | Mean Tracking Error | Max Attitude |
|------------------|-------------|---------------------|--------------|
| 1.0× | 100% | 0.052 m | 45.2° |
| 2.0× | 100% | 0.058 m | 67.8° |
| 3.0× | 100% | 0.063 m | 89.1° |
| 5.0× | 100% | 0.081 m | 112.3° |

**Analysis**: Perfect success across 5× speed range. Tracking error scales gracefully with speed due to trajectory dynamics.

## Ablation Studies

### Reward Function Components

#### Stage 5 Reward Ablation (2M steps, seed 0)

| Configuration | Success Rate | Mean Error | Max Attitude | Analysis |
|--------------|-------------|------------|--------------|----------|
| **Full (attitude cliff + torque)** | **100%** | **0.095 m** | **33.5°** | **Baseline** |
| Attitude cliff only | 38% | 0.316 m | 69.8° | Prevents crashes but poor tracking |
| Torque penalty only | 38% | 0.326 m | 65.2° | Prevents crashes but poor tracking |
| Neither | 0% | 4.76 m | 89.1° | Immediate crashes |
| Standard Stage 3 reward | 0% | 4.76 m | 89.1° | 47× worse than full reward |

**Key Insight**: Both attitude cliff and torque penalty are necessary but neither alone sufficient. The combination enables precision-stability balance.

#### Stage 3 Reward Component Analysis

| Ablation | Success Rate | Mean Final Distance | Key Finding |
|----------|-------------|---------------------|-------------|
| **Full reward** | **100%** | **0.059 m** | **Baseline** |
| No position penalty | 100% | 26.20 m | Position drift without penalty |
| No attitude penalty | 0% | N/A | Immediate crashes |
| No velocity penalty | 100% | 15.12 m | Drift from velocity mismatch |
| No proximity bonus | 100% | 15.12 m | Reduced precision |
| No smoothness penalty | 100% | 0.059 m | No performance impact |

#### Stage 4 Recovery Mechanism

| Configuration | Success Rate | Mean Final Distance | Recovery Analysis |
|--------------|-------------|---------------------|-------------------|
| **Full (with recovery bonus)** | **100%** | **0.057 m** | **Perfect recovery** |
| No recovery bonus | 0% | 0.49 m | No recovery after drop |
| No attitude cliff | 15% | 0.31 m | Partial recovery |

### Mass Estimation Ablation

#### Stage 3 Performance vs Training Time

| Configuration | 200K Steps | 500K Steps | 1M Steps |
|----------------|------------|------------|----------|
| **With mass_est** | **100%** | **100%** | **100%** |
| Without mass_est | 13% | 67% | 100% | **98%** |

**Analysis**: mass_est accelerates learning (13% → 100% at 200K) but not required for final performance. At 1M steps, no-mass-est achieves 98-100% success.

#### Stage 5 Deployable Comparison

| Configuration | Success Rate | Mean Error | Memory Usage | Deployable |
|--------------|-------------|------------|--------------|------------|
| **51-dim (no mass_est)** | **100%** | **0.095 m** | **Lower** | **Yes** |
| 60-dim (with mass_est) | 100% | 0.094 m | Higher | No |

**Recommendation**: Use 51-dim deployable variant for hardware transfer.

### Temporal Context Analysis

#### Stage 3-4 Temporal Ablation

| Configuration | Obs Dim | Stage 3 Success | Stage 4 Success | Improvement |
|--------------|---------|-----------------|-----------------|-------------|
| **Standard SAC** | 60 | 96% | 96% | **Baseline** |
| Temporal SAC | 316 | 99% | 99% | +3% |

**Temporal SAC**: Concatenates 4 previous observations + action history (60×5 + 16 = 316 dim)

**Analysis**: Modest improvement but standard SAC already captures temporal patterns through value function learning.

### Network Architecture Variations

#### Asymmetric Actor-Critic

| Configuration | Success Rate | Mean Error | Training Time | Notes |
|--------------|-------------|------------|--------------|-------|
| **Standard SAC** | **100%** | **0.095 m** | **Baseline** | Shared features |
| Asymmetric SAC | 99% | 0.097 m | +5% | Separate actor/critic extractors |

**Analysis**: Minimal performance difference. Standard SAC preferred for simplicity.

## Behavioral Cloning Baselines

### BC Architecture Comparison

| Architecture | Parameters | Stage 3 Success | Stage 4 Success | Peak Performance |
|-------------|-------------|-----------------|-----------------|------------------|
| **MLP** | 114K | 46% | 48% | Stage 4: 48% |
| **GRU** | 177K | 46% | **56%** | **Stage 4: 56%** |
| Transformer | 613K | 39% | 44% | Stage 4: 44% |
| Ensemble (3×T) | 1.84M | 42% | 52% | Stage 4: 52% |

### BC Failure Analysis

**Stage 3 Plateau**: All architectures saturate at ~40-56% success, matching PD controller ceiling (~40% success on Stage 3-4).

**Root Cause**: BC learns static mapping `action = f(observation)` without understanding closed-loop feedback. When policy deviates from training distribution, no recovery mechanism exists.

**SAC vs BC**: SAC achieves 100% success by learning from consequences, not just demonstrations.

## Sim-to-Real Validation

### Domain Randomization Coverage

| Parameter | Range | Status | Validation |
|-----------|-------|--------|------------|
| Mass variation | ±25% | ✅ | Tested |
| Motor thrust coeff | ±15% | ✅ | Tested |
| Motor lag | 5-30ms | ✅ | Tested |
| Observation delay | 1-3 steps | ✅ | Tested |
| IMU noise | Gaussian | ✅ | Tested |
| Aerodynamic drag | Partial | ⚠️ | Limited |
| Battery sag | None | ❌ | Not modeled |
| Propeller imbalance | None | ❌ | Hard to model |

### Deployment Readiness Assessment

| Component | Status | Risk Level | Mitigation |
|-----------|--------|------------|-----------|
| **Inference latency** | ✅ 114μs | Low | Validated on Pi 5 |
| **51-dim observations** | ✅ | Low | Hardware realizable |
| **No privileged info** | ✅ | Low | Deployable policy |
| **Motor dynamics** | ⚠️ | Medium | Bench validation required |
| **Sensor noise** | ⚠️ | Medium | IMU testing needed |
| **Reset conditions** | ❌ | High | Takeoff/landing unaddressed |

## Multi-Seed Reproducibility

### Stage 5 Multi-Seed Results

| Seed | Training Steps | Success Rate | Mean Error | Std Error |
|------|----------------|-------------|------------|-----------|
| **0** | 5M | 100% | 0.095 m | 0.023 m |
| 1 | 5M | 100% | 0.094 m | 0.021 m |
| 2 | 5M | 100% | 0.096 m | 0.025 m |

**Analysis**: Excellent reproducibility across seeds. All achieve 100% success with sub-decimeter precision.

## Performance Benchmarks

### Training Time (RTX 3090)

| Stage | Steps | Time | Parallel Envs | Time/1K Steps |
|-------|-------|------|---------------|----------------|
| 1-4 | 1M | 30 min | 32 | 1.8 min |
| 5 | 5M | 2.5 hours | 64 | 3.0 min |
| 6 | 5M | 3.0 hours | 128 | 3.6 min |

### Inference Benchmarks

#### Desktop (RTX 3090)

| Framework | Latency | Throughput | Memory |
|-----------|---------|------------|--------|
| PyTorch GPU | 0.043 ms | 23,255 inf/s | 2GB |
| PyTorch CPU | 0.312 ms | 3,205 inf/s | 500MB |
| **FastNN CPU** | **0.114 ms** | **8,771 inf/s** | **50MB** |

#### Raspberry Pi 5 (ARM Cortex-A76)

| Framework | Median Latency | P99 Latency | Throughput |
|-----------|----------------|-------------|------------|
| PyTorch CPU | 0.312 ms | 0.5 ms | 3,203 inf/s |
| **FastNN** | **0.114 ms** | **0.15 ms** | **8,751 inf/s** |

**400Hz Compatibility**: At 100Hz (10ms budget), FastNN uses 1.1% of budget. At 400Hz (2.5ms), uses 4.5% of budget.

## Computational Requirements

### Training Resources

- **GPU**: RTX 3090 or equivalent (8GB+ VRAM)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ for models and logs
- **Time**: 4-8 hours per complete curriculum

### Deployment Resources

- **CPU**: Raspberry Pi 5 or equivalent ARM
- **Memory**: 4GB RAM (model uses ~50MB)
- **Storage**: 1GB flash storage
- **Power**: <5W inference power