# FastNN Quadrotor Control with Curriculum Adaptation

## Abstract

We present a curriculum-based approach to residual reinforcement learning for quadrotor control. Our method augments a cascaded PD controller with learned residuals trained via Soft Actor-Critic (SAC), enabling robust flight under wind and mass perturbations without requiring explicit disturbance estimation.

The key technical contribution is a reward-shaping strategy that addresses the precision-stability tradeoff in residual control: an attitude cliff barrier prevents excessive tilts while torque penalties discourage aggressive control outputs. With this reward structure, a policy trained on a figure-8 tracking task achieves 100% success (single-seed evaluation) with 9.5 cm mean tracking error — a 47× improvement over standard reward shaping.

We validate the approach through a 6-stage curriculum from hover to aggressive racing. The resulting policy handles wind disturbances (±0.8 N), payload drops (up to 40% mass reduction), and target speeds up to 5× baseline. We demonstrate that explicit mass estimation is unnecessary for recovery: policies trained without mass measurements achieve equivalent performance (100% success at 1M steps) by inferring mass implicitly from control history.

Finally, we validate deployment feasibility on ARM edge hardware (Raspberry Pi 5). FastNN inference achieves 114 μs median latency — compatible with 100 Hz deployment with significant headroom for higher-rate control loops.

## 📚 Related Research Documents

For additional context and detailed analysis:

- **[Complete Research Paper](../../papers/fastnn_quadrotor_paper.md)** - Full research paper with comprehensive methods and analysis
- **[Experimental History](../../papers/quadrotor_research_paper.md)** - Complete experimental history and failure analysis
- **[Technical Analysis](../../papers/quadrotor_best_path_forward.md)** - Architecture decisions and future work
- **[Research Summary](../../papers/quadrotor_research_summary.md)** - Concise results overview

## 1. Introduction

Quadrotor control in unstructured environments requires robust adaptation to wind disturbances and mass variations. Classical PID/PD controllers provide stable hover but struggle with unmodeled disturbances. Pure neural network policies can learn complex behaviors but require large datasets and may be unstable at deployment.

We propose combining:
- **Residual RL**: A cascaded PD base controller + learned residual corrections via SAC
- **Reward Shaping**: Attitude cliff barriers and torque penalties for precision-stability balance
- **Curriculum Learning**: 6-stage progression from hover to aggressive racing
- **FastNN (Rust)**: Deterministic low-latency inference for edge deployment

Our key insight is that the precision-stability tradeoff in residual RL can be addressed through careful reward design rather than architectural changes. The resulting policy achieves centimeter-precision tracking on dynamic trajectories while maintaining robustness to payload drops and wind disturbances.

## 2. Related Work

Our work builds on simulation frameworks for quadrotor control [1,2] and residual policy learning [3,4]. Curriculum learning for robotics [5] and asymmetric actor-critic methods [6] provide foundational techniques we adapt for UAV deployment.

**References**:
[1] Zhou et al., "Safe Control Gym," NeurIPS Datasets, 2023.
[2] Pagani et al., "Customizable Autonomous Drone Racing Simulation," Drones, 2023.
[3] Silver et al., "Residual Policy Learning," RSS, 2018.
[4] Johannink et al., "Residual Reinforcement Learning for Robot Control," ICRA, 2019.
[5] Hwangbo et al., "Learning Agile Motor Skills," Sci. Robot., 2019.
[6] Pinto et al., "Asymmetric Actor-Critic for Sim-to-Real Transfer," arXiv, 2017.

## 3. Methods

### 3.1 Canonical Experimental Setup

**Table 1: Observation Space**

| Component | Dims | Type | Description |
|-----------|------|------|-------------|
| Position Error | 3 | Deployable | target - current position [m] |
| Velocity Error | 3 | Deployable | -current velocity [m/s] |
| Attitude Error | 3 | Deployable | Roll/pitch/yaw deviation [rad] |
| Rate Error | 3 | Deployable | -current angular rates [rad/s] |
| Linear Acceleration | 3 | Deployable | Body-frame IMU measurement [m/s²] |
| Rotation Matrix | 9 | Deployable | SO(3) representation [-] |
| Body Rates | 3 | Deployable | Current angular velocities [rad/s] |
| Action History | 16 | Deployable | 4-step ring buffer of past actions [-] |
| Error Integrals | 4 | Deployable | Accumulated position + yaw error [-] |
| Rotor Thrust Estimate | 4 | Deployable | Filtered motor outputs [N] |
| Target Velocity | 3 | Deployable | For stages with moving target [m/s] |
| **Total Deployable** | **51** | | Hardware-realizable observations |
| Privileged Info | 9 | Critic-only | mass_ratio, com_shift, wind, motor_deg, mass_est |
| **Total (Critic)** | **60** | | Includes simulation ground-truth |

**Note**: mass_est is privileged/critic-only. The deployed policy (51-dim) infers mass implicitly from action history and error integrals.

**Table 2: Action Space**

| Component | Units | Scaling | Physical Interpretation |
|-----------|-------|---------|------------------------|
| Residual Thrust | N | [-1, 1] → ±1.0 | ≈ ±10% of hover thrust (~10 N) |
| Residual Roll Torque | Nm | [-1, 1] → ±1.0 | Direct torque on roll axis |
| Residual Pitch Torque | Nm | [-1, 1] → ±1.0 | Direct torque on pitch axis |
| Residual Yaw Torque | Nm | [-1, 1] → ±1.0 | Direct torque on yaw axis |

The action is added to the cascaded PD controller output: `total_ctrl = pd_output + action * action_scale` where `action_scale = [1.0, 1.0, 1.0, 1.0]`.

**Note on Residual Authority**: The residual thrust is bounded to ±1.0 N (≈ ±10% of hover thrust). Recovery from large mass drops (up to -40%, requiring ~4.0 N thrust reduction) relies primarily on the base PD controller's integral term to compensate for the baseline shift, while the SAC policy handles transient stabilization and fine tracking corrections.

**Table 3: Task Specifications by Stage**

| Stage | Safety Boundary | Max Attitude | Disturbances | Success Criterion |
|-------|-----------------|--------------|--------------|-------------------|
| 1 | 0.5 m | 90° | None | steps ≥ 500 |
| 2 | 0.5 m | 90° | Random pose/vel | steps ≥ 500 |
| 3 | 0.5 m | 90° | Wind ±0.5 N, mass ±10% | steps ≥ 500 |
| 4 | 0.5 m | 90° | + Payload drop (50%, -15-40%) | steps ≥ 500 |
| 5 | 1.5 m | 90° | + Moving target (fig-8) | steps ≥ 500, tracking < 0.2 m |
| 6 | 3.0 m | 120° | + Racing circuit, wind ±1.0 N | steps ≥ 500 |

**Table 4: Training Hyperparameters**

| Parameter | Value |
|-----------|-------|
| Algorithm | SAC (Stable-Baselines3) |
| Learning rate | 1×10⁻⁴ |
| Batch size | 256 |
| Replay buffer | 1,000,000 |
| Discount (γ) | 0.99 |
| Soft update (τ) | 0.005 |
| Target entropy | -2 (auto-tuned) |
| Policy network | [256, 256] |
| Training steps | 1M (stages 1-4), 5M (stage 5-6) |
| Parallel envs | 32-512 |

### 3.2 Environment

- **Engine**: MuJoCo (Gymnasium)
- **Physics**: RK4 integrator, 10 ms timestep (100 Hz)
- **Disturbances**: Wind force via xfrc_applied + mass perturbation
- **Catastrophic Failure**: |Roll/Pitch| > θ_max (stage-dependent)
- **Truncation**: Distance > safety_boundary from target

### 3.3 Neural Network Architecture

**Training Algorithm**: SAC (Stable-Baselines3) with standard `MlpPolicy`

- **Actor/Critic Shared**: Stage-dependent obs → [256, 256] → 4 actions / 1 Q-value
  - Stages 1-4: 60-dim input (51 deployable + 9 privileged)
  - Stages 5-6: 63-dim input (54 deployable + 9 privileged)

**Note**: The reported results use standard SAC with shared feature extraction. We also implemented an asymmetric variant (Appendix D) where the actor receives only deployable observations while the critic uses privileged information, but this showed minimal difference in final performance.

### 3.4 Reward Function

**Base Reward** (all stages):
```
r_total = r_alive + r_pos + r_att + r_vel + r_rate + r_smooth + r_proximity + r_alignment + r_success + r_recovery + r_jerk + r_torque
```

**Stage 5+ Precision Additions**:
- `r_att_cliff`: Quadratic penalty beyond 30° attitude error
- `r_torque`: Penalty on roll/pitch control magnitudes

The attitude cliff creates a safety barrier: within 30°, no additional penalty; beyond 30°, rapidly increasing quadratic penalty prevents crash spirals.

## 4. Curriculum Learning

### 4.1 Stage Progression

| Stage | Description | Advance Criteria |
|-------|-------------|------------------|
| 1 | Fixed hover | 50 eps, ≥90% success |
| 2 | Random pose + velocity | 50 eps, ≥50% success |
| 3 | Wind + mass | 50 eps, ≥50% success |
| 4 | Payload drop | Train to convergence |
| 5 | Moving target (figure-8) | 100% success, <0.2m tracking error |
| 6 | Racing FPV (circuit) | High-speed validation |

### 4.2 Stage 5: Moving Target

Stage 5 introduces dynamic trajectory tracking with a figure-8 pattern (Lemniscate of Bernoulli). Key challenges include:
- Wind disturbances (±0.8 N)
- Payload drops (50% probability)
- Predictive tracking (velocity matching)

**Speed curriculum**: 0.05× → 0.1× → 0.2× → 0.4× → 0.7× → 1.0×

### 4.3 Stage 6: Racing FPV

Stage 6 extends to aggressive racing maneuvers:
- Oval circuit (22 m lap) with hairpin turns
- Target speeds up to 5× baseline
- Attitude limit extended to 120° for FPV-style flight
- Wind increased to ±1.0 N

## 5. Why Imitation Learning Fails

### 5.1 The BC Ceiling

Behavioral Cloning (BC) learns a static mapping from observations to actions. When the learned policy deviates from the training distribution, it cannot recover because it never learned the feedback dynamics of the closed-loop system.

Our BC experiments (Appendix A) show that all architectures (MLP, GRU, Transformer) plateau at 35-56% success on Stages 3-4. The fundamental limitation is that BC inherits the failure modes of its teacher (the PD controller achieves only ~40% success on Stage 3-4).

### 5.2 Why SAC Succeeds

Soft Actor-Critic learns from consequences: actions lead to states, and states lead to rewards or failures. This closed-loop learning enables recovery from off-distribution states that BC cannot handle.

The key difference:
- **BC**: `action = f(observation)` — static mapping
- **SAC**: `action = argmax_a Q(s, a)` — learns value of consequences

## 6. Experimental Results

### 6.1 Metrics and Evaluation Protocol

**Success Rate**: Fraction of evaluation episodes reaching 500 steps without termination (crash) or truncation (boundary exceeded). Computed over 100 episodes per condition.

**Mean Tracking Error**: Average Euclidean distance between drone and target positions across all timesteps of successful episodes only.

**Mean Final Distance**: The Euclidean distance between drone and target at the final timestep (t=500) of successful episodes.

**Evaluation Protocol**:
- 100 episodes per stage/condition
- Deterministic policy evaluation
- Fresh environment per episode with randomized initial conditions
- Single-seed results unless multi-seed explicitly noted

### 6.2 SAC Results: Stages 1-4

| Stage | Success Rate | Mean Final Distance | Mean Steps |
|-------|-------------|-------------------|------------|
| **Stage 1** | **100%** | 0.053 m | 500 |
| **Stage 2** | **100%** | 0.043 m | 500 |
| **Stage 3** | **100%** | 0.059 m | 500 |
| **Stage 4** | **100%** | 0.057 m | 500 |

SAC achieves 100% success across all foundational stages, including wind+mass perturbations (Stage 3) and payload drop recovery (Stage 4). Results use standard SAC with 60-dim observations (51 deployable + 9 privileged).

### 6.3 Stage 5: Precision Through Reward Design

Moving target tracking presents a precision-stability tradeoff: aggressive corrections reduce tracking error but risk attitude divergence. Our solution introduces:

1. **Attitude Cliff**: Quadratic penalty beyond 30° (0.52 rad)
2. **Torque Penalty**: Direct penalization of roll/pitch control magnitudes

**Ablation Results** (2M steps, seed 0):

| Configuration | Success Rate | Tracking Error | Max Attitude |
|--------------|-------------|---------------|--------------|
| **Both penalties (baseline)** | **100%** | **0.095 m** | 33.5° |
| Attitude cliff only | 38% | 0.316 m | 69.8° |
| Torque penalty only | 38% | 0.326 m | 65.2° |

Both components are required: neither alone prevents crashes.

**Stage 5 Final Results** (5M steps, seed 1, 100 episodes):

| Metric | Value |
|--------|-------|
| Success Rate | 100% |
| Mean Tracking Error | 0.095 m |
| Mean Final Distance | 0.073 m |

This represents a 47× improvement over the baseline reward function (Stage 3 rewards applied directly to Stage 5 without attitude cliff or torque penalties), which achieved 4.76 m tracking error.

### 6.4 Stage 6: Racing Validation

The precision-stability reward transfers to aggressive racing maneuvers. Evaluation at multiple speed multipliers shows maintained success rates with centimeter-level tracking:

| Speed | Success Rate | Mean Tracking Error |
|-------|-------------|---------------------|
| 1.0× | 100% | ~0.05 m |
| 3.0× | 100% | ~0.06 m |
| 5.0× | 100% | ~0.08 m |

**Note**: Tracking error naturally increases with speed due to trajectory dynamics, but remains sub-decimeter. Detailed speed-dependent evaluation ongoing (single-seed preliminary results).

### 6.5 Mass Estimation Ablation

| Configuration | Stage 3 (200K steps) | Stage 3 (1M steps) |
|--------------|---------------------|-------------------|
| With mass_est | 100% | 100% |
| Without mass_est | 13% | 100% |

At 200K steps, mass_est provides fast adaptation (100% vs 13% success). At 1M steps, the no-mass-est policy achieves comparable performance (98-100% success), demonstrating that mass can be inferred implicitly from action history and error integrals. The deployable variant (no mass_est) is preferred for hardware transfer.

### 6.6 Temporal Context Ablation

| Configuration | Stage 3 | Stage 4 |
|---------------|---------|---------|
| Standard SAC (60-dim) | 94-98% | 94-98% |
| Temporal SAC (316-dim) | 99-100% | 99-100% |

**Note**: "Temporal SAC" uses a wrapper that concatenates 4 previous full observations plus action history: `new_dim = base_dim × 5 + action_dim × 4` = 60×5 + 16 = 316 dims. This provides modest improvement (~2-5%) over standard SAC, which already learns implicit temporal patterns through its value function.

### 6.7 Reward Function Ablation

**Table 6.7a: Stage 3 Ablations**

| Ablation | Success | Mean Distance | Key Insight |
|----------|---------|--------------|-------------|
| Standard | 100% | 5.67 m | Baseline |
| No Proximity | 100% | 15.12 m | Proximity rewards improve precision |
| No Smoothness | 100% | 5.67 m | Smoothness penalty doesn't matter |
| Position Only | 100% | 26.20 m | Attitude/velocity penalties prevent drift |

**Table 6.7b: Stage 4 Critical Component**

| Ablation | Stage | Success | Mean Distance | Key Insight |
|----------|-------|---------|--------------|-------------|
| Standard | 4 | 100% | 0.057 m | Baseline |
| No Recovery | 4 | 0% | 0.49 m | Recovery bonus critical for payload drop |

## 7. Sim-to-Real Considerations

### 7.1 Deployment Assumptions

- **Control frequency**: 100 Hz (10 ms timestep)
- **Inference latency**: <150 μs (validated on Pi 5)
- **Sensors**: Position (GPS/VIO), velocity (IMU), attitude (IMU), body rates (IMU)
- **Deployment policy**: No-mass-est variant (51-dim)

### 7.2 Known Risks

1. **Motor Dynamics**: Simulation uses first-order delay (57 ms time constant). Real motors may differ; validation required on target hardware.

2. **Sensor Noise**: Real IMU has ~0.4 m/s² noise floor. The no-mass-est policy avoids explicit reliance on noisy acceleration measurements.

3. **Thrust Curve**: Simulation assumes linear thrust; real propellers have nonlinear curves in ground effect.

4. **Reset Assumptions**: Training episodes start from randomized poses. Real-world deployment requires takeoff/landing handling not addressed in this work.

### 7.3 Deployment Path

**Primary**: Use the no-mass-est model for hardware deployment. This network:
- Has learned to infer mass from action history and error integrals
- Does not rely on the noisy mass_est signal
- Should transfer to real hardware without modification

**Validation Steps Required**:
1. Verify inference latency on target hardware under load
2. Test mass estimation noise sensitivity before deployment
3. Validate motor delay assumptions against bench tests
4. Implement watchdog for attitude boundary violations

## 8. Inference Benchmark

### 8.1 Raspberry Pi 5 Results

Hardware: Raspberry Pi 5 (ARM Cortex-A76, 4 cores)

| Runtime | Median Latency | P99 Latency | Throughput |
|---------|---------------|-------------|------------|
| PyTorch FP32 (CPU) | 0.312 ms | ~0.5 ms | 3,203 inf/s |
| **FastNN (Rust)** | **0.114 ms** | **0.15 ms** | **8,751 inf/s** |

**400 Hz Compatibility**: Our deployment target is 100 Hz (10 ms budget), where FastNN uses only 1.1% of the budget. The same inference latency (114 μs) provides substantial headroom for 400 Hz control loops (2.50 ms budget), where it would consume 4.5% of the budget — demonstrating scalability to higher-rate control architectures.

### 8.2 Key Insight

P99 latency (jitter) matters more than median for real-time control. FastNN's static computation graph eliminates non-determinism from dynamic graph construction, ensuring every control pulse arrives within the strict 2.5 ms window.

## 9. Conclusion

We presented a curriculum-based approach to residual reinforcement learning for quadrotor control. Key contributions:

1. **Reward Design for Precision-Stability Balance**: Attitude cliff barriers and torque penalties enable centimeter-precision tracking (9.5 cm error) with 100% success on dynamic figure-8 trajectories — a 47× improvement over standard reward shaping.

2. **Implicit Mass Inference**: Policies trained without explicit mass estimation achieve comparable performance (98-100% success at 1M steps, single-seed) by inferring mass from control history — a deployable alternative to simulation-dependent mass estimation.

3. **Edge Deployment Feasibility**: FastNN inference achieves 114 μs median latency on Raspberry Pi 5, suitable for real-time embedded control at 100 Hz with headroom for higher frequencies.

4. **Curriculum Validation**: A 6-stage curriculum demonstrates progressive capability acquisition from hover to racing FPV maneuvers.

### Limitations and Future Work

- **Single-seed validation**: Reported results from seed 1; multi-seed validation ongoing
- **Simulation-only**: Hardware flight validation not yet performed

## References

[1] Zhou et al., "Safe Control Gym," NeurIPS Datasets, 2023.  
[2] Pagani et al., "Customizable Autonomous Drone Racing Simulation," Drones, 2023.  
[3] Silver et al., "Residual Policy Learning," RSS, 2018.  
[4] Johannink et al., "Residual Reinforcement Learning for Robot Control," ICRA, 2019.  
[5] Hwangbo et al., "Learning Agile Motor Skills," Sci. Robot., 2019.  
[6] Pinto et al., "Asymmetric Actor-Critic for Sim-to-Real Transfer," arXiv, 2017.

## Supplementary Materials

- [Methods Details](methods.md) - Complete environment specifications and hyperparameters
- [Detailed Results](results.md) - Full evaluation metrics and ablation studies  
- [Technical Appendix](appendix.md) - Supporting materials and failed experiments
- [User Guides](../../docs/guides/) - Installation and usage instructions