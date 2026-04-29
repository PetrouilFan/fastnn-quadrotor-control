# Technical Paper Audit Report: FastNN Quadrotor Control

## Executive Diagnosis

### Critical Issues (Must Fix Before Submission)

1. **FUNDAMENTAL NARRATIVE CONTRADICTION**: Section 5.3 claims "residual RL cannot overcome PD limitations" (46% oracle success), while Sections 10-13 celebrate SAC residual RL achieving 100% success. These cannot both be true. The resolution: The "oracle residual" was open-loop feedforward, not closed-loop RL. SAC learns feedback control.

2. **IN-PROGRESS WORK IN ABSTRACT**: Stages 7 and 8 are described as achievements in the abstract (lines 7-13), but results sections clearly state "Training in progress" (Section 13.10, 13.11). Stage 7 shows 0-32% success; Stage 8 shows 76-93% intermediate results with "full 10M training in progress."

3. **DIMENSIONAL INCONSISTENCIES**: Paper states 60-dim obs (line 5, Section 3.2), 52-dim (Section 3.4), 51-dim (Section 10.6), and 63-dim (Section 13.5). No canonical specification exists.

4. **IMPOSSIBLE METRIC**: Abstract claims "0.095m tracking error" with "1.5m safety boundary" — this is physically consistent, BUT Section 6.1 reports 4.76m tracking error with 100% success, which is impossible if safety boundary is 0.5m (Section 3.1) or 1.5m (Section 13.5).

5. **SYMMETRIC VS ASYMMETRIC AC CONTRADICTION**: Section 3.4 says "symmetric actor-critic" but Section 3.5 shows both actor and critic receiving 52-dim input (which is symmetric). However, Section 5.4 discusses privileged mass_est for critic, implying asymmetric design. The actual code shows 51 deployable + 9 privileged = 60 total, with critic seeing all 60 and actor seeing 51.

6. **OVERCLAIMED SIM-TO-REAL**: Section 16 presents noise-injected mass estimation as a completed robustification strategy, but no results are shown. The no-mass-est model is the only validated deployment candidate.

### High Severity Issues

7. **SINGLE-SEED RESULTS PRESENTED AS CONCLUSIVE**: Stage 5 "100% success" is from seed_1 only (results file shows 1 seed). Stage 6 "100%" appears to be single-seed from training script default.

8. **BENCHMARK TABLE MIXES HARDWARE**: Section 14 presents Pi 5 results and Desktop results without clear separation. Pi 5 shows 0.114ms median, Desktop shows 43.90μs — different hardware, cannot be directly compared in same table.

9. **MASS EST CLAIM CONTRADICTION**: Section 10.4 says "mass_est is NOT load-bearing" and "network CAN succeed without mass_est" but also says 200K steps without mass_est achieves only 13% vs 100% with mass_est. The claim is only true at 1M steps, not "fast adaptation."

10. **REWARD FUNCTION ABLATION INCONSISTENT**: Section 10.7 shows "Standard" achieving 100% with 5.67m mean distance, but Section 10.2 shows 100% with 0.059m. Which is correct? These appear to be different metrics (Stage 3 vs Stage 5) but are presented without distinction.

### Medium Severity Issues

11. **TRANSFORMER INTERPRETATION CONTRADICTION**: Section 9.8 claims "Transformer's underperformance is NOT data-limited" but then states "with sufficient data or pre-training, Transformers may match or exceed GRU." Pick one interpretation.

12. **STAGE 4 SUCCESS INCONSISTENCY**: Section 6.1 reports 30-50% success; Section 10.2 reports 100%. These are clearly different experiments (old vs new training) but not clearly distinguished.

13. **MISSING BC EXPERIMENT DETAILS**: Section 9.6 states BC is trained on PD controller data achieving 40-88% success, but Section 5.2 mentions "Propose training BC on SAC demonstrations" as future work. This validation experiment was apparently never run.

14. **CPT REWARD UNVALIDATED**: Section 13.10 describes Convergence-Predictive Tracking as novel, but training results show 0-32% success. Cannot claim novelty without validation.

### Low Severity Issues

15. **"SOLVED" LANGUAGE**: Line 767: "precision-stability tradeoff SOLVED" — remove hype words per guidelines.

16. **"SEAMLESSLY"**: Line 1223: "scales seamlessly" — unsupported causal claim.

17. **"EXTREME AGILITY"**: Line 9: "demonstrating that our approach scales to extreme agility" — overclaim; only tested to 5x speed.

18. **50X vs 47X IMPROVEMENT**: Line 7 claims 50x improvement, line 851 claims 47x. Inconsistent.

---

## Canonical Decisions

| Topic | Conflicting Versions | Final Chosen Version | Rationale |
|-------|---------------------|----------------------|-----------|
| **Observation Dim** | 60-dim (abstract), 52-dim (Section 3.4), 51-dim (Section 10.6), 63-dim (Section 13.5) | **51 deployable + 9 privileged = 60 total** | Code: `env_rma.py:1552-1594` shows 51 deployable (no mass_est) + 9 privileged |
| **Actor Input Dim** | 52 (Section 3.5), 51 (Section 5.4) | **51** | Code confirms mass_est removed from deployable; critic sees 60 |
| **Safety Boundary** | 0.5m (Section 3.1), 1.5m (Stage 5), 3.0m (Stage 6), 6.0m (Stage 7), 5.0m (Stage 8) | **Stage-dependent: 0.5m (1-4), 1.5m (5), 3.0m (6), 6.0m (7), 5.0m (8)** | Code: `env_rma.py:870-879` shows stage-dependent boundaries |
| **Action Scale** | [1.0, 1.0, 1.0, 1.0] (Section 3.3), 0.5→1.0 (Section 3.4) | **[1.0, 1.0, 1.1, 1.0]** | Code: `env_rma.py:87` shows action_scale=1.0 for all dims |
| **Timestep** | 10ms (Section 3.1), 0.01s (code) | **0.01s (100 Hz)** | Code: `env_rma.py:61` and `env_rma.py:51` confirm 0.01s |
| **Control Frequency** | Implied 100Hz from timestep | **100 Hz** | dt=0.01s implies 100Hz |
| **Residual Type** | Oracle 46% (Section 5.1) vs SAC 100% (Section 10) | **Oracle was open-loop; SAC is closed-loop** | Section 5.4 clarifies oracle was open-loop feedforward |
| **Stage 5 Success** | 90% average (Section 13.5), 100% (abstract) | **100% with new reward (seed 1 only)** | `results_stage5_curriculum/stage_5.json` shows seed_1: 100% success, 0.094m tracking error |
| **Mass Est Status** | Deployable (early), Privileged (Section 5.4), Removed (code) | **Removed from deployable; exists in privileged for critic only** | Code: `env_rma.py:1552` shows mass_est NOT in deployable parts |
| **Stage 7/8 Status** | "Implemented" (curriculum table), "In Progress" (results) | **IN PROGRESS — remove from abstract** | Sections 13.10, 13.11 show training incomplete |
| **Success Metric** | "steps >= 500" (Section 13.5 code), "100 episodes" (Section 10.2) | **Episode completion without truncation/termination** | Code: `train_stage5_curriculum.py:190-191` shows steps >= max_episode_steps |

---

## Paper Restructure Plan

### Proposed Section Order

1. **Abstract** — Conservative, publication-style, exclude in-progress work
2. **Introduction** — Frame as "SAC-based residual control with curriculum"
3. **Related Work** — Keep minimal
4. **Methods**
   - 4.1 Canonical Setup Table (THE authoritative spec)
   - 4.2 Environment (MuJoCo, physics, disturbances)
   - 4.3 Observation and Action Space (frozen dims)
   - 4.4 Reward Design (attitude cliff explanation)
   - 4.5 Training (SAC hyperparameters)
5. **Curriculum Design** — Stages 1-6 only (completed)
6. **Theoretical Analysis** — BC ceiling, why imitation fails
7. **Results**
   - 7.1 Stage 1-4: SAC achieves 100%
   - 7.2 Stage 5: Moving target with precision
   - 7.3 Stage 6: Racing validation
   - 7.4 Ablations (reward, mass_est, temporal)
8. **Sim-to-Real Considerations** — Conservative deployment path
9. **Inference Benchmarks** — Pi 5 results only for deployment relevance
10. **Conclusion** — Defensible claims only
11. **Appendix** — BC experiments (failed), Stage 7/8 (ongoing), full hyperparameters

### Sections to Cut/Merge/Demote

| Section | Action | Rationale |
|---------|--------|-----------|
| 5.1-5.3 (Root Cause: residual RL failure) | **REWRITE** | Contradicts later SAC success; frame as "BC ceiling" not "residual RL invalid" |
| 5.4 (Oracle vs SAC) | **PROMOTE** | Key insight: open-loop vs closed-loop |
| 9 (Temporal Models/BC) | **APPENDIX** | Failed approaches belong in appendix |
| 12 (Complete Journey) | **DELETE** | Timeline narrative; not publication structure |
| 13.10 (Stage 7) | **APPENDIX** | Incomplete (0-32% success) |
| 13.11 (Stage 8) | **APPENDIX** | Incomplete (76-93% intermediate) |
| 13.6-13.9 | **KEEP** | Stage 5-6 completed results |

---

## Rewritten Contribution List (Max 4)

**OLD (problematic):**
> 6. BREAKTHROUGH: We solved the precision-stability tradeoff... 50x improvement... seamless scaling to extreme agility

**NEW (defensible):**

1. **Reward Design for Precision-Stability Balance**: We identify that the precision-stability tradeoff in residual RL can be addressed through reward shaping (attitude cliff barrier + torque penalties). A SAC policy trained with these modifications achieves 100% success (single-seed) with 9.5cm mean tracking error on a dynamic figure-8 trajectory — a 47× improvement over policies trained with standard reward shaping.

2. **Curriculum Progression Validation**: We demonstrate that a 6-stage curriculum (hover → random pose → wind+mass → payload drop → moving target → racing) enables single-policy progression through tasks of increasing complexity. The final policy handles wind disturbances (±0.8N), payload drops (up to 40% mass reduction), and target speeds up to 5× baseline.

3. **Implicit Mass Inference**: We show that explicit mass estimation (thrust/acceleration) is not required for payload drop recovery. A policy trained without mass_est achieves equivalent final performance (100% success at 1M steps vs 200K with mass_est) by inferring mass implicitly from action history and error integrals — a deployable alternative to simulation-dependent mass estimation.

4. **Edge Deployment Feasibility**: We validate that the trained policies can run within real-time constraints on ARM edge hardware (Raspberry Pi 5). FastNN inference achieves 114μs median latency (8,751 inf/s), consuming 4.5% of a 400Hz control loop budget.

**NOT INCLUDED (unsupported):**
- Stage 7/8 results (incomplete)
- "Breakthrough" / "seamless" / "extreme agility" language
- Multi-seed statistical validation (only seed 1 validated for Stage 5-6)
- Hardware flight validation (simulation-only)

---

## Rewritten Abstract

> **Abstract**: We present a curriculum-based approach to residual reinforcement learning for quadrotor control. Our method augments a cascaded PD controller with learned residuals trained via Soft Actor-Critic (SAC), enabling robust flight under wind and mass perturbations without requiring explicit disturbance estimation.
>
> The key technical contribution is a reward-shaping strategy that addresses the precision-stability tradeoff in residual control: an attitude cliff barrier prevents excessive tilts while torque penalties discourage aggressive control outputs. With this reward structure, a policy trained on a figure-8 tracking task achieves 100% success (single-seed evaluation) with 9.5cm mean tracking error — a 47× improvement over standard reward shaping.
>
> We validate the approach through a 6-stage curriculum from hover to aggressive racing. The resulting policy handles wind disturbances (±0.8N), payload drops (up to 40% mass reduction), and target speeds up to 5× baseline. We demonstrate that explicit mass estimation is unnecessary for recovery: policies trained without mass measurements achieve equivalent performance (100% success at 1M steps) by inferring mass implicitly from control history.
>
> Finally, we validate deployment feasibility on ARM edge hardware (Raspberry Pi 5). FastNN inference achieves 114μs median latency, consuming 4.5% of a 400Hz control loop budget — suitable for real-time embedded deployment.

**Changes from original:**
- Removed "solved", "breakthrough", "seamless", "extreme agility"
- Removed Stage 7/8 (incomplete)
- Added "single-seed" qualifier where appropriate
- Specified "47×" consistently (was "50×")
- Added "up to" for speed claims

---

## Rewritten Setup + Metrics Sections

### 4.1 Canonical Experimental Setup

**Table 1: Canonical Observation Space**

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

| Component | Units | Range | Physical Interpretation |
|-----------|-------|-------|------------------------|
| Residual Thrust | N | [-1, 1] scaled to ±1.0 N (≈ ±10% hover) |
| Residual Roll Torque | Nm | [-1, 1] scaled to ±1.0 Nm |
| Residual Pitch Torque | Nm | [-1, 1] scaled to ±1.0 Nm |
| Residual Yaw Torque | Nm | [-1, 1] scaled to ±1.0 Nm |

The action is added to the cascaded PD controller output: `total_ctrl = pd_output + action * action_scale` where `action_scale = [1.0, 1.0, 1.0, 1.0]`.

**Table 3: Task Specifications by Stage**

| Stage | Safety Boundary | Max Attitude | Disturbances | Success Criterion |
|-------|-----------------|--------------|--------------|-------------------|
| 1 | 0.5m | 90° | None | steps ≥ 500 |
| 2 | 0.5m | 90° | Random pose/vel | steps ≥ 500 |
| 3 | 0.5m | 90° | Wind ±0.5N, mass ±10% | steps ≥ 500 |
| 4 | 0.5m | 90° | + Payload drop (50%, -15-40%) | steps ≥ 500 |
| 5 | 1.5m | 90° | + Moving target (fig-8) | steps ≥ 500, tracking error < 0.5m |
| 6 | 3.0m | 120° | + Racing circuit, wind ±1.0N | steps ≥ 500 |

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

### 7. Metrics and Evaluation Protocol

**Metric Definitions:**

- **Success Rate**: Fraction of evaluation episodes reaching 500 steps without termination (crash) or truncation (boundary exceeded). Computed over 100 episodes per condition.

- **Mean Tracking Error**: Average Euclidean distance between drone and target positions across all timesteps of successful episodes only. Does not include failed episodes.

- **Mean Final Distance**: Final distance between drone and target at episode end, averaged across all episodes (success and failure).

- **Episode Length / Mean Steps**: Average number of steps before episode termination. For 100% success rate, this equals 500.

- **Catastrophic Failure**: Episode terminates due to |roll| > θ_max or |pitch| > θ_max (θ_max stage-dependent: 90° stages 1-5, 120° stage 6).

- **Truncation**: Episode ends due to boundary violation (dist > safety_radius) or step limit (500).

- **Recovery Success**: After payload drop (Stage 4), percentage of episodes where drone returns to within 0.15m position error and 0.15rad attitude error within remaining episode duration.

**Evaluation Protocol:**
- 100 episodes per stage/condition
- Deterministic policy evaluation (no exploration noise)
- Fresh environment per episode with randomized initial conditions per stage specification
- Reported results from single seed unless multi-seed explicitly noted

---

## Rewritten Results Narrative

### 7.1 From BC Ceiling to RL Success

Early experiments with Behavioral Cloning (BC) plateaued at 46-56% success across MLP, GRU, and Transformer architectures (Appendix A). The fundamental limitation is that BC learns a static mapping from observations to actions without understanding the feedback dynamics of the closed-loop system. When the learned policy deviates from the training distribution, it cannot recover — a consequence of the imitation ceiling inherited from the PD controller's limited capability (~40% success on Stage 3-4).

Switching to Soft Actor-Critic (SAC) enables learning from consequences: the policy discovers that actions lead to states, and states lead to rewards or failures. This closed-loop learning achieves 100% success on Stages 1-4 (Table X), surpassing the BC ceiling.

### 7.2 Stage 5: Precision Through Reward Design

Moving target tracking presents a precision-stability tradeoff: aggressive corrections reduce tracking error but risk attitude divergence. Our solution introduces two reward modifications:

1. **Attitude Cliff**: Quadratic penalty activates only beyond 0.52 rad (30°), creating a safety barrier while permitting maneuvering freedom within the safe zone.

2. **Torque Penalty**: Direct penalization of roll/pitch control outputs discourages aggressive flips.

Ablation shows both components are necessary: attitude cliff alone achieves 38% success; torque penalty alone achieves 38%; combined achieves 100% (Table Y).

With these modifications, the policy achieves 100% success (seed 1, 100 episodes) with 9.5cm mean tracking error — a 47× improvement over the 4.76m error from standard reward shaping.

### 7.3 Stage 6: Racing Validation

The Stage 6 racing circuit (22m lap, hairpin turns) validates that the precision-stability reward transfers to aggressive maneuvers. At 5× target speed, the policy maintains 100% success with 5.2cm tracking error. The wider safety boundary (3.0m vs 1.5m for Stage 5) accommodates the higher speeds while the attitude limit (120°) permits FPV-style aggressive flight.

### 7.4 Mass Estimation Ablation

We evaluate whether the policy requires explicit mass estimation for payload drop recovery. At 200K steps, policies with mass_est achieve 100% success vs 13% without — the signal accelerates learning. However, at 1M steps, the no-mass-est policy achieves 100% success, demonstrating that mass can be inferred implicitly from action history and error integrals. The deployable variant (no mass_est) is preferred for hardware transfer.

---

## Sim-to-Real Section Rewrite

### 8. Sim-to-Real Considerations

**Deployment Assumptions:**
- Control frequency: 100 Hz (10ms timestep)
- Inference latency: <150μs (validated on Pi 5)
- Sensors: Position (GPS/VIO), velocity (IMU), attitude (IMU), body rates (IMU)
- No explicit mass estimation required (use no-mass-est policy)

**Known Risks:**

1. **Motor Dynamics**: Simulation uses first-order delay (57ms time constant). Real motors may differ; validation required on target hardware.

2. **Sensor Noise**: Training uses ideal sensors. Real IMU has ~0.4 m/s² noise floor; wind estimation (0.5 m/s² signal) may be unreliable. The no-mass-est policy avoids explicit reliance on noisy acceleration measurements.

3. **Thrust Curve**: Simulation assumes linear thrust; real propellers have nonlinear curves in ground effect.

4. **Reset Assumptions**: Training episodes start from randomized poses. Real-world deployment requires takeoff/landing handling not addressed in this work.

**Recommended Validation Steps:**
1. Verify inference latency on target hardware under load
2. Test mass estimation noise sensitivity before deployment
3. Validate motor delay assumptions against bench tests
4. Implement watchdog for attitude boundary violations

---

## Red-Team Pass

### Dimensional Consistency Check
| Claim | Status | Resolution |
|-------|--------|------------|
| 60-dim obs space | ✓ | 51 deployable + 9 privileged = 60 |
| 51-dim actor input | ✓ | mass_est removed from deployable |
| 52-dim (Section 3.5) | ✗ | Incorrect — code shows 51 |
| 63-dim (Section 13.5) | ✗ | Includes target velocity (3) — should be 54 deployable + 9 privileged = 63 for Stage 5 |

**Resolution**: Observation dim varies by stage (51-54 deployable + 9 privileged). Canonical table must reflect stage-dependent dimensions.

### Unit Consistency Check
| Unit | Status |
|------|--------|
| Action scale = [1.0, 1.0, 1.0, 1.0] N/Nm | ✓ Physical units match |
| Safety boundary in meters | ✓ |
| Tracking error in meters | ✓ |
| Attitude in radians (0.52 rad = 30°) | ✓ |
| Timestep 0.01s = 10ms | ✓ |

### Impossible Metrics Check
| Metric | Expected | Reported | Status |
|--------|----------|----------|--------|
| Stage 5 success 100% with 4.76m error | Impossible (boundary 1.5m) | Section 6.1 | ✗ Old training |
| Stage 5 success 100% with 0.095m error | Possible | Results file | ✓ |
| Stage 3 oracle 46% vs SAC 100% | Contradiction | Section 5.1 vs 10 | ✗ Oracle was open-loop |

### Unsupported Causal Claims
| Claim | Evidence | Status |
|-------|----------|--------|
| "Solved" precision-stability tradeoff | Single-seed, single condition | Downgrade to "addressed" |
| "Seamless" scaling to racing | Single-seed | Remove |
| "Extreme agility" | 5× speed tested | Remove or specify "up to 5×" |
| Transformer "may improve with scale" | No data beyond 166K samples | Remove or label speculative |

### Unfinished Work in Headlines
| Section | Claim | Status | Action |
|---------|-------|--------|--------|
| Abstract line 10-13 | "Extreme Extended Racing...76-93% success" | Incomplete | Delete |
| Abstract line 13-14 | "Yaw Control...Training in progress" | Incomplete | Delete |
| Conclusion bullet 8 | "Stage 7 Yaw Control" | Incomplete | Move to Future Work |
| Conclusion bullet 9 | "Stage 8 Extreme Extended Racing" | Incomplete | Move to Future Work |

---

## Summary

This manuscript requires significant restructuring before submission:

**Must Do:**
1. Resolve residual RL narrative contradiction (oracle was open-loop, SAC is closed-loop)
2. Remove Stage 7/8 from abstract and conclusions (move to appendix/future work)
3. Create canonical setup table with frozen dimensions
4. Add "single-seed" qualifiers where appropriate
5. Remove hype words: "solved", "breakthrough", "seamless", "extreme"
6. Clarify sim-to-real: no-mass-est is deployment path, noise-injection is future work

**Should Do:**
7. Move BC experiments to appendix
8. Separate Pi 5 and Desktop benchmarks clearly
9. Add statistical qualifiers (95% CI for success rates)
10. Standardize "47×" improvement claim

**Paper Structure Decision:**
This should be Option 2: "SAC-based residual quadrotor control with curriculum and sim-to-real considerations." The BC experiments support this as failure analysis (why imitation doesn't work) rather than a separate contribution.

The core contribution is the reward design (attitude cliff + torque penalties) enabling precision residual control, validated through curriculum progression. This is defensible with current data.
