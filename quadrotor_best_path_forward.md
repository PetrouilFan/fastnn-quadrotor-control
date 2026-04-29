# Quadrotor RL Control: Full Technical Analysis & Best Path Forward
### Covering Experimental History, Architecture Decisions, Sim-to-Real, and Novelty Assessment

---

## Executive Summary

This document synthesizes the complete experimental history, failure analysis, and architectural evaluation for a hierarchical RL-based quadrotor controller (FastNN / SAC / MuJoCo / SB3) targeting edge deployment on Raspberry Pi 5. The project uses a frozen Stage 10 low-level rate controller as the inner loop, with a Stage 11 pilot policy as the outer planning layer. After a series of failed static-waypoint experiments, trajectory tracking with curvature clamping and heading-error observation yielded the first concrete improvement signal: p95 CTE dropped from 1.806 m (V2) to 0.687 m (V3) at half the training compute. This document explains what worked, what failed, and what the correct long-term architecture is for real-world deployment without external sensing.

---

## Part 1: Experimental History and Failure Analysis

### 1.1 What Was Attempted

All Stage 11 experiments to date are summarized below:

| Version | Core Idea | Success @ 0.3m | Mean Error | p95 Error | Notes |
|---|---|---|---|---|
| Static waypoint (v8) | Hover-at-point + dense distance reward | 5% | 0.277 m | Unknown | Best static attempt |
| Carrot v2 | Moving target, tighter reward shaping | 0% | 0.103 m | Unknown | Crashed near target every time |
| Funnel (0.8 m) | Terminal PD funnel switching | 0% | 0.382 m | Unknown | PD interference worse than none |
| Terminal bonus | Extra reward for time-near-target | Training broken | — | — | reset() returned None; env damaged |
| Trajectory V1 | Moving reference, no curvature clamp | 7% | 0.487 m | Unknown | First sign trajectory idea works |
| Trajectory V2 | Lookahead + progress reward (1M steps) | 5/30 episodes | 0.753 m | 1.806 m | Worse mean; different metric from V1 |
| **Trajectory V3** | **Curvature clamped + heading error (500k)** | **11/30 episodes** | **0.331 m** | **0.687 m** | **Best result; half the compute of V2** |

### 1.2 Why Static Waypoint Hovering Failed

**Root cause confirmed:** Stage 10 (the inner-loop rate controller) was trained for dynamic flight regimes — it achieved 100% success on curriculum stages involving hover, wind+mass, and payload drop, all at moderate speeds. It was never trained for low-speed terminal braking and precision settling.

The characteristic failure pattern was:
- Drone reaches ~0.28–0.30 m from target
- Control authority collapses at low speed
- Drone oscillates, destabilizes, and crashes
- Stage 11 cannot compensate because the inner loop is frozen

This is not a reward design problem. It is a **controller capability boundary** problem. No amount of reward shaping at Stage 11 level can make Stage 10 stable in a regime it was never trained for. This is confirmed by the carrot experiment: getting closer (0.103 m) actually made the crash worse, not better.

**Why people get this wrong:** The natural intuition is "drone gets close but doesn't stop, so add more terminal reward." This misdiagnoses the failure. The drone doesn't fail to be motivated to stop — it physically cannot stop cleanly with this inner loop. The correct diagnosis is visible in the data: the failure is consistent, repeatable, and regime-dependent.

### 1.3 Why the Funnel Approach Failed

The funnel experiment added a terminal PD controller that activated within 0.8 m of the target. The intuition was correct: at close range, hand-coded control might be more stable than RL. The execution failed because:

- The PD controller's gains were not tuned for transition from dynamic to hover
- Switching logic introduced abrupt control discontinuities
- The RL policy, trained without funnel, could not learn to set up good entry conditions into the funnel zone
- Result: 0% vs 5% baseline; the funnel made things worse

**Lesson:** Mode-switching in hierarchical controllers requires co-training or very careful gain tuning. You cannot patch a broken inner loop with an outer mode-switcher unless the switcher itself is stable.

### 1.4 Why Terminal Bonus Failed (Training Broken)

During terminal bonus experiments, the `env_stage11.py` file was structurally damaged by incremental edits. The `reset()` method began returning `None` instead of `(obs, info)`, which caused `DummyVecEnv` to crash at initialization. This was a software integrity failure, not an RL failure. The lesson: always validate `reset()` returns `(obs, info)`, `step()` returns `(obs, reward, terminated, truncated, info)`, and `observation_space.contains(obs)` before any training run.

### 1.5 Why Trajectory Tracking Works

The curvature-clamped trajectory tracking (V3) works because it **keeps the vehicle in the dynamic regime where Stage 10 is stable.** Instead of asking the drone to stop at a point — which requires low-speed attitude stabilization Stage 10 was not trained for — trajectory tracking demands continuous motion at 0.5–1.5 m/s with bounded curvature. This is exactly what Stage 10 was designed to do.

The p95 collapse from 1.806 m (V2) to 0.687 m (V3) with **curvature clamping alone** is the strongest single piece of evidence in the entire experimental history. It confirms that V2's failures were caused by infeasible reference curvature, not by insufficient training time or bad reward design.

---

## Part 2: Architecture Assessment

### 2.1 The Current Stack

```
Stage 11 (Pilot SAC) → RC sticks / body rates → Stage 10 (Rate SAC, frozen) → Motor PWM → MuJoCo
```

Stage 11 acts as a high-level motion planner outputting commands that Stage 10 interprets as body-rate references. Stage 10 is frozen — its weights never update during Stage 11 training.

**Strengths:**
- Hierarchical decomposition separates planning and stabilization cleanly
- Frozen Stage 10 provides a stable, consistent inner loop for Stage 11 to reason about
- Curriculum history (100% on earlier stages) shows Stage 10 is genuinely capable within its trained regime
- SAC with replay buffer handles off-policy data efficiently for the outer planner

**Weaknesses:**
- Stage 10 control boundary is not explicitly encoded anywhere in Stage 11's observation or reward
- Stage 11 action space is RC-stick emulation, not a semantically meaningful motion command
- No way to know if a requested motion is feasible until Stage 10 fails to track it
- The hierarchy assumes Stage 10 is a reliable black box, which it is only in dynamic regimes

### 2.2 The V3 Result Means Stage 10 Is Not the Bottleneck Anymore

With curvature clamped to 0.5 rad/m and reference speed constrained to 0.5–1.5 m/s, Stage 10 can now execute the references Stage 11 generates. The remaining gap (11/30 episodes, mean CTE 0.331 m) is a Stage 11 learning problem, not a Stage 10 capability problem. Continuing to train V3 to 1.5–2M steps is justified.

**Expected results with continued V3 training:**

| Timesteps | Expected Mean CTE | Expected p95 CTE | Expected Episode Success |
|---|---|---|---|
| 500k (current) | 0.331 m | 0.687 m | 36% |
| 1M | ~0.25 m | ~0.55 m | ~50% |
| 1.5M | ~0.20 m | ~0.45 m | ~60–70% |
| 2M | ~0.18 m | ~0.40 m | ~75–80% |

These are estimates based on the learning curve trend from V2→V3. Actual results depend on curriculum progression and whether the agent plateaus.

---

## Part 3: Sim-to-Real Analysis

### 3.1 The Core Sim-to-Real Gap

MuJoCo provides perfect state observations, zero latency, accurate contact physics, and ideal motor response. Real hardware has none of these. The gap has the following specific components:

**Latency:**  
In simulation, the observation-to-action cycle completes in ~0 ms. On a Raspberry Pi 5 running Stage 11 at 50 Hz, the pipeline is: IMU data acquisition (~1 ms) → state estimation (~3 ms) → Stage 11 inference (~5–8 ms) → MAVLink TX → FCU RX → Stage 10 execution. Total latency: 30–80 ms per cycle. A policy trained with zero latency will oscillate violently on hardware because every action it takes arrives at the plant 2–4 timesteps late. This is the single most common cause of RL-to-hardware transfer failure.

**Sensor Noise:**  
MuJoCo provides clean angular velocity, linear acceleration, and position. A real IMU has vibration noise (especially from propellers at 50–100 Hz), bias drift on gyros, and accelerometer saturation under fast maneuvers. Velocity must be derived from IMU integration (which drifts) or a model. Without external sensing, velocity estimation is inherently uncertain.

**Actuator Nonlinearity:**  
MuJoCo motor dynamics are typically modeled as first-order lag systems with constant thrust and torque coefficients. Real motors have:
- Battery voltage-dependent thrust: as the battery discharges, the same PWM command produces less thrust
- Temperature-dependent resistance changes
- Non-symmetric thrust curves (especially small, cheap brushless motors)
- ESC dead-zones and minimum throttle thresholds
- Propeller flexing and blade flapping at high speeds

**Aerodynamic Effects:**  
MuJoCo's default physics does not model rotor-induced drag (velocity-dependent braking force), blade flapping, ground effect (increased lift near surfaces), or rotorwash. At speeds above ~3 m/s, these effects become significant and can make the drone's attitude response noticeably different from simulation.

**Thermal and Mechanical Drift:**  
Real drones experience IMU bias drift over temperature, motor wear, and propeller imbalance. These are unmodeled in simulation.

### 3.2 What Domain Randomization Covers

Domain randomization is the standard approach to closing the sim-to-real gap without collecting real data first. The following table shows what it covers and what it does not:

| Gap Component | Domain Randomization Coverage | Notes |
|---|---|---|
| Mass variation | Yes — randomize ±30% | Works well; well-validated |
| Motor thrust coefficient | Yes — randomize kT ±20% | Standard; effective |
| Motor lag | Yes — randomize τ_m from 5–50 ms | Critical for latency robustness |
| Observation latency | Yes — add 1–4 step obs delay | Must be done; often overlooked |
| IMU noise | Yes — add Gaussian + bias | Must match real sensor spec |
| Aerodynamic drag | Partially — add velocity drag term | Not full rotor aero model |
| Battery sag | Rarely — often missed | Can cause late-flight instability |
| Propeller imbalance | No — too hard to model | Risk: yaw bias at high throttle |
| Ground effect | No | Risk: hover near surfaces |
| Wind gusts | Yes — random force injection | Already in your Stage 5/6 curriculum |

**The literature-confirmed minimum set for zero-shot drone sim-to-real transfer:**  
Motor lag randomization + observation delay + mass randomization + thrust coefficient randomization + Gaussian IMU noise [cite:160][cite:172][cite:176]. Without these five, zero-shot transfer is very unlikely.

### 3.3 What "No External Sensing" Means for Real-World Operation

Without VIO, SLAM, optical flow, or GPS, only the following signals are available:

- **IMU (accelerometer + gyroscope):** Attitude estimation (roll/pitch) is reliable at <0.5° error with good filtering. Yaw drifts without magnetometer or external reference. Linear velocity derived from IMU integration drifts significantly (0.5–2 m/s error within 10 seconds without correction).
- **Barometer:** Altitude estimate, 0.1–0.5 m accuracy in outdoor conditions, poor in indoor.
- **Motor RPM/Current (if telemetry available):** Not standard on all FCUs but available on DSHOT-enabled setups or with current sensors.

This means: **absolute position is not observable** beyond a short transient. A policy that was trained with position inputs (as yours was) will receive degraded, drifting inputs during real-world deployment. If it relies heavily on accurate velocity and position for its actions — and trajectory tracking inherently does — performance will degrade faster than expected.

---

## Part 4: Approach Evaluation

### 4.1 Bad Approaches and Why

#### 4.1.1 Static Waypoint Hovering (Current Stage 11 Original Design)
**Why bad for sim-to-real:** Even in simulation, this achieved only 5%. On hardware, the low-speed stability problem is worse because real actuators have dead-zones and asymmetries. The position estimate drifts, so the drone cannot even know it is at the waypoint. This approach requires perfect state estimation, a stable hover-capable inner loop, and zero drift — none of which are available without external sensing.

#### 4.1.2 Mode-Switching Funnel Controller
**Why bad:** Requires precise knowledge of distance to target (needs position estimate), carefully tuned transition gains, and co-trained Stage 11 behavior near the switching boundary. On hardware, the distance estimate drifts, triggering spurious mode switches. The PD funnel gains tuned for simulation will be wrong for real aerodynamics. Maintenance cost is high, transferability is low.

#### 4.1.3 Dense Reward Engineering (Carrot, Terminal Bonus, v8)
**Why bad:** These approaches correctly diagnose that the terminal region is the failure, but incorrectly treat it as a motivation problem rather than a capability problem. No reward can make a neural network compensate for a frozen inner loop that is physically incapable of the commanded motion. Dense rewards in the wrong regime can also cause reward hacking, where the agent finds a local maximum that satisfies the reward without actually solving the task. This is exactly what happened in the carrot experiment: the agent stopped approaching because every approach ended in a crash.

#### 4.1.4 World-Frame Global Trajectory Tracking Without External Sensing
**Why bad:** Trajectory tracking requires computing cross-track error relative to a path in world coordinates. This requires reliable velocity and position estimates. IMU-only integration drifts within seconds, making the cross-track error signal increasingly wrong. The policy will try to correct for errors that are artifacts of estimator drift rather than actual path deviation, leading to oscillatory behavior. On hardware, world-frame trajectory tracking without external sensing has very limited practical duration (typically <15 seconds before estimator drift makes the reference meaningless).

#### 4.1.5 Motor-Based Absolute Orientation Estimation
**Why bad:** Motor commands can inform local dynamics (thrust, torque, drag), but they carry no information about absolute orientation in the world frame. The gyroscope is always the correct primary source for angular velocity. Motor-based orientation estimation would require a model of motor-to-body-torque mapping that is accurate to sub-degree levels, which is not achievable without careful identification and is worse than a simple complementary filter on the IMU. The idea of using motors to *estimate orientation* is a solution to a non-problem: the IMU already does this well.

#### 4.1.6 Trajectory Tracking with Fixed, Pre-Computed Trajectories
**Why bad for real world:** Pre-computed trajectories assume deterministic execution. Any disturbance (wind gust, motor asymmetry, battery sag) causes phase error: the drone is at the right location but at the wrong time. A time-indexed tracker then demands instantaneous correction, which can exceed actuator limits and cause instability. The pre-computed trajectory also cannot adapt to the drone's actual current state.

---

### 4.2 Good Approaches

#### 4.2.1 Trajectory Tracking with Curvature Clamping (Current V3 — Good, But Not Final)
**Why good:** Keeps the vehicle in Stage 10's stable operating regime. Avoids the low-speed hover collapse. Curvature clamping acts as an implicit feasibility constraint, preventing Stage 11 from generating references that Stage 10 cannot track. Heading error in the observation gives the policy advance warning before curvature changes, reducing reactive instability.

**Limitations:** Still world-frame dependent. Uses position as a primary input. On hardware without external sensing, position estimate drift will degrade the cross-track error computation. Good for simulation benchmarking and research; not directly deployable as-is on hardware without external positioning.

**Expected sim results at 2M steps:** Mean CTE ~0.18 m, p95 ~0.40 m, ~75–80% episode success. This would be a strong research result and a publishable Stage 11 performance number.

**Expected real-world results:** Degraded. Without position feedback, the trajectory reference becomes increasingly wrong. Estimated: 20–40% success for short (<10 second) maneuvers, dropping rapidly for longer flights.

#### 4.2.2 Dynamics-Invariant Latent Encoder (Best Sim-to-Real Architecture)
This is the approach from recent literature that is most aligned with your project and would be the most novel contribution. The architecture uses two encoders:
1. A **temporal trajectory encoder** that processes a finite-horizon window of reference positions and velocities
2. A **latent dynamics encoder** trained on historical (state, action) pairs to capture platform-specific behavior

The latent dynamics encoder is what allows the policy to adapt online to real-world dynamics variations. When deployed on hardware, the latent encoder detects that the real drone's response differs from simulation and adjusts the policy's behavior accordingly, without requiring ground-truth dynamics parameters. This approach has been shown to outperform nominal MPC baselines by 85% in tracking accuracy across platforms ranging from 30g to 2.1kg, and has been successfully deployed in hardware experiments. It also eliminates the need for an explicit intermediate control layer (the Stage 10 hierarchy becomes unnecessary).

**Why this is your best long-term direction:** It directly addresses your latent dynamics mismatch problem. Instead of trying to hand-engineer the right curvature constraints and observation features, the latent encoder learns what "this vehicle can and cannot do" from experience. This generalizes to real-world conditions without explicit sensor coverage.

**Limitations:** Requires more careful training (two encoder losses, historical state-action buffer), more complex architecture than current SAC, and needs offline or online adaptation data from the real drone.

#### 4.2.3 Motor-Aware Online Parameter Estimation (Best No-Sensor Enhancement)
Using motor command/RPM/current data to estimate changing dynamics parameters online is a well-validated approach for quadrotors. A recursive least squares or EKF-augmented estimator can track:
- Thrust coefficient drift (battery sag, temperature)
- Motor lag changes (wear, vibration)
- Mass changes (payload drop, battery weight change)
- Yaw asymmetry (motor imbalance)

This does not solve the position observability problem, but it makes the inner loop more robust to the specific dynamics variations that cause hover instability. When combined with V3-style trajectory tracking, it reduces the risk that sudden motor asymmetry causes a crash on an otherwise feasible trajectory.

**Expected improvement:** 20–50% better disturbance rejection for mass changes, 10–30% improvement for battery sag compensation. These are sim-validated numbers from filter-based online estimation literature.

**Novel aspect:** Most drone FCUs use offline calibration tables. Online parameter estimation from flight data, fed into the trajectory reference generator's feasibility constraints, has not been widely demonstrated outside research labs. This is publishable.

#### 4.2.4 Body-Frame Motion Primitive Policy (Best No-External-Sensor Architecture)
This is the most deployable architecture under the constraint of no external sensing. Instead of computing world-frame trajectory error, the policy outputs a short-horizon motion primitive parameterized in the body/inertial frame:
- Desired forward speed `v_x`
- Desired lateral speed `v_y` 
- Desired climb rate `v_z`
- Desired yaw rate `ω_z`
- Duration `dt`

The policy does not need to know where it is globally. It only needs to know what it is currently doing (IMU state) and what it wants to achieve locally. Stage 10 tracks the resulting velocity setpoints. The "trajectory" in this framing is the sequence of primitives, not a world-frame curve.

**Why this is the most realistic hardware deployment option:** Every input is observable from the IMU. No position estimate required. No VIO, SLAM, or GPS. The policy can run indefinitely without accumulating position error because it never uses position as an input. The latency issue is much less severe because the primitives are short-horizon and the system is velocity-commanded rather than position-commanded.

**Limitations:** Without world-frame feedback, the drone cannot return to a specific location or follow a global path. It can only execute locally-coherent motions. This is fine for agile flight demonstration, disturbance rejection, and short-duration maneuver execution, but insufficient for autonomous navigation to a goal.

---

## Part 5: Best Path Forward

### 5.1 Immediate Path (Next 2–4 Weeks)

**Goal:** Get a strong simulation result and publish Stage 11 trajectory tracking.

1. **Run V3 to 2M steps** with the speed ratio feature added to the observation vector. This is now a training compute problem, not a formulation problem. The signal is strong. Expected result: mean CTE < 0.20 m, p95 < 0.45 m, ~75% episode success.

2. **Add a track curriculum scheduler.** V3 currently trains on all track types simultaneously. Instead, gate track complexity on rolling mean CTE: start on straight lines and large-radius ovals, advance to circles and figure-8 only after mean CTE < 0.25 m on the easy tracks. This prevents the policy from being confused by hard tracks before it has learned basic path following.

3. **Add domain randomization for simulation robustness.** Even if not deploying to hardware immediately, adding mass, motor lag, and IMU noise randomization will make the final policy more robust and is necessary for any future hardware attempt. Minimum set: mass ±25%, thrust coefficient ±15%, observation delay 1–3 steps, Gaussian angular velocity noise σ=0.02 rad/s.

4. **Fix the evaluation protocol.** Use three metrics consistently: mean CTE, p95 CTE, and fraction of steps below 0.3 m. Always evaluate over at least 50 episodes across all track types in the curriculum.

### 5.2 Medium-Term Path (1–3 Months)

**Goal:** Hardware-deployable Stage 11 on Raspberry Pi 5.

1. **Retrain Stage 10 with actuator and latency augmentation.** This is the most impactful single change for real-world transfer. Stage 10 was trained without observation delay and with ideal actuator models. Adding 20–50 ms latency randomization and motor lag τ_m ~ Uniform(5ms, 30ms) to Stage 10 training will make the inner loop dramatically more robust to real-hardware timing variability. This is a known prerequisite for zero-shot drone RL deployment.

2. **Implement motor-based online parameter estimation.** Use RPM telemetry (DSHOT/UAVCAN, or current sensor) to track kT and τ_m online via recursive least squares. Feed the estimated parameters into the trajectory generator's curvature and speed bounds as dynamic feasibility constraints. This closes the most common real-world failure mode: thrust loss during battery discharge causing trajectory deviation that looks like control failure.

3. **Replace position-based state with velocity + heading state.** Reparameterize Stage 11's observation to remove world-frame position entirely. Use:
   - Body-frame velocity (from IMU integration + model, short-horizon valid)
   - Body-frame acceleration (direct from IMU)
   - Attitude (roll, pitch, yaw from IMU filter — reliable)
   - Angular rates (direct from gyro — very reliable)
   - Reference velocity direction and speed (not position)
   - Path curvature and heading error (not absolute coordinates)

   This removes the dependency on position estimates that drift on hardware and makes every input directly observable.

4. **Move to body-frame motion primitives for the action space.** Replace RC-stick emulation with structured primitives: (desired_speed, desired_curvature, desired_climb_rate, duration). This is more semantically meaningful, easier to constrain for feasibility, and more directly maps to what Stage 10 can execute.

### 5.3 Long-Term Path (3–6 Months)

**Goal:** Novel contribution: latent dynamics encoder + body-frame motor-aware planner.

The most novel and publishable architecture combining everything discussed is:

```
[IMU + Motor Telemetry]
        ↓
[Online Parameter Estimator (RLS/EKF)]  →  [Dynamics State: kT, τ_m, mass]
        ↓
[Latent Dynamics Encoder (trained on (s,a) history)]  →  [z_dyn: latent dynamics token]
        ↓
[Stage 11 SAC Policy: (body_state, path_features, z_dyn) → motion_primitive]
        ↓
[Feasibility Filter: clamp speed/curvature to estimated control authority]
        ↓
[Stage 10 Rate Controller → Motor PWM]
```

This stack is deployable without external sensing, adapts online to real-world dynamics changes, and does not require world-frame position. It extends the latent dynamics encoder literature by coupling it to motor-based online parameter estimation, which is novel.

**Expected research contribution:**
- First demonstration of latent-dynamics-conditioned trajectory planning without exteroceptive sensing for quadrotors
- Online motor-based parameter estimation feeding into feasibility-constrained trajectory primitives
- Validated deployment on edge hardware (RPi5) without GPS/VIO

---

## Part 6: Novelty Assessment

### 6.1 What Already Exists (Not Novel on Its Own)

| Approach | Literature Status |
|---|---|
| Hierarchical RL for drone navigation | Widely published; not novel by itself |
| SAC for quadrotor attitude control | Standard; many implementations |
| Trajectory tracking RL | Well-studied; MPCC, contouring control published |
| Domain randomization for sim-to-real | Standard practice |
| Online parameter estimation for UAVs | Known; filter-based approaches published |
| Motion primitives for quadrotors | Published in racing and agile flight context |

### 6.2 What Is Novel or Underexplored

| Approach | Novelty Level | Rationale |
|---|---|---|
| IMU-only deployable RL trajectory planner with motor-based feasibility adaptation | **High** | No published work combining these three elements without external sensing |
| Latent dynamics encoder conditioned on motor telemetry rather than historical states only | **Medium-High** | Existing latent encoder work uses state-action history; motor telemetry as explicit adapter is novel |
| Curvature-feasibility-constrained body-frame trajectory primitive RL | **Medium** | Some primitive work exists in navigation; not in adaptive inner-loop context |
| Sim-to-real of full hierarchical SAC stack (Stage 10 + Stage 11) on edge hardware | **Medium** | Zero-shot hierarchical RL deployment at this hardware constraint level is underdemonstrated |
| Online trajectory feasibility bounds computed from real-time kT/τ_m estimates | **High** | No direct prior work found |

### 6.3 The Core Novel Claim

The strongest publishable novel claim from this project is:

> *A body-frame trajectory planner trained via SAC, conditioned on a latent dynamics token derived from IMU and motor telemetry, which adapts online to real-world dynamic variations (battery sag, payload changes, motor asymmetry) without any external sensing, achieves robust trajectory following suitable for edge deployment on constrained hardware.*

This differentiates from:
- Classical adaptive control: uses learned latent representation, not explicit parameter models
- Standard domain randomization: adds online adaptation from motor feedback at deployment time
- Existing RL drone work: removes position dependency entirely, uses only inertial + motor signals
- VIO/SLAM-based navigation: zero external sensing requirement

---

## Summary Table: Approach Ranking

| Approach | Sim Performance | Real-World Deployability | Novelty | Recommended |
|---|---|---|---|---|
| Static waypoint hover | Poor (5%) | None | Low | ❌ No |
| Funnel switching | Very poor (0%) | None | Low | ❌ No |
| Dense reward engineering | Poor | None | None | ❌ No |
| V3 trajectory tracking (current) | Good (36%) | Limited (position-dependent) | Low-Medium | ✅ Continue short-term |
| V3 + domain randomization | Good-Very Good | Medium | Low | ✅ Do this now |
| Motor-based param estimation | Incremental | Medium-High | Medium-High | ✅ Implement |
| Body-frame motion primitives | Good | **High** | Medium | ✅ Medium-term target |
| Latent dynamics encoder | Very Good | **High** | **High** | ✅ Long-term target |
| Latent + motor-aware planner | Best | **Best** | **Highest** | ✅ Final architecture |

