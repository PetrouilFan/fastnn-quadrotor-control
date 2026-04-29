# Quadrotor RL Control: Complete Research Summary
## All experiments, lessons learned, and definitive best path forward

---

## Executive Summary

After extensive experimentation (10+ versions tested), the trajectory tracking formulation with curvature clamping (V3) is the best performing. Key findings:

- **Best checkpoint: V3 (500k)** - mean CTE 0.331m, p95 0.687m
- **V6 best checkpoint: 700k** - mean 0.322m, p95 0.694m (ties V3)
- Training is non-monotonic - degrades after ~700k
- Problem is optimization stability, not architecture

---

## Complete Results Table

| Version | Steps | Mean CTE | p95 CTE | Steps <0.3m | Episodes Success |
|---------|-------|---------|---------|------------|-----------------|
| v8 (static) | 500k | 0.277m | — | — | 5% |
| Carrot v2 | 500k | 0.103m | — | — | 0% |
| Trajectory V1 | 500k | 0.487m | — | — | 7% |
| Trajectory V2 | 1M | 0.753m | 1.806m | 27.6% | 5/30 |
| **Trajectory V3** | **500k** | **0.331m** | **0.687m** | **48.1%** | **11/30** |
| Trajectory V4 | 2M | 0.533m | 1.654m | 38.5% | 8/30 |
| V6 final (1.5M) | 1.5M | 0.431m | 1.366m | 45.6% | 14/30 |
| **V6-700k checkpoint** | **700k** | **0.322m** | **0.694m** | **50.5%** | **~15/30** |

---

## Key Discoveries

### 1. Inner Loop Capability Boundary
Stage 10 was trained for dynamic flight, NOT low-speed precision hover. This is why static waypoint hovering failed (5% max).

### 2. Curvature Clamping Works
p95 dropped from 1.806m (V2) to 0.687m (V3) with curvature clamping alone. Reference feasibility is primarily about curvature.

### 3. Training is Non-Monotonic
V6 sweep showed:
- 300k: poor (p95=1.699m)
- 500k: moderate (p95=0.779m)
- **700k: best (p95=0.694m)** ← ties V3
- 900k: collapse (p95=1.569m)
- 1100k: collapse (p95=1.775m)
- 1200k: partial recovery (p95=0.784m)
- 1500m: moderate (p95=1.366m)

**Critical insight:** Longer training degrades after ~700k. Problem is optimization stability, not formulation.

### 4. Replay Buffer Contamination
V4 regression caused by premature curriculum exposure, filling replay buffer with high-error experiences.

### 5. Best-Model Selection Mismatch
SB3's best_model.zip is selected on eval reward, NOT on CTE metrics. True best is found by sweeping checkpoints.

---

## Current Best Architecture

### V3 Formulation (Recommended)
```
- Curvature clamping: radius >= 1.2m, velocity 0.8-1.2 m/s
- Heading error in observation
- Speed ratio in observation
- Simple curriculum ["line", "oval", "circle", "figure8"]
- 500k steps, evaluate at 500k
```

### V6-700k (Alternative)
Same as V3 but with:
- Gated curriculum (rolling 200 episode CTE window)
- Advance threshold: mean CTE < 0.25m AND p95 < 0.8m
- Stop at 700k or first p95 degradation
- Metric-based checkpoint selection (p95 primary)

---

## What Does NOT Work

| Approach | Reason Failed |
|----------|---------------|
| Static waypoint | Stage 10 not trained for low-speed hover |
| PD funnel | Mode switching requires co-training |
| Dense reward | Motivation ≠ capability |
| Long continuous training | Non-monotonic, degrades after ~700k |
| Reward-based checkpoint selection | Diverges from CTE metrics |

---

## Definitive Best Path Forward

### Immediate (Do Now)

1. **Use V3 (500k) as baseline** - proven stable
2. **Add dual-threshold gating:**
   - rolling mean CTE < 0.25m AND
   - rolling p95 CTE < 0.8m
3. **Stop at first p95 degradation** (around 700k)
4. **Select checkpoint by p95**, not reward

### To Evaluate (Track Bucket)
Split evaluation by track type:
- Easy: line, oval
- Medium: large_circle, small_circle  
- Hard: figure8, spline

This reveals if collapse is track-specific.

### Medium-Term
- Body-frame motion primitives (no position dependency)
- Latent dynamics encoder (adapts to real hardware)
- Model-based trajectory generation

---

## Technical Lessons

1. **Validate env methods before training** - reset() returning None crashed multiple runs
2. **Log entropy coefficient** - low α (< 0.05) signals over-commitment
3. **Checkpoint frequently** - enables offline sweep
4. **Smaller training runs > one long run** - prevents degradation
5. **Three-metric evaluation:** mean CTE, p95 CTE, fraction < 0.3m

---

## Research Paper Update Summary

The complete paper now includes:
- Full experimental history (10+ versions)
- Root cause analysis (inner loop capability)
- Trajectory breakthrough (curvature clamping)
- Non-monotonic training finding
- Checkpoint sweep methodology
- Definitive best path forward with dual-threshold gating

**Key insight:** The formulation (V3) is correct. The remaining problem is training protocol optimization, not architecture.

---

*Last updated: Complete sweep conducted, V3 + V6-700k identified as co-champions*
*Next: Metric-based checkpoint selection + tighter dual-threshold gating*