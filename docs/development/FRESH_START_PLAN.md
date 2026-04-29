# Fresh Start: Complete Validation Plan

**Created:** 2026-04-28  
**Purpose:** Start fresh branch with validated experiments only

---

## Phase 0: Branch Setup

### Step 0.1: Create New Branch
```bash
git checkout -b fresh-start-validated master
```

### Step 0.2: Archive Old Files
```bash
mkdir -p docs/archived/2026_archive
mv archive/ docs/archived/2026_archive/
# Keep: env_rma.py, quadrotor/, train_*.py (working infrastructure)
```

---

## Phase 1: Training Infrastructure

### Scripts to Create

#### 1. train_gru_delay.py
Based on `train_gru_sac.py` with:
- Delay injection in environment wrapper
- 50ms delay (fixed for training)
- GRU policy with history buffer (4 past observations)

#### 2. eval_robustness.py
Standardized evaluation with:
- Delay sweep: 0, 25, 50, 100ms
- Noise sweep: 0x, 0.5x, 1x baseline
- Mass variation: 0.6, 0.8, 1.0, 1.2, 1.4
- Wind: 0, ±0.5, ±1.0, ±2.0 N

---

## Phase 2: Execution Order

### Step 1: Baseline (E01)
- Use existing Stage 3-5 checkpoints
- Run delay sweep for baseline metrics
- **Expected:** ~80% at 50ms delay (MLP degrades)

### Step 2: GRU Delay Training (E02)
```bash
python train_gru_delay.py \
  --steps 500000 \
  --n-envs 32 \
  --delay-ms 50 \
  --seed 0
```
- Output: runs/gru_delay_50ms/
- Expected: 2-3 hours on 32 envs

### Step 3: GRU Evaluation
- Run same delay sweep as baseline
- Compare success rates

### Step 4: Continue Based on Results
| Result | Action |
|--------|--------|
| GRU ≥90% at 50ms | Proceed to TCN (E07) |
| GRU <90% at 50ms | Skip temporal, go to adaptation (E03) |

---

## Complete Testing Order (All Experiments)

### Phase 1: Temporal Controllers
| Exp | Architecture | Training | Expected Benefit |
|-----|--------------|----------|------------------|
| E01 | MLP Baseline | Existing | N/A (baseline) |
| E02 | GRU Temporal | New 500K | Delay robustness |
| E07 | TCN | New 500K | Alternative to GRU |

### Phase 2: Adaptation
| Exp | Architecture | Purpose |
|-----|--------------|----------|
| E03 | Latent Encoder | Mass inference |
| E05 | Feasibility Shield | Safety |

### Phase 3: Hierarchical
| Exp | Architecture | Purpose |
|-----|--------------|----------|
| E04 | Body-frame Pilot | Separate pilot/controller |
| E10 | Multi-timescale | Different update rates |

### Phase 4: Integration
| Exp | Components | Purpose |
|-----|------------|----------|
| E06 | GRU + Adapter + Shield | Combined best |

### Phase 5: Hardware
| Test | Target | Metric |
|------|--------|--------|
| Latency | Raspberry Pi 5 | <1ms inference |
| Model size | All | <10MB |

---

## Test Variables

| Variable | Levels | Rationale |
|----------|--------|----------|
| Delay | 0, 25, 50, 100 ms | Common sensor-actuator delays |
| Sensor noise | 0x, 0.5x, 1x baseline | Real IMU noise levels |
| Mass ratio | 0.6, 0.8, 1.0, 1.2, 1.4 | Typical payload variations |
| Wind | 0, ±0.5, ±1.0, ±2.0 N | Increasing disturbance |
| Motor lag | 0, 50, 100 ms | Actuator delay |

---

## Evaluation Metrics

### Primary
- Success rate (% episodes reaching 500 steps)
- Mean tracking error (m)
- P95 tracking error (m)

### Robustness
- Success rate under delay
- Success rate under noise
- Success rate under mass variation
- Success rate under wind

### Safety
- Crash rate (policy-induced failures)
- Max attitude at crash

### Deployment
- Inference latency (ms)
- Model size (MB)

---

## Promotion Gates

| Phase | Gate | Criteria |
|-------|------|----------|
| E02 GRU | Phase 1 Pass | Success rate ≥90% at 50ms delay (vs baseline <80%) |
| E03 Adapter | Phase 1 Pass | Mass prediction correlation >0.7 |
| E05 Shield | Phase 1 Pass | Crash rate <5% under extreme conditions |
| Combined | Phase 2 Pass | Robustness >baseline AND capability ≥95% |

---

## Results Template

### Delay Sweep Results
| Delay | MLP Success | GRU Success | Delta |
|-------|------------|-------------|-------|
| 0ms   |            |             |       |
| 25ms  |            |             |       |
| 50ms  |            |             |       |
| 100ms |            |             |       |

### Noise Sweep Results
| Noise Level | MLP Success | GRU Success |
|------------|-------------|-------------|
| 0x         |             |             |
| 0.5x       |             |             |
| 1x         |             |             |

### Mass Variation Results
| Mass Ratio | MLP Success | GRU Success |
|------------|-------------|-------------|
| 0.6        |             |             |
| 0.8        |             |             |
| 1.0        |             |             |
| 1.2        |             |             |
| 1.4        |             |             |

---

## Execution Log

| Date | Step | Action | Result |
|------|------|--------|--------|
| 2026-04-28 | Branch | Created fresh-start-validated branch | ✓ |
| 2026-04-28 | Archive | Moved archive/ to docs/archived/ | ✓ |
| 2026-04-28 | Train 2M | Stage 5 baseline | **99% success** |
| 2026-04-28 | Test delay | 0ms=36%, 25ms=22%, 50ms=0% | Baseline dies at delay |
| 2026-04-28 | Train delay 50ms | 1M with 50ms delay | 0% (too hard) |

---

## Key Findings

### What Works
- **Clean training**: 2M steps → 99% success on moving target
- Checkpoint: `models_stage5_curriculum/stage_5/seed_0/final.zip`

### What Doesn't Work
- **Immediate delay training**: 50ms delay during training makes it too hard to learn
- Adding 0.1 noise during training → 0% (too aggressive)

### The Real Problem
- MLP doesn't have memory - it only sees current observation
- Delay = missing information = impossible to recover without history

---

## Solutions (Future Work)

1. **GRU architecture**: Encodes observation history in hidden state
2. **Curriculum delay**: Start with 0ms → gradually increase to 50ms over training
3. **Smaller noise first**: 0.01-0.05 instead of 0.1

---

## Notes

- Always log metrics to JSON/CSV from start
- Save checkpoints every 100K steps
- Document results as you go (no retroactively)
- Only write paper AFTER training completes with evidence