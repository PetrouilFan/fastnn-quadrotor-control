# Deployment Guide

This guide covers deploying trained models to hardware, real-time inference optimization, and sim-to-real considerations for the FastNN Quadrotor Control framework.

## Hardware Requirements

### Supported Platforms

| Platform | CPU | RAM | Storage | Performance |
|----------|-----|-----|---------|-------------|
| **Raspberry Pi 5** | Cortex-A76 (4 cores) | 4GB/8GB | MicroSD | Primary target |
| **NVIDIA Jetson** | Xavier NX/Orin | 8GB+ | NVMe | High-performance |
| **x86 Linux** | i5/i7 equivalent | 4GB+ | SSD | Development/testing |
| **ARM Linux** | RK3588, etc. | 4GB+ | eMMC | Embedded systems |

### Sensor Requirements

**Core Sensors** (51-dim deployable):
- **IMU**: 3-axis accelerometer, 3-axis gyroscope (body-frame)
- **Position**: GPS or VIO (NED frame)
- **Attitude**: IMU fusion or external estimate
- **Motor Feedback**: Thrust estimates from ESC telemetry

**Optional Sensors**:
- **Barometer**: Altitude hold
- **Magnetometer**: Yaw reference
- **Camera**: Vision-based positioning

## Model Export and Optimization

### Export to ONNX

```python
# Export PyTorch model to ONNX
import torch
from stable_baselines3 import SAC

# Load trained model
model = SAC.load("models_stage5_curriculum/stage_5/seed_0/final.zip")

# Create sample input (51-dim deployable observations)
sample_obs = torch.randn(1, 51)

# Export to ONNX
torch.onnx.export(
    model.policy,
    (sample_obs,),
    "model.onnx",
    input_names=['obs'],
    output_names=['action'],
    dynamic_axes={'obs': {0: 'batch'}},
    opset_version=11
)
```

### FastNN Optimization

```bash
# Convert ONNX to FastNN format
./fastnn_converter model.onnx model.fastnn

# Optimize for target platform
./fastnn_optimizer model.fastnn \
  --target raspberry_pi_5 \
  --quantize int8 \
  --optimize latency
```

### Performance Comparison

| Runtime | Latency (median) | Throughput | Memory |
|---------|------------------|------------|--------|
| PyTorch CPU | 312 μs | 3,205 inf/s | 500 MB |
| **FastNN** | **114 μs** | **8,751 inf/s** | **50 MB** |
| FastNN + INT8 | 89 μs | 11,236 inf/s | 25 MB |

## Real-Time Control Loop

### Control Architecture

```
100 Hz Control Loop:
├── Sensor Reading (5ms)
├── State Estimation (10ms)
├── Policy Inference (1ms)
├── PD Controller (5ms)
├── Motor Commands (5ms)
└── Actuator Update (5ms)
```

### Implementation Example

```python
import time
from fastnn_inference import FastNNModel

class QuadrotorController:
    def __init__(self, model_path):
        # Load optimized model
        self.model = FastNNModel(model_path)

        # PD controller gains
        self.kp_pos = np.array([2.0, 2.0, 1.0])
        self.kd_pos = np.array([1.0, 1.0, 0.5])
        self.kp_att = np.array([5.0, 5.0, 2.0])
        self.kd_att = np.array([1.0, 1.0, 0.5])

    def control_loop(self):
        target_freq = 100  # Hz
        dt = 1.0 / target_freq

        while True:
            start_time = time.time()

            # 1. Read sensors
            obs = self.read_sensors()

            # 2. Policy inference
            action = self.model.infer(obs)

            # 3. PD controller
            pd_cmd = self.pd_controller(obs)

            # 4. Residual addition
            total_cmd = pd_cmd + action * self.action_scale

            # 5. Send to motors
            self.send_motors(total_cmd)

            # 6. Maintain frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, dt - elapsed)
            time.sleep(sleep_time)
```

## Sensor Integration

### IMU Processing

```python
def process_imu(self, accel, gyro):
    """Process IMU data for observation space"""

    # Body-frame acceleration (gravity-removed)
    accel_body = accel - np.array([0, 0, -9.81])

    # Integrate for velocity (simple, use proper estimator in production)
    self.velocity += accel_body * self.dt

    # Observation components
    obs = np.concatenate([
        accel_body,           # Linear acceleration (3)
        gyro,                 # Angular rates (3)
        self.rotation_matrix, # SO(3) matrix (9)
        # ... other components
    ])

    return obs
```

### Position Integration

```python
def process_position(self, gps_pos, gps_vel=None):
    """Process position data"""

    # Position error (target - current)
    pos_error = self.target_pos - gps_pos

    # Velocity error
    if gps_vel is not None:
        vel_error = self.target_vel - gps_vel
    else:
        vel_error = -self.velocity  # From IMU integration

    return pos_error, vel_error
```

### Motor Telemetry

```python
def process_motor_feedback(self, motor_rpms, motor_currents):
    """Extract motor thrust estimates"""

    # Simplified thrust estimation
    # In practice, use motor model: thrust = k * rpm^2
    thrust_estimates = []
    for rpm in motor_rpms:
        thrust = self.thrust_coeff * (rpm / self.max_rpm)**2
        thrust_estimates.append(thrust)

    return np.array(thrust_estimates)
```

## Safety and Monitoring

### Attitude Protection

```python
def safety_check(self, obs):
    """Critical safety monitoring"""

    roll, pitch, yaw = obs[6:9]  # Attitude error

    # Attitude crash detection
    max_attitude = np.pi/2  # 90 degrees
    if abs(roll) > max_attitude or abs(pitch) > max_attitude:
        self.emergency_stop("Attitude violation")
        return False

    # Position boundary check
    position = obs[0:3]
    if np.linalg.norm(position) > self.safety_boundary:
        self.emergency_stop("Boundary violation")
        return False

    return True
```

### Watchdog Timer

```python
class Watchdog:
    def __init__(self, timeout_ms=100):
        self.timeout = timeout_ms / 1000.0
        self.last_feed = time.time()

    def feed(self):
        self.last_feed = time.time()

    def check(self):
        if time.time() - self.last_feed > self.timeout:
            # Control loop stalled - emergency stop
            emergency_stop("Control loop timeout")
            return False
        return True
```

### Failsafe Modes

```python
def emergency_stop(self, reason):
    """Multiple levels of failsafe"""

    print(f"EMERGENCY STOP: {reason}")

    # Level 1: Stop residual control (pure PD)
    self.use_residual = False

    # Level 2: Hover in place
    self.target_pos = self.current_pos
    self.target_vel = np.zeros(3)

    # Level 3: Land immediately
    if self.critical_failure:
        self.land_now()
```

## Sim-to-Real Transfer

### Known Gaps

| Gap | Simulation | Reality | Mitigation |
|-----|------------|---------|------------|
| **Motor Dynamics** | 57ms lag | Variable | Bench test motors |
| **Sensor Noise** | Ideal | IMU bias/drift | Kalman filtering |
| **Thrust Curve** | Linear | Nonlinear | Motor characterization |
| **Aerodynamics** | Simplified | Complex | Wind tunnel testing |
| **Vibration** | None | High-frequency | Sensor filtering |

### Domain Randomization Validation

```python
# Test trained policy against real-world variations
test_conditions = [
    {'mass': 1.2, 'wind': 0.5, 'motor_eff': 0.9},
    {'mass': 0.8, 'wind': -0.3, 'motor_eff': 1.1},
    # ... more conditions
]

for condition in test_conditions:
    # Simulate real-world variation
    env.set_mass(condition['mass'])
    env.set_wind(condition['wind'])
    # Evaluate policy performance
```

### Hardware Validation Steps

1. **Bench Testing**:
   - Motor characterization (thrust curves)
   - Sensor calibration (IMU bias)
   - Control loop timing verification

2. **Tethered Flight**:
   - Secure vehicle during initial tests
   - Gradual authority increase (PD → residual)
   - Emergency stop verification

3. **Free Flight**:
   - Low-speed hover first
   - Progressive complexity increase
   - Telemetry monitoring throughout

## Performance Optimization

### FastNN Tuning

```bash
# Profile model performance
./fastnn_profiler model.fastnn \
  --platform raspberry_pi_5 \
  --batch-size 1 \
  --iterations 1000

# Optimize for latency
./fastnn_optimizer model.fastnn \
  --optimize latency \
  --threads 4 \
  --cache-line 64
```

### Memory Optimization

```python
# Pre-allocate buffers
self.obs_buffer = np.zeros(51, dtype=np.float32)
self.action_buffer = np.zeros(4, dtype=np.float32)

# Reuse tensors
self.input_tensor = torch.from_numpy(self.obs_buffer)
self.output_tensor = torch.empty(4, dtype=torch.float32)
```

### CPU Affinity

```python
# Pin control loop to specific CPU core
import os
os.sched_setaffinity(0, {3})  # Use CPU core 3

# Set real-time priority
import sched
sched.setscheduler(0, sched.SCHED_FIFO, sched.sched_param(50))
```

## Monitoring and Debugging

### Telemetry Collection

```python
def collect_telemetry(self):
    """Real-time performance monitoring"""

    telemetry = {
        'timestamp': time.time(),
        'position_error': np.linalg.norm(self.obs[0:3]),
        'attitude_error': np.linalg.norm(self.obs[6:9]),
        'residual_action': self.action,
        'pd_command': self.pd_cmd,
        'motor_commands': self.motor_cmds,
        'inference_time': self.inference_time,
        'loop_time': self.loop_time
    }

    # Log to file/SD card
    self.logger.log(telemetry)

    return telemetry
```

### Real-Time Visualization

```python
# Stream telemetry to ground station
import socket

def send_telemetry(self, telemetry):
    try:
        self.telemetry_socket.sendto(
            json.dumps(telemetry).encode(),
            (self.ground_station_ip, self.telemetry_port)
        )
    except:
        pass  # Don't fail control loop
```

## Certification Considerations

### DO-178C Compliance (Aerospace)

**Software Level**: Level C (minor injury possible)

**Required Artifacts**:
- Requirements traceability
- Code reviews and testing
- Structural coverage analysis
- Verification and validation

### Safety-Critical Features

- **Deterministic execution**: No dynamic memory allocation
- **Timing guarantees**: Worst-case execution time bounds
- **Error handling**: Comprehensive exception management
- **Redundancy**: Multiple failsafe levels

## Example Deployment Scripts

### Raspberry Pi Setup

```bash
# Install dependencies
sudo apt update
sudo apt install python3-pip libatlas-base-dev
pip3 install fastnn-runtime numpy

# Configure real-time kernel
sudo apt install linux-image-rt-arm64
# Add to /boot/firmware/config.txt:
# kernel=linux-image-rt-arm64

# Set CPU governor to performance
sudo cpupower frequency-set -g performance
```

### Startup Script

```bash
#!/bin/bash
# /home/pi/start_quadrotor.sh

cd /home/pi/fastnn_quadrotor
python3 deploy_raspberry_pi.py \
  --model model.fastnn \
  --config pi_config.yaml \
  --log-dir /home/pi/logs
```

This deployment guide provides a foundation for hardware implementation. Always start with simulation validation and progress gradually to real hardware testing.