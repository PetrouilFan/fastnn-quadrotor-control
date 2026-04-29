#!/usr/bin/env python3
"""
RMA (Rapid Motor Adaptation) Environment for Quadrotor
With curriculum learning support and residual RL on top of cascaded PD controller.

Key features:
- Curriculum stages 1-4
- Residual RL action on top of cascaded PD base controller
- 60-dim observation (52 deployable + 8 privileged)
- IMU-based mass estimation (body-frame gravity projection)
- Motor delay simulation (first-order lag)
- Direct control mode for standalone PD evaluation

Asymmetric Actor-Critic:
- Actor receives ONLY 52 deployable obs (indices 0-51)
- Critic receives full 60-dim obs (deployable + privileged)
- This ensures the learned policy is deployable on real hardware
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mujoco import MjModel, MjData, mj_step


class RMAQuadrotorEnv(gym.Env):
    """
    MuJoCo quadrotor environment with curriculum learning for RMA.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        curriculum_stage: int = 1,
        imu_shock_std: float = 5.0,
        imu_shock_steps: int = 3,
        use_direct_control: bool = False,
        max_episode_steps: int = 500,
    ):
        super().__init__()

        self.curriculum_stage = curriculum_stage
        self.imu_shock_std = imu_shock_std
        self.imu_shock_steps = imu_shock_steps
        self.use_direct_control = use_direct_control

        # Load MuJoCo model
        self.model = MjModel.from_xml_path("quadrotor.xml")
        self.data = MjData(self.model)
        self.model.opt.timestep = 0.01
        self.model.opt.integrator = 2

        # Max episode steps (time limit, configurable)
        self.max_episode_steps = max_episode_steps
        # Constants
        self.nominal_mass = 1.0
        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0
        self._sync_visual_target()
        self.dt = 0.01
        self._use_moving_target = False  # for Stage 5/6 moving target
        self._prev_target_pos = np.array([0.0, 0.0, 1.0])
        self._prev_focal_pos = np.array([0.0, 0.0, 1.0])
        self._target_velocity = np.zeros(3)
        self._target_speed = 1.0  # speed multiplier for curriculum
        self._target_trajectory = "figure8"  # 'figure8' or 'racing'
        self._racing_lap_time = 0.0  # Track lap time for racing

        # Stage 7 yaw control parameters
        self._yaw_reward_weight = 0.0
        self._figure8_amplitude = 3.0  # default for Stage 7
        self._current_target_yaw = 0.0
        self._prev_target_yaw = 0.0
        self._target_yaw_rate = 0.0
        self._prev_cpt_pos_err = None
        self._future_target_100ms = np.zeros(3)
        self._future_target_200ms = np.zeros(3)
        self._future_target_300ms = np.zeros(3)
        self._yaw_only_mode = (
            False  # Stage 7 yaw-only: drone hovers at origin, focal point moves
        )
        self._yaw_rate_scale = 3.0  # max yaw rate command: action[3]=1 → 3 rad/s

        # Stage 8 progressive curriculum parameters (defaults, overridden by callback)
        self._phase_pos_range = 0.2
        self._phase_angle_range = 10.0

        # Store nominal inertia for proportional scaling on mass changes
        self._nominal_inertia = self.model.body_inertia[1].copy()

        # Hover thrust per motor
        # Residual RL action scale (added on top of PD base controller)
        # [dT, dR, dP, dY] - residual around PD output
        # INCREASED from 0.3 to 1.0 to allow full compensation for wind perturbations
        # With action_scale=1.0, max residual = ±5N thrust = ±50% of hover thrust,
        # sufficient to counteract ±0.5N wind forces (+ 50% safety margin)
        self.action_scale = np.array([1.0, 1.0, 1.0, 1.0])

        # Action space: residual around PD controller (-1 to 1)
        # In direct control mode, action = [thrust, roll_torque, pitch_torque, yaw_torque]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation space: stage-dependent
        # Stages 1-4: 60 dims (51 deployable + 9 privileged)
        # Stages 5-6: 63 dims (54 deployable + 9 privileged)
        # Stage 7: 75 dims (66 deployable + 9 privileged)
        obs_dim = 75 if curriculum_stage == 7 else (63 if curriculum_stage >= 5 else 60)
        self.observation_space = spaces.Box(
            low=-10, high=10, shape=(obs_dim,), dtype=np.float32
        )

        # State tracking
        self.prev_action = np.zeros(4)
        self.prev_body_rates = np.zeros(3)
        self.prev_angular_vel = np.zeros(3)
        self.drop_occurred = False
        self.drop_triggered = False
        self.steps_since_drop = 0

        # Action history ring buffer: 4 most-recent actions (4 x 4)
        self.action_history = np.zeros((4, 4), dtype=np.float32)

        # Error integrals: pos (3) + yaw (1), clamped to [-2, 2]
        self.pos_integral = np.zeros(3, dtype=np.float32)
        self.yaw_integral = 0.0

        # Rotor thrust estimates: 1st-order lag filter on commanded per-motor thrust
        # tau = 40ms, alpha = dt / (tau + dt)
        self._rotor_tau = 0.04
        self._rotor_alpha = self.dt / (self._rotor_tau + self.dt)
        self.rotor_thrusts = np.zeros(4, dtype=np.float32)

        # Motor delay simulation: first-order lag filter for ESC/rotor dynamics
        # alpha = 0.15 gives ~57ms time constant (simulates real ESC + rotor response)
        # tau = (1-alpha)/alpha * dt => tau = 0.85/0.15 * 0.01 ≈ 57ms
        self._motor_delay_alpha = 0.15
        self._motor_thrust_delayed = np.zeros(4)  # delayed thrust per motor
        self._motor_torque_delayed = np.zeros(3)  # delayed torques

        # Motor efficiency factors for Stage 5 motor degradation
        # 1.0 = full efficiency, 0.0 = complete failure
        self._motor_efficiency = np.ones(4, dtype=np.float32)  # per-motor efficiency

        # PD base controller constants (tuned for reliable Stage 1 hover with motor delay)
        self.max_thrust = 20.0
        # Outer gains: [kp_x, kp_y, kp_z, kd_z]
        # Cascaded PD gains: outer loop (position → desired attitude), inner loop (attitude → torques)
        # Tuned for >95% hover success in Stage 1 with motor delay
        # X/Y: PD + I → desired roll/pitch (cascaded through attitude inner loop)
        # Z: PD + I → thrust
        self.outer_gains = np.array(
            [8.0, 8.0, 16.0, 8.0]
        )  # [kp_x, kp_y, kp_z, kd_xy_z]
        # Inner gains: [kp_roll, kp_pitch, kp_yaw]
        # Inner/outer ratio ≈ 1.25 (aggressive for tight tracking with motor delay)
        self.inner_gains = np.array([10.0, 10.0, 2.5])
        self.rate_damping = 6.0  # angular rate damping gain
        self.torque_max = 5.0
        # Integral gains for position control (eliminates steady-state drift from motor delay)
        self.ki_pos = 4.0  # position integral gain (x/y)
        self.ki_z = 5.0  # z-axis integral gain
        self.rescue_mode = False

        # IMU-based mass estimator (replaces ground-truth mass)
        self._mass_hat = self.nominal_mass  # estimated mass
        self._mass_estimator_alpha = 0.02  # slow drift tracking
        self._mass_estimator_alpha_transient = 0.30  # fast tracking after jump
        self._mass_jump_detected = False
        self._mass_jump_timer = 0
        self._mass_jump_threshold = 0.01  # ~10g, achievable with alpha=0.02

        # Privileged state - initialized to defaults
        self.wind_force = np.zeros(3)
        self.com_shift = np.zeros(3)
        self.payload_mass = self.nominal_mass
        self.drop_time = None
        self.drop_magnitude = 0.0
        self.imu_shock_counter = 0

        # Store total_ctrl for mass estimator
        self._total_ctrl = np.zeros(4)

    def set_curriculum_stage(self, stage: int):
        """Set curriculum stage (called by callback)."""
        assert 1 <= stage <= 7, f"Invalid stage: {stage}"
        self.curriculum_stage = stage
        # Reset integrals and estimator on stage change (Bug #4 fix)
        self.pos_integral = np.zeros(3, dtype=np.float32)
        self.yaw_integral = 0.0
        self._mass_hat = self.nominal_mass
        self._mass_jump_detected = False
        self._mass_jump_timer = 0

    def set_target_speed(self, speed: float):
        """Set target movement speed multiplier for curriculum (Stage 5/6)."""
        assert 0.01 <= speed <= 5.0, f"Invalid speed: {speed}"
        self._target_speed = speed

    def set_moving_target(self, enabled: bool):
        """Enable/disable moving target mode."""
        self._use_moving_target = enabled
        if enabled:
            self._target_initial_pos = self.target_pos.copy()
            self._prev_target_pos = self.target_pos.copy()
            self._prev_focal_pos = self._target_initial_pos.copy()
            self._target_velocity = np.zeros(3)
            self._target_traj_time = 0.0

    def set_target_trajectory(self, trajectory: str):
        """Set target trajectory type ('figure8', 'figure8_yaw', 'racing', 'extended', 'linear_short', 'static')."""
        self._target_trajectory = trajectory

    def set_yaw_reward_weight(self, weight: float):
        """Set yaw reward weight for Stage 7 curriculum (0.0 to 3.0)."""
        self._yaw_reward_weight = weight

    def set_figure8_amplitude(self, amplitude: float):
        """Set figure-8 amplitude for Stage 7."""
        self._figure8_amplitude = amplitude

    def set_yaw_only_mode(self, enabled: bool):
        """Enable yaw-only mode: drone hovers at origin, focal point traces figure-8."""
        self._yaw_only_mode = enabled

    def _sync_visual_target(self):
        """Sync the MuJoCo target body position to match target_pos.

        The controller uses self.target_pos internally, but the MuJoCo
        viewer renders model.body_pos. Without this sync, the visual
        target stays at [0,0,1] even when target_pos moves (Stage 5).
        """
        self.model.body_pos[2][:3] = self.target_pos

    def _reset_fixed_hover(self):
        """Stage 1: Fixed hover start with small perturbations."""
        self.data.qpos[:] = 0
        self.data.qpos[2] = 1.0  # z = 1m
        self.data.qvel[:] = 0

        self.data.qpos[:3] += self.np_random.uniform(-0.05, 0.05, size=3)
        self.data.qpos[2] += self.np_random.uniform(-0.1, 0.1)
        self.data.qvel[:3] = self.np_random.uniform(-0.1, 0.1, size=3)
        self.data.qvel[3:6] = self.np_random.uniform(-0.2, 0.2, size=3)
        # Small random tilt (up to 5 degrees)
        angle_rad = np.deg2rad(5.0)
        roll = self.np_random.uniform(-angle_rad, angle_rad)
        pitch = self.np_random.uniform(-angle_rad, angle_rad)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = 1.0, 0.0  # yaw = 0
        self.data.qpos[3:7] = [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]

    def _reset_random_pose(self, pos_range=0.3, vel_range=0.3, angle_range=10.0):
        """Stages 2-4: Random initial pose."""
        # Position - randomize AROUND TARGET, not origin
        # Bug fix: was sampling from [-pos_range, pos_range] centered on origin [0,0,0],
        # but target is at [0,0,1.0]. This caused immediate truncation when starting
        # position was >0.5m from target.
        self.data.qpos[:3] = self.target_pos + self.np_random.uniform(
            -pos_range, pos_range, size=3
        )
        # Clamp z to reasonable range near target
        self.data.qpos[2] = np.clip(self.data.qpos[2], 0.3, 2.0)

        # Velocity
        self.data.qvel[:3] = self.np_random.uniform(-vel_range, vel_range, size=3)

        # Attitude (small random tilt)
        angle_rad = np.deg2rad(angle_range)
        roll = self.np_random.uniform(-angle_rad, angle_rad)
        pitch = self.np_random.uniform(-angle_rad, angle_rad)
        yaw = 0.0

        # Convert to quaternion
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)

        self.data.qpos[3:7] = [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]

        self.data.qvel[3:6] = self.np_random.uniform(-0.5, 0.5, size=3)

    def reset(self, seed=None, options=None):
        """Reset environment based on curriculum stage."""
        super().reset(seed=seed)

        # Reset target to default position (fixes Stage 5 drift between episodes)
        self.target_pos = np.array([0.0, 0.0, 1.0])
        # NOTE: _use_moving_target persists across resets (set by set_moving_target())
        self._target_velocity = np.zeros(3)
        self._target_traj_time = 0.0

        self.prev_action = np.zeros(4)
        self.prev_body_rates = np.zeros(3)
        self.prev_angular_vel = np.zeros(3)
        self.drop_occurred = False
        self.drop_triggered = False
        self.steps_since_drop = 0
        self._step_count = (
            0  # track time via steps, not data.time (data.time accumulates)
        )

        # Reset obs state
        self.action_history = np.zeros((4, 4), dtype=np.float32)
        self.pos_integral = np.zeros(3, dtype=np.float32)
        self.yaw_integral = 0.0
        self.rotor_thrusts = np.zeros(4, dtype=np.float32)
        self._motor_thrust_delayed = np.zeros(4)
        self._motor_torque_delayed = np.zeros(3)

        # Reset mass estimator
        self._mass_hat = self.nominal_mass
        self._mass_jump_detected = False
        self._mass_jump_timer = 0

        # Reset rescue mode
        self.rescue_mode = False

        # Stage-gated reset
        if self.curriculum_stage == 1:
            self._reset_fixed_hover()
            self.wind_force = np.zeros(3)
            self.payload_mass = self.nominal_mass
            self.model.body_mass[1] = self.payload_mass
            self.drop_time = None
            self.drop_magnitude = 0.0
            self.com_shift = np.zeros(3)
        elif self.curriculum_stage == 2:
            self._reset_random_pose(pos_range=0.2, vel_range=0.3, angle_range=10.0)
            self.wind_force = np.zeros(3)
            self.payload_mass = self.nominal_mass
            self.model.body_mass[1] = self.payload_mass
            self.drop_time = None
            self.drop_magnitude = 0.0
            self.com_shift = np.zeros(3)
        elif self.curriculum_stage == 3:
            self._reset_random_pose(pos_range=0.25, vel_range=0.5, angle_range=20.0)
            self.wind_force = self.np_random.uniform(-0.5, 0.5, size=3)
            self.payload_mass = self.nominal_mass * self.np_random.uniform(0.9, 1.1)
            self.model.body_mass[1] = self.payload_mass
            self.drop_time = None
            self.drop_magnitude = 0.0
            self.com_shift = np.zeros(3)
        elif self.curriculum_stage == 4:
            self._reset_random_pose(pos_range=0.25, vel_range=0.5, angle_range=20.0)
            self.wind_force = self.np_random.uniform(-0.5, 0.5, size=3)
            self.payload_mass = self.nominal_mass * self.np_random.uniform(0.9, 1.1)
            self.model.body_mass[1] = self.payload_mass
            # Drop event (50% probability)
            if self.np_random.random() < 0.5:
                self.drop_time = self.np_random.uniform(2.0, 8.0)
                self.drop_magnitude = self.np_random.uniform(0.15, 0.40)
                self.com_shift = self.np_random.uniform(-0.05, 0.05, size=3)
            else:
                self.drop_time = None
                self.drop_magnitude = 0.0
                self.com_shift = np.zeros(3)
        elif self.curriculum_stage == 5:
            # Stage 5: Moving target + all previous challenges
            # Target moves in a figure-8 pattern while handling wind and payload drops
            self._reset_random_pose(pos_range=0.2, vel_range=0.2, angle_range=15.0)
            self.wind_force = self.np_random.uniform(-0.8, 0.8, size=3)
            self.payload_mass = self.nominal_mass * self.np_random.uniform(0.9, 1.1)
            self.model.body_mass[1] = self.payload_mass
            # Drop event (50% probability)
            if self.np_random.random() < 0.5:
                self.drop_time = self.np_random.uniform(2.0, 8.0)
                self.drop_magnitude = self.np_random.uniform(0.15, 0.40)
                self.com_shift = self.np_random.uniform(-0.05, 0.05, size=3)
            else:
                self.drop_time = None
                self.drop_magnitude = 0.0
                self.com_shift = np.zeros(3)
            # Moving target parameters
            self._target_traj_time = 0.0
            self._target_initial_pos = self.target_pos.copy()
            self._prev_target_pos = self.target_pos.copy()
            self._prev_focal_pos = self._target_initial_pos.copy()
            self._target_velocity = np.zeros(3)
            # NOTE: _target_speed and _use_moving_target persist across resets (curriculum-controlled)
            self._target_trajectory = "figure8"

        elif self.curriculum_stage == 6:
            # Stage 6: Racing FPV - extreme speed with aggressive maneuvers
            # No starting perturbations - start clean for racing
            self._reset_random_pose(pos_range=0.1, vel_range=0.1, angle_range=5.0)
            # Stronger wind for racing turbulence
            self.wind_force = self.np_random.uniform(-1.0, 1.0, size=3)
            # No payload drops in racing - mass stays nominal
            self.payload_mass = self.nominal_mass
            self.model.body_mass[1] = self.payload_mass
            self.drop_time = None
            self.drop_magnitude = 0.0
            self.com_shift = np.zeros(3)
            # Racing target parameters
            self._target_traj_time = 0.0
            self._target_initial_pos = np.array([0.0, 0.0, 1.0])  # Center of track
            self._prev_target_pos = self.target_pos.copy()
            self._prev_focal_pos = self._target_initial_pos.copy()
            self._target_velocity = np.zeros(3)
            # NOTE: _target_speed, _use_moving_target, _target_trajectory persist across resets

        elif self.curriculum_stage == 7:
            # Stage 7: Yaw control with figure-8 + gaze targets
            # In yaw-only mode: drone hovers at random position, focal point moves relative to drone
            self._reset_random_pose(pos_range=0.5, vel_range=0.3, angle_range=20.0)
            self.wind_force = self.np_random.uniform(-0.8, 0.8, size=3)
            self.payload_mass = self.nominal_mass * self.np_random.uniform(0.9, 1.1)
            self.model.body_mass[1] = self.payload_mass
            if self.np_random.random() < 0.5:
                self.drop_time = self.np_random.uniform(2.0, 8.0)
                self.drop_magnitude = self.np_random.uniform(0.15, 0.40)
                self.com_shift = self.np_random.uniform(-0.05, 0.05, size=3)
            else:
                self.drop_time = None
                self.drop_magnitude = 0.0
                self.com_shift = np.zeros(3)
            # Yaw figure-8 target parameters
            self._target_traj_time = 0.0
            # Randomize initial position: focal center varies ±1m from origin
            self._target_initial_pos = np.array(
                [
                    self.np_random.uniform(-1.0, 1.0),
                    self.np_random.uniform(-1.0, 1.0),
                    self.np_random.uniform(0.8, 1.2),
                ]
            )
            self._prev_target_pos = self.target_pos.copy()
            self._prev_focal_pos = self._target_initial_pos.copy()
            self._target_velocity = np.zeros(3)
            # NOTE: _target_speed, _target_trajectory, _use_moving_target, _yaw_only_mode
            # are NOT reset here — they are curriculum-controlled and persist across episodes.
            # Reset CPT state
            self._prev_cpt_pos_err = None
            self._current_target_yaw = 0.0
            self._prev_target_yaw = 0.0
            self._target_yaw_rate = 0.0
            # In yaw-only mode, initialize PD target_yaw to face the focal
            if self._yaw_only_mode:
                self.target_yaw = np.arctan2(
                    self._target_initial_pos[1] - self.data.qpos[1],
                    self._target_initial_pos[0] - self.data.qpos[0],
                )
            self._future_target_100ms = np.zeros(3)
            self._future_target_200ms = np.zeros(3)
            self._future_target_300ms = np.zeros(3)

        elif self.curriculum_stage == 8:
            # Stage 8: Extreme Extended Racing
            # Perturbation ranges are controlled by curriculum phases via _phase_pos_range, _phase_angle_range
            pos_range = getattr(self, "_phase_pos_range", 0.2)
            angle_range = getattr(self, "_phase_angle_range", 10.0)
            self._reset_random_pose(
                pos_range=pos_range, vel_range=0.2, angle_range=angle_range
            )
            # Wind is set by curriculum callback (not here)
            # Ensure wind_force exists; will be overwritten by callback
            if not hasattr(self, "wind_force"):
                self.wind_force = np.zeros(3)
            # Mass stays nominal (no drops during racing)
            self.payload_mass = self.nominal_mass
            self.model.body_mass[1] = self.payload_mass
            self.drop_time = None
            self.drop_magnitude = 0.0
            self.com_shift = np.zeros(3)
            # Extended racing target parameters
            self._target_traj_time = 0.0
            # Target near drone start position
            self._target_initial_pos = np.array([0.0, 0.0, 1.0])
            self._prev_target_pos = self.target_pos.copy()
            self._prev_focal_pos = self._target_initial_pos.copy()
            self._target_velocity = np.zeros(3)
            # NOTE: _target_speed, _target_trajectory, _use_moving_target persist across resets

        # Initialize mass estimator to true mass (warm start)
        self._mass_hat = self.payload_mass

        # Apply CoM shift to MuJoCo model (Bug 11 fix: was dead variable before)
        self.model.body_ipos[1][:3] = self.com_shift

        # Scale inertia proportionally with mass (Bug 15 fix)
        mass_ratio = self.payload_mass / self.nominal_mass
        self.model.body_inertia[1] = self._nominal_inertia * mass_ratio

        # Initialize delayed thrust with correct mass estimate (Bug 10 fix)
        self._motor_thrust_delayed[0] = self._mass_hat * 9.81

        # Sync visual target position for viewer
        self._sync_visual_target()

        # Return combined obs (deployable + privileged)
        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def _apply_disturbances(self):
        """Apply wind force additively (on top of thrust already set)."""
        self.data.xfrc_applied[1, :3] += self.wind_force

    def step(self, action):
        """Execute one timestep."""
        self._step_count += 1

        # Clear external forces/torques from previous step to prevent accumulation
        self.data.xfrc_applied[1] = 0.0

        # Capture angular velocity at start of step (before physics advance)
        self.prev_angular_vel = self.data.qvel[3:6].copy()

        # Apply disturbances (wind) before physics
        self._apply_disturbances()

        # Compute total control signal
        if self.use_direct_control:
            # Direct control mode: action = [thrust, roll_torque, pitch_torque, yaw_torque]
            # Scale action from [-1,1] to physical ranges
            # Torque limits match env's max_thrust and reasonable torque authority
            self._total_ctrl = np.array(
                [
                    (action[0] + 1.0) * 10.0,  # thrust: [0, 20] N
                    action[1] * 3.0,  # roll torque: [-3, 3] Nm
                    action[2] * 3.0,  # pitch torque: [-3, 3] Nm
                    action[3] * 2.0,  # yaw torque: [-2, 2] Nm
                ]
            )
        else:
            # Residual RL: action is residual on top of PD base controller
            # In yaw-only mode, action[3] is a yaw RATE command, not residual torque
            if self._yaw_only_mode:
                # Integrate yaw rate command into PD target
                self.target_yaw = self._wrap_angle(
                    self.target_yaw + action[3] * self._yaw_rate_scale * self.dt
                )
                # Compute PD with updated yaw target, zero out residual yaw
                base_ctrl = self._cascaded_controller()
                residual = action.copy()
                residual[3] = 0.0  # yaw handled by target update, not residual
                self._total_ctrl = base_ctrl + residual * self.action_scale
            else:
                base_ctrl = self._cascaded_controller()
                self._total_ctrl = base_ctrl + action * self.action_scale

        # Apply motor control with delay
        self._apply_pd_control(self._total_ctrl)

        # Step physics
        mj_step(self.model, self.data)

        # Trigger drop event (Stages 4 and 5 both have payload drops)
        t = self._step_count * self.dt
        if self.curriculum_stage >= 4 and self.drop_time is not None:
            if t >= self.drop_time and not self.drop_triggered:
                # Drop relative to nominal mass for deterministic magnitude
                # (Bug fix: *= was relative to current mass which varies)
                self.model.body_mass[1] = (
                    self.model.body_mass[1] - self.nominal_mass * self.drop_magnitude
                )
                self.payload_mass = self.model.body_mass[1]
                # Scale inertia proportionally with mass (Bug 15 fix)
                mass_ratio_drop = self.payload_mass / self.nominal_mass
                self.model.body_inertia[1] = self._nominal_inertia * mass_ratio_drop
                self.drop_occurred = True
                self.drop_triggered = True
                self.imu_shock_counter = self.imu_shock_steps

        # Update moving target for Stage 5, 7, 8 or custom trajectories
        if (
            self.curriculum_stage in [5, 6, 7, 8] and self._use_moving_target
        ) or self._target_trajectory in ["linear_short", "extended"]:
            self._target_traj_time += (
                self.dt * self._target_speed
            )  # apply speed multiplier

            if self._target_trajectory == "figure8":
                # Figure-8 trajectory (Lemniscate of Bernoulli)
                a = 0.5  # amplitude
                t_traj = self._target_traj_time
                dx = a * np.sin(t_traj)
                dy = a * np.sin(2 * t_traj) / 2  # different frequency for figure-8
                dz = 0.3 * np.sin(t_traj / 3)  # slow vertical oscillation
                new_target = self._target_initial_pos + np.array([dx, dy, dz])

            elif self._target_trajectory == "racing":
                # Racing circuit: oval track with hairpin turns
                # Track: length 8m straight, 3m curves, total ~22m lap
                t_traj = self._target_traj_time
                track_length = 8.0  # straight section length
                curve_radius = 1.5  # turn radius

                # Normalized position along track (0 to 1 = one lap)
                lap_pos = (t_traj * 0.3) % 1.0  # ~3.3 second lap at 1x speed

                if lap_pos < 0.4:
                    # Straight 1: going forward (+x direction)
                    x = -track_length / 2 + lap_pos / 0.4 * track_length
                    y = -curve_radius
                    z = 1.0
                elif lap_pos < 0.5:
                    # Turn 1: 90° hairpin
                    angle = (lap_pos - 0.4) / 0.1 * np.pi / 2
                    x = track_length / 2
                    y = -curve_radius + curve_radius * np.sin(angle)
                    z = 1.0
                elif lap_pos < 0.9:
                    # Straight 2: coming back (-x direction)
                    x = track_length / 2 - (lap_pos - 0.5) / 0.4 * track_length
                    y = curve_radius
                    z = 1.0
                else:
                    # Turn 2: 90° hairpin back to start
                    angle = (lap_pos - 0.9) / 0.1 * np.pi / 2
                    x = -track_length / 2
                    y = curve_radius * np.cos(angle)
                    z = 1.0

                new_target = np.array([x, y, z])

            elif self._target_trajectory == "figure8_medium":
                # Medium figure-8 for Phase 2.5: 1.5m amplitude, horizontal only
                a = 1.5
                t_traj = self._target_traj_time
                dx = a * np.sin(t_traj)
                dy = a * np.sin(2 * t_traj) / 2
                dz = 0.0  # No vertical yet
                new_target = self._target_initial_pos + np.array([dx, dy, dz])

            elif self._target_trajectory == "figure8_large":
                # Large figure-8 for Phase 3: 2.0m amplitude, light vertical
                a = 2.0
                t_traj = self._target_traj_time
                dx = a * np.sin(t_traj)
                dy = a * np.sin(2 * t_traj) / 2
                dz = 0.2 * np.sin(t_traj / 3)  # Light vertical
                new_target = self._target_initial_pos + np.array([dx, dy, dz])

            elif self._target_trajectory == "figure8_large2":
                # Phase 3.5: Intermediate large figure-8: 2.5m amplitude, moderate vertical
                a = 2.5
                t_traj = self._target_traj_time
                dx = a * np.sin(t_traj)
                dy = a * np.sin(2 * t_traj) / 2
                dz = 0.25 * np.sin(t_traj / 3)  # Moderate vertical
                new_target = self._target_initial_pos + np.array([dx, dy, dz])

            elif self._target_trajectory == "extended":
                # Extended figure-8 for Phase 4: 3.0m amplitude, full vertical
                a = 3.0
                t_traj = self._target_traj_time
                dx = a * np.sin(t_traj)
                dy = a * np.sin(2 * t_traj) / 2
                dz = 0.3 * np.sin(
                    t_traj / 3
                )  # Same vertical oscillation as figure8_yaw
                new_target = self._target_initial_pos + np.array([dx, dy, dz])

                # Compute future targets for potential predictive rewards (future use)
                for tau, attr in [
                    (0.1, "_future_target_100ms"),
                    (0.2, "_future_target_200ms"),
                    (0.3, "_future_target_300ms"),
                ]:
                    t_future = t_traj + tau * self._target_speed
                    fdx = a * np.sin(t_future)
                    fdy = a * np.sin(2 * t_future) / 2
                    fdz = 0.3 * np.sin(t_future / 3)
                    setattr(
                        self, attr, self._target_initial_pos + np.array([fdx, fdy, fdz])
                    )

            elif self._target_trajectory == "linear_short":
                # Short linear trajectory for Phase 2: 5m line back and forth
                line_length = 5.0
                t_traj = self._target_traj_time
                # Oscillate along x-axis
                x = self._target_initial_pos[0] + (line_length / 2) * np.sin(
                    t_traj * 0.2
                )
                y = self._target_initial_pos[1]
                z = self._target_initial_pos[2] + 0.2 * np.sin(
                    t_traj * 0.5
                )  # Small vertical
                new_target = np.array([x, y, z])

                # Simple future targets (linear extrapolation)
                dt_future = 0.2
                future_x = self._target_initial_pos[0] + (line_length / 2) * np.sin(
                    (t_traj + dt_future) * 0.2
                )
                self._future_target_100ms = np.array([future_x, y, z])
                self._future_target_200ms = np.array(
                    [future_x, y, z]
                )  # Same for simplicity
                self._future_target_300ms = np.array([future_x, y, z])

            elif self._target_trajectory == "static":
                # Static target: no movement
                new_target = self._target_initial_pos.copy()

            elif self._target_trajectory == "figure8_yaw":
                # Figure-8 with yaw gaze targets (Stage 7)
                # Large amplitude (3m) figure-8 with focal point blending
                a = self._figure8_amplitude
                t_traj = self._target_traj_time
                dx = a * np.sin(t_traj)
                dy = a * np.sin(2 * t_traj) / 2
                dz = 0.3 * np.sin(t_traj / 3)
                focal = self._target_initial_pos + np.array([dx, dy, dz])

                if self._yaw_only_mode:
                    # Drone hovers at origin, focal point traces figure-8
                    # target_pos stays at hover point (drone doesn't fly to the focal)
                    new_target = self._target_initial_pos.copy()
                    # Yaw: point at focal
                    pos_now = self.data.qpos[:3]
                    self._current_target_yaw = np.arctan2(
                        focal[1] - pos_now[1], focal[0] - pos_now[0]
                    )
                else:
                    # Normal: drone flies toward the moving target
                    new_target = focal
                    # Compute gaze target yaw using focal point blending
                    pos_now = self.data.qpos[:3]
                    focal_right = np.array([2 * a / 3, 0.0, new_target[2]])
                    focal_left = np.array([-2 * a / 3, 0.0, new_target[2]])
                    blend = 0.5 + 0.5 * np.tanh(dx / (a * 0.15))
                    focal_pt = focal_right * (1 - blend) + focal_left * blend
                    self._current_target_yaw = np.arctan2(
                        focal_pt[1] - pos_now[1], focal_pt[0] - pos_now[0]
                    )

                # Compute future targets for CPT predictive reward
                for tau, attr in [
                    (0.1, "_future_target_100ms"),
                    (0.2, "_future_target_200ms"),
                    (0.3, "_future_target_300ms"),
                ]:
                    t_future = t_traj + tau * self._target_speed
                    fdx = a * np.sin(t_future)
                    fdy = a * np.sin(2 * t_future) / 2
                    fdz = 0.3 * np.sin(t_future / 3)
                    setattr(
                        self, attr, self._target_initial_pos + np.array([fdx, fdy, fdz])
                    )

            elif self._target_trajectory == "static":
                # Static target: no movement
                new_target = self._target_initial_pos.copy()

            else:
                # Default figure-8 (Stage 5 standard)
                a = 0.5
                t_traj = self._target_traj_time
                dx = a * np.sin(t_traj)
                dy = a * np.sin(2 * t_traj) / 2
                dz = 0.3 * np.sin(t_traj / 3)
                new_target = self._target_initial_pos + np.array([dx, dy, dz])

            # Compute target velocity for predictive reward
            if self._yaw_only_mode and self._target_trajectory == "figure8_yaw":
                # In yaw-only mode, target_pos stays fixed but focal moves.
                # Provide focal velocity so the model knows how fast to yaw.
                self._target_velocity = (focal - self._prev_focal_pos) / (
                    self.dt + 1e-8
                )
                self._prev_focal_pos = focal.copy()
            else:
                self._target_velocity = (new_target - self._prev_target_pos) / (
                    self.dt * self._target_speed + 1e-8
                )
                self._prev_target_pos = self.target_pos.copy()
            self.target_pos = new_target

            # Compute yaw rate for observation
            self._target_yaw_rate = self._wrap_angle(
                self._current_target_yaw - self._prev_target_yaw
            ) / (self.dt + 1e-8)
            self._prev_target_yaw = self._current_target_yaw

            # Sync visual target so viewer shows the moving target
            self._sync_visual_target()

        # Update mass estimator (after physics step, uses body-frame IMU)
        self._update_mass_estimator()

        # Update action history (shift and prepend latest action)
        self.action_history = np.roll(self.action_history, 1, axis=0)
        self.action_history[0] = action

        # Update rotor thrust estimate (1st-order lag on per-motor thrust)
        thrust_per_rotor = self._total_ctrl[0] / 4.0
        self.rotor_thrusts = (1 - self._rotor_alpha) * self.rotor_thrusts
        +self._rotor_alpha * np.array([thrust_per_rotor] * 4, dtype=np.float32)

        # Update error integrals (clamped to [-5, 5] for stronger integral action)
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        rpy = self._quat_to_rpy(quat)
        self.pos_integral = np.clip(
            self.pos_integral + (self.target_pos - pos) * self.dt, -5.0, 5.0
        ).astype(np.float32)
        self.yaw_integral = float(
            np.clip(self.yaw_integral + (0.0 - rpy[2]) * self.dt, -2.0, 2.0)
        )

        # Get observation
        obs_raw = self._get_obs_raw()

        # Apply IMU shock if active
        if self.imu_shock_counter > 0:
            shock = self.np_random.normal(0, self.imu_shock_std, size=3)
            obs_raw["lin_accel"] += shock
            self.imu_shock_counter -= 1

        # Combine deployable + privileged
        obs = self._get_obs_from_raw(obs_raw)

        # Check termination
        pos = self.data.qpos[:3]
        quat = self.data.qpos[3:7]
        rpy = self._quat_to_rpy(quat)

        # Safety envelope - distance from TARGET (not origin)
        # 0.5m radius from target position [0, 0, 1.0]
        dist_from_target = np.linalg.norm(pos - self.target_pos)
        roll, pitch = rpy[0], rpy[1]

        terminated = False
        truncated = False

        # Crash: drone is unrecoverable → terminated (zero future value for bootstrap)
        # Stages 6, 7, 8 allow aggressive 120° attitudes for extreme maneuvers
        if self.curriculum_stage in [6, 7, 8]:
            crash_angle = np.deg2rad(120)
        else:
            crash_angle = np.deg2rad(90)
        if abs(roll) > crash_angle or abs(pitch) > crash_angle:
            terminated = True

        # Boundary/timeout: drone is OK but out of zone → truncated (bootstrap OK)
        # Stage 5 has wider safety zone for figure-8 tracking
        # Stage 6 (racing) has even wider zone for high-speed maneuvers
        # Stage 7 (yaw) needs wide zone for 3m figure-8
        # Stage 8 (extended) needs largest zone for 29m track
        if self.curriculum_stage == 8:
            safety_radius = 8.0  # Extended racing: even wider boundary for long track
        elif self.curriculum_stage == 6:
            safety_radius = 3.0  # Racing: allow more tracking lag at high speed
        elif self.curriculum_stage == 7:
            safety_radius = 6.0  # Yaw: 3m amplitude figure-8 needs wide boundary
        elif self.curriculum_stage == 5:
            safety_radius = 1.5  # Figure-8: moderate tracking lag
        else:
            safety_radius = 0.5  # Stages 1-4: tight hovering
        if dist_from_target > safety_radius:
            truncated = True
        if self._step_count >= self.max_episode_steps:
            truncated = True

        # Compute reward
        reward = self._compute_reward(action)

        # Update previous action
        self.prev_action = action.copy()

        # prev_body_rates used externally if needed; keep in sync
        self.prev_body_rates = self.data.qvel[3:6].copy()

        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _apply_pd_control(self, ctrl):
        """Apply control output to the quadrotor with motor delay.

        ctrl: [thrust_z, roll_torque, pitch_torque, yaw_torque].
        Thrust is along the body z-axis, projected to world frame.
        Motor delay simulates ESC/rotor spin-up time via first-order lag filter.
        """
        quat = self.data.qpos[3:7]
        rotation = self._quat_to_rotmat(quat)

        # Compute commanded thrust and torques
        thrust_body = np.array([0, 0, ctrl[0]])
        thrust_world = rotation @ thrust_body

        # Apply first-order lag filter for motor delay (ESC + rotor inertia)
        # The delay applies to the thrust MAGNITUDE, not the direction.
        # The direction comes from the current body orientation (rotation matrix).
        alpha = self._motor_delay_alpha

        # Delayed thrust magnitude (body z-axis force magnitude)
        thrust_cmd = ctrl[0]
        self._motor_thrust_delayed[0] = (1 - alpha) * self._motor_thrust_delayed[
            0
        ] + alpha * thrust_cmd

        # Delayed torques
        torque_cmd = ctrl[1:4]
        self._motor_torque_delayed = (
            1 - alpha
        ) * self._motor_torque_delayed + alpha * torque_cmd

        # Apply DELAYED thrust along current body orientation
        # This correctly generates XY forces when the body is tilted
        delayed_thrust_body = np.array([0, 0, self._motor_thrust_delayed[0]])
        delayed_thrust_world = rotation @ delayed_thrust_body

        # Use += to accumulate on top of wind/disturbances (already set in xfrc_applied)
        self.data.xfrc_applied[1, :3] += delayed_thrust_world

        # Rotate body-frame torques to world frame before applying
        # (xfrc_applied expects world-frame torques, controller computes body-frame)
        torque_world = rotation @ self._motor_torque_delayed
        self.data.xfrc_applied[1, 3:6] += torque_world

    def _cascaded_controller(self):
        """Cascaded PD controller with estimated mass for hover thrust."""
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        ang_vel = self.data.qvel[3:6]
        quat = self.data.qpos[3:7]

        pos_err = self.target_pos - pos
        vel_err = -vel

        # Use ESTIMATED mass for hover (not ground-truth payload_mass)
        hover = self._mass_hat * 9.81

        current_rpy = self._quat_to_rpy(quat)
        current_roll = current_rpy[0]
        current_pitch = current_rpy[1]

        # Rescue mode thresholds
        rescue_threshold = np.deg2rad(25)
        rescue_torque_mult = 10.0
        max_torque_rescue = 10.0
        yaw_deprioritize_factor = 0.3

        rescue_triggered = (
            abs(current_roll) > rescue_threshold
            or abs(current_pitch) > rescue_threshold
        )

        if rescue_triggered:
            self.rescue_mode = True
            desired_thrust_z = max(hover - 2.0, 5.0)
            desired_thrust_z = np.clip(desired_thrust_z, 5.0, self.max_thrust)

            roll_error = -current_roll
            pitch_error = -current_pitch

            inner_kp_rescue = self.inner_gains * rescue_torque_mult
            rate_damping = 3.0
            desired_roll_rate = (
                inner_kp_rescue[0] * roll_error - rate_damping * ang_vel[0]
            )
            desired_pitch_rate = (
                inner_kp_rescue[1] * pitch_error - rate_damping * ang_vel[1]
            )
            # Fix: add yaw error feedback (not just rate damping)
            yaw_error = self.target_yaw - current_rpy[2]
            desired_yaw_rate = (
                yaw_error * self.inner_gains[2] * yaw_deprioritize_factor
                - ang_vel[2] * yaw_deprioritize_factor
            )

            roll_torque = np.clip(
                desired_roll_rate, -max_torque_rescue, max_torque_rescue
            )
            pitch_torque = np.clip(
                desired_pitch_rate, -max_torque_rescue, max_torque_rescue
            )
            yaw_torque = np.clip(desired_yaw_rate, -self.torque_max, self.torque_max)

            return np.array([desired_thrust_z, roll_torque, pitch_torque, yaw_torque])

        self.rescue_mode = False

        # Normal cascaded control with integral terms
        # Z-axis: PD + I (eliminates steady-state drift from motor delay)
        desired_thrust_z = (
            self.outer_gains[2] * pos_err[2]
            + self.outer_gains[3] * vel_err[2]
            + self.ki_z * self.pos_integral[2]
            + hover
        )
        desired_thrust_z = np.clip(desired_thrust_z, 0.0, self.max_thrust)

        # X/Y position: PD + I → desired attitude
        # Sign convention verified: positive pitch tilts body z toward +X,
        # negative roll tilts body z toward +Y.
        # X-position error → pitch command (positive pitch = +X force)
        # Y-position error → roll command (negative roll = +Y force)
        desired_pitch = np.arctan2(
            self.outer_gains[0] * pos_err[0]
            + self.outer_gains[3] * vel_err[0]
            + self.ki_pos * self.pos_integral[0],
            hover,
        )
        desired_roll = np.arctan2(
            -(
                self.outer_gains[1] * pos_err[1]
                + self.outer_gains[3] * vel_err[1]
                + self.ki_pos * self.pos_integral[1]
            ),
            hover,
        )

        roll_error = desired_roll - current_roll
        pitch_error = desired_pitch - current_pitch

        desired_roll_rate = (
            self.inner_gains[0] * roll_error - self.rate_damping * ang_vel[0]
        )
        desired_pitch_rate = (
            self.inner_gains[1] * pitch_error - self.rate_damping * ang_vel[1]
        )
        # Fix: yaw control with proportional error + rate damping (matches roll/pitch)
        # Bug fix: was `desired_yaw_rate = -ang_vel[2]; torque = desired - ang_vel[2]`
        # which applied rate damping TWICE
        yaw_error = self.target_yaw - current_rpy[2]
        desired_yaw_rate = (
            self.inner_gains[2] * yaw_error - self.rate_damping * ang_vel[2]
        )

        roll_torque = np.clip(desired_roll_rate, -self.torque_max, self.torque_max)
        pitch_torque = np.clip(desired_pitch_rate, -self.torque_max, self.torque_max)
        yaw_torque = np.clip(desired_yaw_rate, -self.torque_max, self.torque_max)

        return np.array([desired_thrust_z, roll_torque, pitch_torque, yaw_torque])

    def _update_mass_estimator(self):
        """Update mass estimate using body-frame IMU data.

        Uses body-frame specific force (what an IMU actually measures) to estimate mass.
        The key physics: T = m * a_body_z, where T is thrust along body z-axis
        and a_body_z is the z-component of body-frame specific force.
        Therefore: m = T / a_body_z.

        Body-frame specific force = R^T * (a_world - g_world)
        This correctly handles tilt (gravity projects through rotation matrix).

        IMPORTANT: The mass estimate is only reliable near steady state (hover).
        During transients, commanded thrust ≠ actual thrust (due to motor delay),
        so m_meas would be wildly wrong. We gate updates to:
        1. Skip first 50 steps (let motor delay settle)
        2. Only update when near hover (small pos error + small velocity)
        3. Detect acceleration anomalies for fast adaptation after mass drops
        """
        # Skip estimation during settling (motor delay hasn't stabilized)
        if self._step_count < 50:
            return

        quat = self.data.qpos[3:7]
        rotmat = self._quat_to_rotmat(quat)

        # Body-frame specific force (what IMU measures)
        body_accel = self._get_body_accel(rotmat)
        body_accel_z = body_accel[2]

        # Actual thrust on the body (delayed, not commanded)
        # Bug fix: was using _total_ctrl (commanded) which differs from actual during transients
        thrust = self._motor_thrust_delayed[0]

        # Near-hover gate: only update mass estimate when drone is close to hover.
        # During transients, commanded thrust != actual thrust (motor delay),
        # so m_meas would be garbage. Relaxed thresholds to allow updates during
        # moderate oscillations (e.g. after mass drops).
        pos_err_norm = np.linalg.norm(self.target_pos - self.data.qpos[:3])
        vel_norm = np.linalg.norm(self.data.qvel[:3])
        # Also check thrust is near hover (not actively correcting)
        thrust_near_hover = abs(thrust - self._mass_hat * 9.81) < 3.0
        near_hover = pos_err_norm < 0.3 and vel_norm < 0.5 and thrust_near_hover

        # Compute instantaneous mass estimate when valid and near hover
        if near_hover and thrust > 1.0 and abs(body_accel_z) > 1.0:
            m_meas = thrust / body_accel_z
            m_meas = np.clip(m_meas, 0.3, 2.5)  # physical limits

            # Adaptive EMA: fast after jump detection, slow otherwise
            if self._mass_jump_detected:
                alpha = self._mass_estimator_alpha_transient  # 0.30
            else:
                alpha = self._mass_estimator_alpha  # 0.02

            prev_m_hat = self._mass_hat
            self._mass_hat = (1 - alpha) * self._mass_hat + alpha * m_meas

            # Jump detection: large mass change triggers fast-tracking mode
            if abs(self._mass_hat - prev_m_hat) > self._mass_jump_threshold:
                self._mass_jump_detected = True
                self._mass_jump_timer = 50  # ~0.5s of fast tracking
            elif self._mass_jump_timer > 0:
                self._mass_jump_timer -= 1
            else:
                self._mass_jump_detected = False

        # Acceleration anomaly detection for mass jumps (even when not near hover)
        # After a mass drop, body_accel_z spikes because thrust is now too high
        # This detects the spike and triggers fast-tracking mode
        if not near_hover and not self._mass_jump_detected:
            # Expected body accel for current mass estimate at hover
            expected_accel_z = 9.81  # at hover, body z-accel = g
            accel_anomaly = abs(body_accel_z - expected_accel_z)
            if accel_anomaly > 3.0 and thrust > 1.0:
                # Large acceleration anomaly: likely a mass change
                self._mass_jump_detected = True
                self._mass_jump_timer = 50

    def _pd_controller(self):
        """Alias for cascaded controller (matches env.py API)."""
        return self._cascaded_controller()

    def _compute_reward(self, action):
        """Compute reward using exact formula from spec with improved shaping."""
        # Get state
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        quat = self.data.qpos[3:7]
        rpy = self._quat_to_rpy(quat)
        body_rates = self.data.qvel[3:6]

        pos_err = np.linalg.norm(self.target_pos - pos)
        vel_err = np.linalg.norm(vel)
        att_err = np.linalg.norm(rpy[:2])  # roll + pitch only
        rate_err = np.linalg.norm(body_rates)

        # Action change (delta from previous)
        act_delta = np.linalg.norm(action - self.prev_action)

        r_alive = 1.0
        r_pos = -2.0 * np.tanh(pos_err)
        r_att = -1.5 * np.tanh(att_err)
        # Attitude cliff: steep penalty for tilting past threshold to prevent crash spiral
        # Stage 6/7 need higher threshold for aggressive maneuvering (racing/yaw)
        if self.curriculum_stage in [6, 7]:
            att_cliff_threshold = 0.87  # ~50 degrees - needed for 3m figure-8 banking
        else:
            att_cliff_threshold = 0.52  # ~30 degrees for hover/station-keeping
        if att_err > att_cliff_threshold:
            excess = att_err - att_cliff_threshold
            r_att -= 5.0 * excess * excess  # up to -5.0 at 60° tilt
        # Velocity penalty: higher for stages that need speed matching
        vel_weight = 0.5 if self.curriculum_stage == 7 else 0.05
        r_vel = -vel_weight * np.tanh(vel_err)
        r_rate = -0.05 * np.tanh(rate_err)
        r_smooth = -0.01 * np.tanh(act_delta)

        # Success bonus (sparse but significant)
        r_success = 5.0 if pos_err < 0.1 and att_err < 0.1 else 0.0

        # DENSE proximity reward: reward increases continuously as drone approaches target
        # This provides much stronger learning signal than sparse success bonus alone
        r_proximity = 0.5 if pos_err < 0.2 else 0.0
        r_alignment = 0.3 if pos_err < 0.3 and att_err < 0.1 else 0.0

        # Post-drop recovery bonus (stages 4 and 5)
        r_recovery = 0.0
        if self.curriculum_stage >= 4 and self.drop_occurred:
            pos_err_drop = np.linalg.norm(self.target_pos - pos)
            att_err_drop = np.linalg.norm(rpy[:2])
            if pos_err_drop < 0.15 and att_err_drop < 0.15:
                r_recovery = 2.0

        # Angular acceleration penalty
        angular_accel = np.linalg.norm(body_rates - self.prev_angular_vel) / self.dt
        r_jerk = -0.05 * np.tanh(angular_accel / 50.0)

        # PREDICTIVE TRACKING REWARD (Stages 5, 7, 8):
        # Reward the drone for matching the target's velocity DIRECTION (not magnitude).
        # This teaches the network: "if target moves right, move right too"
        # Unlike the aggressive velocity matching we tried before (which caused crashes),
        # this only rewards direction alignment, giving the network freedom on speed.
        r_track = 0.0
        r_stability = 0.0  # Stage 5 stability bonus
        if self.curriculum_stage in [5, 7, 8] and self._use_moving_target:
            target_vel = self._target_velocity
            target_speed = np.linalg.norm(target_vel)
            if target_speed > 0.01:  # only when target is actually moving
                target_dir = target_vel / target_speed
                drone_speed = np.linalg.norm(vel)
                if drone_speed > 0.01:
                    drone_dir = vel / drone_speed
                    # Cosine similarity of velocity directions (1 = same direction, -1 = opposite)
                    direction_alignment = np.dot(drone_dir, target_dir)
                    r_track = (
                        0.5 * direction_alignment
                    )  # max +0.5 for perfect alignment
                else:
                    r_track = 0.0  # no reward if drone is stationary

            # STABILITY: Penalize large attitude angles to prevent aggressive flips
            # Only activate when drone is already close (reduces crash risk without
            # hurting exploration at the start of episodes)
            r_stability = 0.0

        # RACING REWARD (Stages 6 & 8):
        # Rewards for aggressive FPV racing: speed matching, G-load limits, progress
        r_racing = 0.0
        if self.curriculum_stage in [6, 8] and self._use_moving_target:
            # 1. Speed matching: reward for matching target speed magnitude
            target_speed = np.linalg.norm(self._target_velocity)
            drone_speed = np.linalg.norm(vel)
            if target_speed > 0.1 and drone_speed > 0.1:
                speed_ratio = min(drone_speed / target_speed, 2.0)  # Cap at 2x
                r_racing += 1.0 * speed_ratio  # Up to +2.0

            # 2. G-load penalty: penalize excessive acceleration (G's > 4.0)
            lin_accel = self._get_body_accel(self._quat_to_rotmat(quat))
            g_load = np.linalg.norm(lin_accel) / 9.81
            if g_load > 4.0:
                r_racing -= 0.5 * (g_load - 4.0)
            elif g_load > 2.0:
                r_racing -= 0.1 * (g_load - 2.0)

            # 3. Angular rate penalty: typical FPV limit 5 rad/s
            max_rate = 5.0
            if abs(body_rates[0]) > max_rate:
                r_racing -= 0.3 * (abs(body_rates[0]) - max_rate)
            if abs(body_rates[1]) > max_rate:
                r_racing -= 0.3 * (abs(body_rates[1]) - max_rate)

            # 4.WAYPOINT PROGRESS BONUS for extended track (Stage 8 only)
            if self.curriculum_stage == 8:
                # Reward for making forward progress along trajectory
                # Use distance traveled along track as proxy
                if hasattr(self, "_prev_target_traj_time"):
                    time_progress = self._target_traj_time - self._prev_target_traj_time
                    if time_progress > 0:  # Target advancing
                        # Reward drone for staying near target (progress indicator)
                        progress_bonus = 0.2 * np.exp(-pos_err / 1.0)
                        r_racing += progress_bonus
                self._prev_target_traj_time = self._target_traj_time

                # 5. CROSS-TRACK ERROR penalty: keep drone near trajectory line
                # Compute perpendicular distance from drone to target direction
                if target_speed > 0.01:
                    # Vector from drone to target
                    to_target = self.target_pos - pos
                    # Project onto target velocity direction
                    target_dir = self._target_velocity / target_speed
                    lateral_error = np.linalg.norm(
                        to_target - np.dot(to_target, target_dir) * target_dir
                    )
                    # Penalty for large cross-track error (>1.0m)
                    if lateral_error > 1.0:
                        r_racing -= 0.2 * (lateral_error - 1.0)

            # 6. Success/close tracking bonus
            if pos_err < 0.3 and att_err < 0.3:
                r_racing += 2.0

        # YAW REWARD (Stage 7):
        r_yaw = 0.0
        if self.curriculum_stage == 7 and self._use_moving_target:
            yaw_weight = self._yaw_reward_weight
            if yaw_weight > 0:
                if self._yaw_only_mode:
                    # Yaw-only mode: drone hovers at origin, points at focal
                    # Position penalty: keep drone near hover
                    pos_err_hover = np.linalg.norm(pos - np.array([0.0, 0.0, 1.0]))
                    r_yaw = -yaw_weight * 0.5 * np.tanh(pos_err_hover)

                    # Yaw error: reward for facing the focal point
                    rpy = self._quat_to_rpy(quat)
                    yaw_err = abs(self._wrap_angle(rpy[2] - self._current_target_yaw))
                    r_yaw -= yaw_weight * np.tanh(yaw_err)

                    # Extreme yaw penalty
                    if yaw_err > np.pi / 2:
                        r_yaw -= 1.0 * yaw_weight

                    # Attitude stability: keep level (not spinning)
                    r_yaw -= yaw_weight * 0.3 * np.tanh(np.linalg.norm(rpy[:2]))
                else:
                    # Normal: drone flies figure-8, yaw gaze at focal
                    rpy = self._quat_to_rpy(quat)
                    yaw_err = abs(self._wrap_angle(rpy[2] - self._current_target_yaw))
                    r_yaw_gaze = -yaw_weight * np.tanh(yaw_err)

                    if yaw_err > np.pi / 2:
                        r_yaw_gaze -= 1.0 * yaw_weight

                    target_vel = self._target_velocity
                    target_speed_val = np.linalg.norm(target_vel)
                    if target_speed_val > 0.01 and np.linalg.norm(vel) > 0.01:
                        target_dir = target_vel / target_speed_val
                        drone_dir = vel / np.linalg.norm(vel)
                        direction_alignment = np.dot(drone_dir, target_dir)
                        r_yaw_gaze += 0.5 * yaw_weight * direction_alignment

                    r_yaw = r_yaw_gaze

        # CONVERGENCE-PREDICTIVE TRACKING (CPT) REWARD (Stage 7):
        r_cpt = 0.0
        if self.curriculum_stage == 7 and self._use_moving_target:
            if self._yaw_only_mode:
                # Yaw-only mode: drone hovers, CPT tracks whether drone points at FUTURE focal position
                # Convergence: does yaw rotation align with the direction to the focal?
                rpy = self._quat_to_rpy(quat)
                yaw_current = rpy[2]
                yaw_to_focal = self._current_target_yaw

                # Is the drone's yaw heading toward the focal?
                yaw_err = abs(self._wrap_angle(yaw_current - yaw_to_focal))
                # Cosine of yaw alignment: 1=facing focal, -1=facing away
                cos_yaw_alignment = np.cos(yaw_err)

                # Predictive: will the future focal be in the drone's field of view?
                # At T+100ms, 200ms, 300ms where will the focal be?
                future_yaw_errors = []
                for tau in [0.1, 0.2, 0.3]:
                    t_future = self._target_traj_time + tau * self._target_speed
                    a = self._figure8_amplitude
                    fdx = a * np.sin(t_future)
                    fdy = a * np.sin(2 * t_future) / 2
                    focal_future = self._target_initial_pos + np.array(
                        [fdx, fdy, 0.3 * np.sin(t_future / 3)]
                    )
                    yaw_future = np.arctan2(focal_future[1], focal_future[0])
                    future_yaw_errors.append(
                        abs(self._wrap_angle(yaw_current - yaw_future))
                    )

                r_convergence = 1.5 * np.tanh(
                    cos_yaw_alignment
                )  # positive = facing focal
                r_predictive = 2.0 * np.exp(
                    -np.mean(future_yaw_errors) / 1.0
                )  # lower future yaw error = better
                r_closure = 0.0  # no position closure in hover mode
                r_cpt = r_convergence + r_predictive
            else:
                # Normal: drone flies figure-8 with CPT reward
                if pos_err > 0.01:
                    err_dir = (self.target_pos - pos) / pos_err
                    convergence = np.dot(vel, err_dir)
                else:
                    convergence = 0.0
                r_convergence = 1.5 * np.tanh(convergence / 2.0)

                pred_errors = []
                for tau, future_target in [
                    (0.1, self._future_target_100ms),
                    (0.2, self._future_target_200ms),
                    (0.3, self._future_target_300ms),
                ]:
                    pred_pos = pos + vel * tau
                    pred_errors.append(np.linalg.norm(pred_pos - future_target))
                r_predictive = 2.0 * np.exp(-np.mean(pred_errors) / 1.0)

                if self._prev_cpt_pos_err is not None:
                    error_reduction = self._prev_cpt_pos_err - pos_err
                    r_closure = 2.0 * np.tanh(error_reduction / 0.5)
                else:
                    r_closure = 0.0
                self._prev_cpt_pos_err = pos_err

                r_cpt = r_convergence + r_predictive + r_closure

        # Roll/pitch torque penalty: penalize extreme angular torques that cause flips
        r_torque = -0.2 * (action[1] ** 2 + action[2] ** 2)

        return (
            r_alive
            + r_pos
            + r_att
            + r_vel
            + r_rate
            + r_smooth
            + r_proximity
            + r_alignment
            + r_success
            + r_recovery
            + r_jerk
            + r_track
            + r_stability
            + r_torque
            + r_racing
            + r_yaw
            + r_cpt
        )

    def _get_obs(self):
        """Get combined observation (deployable + privileged)."""
        raw = self._get_obs_raw()
        return self._get_obs_from_raw(raw)

    def _get_obs_raw(self):
        """Get raw observation components.

        Deployable observations (52 dims) use only sensor data available on real hardware:
        - Body-frame IMU acceleration (not world-frame)
        - Estimated mass deviation (not ground-truth)
        """
        pos = self.data.qpos[:3]
        vel = self.data.qvel[:3]
        quat = self.data.qpos[3:7]
        ang_vel = self.data.qvel[3:6]

        # Position error
        pos_err = self.target_pos - pos
        vel_err = -vel

        # Rotation matrix and RPY
        rotmat = self._quat_to_rotmat(quat)
        rpy = self._quat_to_rpy(quat)
        att_err = -rpy  # Full 3D attitude error
        rate_err = -ang_vel

        # Linear acceleration in BODY FRAME (what an IMU measures)
        # This is specific force: R^T * (a_world - g_world)
        lin_accel = self._get_body_accel(rotmat)

        # Mass estimate: normalized deviation from nominal
        # 0 = nominal mass, positive = heavy, negative = light
        mass_est = (self._mass_hat - self.nominal_mass) / self.nominal_mass

        return {
            "pos_err": pos_err,
            "vel_err": vel_err,
            "att_err": att_err,
            "rate_err": rate_err,
            "lin_accel": lin_accel,
            "rotmat": rotmat.flatten(),
            "ang_vel": ang_vel,
            "action_hist": self.action_history.flatten(),  # 16 dims
            "error_integrals": np.array(  # 4 dims
                [*self.pos_integral, self.yaw_integral], dtype=np.float32
            ),
            "rotor_thrust": self.rotor_thrusts,  # 4 dims
            "mass_est": np.array([mass_est], dtype=np.float32),
            "target_vel": self._target_velocity,  # 3 dims for Stage 5+ predictive tracking
            "yaw_error": np.float32(self._wrap_angle(rpy[2] - self._current_target_yaw))
            if self.curriculum_stage == 7
            else np.float32(0.0),
            "target_yaw": np.float32(self._current_target_yaw)
            if self.curriculum_stage == 7
            else np.float32(0.0),
        }

    def _get_obs_from_raw(self, raw):
        """Convert raw observation to stage-dependent vector.

        Stages 1-4: 60 dims (51 deployable + 9 privileged)
        Stages 5-6: 63 dims (54 deployable + 9 privileged)
        Stage 7:    75 dims (66 deployable + 9 privileged)
          - 54 base + 2 yaw error/target_yaw + 9 future targets
        """
        # Deployable obs - sensor data available on real hardware
        # NOTE: mass_est removed - network must infer mass from action history + error integrals
        deployable = np.concatenate(
            [
                raw["pos_err"],  # 3
                raw["vel_err"],  # 3
                raw["att_err"],  # 3
                raw["rate_err"],  # 3
                raw["lin_accel"],  # 3
                raw["rotmat"],  # 9
                raw["ang_vel"],  # 3
                raw["action_hist"],  # 16
                raw["error_integrals"],  # 4
                raw["rotor_thrust"],  # 4
                raw["target_vel"],  # 3 (zeros for stages 1-4, computed for stage 5+)
                # mass_est removed - was 1 dim
            ]
        )
        # Base: 3+3+3+3+3+9+3+16+4+4+3 = 54 dims

        # Stage 7: add yaw error, target yaw, yaw rate, and future target positions
        if self.curriculum_stage == 7:
            yaw_obs = np.array(
                [
                    raw.get("yaw_error", 0.0),  # 1: yaw error (wrapped to [-pi, pi])
                    raw.get("target_yaw", 0.0),  # 1: current target yaw
                    np.float32(
                        self._target_yaw_rate
                    ),  # 1: how fast the focal is sweeping (rad/s)
                ],
                dtype=np.float32,
            )
            future_targets = np.concatenate(
                [
                    self._future_target_100ms,  # 3
                    self._future_target_200ms,  # 3
                    self._future_target_300ms,  # 3
                ]
            )
            deployable = np.concatenate([deployable, yaw_obs, future_targets])
            # Stage 7: 54 + 3 + 9 = 66 deployable dims

        # Privileged obs (9 dims) - sim-only, used by critic during training
        privileged = self.get_privileged_info()

        return np.concatenate([deployable, privileged]).astype(np.float32)

    def get_privileged_info(self):
        """Get 9-dim privileged information (critic-only during training).

        NOTE: mass_est is privileged, not deployable. The actor cannot see
        the mass estimate directly. This prevents sim-to-real transfer issues
        where the network exploits perfect mass estimation in simulation.

        The network must infer mass changes from Action History and Error Integrals
        implicitly - a heavy drone requires larger action history to maintain the
        same vertical error integral.
        """
        mass_ratio = self.payload_mass / self.nominal_mass
        motor_deg = 1.0
        mass_est = (self._mass_hat - self.nominal_mass) / self.nominal_mass

        return np.array(
            [
                mass_ratio,
                self.com_shift[0],
                self.com_shift[1],
                self.com_shift[2],
                self.wind_force[0],
                self.wind_force[1],
                self.wind_force[2],
                motor_deg,
                mass_est,
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        """Get info dict."""
        quat = self.data.qpos[3:7]
        rpy = self._quat_to_rpy(quat)

        return {
            "pos": self.data.qpos[:3].copy(),
            "quat": quat.copy(),
            "rpy": rpy.copy(),
            "mass": self.payload_mass,
            "mass_est": self._mass_hat,
            "drop_occurred": self.drop_occurred,
        }

    def _quat_to_rotmat(self, quat):
        """Convert quaternion to rotation matrix."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def _quat_to_rpy(self, quat):
        """Convert quaternion to roll-pitch-yaw."""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])

    def _wrap_angle(self, angle):
        """Wrap angle to [-pi, pi]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _get_body_accel(self, rotmat):
        """Project world acceleration to body frame, accounting for gravity.

        IMU measures specific force in the body frame:
        specific_force = R^T * (acceleration_world - gravity_world)

        This correctly handles tilt: when the drone pitches, gravity
        projects through the rotation matrix into body x/y/z axes,
        not just body-z. Subtracting scalar 9.81 from z-accel alone
        would cause mass estimation spikes during tilt.
        """
        # World-frame acceleration (from MuJoCo qacc)
        accel_world = self.data.qacc[:3].copy()
        # Gravity in world frame
        gravity_world = np.array([0.0, 0.0, -9.81])
        # Specific force (what IMU measures), rotated to body frame
        specific_force_world = accel_world - gravity_world
        body_accel = rotmat.T @ specific_force_world
        return body_accel
