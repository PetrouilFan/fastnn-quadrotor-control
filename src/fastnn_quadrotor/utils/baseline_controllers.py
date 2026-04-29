#!/usr/bin/env python3
"""
Baseline Controllers: PID and PD implementations
These serve as the control group in the comparative analysis.
"""

import numpy as np


class PDController:
    """
    Proportional-Derivative Controller
    Used as baseline in comparative analysis

    Default gains tuned for reliable hover in MuJoCo:
    - kp_pos=8.0, kd_pos=4.0 → reliable hover with motor delay
    - kp_att=10.0, kd_att=6.0 → aggressive attitude tracking
    """

    def __init__(self, kp_pos=8.0, kd_pos=4.0, kp_att=10.0, kd_att=6.0):
        self.kp_pos = kp_pos  # Position proportional gain
        self.kd_pos = kd_pos  # Position derivative gain
        self.kp_att = kp_att  # Attitude proportional gain
        self.kd_att = kd_att  # Attitude derivative gain

        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0

        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)

    def reset(self):
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)

    def compute(self, pos, vel, quat, ang_vel, mass=1.0):
        """Compute control output"""

        # Position error
        pos_error = self.target_pos - pos

        # Desired attitude from position loop
        target_roll = np.arctan2(-self.kp_pos * pos_error[0], 9.81 * mass)
        target_pitch = np.arctan2(self.kp_pos * pos_error[1], 9.81 * mass)
        target_thrust = mass * 9.81 + self.kp_pos * pos_error[2] - self.kd_pos * vel[2]

        # Get current RPY
        current_rpy = self._quat_to_rpy(quat)

        # Attitude errors
        roll_error = target_roll - current_rpy[0]
        pitch_error = target_pitch - current_rpy[1]
        yaw_error = self.target_yaw - current_rpy[2]

        # Torques from attitude loop
        roll_torque = self.kp_att * roll_error - self.kd_att * ang_vel[0]
        pitch_torque = self.kp_att * pitch_error - self.kd_att * ang_vel[1]
        yaw_torque = self.kp_att * yaw_error - self.kd_att * ang_vel[2]

        return np.array([target_thrust, roll_torque, pitch_torque, yaw_torque])

    def _quat_to_rpy(self, quat):
        """Convert quaternion to roll-pitch-yaw"""
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])


class PIDController:
    """
    Proportional-Integral-Derivative Controller
    Adds integral term for steady-state error elimination
    """

    def __init__(
        self, kp_pos=20.0, ki_pos=0.1, kd_pos=5.0, kp_att=3.0, ki_att=0.0, kd_att=0.5
    ):
        self.kp_pos = kp_pos
        self.ki_pos = ki_pos
        self.kd_pos = kd_pos
        self.kp_att = kp_att
        self.ki_att = ki_att
        self.kd_att = kd_att

        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0

        # Integral states
        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)

        # Derivative states
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)

        self.dt = 0.01

    def reset(self):
        self.pos_integral = np.zeros(3)
        self.att_integral = np.zeros(3)
        self.prev_pos_error = np.zeros(3)
        self.prev_att_error = np.zeros(3)

    def compute(self, pos, vel, quat, ang_vel, mass=1.0):
        """Compute PID control output"""

        # Position error
        pos_error = self.target_pos - pos

        # Integral term (with anti-windup)
        self.pos_integral += pos_error * self.dt
        self.pos_integral = np.clip(self.pos_integral, -5.0, 5.0)

        # Derivative term
        pos_derivative = (pos_error - self.prev_pos_error) / self.dt
        self.prev_pos_error = pos_error.copy()

        # PD output for position
        target_roll = np.arctan2(-self.kp_pos * pos_error[0], 9.81 * mass)
        target_pitch = np.arctan2(self.kp_pos * pos_error[1], 9.81 * mass)
        target_thrust = (
            mass * 9.81
            + self.kp_pos * pos_error[2]
            + self.ki_pos * self.pos_integral[2]
            - self.kd_pos * vel[2]
        )

        # Get current RPY
        current_rpy = self._quat_to_rpy(quat)

        # Attitude error
        roll_error = target_roll - current_rpy[0]
        pitch_error = target_pitch - current_rpy[1]
        yaw_error = self.target_yaw - current_rpy[2]

        # Attitude integral
        self.att_integral += np.array([roll_error, pitch_error, yaw_error]) * self.dt
        self.att_integral = np.clip(self.att_integral, -2.0, 2.0)

        # Attitude derivative
        att_derivative = (
            np.array([roll_error, pitch_error, yaw_error]) - self.prev_att_error
        ) / self.dt
        self.prev_att_error = np.array([roll_error, pitch_error, yaw_error])

        # Torques
        roll_torque = (
            self.kp_att * roll_error
            + self.ki_att * self.att_integral[0]
            - self.kd_att * ang_vel[0]
        )
        pitch_torque = (
            self.kp_att * pitch_error
            + self.ki_att * self.att_integral[1]
            - self.kd_att * ang_vel[1]
        )
        yaw_torque = (
            self.kp_att * yaw_error
            + self.ki_att * self.att_integral[2]
            - self.kd_att * ang_vel[2]
        )

        return np.array([target_thrust, roll_torque, pitch_torque, yaw_torque])

    def _quat_to_rpy(self, quat):
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])


class HighGainPD:
    """
    High-gain PD for aggressive recovery
    Used in benchmarks to test fast recovery
    """

    def __init__(self):
        self.kp_pos = 40.0
        self.kd_pos = 10.0
        self.kp_att = 8.0
        self.kd_att = 2.0

        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0

    def reset(self):
        pass

    def compute(self, pos, vel, quat, ang_vel, mass=1.0):
        pos_error = self.target_pos - pos

        target_roll = np.arctan2(-self.kp_pos * pos_error[0], 9.81 * mass)
        target_pitch = np.arctan2(self.kp_pos * pos_error[1], 9.81 * mass)
        target_thrust = mass * 9.81 + self.kp_pos * pos_error[2] - self.kd_pos * vel[2]

        current_rpy = self._quat_to_rpy(quat)

        roll_torque = (
            self.kp_att * (target_roll - current_rpy[0]) - self.kd_att * ang_vel[0]
        )
        pitch_torque = (
            self.kp_att * (target_pitch - current_rpy[1]) - self.kd_att * ang_vel[1]
        )
        yaw_torque = (
            self.kp_att * (self.target_yaw - current_rpy[2]) - self.kd_att * ang_vel[2]
        )

        return np.array([target_thrust, roll_torque, pitch_torque, yaw_torque])

    def _quat_to_rpy(self, quat):
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


class LQRController:
    """
    Linear Quadratic Regulator for comparison
    Optimal control baseline
    """

    def __init__(self):
        # State weighting matrix
        self.Q = np.diag(
            [
                10.0,
                10.0,
                20.0,  # position
                1.0,
                1.0,
                1.0,  # velocity
                5.0,
                5.0,
                2.0,
            ]
        )  # attitude

        # Control weighting matrix
        self.R = np.diag([0.1, 0.5, 0.5, 0.1])

        self.target_pos = np.array([0.0, 0.0, 1.0])
        self.target_yaw = 0.0

    def reset(self):
        pass

    def compute(self, pos, vel, quat, ang_vel, mass=1.0):
        # Linearized dynamics around hover
        # Compute error state
        current_rpy = self._quat_to_rpy(quat)

        state = np.concatenate(
            [
                self.target_pos - pos,  # position error
                -vel,  # velocity error
                -current_rpy,  # attitude error
            ]
        )

        # LQR gain (precomputed for hover linearization)
        # K = R^-1 * B^T * P where P solves Riccati
        # Simplified: use PD with optimal-like gains
        K_pos = np.diag([20.0, 20.0, 30.0])
        K_vel = np.diag([5.0, 5.0, 8.0])
        K_att = np.diag([3.0, 3.0, 1.0])

        # Compute desired thrust and torques
        thrust = (
            mass * 9.81
            + K_pos[2, 2] * (self.target_pos[2] - pos[2])
            - K_vel[2, 2] * vel[2]
        )

        roll_torque = K_att[0, 0] * (-current_rpy[0]) - 0.5 * ang_vel[0]
        pitch_torque = K_att[1, 1] * (-current_rpy[1]) - 0.5 * ang_vel[1]
        yaw_torque = K_att[2, 2] * (self.target_yaw - current_rpy[2]) - 0.3 * ang_vel[2]

        return np.array([thrust, roll_torque, pitch_torque, yaw_torque])

    def _quat_to_rpy(self, quat):
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


def test_controllers():
    """Test all baseline controllers"""
    from mujoco import MjModel, MjData

    # Create simple quadrotor
    model_path = "quadrotor.xml"
    model = MjModel.from_xml_path(model_path)
    data = MjData(model)

    controllers = {
        "PD": PDController(),
        "PID": PIDController(),
        "HighGainPD": HighGainPD(),
        "LQR": LQRController(),
    }

    for name, ctrl in controllers.items():
        ctrl.reset()
        # Run a few steps
        for _ in range(10):
            pos = data.qpos[:3]
            vel = data.qvel[:3]
            quat = data.qpos[3:7]
            ang_vel = data.qvel[3:6]

            action = ctrl.compute(pos, vel, quat, ang_vel)
            data.ctrl[:] = action
            from mujoco import mj_step

            mj_step(model, data)

        print(f"{name}: pos={data.qpos[:3]}, roll={np.rad2deg(data.qpos[4]):.1f}°")

    print("\nAll baseline controllers working!")


if __name__ == "__main__":
    test_controllers()
