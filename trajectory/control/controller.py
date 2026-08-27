"""Basic attitude controller stub."""

import numpy as np


class Controller:
    """Base controller interface."""

    def __init__(self):
        self.active = False

    def compute(self, state: dict, target: dict, dt: float) -> tuple:
        """
        Compute control outputs.

        Args:
            state: Current state dict (pos, vel, quat, omega)
            target: Target state dict
            dt: Time step

        Returns:
            (pitch_gimbal, yaw_gimbal) in radians
        """
        return (0.0, 0.0)


class PIDController(Controller):
    """PID controller for pitch/yaw control."""

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.1):
        super().__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error: float, dt: float) -> float:
        """Compute single-axis PID output."""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)

        return np.clip(output, -np.radians(8), np.radians(8))
