"""Translational equations of motion: F = ma in inertial frame."""

import numpy as np


def TranslationalEOM(state: np.ndarray, forces: np.ndarray, mass: float) -> np.ndarray:
    """
    Compute translational acceleration in inertial frame.

    Args:
        state: [x, y, z, vx, vy, vz] - position and velocity [m, m/s]
        forces: Total force vector in inertial frame [N]
        mass: Current mass [kg]

    Returns:
        Derivative array [vx, vy, vz, ax, ay, az]
    """
    acc = forces / mass
    return np.array([state[3], state[4], state[5], acc[0], acc[1], acc[2]])
