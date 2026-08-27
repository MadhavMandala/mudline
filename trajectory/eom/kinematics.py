"""Quaternion kinematics and rotation utilities."""

import numpy as np


def quat_to_dcm(q: np.ndarray) -> np.ndarray:
    """Rotation matrix of the attitude quaternion [w, x, y, z].

    Maps body-frame vectors into the inertial frame -- ``R @ v_body`` -- for
    a right-handed rotation, which is the same quaternion
    ``quaternion_derivative`` propagates as ``dq/dt = 0.5 * q (x) omega``.
    Inertial-to-body is the transpose.

    The simulator used to read this matrix the other way round and use its
    transpose as body-to-inertial, while the derivative propagated the
    standard quaternion. The attitude then rotated *opposite* to the body
    rate: a moment that should have turned the nose into the wind turned it
    away, and a statically stable vehicle diverged. It went unnoticed for as
    long as it did because a second sign error in the aero model made the
    static moment diverge too, and the two cancelled.
    """
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ])


def quaternion_rate_matrix(omega: np.ndarray) -> np.ndarray:
    """The 4x4 matrix Omega(w) in dq/dt = 0.5 * Omega(w) * q."""
    p, q_r, r = omega
    return np.array([
        [0,   -p,   -q_r, -r],
        [p,    0,    r,   -q_r],
        [q_r, -r,    0,    p],
        [r,    q_r, -p,    0]
    ])


def quaternion_derivative(q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Exact quaternion derivative dq/dt = 0.5 * Omega(w) * q.

    Use this inside a derivative function. Do **not** compute a derivative by
    taking a finite step with ``propagate_quaternion`` and subtracting: that
    function renormalises before returning, so the difference carries a
    spurious norm-correction term. The error is second order in |w|*dt, and
    with the unit step the simulator used it reached 3.7% at 0.15 rad/s, 24% at
    1 rad/s and 41% at 2 rad/s. Worse than the magnitude, the stray term always
    points along -q, so it steadily shrinks the quaternion norm -- an
    artificial damping the integrator then has to fight.

    This form is exactly norm-preserving: q . dq/dt is identically zero,
    because Omega(w) is skew-symmetric.
    """
    return 0.5 * quaternion_rate_matrix(omega) @ np.asarray(q, dtype=float)


def propagate_quaternion(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """
    Propagate quaternion using angular velocity.
    Uses first-order integration.

    Args:
        q: Quaternion [w, x, y, z]
        omega: Angular velocity in body frame [rad/s]
        dt: Time step [s]

    Returns:
        Updated quaternion (normalized)
    """
    p, q_r, r = omega
    # Quaternion derivative: dq/dt = 0.5 * Q(omega) * q
    omega_mat = np.array([
        [0,   -p,   -q_r, -r],
        [p,    0,    r,   -q_r],
        [q_r, -r,    0,    p],
        [r,    q_r, -p,    0]
    ])
    q_dot = 0.5 * omega_mat @ q
    q_new = q + q_dot * dt
    return q_new / np.linalg.norm(q_new)


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to quaternion."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    return np.array([
        cr*cp*cy + sr*sp*sy,
        sr*cp*cy - cr*sp*sy,
        cr*sp*cy + sr*cp*sy,
        cr*cp*sy - sr*sp*cy
    ])
