"""Tests for quaternion kinematics and apogee arming.

Two defects found by running the pipeline end to end on a real vehicle:

* the quaternion derivative was computed by taking a unit-length step with
  ``propagate_quaternion`` and subtracting, but that function renormalises, so
  the result carried a stray norm-correction term;
* the apogee event fired at t = 0, because a vehicle at rest on the rail has
  exactly zero vertical velocity -- which terminated the ascent phase
  immediately and flew the whole flight under the drogue.

Runs under pytest, and standalone via
``python trajectory/tests/test_kinematics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.eom.kinematics import (  # noqa: E402
    propagate_quaternion,
    quat_to_dcm,
    quaternion_derivative,
    quaternion_rate_matrix,
)
from trajectory.sim.events import EventDetector  # noqa: E402


def _omega_matrix(omega):
    p, q, r = omega
    return np.array([
        [0, -p, -q, -r],
        [p, 0, r, -q],
        [q, -r, 0, p],
        [r, q, -p, 0],
    ])


# ------------------------------------------------------- quaternion derivative


def test_derivative_matches_the_defining_formula():
    q = np.array([0.9239, 0.0, 0.3827, 0.0])
    q /= np.linalg.norm(q)
    omega = np.array([0.1, -0.4, 0.25])
    assert np.allclose(quaternion_derivative(q, omega), 0.5 * _omega_matrix(omega) @ q)


def test_rate_matrix_is_skew_symmetric():
    """Which is what makes the derivative norm-preserving."""
    m = quaternion_rate_matrix([0.3, -0.2, 0.7])
    assert np.allclose(m, -m.T)


def test_derivative_is_norm_preserving():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for omega in [[0.0, 0.0, 0.05], [0.5, -0.3, 0.9], [2.0, 2.0, 2.0]]:
        assert abs(float(np.dot(q, quaternion_derivative(q, np.array(omega))))) < 1e-12


def test_derivative_is_exactly_linear_in_omega():
    q = np.array([0.7071, 0.7071, 0.0, 0.0])
    omega = np.array([0.2, 0.1, -0.3])
    assert np.allclose(
        quaternion_derivative(q, 3.0 * omega), 3.0 * quaternion_derivative(q, omega)
    )


def test_finite_difference_of_propagate_is_not_the_derivative():
    """Pins the old bug: renormalisation makes the difference quotient wrong.

    The error grows with rate, and always points along -q, so it shrinks the
    quaternion rather than rotating it.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])
    for omega_z, floor in [(0.15, 0.02), (1.0, 0.15), (2.0, 0.3)]:
        omega = np.array([0.0, 0.0, omega_z])
        exact = quaternion_derivative(q, omega)
        naive = propagate_quaternion(q, omega, 1.0) - q
        relative = np.linalg.norm(naive - exact) / np.linalg.norm(exact)
        assert relative > floor, (omega_z, relative)
        assert naive[0] < 0.0, "the stray term should shrink the scalar part"


def test_small_step_propagation_agrees_with_the_derivative():
    """They must converge as dt -> 0, which is why the bug was easy to miss."""
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.4])
    exact = quaternion_derivative(q, omega)
    dt = 1e-6
    numeric = (propagate_quaternion(q, omega, dt) - q) / dt
    assert np.allclose(numeric, exact, atol=1e-6)


def test_derivative_integrates_to_the_right_rotation_angle_and_sense():
    """Integrating a constant body rate turns through |omega| * t, right-handed.

    The sense is asserted, not just the magnitude. This docstring used to
    say pinning it would "only encode one reading of the convention" and
    that the closed loop -- a stable vehicle weathercocking into the flow --
    was checked elsewhere. It was not: the test it named integrated the
    attitude by hand and never touched the quaternion. The combination the
    simulator actually used (this derivative with the *transpose* of
    ``quat_to_dcm`` as body-to-inertial) rotated the vehicle backwards.

    ``quat_to_dcm`` is body-to-inertial. A positive rate about body z, from
    identity, must carry body +X toward inertial +Y.
    """
    q = np.array([1.0, 0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.5])
    dt = 1e-4
    for _ in range(20000):                      # 2 s at 0.5 rad/s -> 1.0 rad
        q = q + quaternion_derivative(q, omega) * dt
        q /= np.linalg.norm(q)

    rotated = quat_to_dcm(q) @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(rotated, [np.cos(1.0), np.sin(1.0), 0.0], atol=1e-3), rotated
    assert np.isclose(np.linalg.norm(rotated), 1.0, atol=1e-9)
    assert abs(rotated[2]) < 1e-9, "a body-z rate must not tilt the vehicle out of plane"


def test_dcm_is_body_to_inertial():
    """A quarter turn about +z carries body +Y to inertial -X."""
    q = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    assert np.allclose(quat_to_dcm(q) @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], atol=1e-12)


def test_integration_agrees_with_repeated_small_step_propagation():
    """The derivative and the stepper must describe the same motion."""
    omega = np.array([0.1, 0.0, 0.4])
    dt = 1e-4
    steps = 5000

    q_derivative = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(steps):
        q_derivative = q_derivative + quaternion_derivative(q_derivative, omega) * dt
        q_derivative /= np.linalg.norm(q_derivative)

    q_stepped = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(steps):
        q_stepped = propagate_quaternion(q_stepped, omega, dt)

    assert np.allclose(q_derivative, q_stepped, atol=1e-8)


# ------------------------------------------------------------- apogee arming


def _state(altitude, vy):
    s = np.zeros(14)
    s[1], s[4] = altitude, vy
    return s


def test_apogee_does_not_fire_on_the_pad():
    """A vehicle at rest has vy = 0, which is a legitimate root of the old event."""
    event = EventDetector.apogee()
    assert event(0.0, _state(0.0, 0.0)) > 0.0


def test_apogee_is_suppressed_below_the_arming_altitude():
    event = EventDetector.apogee(arm_altitude_m=10.0)
    assert event(0.0, _state(5.0, -50.0)) > 0.0     # descending, but not armed


def test_apogee_fires_once_armed():
    event = EventDetector.apogee(arm_altitude_m=10.0)
    assert event(0.0, _state(500.0, 20.0)) > 0.0    # climbing
    assert event(0.0, _state(500.0, -20.0)) < 0.0   # descending


def test_no_zero_crossing_while_climbing_through_the_arming_altitude():
    """The guard is a jump, but never one the root finder can trip on."""
    event = EventDetector.apogee(arm_altitude_m=10.0)
    values = [event(0.0, _state(h, 60.0)) for h in np.linspace(0.0, 40.0, 200)]
    assert all(v > 0.0 for v in values)


def test_apogee_event_is_terminal_and_downward():
    event = EventDetector.apogee()
    assert event.terminal and event.direction == -1


# ------------------------------------------------------------------ fog range
