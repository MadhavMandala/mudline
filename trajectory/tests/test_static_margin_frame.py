"""The centre of pressure and the centre of gravity share a frame.

The defect, now fixed
---------------------
Centre of pressure and centre of gravity reached ``RasaeroAeroModel`` through
different frames. ``trajectory.frames.station_to_body`` maps a station ``s``
to body coordinates as ``y = -s`` -- stations increase aft, body +Y points
forward. The CG arrived that way. The CP did not: the model built it as
``nose + body_axis * x_cp``, which put it ``x_cp`` *ahead* of the nose.

Two consequences, one hiding the other. The moment arm pointed the wrong
way, so every statically stable vehicle carried a diverging aerodynamic
moment. And the damping estimate, which squares the arm, used
``x_cp + x_cg`` instead of ``x_cp - x_cg`` -- on the basic rocket 2.67 m
instead of 0.17 m, some 250 times too much damping. That froze the attitude,
and a thrusting rocket with a frozen attitude flies along its own axis, so
alpha collapsed to zero within a second of leaving the rail and nothing looked
wrong. No flown trajectory validated stability; none weathercocked.

Fixing the arm alone exposed a second error it had been cancelling: the
simulator read ``quat_to_dcm`` as inertial-to-body and used its transpose,
while ``quaternion_derivative`` propagates the standard body-to-inertial
quaternion, so the attitude rotated *opposite* to the body rate. With the
arm corrected and the kinematics still reversed, the basic rocket tumbled to
90 deg alpha and a 376 m apogee. See ``test_kinematics`` for the sense now
pinned, and ``trajectory.sim.launch_rail`` for the initial attitude that
had been conjugated to match the wrong reading.

Measured on the basic rocket in an 8 m/s crosswind, dt 0.02 s. Before: the
static moment at 0.27 deg alpha opposed the restoring direction, and alpha
read 0.86, 0.05, 0.01 deg at 0.2, 0.5 and 1.0 s after rail exit -- a frozen
vehicle. After both fixes: the moment restores, and alpha reads 11.4 deg at
the rail, 4.4, 2.4 and 0.2 deg at 0.2, 0.5 and 1.0 s -- a damped weathercock,
as a finned rocket should. ``parametric.tests.test_flight`` flies that case.

The convention decided with the fix: ``AeroForces.static_margin_m`` is
CP minus CG measured *aft*, positive for a stable vehicle -- the same sign the
status bar and the aero report quote in calibres.
"""

from __future__ import annotations

import numpy as np
import pytest

from trajectory.frames import station_to_body
from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase
from trajectory.vehicle.aero_model import RasaeroAeroModel

BODY_AXIS = np.array([0.0, 1.0, 0.0])


def _model(cp_station: float) -> RasaeroAeroModel:
    rows = [
        AeroCoefficients(mach=m, alpha_deg=a, cd=0.4, cn=0.09 * a, cm=0.0,
                         x_cp_m=cp_station)
        for m in (0.0, 1.0, 3.0) for a in (0.0, 5.0, 10.0)
    ]
    return RasaeroAeroModel(
        AeroDatabase(rows, reference_length_m=2.0),
        reference_area_m2=0.01, reference_length_m=2.0, body_axis=BODY_AXIS,
    )


def _flow(alpha_deg: float, speed: float = 100.0) -> np.ndarray:
    a = np.radians(alpha_deg)
    return np.array([speed * np.sin(a), speed * np.cos(a), 0.0])


def test_station_to_body_negates_the_station():
    """The premise: the bridge flips sign, so anything bypassing it is wrong."""
    assert station_to_body(2.5)[1] == pytest.approx(-2.5)
    assert station_to_body(0.0)[1] == pytest.approx(0.0)


def test_cp_and_cg_share_a_frame():
    """CG at station 1.0 m, CP at 1.3 m: stable by 0.3 m, and reported so."""
    forces = _model(cp_station=1.3).forces_and_moments(
        _flow(4.0), 1.2, 340.0, station_to_body(1.0),
    )
    assert forces.static_margin_m == pytest.approx(0.3, abs=1e-9)
    assert forces.cp_body_m[1] == pytest.approx(-1.3)


def test_a_stable_vehicle_is_restored_toward_the_wind():
    """The moment must rotate the nose toward the relative wind, not away."""
    v_body = _flow(4.0)
    forces = _model(cp_station=1.3).forces_and_moments(
        v_body, 1.2, 340.0, station_to_body(1.0),
    )
    # Rotating the body axis toward the flow is a rotation about axis x flow.
    toward_wind = np.cross(BODY_AXIS, v_body)
    assert np.dot(forces.static_moment_body_nm, toward_wind) > 0.0


def test_an_unstable_vehicle_is_reported_negative_and_diverges():
    """CP ahead of CG: margin negative, moment pushes the nose away."""
    v_body = _flow(4.0)
    forces = _model(cp_station=0.7).forces_and_moments(
        v_body, 1.2, 340.0, station_to_body(1.0),
    )
    assert forces.static_margin_m == pytest.approx(-0.3, abs=1e-9)
    assert np.dot(forces.static_moment_body_nm, np.cross(BODY_AXIS, v_body)) < 0.0


def test_damping_arm_is_the_margin_not_the_sum():
    """Squared on x_cp - x_cg. The sum gave hundreds of times too much."""
    model = _model(cp_station=1.3)
    forces = model.forces_and_moments(_flow(4.0), 1.2, 340.0, station_to_body(1.0))
    expected = model.estimate_cmq(forces.mach, forces.alpha_deg, 0.3)
    assert forces.cmq == pytest.approx(expected)
    assert abs(forces.cmq) < abs(model.estimate_cmq(forces.mach, forces.alpha_deg, 2.3))
