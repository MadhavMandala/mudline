"""Tests for aerodynamic damping.

Pins the defect where the aero model produced only a static restoring moment.
With a restoring moment and no damping, the rotational equations describe a
frictionless pendulum: a disturbance oscillates forever instead of decaying.

Runs under pytest, and standalone via
``python trajectory/tests/test_aero_damping.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.eom.rotational import RotationalEOM  # noqa: E402
from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase  # noqa: E402
from trajectory.vehicle.aero_model import RasaeroAeroModel  # noqa: E402


BODY_AXIS = np.array([0.0, 1.0, 0.0])


def _model(cmq=None, clp=0.0) -> RasaeroAeroModel:
    """A stable vehicle: CP 0.5 m behind the CG we will pass in."""
    rows = []
    for mach in [0.0, 0.5, 1.0, 2.0, 5.0]:
        for alpha in [0.0, 5.0, 10.0, 15.0]:
            rows.append(AeroCoefficients(
                mach=mach, alpha_deg=alpha,
                cd=0.4, cn=0.09 * alpha, cm=0.0, x_cp_m=2.0,
            ))
    db = AeroDatabase(rows, reference_length_m=3.0)
    return RasaeroAeroModel(
        db, reference_area_m2=0.07, reference_length_m=3.0,
        body_axis=BODY_AXIS, cmq=cmq, clp=clp,
    )


def _flow(alpha_deg: float = 4.0, speed: float = 200.0) -> np.ndarray:
    """Velocity in the body frame at a given angle of attack."""
    a = np.radians(alpha_deg)
    return np.array([speed * np.sin(a), speed * np.cos(a), 0.0])


# Body +Y is forward and stations run aft, so the tabulated CP at station
# 2.0 m sits at y = -2.0. A CG at station 1.5 puts it 0.5 m ahead of the CP:
# a stable vehicle, by half a metre.
CG = np.array([0.0, -1.5, 0.0])


def test_cn_alpha_slope_is_recovered_from_the_table():
    db = _model().database
    # cn = 0.09 per degree -> 0.09 * 180/pi per radian.
    assert np.isclose(db.cn_alpha_per_rad(1.0, 0.0), 0.09 * 180.0 / np.pi, rtol=1e-6)


def test_cn_alpha_is_nonzero_at_zero_alpha():
    """A central difference across the |alpha| fold would return 0 here."""
    assert _model().database.cn_alpha_per_rad(1.0, 0.0) > 1.0


def test_estimated_cmq_is_negative():
    model = _model()
    assert model.estimate_cmq(1.0, 4.0, static_margin_m=-0.5) < 0.0


def test_cmq_grows_with_static_margin_squared():
    model = _model()
    near = abs(model.estimate_cmq(1.0, 4.0, static_margin_m=-0.5))
    far = abs(model.estimate_cmq(1.0, 4.0, static_margin_m=-1.0))
    assert np.isclose(far / near, 4.0, rtol=1e-9)


def test_damping_opposes_the_rotation_rate():
    model = _model()
    omega = np.array([0.0, 0.0, 0.6])          # pitch rate about z
    aero = model.forces_and_moments(_flow(), 1.0, 340.0, CG, omega)
    # The damping component must point against omega.
    assert np.dot(aero.damping_moment_body_nm, omega) < 0.0


def test_damping_is_absent_without_omega():
    aero = _model().forces_and_moments(_flow(), 1.0, 340.0, CG)
    assert np.allclose(aero.damping_moment_body_nm, 0.0)
    assert np.allclose(aero.moment_body_nm, aero.static_moment_body_nm)


def test_damping_is_linear_in_rate():
    model = _model()
    m1 = model.forces_and_moments(_flow(), 1.0, 340.0, CG, np.array([0, 0, 0.2]))
    m2 = model.forces_and_moments(_flow(), 1.0, 340.0, CG, np.array([0, 0, 0.4]))
    assert np.allclose(2 * m1.damping_moment_body_nm, m2.damping_moment_body_nm)


def test_damping_vanishes_with_airspeed_no_singularity():
    """q_dyn carries V^2 and the reduced rate carries 1/V: the product is linear."""
    model = _model()
    magnitudes = []
    for speed in [100.0, 10.0, 1.0, 0.1, 0.01]:
        aero = model.forces_and_moments(
            _flow(speed=speed), 1.0, 340.0, CG, np.array([0.0, 0.0, 0.5])
        )
        magnitudes.append(np.linalg.norm(aero.damping_moment_body_nm))
    assert np.all(np.isfinite(magnitudes))
    assert magnitudes == sorted(magnitudes, reverse=True)
    assert magnitudes[-1] < 1e-3


def test_roll_damping_is_off_unless_supplied():
    roll = np.array([0.0, 0.7, 0.0])           # rate about the body axis
    undamped = _model().forces_and_moments(_flow(), 1.0, 340.0, CG, roll)
    assert np.allclose(np.dot(undamped.damping_moment_body_nm, BODY_AXIS), 0.0)

    damped = _model(clp=-0.02).forces_and_moments(_flow(), 1.0, 340.0, CG, roll)
    assert np.dot(damped.damping_moment_body_nm, BODY_AXIS) < 0.0


def test_explicit_cmq_overrides_the_estimate():
    model = _model(cmq=-9.0)
    assert model.estimate_cmq(1.0, 4.0, static_margin_m=-0.5) == -9.0


def _swing(model, damped: bool, steps: int = 4000, dt: float = 0.002):
    """Integrate free rotation about the CG and return the pitch-rate history."""
    inertia = np.diag([40.0, 4.0, 40.0])
    inertia_inv = np.linalg.inv(inertia)
    omega = np.array([0.0, 0.0, 0.35])
    attitude = 0.0        # pitch angle relative to the flow [rad]
    history = []
    for _ in range(steps):
        v_body = _flow(alpha_deg=np.degrees(attitude))
        aero = model.forces_and_moments(
            v_body, 1.0, 340.0, CG, omega if damped else None
        )
        omega = omega + RotationalEOM(
            omega, aero.moment_body_nm, inertia, inertia_inv
        ) * dt
        attitude += omega[2] * dt
        history.append(abs(omega[2]))
    return np.array(history)


def test_undamped_oscillation_does_not_decay():
    """Characterises the old behaviour, so the contrast below is meaningful."""
    history = _swing(_model(), damped=False)
    early = history[:500].max()
    late = history[-500:].max()
    assert late > 0.8 * early, f"expected a persistent swing, got {late}/{early}"


def test_damping_decays_the_oscillation():
    history = _swing(_model(), damped=True)
    early = history[:500].max()
    late = history[-500:].max()
    assert late < 0.5 * early, f"oscillation did not decay: {late}/{early}"


def test_damping_removes_rotational_energy_monotonically():
    """Peak rate must never grow when damping is the only moment acting."""
    history = _swing(_model(), damped=True)
    windows = [history[i:i + 400].max() for i in range(0, 3600, 400)]
    assert all(b <= a + 1e-9 for a, b in zip(windows, windows[1:])), windows


if __name__ == "__main__":
    failures = 0
    names = sorted(n for n in globals() if n.startswith("test_"))
    for name in names:
        try:
            globals()[name]()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(names) - failures}/{len(names)} passed")
    raise SystemExit(1 if failures else 0)
