"""The rotational degrees of freedom against what must be conserved.

Runs under pytest, and standalone via
``python trajectory/tests/test_rotational_verification.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory import simulation as tm  # noqa: E402
from validation.imu import ImuLog, compare, load_imu_csv, simulated_imu, write_imu_csv  # noqa: E402
from validation.rotational import fit_pitch_oscillation, free_body, torque_free_tumble  # noqa: E402

INERTIA = np.diag([10.0, 2.0, 10.0])


def test_a_torque_free_tumble_conserves_momentum_and_energy():
    record = torque_free_tumble(INERTIA, [1.0, 0.3, 2.0], duration_s=10.0)
    momentum0 = record.momentum[0]
    drift = np.linalg.norm(record.momentum - momentum0, axis=1) / np.linalg.norm(momentum0)
    assert np.max(drift) < 1e-7
    assert np.max(np.abs(record.energy / record.energy[0] - 1.0)) < 1e-7
    assert np.max(np.abs(record.quaternion_norm - 1.0)) < 1e-6


def test_an_asymmetric_tumble_actually_tumbles():
    """The rate about the middle axis is unstable: it does not just spin."""
    record = torque_free_tumble(np.diag([10.0, 2.0, 6.0]), [0.01, 0.0, 3.0], duration_s=20.0)
    axis_z = np.array([r[:, 2] for r in record.dcm])          # body z in inertial
    assert np.min(axis_z @ axis_z[0]) < 0.0, "body z has turned past 90 degrees"
    drift = np.linalg.norm(record.momentum - record.momentum[0], axis=1) / np.linalg.norm(record.momentum[0])
    assert np.max(drift) < 1e-6


def test_a_steady_spin_is_the_exact_rotation():
    """Spinning about a principal axis the attitude is the rotation by ``p t``."""
    p = 20.0
    record = torque_free_tumble(INERTIA, [0.0, p, 0.0], duration_s=2.0, dt=0.05)
    for t, dcm in zip(record.time_s, record.dcm):
        c, s = np.cos(p * t), np.sin(p * t)
        expected = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        assert np.allclose(dcm, expected, atol=1e-7), t


def test_the_peak_fit_recovers_a_known_oscillation():
    t = np.linspace(0.0, 3.0, 3001)
    omega_n, zeta = 12.0, 0.08
    omega_d = omega_n * np.sqrt(1.0 - zeta ** 2)
    alpha = np.abs(5.0 * np.exp(-zeta * omega_n * t) * np.cos(omega_d * t))
    fit = fit_pitch_oscillation(t, alpha)
    assert fit.damped_frequency_radps == pytest.approx(omega_d, rel=0.01)
    assert fit.damping_ratio == pytest.approx(zeta, rel=0.05)


# ------------------------------------------------------------ the IMU path


def test_a_synthetic_imu_round_trips_through_the_comparison(tmp_path):
    sim = tm.RocketSimulation()
    result = sim.run(launch_elevation=np.radians(85.0), t_max=40.0, dt=0.02, rail_buttons_m=(1.0, 3.5))
    truth = simulated_imu(sim, result)
    assert truth.axial_g.max() > 3.0
    rng = np.random.default_rng(0)
    noisy = ImuLog(
        truth.time_s + 2.5,                                          # the recorder started early
        truth.gyro_radps + rng.normal(0.0, 0.01, truth.gyro_radps.shape),
        truth.accel_mps2 + rng.normal(0.0, 0.2, truth.accel_mps2.shape),
        source="synthetic",
    )
    path = write_imu_csv(noisy, tmp_path / "imu.csv")
    measured = load_imu_csv(path)
    comparison = compare(measured, truth)
    assert comparison.offset_s == pytest.approx(2.5, abs=0.03)
    assert comparison.rms_boost["axial_g"] < 0.05
    assert comparison.rms_boost["roll_rate_dps"] < 1.0
    assert "aligned at liftoff" in comparison.report()


def test_units_and_axes_are_honoured(tmp_path):
    t = np.array([0.0, 1.0])
    with open(tmp_path / "raw.csv", "w", encoding="utf-8") as f:
        f.write("t,p,q,r,fx,fy,fz\n0,57.29577951,0,0,1,0,0\n1,0,0,0,0,1,0\n")
    # IMU x points out the nose: its x is body y.
    axes = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    log = load_imu_csv(tmp_path / "raw.csv", time="t", gyro=("p", "q", "r"), accel=("fx", "fy", "fz"),
                       gyro_units="dps", accel_units="g", axes=axes)
    assert np.allclose(log.time_s, t)
    assert log.gyro_radps[0, 1] == pytest.approx(1.0)                # 57.3 dps about IMU x -> body y
    assert log.accel_mps2[0, 1] == pytest.approx(9.80665)


def test_a_free_body_falls_without_turning():
    sim = free_body(INERTIA)
    state = np.zeros(tm.STATE_SIZE)
    state[1], state[6] = 1000.0, 1.0
    result = sim._integrate_phase(state, 0.0, 5.0, 0.5, [])
    assert np.allclose(result.y[10:13], 0.0)
    assert np.allclose(result.y[6:10, -1], [1.0, 0.0, 0.0, 0.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
