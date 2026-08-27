"""Rate derivatives: jet damping, the inertia-rate term, table Cmq, and roll.

Runs under pytest, and standalone via
``python trajectory/tests/test_derivatives.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory import simulation as tm  # noqa: E402
from trajectory.eom.rotational import RotationalEOM  # noqa: E402
from trajectory.simulation import jet_damping_moment  # noqa: E402
from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase  # noqa: E402
from trajectory.vehicle.aero_model import RasaeroAeroModel  # noqa: E402

BODY_AXIS = np.array([0.0, 1.0, 0.0])


# ------------------------------------------------------------ jet damping


def test_jet_damping_opposes_transverse_rates_and_ignores_roll():
    r_exit = np.array([0.0, -2.0, 0.0])           # nozzle 2 m aft of the CG
    omega = np.array([0.3, 0.7, 0.0])             # x is transverse, y is roll
    moment = jet_damping_moment(5.0, r_exit, omega)
    assert np.isclose(moment[0], -5.0 * 4.0 * 0.3)
    assert np.isclose(moment[1], 0.0)
    assert np.isclose(moment[2], 0.0)


def test_no_mass_flow_means_no_jet_damping():
    assert np.allclose(jet_damping_moment(0.0, [0.0, -2.0, 0.0], [1.0, 1.0, 1.0]), 0.0)


def test_the_inertia_falls_while_the_motor_burns():
    sim = tm.RocketSimulation()
    state = np.zeros(tm.STATE_SIZE)
    state[1] = 1000.0
    state[6] = 1.0
    state[tm.PROP_IDX] = sim.mass_props.prop_mass
    burning = sim.evaluate(state, 5.0)
    assert burning.mass_flow_kgps > 0.0
    assert np.trace(burning.inertia_rate) < 0.0

    state[tm.PROP_IDX] = 0.0
    empty = sim.evaluate(state, 5.0)
    assert np.allclose(empty.inertia_rate, 0.0)
    assert np.allclose(empty.jet_damping_moment_body_nm, 0.0)


def test_the_force_model_applies_jet_damping_to_a_pitching_burner():
    sim = tm.RocketSimulation()
    sim.thrust_position_body_m = np.array([0.0, -1.5, 0.0])
    state = np.zeros(tm.STATE_SIZE)
    state[1] = 200_000.0                          # no air to speak of
    state[6] = 1.0
    state[10] = 0.2                               # pitching about body x
    state[tm.PROP_IDX] = sim.mass_props.prop_mass
    point = sim.evaluate(state, 5.0)
    _, cg, _ = sim.mass_props.at_propellant(sim.mass_props.prop_mass)
    r_exit = sim.thrust_position_body_m - cg
    expected = jet_damping_moment(point.mass_flow_kgps, r_exit, state[10:13])
    assert np.allclose(point.jet_damping_moment_body_nm, expected)
    assert point.jet_damping_moment_body_nm[0] < 0.0


def test_euler_with_falling_inertia_spins_a_free_body_up():
    """Angular momentum is kept as I shrinks: no moment, rising rate."""
    inertia = np.diag([10.0, 2.0, 10.0])
    rate = np.diag([-1.0, 0.0, -1.0])
    omega = np.array([0.5, 0.0, 0.0])
    omega_dot = RotationalEOM(omega, np.zeros(3), inertia, np.linalg.inv(inertia), rate)
    assert np.isclose(omega_dot[0], 0.05)
    rigid = RotationalEOM(omega, np.zeros(3), inertia, np.linalg.inv(inertia))
    assert np.allclose(rigid, 0.0)


# -------------------------------------------------------------- table Cmq


def _table(with_moments: bool, clp=None, cl_roll=None) -> AeroDatabase:
    """Two lifting parts: a nose (CN_alpha 2 at 0.3 m) and fins (10 at 1.6 m)."""
    moments = dict(cna_sum=12.0, cna_x_m=2 * 0.3 + 10 * 1.6,
                   cna_x2_m2=2 * 0.3 ** 2 + 10 * 1.6 ** 2) if with_moments else {}
    rows = [
        AeroCoefficients(mach=m, alpha_deg=a, cd=0.4, cn=12.0 * np.radians(a), cm=0.0,
                         x_cp_m=(2 * 0.3 + 10 * 1.6) / 12.0, clp=clp, cl_roll=cl_roll,
                         **moments)
        for m in (0.0, 1.0, 3.0) for a in (0.0, 5.0, 10.0)
    ]
    return AeroDatabase(rows, reference_length_m=2.0)


def _model(db: AeroDatabase, **kwargs) -> RasaeroAeroModel:
    return RasaeroAeroModel(db, reference_area_m2=0.01, reference_length_m=2.0,
                            body_axis=BODY_AXIS, **kwargs)


def _flow(alpha_deg: float, speed: float = 100.0) -> np.ndarray:
    a = np.radians(alpha_deg)
    return np.array([speed * np.sin(a), speed * np.cos(a), 0.0])


def test_table_cmq_is_the_sum_about_the_cg_given():
    model = _model(_table(True))
    cg = np.array([0.0, -1.25, 0.0])              # station 1.25
    coeffs = model.database.lookup(1.0, 0.0)
    direct = -2.0 * (2 * (0.3 - 1.25) ** 2 + 10 * (1.6 - 1.25) ** 2) / 2.0 ** 2
    assert np.isclose(model.cmq_from_table(coeffs, cg), direct)
    forces = model.forces_and_moments(_flow(4.0), 1.2, 340.0, cg, np.array([0.0, 0.0, 0.1]))
    assert np.isclose(forces.cmq, direct)


def test_the_component_sum_exceeds_the_lumped_estimate():
    """Lumping the slope at the total CP cancels the nose's arm against the fins'."""
    model = _model(_table(True))
    cg = np.array([0.0, -1.25, 0.0])
    forces = model.forces_and_moments(_flow(4.0), 1.2, 340.0, cg)
    lumped = model.estimate_cmq(forces.mach, forces.alpha_deg, forces.static_margin_m)
    assert forces.cmq < 0.0 and lumped < 0.0
    assert abs(forces.cmq) > 2.0 * abs(lumped), (forces.cmq, lumped)


def test_a_table_without_moments_falls_back_to_the_estimate():
    model = _model(_table(False))
    cg = np.array([0.0, -1.25, 0.0])
    assert not model.database.has_damping
    forces = model.forces_and_moments(_flow(4.0), 1.2, 340.0, cg)
    assert np.isclose(
        forces.cmq, model.estimate_cmq(forces.mach, forces.alpha_deg, forces.static_margin_m)
    )


def test_an_explicit_cmq_still_overrides_the_table():
    model = _model(_table(True), cmq=-9.0)
    forces = model.forces_and_moments(_flow(4.0), 1.2, 340.0, np.array([0.0, -1.25, 0.0]))
    assert forces.cmq == -9.0


def test_the_moments_round_trip_through_csv(tmp_path):
    db = _table(True, clp=-10.0, cl_roll=0.05)
    reloaded = AeroDatabase.from_csv(db.to_csv(tmp_path / "t.csv"), reference_length_m=2.0)
    assert reloaded.has_damping and reloaded.has_roll
    row = reloaded.lookup(1.0, 5.0)
    assert np.isclose(row.cna_x2_m2, 2 * 0.3 ** 2 + 10 * 1.6 ** 2)
    assert np.isclose(row.clp, -10.0) and np.isclose(row.cl_roll, 0.05)


# ------------------------------------------------------------------- roll


def test_roll_damping_and_forcing_come_from_the_table():
    model = _model(_table(True, clp=-10.0, cl_roll=0.05))
    cg = np.array([0.0, -1.25, 0.0])
    speed, rho = 100.0, 1.2
    diameter = model.reference_diameter_m
    q = 0.5 * rho * speed ** 2

    forcing = model.forces_and_moments(_flow(0.0, speed), rho, 340.0, cg)
    assert np.isclose(np.dot(forcing.static_moment_body_nm, BODY_AXIS),
                      q * 0.01 * diameter * 0.05)

    rolling = model.forces_and_moments(
        _flow(0.0, speed), rho, 340.0, cg, np.array([0.0, 2.0, 0.0]),
    )
    expected = 0.25 * rho * speed * 0.01 * diameter ** 2 * (-10.0) * 2.0
    assert np.isclose(np.dot(rolling.damping_moment_body_nm, BODY_AXIS), expected)
    assert forcing.clp == -10.0 and forcing.cl_roll == 0.05


def test_the_steady_roll_rate_is_the_closed_form():
    """Forcing and damping balance at p = -2 V Cl / (d Clp)."""
    model = _model(_table(True, clp=-10.0, cl_roll=0.05))
    cg = np.array([0.0, -1.25, 0.0])
    speed = 100.0
    p = -2.0 * speed * 0.05 / (model.reference_diameter_m * -10.0)
    forces = model.forces_and_moments(
        _flow(0.0, speed), 1.2, 340.0, cg, np.array([0.0, p, 0.0]),
    )
    assert abs(np.dot(forces.moment_body_nm, BODY_AXIS)) < 1e-9


def test_a_table_without_roll_leaves_roll_alone():
    model = _model(_table(True))
    forces = model.forces_and_moments(
        _flow(0.0), 1.2, 340.0, np.array([0.0, -1.25, 0.0]), np.array([0.0, 2.0, 0.0]),
    )
    assert np.isclose(np.dot(forces.damping_moment_body_nm, BODY_AXIS), 0.0)
    assert np.isclose(np.dot(forces.static_moment_body_nm, BODY_AXIS), 0.0)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
