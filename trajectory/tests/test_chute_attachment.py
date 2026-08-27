"""The canopy pulls at the harness, and the vehicle hangs from it.

Runs under pytest, and standalone via
``python trajectory/tests/test_chute_attachment.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory import simulation as tm  # noqa: E402
from trajectory.analysis.flightlog import FlightLog  # noqa: E402
from trajectory.eom import quat_to_dcm  # noqa: E402
from trajectory.vehicle.recovery import Parachute, standard_recovery  # noqa: E402


def _descent_state(sim, pitch_deg: float, rate: float = 0.0) -> np.ndarray:
    """Falling at 6 m/s with the axis tilted off the vertical about body x."""
    state = np.zeros(tm.STATE_SIZE)
    state[1] = 500.0
    state[4] = -6.0
    half = np.radians(pitch_deg) / 2.0
    state[6:10] = [np.cos(half), np.sin(half), 0.0, 0.0]
    state[10] = rate
    return state


def _armed(sim, chute):
    sim.launch_rail = None
    sim._active_chute = chute
    sim._deploy_trigger_s = -100.0            # long since inflated
    return sim


def test_through_the_cg_is_what_it_was():
    sim = _armed(tm.RocketSimulation(), Parachute(cda_m2=20.0))
    point = sim.evaluate(_descent_state(sim, 30.0, rate=0.5), 0.0)
    assert np.allclose(point.chute_moment_body_nm, 0.0)
    assert point.chute_force_inertial_n[1] == pytest.approx(0.5 * point.rho_kg_m3 * 36.0 * 20.0, rel=1e-9)


def test_the_canopy_pulls_at_the_attachment_and_sees_its_motion():
    """The moment is r x F about the CG, and the drag opposes the point's
    own velocity v + w x r, not the CG's."""
    sim = _armed(tm.RocketSimulation(), Parachute(cda_m2=20.0, attachment_station_m=0.2))
    still = sim.evaluate(_descent_state(sim, 30.0), 0.0)
    r = np.array([0.0, -0.2, 0.0]) - still.cg_body_m
    force_body = still.dcm_b2i.T @ still.chute_force_inertial_n
    assert np.allclose(still.chute_moment_body_nm, np.cross(r, force_body))
    assert np.linalg.norm(still.chute_moment_body_nm) > 50.0

    rate = 2.0
    swinging = sim.evaluate(_descent_state(sim, 30.0, rate=rate), 0.0)
    v_point = np.array([0.0, -6.0, 0.0]) + swinging.dcm_b2i @ np.cross([rate, 0.0, 0.0], r)
    expected = -0.5 * swinging.rho_kg_m3 * np.linalg.norm(v_point) * 20.0 * v_point
    assert np.allclose(swinging.chute_force_inertial_n, expected)
    assert not np.allclose(swinging.chute_force_inertial_n, still.chute_force_inertial_n)


def test_the_moment_restores_toward_nose_up():
    """Tilted 30 degrees, the moment about body x turns the nose back up."""
    sim = _armed(tm.RocketSimulation(), Parachute(cda_m2=20.0, attachment_station_m=0.2))
    point = sim.evaluate(_descent_state(sim, 30.0), 0.0)
    # A positive rotation about body x took the nose over; the restoring
    # moment is negative about x.
    assert point.chute_moment_body_nm[0] < 0.0
    assert abs(point.chute_moment_body_nm[1]) < 1e-9 and abs(point.chute_moment_body_nm[2]) < 1e-9


def test_a_flight_hangs_nose_up_by_landing():
    """Launched 30 degrees off the vertical with nothing to turn it -- the
    placeholder has no table -- the vehicle keeps that attitude to apogee
    and the canopy then swings it to hang within a few degrees of nose-up."""
    sim = tm.RocketSimulation()
    recovery = standard_recovery(dry_mass_kg=50.0, main_deploy_altitude_m=500.0,
                                 attachment_station_m=0.3)
    result = sim.run(launch_elevation=np.radians(60.0), dt=0.5, recovery=recovery)
    assert result.landed
    log = FlightLog.from_flight(sim, result)
    assert log.axis_from_vertical_deg[log.apogee_index] == pytest.approx(30.0, abs=1.0)
    hang = log.hang_angle_deg()
    assert hang is not None and hang < 3.0, hang


def test_a_flight_through_the_cg_says_nothing_about_attitude():
    """No moment from the canopy, none from the fallback drag law: the
    attitude the rail gave it is the attitude it lands with."""
    sim = tm.RocketSimulation()
    recovery = standard_recovery(dry_mass_kg=50.0, main_deploy_altitude_m=500.0)
    result = sim.run(launch_elevation=np.radians(60.0), dt=0.5, recovery=recovery)
    log = FlightLog.from_flight(sim, result)
    assert log.hang_angle_deg() == pytest.approx(30.0, abs=1.0)


def test_the_swing_starts_as_a_pendulum_and_dies_away():
    """At the first instant the angular acceleration is ``-F L sin(theta) / I``
    for the canopy force ``F`` at the harness; then the harness point's own
    motion through the air damps the swing within a couple of seconds --
    the linearised damping ratio is about 0.8 for this canopy."""
    sim = tm.RocketSimulation()
    sim.mass_props.prop_mass = 0.0
    sim.mass_props.mass_0 = sim.mass_props.dry_mass
    sim.reference_area = 0.0                      # no body drag to confuse the swing
    chute = Parachute(cda_m2=20.0, attachment_station_m=0.2)
    _armed(sim, chute)
    state = _descent_state(sim, 10.0)
    state[4] = -chute.terminal_velocity(sim.mass_props.dry_mass)
    state[tm.PROP_IDX] = 0.0

    point = sim.evaluate(state, 0.0)
    _, cg, inertia = sim.mass_props.at_propellant(0.0)
    arm = abs(0.2 - (-cg[1]))
    force = float(np.linalg.norm(point.chute_force_inertial_n))
    # Terminal velocity was sized at sea level; the state is at 500 m.
    assert force == pytest.approx(0.5 * point.rho_kg_m3 * state[4] ** 2 * 20.0, rel=1e-9)
    assert force == pytest.approx(sim.mass_props.dry_mass * 9.80665, rel=0.06)
    derivative = sim.state_derivative(0.0, state)
    expected = -force * arm * np.sin(np.radians(10.0)) / inertia[0, 0]
    assert derivative[10] == pytest.approx(expected, rel=1e-6)
    assert abs(derivative[11]) < 1e-9 and abs(derivative[12]) < 1e-9

    sim.rtol, sim.atol = 1e-8, 1e-10
    result = sim._integrate_phase(state, 0.0, 4.0, 0.01, [])
    tilt = np.array([
        np.degrees(np.arccos(np.clip((quat_to_dcm(y[6:10]) @ tm.BODY_AXIS)[1], -1, 1)))
        for y in result.y.T
    ])
    assert tilt[0] == pytest.approx(10.0, abs=1e-6)
    assert np.max(tilt[result.t > 2.0]) < 1.0, "damped out within two seconds"
    assert np.max(tilt) <= 10.0 + 1e-6, "never swings wider than it started"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
