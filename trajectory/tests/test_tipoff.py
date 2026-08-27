"""Rail tip-off on two buttons.

Runs under pytest, and standalone via ``python trajectory/tests/test_tipoff.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory import simulation as tm  # noqa: E402
from trajectory.eom import quat_to_dcm  # noqa: E402
from trajectory.sim import LaunchRail  # noqa: E402
from trajectory.vehicle.engine import Engine  # noqa: E402

BUTTONS = (1.0, 3.5)          # stations on the four-metre placeholder


def _vacuum_sim(thrust_n: float = 20000.0) -> tm.RocketSimulation:
    """The placeholder with no drag, no wind and a motor that loses no mass."""
    sim = tm.RocketSimulation()
    sim.engine = Engine(
        thrust_curve=np.array([thrust_n, thrust_n]), time_points=np.array([0.0, 100.0]),
        isp_vac=1e9, isp_sl=1e9, nozzle_area=0.0, thrust_reference="vacuum",
    )
    sim.reference_area = 0.0
    sim.rtol, sim.atol = 1e-9, 1e-11
    return sim


def _tilt_from_rail(state: np.ndarray, rail: LaunchRail) -> float:
    axis = quat_to_dcm(state[6:10]) @ tm.BODY_AXIS
    return float(np.degrees(np.arccos(np.clip(np.dot(axis, rail.direction), -1.0, 1.0))))


# ------------------------------------------------------------ the rail itself


def test_without_buttons_the_rail_is_what_it_was():
    rail = LaunchRail(length_m=5.0)
    state = np.zeros(tm.STATE_SIZE)
    state[6] = 1.0
    state[0:3] = rail.direction * 2.0
    assert rail.phase_of(state) == "rail" and rail.is_on_rail(state[0:3])
    state[0:3] = rail.direction * 5.0
    assert rail.phase_of(state) == "free"
    assert not rail.has_buttons
    assert rail.start_offset_m(np.array([0.0, -1.5, 0.0]), tm.BODY_AXIS) == 0.0


def test_buttons_are_ordered_and_distinct():
    assert LaunchRail(buttons_m=(3.5, 1.0)).buttons_m == (1.0, 3.5)
    with pytest.raises(ValueError):
        LaunchRail(buttons_m=(2.0, 2.0))


def test_the_aft_button_starts_at_the_foot_and_the_phases_follow_it():
    sim = _vacuum_sim()
    rail = LaunchRail(elevation_rad=np.radians(80.0), length_m=5.0, buttons_m=BUTTONS)
    state = sim.initial_state(rail)
    cg = sim._cg_of(state)
    forward, aft = rail.button_bodies()
    assert rail.travel_of(state, aft, cg) == pytest.approx(0.0, abs=1e-12)
    assert rail.travel_of(state, forward, cg) == pytest.approx(2.5, abs=1e-12)
    assert rail.phase_of(state, cg) == "rail"
    state[0:3] += rail.direction * 2.6
    assert rail.phase_of(state, cg) == "tipoff"
    state[0:3] += rail.direction * 2.5
    assert rail.phase_of(state, cg) == "free"


# ------------------------------------------------------------ the constraint


def test_the_pinned_button_has_no_acceleration_across_the_rail():
    """Whatever the forces, the aft button's acceleration lies along the
    rail and the roll acceleration is zero -- to round-off."""
    sim = _vacuum_sim()
    sim.wind.surface_wind = np.array([8.0, 0.0])
    sim.wind.surface_dir = np.pi / 2
    sim.reference_area = np.pi * 0.15 ** 2          # some drag, off the pivot
    rail = LaunchRail(elevation_rad=np.radians(80.0), length_m=5.0, buttons_m=BUTTONS)
    sim.launch_rail = rail
    state = sim.initial_state(rail)
    state[0:3] += rail.direction * 3.0
    state[3:6] = rail.direction * 20.0
    state[10:13] = [0.3, 0.0, -0.2]                # already turning a little
    point = sim.evaluate(state, 0.5)
    assert point.rail_phase == "tipoff"
    a_cg, alpha = sim.tipoff_accelerations(point, state)
    dcm = point.dcm_b2i
    r = rail.button_bodies()[1] - point.cg_body_m
    omega = state[10:13]
    a_button = a_cg + dcm @ (np.cross(alpha, r) + np.cross(omega, np.cross(omega, r)))
    across = a_button - np.dot(a_button, rail.direction) * rail.direction
    assert np.linalg.norm(across) < 1e-9 * max(1.0, np.linalg.norm(a_button))
    assert abs(np.dot(alpha, tm.BODY_AXIS)) < 1e-12
    derivative = sim.state_derivative(0.5, state)
    assert np.allclose(derivative[3:6], a_cg) and np.allclose(derivative[10:13], alpha)


def test_the_button_stays_on_the_line_through_the_tip_off():
    sim = _vacuum_sim()
    rail_state = sim.run(launch_elevation=np.radians(80.0), rail_length_m=5.0,
                         t_max=1.0, dt=0.001, rail_buttons_m=BUTTONS)
    rail = sim.launch_rail
    aft = rail.button_bodies()[1]
    for state in rail_state.y.T:
        cg = sim._cg_of(state)
        if rail.phase_of(state, cg) == "free":
            break
        point = state[0:3] + quat_to_dcm(state[6:10]) @ (aft - cg)
        offset = point - rail.position_m
        across = offset - np.dot(offset, rail.direction) * rail.direction
        assert np.linalg.norm(across) < 1e-5
    exit_state = rail_state.rail_exit
    assert exit_state["exact"] and exit_state["tipoff_time_s"] > 0.0


# ------------------------------------------------------------ the physics


def test_a_vertical_rail_in_still_air_has_no_tip_off():
    sim = _vacuum_sim()
    result = sim.run(launch_elevation=np.radians(90.0), rail_length_m=5.0, t_max=1.0,
                     dt=0.01, rail_buttons_m=BUTTONS)
    assert result.rail_exit["pitch_rate_dps"] < 1e-6
    assert result.rail_exit["tip_off_deg"] < 1e-6


def test_gravity_tips_the_nose_down_at_the_closed_form_rate():
    """``theta_dot = m g d cos(el) dt / (I + m d^2)``: gravity's moment about
    the aft button, over the time the forward button is off the rail, on
    the inertia about that button. The thrust passes through the button
    and the acceleration term is second order in the small angle."""
    sim = _vacuum_sim()
    elevation = np.radians(80.0)
    result = sim.run(launch_elevation=elevation, rail_length_m=5.0, t_max=1.0,
                     dt=0.001, rail_buttons_m=BUTTONS)
    exit_state = result.rail_exit
    rate = np.radians(exit_state["pitch_rate_dps"])
    duration = exit_state["tipoff_time_s"]
    assert 0.02 < duration < 0.3, duration

    point = sim.evaluate(sim._tipoff_record[1], sim._tipoff_record[0])
    d = float(np.linalg.norm(sim.launch_rail.button_bodies()[1] - point.cg_body_m))
    inertia_about_button = point.inertia_kg_m2[0, 0] + point.mass_kg * d * d
    expected = point.mass_kg * 9.80665 * d * np.cos(elevation) * duration / inertia_about_button
    assert rate == pytest.approx(expected, rel=0.1), (rate, expected)

    # And the nose went *down*: further from the vertical than the rail.
    y_exit = sim._rail_exit_record[1]
    axis = quat_to_dcm(y_exit[6:10]) @ tm.BODY_AXIS
    assert axis[1] < sim.launch_rail.direction[1]
    assert exit_state["tip_off_deg"] > 0.0


def test_the_constraint_does_no_work():
    """Kinetic plus potential energy changes only by the thrust's work."""
    sim = _vacuum_sim()
    result = sim.run(launch_elevation=np.radians(80.0), rail_length_m=5.0, t_max=0.8,
                     dt=0.0005, rail_buttons_m=BUTTONS)
    assert result.rail_exit["exact"]
    times, states = result.t, result.y.T
    energy, power = [], []
    for t, state in zip(times, states):
        point = sim.evaluate(state, float(t))
        v = state[3:6]
        omega = state[10:13]
        kinetic = 0.5 * point.mass_kg * v @ v + 0.5 * omega @ point.inertia_kg_m2 @ omega
        potential = point.mass_kg * 9.80665 * state[1]
        energy.append(kinetic + potential)
        r_nozzle = sim.thrust_position_body_m - point.cg_body_m
        v_nozzle = v + point.dcm_b2i @ np.cross(omega, r_nozzle)
        power.append(float(np.dot(point.thrust_inertial_n, v_nozzle)))
    work = np.concatenate([[0.0], np.cumsum(0.5 * (np.array(power[1:]) + np.array(power[:-1])) * np.diff(times))])
    balance = np.array(energy) - energy[0] - work
    assert np.max(np.abs(balance)) < 2e-3 * work[-1], np.max(np.abs(balance)) / work[-1]


def test_a_longer_tip_off_leaves_a_larger_rate():
    sim = _vacuum_sim()
    short = sim.run(launch_elevation=np.radians(80.0), rail_length_m=5.0, t_max=1.0,
                    dt=0.01, rail_buttons_m=(2.5, 3.5)).rail_exit
    sim = _vacuum_sim()
    long = sim.run(launch_elevation=np.radians(80.0), rail_length_m=5.0, t_max=1.0,
                   dt=0.01, rail_buttons_m=(0.5, 3.5)).rail_exit
    assert long["tipoff_time_s"] > short["tipoff_time_s"]
    assert long["pitch_rate_dps"] > 2.0 * short["pitch_rate_dps"]


def test_a_whole_flight_flies_with_buttons():
    sim = tm.RocketSimulation()
    result = sim.run(launch_elevation=np.radians(85.0), rail_length_m=5.0, dt=0.5,
                     rail_buttons_m=BUTTONS)
    assert result.landed and np.all(np.isfinite(result.y))
    assert result.rail_exit["exact"] and "pitch_rate_dps" in result.rail_exit
    assert result.rail_exit["velocity_mps"] > 15.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
