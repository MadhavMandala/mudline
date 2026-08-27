"""The full-six-degree-of-freedom sprint: what the frame and the solver do.

Each test here computes an expected number from a closed form and holds
the simulator to it. Runs under pytest, and standalone via
``python trajectory/tests/test_sixdof.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory import simulation as tm  # noqa: E402
from trajectory.environment.gravity import (  # noqa: E402
    G0,
    OMEGA_EARTH_RADPS,
    coriolis_acceleration,
    earth_rotation_enu,
)
from trajectory.sim import LaunchRail  # noqa: E402
from trajectory.sim.integrator import TrajectoryIntegrator  # noqa: E402
from trajectory.vehicle.engine import Engine  # noqa: E402


def _ballistic_sim() -> tm.RocketSimulation:
    """A point mass in vacuum: no thrust, no propellant, no drag."""
    sim = tm.RocketSimulation()
    sim.engine = Engine(
        thrust_curve=np.array([0.0, 0.0]), time_points=np.array([0.0, 1.0]),
        isp_vac=250.0, isp_sl=250.0, nozzle_area=0.0, thrust_reference="vacuum",
    )
    sim.mass_props.prop_mass = 0.0
    sim.mass_props.mass_0 = sim.mass_props.dry_mass
    sim.reference_area = 0.0          # the fallback drag law has nothing to act on
    sim.launch_rail = LaunchRail(length_m=0.0)
    return sim


def _thrown_up(sim: tm.RocketSimulation, speed: float, rtol=1e-9, atol=1e-11):
    state0 = sim.initial_state(sim.launch_rail)
    state0[4] = speed
    sim.rtol, sim.atol = rtol, atol
    return sim._integrate_phase(state0, 0.0, None, 0.1, [])


# ------------------------------------------------------------- Coriolis


def test_earth_rotation_points_at_the_pole():
    omega = earth_rotation_enu(np.radians(90.0))
    assert np.allclose(omega, [0.0, OMEGA_EARTH_RADPS, 0.0])
    omega = earth_rotation_enu(0.0)
    assert np.allclose(omega, [0.0, 0.0, OMEGA_EARTH_RADPS])


def test_a_rising_body_is_deflected_west():
    """Eastward velocity from the ground's rotation is left behind on the way up."""
    a = coriolis_acceleration(np.array([0.0, 100.0, 0.0]), np.radians(45.0))
    assert a[0] < 0.0 and abs(a[1]) < 1e-12 and abs(a[2]) < 1e-12


def test_a_vertical_throw_lands_west_by_the_textbook_amount():
    """Landing deflection ``(4/3) Omega cos(lat) v0^3 / g^2`` west of the pad."""
    latitude = np.radians(45.0)
    speed = 100.0
    sim = _ballistic_sim()
    sim.latitude_rad = latitude
    result = _thrown_up(sim, speed)
    assert result.status == 1, "the ground event should have ended the flight"
    y = np.asarray(result.y_events[0][0], dtype=float)
    expected = -(4.0 / 3.0) * OMEGA_EARTH_RADPS * np.cos(latitude) * speed ** 3 / G0 ** 2
    assert expected < -0.5, "the effect must be resolvable"
    # g falls by 3e-5 over the 510 m climb, which is why this is 0.5%, not 1e-6.
    assert y[0] == pytest.approx(expected, rel=5e-3)
    assert abs(y[2]) < 1e-3


def test_no_latitude_means_no_coriolis():
    sim = _ballistic_sim()
    result = _thrown_up(sim, 100.0)
    y = np.asarray(result.y_events[0][0], dtype=float)
    assert abs(y[0]) < 1e-9 and abs(y[2]) < 1e-9
    point = sim.evaluate(result.y[:, 5], float(result.t[5]))
    assert np.allclose(point.coriolis_inertial_n, 0.0)


# ------------------------------------------------------------- tolerances


def test_the_simulation_hands_its_tolerances_to_the_solver():
    sim = tm.RocketSimulation()
    sim.rtol, sim.atol, sim.max_step_s = 1e-4, 1e-7, 0.25
    sim.run(launch_elevation=np.radians(85.0), t_max=2.0, dt=0.5)
    assert (sim.integrator.rtol, sim.integrator.atol, sim.integrator.max_step) == (1e-4, 1e-7, 0.25)


def test_the_default_tolerance_is_tighter_than_scipys():
    assert TrajectoryIntegrator.DEFAULT_RTOL < 1e-3
    assert TrajectoryIntegrator(lambda t, y: y).rtol == TrajectoryIntegrator.DEFAULT_RTOL


def test_a_tight_tolerance_costs_evaluations_and_changes_little():
    loose = tm.RocketSimulation()
    loose.rtol, loose.atol = 1e-3, 1e-6
    tight = tm.RocketSimulation()
    tight.rtol, tight.atol = 1e-8, 1e-10
    a = loose.run(launch_elevation=np.radians(85.0), t_max=40.0, dt=0.5)
    b = tight.run(launch_elevation=np.radians(85.0), t_max=40.0, dt=0.5)
    assert b.nfev > a.nfev
    assert np.max(b.y[1]) == pytest.approx(np.max(a.y[1]), rel=2e-3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
