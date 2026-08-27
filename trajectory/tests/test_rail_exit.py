"""The rail exit is an event, and the first free instant is looked at.

The exit used to be the first output sample past the rail's length, so
the exit speed -- the go/no-go number for a launch -- was good to one
output step, and nothing was said about the angle of attack the crosswind
makes of a vehicle that leaves the rail still pointing where the rail
pointed.

Runs under pytest, and standalone via
``python trajectory/tests/test_rail_exit.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory import simulation as tm  # noqa: E402
from trajectory.environment.wind import WindModel  # noqa: E402
from trajectory.vehicle.recovery import standard_recovery  # noqa: E402


def _sim(wind: WindModel | None = None) -> tm.RocketSimulation:
    sim = tm.RocketSimulation()
    if wind is not None:
        sim.wind = wind
    return sim


def test_the_exit_is_a_root_not_a_sample():
    sim = _sim()
    result = sim.run(launch_elevation=np.radians(85.0), rail_length_m=5.0,
                     t_max=20.0, dt=0.5)
    exit_state = result.rail_exit
    assert exit_state["exact"]
    travelled = sim.launch_rail.distance_along(exit_state["position_m"])
    assert np.isclose(travelled, 5.0, atol=1e-6)
    assert not np.any(np.isclose(result.t, exit_state["time_s"], atol=1e-9)), \
        "the root lies between output samples"


def test_the_grid_scan_agrees_to_within_a_sample():
    sim = _sim()
    result = sim.run(launch_elevation=np.radians(85.0), rail_length_m=5.0,
                     t_max=20.0, dt=0.5)
    sampled = sim.launch_rail.exit_state(result.t, result.y.T)
    gap = sampled["time_s"] - result.rail_exit["time_s"]
    assert 0.0 <= gap <= 0.5 + 1e-9
    assert sampled["velocity_mps"] >= result.rail_exit["velocity_mps"]


def test_alpha_off_the_rail_is_the_crosswind_kinematics():
    """Pointing where the rail pointed, into the relative wind."""
    wind = WindModel(surface_wind=np.array([8.0, 0.0]), surface_dir=np.pi / 2)
    result = _sim(wind).run(launch_elevation=np.radians(90.0), rail_length_m=5.0,
                            t_max=10.0, dt=0.1)
    exit_state = result.rail_exit
    speed = exit_state["velocity_mps"]
    # The rail's top is below the 10 m reference height, where the profile
    # has not yet reached the full surface wind.
    wind = exit_state["wind_mps"]
    assert 0.6 * 8.0 < wind < 8.0
    assert np.isclose(exit_state["alpha_deg"], np.degrees(np.arctan2(wind, speed)), atol=0.2)
    assert exit_state["airspeed_mps"] > speed


def test_still_air_gives_no_alpha_at_the_exit():
    result = _sim().run(launch_elevation=np.radians(85.0), rail_length_m=5.0,
                        t_max=10.0, dt=0.1)
    assert abs(result.rail_exit["alpha_deg"]) < 1e-6


def test_a_flight_cut_off_on_the_rail_has_no_exit():
    result = _sim().run(launch_elevation=np.radians(85.0), rail_length_m=5000.0,
                        t_max=1.0, dt=0.1)
    assert result.rail_exit is None


def test_no_rail_means_no_exit():
    result = _sim().run(launch_elevation=np.radians(85.0), rail_length_m=0.0,
                        t_max=5.0, dt=0.5)
    assert result.rail_exit is None


def test_the_ascent_does_not_end_at_the_rail():
    """A non-terminal event must not be mistaken for the phase's end."""
    sim = _sim()
    recovery = standard_recovery(dry_mass_kg=50.0, main_deploy_altitude_m=500.0)
    result = sim.run(launch_elevation=np.radians(85.0), rail_length_m=5.0,
                     dt=1.0, recovery=recovery)
    assert [p["name"] for p in result.phases] == ["ascent", "drogue", "main"]
    assert result.phases[0]["t_end"] > 60.0
    assert result.rail_exit["exact"] and result.rail_exit["time_s"] < 2.0
    assert result.landed


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
