"""A flight ends on the ground, not on a clock.

There used to be a ``t_max`` on every run and nothing else; a slow descent
from a high apogee ran into it and the result stopped in mid-air, main
never deployed, landing never recorded. These pin the open-ended run: it
lands, the windows it is integrated in leave no seam, a cutoff still works
when asked for and says so, and the ground is wherever the rail stood.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.vehicle.recovery import standard_recovery  # noqa: E402


def _sim():
    from trajectory import simulation as tm
    return tm.RocketSimulation()


def _recovery():
    return standard_recovery(dry_mass_kg=50.0, main_deploy_altitude_m=500.0)


def test_a_flight_runs_until_it_lands():
    sim = _sim()
    result = sim.run(launch_elevation=np.radians(85), recovery=_recovery())
    assert result.landed
    assert [p["name"] for p in result.phases] == ["ascent", "drogue", "main"]
    # The record ends on the event state itself, 0.1 m below the ground.
    assert abs(result.y[1, -1]) < 0.2
    assert result.y[4, -1] < 0.0, "still descending at the last sample"


def test_a_ballistic_flight_lands_too():
    result = _sim().run(launch_elevation=np.radians(80))
    assert result.landed
    assert result.phases[0]["name"] == "flight"
    assert abs(result.y[1, -1]) < 0.2


def test_windows_leave_no_seam():
    """Integrating in 7 s windows must give the same flight as one window."""
    one = _sim()
    one.phase_window_s = 1e6
    whole = one.run(launch_elevation=np.radians(85), recovery=_recovery())

    many = _sim()
    many.phase_window_s = 7.0
    pieces = many.run(launch_elevation=np.radians(85), recovery=_recovery())

    assert pieces.landed
    assert np.all(np.diff(pieces.t) > 0.0), "duplicate or reversed samples"
    assert np.isclose(pieces.y[1].max(), whole.y[1].max(), rtol=1e-3)
    assert np.isclose(pieces.t[-1], whole.t[-1], rtol=1e-2)
    assert [p["name"] for p in pieces.phases] == [p["name"] for p in whole.phases]


def test_a_cutoff_is_honoured_and_reported():
    result = _sim().run(launch_elevation=np.radians(85), t_max=20.0,
                        recovery=_recovery())
    assert not result.landed
    assert result.t[-1] <= 20.0
    assert result.y[1, -1] > 100.0, "20 s in, this vehicle is still climbing"


def test_the_runaway_guard_stops_a_flight_that_never_ends():
    sim = _sim()
    sim.phase_window_s = 5.0
    sim.runaway_s = 10.0
    result = sim.run(launch_elevation=np.radians(85), recovery=_recovery())
    assert not result.landed
    assert result.t[-1] <= 10.0 + 5.0


def test_the_ground_is_where_the_rail_stood():
    """A pad at 250 m is the ground for its own flight."""
    result = _sim().run(
        launch_elevation=np.radians(85), recovery=_recovery(),
        pad_position_m=np.array([0.0, 250.0, 0.0]),
    )
    assert result.landed
    assert abs(result.y[1, -1] - 250.0) < 0.2
    assert result.y[1].min() > 240.0, "fell through the pad"
