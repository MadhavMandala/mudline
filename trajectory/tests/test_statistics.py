"""Tests for trajectory summary statistics.

Pins the defect where ``flight_statistics`` read state column 0 as the time
base. Column 0 is downrange ``x``, so on a vertical flight every time-derived
statistic silently reported 0.0.

Runs under pytest, and standalone via ``python trajectory/tests/test_statistics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.analysis.statistics import flight_statistics  # noqa: E402


def _vertical_flight(n: int = 101):
    """A purely vertical up-and-down flight: x and z stay exactly zero."""
    times = np.linspace(0.0, 20.0, n)
    states = np.zeros((n, 14))
    # Altitude peaks at t = 10 s.
    states[:, 1] = 100.0 * times - 5.0 * times**2
    states[:, 4] = 100.0 - 10.0 * times
    return states, times


def test_apogee_time_is_a_real_time_not_downrange():
    states, times = _vertical_flight()
    stats = flight_statistics(states, times)
    # The old implementation returned x at the apogee row, which is 0.0 here.
    assert np.isclose(stats["apogee_time"], 10.0)
    assert stats["apogee_time"] != 0.0


def test_flight_time_spans_the_time_base():
    states, times = _vertical_flight()
    stats = flight_statistics(states, times)
    assert np.isclose(stats["flight_time"], 20.0)


def test_max_altitude_and_velocity():
    states, times = _vertical_flight()
    stats = flight_statistics(states, times)
    assert np.isclose(stats["max_altitude"], 500.0)
    assert np.isclose(stats["max_velocity"], 100.0)


def test_apogee_time_survives_nonzero_downrange():
    """A downrange-carrying flight must not change which sample is apogee."""
    states, times = _vertical_flight()
    states[:, 0] = 3.0 * times          # x now grows monotonically
    states[:, 3] = 3.0
    stats = flight_statistics(states, times)
    assert np.isclose(stats["apogee_time"], 10.0)
    assert np.isclose(stats["range"], 60.0)


def test_times_must_match_state_rows():
    states, times = _vertical_flight()
    try:
        flight_statistics(states, times[:-1])
    except ValueError:
        return
    raise AssertionError("mismatched time base was not rejected")


def test_states_need_position_and_velocity_columns():
    try:
        flight_statistics(np.zeros((5, 3)), np.arange(5.0))
    except ValueError:
        return
    raise AssertionError("under-wide state array was not rejected")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"  FAIL  {name}: {exc}")
    total = sum(1 for n in globals() if n.startswith("test_"))
    print(f"\n{total - failures}/{total} passed")
    raise SystemExit(1 if failures else 0)
