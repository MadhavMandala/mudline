"""Rail buttons through the flight settings.

Runs under pytest, and standalone via
``python parametric/tests/test_tipoff_settings.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.flight import FlightSettings, default_rail_buttons_m, fly_model  # noqa: E402
from parametric.standard import basic_rocket  # noqa: E402

_TABLE = None


def _table():
    global _TABLE
    if _TABLE is None:
        from parametric import aero

        _TABLE, _ = aero.run_analysis(basic_rocket(), aero.AeroSettings(
            mach_min=0.05, mach_max=1.5, mach_points=8, alpha_max_deg=8.0, alpha_points=3,
        ))
    return _TABLE


def _settings(**overrides) -> FlightSettings:
    base = dict(dt_s=0.05, couple_aero_altitude=False, elevation_deg=80.0, rail_length_m=3.0)
    base.update(overrides)
    return FlightSettings(**base)


def test_the_default_buttons_sit_at_the_cg_and_the_tail():
    model = basic_rocket()
    forward, aft = default_rail_buttons_m(model)
    start, end = model.station_range_m()
    assert start < forward < aft < end
    assert aft == pytest.approx(end - 0.05)


def test_the_flight_reports_the_tip_off():
    outcome = fly_model(basic_rocket(), _settings(), _table())
    exit_state = outcome.rail_exit
    assert exit_state["exact"]
    assert exit_state["pitch_rate_dps"] > 1.0
    assert exit_state["tipoff_time_s"] > 0.0
    assert "buttons at the CG and the tail" in outcome.settings.describe_rail()
    # The CG starts its own distance up the rail: the aft button is at the foot.
    rail = outcome.simulation.launch_rail
    offset = rail.start_offset_m(outcome.simulation._cg_of(outcome.states[0]), np.array([0.0, 1.0, 0.0]))
    assert 0.0 < offset < 1.0
    assert outcome.states[0, 1] == pytest.approx(offset * rail.direction[1])
    assert outcome.landed


def test_the_fins_arrest_the_tip_off():
    """A table-flown vehicle is back within a degree of its flight path a
    second after the rail; the tip-off is a disturbance, not a heading."""
    outcome = fly_model(basic_rocket(), _settings(), _table())
    log = outcome.log
    t_exit = outcome.rail_exit["time_s"]
    assert log.alpha_deg[log.index_at(t_exit + 1.0)] < 1.0
    assert log.alpha_deg[log.index_at(t_exit + 3.0)] < 0.5


def test_the_cg_rail_is_still_there():
    outcome = fly_model(basic_rocket(), _settings(rail_buttons=False), _table())
    assert "pitch_rate_dps" not in outcome.rail_exit
    assert "CG constrained" in outcome.settings.describe_rail()


def test_no_table_means_no_tip_off():
    """The fallback drag law has no restoring moment, so a rate the rail
    left the vehicle with would never be arrested; the CG rail is used."""
    outcome = fly_model(basic_rocket(), _settings(use_aero_table=False))
    assert "pitch_rate_dps" not in outcome.rail_exit
    assert outcome.states[0, 1] == pytest.approx(0.0)


def test_named_buttons_are_used():
    settings = _settings(rail_buttons_m=(0.4, 0.9))
    outcome = fly_model(basic_rocket(), settings, _table())
    assert outcome.simulation.launch_rail.buttons_m == (0.4, 0.9)
    assert "0.40 and 0.90" in settings.describe_rail()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
