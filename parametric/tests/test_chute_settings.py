"""The harness attachment through the flight settings.

Runs under pytest, and standalone via
``python parametric/tests/test_chute_settings.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.flight import (  # noqa: E402
    FlightSettings,
    default_attachment_station_m,
    fly_model,
    recovery_for,
)
from parametric.standard import basic_rocket  # noqa: E402


def test_the_default_attachment_is_the_nose_shoulder():
    model = basic_rocket()
    station = default_attachment_station_m(model)
    start, end = model.station_range_m()
    assert start < station < start + 0.5 * (end - start)


def test_the_settings_choose_where_the_canopy_pulls():
    named = recovery_for(5.0, FlightSettings(chute_attachment_station_m=0.42), 0.3)
    assert named.drogue.attachment_station_m == 0.42 and named.main.attachment_station_m == 0.42
    default = recovery_for(5.0, FlightSettings(), 0.3)
    assert default.main.attachment_station_m == 0.3
    through_cg = recovery_for(5.0, FlightSettings(chute_at_attachment=False), 0.3)
    assert through_cg.main.attachment_station_m is None
    assert "nose shoulder" in FlightSettings().describe_recovery()
    assert "through the CG" in FlightSettings(chute_at_attachment=False).describe_recovery()


def test_the_basic_rocket_lands_hanging_from_its_nose():
    outcome = fly_model(basic_rocket(), FlightSettings(use_aero_table=False, dt_s=0.1))
    assert outcome.landed
    hang = outcome.log.hang_angle_deg()
    assert hang is not None and hang < 5.0, hang
    main = outcome.result.phases[-1]
    assert main["name"] == "main"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
