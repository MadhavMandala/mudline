"""Winds aloft and turbulence through the flight settings and a dispersion.

Runs under pytest, and standalone via
``python parametric/tests/test_wind_settings.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.dispersion import ModelCaseRunner, dispersions_about  # noqa: E402
from parametric.flight import (  # noqa: E402
    FlightSettings,
    configure_simulation,
    fly_model,
    parse_wind_aloft,
    wind_model_for,
)
from parametric.standard import basic_rocket  # noqa: E402
from trajectory.analysis.dispersion import perturb_simulation  # noqa: E402


def test_winds_aloft_are_read_forgivingly():
    text = "1000 10 270\n# a comment\n2000, 15, 300\n\n500; 8; 250"
    assert parse_wind_aloft(text) == [(500.0, 8.0, 250.0), (1000.0, 10.0, 270.0), (2000.0, 15.0, 300.0)]
    with pytest.raises(ValueError, match="line 2"):
        parse_wind_aloft("1000 10 270\n1000 10")
    with pytest.raises(ValueError, match="not three numbers"):
        parse_wind_aloft("1000 ten 270")
    assert parse_wind_aloft("") == []


def test_calm_smooth_air_builds_no_wind_model():
    assert wind_model_for(FlightSettings()) is None
    assert wind_model_for(FlightSettings(wind_aloft=[(1000.0, 10.0, 90.0)])) is not None
    assert wind_model_for(FlightSettings(turbulence="light")) is not None


def test_the_settings_build_the_sounding_and_the_turbulence():
    settings = FlightSettings(wind_speed_mps=5.0, wind_direction_deg=0.0,
                              wind_aloft=[(1000.0, 10.0, 90.0)], turbulence="moderate",
                              turbulence_seed=11, use_aero_table=False)
    sim = configure_simulation(basic_rocket(), settings)
    assert sim.wind.aloft == [(1000.0, 10.0, 90.0)]
    assert sim.wind.turbulence is not None and sim.wind.turbulence.seed == 11
    assert np.allclose(sim.wind.mean_wind(1000.0), [-10.0, 0.0, 0.0])
    assert "1 level aloft" in settings.describe_wind() and "moderate" in settings.describe_wind()


def test_a_dispersed_surface_wind_keeps_the_sounding_and_reseeds():
    settings = FlightSettings(wind_speed_mps=5.0, wind_aloft=[(1000.0, 10.0, 90.0)],
                              turbulence="light", turbulence_seed=1, use_aero_table=False)
    sim = configure_simulation(basic_rocket(), settings)
    perturb_simulation(sim, {"wind_speed_mps": 8.0, "wind_direction_deg": 45.0, "turbulence_seed": 99.0})
    assert sim.wind.aloft == [(1000.0, 10.0, 90.0)]
    assert sim.wind.surface_speed() == pytest.approx(8.0)
    assert sim.wind.surface_dir == pytest.approx(np.radians(45.0))
    assert sim.wind.turbulence.seed == 99


def test_a_turbulent_study_draws_a_seed_per_case():
    turbulent = dispersions_about(FlightSettings(turbulence="light"), 5.0)
    assert turbulent["turbulence_seed"][0] == "uniform"
    assert "turbulence_seed" not in dispersions_about(FlightSettings(), 5.0)
    runner = ModelCaseRunner(basic_rocket(), FlightSettings(turbulence="light", use_aero_table=False, dt_s=0.5))
    a = runner({"turbulence_seed": 5.0, "t_max": 3.0})
    b = runner({"turbulence_seed": 6.0, "t_max": 3.0})
    assert a["params"]["turbulence_seed"] != b["params"]["turbulence_seed"]


def test_the_log_carries_the_wind_and_the_sounding_is_felt():
    settings = FlightSettings(wind_speed_mps=2.0, wind_aloft=[(1500.0, 25.0, 90.0)],
                              use_aero_table=False, dt_s=0.1)
    outcome = fly_model(basic_rocket(), settings)
    log = outcome.log
    high = np.asarray(log.altitude_agl_m) > 1500.0
    assert np.all(np.asarray(log.wind_mps)[high] == pytest.approx(25.0))
    calm = fly_model(basic_rocket(), FlightSettings(wind_speed_mps=2.0, use_aero_table=False, dt_s=0.1))
    assert outcome.states[-1, 0] < calm.states[-1, 0] - 200.0, "carried west by the wind aloft"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
