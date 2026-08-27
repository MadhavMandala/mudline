"""Tests for wind profiles, event detectors and the cm moment fallback.

Covers three loose ends found in the audit:

* the logarithmic wind profile divided by zero at ground level and reversed
  direction below the roughness length;
* ``cm`` was parsed from every aero CSV, stored, and never used, so a table
  that describes its pitching moment through cm rather than x_cp produced a
  vehicle with no restoring moment at all;
* several event detectors were defined and never referenced, so nothing
  checked whether they fired correctly.

Runs under pytest, and standalone via
``python trajectory/tests/test_environment_and_events.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.environment.wind import WindModel  # noqa: E402
from trajectory.sim.events import EventDetector  # noqa: E402
from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase  # noqa: E402
from trajectory.vehicle.aero_model import RasaeroAeroModel  # noqa: E402


# ------------------------------------------------------------------- wind


def _log_wind() -> WindModel:
    return WindModel(surface_wind=np.array([10.0, 0.0]), profile_type="log")


def test_log_profile_is_finite_at_ground_level():
    """log(0/z0) is -inf, which used to poison every force in the simulation."""
    wind = _log_wind().mean_wind(0.0)
    assert np.all(np.isfinite(wind))


def test_log_profile_never_reverses_near_the_ground():
    """Between 0 and the roughness length the raw logarithm goes negative."""
    wind = _log_wind()
    for altitude in [0.0, 0.001, 0.01, 0.04, 0.05, 0.1]:
        along = np.dot(wind.mean_wind(altitude), wind.blows_toward())
        assert along >= 0.0, altitude


def test_log_profile_is_finite_and_increasing_with_height():
    wind = _log_wind()
    speeds = [np.linalg.norm(wind.mean_wind(h)) for h in [1, 10, 100, 1000, 10000]]
    assert np.all(np.isfinite(speeds))
    assert speeds == sorted(speeds)


def test_log_profile_matches_the_reference_height():
    """At z_ref the profile must return exactly the surface wind."""
    wind = _log_wind()
    assert np.isclose(np.linalg.norm(wind.mean_wind(wind.z_ref)), 10.0)


def test_power_law_is_finite_at_ground_level():
    wind = WindModel(surface_wind=np.array([10.0, 0.0]), profile_type="power_law")
    assert np.all(np.isfinite(wind.mean_wind(0.0)))


def test_calm_wind_is_zero_everywhere():
    wind = WindModel()
    for h in [0.0, 100.0, 10000.0]:
        assert np.allclose(wind.total_wind(h, 12.0), 0.0)


def test_negative_altitude_does_not_produce_nan():
    assert np.all(np.isfinite(_log_wind().mean_wind(-50.0)))


def test_gust_adds_to_the_mean():
    wind = _log_wind()
    base = wind.mean_wind(100.0)
    wind.set_gust(amplitude=5.0, frequency=0.25, phase=np.pi / 2)
    assert not np.allclose(wind.total_wind(100.0, 0.0), base)
    wind.clear_gust()
    assert np.allclose(wind.total_wind(100.0, 0.0), base)


def test_wind_from_the_north_blows_south():
    """A meteorological bearing says where the wind comes *from*.

    The conversion dropped both minus signs, so every wind blew toward the
    bearing it was said to come from -- 180 degrees wrong.
    """
    wind = WindModel(surface_wind=np.array([10.0, 0.0]), surface_dir=0.0)
    east, _, north = wind.mean_wind(wind.z_ref)
    assert np.isclose(east, 0.0, atol=1e-9)
    assert np.isclose(north, -10.0)


def test_wind_from_the_east_blows_west():
    wind = WindModel(surface_wind=np.array([10.0, 0.0]), surface_dir=np.pi / 2)
    east, _, north = wind.mean_wind(wind.z_ref)
    assert np.isclose(east, -10.0)
    assert np.isclose(north, 0.0, atol=1e-9)


def test_apogee_arms_above_the_ground_not_above_the_sea():
    """A pad at 2,000 m must not arm the event while the vehicle sits on it."""
    event = EventDetector.apogee(arm_altitude_m=10.0, ground_m=2000.0)
    on_pad = _state(altitude=2000.0, vy=0.0)
    assert event(0.0, on_pad) > 0.0
    climbing = _state(altitude=2015.0, vy=40.0)
    assert event(1.0, climbing) == 40.0


def test_a_gust_blows_along_the_wind_not_along_east():
    wind = WindModel(surface_wind=np.array([10.0, 0.0]), surface_dir=np.radians(30.0))
    wind.set_gust(amplitude=5.0, frequency=0.25, phase=np.pi / 2)
    gust = wind.gust(0.0)
    mean = wind.mean_wind(100.0)
    assert np.isclose(np.linalg.norm(gust), 5.0)
    assert np.allclose(np.cross(gust, mean), 0.0, atol=1e-9)
    assert np.dot(gust, mean) > 0.0


# ----------------------------------------------------------------- events


def _state(altitude=1000.0, vy=100.0, x=0.0, z=0.0):
    s = np.zeros(14)
    s[0], s[1], s[2], s[4] = x, altitude, z, vy
    return s


def test_apogee_event_is_terminal_and_downward_directed():
    event = EventDetector.apogee()
    assert event.terminal and event.direction == -1
    assert event(0.0, _state(vy=50.0)) > 0.0     # climbing
    assert event(0.0, _state(vy=-50.0)) < 0.0    # descending


def test_ground_event_changes_sign_at_the_surface():
    event = EventDetector.ground()
    assert event(0.0, _state(altitude=10.0)) > 0.0
    assert event(0.0, _state(altitude=-1.0, vy=-5.0)) < 0.0
    assert event(0.0, _state(altitude=0.0, vy=-5.0)) == 0.0, "the root is the ground itself"


def test_ground_event_is_not_a_root_on_the_pad():
    """At rest on the pad, or climbing out of it, there is nothing to hit."""
    event = EventDetector.ground(ground_m=1400.0)
    assert event(0.0, _state(altitude=1400.0, vy=0.0)) > 0.0
    assert event(0.0, _state(altitude=1400.3, vy=2.0)) > 0.0
    assert event(0.0, _state(altitude=1400.3, vy=-2.0)) > 0.0
    assert event(0.0, _state(altitude=1399.9, vy=-2.0)) < 0.0


def test_descending_through_fires_only_downward():
    event = EventDetector.descending_through(500.0)
    assert event.terminal and event.direction == -1
    assert event(0.0, _state(altitude=900.0)) > 0.0
    assert event(0.0, _state(altitude=100.0)) < 0.0


def test_max_altitude_event_triggers_on_a_ceiling():
    event = EventDetector.max_altitude(5000.0)
    assert event(0.0, _state(altitude=1000.0)) > 0.0
    assert event(0.0, _state(altitude=6000.0)) < 0.0


def test_max_range_event_measures_horizontal_distance_only():
    event = EventDetector.max_range(1000.0)
    assert event(0.0, _state(x=300.0, z=400.0)) < 0.0     # 500 m out
    assert event(0.0, _state(x=3000.0, z=4000.0)) > 0.0   # 5000 m out
    # Altitude must not count toward downrange.
    assert event(0.0, _state(altitude=50000.0)) < 0.0


def test_burnout_event_detects_thrust_falling_away():
    event = EventDetector.burnout(lambda t: 1000.0 if t < 10.0 else 0.0)
    assert not event.terminal          # observational, not terminating
    assert event(5.0, _state()) > 0.0
    assert event(15.0, _state()) < 0.0


def test_max_range_event_is_usable_as_a_run_limit():
    """The detectors are reachable through run(events=...)."""
    from trajectory import simulation as tm
    sim = tm.RocketSimulation()
    result = sim.run(
        launch_elevation=np.radians(45), t_max=300.0,
        events=[EventDetector.max_range(3000.0)],
    )
    downrange = np.linalg.norm(result.y[[0, 2], -1])
    assert downrange < 3600.0, downrange


# ------------------------------------------------------ cm moment fallback


def _table(x_cp: float, cm_per_deg: float) -> AeroDatabase:
    rows = []
    for mach in [0.0, 1.0, 3.0]:
        for alpha in [0.0, 5.0, 10.0]:
            rows.append(AeroCoefficients(
                mach=mach, alpha_deg=alpha, cd=0.4, cn=0.09 * alpha,
                cm=cm_per_deg * alpha, x_cp_m=x_cp,
            ))
    return AeroDatabase(rows, reference_length_m=3.0)


def _model(db) -> RasaeroAeroModel:
    return RasaeroAeroModel(db, reference_area_m2=0.07, reference_length_m=3.0,
                            body_axis=np.array([0.0, 1.0, 0.0]))


def _flow(alpha_deg=5.0, speed=200.0):
    a = np.radians(alpha_deg)
    return np.array([speed * np.sin(a), speed * np.cos(a), 0.0])


def test_database_reports_which_moment_data_it_has():
    assert _table(x_cp=2.0, cm_per_deg=0.0).has_x_cp
    assert not _table(x_cp=0.0, cm_per_deg=-0.01).has_x_cp
    assert _table(x_cp=0.0, cm_per_deg=-0.01).has_cm


def test_cm_only_table_still_produces_a_restoring_moment():
    """Previously this vehicle had no static moment at all."""
    model = _model(_table(x_cp=0.0, cm_per_deg=-0.01))
    aero = model.forces_and_moments(_flow(), 1.0, 340.0, np.array([0.0, 2.5, 0.0]))
    assert np.linalg.norm(aero.static_moment_body_nm) > 0.0


def test_x_cp_table_is_preferred_when_both_are_present():
    """x_cp tracks CG migration; cm is referenced to a fixed point."""
    model = _model(_table(x_cp=2.0, cm_per_deg=-0.01))
    forward = model.forces_and_moments(_flow(), 1.0, 340.0, np.array([0.0, 2.2, 0.0]))
    aft = model.forces_and_moments(_flow(), 1.0, 340.0, np.array([0.0, 3.0, 0.0]))
    # Moving the CG must change the moment; a cm-only model could not see it.
    assert not np.allclose(forward.static_moment_body_nm, aft.static_moment_body_nm)


def test_cm_fallback_still_damps():
    model = _model(_table(x_cp=0.0, cm_per_deg=-0.01))
    omega = np.array([0.0, 0.0, 0.5])
    aero = model.forces_and_moments(_flow(), 1.0, 340.0, np.array([0.0, 2.5, 0.0]), omega)
    assert np.dot(aero.damping_moment_body_nm, omega) <= 0.0


if __name__ == "__main__":
    failures = 0
    names = sorted(n for n in globals() if n.startswith("test_"))
    for name in names:
        try:
            globals()[name]()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(names) - failures}/{len(names)} passed")
    raise SystemExit(1 if failures else 0)
