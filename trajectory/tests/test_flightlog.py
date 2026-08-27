"""The flight log: what the integrator computed and threw away, kept.

Runs under pytest, and standalone via
``python trajectory/tests/test_flightlog.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory import simulation as tm  # noqa: E402
from trajectory.analysis.flightlog import G0, FlightLog  # noqa: E402
from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase  # noqa: E402
from trajectory.vehicle.recovery import standard_recovery  # noqa: E402


def _ascent(sim=None, **run):
    """The default vehicle, boost and coast only, coarse and quick."""
    sim = sim or tm.RocketSimulation()
    kwargs = dict(launch_elevation=np.radians(90.0), t_max=120.0, dt=0.5)
    kwargs.update(run)
    result = sim.run(**kwargs)
    return sim, result, FlightLog.from_flight(sim, result)


def _table() -> AeroDatabase:
    rows = [
        AeroCoefficients(mach=m, alpha_deg=a, cd=0.4, cn=0.09 * a, cm=0.0, x_cp_m=2.0)
        for m in (0.0, 1.0, 3.0, 8.0) for a in (0.0, 5.0, 10.0)
    ]
    return AeroDatabase(rows, reference_length_m=3.0)


def test_the_log_covers_every_sample():
    _, result, log = _ascent()
    assert len(log) == len(result.t)
    assert np.array_equal(log.time_s, result.t)
    assert len(log.phase) == len(result.t)


def test_thrust_and_mass_follow_the_burn():
    """The tank runs dry before the 30 s curve does; the log follows the tank."""
    sim, _, log = _ascent()
    burnout = log.burnout_index
    assert burnout is not None
    expected = sim.mass_props.prop_mass / sim.engine.mass_flow_at(1.0)
    assert abs(log.time_s[burnout] - expected) <= 0.5 + 1e-9
    burn = log.time_s < log.time_s[burnout] - 0.5
    coast = log.time_s >= log.time_s[burnout]
    assert np.all(log.thrust_n[burn] > 0.0)
    assert np.all(log.thrust_n[coast] == 0.0)
    assert np.all(np.diff(log.mass_kg[burn]) < 0.0)
    assert np.allclose(np.diff(log.mass_kg[coast]), 0.0)


def test_a_vertical_flight_in_still_air_has_no_angle_of_attack():
    _, _, log = _ascent()
    flying = log.airspeed_mps > 1.0
    assert np.max(log.alpha_deg[flying]) < 0.5


def test_acceleration_is_what_the_airframe_feels():
    """Thrust over mass at ignition, not thrust minus weight."""
    sim, _, log = _ascent()
    pressure = sim.atm.get_conditions(0.0)[1]
    thrust, _ = sim.engine.thrust_at(0.0, pressure)
    expected = thrust / (sim.mass_props.mass_0 * G0)
    assert np.isclose(log.acceleration_g[0], expected, rtol=1e-6)
    assert np.isclose(log.axial_g[0], expected, rtol=1e-6)
    coast = log.time_s > 40.0
    assert np.all(log.acceleration_g[coast] < 1.0), "only drag once the motor is out"


def test_the_static_margin_needs_a_table():
    _, _, without = _ascent()
    assert np.all(np.isnan(without.static_margin_cal))
    assert np.all(np.isnan(without.cp_station_m))

    sim = tm.RocketSimulation()
    sim.set_aero_database(_table())
    _, _, with_table = _ascent(sim)
    free = ~with_table.on_rail & (with_table.airspeed_mps > 1.0)
    assert np.all(np.isfinite(with_table.static_margin_cal[free]))
    i = int(np.flatnonzero(free)[0])
    assert np.isclose(with_table.cp_station_m[i], 2.0)
    expected = (2.0 - with_table.cg_station_m[i]) / with_table.reference_diameter_m
    assert np.isclose(with_table.static_margin_cal[i], expected)
    assert np.isfinite(with_table.min_static_margin_cal())


def test_the_rail_is_where_the_log_says():
    _, result, log = _ascent(rail_length_m=5.0)
    assert log.rail_exit_index is not None
    assert log.on_rail[0]
    exit_t = log.time_s[log.rail_exit_index]
    assert np.isclose(exit_t, result.rail_exit["time_s"], atol=0.5 + 1e-9)


def test_recovery_phases_are_replayed():
    sim = tm.RocketSimulation()
    recovery = standard_recovery(dry_mass_kg=50.0, main_deploy_altitude_m=500.0)
    result = sim.run(launch_elevation=np.radians(85.0), dt=1.0, recovery=recovery)
    as_found = (sim._active_chute, sim._deploy_trigger_s)
    log = FlightLog.from_flight(sim, result)
    phases = np.asarray(log.phase)
    assert set(phases) == {"ascent", "drogue", "main"}
    assert np.all(log.chute_cda_m2[phases == "ascent"] == 0.0)
    # The first sample of a chute phase sits at the foot of its inflation
    # ramp; from the second on the canopy is open.
    assert np.all(log.chute_cda_m2[phases == "main"][2:] > 0.0)
    assert np.all(log.chute_cda_m2[phases == "drogue"][2:] > 0.0)
    # Replaying must leave the simulation as it found it -- which after a
    # flight is holding the last canopy, not nothing.
    assert (sim._active_chute, sim._deploy_trigger_s) == as_found


def test_the_csv_carries_the_log(tmp_path):
    from trajectory.analysis.export import write_trajectory_csv

    _, result, log = _ascent()
    path = write_trajectory_csv(result, tmp_path / "flight.csv", log=log)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    assert {"alpha_deg", "thrust_n", "acceleration_g", "static_margin_cal", "phase"} <= set(header)
    assert len(lines) == len(log) + 1
    # NaN is blank, not the word.
    assert "nan" not in path.read_text(encoding="utf-8").lower()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
