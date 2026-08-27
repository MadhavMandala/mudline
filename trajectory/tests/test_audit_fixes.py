"""Regressions from the correctness audit of 25 August 2026: the simulator.

Each of these was found by computing an expected number and comparing, not
by reading -- the class of defect reading had missed. They are kept
together so the audit's findings stay visible as a set.

Runs under pytest, and standalone via
``python trajectory/tests/test_audit_fixes.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory import simulation as tm  # noqa: E402
from trajectory.analysis.dispersion import reduce_outcome, run_dispersion  # noqa: E402
from trajectory.analysis.flightlog import FlightLog  # noqa: E402
from trajectory.analysis.statistics import compute_cepxy, landing_ellipse  # noqa: E402
from trajectory.environment.gravity import G0, gravity_simple  # noqa: E402
from trajectory.environment.wind import WindModel  # noqa: E402
from trajectory.vehicle.engine import Engine  # noqa: E402
from trajectory.vehicle.mass_properties import RADIAL, _column_inertia  # noqa: E402
from trajectory.vehicle.recovery import Parachute, RecoverySystem, standard_recovery  # noqa: E402


# ------------------------------------------------------------ environment


def test_the_wind_profile_is_read_at_height_above_the_pad():
    """A 10 m/s surface wind was applied as 20 m/s on a 2,000 m pad."""
    sim = tm.RocketSimulation()
    sim.wind = WindModel(surface_wind=np.array([10.0, 0.0]), surface_dir=np.pi / 2)
    pad = 2000.0
    sim.launch_rail = tm.LaunchRail(position_m=np.array([0.0, pad, 0.0]), length_m=0.0)
    state = np.zeros(tm.STATE_SIZE)
    state[6] = 1.0
    state[tm.PROP_IDX] = sim.mass_props.prop_mass

    state[1] = pad + 10.0                      # the reference height above the pad
    at_ten = sim.evaluate(state, 0.0)
    assert np.isclose(np.linalg.norm(at_ten.wind_inertial_mps), 10.0)

    state[1] = pad                             # on the pad itself
    assert np.linalg.norm(sim.evaluate(state, 0.0).wind_inertial_mps) < 1e-9


def test_the_power_law_follows_the_log_law_below_the_reference_height():
    """The floor held the full surface wind down to the pad; a rail at 3 m
    sees three quarters of it."""
    power = WindModel(surface_wind=np.array([10.0, 0.0]), profile_type="power_law")
    log = WindModel(surface_wind=np.array([10.0, 0.0]), profile_type="log")
    for height in (1.0, 3.0, 5.0):
        assert np.isclose(np.linalg.norm(power.mean_wind(height)),
                          np.linalg.norm(log.mean_wind(height)))
        assert np.linalg.norm(power.mean_wind(height)) < 10.0
    assert np.isclose(np.linalg.norm(power.mean_wind(10.0)), 10.0)
    assert np.isclose(np.linalg.norm(power.mean_wind(100.0)), 10.0 * 10.0 ** 0.14)


def test_surface_gravity_is_standard_gravity():
    assert np.isclose(gravity_simple(0.0), G0)
    assert np.isclose(gravity_simple(100_000.0), G0 * (6371000.0 / 6471000.0) ** 2)


# ------------------------------------------------------------- the default


def test_the_default_vehicle_is_in_body_axes():
    """Its dry CG sat ahead of the nose tip, in the model's axes."""
    sim = tm.RocketSimulation()
    assert sim.mass_props.cg_dry[1] < 0.0, "aft of the nose: body +Y is forward"
    inertia = np.diag(sim.mass_props.i_tensor_dry)
    assert inertia[1] < inertia[0] and inertia[1] < inertia[2], "roll on Y is the small one"
    assert sim.thrust_position_body_m[1] < sim.mass_props.cg_dry[1], "nozzle behind the CG"


# ---------------------------------------------------------- phases, events


def _flight(**kwargs):
    sim = tm.RocketSimulation()
    defaults = dict(launch_elevation=np.radians(85.0), dt=1.0,
                    recovery=standard_recovery(dry_mass_kg=50.0, main_deploy_altitude_m=500.0))
    defaults.update(kwargs)
    return sim, sim.run(**defaults)


def test_phases_end_at_their_events():
    """Each phase's end is the next one's start, not the last sample before."""
    _, result = _flight()
    for earlier, later in zip(result.phases, result.phases[1:]):
        assert later["t_start"] == earlier["t_end"]
    assert np.isclose(result.phases[-1]["t_end"], result.t[-1])


def test_a_main_only_system_falls_to_its_deployment_height():
    """The height trigger lived inside the drogue branch, so a main-only
    system deployed at apogee and its deployment height meant nothing."""
    sim = tm.RocketSimulation()
    main = Parachute(cda_m2=22.0, deploy_altitude_m=300.0, inflation_time_s=1.2)
    result = sim.run(launch_elevation=np.radians(85.0), dt=1.0,
                     recovery=RecoverySystem(drogue=None, main=main))
    assert [p["name"] for p in result.phases] == ["ascent", "freefall", "main"]
    opened = next(p for p in result.phases if p["name"] == "main")
    i = int(np.searchsorted(result.t, opened["t_start"]))
    assert abs(result.y[1, i] - 300.0) < 5.0
    assert result.landed


def test_window_joins_never_repeat_a_sample():
    sim = tm.RocketSimulation()
    sim.phase_window_s = 20.0
    result = sim.run(launch_elevation=np.radians(85.0), dt=0.5,
                     recovery=standard_recovery(dry_mass_kg=50.0, main_deploy_altitude_m=500.0))
    assert np.all(np.diff(result.t) > 0.0)
    assert result.rail_exit["exact"]


def test_the_flight_lands_exactly_on_the_pad():
    """The ground event used to root 0.1 m under the pad."""
    pad = 1400.0
    _, result = _flight(pad_position_m=np.array([0.0, pad, 0.0]))
    assert result.landed
    assert abs(result.y[1, -1] - pad) < 1e-6


def test_a_slow_start_does_not_trip_the_ground_event_at_ignition():
    """The 0.1 m offset existed to guard this case; the arming does it now."""
    sim = tm.RocketSimulation()
    sim.engine = Engine(
        thrust_curve=np.array([0.0, 20000.0, 20000.0, 0.0]),
        time_points=np.array([0.0, 0.5, 30.0, 31.0]),
        isp_vac=280.0, isp_sl=250.0, nozzle_area=0.0, thrust_reference="vacuum",
    )
    result = sim.run(launch_elevation=np.radians(85.0), t_max=5.0, dt=0.1)
    assert not result.landed, "the ground event fired on the pad"
    assert result.t[-1] >= 4.9 - 1e-9
    assert result.y[1, -1] > 100.0


# --------------------------------------------------------------- the mass


def test_a_core_burner_keeps_its_radius_and_opens_a_bore():
    """Roll inertia grows per kilogram as the grain burns out toward the case."""
    radius, length = 0.05, 1.0
    full = _column_inertia(1.0, 1.0, length, radius, RADIAL, roll_axis=1)[1, 1]
    half = _column_inertia(1.0, 0.5, length, radius, RADIAL, roll_axis=1)[1, 1]
    assert np.isclose(full, 0.5 * radius ** 2)
    assert np.isclose(half, 0.5 * (radius ** 2 + 0.5 * radius ** 2))
    assert half > full, "an annulus of the same mass has more roll inertia than a rod"


# ----------------------------------------------------------- statistics


def test_cep_uses_the_sample_deviation_and_the_stated_constant():
    rng = np.random.default_rng(3)
    cloud = rng.normal(0.0, [120.0, 45.0], size=(50, 2))
    cep, _, sx, sy = compute_cepxy(cloud)
    assert np.isclose(sx, np.std(cloud[:, 0], ddof=1))
    assert np.isclose(cep, 0.5887 * (sx + sy))


def test_the_ellipse_orientation_is_a_bearing_not_a_sign_accident():
    rng = np.random.default_rng(5)
    angle = np.radians(30.0)
    raw = rng.normal(0.0, [200.0, 50.0], size=(4000, 2))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    _, _, _, orientation = landing_ellipse(raw @ rotation.T)
    assert 0.0 <= orientation < np.pi
    assert abs(np.degrees(orientation) - 30.0) < 3.0


# ------------------------------------------------------------ dispersion


def test_a_cut_off_flight_is_not_a_landing():
    sim = tm.RocketSimulation()
    result = sim.run(launch_elevation=np.radians(85.0), t_max=60.0, dt=0.5)
    assert reduce_outcome(result, {})["landed"] is False
    try:
        run_dispersion(n_cases=2, fixed={"t_max": 60.0, "dt": 0.5}, seed=1)
    except RuntimeError as exc:
        assert "ground" in str(exc)
    else:
        raise AssertionError("truncated cases must not make a landing ellipse")
    batch = run_dispersion(n_cases=2, fixed={"t_max": 60.0, "dt": 0.5}, seed=1,
                           require_landing=False)
    assert batch.n_cases == 2 and batch.discarded == 0


# ------------------------------------------------------------- the log


def test_burnout_is_the_end_of_the_last_burn():
    sim = tm.RocketSimulation()
    sim.engine = Engine(
        thrust_curve=np.array([0.0, 20000.0, 20000.0, 0.0, 0.0, 15000.0, 15000.0, 0.0]),
        time_points=np.array([0.0, 0.1, 5.0, 5.5, 6.0, 6.1, 25.0, 26.0]),
        isp_vac=280.0, isp_sl=280.0, nozzle_area=0.0, thrust_reference="vacuum",
    )
    sim.mass_props.prop_mass = 150.0
    sim.mass_props.mass_0 = 200.0
    result = sim.run(launch_elevation=np.radians(90.0), t_max=40.0, dt=0.5)
    log = FlightLog.from_flight(sim, result)
    assert 25.0 <= log.time_s[log.burnout_index] <= 26.5


def test_the_csv_mass_column_is_the_vehicle_not_the_propellant(tmp_path):
    from trajectory.analysis.export import write_trajectory_csv

    sim = tm.RocketSimulation()
    result = sim.run(launch_elevation=np.radians(90.0), t_max=20.0, dt=1.0)
    log = FlightLog.from_flight(sim, result)
    path = write_trajectory_csv(result, tmp_path / "f.csv", log=log)
    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split(",")
    first = dict(zip(header, lines[1].split(",")))
    assert np.isclose(float(first["mass_kg"]), sim.mass_props.mass_0, rtol=1e-4)
    assert np.isclose(float(first["mass_kg"]), float(first["mass_total_kg"]), rtol=1e-6)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
