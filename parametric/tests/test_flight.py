"""Tests for the one launch sequence every entry point flies.

Run Flight, the design sweep and the dispersion study each used to carry
their own copy of the launch, and the copies disagreed: the sweep dropped
wind, azimuth and the pad and integrated coarser; the dispersion flew the
simulator's placeholder vehicle. These pin the single path -- and the pad
altitude, which used to move nothing but the coupled-aero profile.

Runs under pytest, and standalone via ``python parametric/tests/test_flight.py``.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.flight import FlightSettings, fly_model  # noqa: E402
from parametric.standard import basic_rocket  # noqa: E402


def _settings(**overrides) -> FlightSettings:
    # The fallback drag law: no table to build, so these run without CAD
    # and in a second or so each.
    base = dict(use_aero_table=False, dt_s=0.1)
    base.update(overrides)
    return FlightSettings(**base)


# ------------------------------------------------------------- the pad


def test_the_flight_leaves_from_and_returns_to_the_pad():
    outcome = fly_model(basic_rocket(), _settings(pad_altitude_m=1500.0))
    states = outcome.states
    assert states[0, 1] == pytest.approx(1500.0)
    assert outcome.landed
    assert states[-1, 1] == pytest.approx(1500.0, abs=0.5)
    assert outcome.apogee_agl_m == pytest.approx(outcome.stats["max_altitude"] - 1500.0)
    assert np.allclose(outcome.positions_from_pad()[0], 0.0)


def test_the_main_opens_at_the_set_height_above_the_pad():
    settings = _settings(pad_altitude_m=1500.0, main_deploy_altitude_m=200.0)
    outcome = fly_model(basic_rocket(), settings)
    main = next(p for p in outcome.result.phases if p["name"] == "main")
    i = int(np.searchsorted(outcome.times, main["t_start"]))
    assert outcome.states[i, 1] == pytest.approx(1700.0, abs=5.0)


def test_thin_air_on_a_high_pad_raises_the_apogee():
    """The pad used to be cosmetic: every flight left from sea level."""
    low = fly_model(basic_rocket(), _settings(pad_altitude_m=0.0))
    high = fly_model(basic_rocket(), _settings(pad_altitude_m=2000.0))
    assert high.apogee_agl_m > low.apogee_agl_m * 1.02


# ------------------------------------------------------------- the wind


def test_a_wind_from_the_east_carries_the_vehicle_west():
    calm = fly_model(basic_rocket(), _settings())
    windy = fly_model(
        basic_rocket(), _settings(wind_speed_mps=10.0, wind_direction_deg=90.0)
    )
    assert windy.states[-1, 0] < calm.states[-1, 0] - 100.0


def test_the_log_rides_with_the_outcome():
    outcome = fly_model(basic_rocket(), _settings())
    log = outcome.log
    assert log is not None and len(log) == len(outcome.times)
    assert log.max_acceleration_g > 1.0
    assert np.isnan(log.static_margin_cal).all(), "the fallback law has no CP"
    assert outcome.log is log, "built once and kept"


def test_a_finned_rocket_weathercocks_into_the_wind_and_settles():
    """The closed loop: table, restoring moment, right-handed kinematics, damping.

    Two sign errors used to cancel here -- the CP placed ahead of the nose and
    the attitude rotating opposite to the body rate -- and the vehicle flew
    with a frozen attitude. Fixing one without the other tumbled it. This
    flies the case through the real drag table and asks for what a finned
    rocket does: a kick of alpha off the rail, a decay over the next second,
    and a nose that ends up pointing into the wind.
    """
    from parametric import aero
    from trajectory.eom import quat_to_dcm

    model = basic_rocket()
    table, _ = aero.run_analysis(model, aero.AeroSettings(
        mach_min=0.05, mach_max=1.5, mach_points=8, alpha_max_deg=16.0, alpha_points=5,
    ))
    settings = FlightSettings(
        wind_speed_mps=8.0, wind_direction_deg=90.0, dt_s=0.02,
        couple_aero_altitude=False, use_recovery=False,
    )
    outcome = fly_model(model, settings, table)
    alpha = outcome.log.alpha_deg
    t = outcome.times
    exit_t = outcome.result.rail_exit["time_s"]

    kick = alpha[(t > exit_t) & (t < exit_t + 0.5)].max()
    assert 5.0 < kick < 20.0, kick
    one_s = alpha[int(np.searchsorted(t, exit_t + 1.0))]
    three_s = alpha[int(np.searchsorted(t, exit_t + 3.0))]
    assert one_s < 0.3 * kick, (kick, one_s)
    assert three_s < 0.05 * kick, (kick, three_s)

    # Wind from the east: the relative wind has an eastward component, and
    # the nose follows it. Body +Y in the inertial frame leans east.
    state = outcome.states[int(np.searchsorted(t, exit_t + 3.0))]
    axis = quat_to_dcm(state[6:10]) @ np.array([0.0, 1.0, 0.0])
    assert axis[0] > 0.0, axis

    # And the log can say so directly: margin positive throughout the boost.
    log = outcome.log
    assert log.min_static_margin_cal() > 0.5
    burnout = log.burnout_index
    assert burnout is not None and np.isfinite(log.static_margin_cal[burnout])

    # The exit itself: exact, and the alpha the crosswind makes of it. The
    # wind at rail height is below the 10 m surface value -- the profile
    # follows the log law down there -- so the kinematics use the wind the
    # exit actually recorded.
    exit_state = outcome.rail_exit
    assert exit_state["exact"]
    assert 0.6 * 8.0 < exit_state["wind_mps"] < 8.0
    expected = np.degrees(np.arctan2(exit_state["wind_mps"], exit_state["velocity_mps"]))
    assert exit_state["alpha_deg"] == pytest.approx(expected, abs=0.3)
    assert outcome.rail_check() == [], "16 deg covers a 13 deg exit"


def test_the_rail_check_names_an_uncovered_exit_alpha():
    from parametric import aero

    model = basic_rocket()
    narrow, _ = aero.run_analysis(model, aero.AeroSettings(
        mach_min=0.05, mach_max=1.5, mach_points=8, alpha_max_deg=8.0, alpha_points=3,
    ))
    settings = FlightSettings(
        wind_speed_mps=8.0, wind_direction_deg=90.0, dt_s=0.05,
        couple_aero_altitude=False, use_recovery=False,
    )
    outcome = fly_model(model, settings, narrow)
    notes = outcome.rail_check()
    assert len(notes) == 1 and "8 deg" in notes[0], notes
    assert outcome.rail_exit_alpha_deg > 8.0


def test_the_plume_filled_base_flies_higher():
    """Less base drag while burning; the same drag after. Apogee goes up."""
    from parametric import aero

    model = basic_rocket()
    grid = dict(mach_min=0.05, mach_max=1.5, mach_points=8,
                alpha_max_deg=8.0, alpha_points=3)
    settings = _settings(use_aero_table=True, couple_aero_altitude=False,
                         use_recovery=False, dt_s=0.05)
    single, _ = aero.run_analysis(model, aero.AeroSettings(power_on_base=False, **grid))
    both, _ = aero.run_analysis(model, aero.AeroSettings(power_on_base=True, **grid))
    coasting = fly_model(model, settings, single)
    burning = fly_model(model, settings, both)
    assert burning.used_table and coasting.used_table
    assert burning.apogee_agl_m > coasting.apogee_agl_m


# ------------------------------------------------------- the derivatives


def _table(model, alpha_max=8.0):
    from parametric import aero

    table, _ = aero.run_analysis(model, aero.AeroSettings(
        mach_min=0.05, mach_max=1.5, mach_points=8, alpha_max_deg=alpha_max,
        alpha_points=3,
    ))
    return table


def test_the_table_carries_damping_and_roll_derivatives():
    model = basic_rocket()
    table = _table(model)
    assert table.has_damping and table.has_roll
    row = table.lookup(0.3, 0.0)
    assert row.cna_sum > 0.0 and row.cna_x2_m2 > 0.0
    assert row.clp < 0.0, "three fins damp roll"
    assert row.cl_roll == 0.0, "no cant, no forcing"

    model.fin_sets[0].set("cant", 2.0)
    canted = _table(model).lookup(0.3, 0.0)
    assert canted.cl_roll > 0.0
    assert np.isclose(canted.clp, row.clp)


def test_roll_damping_matches_a_numerical_span_integral():
    from parametric.aero import roll_derivatives

    fins = basic_rocket().fin_sets[0]
    clp, _, _ = roll_derivatives(fins, cn_alpha_panel=3.0, diameter_m=0.1)
    r, s = fins.body_radius_m(), fins.get("span")
    xi = np.linspace(r, r + s, 20001)
    chord = fins.get("root_chord") + (fins.get("tip_chord") - fins.get("root_chord")) * (xi - r) / s
    integral = np.trapezoid(xi ** 2 * chord, xi)
    expected = -2.0 * fins.count * 3.0 * integral / (fins.area_per_fin_m2 * 0.1 ** 2)
    assert np.isclose(clp, expected, rtol=1e-6)


def test_component_damping_exceeds_the_lumped_estimate_on_the_basic_rocket():
    from parametric.flight import configure_simulation

    model = basic_rocket()
    table = _table(model)
    sim = configure_simulation(model, _settings(use_aero_table=True), table)
    aero_model = sim.aero_model
    _, cg, _ = sim.mass_props.at_propellant(sim.mass_props.prop_mass)
    coeffs = table.lookup(0.3, 0.0)
    summed = aero_model.cmq_from_table(coeffs, cg)
    forces = aero_model.forces_and_moments(
        np.array([0.0, 100.0, 0.0]), 1.2, 340.0, cg,
    )
    lumped = aero_model.estimate_cmq(0.3, 0.0, forces.static_margin_m)
    assert summed < 0.0 and abs(summed) > 1.5 * abs(lumped), (summed, lumped)


def test_an_uncanted_rocket_does_not_roll():
    model = basic_rocket()
    outcome = fly_model(
        model, _settings(use_aero_table=True, use_recovery=False, dt_s=0.05,
                         couple_aero_altitude=False, elevation_deg=90.0),
        _table(model),
    )
    assert np.max(np.abs(outcome.log.roll_rate_radps)) < 1e-6


def test_a_canted_fin_set_rolls_at_the_closed_form_rate():
    """Forcing against damping, quasi-steady once the roll time constant
    (a fraction of a second here) has passed."""
    model = basic_rocket()
    model.fin_sets[0].set("cant", 2.0)
    table = _table(model)
    outcome = fly_model(
        model, _settings(use_aero_table=True, use_recovery=False, dt_s=0.02,
                         couple_aero_altitude=False, elevation_deg=90.0),
        table,
    )
    log = outcome.log
    i = log.index_at(1.2)
    speed, mach = log.airspeed_mps[i], log.mach[i]
    row = table.lookup(mach, 0.0)
    expected = -2.0 * speed * row.cl_roll / (log.reference_diameter_m * row.clp)
    measured = log.roll_rate_radps[i]
    assert expected > 1.0, expected
    assert measured == pytest.approx(expected, rel=0.15), (measured, expected)


def test_jet_damping_slows_a_pitching_burner_in_vacuum():
    """Off the rail before there is airspeed, the exhaust is what damps."""
    from parametric.flight import configure_simulation
    from trajectory import simulation as tm

    model = basic_rocket()
    sim = configure_simulation(model, _settings())
    state = np.zeros(tm.STATE_SIZE)
    state[1] = 150_000.0
    state[6] = 1.0
    state[10] = 0.3
    state[tm.PROP_IDX] = sim.mass_props.prop_mass
    burning = sim.state_derivative(0.5, state)
    assert burning[10] < 0.0, "the pitch rate decays while the motor burns"
    state[tm.PROP_IDX] = 0.0
    coasting = sim.state_derivative(0.5, state)
    assert abs(coasting[10]) < 1e-9


# ------------------------------------------------------ beyond the table


def test_the_table_carries_the_planform_for_the_extension():
    model = basic_rocket()
    table = _table(model)
    shape = table.high_alpha
    d, length = model.max_diameter_m, model.total_length_m
    assert 0.6 * d * length < shape.planform_area_m2 < d * length
    assert 0.3 * length < shape.planform_centroid_m < 0.7 * length
    assert shape.nose_length_m > 0.0
    fins = model.fin_sets[0]
    # Half the panels: roll-averaged, a cruciform set's cos^2 sums to N/2.
    assert np.isclose(shape.fin_area_m2, 0.5 * fins.count * fins.area_per_fin_m2)
    low, high = fins.station_range_m()
    assert low <= shape.fin_centroid_m <= high


def test_broadside_the_basic_rocket_is_a_flat_plate_of_its_planform():
    from parametric.flight import configure_simulation
    from trajectory.vehicle.aero_model import (
        FLAT_PLATE_CN_90, crossflow_drag_coefficient, crossflow_proportionality,
    )

    model = basic_rocket()
    table = _table(model)
    sim = configure_simulation(model, _settings(use_aero_table=True), table)
    _, cg, _ = sim.mass_props.at_propellant(0.0)
    rho, speed = 1.2, 30.0
    forces = sim.aero_model.forces_and_moments(
        np.array([speed, 0.0, 0.0]), rho, 340.0, cg,
    )
    shape = table.high_alpha
    eta = crossflow_proportionality(shape.length_m / shape.diameter_m)
    expected_cn = (
        eta * crossflow_drag_coefficient(0.0) * shape.planform_area_m2
        + FLAT_PLATE_CN_90 * shape.fin_area_m2
    ) / sim.reference_area
    q = 0.5 * rho * speed ** 2
    assert np.isclose(np.linalg.norm(forces.force_body_n), q * sim.reference_area * expected_cn,
                      rtol=1e-6)
    assert abs(forces.ca_applied) < 1e-9


def test_the_extension_is_continuous_on_the_basic_rocket():
    from parametric.flight import configure_simulation

    model = basic_rocket()
    table = _table(model, alpha_max=16.0)
    sim = configure_simulation(model, _settings(use_aero_table=True), table)
    _, cg, _ = sim.mass_props.at_propellant(sim.mass_props.prop_mass)
    edge = table.alpha_range_deg[1]
    for boundary in (edge, edge + 15.0):
        pair = []
        for alpha in (boundary - 0.01, boundary + 0.01):
            a = np.radians(alpha)
            pair.append(sim.aero_model.forces_and_moments(
                np.array([100.0 * np.sin(a), 100.0 * np.cos(a), 0.0]), 1.2, 340.0, cg,
            ))
        assert np.allclose(pair[0].force_body_n, pair[1].force_body_n, rtol=2e-2)
        assert np.allclose(pair[0].static_moment_body_nm, pair[1].static_moment_body_nm,
                           rtol=2e-2, atol=1e-6)


# ------------------------------------------------------------ the sweep


def test_the_sweep_flies_the_same_flight_as_run_flight():
    from parametric.sweep import _fly

    settings = _settings(
        wind_speed_mps=6.0, wind_direction_deg=45.0, pad_altitude_m=800.0,
        elevation_deg=80.0,
    )
    model = basic_rocket()
    swept = _fly(model, None, settings)
    direct = fly_model(model, settings)
    assert swept["apogee m"] == pytest.approx(direct.apogee_agl_m)
    assert swept["rail exit m/s"] == pytest.approx(direct.rail_exit_mps)
    assert swept["max-Q kPa"] == pytest.approx(direct.peak["pressure_pa"] / 1000.0)


# ------------------------------------------------------- the dispersion


def test_the_runner_flies_the_model_not_the_placeholder():
    from parametric.dispersion import ModelCaseRunner

    model = basic_rocket()
    runner = ModelCaseRunner(model, _settings(dt_s=0.2))
    outcome = runner({
        "dry_mass_kg": model.mass_summary().dry_mass_kg, "t_max": 400.0,
    })
    # The placeholder reaches ~150 km; the basic rocket a few.
    assert 1000.0 < outcome["max_altitude"] < 10000.0
    assert outcome["success"]


def test_the_runner_survives_the_trip_to_a_worker():
    from parametric.dispersion import ModelCaseRunner

    runner = pickle.loads(pickle.dumps(
        ModelCaseRunner(basic_rocket(), _settings(dt_s=0.2))
    ))
    assert runner({"t_max": 60.0})["success"]


def test_the_runner_never_couples_the_table_per_case():
    from parametric.dispersion import ModelCaseRunner

    runner = ModelCaseRunner(basic_rocket(), _settings(couple_aero_altitude=True))
    assert runner.settings.couple_aero_altitude is False


def test_the_runner_quotes_apogee_above_the_pad():
    from parametric.dispersion import ModelCaseRunner

    model = basic_rocket()
    sea = ModelCaseRunner(model, _settings(dt_s=0.2))({"t_max": 400.0})
    high = ModelCaseRunner(model, _settings(dt_s=0.2, pad_altitude_m=3000.0))(
        {"t_max": 400.0}
    )
    assert high["max_altitude"] < 2.0 * sea["max_altitude"]


def test_dispersions_are_centred_on_the_flight_and_the_vehicle():
    from parametric.dispersion import dispersions_about

    settings = _settings(
        elevation_deg=80.0, azimuth_deg=30.0, wind_speed_mps=4.0,
        wind_direction_deg=200.0,
    )
    spread = dispersions_about(settings, 12.5)
    assert spread["launch_elevation_deg"][0] == 80.0
    assert spread["launch_azimuth_deg"][0] == 30.0
    assert spread["wind_speed_mps"][0] == 4.0
    assert spread["wind_direction_deg"][0] == 200.0
    assert spread["dry_mass_kg"][0] == 12.5
    for name, (mean, sd, low, high) in spread.items():
        assert sd > 0.0, name
        assert low <= mean <= high and low < high, name


def test_dispersion_bounds_respect_the_physics():
    from parametric.dispersion import dispersions_about

    spread = dispersions_about(
        _settings(elevation_deg=89.0, wind_speed_mps=0.0), 10.0,
        elevation_sd_deg=2.0, wind_speed_sd_mps=3.0, dry_mass_sd_kg=0.0,
    )
    assert spread["launch_elevation_deg"][3] == 90.0
    assert spread["wind_speed_mps"][2] == 0.0
    _, sd, low, high = spread["dry_mass_kg"]
    assert sd > 0.0 and low < high


def test_a_dispersion_of_the_model_scatters_around_the_model():
    from parametric.dispersion import ModelCaseRunner, dispersions_about
    from trajectory.analysis.dispersion import run_dispersion

    model = basic_rocket()
    settings = _settings(dt_s=0.25)
    result = run_dispersion(
        n_cases=4,
        dispersions=dispersions_about(
            settings, model.mass_summary().dry_mass_kg, dry_mass_sd_kg=0.2,
        ),
        seed=7,
        case_fn=ModelCaseRunner(model, settings),
        fixed={"t_max": 400.0},
    )
    assert result.n_cases == 4
    assert 1000.0 < result.summary["max_altitude"]["mean"] < 10000.0
    assert result.landing_points.std(axis=0).max() > 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
