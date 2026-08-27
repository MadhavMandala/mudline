"""Tests for Monte Carlo dispersion.

Pins the gap where MonteCarlo and the CEP / landing-ellipse statistics were
both fully implemented and never called: the sampler had no simulation to
drive, and the statistics had no landing points to consume.

Flights here are deliberately truncated (short t_max, coarse dt) -- these test
the plumbing and the parameter sensitivities, not trajectory accuracy, which
is covered elsewhere.

Runs under pytest, and standalone via
``python trajectory/tests/test_dispersion.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.analysis.dispersion import (  # noqa: E402
    DEFAULT_DISPERSIONS,
    build_case_simulation,
    run_case,
    run_dispersion,
)

# Short flights: enough to fly, cheap enough to test.
FAST = {"t_max": 90.0, "dt": 0.5}


def _case(**overrides) -> dict:
    params = {
        "impulse_scale": 1.0,
        "thrust_scale": 1.0,
        "dry_mass_kg": 50.0,
        "aero_scale": 1.0,
        "launch_elevation_deg": 85.0,
        "launch_azimuth_deg": 0.0,
        "wind_speed_mps": 0.0,
        "wind_direction_deg": 0.0,
        **FAST,
    }
    params.update(overrides)
    return params


# ------------------------------------------------------ parameter plumbing


def test_thrust_scale_reaches_the_engine():
    sim = build_case_simulation(_case(thrust_scale=2.0))
    assert np.isclose(sim.engine.vacuum_thrust_at(10.0), 40000.0)


def test_thrust_scale_leaves_propellant_alone():
    """Thrust alone: total impulse fixed, burn time halved."""
    nominal = build_case_simulation(_case())
    scaled = build_case_simulation(_case(thrust_scale=2.0))
    assert scaled.mass_props.prop_mass == nominal.mass_props.prop_mass
    assert np.isclose(scaled.engine.mass_flow_at(10.0),
                      2 * nominal.engine.mass_flow_at(10.0))


def test_impulse_scale_moves_thrust_and_propellant_together():
    """Impulse: burn time fixed, delivered impulse scaled."""
    nominal = build_case_simulation(_case())
    scaled = build_case_simulation(_case(impulse_scale=1.2))
    assert np.isclose(scaled.engine.vacuum_thrust_at(10.0),
                      1.2 * nominal.engine.vacuum_thrust_at(10.0))
    assert np.isclose(scaled.mass_props.prop_mass,
                      1.2 * nominal.mass_props.prop_mass)
    # Burn time is thrust-independent when both scale together.
    nominal_burn = nominal.mass_props.prop_mass / nominal.engine.mass_flow_at(10.0)
    scaled_burn = scaled.mass_props.prop_mass / scaled.engine.mass_flow_at(10.0)
    assert np.isclose(nominal_burn, scaled_burn)


def test_total_mass_is_consistent_after_any_perturbation():
    for params in [_case(impulse_scale=1.15), _case(dry_mass_kg=61.0),
                   _case(impulse_scale=0.9, dry_mass_kg=44.0)]:
        sim = build_case_simulation(params)
        assert np.isclose(
            sim.mass_props.mass_0,
            sim.mass_props.dry_mass + sim.mass_props.prop_mass,
        )


def test_dry_mass_updates_the_total_mass():
    sim = build_case_simulation(_case(dry_mass_kg=80.0))
    assert sim.mass_props.dry_mass == 80.0
    assert sim.mass_props.mass_0 == 80.0 + sim.mass_props.prop_mass


def test_wind_reaches_the_wind_model():
    sim = build_case_simulation(_case(wind_speed_mps=12.0, wind_direction_deg=90.0))
    assert np.isclose(np.linalg.norm(sim.wind.surface_wind), 12.0)
    assert np.isclose(sim.wind.surface_dir, np.pi / 2)


def test_aero_scale_reaches_the_simulation():
    assert build_case_simulation(_case(aero_scale=1.3)).aero_scale == 1.3


def test_calm_case_leaves_wind_model_at_default():
    sim = build_case_simulation(_case(wind_speed_mps=0.0))
    assert np.allclose(sim.wind.mean_wind(100.0), 0.0)


# ----------------------------------------------------------- sensitivities


def test_more_impulse_flies_higher():
    low = run_case(_case(impulse_scale=0.92))
    high = run_case(_case(impulse_scale=1.08))
    assert high["max_altitude"] > low["max_altitude"]


def test_more_thrust_at_fixed_impulse_flies_lower_on_this_vehicle():
    """Not a typo, and not a bug: this vehicle is drag-dominated.

    thrust_scale holds propellant and Isp fixed, so total impulse -- and
    therefore ideal delta-v -- is unchanged; only burn duration moves. Near
    Mach 4 at 10 km this vehicle sees roughly 16 kN of drag against 20 kN of
    thrust, so reaching a given speed lower in the atmosphere costs more to
    drag than the shorter burn saves against gravity. Documented as a test
    because the direction is counter-intuitive and worth not 'fixing' later.
    """
    slow = run_case(_case(thrust_scale=0.9, t_max=400.0))
    fast = run_case(_case(thrust_scale=1.1, t_max=400.0))
    assert fast["max_altitude"] < slow["max_altitude"]


def test_heavier_vehicle_flies_lower():
    light = run_case(_case(dry_mass_kg=45.0))
    heavy = run_case(_case(dry_mass_kg=56.0))
    assert heavy["max_altitude"] < light["max_altitude"]


def test_more_drag_flies_lower():
    clean = run_case(_case(aero_scale=0.8))
    draggy = run_case(_case(aero_scale=1.25))
    assert draggy["max_altitude"] < clean["max_altitude"]


def test_azimuth_steers_the_landing_point():
    north = run_case(_case(launch_azimuth_deg=0.0, launch_elevation_deg=75.0))
    east = run_case(_case(launch_azimuth_deg=90.0, launch_elevation_deg=75.0))
    assert north["landing_north_m"] > abs(north["landing_east_m"])
    assert east["landing_east_m"] > abs(east["landing_north_m"])


def test_a_case_reports_the_fields_statistics_needs():
    outcome = run_case(_case())
    for key in ("landing_east_m", "landing_north_m", "max_altitude",
                "max_velocity", "flight_time", "apogee_time", "rail_exit_mps"):
        assert key in outcome, key
    assert outcome["success"]


def test_case_records_its_own_parameters():
    outcome = run_case(_case(thrust_scale=1.05))
    assert outcome["params"]["thrust_scale"] == 1.05


# --------------------------------------------------------------- the batch


def _batch(n=6, **kwargs):
    return run_dispersion(n_cases=n, seed=99, fixed=FAST, require_landing=False, **kwargs)


def test_batch_runs_every_case():
    result = _batch(n=6)
    assert result.n_cases == 6
    assert result.landing_points.shape == (6, 2)


def test_batch_is_reproducible_under_a_seed():
    a = run_dispersion(n_cases=4, seed=4242, fixed=FAST, require_landing=False)
    b = run_dispersion(n_cases=4, seed=4242, fixed=FAST, require_landing=False)
    assert np.allclose(a.landing_points, b.landing_points)


def test_different_seeds_give_different_dispersions():
    a = run_dispersion(n_cases=4, seed=1, fixed=FAST, require_landing=False)
    b = run_dispersion(n_cases=4, seed=2, fixed=FAST, require_landing=False)
    assert not np.allclose(a.landing_points, b.landing_points)


def test_dispersion_actually_disperses():
    """Landing points must scatter -- a zero spread means nothing is wired."""
    result = _batch(n=8)
    assert result.landing_points.std(axis=0).max() > 1.0
    assert result.cep_m > 0.0


def test_zero_variance_dispersion_collapses_to_one_point():
    """The control case: no spread in, no spread out."""
    tight = {name: (mean, 1e-9, mean - 1e-6, mean + 1e-6)
             for name, (mean, _, _, _) in DEFAULT_DISPERSIONS.items()}
    result = run_dispersion(n_cases=4, dispersions=tight, seed=5, fixed=FAST, require_landing=False)
    assert result.landing_points.std(axis=0).max() < 1.0
    assert result.cep_m < 1.0


def test_samples_respect_their_truncation_bounds():
    result = _batch(n=10)
    for case in result.cases:
        for name, (_, _, low, high) in DEFAULT_DISPERSIONS.items():
            assert low <= case["params"][name] <= high, (name, case["params"][name])


def test_summary_reports_spread():
    result = _batch(n=6)
    assert set(result.summary) >= {"max_altitude", "max_velocity", "flight_time"}
    assert result.summary["max_altitude"]["std"] >= 0.0
    assert result.summary["max_altitude"]["min"] <= result.summary["max_altitude"]["max"]


def test_ellipse_axes_are_ordered_and_positive():
    result = _batch(n=8)
    _, semi_major, semi_minor, _ = result.ellipse
    assert semi_major >= semi_minor > 0.0


def test_report_is_printable():
    text = _batch(n=4).report()
    assert "CEP" in text and "ellipse" in text


def test_fixed_parameters_travel_to_a_worker_in_a_picklable_wrapper():
    """Constants ride in a class, not a closure, so they survive pickling."""
    import pickle

    from trajectory.analysis.dispersion import _WithFixed

    runner = pickle.loads(pickle.dumps(_WithFixed(run_case, FAST)))
    outcome = runner({"impulse_scale": 1.0})
    assert outcome["params"]["t_max"] == FAST["t_max"]


def test_a_perturbation_applies_to_the_simulation_it_is_given():
    """Dispersing a real vehicle rides on this: perturb, do not rebuild."""
    from trajectory import simulation as tm
    from trajectory.analysis.dispersion import perturb_simulation

    sim = tm.RocketSimulation()
    sim.mass_props.dry_mass = 7.0
    sim.mass_props.mass_0 = 7.0 + sim.mass_props.prop_mass
    same = perturb_simulation(sim, {"aero_scale": 1.1})
    assert same is sim
    assert sim.aero_scale == 1.1
    assert sim.mass_props.dry_mass == 7.0


def test_apogee_is_quoted_above_the_pad():
    from trajectory.analysis.dispersion import reduce_outcome

    sim = build_case_simulation(_case())
    result = sim.run(
        launch_elevation=np.radians(85.0),
        pad_position_m=np.array([0.0, 1000.0, 0.0]), t_max=90.0, dt=0.5,
    )
    raw = reduce_outcome(result, {})
    above = reduce_outcome(result, {}, ground_m=1000.0)
    assert np.isclose(raw["max_altitude"] - above["max_altitude"], 1000.0)


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
