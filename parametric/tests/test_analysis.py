"""Tests for the analysis bridge.

Checks that a parametric model reproduces what the previous pipeline produced,
since the geometry source changed but the physics did not. Where the two
disagree it should be because the parametric path is better informed -- meshed
inertia rather than a slender-rod estimate -- not because something broke.

Runs under pytest, and standalone via
``python parametric/tests/test_analysis.py``.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.standard import basic_rocket, boattailed_rocket  # noqa: E402

pytest.importorskip("cadquery", reason="analysis needs the cad extra")

from parametric import analysis  # noqa: E402


# ------------------------------------------------------------------ aero


def test_canonical_lumping_finds_the_body_and_fins():
    model = basic_rocket()
    canonical = analysis.to_canonical(model)
    kinds = [segment.kind for segment in canonical.segments]
    assert "nose" in kinds
    assert "tube" in kinds
    assert len(canonical.fins) == 1
    assert canonical.fins[0].count == 3


def test_canonical_residual_is_small_for_a_clean_body():
    canonical = analysis.to_canonical(basic_rocket())
    assert canonical.residual_rms_m < 0.005


def test_canonical_carries_mass_and_cg():
    """The dry CG in, the loaded one out: RASAero gets a launch weight."""
    model = basic_rocket()
    canonical = analysis.to_canonical(model, cg_station_m=1.25)
    assert np.isclose(canonical.cg_from_nose_m, analysis.loaded_cg_station_m(model, 1.25))
    assert canonical.cg_from_nose_m > 1.25, "propellant sits aft of this dry CG"
    assert canonical.wet_mass_kg > 0


def test_boattail_survives_lumping():
    canonical = analysis.to_canonical(boattailed_rocket())
    assert any(s.kind == "boattail" for s in canonical.segments)


def test_cdx1_is_written():
    model = basic_rocket()
    with tempfile.TemporaryDirectory() as directory:
        path, canonical = analysis.write_cdx1(model, Path(directory) / "v.cdx1")
        assert path.exists()
        text = path.read_text(encoding="utf-8")
    assert "<NoseCone>" in text and "<Fin>" in text


def test_the_nozzle_reaches_the_canonical_model():
    """No nozzle ever reached the engine, so power-on drag equalled power-off."""
    model = basic_rocket()
    canonical = analysis.to_canonical(model)
    assert 0.0 < canonical.nozzle_exit_diameter_m <= model.max_diameter_m


def test_the_nozzle_reaches_the_project_file():
    with tempfile.TemporaryDirectory() as directory:
        path, _ = analysis.write_cdx1(basic_rocket(), Path(directory) / "n.cdx1")
        text = path.read_text(encoding="utf-8")
    match = re.search(r"<SustainerNozzle>([^<]+)</SustainerNozzle>", text)
    assert match is not None and float(match.group(1)) > 0.0


def test_a_burning_motor_fills_the_base_and_cuts_the_drag():
    """Both columns are tabulated; the burning one is the lower."""
    from parametric import aero

    model = basic_rocket()
    grid = dict(mach_min=0.3, mach_max=0.9, mach_points=4,
                alpha_max_deg=4.0, alpha_points=2)
    both, _ = aero.run_analysis(model, aero.AeroSettings(power_on_base=True, **grid))
    row = both.lookup(0.5, 0.0)
    assert both.has_power_on
    assert row.cd_power_on < row.cd

    single, _ = aero.run_analysis(model, aero.AeroSettings(power_on_base=False, **grid))
    assert not single.has_power_on
    assert np.isclose(single.lookup(0.5, 0.0).cd, row.cd)


# --------------------------------------------------------- static margin


def test_static_margin_is_positive_for_a_stable_rocket():
    assert analysis.static_margin(basic_rocket()) > 0.0


def test_margin_is_reported_for_the_loaded_vehicle_by_default():
    """The rail is where a rocket goes unstable, and there the tanks are full.

    This used to default to the dry CG, which flattered every design: on the
    standard rocket it reported 1.66 calibres for a vehicle that actually
    leaves the rail with 0.38.
    """
    model = basic_rocket()
    loaded = analysis.static_margin(model)
    burnout = analysis.static_margin(model, loaded=False)
    assert loaded < burnout
    assert analysis.static_margin(model) == loaded


def test_propellant_moves_the_cg_aft():
    summary = basic_rocket().mass_summary()
    assert summary.propellant_mass_kg > 0
    assert summary.propellant_cg_station_m > summary.cg_station_m
    assert summary.cg_station_m < summary.wet_cg_station_m < summary.propellant_cg_station_m


def test_wet_cg_is_the_mass_weighted_balance():
    summary = basic_rocket().mass_summary()
    expected = (
        summary.cg_station_m * summary.dry_mass_kg
        + summary.propellant_cg_station_m * summary.propellant_mass_kg
    ) / summary.wet_mass_kg
    assert abs(summary.wet_cg_station_m - expected) < 1e-12


def test_a_dry_motor_leaves_the_cg_alone():
    """With no propellant, loaded and burnout are the same vehicle."""
    model = basic_rocket()
    model.motors[0].set("propellant_mass", 0.0)
    summary = model.mass_summary()
    assert summary.wet_cg_station_m == summary.cg_station_m
    assert analysis.static_margin(model) == analysis.static_margin(model, loaded=False)


def test_a_solved_dry_cg_is_still_corrected_for_propellant():
    """The CAD solver reports structure, so its CG is the burnout one."""
    model = basic_rocket()
    summary = model.mass_summary()
    corrected = analysis.loaded_cg_station_m(model, summary.cg_station_m)
    assert abs(corrected - summary.wet_cg_station_m) < 1e-12


def test_bigger_fins_increase_the_margin():
    model = basic_rocket()
    before = analysis.static_margin(model)
    model.find("fins").set("span", model.find("fins").get("span") * 2.0)
    assert analysis.static_margin(model) > before


def test_moving_fins_aft_increases_the_margin():
    model = basic_rocket()
    fins = model.find("fins")
    before = analysis.static_margin(model)
    fins.set("station", fins.get("station") + 0.15)
    assert analysis.static_margin(model) > before


def test_nose_ballast_increases_the_margin():
    """Moving the CG forward is the other lever, and it must act the right way."""
    from parametric.components import PointMass

    model = basic_rocket()
    before = analysis.static_margin(model)
    model.add(PointMass("ballast", 2.0, 0.15))
    assert analysis.static_margin(model) > before


# ----------------------------------------------------------- simulation


def test_simulation_takes_its_mass_from_the_model():
    model = basic_rocket()
    sim = analysis.build_simulation(model)
    summary = model.mass_summary()
    assert np.isclose(sim.mass_props.dry_mass, summary.dry_mass_kg, rtol=1e-9)
    assert np.isclose(sim.mass_props.prop_mass, summary.propellant_mass_kg, rtol=1e-9)


def test_simulation_reference_area_matches_the_body():
    model = basic_rocket()
    sim = analysis.build_simulation(model)
    assert np.isclose(sim.reference_area, model.reference_area_m2)


def test_cg_is_mapped_into_the_body_frame():
    """Definition +Z aft becomes simulator -Y; a station must land at negative y."""
    model = basic_rocket()
    sim = analysis.build_simulation(model)
    assert sim.mass_props.cg_dry[1] < 0.0
    assert np.isclose(
        abs(sim.mass_props.cg_dry[1]), model.mass_summary().cg_station_m, rtol=1e-9
    )


def test_roll_inertia_lands_on_the_thrust_axis():
    """The rotation, not a permutation: roll must end up on body Y."""
    model = basic_rocket()
    sim = analysis.build_simulation(model)
    ixx, iyy, izz = np.diag(sim.mass_props.i_tensor_dry)
    assert iyy < 0.2 * ixx, (ixx, iyy, izz)
    assert np.isclose(ixx, izz, rtol=0.05)


def test_thrust_acts_at_the_tail():
    model = basic_rocket()
    sim = analysis.build_simulation(model)
    assert np.isclose(abs(sim.thrust_position_body_m[1]), model.total_length_m)


@pytest.mark.slow
def test_flight_runs_end_to_end():
    from trajectory.analysis.statistics import flight_statistics
    from trajectory.vehicle.recovery import standard_recovery

    model = basic_rocket()
    sim = analysis.build_simulation(model)
    recovery = standard_recovery(
        dry_mass_kg=model.mass_summary().dry_mass_kg,
        main_descent_mps=5.0, drogue_descent_mps=18.0, main_deploy_altitude_m=150.0,
    )
    result = sim.run(launch_elevation=np.radians(85.0), rail_length_m=3.0,
                     t_max=900.0, dt=0.1, recovery=recovery)

    assert result.success
    stats = flight_statistics(result.y.T, result.t)
    assert 2000.0 < stats["max_altitude"] < 10000.0
    assert [p["name"] for p in result.phases] == ["ascent", "drogue", "main"]


# --------------------------------------------------------- mass solving


@pytest.mark.slow
def test_meshed_mass_matches_the_analytic_roll_up():
    """Two independent computations: meshed B-reps against section integrals."""
    model = basic_rocket()
    with tempfile.TemporaryDirectory() as directory:
        solved = analysis.solve_mass(model, directory)

    analytic = model.mass_summary()
    assert np.isclose(solved.mass_kg, analytic.dry_mass_kg, rtol=0.02)
    assert abs(solved.cg_station_m - analytic.cg_station_m) < 0.02


@pytest.mark.slow
def test_meshed_inertia_is_physically_ordered():
    model = basic_rocket()
    with tempfile.TemporaryDirectory() as directory:
        solved = analysis.solve_mass(model, directory)

    ixx, iyy, izz = np.diag(solved.inertia_kg_m2)
    assert izz < 0.1 * ixx, "roll inertia must be far below pitch on a slender body"
    assert np.isclose(ixx, iyy, rtol=0.1), "axisymmetric body should have Ixx ~ Iyy"


@pytest.mark.slow
def test_point_masses_reach_the_solved_result():
    model = basic_rocket()
    with tempfile.TemporaryDirectory() as directory:
        solved = analysis.solve_mass(model, directory)
    assert "avionics" in solved.per_component_kg
    assert np.isclose(solved.per_component_kg["avionics"], 0.40)


@pytest.mark.slow
def test_meshed_mass_includes_the_tanks_and_the_engine():
    """A biprop solved without its tanks flew tens of kilograms light."""
    from parametric.standard import biprop_testbed

    model = biprop_testbed()
    with tempfile.TemporaryDirectory() as directory:
        solved = analysis.solve_mass(model, directory)

    for tank in model.tanks:
        assert tank.path in solved.per_component_kg
        assert np.isclose(
            solved.per_component_kg[tank.path], tank.mass_kg(), rtol=0.05
        )
    motor = model.motors[0]
    assert solved.per_component_kg[motor.name] == motor.mass_kg()
    assert np.isclose(solved.mass_kg, model.mass_summary().dry_mass_kg, rtol=0.03)


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
