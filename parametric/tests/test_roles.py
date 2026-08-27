"""Tests for aerodynamic roles, Hoerner protuberance drag, and pipeline state.

Three things that share a purpose: making what the analysis is doing explicit
rather than inferred, so it can neither guess wrong nor quietly use a result
that no longer describes the vehicle.

Runs under pytest, and standalone via
``python parametric/tests/test_roles.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.components import Protuberance, Stack  # noqa: E402
from parametric.model import VehicleModel  # noqa: E402
from parametric.roles import (  # noqa: E402
    HOERNER_CD,
    AeroRole,
    HoernerShape,
    boundary_layer_thickness_m,
    compressibility_factor,
    immersion_factor,
    interference_drag_coefficient,
    protuberance_drag_coefficient,
)
from parametric.standard import basic_rocket, boattailed_rocket  # noqa: E402


# ------------------------------------------------------------------- roles


def test_components_default_to_auto():
    assert basic_rocket().find("nose").aero_role is AeroRole.AUTO


def test_internal_is_the_only_non_external_role():
    for role in AeroRole:
        assert role.is_external == (role is not AeroRole.INTERNAL)


def test_allowed_roles_are_narrowed_per_kind():
    model = basic_rocket()
    assert "fin" not in model.find("nose").allowed_roles
    assert "fin" in model.find("fins").allowed_roles
    assert model.motors[0].allowed_roles == ("internal",)


def test_roles_round_trip_through_a_file(tmp_path):
    model = basic_rocket()
    model.find("nose").aero_role = AeroRole.NOSE
    model.find("motor_tube").aero_role = AeroRole.INTERNAL
    reloaded = VehicleModel.load(model.save(tmp_path / "v.json"))
    assert reloaded.find("nose").aero_role is AeroRole.NOSE
    assert reloaded.find("motor_tube").aero_role is AeroRole.INTERNAL


# --------------------------------------------------------------- Hoerner


def test_every_shape_has_a_coefficient():
    for shape in HoernerShape:
        assert HOERNER_CD[shape] > 0


def test_streamlined_is_far_cheaper_than_a_flat_plate():
    assert HOERNER_CD[HoernerShape.STREAMLINED_FAIRING] < 0.2
    assert HOERNER_CD[HoernerShape.FLAT_PLATE_NORMAL] > 1.2


def test_protuberance_drag_scales_with_area_and_count():
    one = Protuberance("a", "rail_button", frontal_area_m2=1e-4, count=1).to_spec()
    two = Protuberance("b", "rail_button", frontal_area_m2=1e-4, count=2).to_spec()
    big = Protuberance("c", "rail_button", frontal_area_m2=2e-4, count=1).to_spec()
    assert np.isclose(two.drag_area_m2(), 2 * one.drag_area_m2())
    assert np.isclose(big.drag_area_m2(), 2 * one.drag_area_m2())


def test_compressibility_rises_through_the_transonic():
    assert compressibility_factor(0.3) == 1.0
    assert compressibility_factor(0.9) > 1.0
    assert compressibility_factor(1.2) > compressibility_factor(0.9)
    assert compressibility_factor(3.0) < compressibility_factor(1.2)


def test_boundary_layer_grows_along_the_body():
    thin = boundary_layer_thickness_m(0.2, 5e6)
    thick = boundary_layer_thickness_m(2.0, 5e6)
    assert 0.0 < thin < thick


def test_immersion_reduces_a_buried_protuberance():
    """Something inside the boundary layer sees less than free stream."""
    buried = immersion_factor(0.001, 0.010)
    exposed = immersion_factor(0.050, 0.010)
    assert buried < exposed <= 1.0
    assert immersion_factor(0.0, 0.01) == 0.0


def test_no_protuberances_means_no_drag():
    assert protuberance_drag_coefficient([], 0.008, 0.5) == 0.0


def test_protuberance_drag_is_referenced_to_the_vehicle_area():
    spec = Protuberance("p", "flat_plate_normal", frontal_area_m2=8e-4).to_spec()
    cd = protuberance_drag_coefficient([spec], 0.008, 0.2)
    expected = HOERNER_CD[HoernerShape.FLAT_PLATE_NORMAL] * 8e-4 / 0.008
    assert np.isclose(cd, expected, rtol=1e-6)


def test_interference_grows_with_fin_thickness():
    model = basic_rocket()
    fins = model.find("fins")
    thin = interference_drag_coefficient([fins], 0.008)
    fins.set("thickness", fins.get("thickness") * 2)
    assert interference_drag_coefficient([fins], 0.008) > thin


# ------------------------------------------------------- effect on the sweep


def _cd(model, mach=0.3):
    from parametric import aero

    database, _ = aero.run_analysis(model)
    return database.lookup(mach, 0.0).cd


@pytest.mark.slow
def test_protuberances_increase_drag():
    """They contributed exactly zero before, which is a missing force."""
    model = basic_rocket()
    before = _cd(model)
    model.add(Protuberance("buttons", "rail_button", frontal_area_m2=1.6e-4,
                           station_m=1.2, count=2))
    model.add(Protuberance("fairing", "streamlined_fairing",
                           frontal_area_m2=1.8e-3, station_m=0.9))
    assert _cd(model) > before


@pytest.mark.slow
def test_a_declared_nose_overrides_the_silhouette_guess():
    from parametric import aero

    model = basic_rocket()
    inferred = aero.extract_geometry(model)
    assert not inferred.nose_declared

    model.find("nose").aero_role = AeroRole.NOSE
    declared = aero.extract_geometry(model)
    assert declared.nose_declared
    assert np.isclose(declared.nose_length_m, model.find("nose").length_m)


@pytest.mark.slow
def test_internal_components_leave_the_aerodynamics():
    from parametric import aero

    model = basic_rocket()
    with_fins = aero.extract_geometry(model)
    assert with_fins.fin_sets

    model.find("fins").aero_role = AeroRole.INTERNAL
    without = aero.extract_geometry(model)
    assert without.fin_sets == []
    assert "fins" in without.excluded


@pytest.mark.slow
def test_internal_components_keep_their_mass():
    """The point of the role: invisible to air, still on the scale."""
    model = basic_rocket()
    before = model.mass_summary().dry_mass_kg
    model.find("fins").aero_role = AeroRole.INTERNAL
    assert np.isclose(model.mass_summary().dry_mass_kg, before)


# ------------------------------------------------------------- pipeline


def test_pipeline_starts_not_run():
    from app.pipeline import Pipeline, StageState

    model = basic_rocket()
    pipeline = Pipeline()
    assert pipeline.state("aero", model) is StageState.NOT_RUN


def test_recording_makes_a_stage_current():
    from app.pipeline import Pipeline, StageState

    model = basic_rocket()
    pipeline = Pipeline()
    pipeline.record("aero", model)
    assert pipeline.state("aero", model) is StageState.CURRENT


def test_editing_geometry_makes_it_stale():
    from app.pipeline import Pipeline, StageState

    model = basic_rocket()
    pipeline = Pipeline()
    pipeline.record("aero", model)
    model.find("fins").set("span", 0.14)
    assert pipeline.state("aero", model) is StageState.STALE


def test_a_role_change_makes_it_stale_although_no_vertex_moves():
    from app.pipeline import Pipeline, StageState

    model = basic_rocket()
    pipeline = Pipeline()
    pipeline.record("aero", model)
    model.find("forward_tube").aero_role = AeroRole.INTERNAL
    assert pipeline.state("aero", model) is StageState.STALE


def test_a_material_change_makes_it_stale():
    from app.pipeline import Pipeline, StageState

    model = basic_rocket()
    pipeline = Pipeline()
    pipeline.record("mass", model)
    model.find("nose").material = "aluminium_6061_t6"
    assert pipeline.state("mass", model) is StageState.STALE


def test_a_thrust_curve_change_makes_it_stale():
    from app.pipeline import Pipeline, StageState

    model = basic_rocket()
    pipeline = Pipeline()
    pipeline.record("flight", model)
    model.motors[0].add_curve_point(5.0, 100.0)
    assert pipeline.state("flight", model) is StageState.STALE


def test_flight_goes_stale_when_aero_does():
    """Downstream stages inherit their upstream's staleness."""
    from app.pipeline import Pipeline, StageState

    model = basic_rocket()
    pipeline = Pipeline()
    pipeline.record("aero", model)
    pipeline.record("flight", model)
    assert pipeline.state("flight", model) is StageState.CURRENT

    model.find("fins").set("span", 0.15)
    pipeline.record("flight", model)          # flight re-run, aero not
    assert pipeline.state("aero", model) is StageState.STALE
    assert pipeline.state("flight", model) is StageState.STALE


def test_stale_warning_is_worded_for_each_case():
    from app.pipeline import Pipeline

    model = basic_rocket()
    pipeline = Pipeline()
    assert "has not been run" in pipeline.stale_warning("aero", model)
    pipeline.record("aero", model)
    assert pipeline.stale_warning("aero", model) is None
    model.find("fins").set("span", 0.13)
    assert "stale" in pipeline.stale_warning("aero", model)


def test_fingerprint_is_stable_for_an_unchanged_model():
    from app.pipeline import model_fingerprint

    model = basic_rocket()
    assert model_fingerprint(model) == model_fingerprint(model)


def test_different_vehicles_fingerprint_differently():
    from app.pipeline import model_fingerprint

    assert model_fingerprint(basic_rocket()) != model_fingerprint(boattailed_rocket())


if __name__ == "__main__":
    import tempfile

    failures = 0
    names = sorted(n for n in globals() if n.startswith("test_"))
    for name in names:
        fn = globals()[name]
        try:
            if fn.__code__.co_argcount:
                with tempfile.TemporaryDirectory() as d:
                    fn(Path(d))
            else:
                fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(names) - failures}/{len(names)} passed")
    raise SystemExit(1 if failures else 0)
