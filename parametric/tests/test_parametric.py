"""Tests for the parametric geometry core.

The load-bearing checks are that the section model reproduces the masses the
previous pipeline produced, and that a lofted B-rep agrees with the analytic
section integral. Those are independent computations, so agreement is evidence
rather than tautology.

Runs under pytest, and standalone via
``python parametric/tests/test_parametric.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.components import FinSet, Motor, PointMass, Stack  # noqa: E402
from parametric.model import VehicleModel  # noqa: E402
from parametric.parm import Parm, ParmContainer  # noqa: E402
from parametric.standard import basic_rocket, boattailed_rocket  # noqa: E402
from parametric.xsec import (  # noqa: E402
    NoseProfile,
    XSec,
    XSecShape,
    generate_nose_sections,
    nose_radius,
)


# ------------------------------------------------------------------ parms


def test_parm_clamps_to_bounds():
    parm = Parm("span", 0.5, minimum=0.0, maximum=1.0)
    parm.set(5.0)
    assert parm.value == 1.0
    parm.set(-3.0)
    assert parm.value == 0.0


def test_parm_reports_whether_it_moved():
    parm = Parm("x", 1.0, 0.0, 10.0)
    assert parm.set(2.0)
    assert not parm.set(2.0)


def test_inverted_bounds_are_rejected():
    with pytest.raises(ValueError):
        Parm("bad", 1.0, minimum=5.0, maximum=1.0)


def test_container_marks_dirty_only_on_change():
    container = ParmContainer("thing")
    container.add_parm("a", 1.0, 0.0, 10.0)
    container.mark_clean()
    assert not container.set("a", 1.0)
    assert not container.dirty
    assert container.set("a", 2.0)
    assert container.dirty


def test_unknown_parm_names_the_alternatives():
    container = ParmContainer("thing")
    container.add_parm("span", 1.0)
    with pytest.raises(KeyError, match="span"):
        container.parm("spam")


def test_duplicate_parm_is_rejected():
    container = ParmContainer("thing")
    container.add_parm("a", 1.0)
    with pytest.raises(ValueError):
        container.add_parm("a", 2.0)


# ------------------------------------------------------------------ xsecs


def test_circle_area_matches_pi_r_squared():
    section = XSec(0.0, XSecShape.CIRCLE, width_m=0.2)
    assert np.isclose(section.area_m2, np.pi * 0.1 ** 2)


def test_ellipse_area():
    section = XSec(0.0, XSecShape.ELLIPSE, width_m=0.2, height_m=0.1)
    assert np.isclose(section.area_m2, np.pi * 0.1 * 0.05)


def test_super_ellipse_with_exponent_two_is_an_ellipse():
    ellipse = XSec(0.0, XSecShape.ELLIPSE, 0.2, 0.1)
    supers = XSec(0.0, XSecShape.SUPER_ELLIPSE, 0.2, 0.1, exponent=2.0)
    assert np.isclose(supers.area_m2, ellipse.area_m2, rtol=1e-9)


def test_high_exponent_super_ellipse_approaches_a_rectangle():
    supers = XSec(0.0, XSecShape.SUPER_ELLIPSE, 0.2, 0.1, exponent=20.0)
    assert np.isclose(supers.area_m2, 0.2 * 0.1, rtol=0.1)


def test_rounded_rectangle_area_between_rect_and_ellipse():
    rect = XSec(0.0, XSecShape.ROUNDED_RECTANGLE, 0.2, 0.1, corner_radius_m=0.02)
    assert rect.area_m2 < 0.2 * 0.1
    assert rect.area_m2 > np.pi * 0.1 * 0.05


def test_outline_is_closed_and_the_right_size():
    section = XSec(0.0, XSecShape.CIRCLE, width_m=0.2)
    outline = section.outline(64)
    assert outline.shape == (64, 2)
    radii = np.linalg.norm(outline, axis=1)
    assert np.allclose(radii, 0.1, atol=1e-9)


def test_outline_area_matches_analytic_area():
    """Shoelace over the traced outline against the closed-form area."""
    for shape, kwargs in [
        (XSecShape.CIRCLE, {"width_m": 0.2}),
        (XSecShape.ELLIPSE, {"width_m": 0.2, "height_m": 0.12}),
        (XSecShape.SUPER_ELLIPSE, {"width_m": 0.2, "height_m": 0.12, "exponent": 4.0}),
        (XSecShape.ROUNDED_RECTANGLE, {"width_m": 0.2, "height_m": 0.12,
                                       "corner_radius_m": 0.03}),
    ]:
        section = XSec(0.0, shape, **kwargs)
        pts = section.outline(720)
        y, z = pts[:, 0], pts[:, 1]
        shoelace = 0.5 * abs(np.dot(y, np.roll(z, -1)) - np.dot(z, np.roll(y, -1)))
        assert np.isclose(shoelace, section.area_m2, rtol=0.01), shape


def test_point_section_is_degenerate():
    section = XSec(0.0, XSecShape.POINT)
    assert section.is_point
    assert section.area_m2 == 0.0
    assert section.outline().shape == (1, 2)


# --------------------------------------------------------- nose generation


def test_nose_profiles_start_at_zero_and_end_at_the_base():
    for profile in NoseProfile:
        r = nose_radius(profile, 1.5, 0.15, np.array([0.0, 1.5]))
        assert np.isclose(r[0], 0.0, atol=1e-9), profile
        assert np.isclose(r[-1], 0.15, rtol=1e-6), profile


def test_generated_sections_span_the_nose():
    sections = generate_nose_sections(NoseProfile.OGIVE, 0.45, 0.10, sections=12)
    assert len(sections) == 12
    assert np.isclose(sections[0].station_m, 0.0)
    assert np.isclose(sections[-1].station_m, 0.45)
    assert np.isclose(sections[-1].width_m, 0.10, rtol=1e-6)


def test_sections_cluster_toward_the_tip():
    """Curvature is highest at the tip; uniform spacing under-resolves it."""
    sections = generate_nose_sections(NoseProfile.VON_KARMAN, 1.0, 0.3, sections=12)
    gaps = np.diff([s.station_m for s in sections])
    assert gaps[0] < gaps[-1]


def test_generated_tip_is_blunted():
    sections = generate_nose_sections(NoseProfile.CONICAL, 0.5, 0.1, tip_radius_m=0.002)
    assert sections[0].width_m > 0.0


def test_too_few_sections_is_rejected():
    with pytest.raises(ValueError):
        generate_nose_sections(NoseProfile.OGIVE, 0.5, 0.1, sections=2)


# ------------------------------------------------------------- components


def test_stack_keeps_sections_sorted():
    stack = Stack("body")
    stack.add_section(XSec(1.0, XSecShape.CIRCLE, 0.1))
    stack.add_section(XSec(0.0, XSecShape.CIRCLE, 0.1))
    assert [s.station_m for s in stack.sorted_sections()] == [0.0, 1.0]


def test_stack_volume_matches_a_cylinder():
    stack = Stack("tube", wall_thickness_m=0.0)
    stack.add_tube(2.0, 0.3)
    assert np.isclose(stack.volume_m3(), np.pi * 0.15 ** 2 * 2.0, rtol=1e-9)


def test_hollow_stack_volume_matches_the_annulus():
    stack = Stack("tube", wall_thickness_m=0.005)
    stack.add_tube(2.0, 0.3)
    expected = np.pi * (0.15 ** 2 - 0.145 ** 2) * 2.0
    assert np.isclose(stack.volume_m3(), expected, rtol=1e-9)


def test_transition_changes_diameter():
    stack = Stack("body", wall_thickness_m=0.0)
    stack.add_tube(1.0, 0.2)
    stack.add_transition(0.3, 0.1, name="boattail")
    assert np.isclose(stack.radius_at(1.3), 0.05, rtol=1e-6)
    assert np.isclose(stack.max_diameter_m, 0.2)


def test_fin_area_and_aspect_ratio():
    fins = FinSet(root_chord_m=0.5, tip_chord_m=0.3, span_m=0.2)
    assert np.isclose(fins.area_per_fin_m2, 0.5 * (0.5 + 0.3) * 0.2)
    assert np.isclose(fins.aspect_ratio, 0.2 ** 2 / fins.area_per_fin_m2)


def test_fins_read_the_parent_radius():
    stack = Stack("body")
    stack.add_tube(2.0, 0.3)
    fins = stack.add(FinSet("fins", station_m=1.0, root_chord_m=0.4))
    assert np.isclose(fins.body_radius_m(), 0.15, rtol=1e-6)


def test_point_mass_growth():
    mass = PointMass("payload", 10.0, 1.0, growth_allowance=0.2)
    assert np.isclose(mass.mass_with_growth_kg, 12.0)


def test_motor_is_not_counted_as_dry_mass():
    assert Motor("m", propellant_mass_kg=50.0).mass_kg() == 0.0


def test_component_paths_reflect_the_tree():
    model = basic_rocket()
    fins = model.find("fins")
    assert fins.path.endswith("motor_tube/fins")


# ------------------------------------------------------------------ model


def test_basic_rocket_matches_the_retired_definition():
    """The numbers the retired fixed-schema definition gave, frozen.

    vehicles/basic.json and the schema that read it are gone; these literals
    are what that pipeline computed at retirement, kept so the parametric
    model cannot silently drift from the vehicle both eras agreed on.
    """
    model = basic_rocket()
    summary = model.mass_summary()

    assert np.isclose(summary.dry_mass_kg, 4.39537860363771, rtol=0.005)
    assert np.isclose(model.total_length_m, 1.85, rtol=1e-9)
    assert np.isclose(model.max_diameter_m, 0.1, rtol=1e-9)
    assert np.isclose(summary.propellant_mass_kg, 3.0)


def test_per_component_masses_match_the_retired_definition():
    reference = {
        "nose": 0.487977,
        "forward_tube": 0.591122,
        "motor_tube": 1.974679,
        "fins": 0.2916,
    }
    masses = basic_rocket().mass_summary().per_component_kg
    for name, expected in reference.items():
        assert np.isclose(masses[name], expected, rtol=0.01), name


def test_cg_lies_inside_the_vehicle():
    model = basic_rocket()
    summary = model.mass_summary()
    assert 0.0 < summary.cg_station_m < model.total_length_m


def test_reference_vehicles_validate():
    assert basic_rocket().validate() == []
    assert boattailed_rocket().validate() == []


def test_solid_nose_tip_is_not_an_error():
    """The wall closes off the tip by design; only a fully solid body is wrong."""
    model = basic_rocket()
    nose = model.find("nose")
    # The tip section is narrow enough that the wall leaves no cavity there.
    assert min(s.width_m for s in nose.sections) <= 2 * nose.get("wall_thickness")
    assert model.validate() == []


def test_wall_thicker_than_the_body_is_rejected():
    model = basic_rocket()
    model.find("motor_tube").set("wall_thickness", 0.5)
    assert any("entirely solid" in p for p in model.validate())


def test_fins_off_the_end_of_the_parent_are_rejected():
    model = basic_rocket()
    model.find("fins").set("station", 1.80)
    assert any("off the end of its parent" in p for p in model.validate())


def test_propellant_that_does_not_fit_is_rejected():
    model = basic_rocket()
    model.find("motor").set("propellant_mass", 500.0)
    assert any("holds" in p for p in model.validate())


def test_boattail_narrows_toward_the_tail():
    model = boattailed_rocket()
    stack = model.find("airframe")
    assert stack.radius_at(model.total_length_m) < stack.radius_at(1.5)


def test_payload_bulge_is_wider_than_the_body():
    """A shape the previous schema could not express at all."""
    stack = boattailed_rocket().find("airframe")
    assert stack.radius_at(1.25) > stack.radius_at(0.8)


def test_editing_a_parm_marks_the_model_dirty():
    model = basic_rocket()
    model.mark_clean()
    assert not model.dirty
    model.find("fins").set("span", 0.12)
    assert model.dirty


def test_changing_span_changes_mass():
    model = basic_rocket()
    before = model.mass_summary().dry_mass_kg
    model.find("fins").set("span", model.find("fins").get("span") * 2)
    assert model.mass_summary().dry_mass_kg > before


# ---------------------------------------------------------- serialisation


def test_round_trip_preserves_the_model():
    original = basic_rocket()
    with tempfile.TemporaryDirectory() as directory:
        path = original.save(Path(directory) / "v.json")
        loaded = VehicleModel.load(path)

    assert loaded.name == original.name
    assert np.isclose(loaded.total_length_m, original.total_length_m)
    assert np.isclose(
        loaded.mass_summary().dry_mass_kg, original.mass_summary().dry_mass_kg, rtol=1e-9
    )
    assert len(list(loaded.walk())) == len(list(original.walk()))
    assert len(loaded.find("nose").sections) == len(original.find("nose").sections)


def test_foreign_format_is_rejected():
    with pytest.raises(ValueError):
        VehicleModel.from_dict({"format": "something.else", "name": "x", "tree": {}})


# ------------------------------------------------------------------ loft

cadquery = pytest.importorskip("cadquery", reason="lofting needs the cad extra")

from parametric.loft import (  # noqa: E402
    LoftCache,
    is_axisymmetric,
    verify_volumes,
)


def test_circular_stacks_take_the_revolve_path():
    assert is_axisymmetric(basic_rocket().find("nose"))


def test_a_non_circular_section_falls_back_to_lofting():
    stack = Stack("body", wall_thickness_m=0.0)
    stack.add_section(XSec(0.0, XSecShape.CIRCLE, 0.2))
    stack.add_section(XSec(1.0, XSecShape.ROUNDED_RECTANGLE, 0.2, 0.15,
                           corner_radius_m=0.03))
    assert not is_axisymmetric(stack)


@pytest.mark.slow
def test_lofted_volumes_match_the_section_integral():
    for build in (basic_rocket, boattailed_rocket):
        model = build()
        results = LoftCache().solids(model)
        assert results, build.__name__
        assert verify_volumes(results) == [], build.__name__


@pytest.mark.slow
def test_cache_rebuilds_only_what_changed():
    model = basic_rocket()
    cache = LoftCache()
    cache.solids(model)

    model.find("fins").set("span", model.find("fins").get("span") * 1.2)
    cache.solids(model)
    rebuilt = cache.last_rebuilt
    assert rebuilt, "changing a fin must rebuild something"
    assert all("fins" in key for key in rebuilt), rebuilt

    cache.solids(model)
    assert cache.last_rebuilt == [], "an unchanged model must rebuild nothing"


@pytest.mark.slow
def test_hollow_stack_is_lighter_than_solid():
    solid = Stack("t", wall_thickness_m=0.0)
    solid.add_tube(0.5, 0.1)
    hollow = Stack("t", wall_thickness_m=0.004)
    hollow.add_tube(0.5, 0.1)

    cache = LoftCache()
    model_a, model_b = VehicleModel("a"), VehicleModel("b")
    model_a.add(solid)
    model_b.add(hollow)
    volume_solid = list(cache.solids(model_a).values())[0].volume_m3
    volume_hollow = list(LoftCache().solids(model_b).values())[0].volume_m3
    assert volume_hollow < volume_solid


@pytest.mark.slow
def test_step_export_writes_one_file_per_part():
    from parametric.loft import export_step

    model = basic_rocket()
    with tempfile.TemporaryDirectory() as directory:
        written = export_step(model, directory)
        assert len(written) == 6      # 3 stacks + 3 fins
        for path in written:
            assert path.exists() and path.stat().st_size > 0


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


def test_a_swept_fin_tip_may_trail_past_the_tail():
    """The root attaches; the tip is cantilevered into free air.

    Real vehicles are built this way, and the oracle's ``finsweep_12p0`` is
    exactly it: a fin flush with the tail whose 12-inch sweep carries the tip
    eight inches beyond. Checking the whole station range refused those.
    """
    model = basic_rocket()
    fins = model.find("fins")
    parent_aft = fins.parent.station_range_m()[1]
    # Root flush with the tail, tip swept well past it.
    fins.set("station", parent_aft - fins.get("root_chord"))
    fins.set("sweep", 0.40)
    assert fins.station_range_m()[1] > parent_aft      # tip really does overhang
    assert model.validate() == []
