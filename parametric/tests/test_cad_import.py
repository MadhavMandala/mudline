"""Tests for STEP import.

Round-trip against known geometry: build a vehicle parametrically, export it,
import it back with no knowledge of the original, and compare. That is a real
test because the two paths share nothing -- one revolves a profile through OCC,
the other measures slices of the resulting solid.

Runs under pytest, and standalone via
``python parametric/tests/test_cad_import.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.components import Stack  # noqa: E402
from parametric.xsec import XSec, XSecShape  # noqa: E402

pytest.importorskip("cadquery", reason="STEP import needs the cad extra")

from parametric.cad_import import (  # noqa: E402
    build_stack_from_profile,
    detect_axis,
    fit_sections,
    import_step,
    sample_profile,
)
from parametric.loft import LoftCache  # noqa: E402
from parametric.standard import basic_rocket, boattailed_rocket  # noqa: E402


# ---------------------------------------------------------------- fitting


def test_a_straight_profile_needs_two_sections():
    stations = np.linspace(0.0, 1.0, 50)
    radii = np.full_like(stations, 0.05)
    assert fit_sections(stations, radii) == [0, 49]


def test_a_linear_taper_needs_two_sections():
    stations = np.linspace(0.0, 1.0, 50)
    radii = np.linspace(0.05, 0.02, 50)
    assert len(fit_sections(stations, radii)) == 2


def test_a_curved_profile_needs_more_sections():
    stations = np.linspace(0.0, 1.0, 80)
    radii = 0.08 * np.sqrt(stations)          # strongly curved near zero
    assert len(fit_sections(stations, radii, tolerance_m=0.0005)) > 4


def test_tighter_tolerance_uses_more_sections():
    stations = np.linspace(0.0, 1.0, 120)
    radii = 0.08 * np.sqrt(stations)
    loose = fit_sections(stations, radii, tolerance_m=0.005)
    tight = fit_sections(stations, radii, tolerance_m=0.0002)
    assert len(tight) > len(loose)


def test_fit_respects_the_section_cap():
    stations = np.linspace(0.0, 1.0, 400)
    radii = 0.05 + 0.02 * np.sin(stations * 60.0)      # deliberately wiggly
    assert len(fit_sections(stations, radii, tolerance_m=1e-9, max_sections=12)) <= 12


def test_fit_includes_both_ends():
    stations = np.linspace(0.0, 2.0, 60)
    radii = np.linspace(0.02, 0.06, 60)
    indices = fit_sections(stations, radii)
    assert indices[0] == 0
    assert indices[-1] == 59


def test_built_stack_reproduces_the_profile():
    stations = np.linspace(0.0, 1.0, 60)
    radii = 0.05 + 0.01 * stations
    stack, fitted = build_stack_from_profile(stations, radii, tolerance_m=1e-4)
    assert np.allclose(fitted, radii, atol=2e-4)
    assert isinstance(stack, Stack)
    assert all(s.shape is XSecShape.CIRCLE for s in stack.sections)


# ---------------------------------------------------------------- sampling


@pytest.fixture(scope="module")
def exported_airframe():
    """Export the boattail airframe once for the whole module."""
    model = boattailed_rocket()
    solids = LoftCache().solids(model)
    key = next(k for k in solids if k.endswith("airframe"))
    directory = Path(tempfile.mkdtemp())
    path = directory / "airframe.stp"
    solids[key].solid.exportStep(str(path))
    return model, path


@pytest.mark.slow
def test_axis_is_detected(exported_airframe):
    from cadquery import importers

    _, path = exported_airframe
    solid = importers.importStep(str(path)).val()
    assert detect_axis(solid) == 2          # built along +Z


@pytest.mark.slow
def test_sampling_returns_outer_and_material_radii(exported_airframe):
    """They must differ on a shell -- that difference is how hollow is detected."""
    from cadquery import importers

    _, path = exported_airframe
    solid = importers.importStep(str(path)).val()
    stations, outer, material = sample_profile(solid, 2, samples=30)

    assert len(stations) == len(outer) == len(material) == 30
    assert np.all(outer > 0)
    assert np.median(material / outer) < 0.9, "a hollow tube should read thin"


# ------------------------------------------------------------- round trip


@pytest.mark.slow
def test_round_trip_recovers_the_geometry(exported_airframe):
    model, path = exported_airframe
    truth = model.find("airframe")
    imported, report = import_step(path, samples=70)
    stack = imported.stacks[0]

    assert np.isclose(stack.length_m, truth.length_m, rtol=0.01)
    assert np.isclose(stack.max_diameter_m, truth.max_diameter_m, rtol=0.02)
    assert report.residual_max_m < 0.003


@pytest.mark.slow
def test_round_trip_detects_the_wall(exported_airframe):
    """The shell thickness is recovered from the material area, not assumed."""
    model, path = exported_airframe
    truth = model.find("airframe")
    imported, report = import_step(path, samples=70)

    recovered = imported.stacks[0].get("wall_thickness")
    assert np.isclose(recovered, truth.get("wall_thickness"), rtol=0.1)
    assert any("Hollow shell" in note for note in report.notes)


@pytest.mark.slow
def test_round_trip_mass_matches(exported_airframe):
    model, path = exported_airframe
    truth = model.find("airframe")
    imported, _ = import_step(path, samples=70)
    imported.stacks[0].material = truth.material
    assert np.isclose(imported.stacks[0].mass_kg(), truth.mass_kg(), rtol=0.05)


@pytest.mark.slow
def test_import_compresses_the_section_count(exported_airframe):
    """34 authored sections should not come back as 34; the fit is the point."""
    model, path = exported_airframe
    imported, _ = import_step(path, samples=70)
    assert len(imported.stacks[0].sections) < len(model.find("airframe").sections)


@pytest.mark.slow
def test_imported_model_is_editable(exported_airframe):
    """The whole reason for fitting rather than meshing."""
    _, path = exported_airframe
    imported, _ = import_step(path, samples=60)
    stack = imported.stacks[0]

    before = stack.mass_kg()
    stack.set("wall_thickness", stack.get("wall_thickness") * 2.0)
    assert stack.mass_kg() > before

    section = stack.sorted_sections()[len(stack.sections) // 2]
    before_diameter = stack.max_diameter_m
    section.set("width", section.width_m * 1.5)
    assert stack.max_diameter_m >= before_diameter


@pytest.mark.slow
def test_imported_model_can_be_rebuilt_as_geometry(exported_airframe):
    """An imported stack is an ordinary component: it lofts like any other."""
    _, path = exported_airframe
    imported, _ = import_step(path, samples=60)
    solids = LoftCache().solids(imported)
    assert len(solids) == 1
    assert list(solids.values())[0].volume_m3 > 0


@pytest.mark.slow
def test_report_is_readable(exported_airframe):
    _, path = exported_airframe
    _, report = import_step(path, samples=50)
    text = report.text()
    for expected in ("axis", "length", "cross-sections", "residual"):
        assert expected in text


@pytest.mark.slow
def test_solid_body_is_not_reported_hollow():
    """The control case: a solid billet must not trigger wall estimation."""
    model = basic_rocket()
    stack = Stack("solid_tube", wall_thickness_m=0.0)
    stack.add_tube(0.5, 0.08)
    from parametric.model import VehicleModel

    holder = VehicleModel("solid")
    holder.add(stack)
    solids = LoftCache().solids(holder)

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "solid.stp"
        list(solids.values())[0].solid.exportStep(str(path))
        imported, report = import_step(path, samples=40)

    assert not any("Hollow" in note for note in report.notes)
    assert imported.stacks[0].get("wall_thickness") == 0.0


if __name__ == "__main__":
    failures = 0
    model = boattailed_rocket()
    solids = LoftCache().solids(model)
    key = next(k for k in solids if k.endswith("airframe"))
    directory = Path(tempfile.mkdtemp())
    export_path = directory / "airframe.stp"
    solids[key].solid.exportStep(str(export_path))
    fixture = (model, export_path)

    names = sorted(n for n in globals() if n.startswith("test_"))
    for name in names:
        fn = globals()[name]
        try:
            fn(fixture) if fn.__code__.co_argcount else fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(names) - failures}/{len(names)} passed")
    raise SystemExit(1 if failures else 0)
