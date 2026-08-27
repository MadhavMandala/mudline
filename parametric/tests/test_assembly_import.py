"""Tests for reading a STEP assembly and assigning its solids.

The fixture is a launch vehicle written as a named assembly -- nose, payload
bay, three tanks, two intertanks, an aft skirt and four fins -- so the numbers
being checked are the ones that went in. That matters most for the fins: the
profile fitter reads them as a body collar of twice the true diameter while
reporting a 0.04 mm residual, and the whole point of assigning solids by hand
is that it cannot.

Runs under pytest, and standalone via
``python parametric/tests/test_assembly_import.py``.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("cadquery", reason="STEP import needs the cad extra")

import cadquery as cq  # noqa: E402

from parametric.assembly_import import (  # noqa: E402
    BLADE_SOLIDITY,
    Assignment,
    PartType,
    build_model,
    measure_finset,
    suggest,
)
from parametric.model import VehicleModel  # noqa: E402
from parametric.step_assembly import read_assembly  # noqa: E402
from parametric.sweep import sweepable_parms  # noqa: E402

# Millimetres, as the fixture is authored.
R_BODY = 50.0
FIN_ROOT, FIN_TIP, FIN_SPAN, FIN_THICK, FIN_SWEEP = 200.0, 100.0, 60.0, 3.0, 80.0
N_FINS = 4
SECTIONS = [
    ("aft_skirt", 150.0),
    ("tank_lox", 400.0),
    ("intertank_1", 120.0),
    ("tank_rp1", 350.0),
    ("intertank_2", 100.0),
    ("tank_press", 200.0),
    ("payload_bay", 180.0),
]
L_NOSE = 250.0


def _write_assembly(directory: Path) -> Path:
    """A named multi-part vehicle, fins deliberately off the radial plane.

    The fins are extruded to one side rather than straddling their plane,
    which is how they usually arrive from CAD and which breaks any attempt to
    take a blade's azimuth from its centroid.
    """
    assembly = cq.Assembly(name="LV-1")
    station = 0.0
    for name, length in SECTIONS:
        assembly.add(
            cq.Workplane("XY").circle(R_BODY).extrude(length),
            name=name,
            loc=cq.Location(cq.Vector(0, 0, station)),
        )
        station += length
    assembly.add(
        cq.Workplane(obj=cq.Solid.makeCone(R_BODY, 0.0, L_NOSE)),
        name="nose_cone",
        loc=cq.Location(cq.Vector(0, 0, station)),
    )

    points = [
        (R_BODY, 0.0),
        (R_BODY + FIN_SPAN, FIN_SWEEP),
        (R_BODY + FIN_SPAN, FIN_SWEEP + FIN_TIP),
        (R_BODY, FIN_ROOT),
    ]
    blade = (
        cq.Workplane("XZ").polyline(points).close()
        .extrude(FIN_THICK).translate((0, -FIN_THICK / 2.0, 0))
    )
    for i in range(N_FINS):
        assembly.add(
            blade.rotate((0, 0, 0), (0, 0, 1), i * 360.0 / N_FINS),
            name=f"fin_{i + 1}",
        )

    path = directory / "lv1_assembly.step"
    # The assembly export, not a compound: flattening it to one shape is
    # exactly what discards the component names this path reads.
    assembly.export(str(path), "STEP")
    return path


@pytest.fixture(scope="module")
def assembly(tmp_path_factory):
    directory = tmp_path_factory.mktemp("assembly")
    return read_assembly(_write_assembly(directory))


def _assign(read) -> list[Assignment]:
    by_name = {c.name: c.index for c in read.components}
    fins = [by_name[f"fin_{i + 1}"] for i in range(N_FINS) if f"fin_{i + 1}" in by_name]
    if not fins:
        # An unnamed export: the blades are whatever sits off the centreline.
        fins = [
            c.index for c in read.components
            if c.offset_m(read.axis) > 0.5 * c.max_radius_m(read.axis)
        ]
    chosen = [Assignment(PartType.FINSET, "fins", fins)]
    for name, part_type in (
        ("nose_cone", PartType.NOSE),
        ("payload_bay", PartType.BODY),
        ("tank_press", PartType.TANK),
        ("intertank_2", PartType.INTERTANK),
        ("tank_rp1", PartType.TANK),
        ("intertank_1", PartType.INTERTANK),
        ("tank_lox", PartType.TANK),
        ("aft_skirt", PartType.INTERTANK),
    ):
        if name in by_name:
            chosen.append(Assignment(part_type, name, [by_name[name]]))
    return chosen


# ---------------------------------------------------------------- reading


def test_every_solid_comes_back_named(assembly):
    names = {c.name for c in assembly.components}
    assert assembly.named
    assert {name for name, _ in SECTIONS} <= names
    assert "nose_cone" in names
    assert sum(1 for n in names if n.startswith("fin_")) == N_FINS


def test_volumes_are_the_solids_own(assembly):
    """No fitting anywhere: a tank measures what a cylinder holds."""
    lox = next(c for c in assembly.components if c.name == "tank_lox")
    expected = 3.14159265 * (R_BODY ** 2) * 400.0 / 1e9
    assert lox.volume_m3 == pytest.approx(expected, rel=1e-6)


def test_blades_separate_from_bodies_on_geometry(assembly):
    """The two signals that tell a fin from a tube without knowing either.

    Both are needed. A cone fills exactly 1/3 of its own swept cylinder, so
    solidity alone would call the nose a blade; it is the centreline test that
    rules it out. Pinned here because the two thresholds have to stay either
    side of both numbers.
    """
    axis = assembly.axis
    for component in assembly.components:
        blade = component.name.startswith("fin_")
        off_axis = component.offset_m(axis) > 0.35 * component.max_radius_m(axis)
        assert (off_axis and component.solidity(axis) < BLADE_SOLIDITY) is blade

    nose = next(c for c in assembly.components if c.name == "nose_cone")
    assert nose.solidity(axis) == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert nose.offset_m(axis) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------ measurement


def test_fin_geometry_is_recovered_exactly(assembly):
    """Every fin number, against what the fixture was authored with.

    The blade lies off its own radial plane, so this also pins the azimuth
    coming from the shape rather than the centroid.
    """
    blades = [c for c in assembly.components if c.name.startswith("fin_")]
    fins = measure_finset(blades, assembly.axis)
    assert fins["count"] == N_FINS
    assert fins["root_chord_m"] * 1000 == pytest.approx(FIN_ROOT, abs=0.5)
    assert fins["tip_chord_m"] * 1000 == pytest.approx(FIN_TIP, abs=0.5)
    assert fins["span_m"] * 1000 == pytest.approx(FIN_SPAN, abs=0.5)
    assert fins["thickness_m"] * 1000 == pytest.approx(FIN_THICK, abs=0.2)


def test_reference_diameter_is_the_body_not_the_fin_span(assembly):
    """The regression this whole path exists for.

    Fitted as one silhouette, this vehicle reports a 220 mm diameter -- the
    fin tip span -- which is 4.8x the true reference area and wrong in every
    coefficient normalised by it.
    """
    from parametric import aero

    model, _ = build_model(assembly, _assign(assembly), name="LV-1")
    geometry = aero.extract_geometry(model)
    assert geometry.reference_diameter_m * 1000 == pytest.approx(2 * R_BODY, abs=1.0)


# ------------------------------------------------------------------ build


def test_parts_match_the_assignment(assembly):
    model, report = build_model(assembly, _assign(assembly), name="LV-1")
    kinds = {c.name: c.kind for c in model.walk() if c.kind != "component"}
    assert kinds["fins"] == "finset"
    assert kinds["tank_lox"] == "tank"
    assert kinds["nose_cone"] == "stack"
    assert report.parts == len(_assign(assembly))


def test_stations_run_from_the_declared_nose(assembly):
    """The nose is at zero even though it sits at the far end of the file."""
    model, _ = build_model(assembly, _assign(assembly), name="LV-1")
    assert model.find("nose_cone").station_range_m()[0] == pytest.approx(0.0, abs=1e-6)
    aft = model.find("aft_skirt").station_range_m()[1]
    assert aft == pytest.approx(assembly.total_length_m, abs=1e-6)


def test_mass_is_the_cad_volume_not_an_integrated_profile(assembly):
    """Stack.volume_m3 trapezoids section area, which overestimates a taper by
    50% when it is described by its two ends. An import has the exact volume,
    so it uses it."""
    from parametric.materials import get_material

    model, _ = build_model(assembly, _assign(assembly), name="LV-1")
    density = get_material("aluminium_6061_t6").density_kg_m3
    expected = sum(c.volume_m3 for c in assembly.components) * density
    total = sum(c.mass_kg() for c in model.walk() if c.kind != "component")
    assert total == pytest.approx(expected, rel=1e-9)


# ------------------------------------------------------------------- lock


def test_imported_parts_are_flagged_and_survive_a_round_trip(assembly):
    model, _ = build_model(assembly, _assign(assembly), name="LV-1")
    assert all(c.imported for c in model.walk() if c.kind != "component")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "lv1.json"
        path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
        reloaded = VehicleModel.from_dict(
            json.loads(path.read_text(encoding="utf-8"))
        )
    parts = [c for c in reloaded.walk() if c.kind != "component"]
    assert parts and all(c.imported for c in parts)


def test_a_sweep_will_not_drive_imported_geometry(assembly):
    model, _ = build_model(assembly, _assign(assembly), name="LV-1")
    assert sweepable_parms(model) == []


# ------------------------------------------------------------- suggestion


def test_the_suggestion_covers_every_solid_once(assembly):
    suggested = suggest(assembly)
    claimed = [index for a in suggested for index in a.indices]
    assert sorted(claimed) == sorted(c.index for c in assembly.components)
    assert len(claimed) == len(set(claimed))


def test_a_solid_assigned_as_a_protuberance_lands_at_its_station(assembly):
    """This raised AttributeError: the builder asked the part for a method
    the parm container never had, so no protuberance could be imported."""
    by_name = {c.name: c.index for c in assembly.components}
    chosen = [a for a in _assign(assembly) if a.name != "intertank_2"]
    chosen.append(
        Assignment(PartType.PROTUBERANCE, "camera", [by_name["intertank_2"]])
    )
    model, _ = build_model(assembly, chosen, name="LV-1")
    (item,) = model.protuberances
    assert item.name == "camera"
    low, high = model.station_range_m()
    assert low <= item.get("station") <= high


def test_the_suggestion_groups_the_fins_and_never_guesses_a_tank(assembly):
    """Geometry can say "blade"; nothing in it says "holds propellant"."""
    suggested = suggest(assembly)
    finsets = [a for a in suggested if a.part_type is PartType.FINSET]
    assert len(finsets) == 1
    assert len(finsets[0].indices) == N_FINS
    assert not [
        a for a in suggested
        if a.part_type in (PartType.TANK, PartType.INTERTANK)
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
