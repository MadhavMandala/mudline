"""Tests for the two facts geometry cannot carry: material and measured mass.

A STEP file is shape. What a part is made of, and what it weighed when someone
put it on a scale, are separate claims -- and if the tool guesses at either it
reports a mass that looks measured and is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.step_metadata import (  # noqa: E402
    PLAUSIBLE_DENSITY_KG_M3,
    StepMetadata,
    normalise_density,
    read_step_metadata,
    read_text_scan,
)

cq = pytest.importorskip("cadquery", reason="needs the cad extra")

from parametric.components import PointMass, Protuberance  # noqa: E402
from parametric.model import VehicleModel  # noqa: E402
from parametric.standard import basic_rocket  # noqa: E402


# ------------------------------------------------------------------- units


def test_kilograms_per_cubic_metre_pass_through():
    value, note = normalise_density(1850.0, "kg/m3")
    assert np.isclose(value, 1850.0)


def test_grams_per_cubic_centimetre_are_scaled():
    """The thousand-fold trap. CAD writes g/cm3 as often as kg/m3."""
    value, _ = normalise_density(1.85, "g/cm3")
    assert np.isclose(value, 1850.0)


def test_pounds_per_cubic_inch():
    value, _ = normalise_density(0.0975, "lb/in3")
    assert 2600.0 < value < 2800.0        # about aluminium


def test_unit_spelling_is_forgiven():
    for spelling in ("kg/m^3", "KG/M3", "kg / m3", "kg/m3."):
        value, _ = normalise_density(1850.0, spelling)
        assert np.isclose(value, 1850.0), spelling


def test_a_missing_unit_is_inferred_from_magnitude():
    """Single digits must be g/cm3; thousands must be kg/m3."""
    small, note = normalise_density(2.7, "")
    assert np.isclose(small, 2700.0)
    assert "g/cm" in note

    large, _ = normalise_density(2700.0, "")
    assert np.isclose(large, 2700.0)


def test_an_unrecognised_unit_is_not_assumed_to_be_si():
    """Guessing wrong by 1000x is worse than declining the number."""
    value, note = normalise_density(1850.0, "furlongs_per_hogshead")
    assert np.isclose(value, 1850.0)      # magnitude is plausible as kg/m3
    value, note = normalise_density(1850.0, "kg/mm3")
    assert value == 0.0                   # 1.85e12 kg/m3 is not a material
    assert "not a real material" in note


def test_implausible_densities_are_rejected():
    low, high = PLAUSIBLE_DENSITY_KG_M3
    assert normalise_density(high * 10, "kg/m3")[0] == 0.0
    assert normalise_density(0.0, "kg/m3")[0] == 0.0
    assert normalise_density(-5.0, "kg/m3")[0] == 0.0


# --------------------------------------------------------------- AP242 read


def write_step_with_material(path: Path, name: str, density: float,
                             unit: str = "kg/m3") -> Path:
    """An AP242 file carrying a material, written by OpenCascade itself."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.Interface import Interface_Static
    from OCP.STEPCAFControl import STEPCAFControl_Writer
    from OCP.TCollection import (
        TCollection_ExtendedString,
        TCollection_HAsciiString,
    )
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    application = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
    application.NewDocument(TCollection_ExtendedString("MDTV-CAF"), document)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())
    material_tool = XCAFDoc_DocumentTool.MaterialTool_s(document.Main())

    solid = BRepPrimAPI_MakeCylinder(50.0, 400.0).Shape()   # millimetres
    label = shape_tool.AddShape(solid, False)
    material_tool.SetMaterial(
        label,
        TCollection_HAsciiString(name),
        TCollection_HAsciiString("written by the test"),
        density,
        TCollection_HAsciiString("DENSITY"),
        TCollection_HAsciiString(unit),
    )

    Interface_Static.SetCVal_s("write.step.schema", "AP242DIS")
    writer = STEPCAFControl_Writer()
    writer.SetMaterialMode(True)
    writer.Transfer(document)
    writer.Write(str(path))
    return path


def write_plain_step(path: Path) -> Path:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
    from OCP.STEPControl import STEPControl_StepModelType, STEPControl_Writer

    solid = BRepPrimAPI_MakeCylinder(50.0, 400.0).Shape()
    writer = STEPControl_Writer()
    writer.Transfer(solid, STEPControl_StepModelType.STEPControl_AsIs)
    writer.Write(str(path))
    return path


@pytest.mark.slow
def test_material_is_read_from_an_ap242_file(tmp_path):
    path = write_step_with_material(tmp_path / "g10.stp", "G10 Fibreglass", 1850.0)
    meta = read_step_metadata(path)
    assert meta.has_material
    assert "G10" in meta.material_name
    assert np.isclose(meta.density_kg_m3, 1850.0)
    assert meta.source == "xcaf"


@pytest.mark.slow
def test_a_plain_step_carries_no_material(tmp_path):
    meta = read_step_metadata(write_plain_step(tmp_path / "plain.stp"))
    assert not meta.has_material
    assert "no material" in meta.text()


@pytest.mark.slow
def test_a_file_written_in_grams_per_cc_still_reads_correctly(tmp_path):
    path = write_step_with_material(
        tmp_path / "alu.stp", "Aluminium", 2.70, unit="g/cm3"
    )
    meta = read_step_metadata(path)
    assert np.isclose(meta.density_kg_m3, 2700.0, rtol=1e-6)


def test_the_text_scan_finds_a_named_material(tmp_path):
    path = tmp_path / "scan.stp"
    path.write_text(
        "ISO-10303-21;\n"
        "#42=MATERIAL_DESIGNATION('Ti-6Al-4V',(#7));\n"
        "END-ISO-10303-21;\n",
        encoding="utf-8",
    )
    meta = read_text_scan(path)
    assert meta.material_name == "Ti-6Al-4V"
    assert meta.source == "text scan"


def test_a_missing_file_does_not_raise(tmp_path):
    meta = read_text_scan(tmp_path / "nope.stp")
    assert not meta.has_material
    assert meta.notes


def test_metadata_text_is_readable():
    meta = StepMetadata("G10", 1850.0, 0.0, "xcaf")
    assert "G10" in meta.text() and "1,850" in meta.text()


# -------------------------------------------------------- import uses them


@pytest.mark.slow
def test_an_import_adopts_the_declared_material(tmp_path):
    from parametric.cad_import import import_step

    path = write_step_with_material(tmp_path / "g10.stp", "G10 Fibreglass", 1850.0)
    model, report = import_step(path)
    stack = model.stacks[0]

    assert report.material_source == "declared in the file"
    # A solid cylinder r=50 mm, h=400 mm of 1850 kg/m3.
    expected = np.pi * 0.05 ** 2 * 0.4 * 1850.0
    assert np.isclose(stack.mass_kg(), expected, rtol=0.02)


@pytest.mark.slow
def test_an_import_without_a_material_says_so(tmp_path):
    from parametric.cad_import import import_step

    _, report = import_step(write_plain_step(tmp_path / "plain.stp"))
    assert report.material_source == "defaulted"
    assert any("provisional" in note for note in report.notes)


@pytest.mark.slow
def test_a_forced_material_beats_the_file(tmp_path):
    from parametric.cad_import import import_step

    path = write_step_with_material(tmp_path / "g10.stp", "G10 Fibreglass", 1850.0)
    model, report = import_step(path, material="steel_4130")
    assert model.stacks[0].material == "steel_4130"
    assert report.material_source == "specified on import"


@pytest.mark.slow
def test_the_file_material_can_be_ignored(tmp_path):
    from parametric.cad_import import import_step

    path = write_step_with_material(tmp_path / "g10.stp", "G10 Fibreglass", 1850.0)
    _, report = import_step(path, use_file_material=False)
    assert report.material_source == "defaulted"


# ------------------------------------------------------------ mass override


def test_an_override_wins_over_the_geometry():
    model = basic_rocket()
    nose = model.find("nose")
    computed = nose.computed_mass_kg()

    nose.mass_override_kg = 0.620
    assert np.isclose(nose.mass_kg(), 0.620)
    # The geometry is untouched, because aero and CAD still read it.
    assert np.isclose(nose.computed_mass_kg(), computed)
    assert np.isclose(nose.volume_m3(), nose.volume_m3())


def test_clearing_an_override_restores_the_computed_mass():
    model = basic_rocket()
    nose = model.find("nose")
    computed = nose.computed_mass_kg()
    nose.mass_override_kg = 99.0
    nose.mass_override_kg = None
    assert np.isclose(nose.mass_kg(), computed)


def test_an_override_moves_the_vehicle_mass_and_cg():
    model = basic_rocket()
    before = model.mass_summary()
    model.find("nose").mass_override_kg = 1.500      # a heavy nose
    after = model.mass_summary()

    assert after.dry_mass_kg > before.dry_mass_kg
    # The nose is forward of the CG, so a heavier one pulls the CG forward.
    assert after.cg_station_m < before.cg_station_m


def test_disagreement_is_reported():
    model = basic_rocket()
    nose = model.find("nose")
    computed = nose.computed_mass_kg()
    nose.mass_override_kg = computed * 1.30
    assert np.isclose(nose.override_disagreement, 0.30, rtol=1e-6)


def test_no_override_means_no_disagreement():
    assert basic_rocket().find("nose").override_disagreement == 0.0


def test_overrides_work_on_point_masses_and_protuberances():
    model = VehicleModel("t")
    point = PointMass("avionics", 0.5, 0.2)
    lug = Protuberance("lug", mass_kg=0.05, count=2)
    model.add(point)
    model.add(lug)

    assert np.isclose(point.mass_kg(), 0.5)
    assert np.isclose(lug.mass_kg(), 0.10)

    point.mass_override_kg = 0.75
    lug.mass_override_kg = 0.20
    assert np.isclose(point.mass_kg(), 0.75)
    assert np.isclose(lug.mass_kg(), 0.20)


def test_growth_applies_to_the_measured_mass():
    """Weighed hardware plus a growth allowance is still the weighed figure."""
    point = PointMass("avionics", 0.5, 0.2)
    point.set("growth", 0.20)
    point.mass_override_kg = 1.0
    assert np.isclose(point.mass_with_growth_kg, 1.20)


def test_a_motor_is_left_alone():
    """Its dry mass is zero by design; propellant is tracked separately."""
    motor = basic_rocket().motors[0]
    assert motor.mass_kg() == 0.0


def test_an_override_survives_a_save(tmp_path):
    model = basic_rocket()
    model.find("nose").mass_override_kg = 0.620
    reloaded = VehicleModel.load(model.save(tmp_path / "v.json"))
    assert np.isclose(reloaded.find("nose").mass_override_kg, 0.620)


def test_a_cleared_override_survives_a_save(tmp_path):
    model = basic_rocket()
    model.find("nose").mass_override_kg = None
    reloaded = VehicleModel.load(model.save(tmp_path / "v.json"))
    assert reloaded.find("nose").mass_override_kg is None


def test_an_older_file_has_no_overrides(tmp_path):
    import json

    model = basic_rocket()
    path = model.save(tmp_path / "old.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    def strip(node):
        node.pop("mass_override_kg", None)
        for child in node.get("children", []):
            strip(child)

    strip(data["tree"])
    path.write_text(json.dumps(data), encoding="utf-8")
    assert VehicleModel.load(path).find("nose").mass_override_kg is None


def test_a_registered_material_round_trips(tmp_path):
    """An imported density becomes a named material, so the file reloads."""
    from parametric.materials import get_material, material_named

    entry = material_named(1850.0, "G10 Fibreglass")
    assert np.isclose(get_material(entry.name).density_kg_m3, 1850.0)

    model = basic_rocket()
    model.find("nose").material = entry.name
    reloaded = VehicleModel.load(model.save(tmp_path / "v.json"))
    assert reloaded.find("nose").material == entry.name
    assert reloaded.find("nose").mass_kg() > 0


def test_registering_does_not_clobber_a_curated_material():
    """The built-in entries carry strength and modulus an import does not."""
    from parametric.materials import Material, get_material, register_material

    before = get_material("steel_4130")
    register_material(Material("steel_4130", 1.0))
    assert get_material("steel_4130").density_kg_m3 == before.density_kg_m3


@pytest.mark.slow
def test_the_solved_mass_honours_an_override(tmp_path):
    """Solving used to discard it, which is exactly when it matters.

    The mesh gives the shape -- centroid and inertia distribution -- and the
    scale gives the amount. Before this, a weighed part reverted to
    volume x density the moment mass properties were run, and the trajectory
    then flew on the reverted figure.
    """
    from parametric import analysis

    model = basic_rocket()
    plain = analysis.solve_mass(model, tmp_path / "plain")

    nose = model.find("nose")
    nose.mass_override_kg = nose.computed_mass_kg() * 2.0
    solved = analysis.solve_mass(model, tmp_path / "over")

    assert solved.mass_kg > plain.mass_kg
    # And it agrees with the analytic roll-up, which already honoured it.
    assert np.isclose(solved.mass_kg, model.mass_summary().dry_mass_kg, rtol=0.02)


@pytest.mark.slow
def test_an_override_reaches_the_simulator(tmp_path):
    from parametric import analysis

    model = basic_rocket()
    before = analysis.build_simulation(model).mass_props.dry_mass

    model.find("nose").mass_override_kg = 1.500
    after = analysis.build_simulation(model).mass_props.dry_mass
    assert after > before
