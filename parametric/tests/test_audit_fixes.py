"""Regressions from the correctness audit of 25 August 2026: the model side.

Runs under pytest, and standalone via
``python parametric/tests/test_audit_fixes.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.canonical import (  # noqa: E402
    CanonicalFin,
    CanonicalModel,
    CanonicalSegment,
    NoseShape,
    to_rasaero_model,
)
from parametric.components import FinSet, Motor, Stack  # noqa: E402
from parametric.model import VehicleModel  # noqa: E402
from parametric.standard import basic_rocket, biprop_testbed, boattailed_rocket  # noqa: E402
from parametric.xsec import NoseProfile  # noqa: E402
from step_to_rasaero.rasaero_writer import write_rasaero_cdx1  # noqa: E402

IN_PER_M = 39.3700787401575


# ------------------------------------------------------------ the parms


def test_a_wall_survives_nose_generation():
    """The linked bound clamped the wall to the tip section and wrote it back."""
    nose = Stack("nose", wall_thickness_m=0.003)
    nose.add_nose(NoseProfile.OGIVE, 0.45, 0.10, sections=16)
    assert nose.get("wall_thickness") == 0.003


def test_the_boattail_demonstrator_builds_at_its_declared_wall():
    model = boattailed_rocket()
    airframe = model.stacks[0]
    assert airframe.get("wall_thickness") == pytest.approx(0.003)


def test_a_set_value_is_still_clamped_to_its_linked_bound():
    """Edits are bounded; only the silent write-back is gone."""
    tube = Stack("tube", wall_thickness_m=0.002)
    tube.add_tube(0.6, 0.10)
    tube.set("wall_thickness", 0.5)
    assert tube.get("wall_thickness") == pytest.approx(0.05)


# ------------------------------------------------------------- the mass


def test_the_grain_stops_at_the_nozzle():
    motor = Motor("m", propellant_mass_kg=3.0, station_m=1.0, length_m=0.5)
    motor.set("nozzle_length", 0.25)
    assert motor.grain_range_m() == (1.0, 1.5)
    assert motor.station_range_m() == (1.0, 1.75)
    assert motor.centroid_station_m == pytest.approx(1.25)


def test_a_shell_centroid_sits_forward_of_the_enclosed_one():
    model = basic_rocket()
    nose = model.find("nose")
    shell = model._stack_centroid(nose)
    stations = np.array([s.station_m for s in nose.sorted_sections()])
    areas = np.array([s.area_m2 for s in nose.sorted_sections()])
    enclosed = float(np.trapezoid(areas * stations, stations) / np.trapezoid(areas, stations))
    assert enclosed - shell > 0.015, (shell, enclosed)


def test_fin_planform_centroid_reduces_to_the_known_shapes():
    rectangle = FinSet("r", count=3, root_chord_m=0.2, tip_chord_m=0.2, span_m=0.1,
                       sweep_m=0.0, station_m=1.0)
    assert rectangle.planform_centroid_station_m == pytest.approx(1.1)
    triangle = FinSet("t", count=3, root_chord_m=0.3, tip_chord_m=0.0, span_m=0.1,
                      sweep_m=0.12, station_m=1.0)
    assert triangle.planform_centroid_station_m == pytest.approx(1.0 + (0.12 + 0.3) / 3.0)


def test_propellant_mass_counts_the_tanks():
    model = biprop_testbed()
    assert model.propellant_mass_kg == pytest.approx(model.mass_summary().propellant_mass_kg)
    assert model.propellant_mass_kg > 50.0


def test_the_station_range_ignores_the_root():
    model = VehicleModel("shifted")
    body = Stack("body", wall_thickness_m=0.002)
    body.add_tube(1.0, 0.1)
    body.set_forward_station_m(0.5)
    model.root.add(body) if hasattr(model.root, "add") else model.add(body)
    assert model.station_range_m() == pytest.approx((0.5, 1.5))
    assert model.total_length_m == pytest.approx(1.0)


def test_a_flat_curve_delivers_the_impulse_it_was_asked_for():
    motor = Motor("m", propellant_mass_kg=3.0)
    for burn in (0.4, 1.0, 3.25, 25.0):
        motor.curve_from_impulse(10000.0, burn)
        assert motor.total_impulse_ns == pytest.approx(10000.0, rel=1e-9), burn


# --------------------------------------------------------- the project file


def _canonical(nose_start: float, fins_at: float, tail: str = "boattail") -> CanonicalModel:
    segments = [
        CanonicalSegment("nose", nose_start, 0.45, 0.0, 0.10, nose_shape=NoseShape.OGIVE),
        CanonicalSegment("tube", nose_start + 0.45, 1.00, 0.10, 0.10),
        CanonicalSegment(tail, nose_start + 1.45, 0.15, 0.10, 0.14 if tail == "transition" else 0.07),
    ]
    model = CanonicalModel("t", segments, fins=[CanonicalFin(
        count=3, root_chord_m=0.2, tip_chord_m=0.1, span_m=0.08, sweep_m=0.05,
        thickness_m=0.004, station_m=fins_at,
    )])
    model.cg_from_nose_m = nose_start + 1.0
    model.wet_mass_kg = 5.0
    return model


def test_every_station_leaves_relative_to_the_nose_tip():
    payload = to_rasaero_model(_canonical(nose_start=0.5, fins_at=1.7))
    assert payload["nose"]["start"] == 0.0
    assert payload["body_sections"][0]["start"] == pytest.approx(0.45)
    assert payload["fins"][0]["axial_location"] == pytest.approx(1.2)
    assert payload["metadata"]["source_metadata"]["cg_from_nose_m"] == pytest.approx(1.0)


def test_an_expansion_is_a_transition_not_a_boattail():
    from aeroengine.cdx1 import load
    from aeroengine.parts import Expansion

    payload = to_rasaero_model(_canonical(nose_start=0.0, fins_at=1.0, tail="transition"))
    assert payload["body_sections"][-1]["type"] == "transition"
    with tempfile.TemporaryDirectory() as directory:
        path = write_rasaero_cdx1(payload, Path(directory) / "t.CDX1")
        root = ET.parse(path).getroot()
        transition = root.find("RocketDesign/Transition")
        assert transition is not None
        assert float(transition.findtext("FrontDiameter")) == pytest.approx(0.10 * IN_PER_M)
        assert float(transition.findtext("Diameter")) == pytest.approx(0.14 * IN_PER_M)
        design = load(path)
    assert any(isinstance(part, Expansion) for part in design.parts)


def test_fins_on_the_boattail_are_written_there():
    from aeroengine.cdx1 import load
    from aeroengine.parts import PartType

    payload = to_rasaero_model(_canonical(nose_start=0.0, fins_at=1.50))
    with tempfile.TemporaryDirectory() as directory:
        path = write_rasaero_cdx1(payload, Path(directory) / "b.CDX1")
        root = ET.parse(path).getroot()
        assert root.find("RocketDesign/BodyTube/Fin") is None
        assert root.find("RocketDesign/BoatTail/Fin") is not None
        assert root.findtext("RocketDesign/BodyTube/BoattailLength") == "0"
        design = load(path)
    fins = [part for part in design.parts if part.part_type is PartType.FINS]
    assert len(fins) == 1
    assert fins[0].x0 / IN_PER_M == pytest.approx(1.50, abs=1e-6)


# ------------------------------------------------------------ the bridge


def test_the_status_bar_and_the_table_agree_on_alpha():
    from parametric import aero, analysis

    model = basic_rocket()
    table, _ = aero.run_analysis(model, aero.AeroSettings(
        mach_min=0.1, mach_max=1.0, mach_points=10, alpha_max_deg=8.0, alpha_points=5,
    ))
    assert analysis.centre_of_pressure(model, 0.3) == pytest.approx(
        table.lookup(0.3, 4.0).x_cp_m, abs=2e-3,
    )


def test_the_report_margin_is_the_loaded_margin():
    from parametric import aero, analysis

    model = basic_rocket()
    table, geometry = aero.run_analysis(model, aero.AeroSettings(
        mach_min=0.1, mach_max=1.0, mach_points=10, alpha_max_deg=8.0, alpha_points=5,
    ))
    report = aero.analysis_report(model, table, geometry, None)
    row = next(line for line in report.splitlines() if line.strip().startswith("0.30"))
    printed = float(row.split()[-1])
    assert printed == pytest.approx(analysis.static_margin(model, loaded=True), abs=0.05)


def test_the_planform_carries_half_the_fins_at_their_centroid():
    from parametric import aero

    model = basic_rocket()
    table, geometry = aero.run_analysis(model, aero.AeroSettings(
        mach_min=0.1, mach_max=0.5, mach_points=3, alpha_max_deg=4.0, alpha_points=2,
    ))
    fins = model.fin_sets[0]
    assert table.high_alpha.fin_area_m2 == pytest.approx(0.5 * fins.count * fins.area_per_fin_m2)
    assert table.high_alpha.fin_centroid_m == pytest.approx(fins.planform_centroid_station_m)


def test_the_planform_survives_a_csv_round_trip(tmp_path):
    from parametric import aero
    from trajectory.vehicle.aero_database import AeroDatabase

    model = basic_rocket()
    table, _ = aero.run_analysis(model, aero.AeroSettings(
        mach_min=0.1, mach_max=0.5, mach_points=3, alpha_max_deg=4.0, alpha_points=2,
    ))
    reloaded = AeroDatabase.from_csv(table.to_csv(tmp_path / "t.csv"), reference_length_m=2.0)
    assert reloaded.high_alpha is not None
    assert reloaded.high_alpha.fin_area_m2 == pytest.approx(table.high_alpha.fin_area_m2)
    assert reloaded.high_alpha.planform_centroid_m == pytest.approx(
        table.high_alpha.planform_centroid_m
    )


@pytest.mark.slow
def test_the_meshed_solve_takes_its_mass_from_the_closed_form():
    """The mesh gives the shape; the shell integral, exact, gives the amount."""
    pytest.importorskip("cadquery")
    from parametric import analysis

    model = boattailed_rocket()
    with tempfile.TemporaryDirectory() as directory:
        solved = analysis.solve_mass(model, directory)
    airframe = model.stacks[0]
    assert solved.per_component_kg[airframe.path] == pytest.approx(airframe.mass_kg(), rel=1e-9)
    assert solved.mass_kg == pytest.approx(model.mass_summary().dry_mass_kg, rel=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
