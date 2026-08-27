"""Every body tube reaches RASAero, and the fins land on theirs.

Only the first tube used to be written. A vehicle that stepped down to a
narrower motor tube -- the boattail demonstrator -- lost that tube and had
its fins clamped to the first one, 0.8 m forward, and its own table then
called it 2.7 calibres unstable and flew it 200 m. Runs under pytest, and
standalone via ``python parametric/tests/test_bridge_tubes.py``.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric import analysis  # noqa: E402
from parametric.canonical import (  # noqa: E402
    CanonicalFin,
    CanonicalModel,
    CanonicalSegment,
    NoseShape,
    to_rasaero_model,
)
from parametric.standard import boattailed_rocket  # noqa: E402
from step_to_rasaero.rasaero_writer import write_rasaero_cdx1  # noqa: E402

IN_PER_M = 39.3700787401575


def _stepped() -> CanonicalModel:
    """Nose, wide tube, reducing step, narrow tube with the fins, boattail."""
    segments = [
        CanonicalSegment("nose", 0.0, 0.5, 0.0, 0.13, nose_shape=NoseShape.OGIVE),
        CanonicalSegment("tube", 0.5, 0.4, 0.13, 0.13),
        CanonicalSegment("boattail", 0.9, 0.1, 0.13, 0.10),
        CanonicalSegment("tube", 1.0, 0.8, 0.10, 0.10),
        CanonicalSegment("boattail", 1.8, 0.1, 0.10, 0.07),
    ]
    model = CanonicalModel("stepped", segments, fins=[CanonicalFin(
        count=4, root_chord_m=0.2, tip_chord_m=0.1, span_m=0.1, sweep_m=0.05,
        thickness_m=0.004, station_m=1.55,
    )])
    model.cg_from_nose_m = 1.0
    model.wet_mass_kg = 5.0
    return model


def _written(model: CanonicalModel):
    with tempfile.TemporaryDirectory() as directory:
        path = write_rasaero_cdx1(to_rasaero_model(model), Path(directory) / "s.CDX1")
        root = ET.parse(path).getroot()
        from aeroengine.cdx1 import load

        return root, load(path)


def test_every_tube_is_written_in_order():
    root, design = _written(_stepped())
    tags = [el.tag for el in root.find("RocketDesign") if el.tag in ("NoseCone", "BodyTube", "Transition", "BoatTail")]
    assert tags == ["NoseCone", "BodyTube", "Transition", "BodyTube", "BoatTail"]
    tubes = root.findall("RocketDesign/BodyTube")
    assert [round(float(t.findtext("Location")) / IN_PER_M, 3) for t in tubes] == [0.5, 1.0]
    assert [round(float(t.findtext("Length")) / IN_PER_M, 3) for t in tubes] == [0.4, 0.8]


def test_a_mid_body_reduction_is_a_transition_and_the_last_a_boattail():
    root, design = _written(_stepped())
    transition = root.find("RocketDesign/Transition")
    assert float(transition.findtext("FrontDiameter")) == pytest.approx(0.13 * IN_PER_M)
    assert float(transition.findtext("Diameter")) == pytest.approx(0.10 * IN_PER_M)
    boattail = root.find("RocketDesign/BoatTail")
    assert float(boattail.findtext("Location")) == pytest.approx(1.8 * IN_PER_M)
    from aeroengine.parts import Reducer

    reducers = [p for p in design.parts if isinstance(p, Reducer)]
    assert len(reducers) == 2


def test_the_fins_sit_on_the_tube_that_carries_them():
    root, design = _written(_stepped())
    tubes = root.findall("RocketDesign/BodyTube")
    assert tubes[0].find("Fin") is None and tubes[1].find("Fin") is not None
    from aeroengine.parts import PartType

    fins = [p for p in design.parts if p.part_type is PartType.FINS]
    assert len(fins) == 1
    assert fins[0].x0 / IN_PER_M == pytest.approx(1.55, abs=1e-4)      # inches, six figures


def test_the_fittings_ride_on_the_first_tube_only():
    payload = to_rasaero_model(_stepped())
    payload["protuberances"] = {"launch_lug_diameter_m": 0.01, "launch_lug_length_m": 0.05}
    with tempfile.TemporaryDirectory() as directory:
        path = write_rasaero_cdx1(payload, Path(directory) / "s.CDX1")
        tubes = ET.parse(path).getroot().findall("RocketDesign/BodyTube")
    assert float(tubes[0].findtext("LaunchLugLength")) > 0.0
    assert float(tubes[1].findtext("LaunchLugLength")) == 0.0


def test_the_boattail_demonstrator_is_stable_on_its_own_table():
    model = boattailed_rocket()
    # Marginal by its own design -- small fins on the narrow tube -- but
    # stable: it was 2.7 calibres unstable with its motor tube dropped.
    margin = analysis.static_margin(model, loaded=True)
    assert margin > 0.5, margin
    assert analysis.centre_of_pressure(model, 0.3) > analysis.loaded_cg_station_m(model)


def test_the_boattail_demonstrator_flies_on_its_table():
    from parametric import aero
    from parametric.flight import FlightSettings, fly_model

    model = boattailed_rocket()
    table, _ = aero.run_analysis(model, aero.AeroSettings(
        mach_min=0.05, mach_max=1.5, mach_points=8, alpha_max_deg=8.0, alpha_points=3,
    ))
    outcome = fly_model(model, FlightSettings(couple_aero_altitude=False, dt_s=0.05), table)
    assert outcome.apogee_agl_m > 2000.0
    log = outcome.log
    t = np.asarray(log.time_s)
    boost = (t > outcome.rail_exit["time_s"] + 0.5) & (t < 10.0)
    assert np.max(np.asarray(log.alpha_deg)[boost]) < 3.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
