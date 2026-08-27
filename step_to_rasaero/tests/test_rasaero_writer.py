"""The CDX1 writer writes every body tube, in station order.

It used to write the first and warn about the rest, on the belief that
RASAero's single-stage schema held one tube. It holds any number; the
demonstrator that steps down to a narrower motor tube lost that tube and
the fins on it, and its own table then called it unstable.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from parametric.canonical import (
    CanonicalFin,
    CanonicalModel,
    CanonicalSegment,
    NoseShape,
    to_rasaero_model,
)
from step_to_rasaero.rasaero_writer import _write_body, write_rasaero_cdx1

IN_PER_M = 39.37007874015748


def _model(*tubes):
    return {
        "nose": {"length": 0.4, "base_diameter": 0.2},
        "rocket": {"max_diameter": 0.2},
        "body_sections": [
            {"type": "tube", "start": start, "length": length, "diameter": 0.2}
            for start, length in tubes
        ],
    }


def _canonical(*tubes, fins_at: float | None = None) -> CanonicalModel:
    segments = [CanonicalSegment("nose", 0.0, 0.4, 0.0, 0.2, nose_shape=NoseShape.OGIVE)]
    segments += [CanonicalSegment("tube", start, length, 0.2, 0.2) for start, length in tubes]
    fins = []
    if fins_at is not None:
        fins.append(CanonicalFin(count=3, root_chord_m=0.2, tip_chord_m=0.1, span_m=0.08,
                                 sweep_m=0.05, thickness_m=0.004, station_m=fins_at))
    model = CanonicalModel("t", segments, fins=fins)
    model.cg_from_nose_m = 1.0
    model.wet_mass_kg = 5.0
    return model


def test_one_tube_is_written_silently():
    model = _model((0.4, 1.5))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        elem = _write_body(ET.Element("RocketDesign"), model, model["body_sections"][0])
    assert float(elem.find("Length").text) == pytest.approx(1.5 * IN_PER_M, abs=1e-3)


def test_every_tube_is_written_in_order(tmp_path):
    """Three tubes, three <BodyTube> parts, each at its own station."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        path = write_rasaero_cdx1(
            to_rasaero_model(_canonical((0.4, 0.6), (1.0, 0.8), (1.8, 0.5), fins_at=2.0)),
            tmp_path / "t.CDX1",
        )
    tubes = ET.parse(path).getroot().findall("RocketDesign/BodyTube")
    # Inches to six figures, so a few microns of rounding.
    assert [float(t.findtext("Location")) / IN_PER_M for t in tubes] == pytest.approx([0.4, 1.0, 1.8], abs=1e-4)
    assert [float(t.findtext("Length")) / IN_PER_M for t in tubes] == pytest.approx([0.6, 0.8, 0.5], abs=1e-4)
    assert [t.find("Fin") is not None for t in tubes] == [False, False, True]


def test_a_fin_on_no_tube_is_placed_loudly(tmp_path):
    """A fin whose leading edge sits in a gap goes on the tube ahead of it, with a warning."""
    with pytest.warns(UserWarning, match=r"lies on no tube"):
        path = write_rasaero_cdx1(
            to_rasaero_model(_canonical((0.4, 0.6), (1.5, 0.5), fins_at=1.2)),
            tmp_path / "t.CDX1",
        )
    tubes = ET.parse(path).getroot().findall("RocketDesign/BodyTube")
    assert tubes[0].find("Fin") is not None
