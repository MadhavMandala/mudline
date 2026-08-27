"""Reading RASAero's parts off the model instead of fitting them to a curve.

The distinction matters because the fit is free to put a breakpoint wherever
it minimises error, which is not the same place as the joint between two real
parts. On the reference rocket that difference is 8 mm of nose length, and
nose length sets both the nose centre of pressure and the fineness ratio the
wave drag is computed from.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric import analysis  # noqa: E402
from parametric.components import NoseProfile, Stack, Tank  # noqa: E402
from parametric.model import VehicleModel  # noqa: E402
from parametric.roles import AeroRole  # noqa: E402
from parametric.standard import basic_rocket  # noqa: E402
from parametric.canonical import CanonicalSegment, surface_finish  # noqa: E402


def test_segment_boundaries_land_on_real_parts():
    model = basic_rocket()
    nose = model.find("nose")
    canonical = analysis.to_canonical(model)

    segments = {segment.kind: segment for segment in canonical.segments}
    assert np.isclose(segments["nose"].length_m, nose.station_range_m()[1])
    # The two same-diameter tubes behind it are one tube to the flow.
    assert np.isclose(segments["tube"].end_m, model.total_length_m)


def test_the_measured_nose_is_not_the_fitted_one():
    """Both are plausible; only one is the length the part actually has."""
    model = basic_rocket()
    truth = model.find("nose").station_range_m()[1]

    measured = analysis.to_canonical(model, measured=True)
    fitted = analysis.to_canonical(model, measured=False)

    measured_nose = next(s for s in measured.segments if s.kind == "nose")
    fitted_nose = next(s for s in fitted.segments if s.kind == "nose")
    assert np.isclose(measured_nose.length_m, truth, atol=1e-9)
    assert not np.isclose(fitted_nose.length_m, truth, atol=1e-4)


def test_a_declared_role_overrides_the_shape():
    """A tapered part told it is a body tube is written as a tube."""
    model = VehicleModel("declared")
    nose = Stack("nose")
    nose.add_nose(NoseProfile.OGIVE, 0.3, 0.1)
    model.add(nose)

    tapered = Stack("skirt")
    tapered.add_tube(0.001, 0.1)
    tapered.add_transition(0.4, 0.08)
    tapered.set_station_m(0.3)
    model.add(tapered)

    inferred = analysis.to_canonical(model)
    assert any(s.kind == "boattail" for s in inferred.segments)

    tapered.aero_role = AeroRole.BODY
    declared = analysis.to_canonical(model)
    assert not any(s.kind == "boattail" for s in declared.segments)
    assert all("declared" in "".join(s.sources) or s.kind == "nose"
               for s in declared.segments if s.kind != "nose")


def test_internal_parts_are_left_out_of_the_mould_line():
    model = basic_rocket()
    before = analysis.to_canonical(model).total_length_m

    sled = Stack("sled")
    sled.add_tube(0.2, 0.04)
    sled.set_station_m(2.5)             # deliberately past the tail
    sled.aero_role = AeroRole.INTERNAL
    model.add(sled)

    assert np.isclose(analysis.to_canonical(model).total_length_m, before)


def test_a_tank_reaches_the_canonical_model():
    """Tanks are outer mould line, and the silhouette walk does not see them."""
    model = VehicleModel("tankship")
    nose = Stack("nose")
    nose.add_nose(NoseProfile.OGIVE, 0.4, 0.2)
    model.add(nose)

    tank = Tank("lox")
    tank.set("diameter", 0.2)
    tank.set("barrel_length", 1.0)
    tank.set("station", 0.4)
    model.add(tank)

    canonical = analysis.to_canonical(model)
    assert canonical.total_length_m > 1.0
    assert any("lox" in "".join(s.sources) for s in canonical.segments)


def test_clipped_tanks_merge_into_one_body_tube():
    """Flush clips bury each dome in the next bay, so the tank's tube and the
    bay's tube overlap by a dome height. The flow sees one straight tube; the
    writer can only take one; and until the merge accepted overlap every
    clipped-tank vehicle was written as its first two parts, fins and all
    dropped without a word."""
    model = VehicleModel("clipped")
    nose = Stack("nose")
    nose.add_nose(NoseProfile.OGIVE, 0.4, 0.2)
    model.add(nose)
    bay = Stack("bay")
    bay.add_tube(0.3, 0.2, name="bay")
    model.add(bay)
    tank = Tank("lox")
    tank.set("diameter", 0.2)
    tank.set("barrel_length", 1.0)
    model.add(tank)
    skirt = Stack("skirt")
    skirt.add_tube(0.5, 0.2, name="skirt")
    model.add(skirt)
    for part, ahead in ((bay, nose), (tank, bay), (skirt, tank)):
        part.clip_to = ahead.path
        part.clip_locked = True
    model.apply_clips()

    canonical = analysis.to_canonical(model)
    tubes = [s for s in canonical.segments if s.kind == "tube"]
    assert len(tubes) == 1, [(s.kind, s.start_m, s.end_m) for s in canonical.segments]
    assert np.isclose(tubes[0].start_m, 0.4, atol=1e-3)   # less the clip overlap
    assert np.isclose(tubes[0].end_m, model.total_length_m, atol=1e-3)
    assert "lox" in "".join(tubes[0].sources)


def test_overlapping_tubes_union_but_a_gap_stays_a_gap():
    def tube(start, length):
        return CanonicalSegment("tube", start, length, 0.2, 0.2, sources=[f"t{start}"])

    overlapping = analysis._merge_tubes([tube(0.0, 0.5), tube(0.42, 0.5)])
    assert len(overlapping) == 1
    assert np.isclose(overlapping[0].end_m, 0.92)
    assert overlapping[0].sources == ["t0.0", "t0.42"]

    # A tube wholly inside another adds nothing and removes nothing.
    nested = analysis._merge_tubes([tube(0.0, 1.0), tube(0.3, 0.2)])
    assert len(nested) == 1 and np.isclose(nested[0].end_m, 1.0)

    # Daylight between two bodies is geometry, not a joint to be closed.
    apart = analysis._merge_tubes([tube(0.0, 0.5), tube(0.6, 0.5)])
    assert len(apart) == 2


def test_a_model_with_nothing_readable_falls_back_to_fitting():
    model = VehicleModel("empty")
    assert analysis.canonical_from_components(model) is None


@pytest.mark.parametrize("roughness_um,expected", [
    (0.0, "Smooth (Zero Roughness)"),
    (0.5, "Polished"),
    (6.0, "Smooth Paint"),
    (30.0, "Rough Camouflage Paint"),
    (150.0, "Galvanized Metal"),
    # RASAero offers eight finishes and this table carried seven, so nothing
    # above ~150 um could ever select the roughest one. 500 um is closer to
    # cast iron (254 um) than to galvanized metal (150 um) in log height,
    # which is the metric surface_finish uses -- this case previously
    # asserted the wrong answer because the right one did not exist.
    (500.0, "Cast Iron (Very Rough)"),
])
def test_roughness_picks_a_finish_rasaero_offers(roughness_um, expected):
    assert surface_finish(roughness_um * 1e-6) == expected


def test_every_rasaero_finish_is_selectable():
    """Guards the omission above from recurring.

    A finish present in the table but unreachable through ``surface_finish``
    is invisible: no roughness selects it and no test notices, which is
    exactly how the eighth entry went missing.
    """
    from parametric.canonical import SURFACE_ROUGHNESS_M

    assert len(SURFACE_ROUGHNESS_M) == 8
    for name, height_m in SURFACE_ROUGHNESS_M.items():
        if height_m > 0.0:
            assert surface_finish(height_m) == name
