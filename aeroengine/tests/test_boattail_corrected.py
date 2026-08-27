"""The provisional boattail law, and the switch that selects it.

The corrected model is not validated against flight yet -- that is the
telemetry work's job -- so these tests pin only what can be pinned now:
the switch is explicit and total, the corrected law changes nothing where
there is no boattail, it can never go negative, and on a steep boattail it
charges less than the port's separation clamp does, which is the direction
both measured flights point.
"""

from __future__ import annotations

import pytest

from aeroengine.cdx1 import load as load_cdx1
from aeroengine.solver import Engine
from oracle.vehicles import test_matrix as _vehicle_matrix
from oracle.vehicles import write_cdx1

VEHICLES = {v.name: v for v in _vehicle_matrix()}


def _design(name, tmp_path):
    path = tmp_path / f"{name}.CDX1"
    write_cdx1(VEHICLES[name], path)
    return load_cdx1(path)


def test_unknown_model_is_refused(tmp_path):
    design = _design("body_only", tmp_path)
    with pytest.raises(ValueError, match="boattail_model"):
        Engine(design, boattail_model="hopeful")


def test_no_boattail_means_no_difference(tmp_path):
    design = _design("body_only", tmp_path)
    port = Engine(design)
    corrected = Engine(design, boattail_model="corrected")
    for mach in (1.2, 2.0, 3.5):
        assert corrected.solve(mach).cd_base == port.solve(mach).cd_base


def test_steep_boattail_charges_less_than_the_clamp(tmp_path):
    design = _design("boattail_steep", tmp_path)
    port = Engine(design)
    corrected = Engine(design, boattail_model="corrected")
    for mach in (1.2, 2.0, 3.5):
        a = port.solve(mach).cd_base
        b = corrected.solve(mach).cd_base
        assert 0.0 < b < a, (mach, a, b)


def test_subsonic_is_untouched(tmp_path):
    design = _design("boattail_steep", tmp_path)
    port = Engine(design)
    corrected = Engine(design, boattail_model="corrected")
    for mach in (0.3, 0.8):
        assert corrected.solve(mach).cd_off == port.solve(mach).cd_off


def test_default_is_the_port(tmp_path):
    design = _design("boattail_shallow", tmp_path)
    assert Engine(design).boattail_model == "rasaero"
