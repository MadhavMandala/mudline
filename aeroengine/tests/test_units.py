"""Guards on the SI-to-English boundary.

A missed unit conversion here is silent: every coefficient comes back
dimensionless and plausible while the vehicle the engine solved is 39 times
too small. Reynolds number is the one quantity that exposes it immediately,
which is why it is asserted first and directly.
"""

from __future__ import annotations

import math

import pytest

from aeroengine.adapters import IN_PER_M, surface_for_roughness
from aeroengine.atmosphere import STANDARD, reynolds
from aeroengine.parts import SURFACE_ROUGHNESS


def test_inches_per_metre_is_exact_reciprocal_of_the_writers():
    """The CDX1 writer uses 0.0254 m/in; a mismatch would break round trips."""
    assert IN_PER_M == pytest.approx(1.0 / 0.0254, rel=0, abs=1e-12)


def test_reynolds_at_sea_level_matches_rasaero():
    """RASAero reports 3 550 359 for a 60 in body at Mach 0.10.

    Taken from its own Run Test output. This is the strongest single assertion
    against a unit error anywhere upstream: a length in metres instead of
    inches moves this by a factor of 39.
    """
    assert reynolds(0.10, 60.0, 0.0, STANDARD) == pytest.approx(3_550_359, rel=2e-6)


def test_run_test_ignores_launch_site():
    """The default atmosphere is standard sea level with every offset zero.

    RASAero's Run Test path never applies the LaunchSite block, so the
    reference data is standard-day referenced regardless of what a CDX1 says.
    Reproducing that is what makes the golden comparison valid.
    """
    assert STANDARD.dh_density == 0.0
    assert STANDARD.dh_temperature == 0.0
    assert STANDARD.dh_pressure == 0.0
    assert STANDARD.speed_of_sound(0.0) == pytest.approx(math.sqrt(2403.07606 * 518.69))


def test_every_rasaero_finish_is_reachable():
    """All eight, including the roughest.

    The tool's own table was short by one for a while, so no roughness however
    large could select Cast Iron. Round-tripping each height back through the
    selector catches a regression of that shape.
    """
    assert len(SURFACE_ROUGHNESS) == 8
    for name, height_in in SURFACE_ROUGHNESS.items():
        if height_in == 0.0:
            continue
        assert surface_for_roughness(height_in / IN_PER_M) == name
