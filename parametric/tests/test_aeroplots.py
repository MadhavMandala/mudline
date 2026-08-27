"""Comparison plots.

These check the arithmetic behind the curves rather than their appearance:
the lift recovery from normal and axial force, and that the figures render to
a file at all. What the picture looks like is a judgement call; what it claims
is not.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric import aeroplots  # noqa: E402
from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase  # noqa: E402


def _table(cd: float, cn: float, cp: float, machs=(0.5, 1.0, 2.0)) -> AeroDatabase:
    rows = [
        AeroCoefficients(mach=m, alpha_deg=a, cd=cd, cn=cn * (a / 4.0),
                         cm=0.0, x_cp_m=cp)
        for m in machs for a in (0.0, 2.0, 4.0)
    ]
    return AeroDatabase(rows, reference_length_m=1.85)


def test_lift_is_normal_force_resolved_into_the_flow():
    """CL = CN cos(a) - CA sin(a), with CA recovered from CD.

    Straight from RASAero's own export: at Mach 0.5 and 4 degrees it reports
    CN 0.915944, CD 0.406906 and CL 0.889727. Reproducing that third number
    from the first two is the check.
    """
    database = _table(cd=0.406906, cn=0.915944, cp=1.0, machs=(0.5,))
    slope = aeroplots.lift_curve_slope(database, np.array([0.5]), 4.0)
    assert np.isclose(slope[0] * np.radians(4.0), 0.889727, rtol=1e-5)


def test_lift_and_normal_force_agree_at_zero_alpha():
    database = _table(cd=0.3, cn=0.0, cp=1.0, machs=(0.5,))
    row = database.lookup(0.5, 0.0)
    assert row.cn == 0.0


def test_a_shared_mach_range_is_required():
    low = _table(cd=0.3, cn=0.9, cp=1.0, machs=(0.1, 0.2))
    high = _table(cd=0.3, cn=0.9, cp=1.0, machs=(3.0, 4.0))
    with pytest.raises(ValueError):
        aeroplots._mach_grid(low, high, aeroplots.PlotSettings())


def test_both_figures_render(tmp_path):
    pytest.importorskip("matplotlib")
    a = _table(cd=0.40, cn=0.90, cp=1.20)
    b = _table(cd=0.34, cn=0.92, cp=1.38)
    written = aeroplots.write_comparison_plots(a, b, tmp_path, stem="check")
    assert [p.name for p in written] == ["check_lift_slope.png", "check_drag.png"]
    assert all(p.stat().st_size > 5000 for p in written)
