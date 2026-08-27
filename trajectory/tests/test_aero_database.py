"""Tests for aerodynamic coefficient interpolation.

Pins the replacement of the nearest-neighbour lookup, which made every
coefficient a step function of Mach: discontinuous forces that are both wrong
between table points and hostile to an adaptive ODE solver.

Runs under pytest, and standalone via
``python trajectory/tests/test_aero_database.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.vehicle.aero_database import (  # noqa: E402
    AeroCoefficients,
    AeroDatabase,
)


def _grid_db() -> AeroDatabase:
    """2 x 2 grid: Mach 0.5/1.5, alpha 0/10 deg, with a transonic Cd rise."""
    rows = [
        AeroCoefficients(mach=0.5, alpha_deg=0.0, cd=0.30, cn=0.0, cm=0.0, x_cp_m=1.0),
        AeroCoefficients(mach=0.5, alpha_deg=10.0, cd=0.40, cn=2.0, cm=0.0, x_cp_m=1.2),
        AeroCoefficients(mach=1.5, alpha_deg=0.0, cd=0.70, cn=0.0, cm=0.0, x_cp_m=2.0),
        AeroCoefficients(mach=1.5, alpha_deg=10.0, cd=0.90, cn=4.0, cm=0.0, x_cp_m=2.4),
    ]
    return AeroDatabase(rows, reference_length_m=2.0)


def test_grid_is_detected():
    assert _grid_db().is_gridded


def test_tabulated_points_are_returned_exactly():
    db = _grid_db()
    for mach, alpha, cd in [
        (0.5, 0.0, 0.30), (0.5, 10.0, 0.40), (1.5, 0.0, 0.70), (1.5, 10.0, 0.90)
    ]:
        assert np.isclose(db.lookup(mach, alpha).cd, cd), (mach, alpha)


def test_midpoint_is_the_average_not_a_neighbour():
    """The whole point of the change: Mach 1.0 is between 0.30 and 0.70."""
    cd = _grid_db().lookup(1.0, 0.0).cd
    assert np.isclose(cd, 0.50)
    # The old nearest lookup would have returned one endpoint verbatim.
    assert not np.isclose(cd, 0.30)
    assert not np.isclose(cd, 0.70)


def test_bilinear_blends_both_axes():
    # Centre of the cell: mean of all four corners.
    assert np.isclose(_grid_db().lookup(1.0, 5.0).cd, (0.30 + 0.40 + 0.70 + 0.90) / 4)


def test_alpha_sign_is_ignored():
    db = _grid_db()
    assert np.isclose(db.lookup(1.0, -7.0).cn, db.lookup(1.0, 7.0).cn)


def test_out_of_range_is_clamped_not_extrapolated():
    db = _grid_db()
    assert np.isclose(db.lookup(0.0, 0.0).cd, 0.30)     # below the table
    assert np.isclose(db.lookup(9.0, 0.0).cd, 0.70)     # above the table
    assert np.isclose(db.lookup(1.5, 90.0).cd, 0.90)    # beyond max alpha


def test_lookup_is_continuous_in_mach():
    """No jump anywhere across the table -- this is what the solver needs."""
    db = _grid_db()
    machs = np.linspace(0.0, 3.0, 2001)
    cds = np.array([db.lookup(m, 3.0).cd for m in machs])
    steps = np.abs(np.diff(cds))
    # Largest single step must be of the order of the sample spacing, not of
    # the 0.4 range of the table.
    assert steps.max() < 0.01, steps.max()


def test_lookup_is_monotonic_where_the_table_is():
    db = _grid_db()
    cds = [db.lookup(m, 0.0).cd for m in np.linspace(0.5, 1.5, 50)]
    assert np.all(np.diff(cds) >= -1e-12)


def test_single_alpha_table_reduces_to_mach_interpolation():
    """The common RASAero export: a Mach sweep at one alpha."""
    rows = [
        AeroCoefficients(mach=m, alpha_deg=0.0, cd=0.2 + 0.1 * m, cn=0.0, cm=0.0, x_cp_m=1.0)
        for m in [0.0, 1.0, 2.0, 3.0]
    ]
    db = AeroDatabase(rows)
    assert db.is_gridded
    assert np.isclose(db.lookup(1.5, 0.0).cd, 0.35)
    assert np.isclose(db.lookup(1.5, 25.0).cd, 0.35)  # alpha axis is degenerate


def test_duplicate_rows_are_averaged():
    rows = [
        AeroCoefficients(mach=0.0, alpha_deg=0.0, cd=0.20, cn=0.0, cm=0.0, x_cp_m=1.0),
        AeroCoefficients(mach=0.0, alpha_deg=0.0, cd=0.40, cn=0.0, cm=0.0, x_cp_m=1.0),
        AeroCoefficients(mach=1.0, alpha_deg=0.0, cd=0.60, cn=0.0, cm=0.0, x_cp_m=1.0),
    ]
    assert np.isclose(AeroDatabase(rows).lookup(0.0, 0.0).cd, 0.30)


def test_ragged_table_still_interpolates_continuously():
    """A grid with a hole is filled from its neighbours and stays a grid.

    It used to fall to the scattered inverse-distance path, whose weights
    jump when a neighbour drops out of the nearest set.
    """
    rows = [
        AeroCoefficients(mach=0.0, alpha_deg=0.0, cd=0.20, cn=0.0, cm=0.0, x_cp_m=1.0),
        AeroCoefficients(mach=1.0, alpha_deg=0.0, cd=0.60, cn=0.0, cm=0.0, x_cp_m=1.0),
        AeroCoefficients(mach=1.0, alpha_deg=8.0, cd=0.80, cn=3.0, cm=0.0, x_cp_m=1.0),
    ]
    db = AeroDatabase(rows)
    assert db.is_gridded
    cds = np.array([db.lookup(m, 0.0).cd for m in np.linspace(0.0, 1.0, 501)])
    assert np.abs(np.diff(cds)).max() < 0.02
    # Endpoints still reproduce their tabulated values.
    assert np.isclose(db.lookup(0.0, 0.0).cd, 0.20)
    assert np.isclose(db.lookup(1.0, 8.0).cd, 0.80)
    # The hole at (Mach 0, 8 deg) is the mean of its two neighbours.
    assert np.isclose(db.lookup(0.0, 8.0).cd, 0.5 * (0.20 + 0.80))
    assert np.isclose(db.lookup(0.0, 8.0).cn, 0.5 * (0.0 + 3.0))


def test_a_scattered_table_is_not_forced_onto_a_grid():
    """Nearly-unique coordinates would make a grid that is almost all holes."""
    rng = np.random.default_rng(1)
    rows = [
        AeroCoefficients(mach=float(m), alpha_deg=float(a), cd=0.3, cn=0.1 * a, cm=0.0, x_cp_m=1.0)
        for m, a in rng.uniform([0.1, 0.0], [2.0, 10.0], size=(30, 2))
    ]
    db = AeroDatabase(rows)
    assert not db.is_gridded
    assert np.isfinite(db.lookup(1.0, 5.0).cd)


def test_csv_round_trip_preserves_interpolation(tmp_path=None):
    import tempfile

    db = _grid_db()
    with tempfile.TemporaryDirectory() as d:
        path = db.to_csv(Path(d) / "aero.csv")
        reloaded = AeroDatabase.from_csv(path, reference_length_m=2.0)
    assert reloaded.is_gridded
    assert np.isclose(reloaded.lookup(1.0, 5.0).cd, db.lookup(1.0, 5.0).cd)


# --------------------------------------------------------- two drag states


def _two_state_db() -> AeroDatabase:
    rows = [
        AeroCoefficients(0.5, 0.0, 0.30, 0.0, 0.0, 1.0, cd_power_on=0.20),
        AeroCoefficients(1.5, 0.0, 0.70, 0.0, 0.0, 2.0, cd_power_on=0.50),
    ]
    return AeroDatabase(rows, reference_length_m=2.0)


def test_power_on_drag_is_optional_and_interpolates():
    db = _two_state_db()
    assert db.has_power_on
    looked = db.lookup(1.0, 0.0)
    assert np.isclose(looked.cd, 0.50)
    assert np.isclose(looked.cd_power_on, 0.35)

    plain = _grid_db()
    assert not plain.has_power_on
    assert plain.lookup(1.0, 5.0).cd_power_on is None


def test_power_on_column_round_trips_and_a_plain_file_still_loads():
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        both = AeroDatabase.from_csv(_two_state_db().to_csv(Path(d) / "both.csv"))
        plain = AeroDatabase.from_csv(_grid_db().to_csv(Path(d) / "plain.csv"))
        header = (Path(d) / "plain.csv").read_text(encoding="utf-8").splitlines()[0]
    assert both.has_power_on
    assert np.isclose(both.lookup(1.0, 0.0).cd_power_on, 0.35)
    assert not plain.has_power_on
    assert "cd_power_on" not in header, "a single-column table writes the old file"


if __name__ == "__main__":
    failures = 0
    names = sorted(n for n in globals() if n.startswith("test_"))
    for name in names:
        try:
            globals()[name]()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(names) - failures}/{len(names)} passed")
    raise SystemExit(1 if failures else 0)
