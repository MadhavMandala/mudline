"""Tests for trajectory export.

Pins the gap where a simulation's only output was six numbers in a modal
dialog: there was no way to get a time history out, plot it, or hand the data
to anyone else.

Runs under pytest, and standalone via
``python trajectory/tests/test_export.py``.
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.analysis.export import (  # noqa: E402
    CSV_COLUMNS,
    derived_quantities,
    export_all,
    max_q,
    plot_dispersion,
    plot_trajectory,
    write_dispersion_csv,
    write_trajectory_csv,
)


class _FakeResult:
    """A minimal stand-in for a solve_ivp result, so these tests stay fast."""

    def __init__(self, n=120):
        self.t = np.linspace(0.0, 60.0, n)
        states = np.zeros((n, 14))
        states[:, 0] = 10.0 * self.t                       # East
        states[:, 1] = 800.0 * self.t - 8.0 * self.t ** 2  # altitude
        states[:, 2] = 4.0 * self.t                        # North
        states[:, 3] = 10.0
        states[:, 4] = 800.0 - 16.0 * self.t
        states[:, 5] = 4.0
        states[:, 6] = 1.0
        states[:, 13] = np.linspace(150.0, 0.0, n)
        self.y = states.T
        self.success = True
        self.phases = [{"name": "ascent", "t_start": 0.0, "t_end": 60.0}]


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


# ------------------------------------------------------- derived quantities


def test_derived_speed_matches_the_velocity_norm():
    result = _FakeResult()
    derived = derived_quantities(result.y.T)
    expected = np.linalg.norm(result.y.T[:, 3:6], axis=1)
    assert np.allclose(derived["speed_mps"], expected)


def test_derived_mach_uses_the_local_speed_of_sound():
    from trajectory.environment.atmosphere import Atmosphere
    result = _FakeResult()
    derived = derived_quantities(result.y.T)
    states = result.y.T
    a = Atmosphere().get_conditions(float(states[10, 1]))[3]
    assert np.isclose(derived["mach"][10], derived["speed_mps"][10] / a)


def test_dynamic_pressure_is_half_rho_v_squared():
    result = _FakeResult()
    d = derived_quantities(result.y.T)
    assert np.allclose(d["dynamic_pressure_pa"],
                       0.5 * d["density_kgm3"] * d["speed_mps"] ** 2)


def test_mass_column_adds_dry_mass_to_propellant():
    d = derived_quantities(_FakeResult().y.T, dry_mass_kg=50.0)
    assert np.allclose(d["mass_kg"], d["propellant_kg"] + 50.0)


def test_max_q_finds_the_peak():
    result = _FakeResult()
    d = derived_quantities(result.y.T)
    peak = max_q(result)
    assert np.isclose(peak["pressure_pa"], d["dynamic_pressure_pa"].max())
    assert 0.0 <= peak["time_s"] <= 60.0


# --------------------------------------------------------------- CSV output


def test_csv_has_a_row_per_sample():
    result = _FakeResult()
    path = write_trajectory_csv(result, _tmp() / "t.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == CSV_COLUMNS
    assert len(rows) == len(result.t) + 1


def test_csv_values_round_trip():
    result = _FakeResult()
    path = write_trajectory_csv(result, _tmp() / "t.csv", dry_mass_kg=50.0)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert np.isclose(float(rows[5]["time_s"]), result.t[5])
    assert np.isclose(float(rows[5]["altitude_m"]), result.y[1, 5], rtol=1e-5)
    assert np.isclose(float(rows[5]["mass_kg"]), result.y[13, 5] + 50.0, rtol=1e-5)


def test_csv_creates_missing_directories():
    target = _tmp() / "nested" / "deeper" / "t.csv"
    assert write_trajectory_csv(_FakeResult(), target).exists()


# -------------------------------------------------------------- plot output


def test_plot_is_written_and_nonempty():
    path = plot_trajectory(_FakeResult(), _tmp() / "t.png")
    assert path.exists() and path.stat().st_size > 5000


def test_export_all_writes_both_artifacts():
    written = export_all(_FakeResult(), _tmp(), dry_mass_kg=50.0)
    assert written["csv"].exists()
    assert written["plot"].exists()


def test_plot_handles_a_recovery_flight():
    """Phase markers must not break plotting."""
    result = _FakeResult()
    result.phases = [
        {"name": "ascent", "t_start": 0.0, "t_end": 30.0},
        {"name": "drogue", "t_start": 30.0, "t_end": 50.0, "cda_m2": 1.3},
        {"name": "main", "t_start": 50.0, "t_end": 60.0, "cda_m2": 22.0},
    ]
    assert plot_trajectory(result, _tmp() / "r.png").exists()


# ---------------------------------------------------------------- dispersion


class _FakeDispersion:
    def __init__(self, n=25):
        rng = np.random.default_rng(3)
        self.landing_points = rng.normal([1000.0, 5000.0], [300.0, 700.0], size=(n, 2))
        self.cep_m = 500.0
        self.ellipse = (
            self.landing_points.mean(axis=0), 1400.0, 600.0, 0.3
        )
        self.cases = [
            {
                "params": {"impulse_scale": 1.0 + 0.01 * i, "dry_mass_kg": 50.0},
                "max_altitude": 100000.0 + 100 * i,
                "landing_east_m": float(self.landing_points[i, 0]),
                "landing_north_m": float(self.landing_points[i, 1]),
                "success": True,
            }
            for i in range(n)
        ]


def test_dispersion_csv_has_a_row_per_case():
    dispersion = _FakeDispersion()
    path = write_dispersion_csv(dispersion, _tmp() / "d.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(dispersion.cases)
    assert "impulse_scale" in rows[0]
    assert "max_altitude" in rows[0]


def test_dispersion_csv_records_sampled_inputs():
    path = write_dispersion_csv(_FakeDispersion(), _tmp() / "d.csv")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert np.isclose(float(rows[3]["impulse_scale"]), 1.03)


def test_dispersion_plot_is_written():
    path = plot_dispersion(_FakeDispersion(), _tmp() / "d.png")
    assert path.exists() and path.stat().st_size > 5000


def test_empty_dispersion_is_rejected():
    class Empty:
        cases = []
    try:
        write_dispersion_csv(Empty(), _tmp() / "d.csv")
    except ValueError:
        return
    raise AssertionError("empty dispersion should not silently write a file")


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
