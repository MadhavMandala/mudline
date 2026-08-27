"""Tests for the US Standard Atmosphere 1976 implementation.

Checked against published USSA76 table values rather than against another
implementation, so a shared misreading of the standard cannot make these pass.

Pins the defect where the model returned rho = 0 above 25 km, which silently
removed all aerodynamics from the majority of a high-altitude flight.

Runs under pytest, and standalone via
``python trajectory/tests/test_atmosphere.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.environment.atmosphere import Atmosphere  # noqa: E402


ATM = Atmosphere()

# The standard's defining layer parameters: (geopotential base height [m],
# base temperature [K], base pressure [Pa]). These are exact by definition, not
# transcribed from a printed table, so they are the right thing to test
# against. Note they are GEOPOTENTIAL heights -- get_conditions takes geometric
# altitude, so the tests convert.
LAYER_BASES = [
    (0.0,     288.15, 101325.0),
    (11000.0, 216.65,  22632.06),
    (20000.0, 216.65,   5474.889),
    (32000.0, 228.65,    868.0187),
    (47000.0, 270.65,    110.9063),
    (51000.0, 270.65,     66.93887),
    (71000.0, 214.65,      3.956420),
    (84852.0, 186.87,      0.3733836),
]


def test_sea_level_matches_constants():
    rho, p, T, a = ATM.get_conditions(0.0)
    assert np.isclose(p, 101325.0)
    assert np.isclose(T, 288.15)
    assert np.isclose(rho, 1.225, rtol=1e-3)
    assert np.isclose(a, 340.29, rtol=1e-3)


def test_reproduces_every_layer_base_exactly():
    """Integrating up from sea level must land on each defined base state."""
    for h_b, t_ref, p_ref in LAYER_BASES:
        z = ATM.geometric_altitude(h_b)
        rho, p, T, _ = ATM.get_conditions(z)
        assert np.isclose(T, t_ref, rtol=1e-4), f"T at H={h_b} m: {T} vs {t_ref}"
        assert np.isclose(p, p_ref, rtol=1e-3), f"p at H={h_b} m: {p} vs {p_ref}"
        assert np.isclose(rho, p_ref / (ATM.R * t_ref), rtol=1e-3)


def test_isothermal_layer_follows_the_barometric_law():
    """11-20 km has zero lapse rate, so pressure must decay exponentially."""
    h1, h2 = 12000.0, 18000.0
    p1 = ATM.get_conditions(ATM.geometric_altitude(h1))[1]
    p2 = ATM.get_conditions(ATM.geometric_altitude(h2))[1]
    expected_ratio = np.exp(-ATM.G0 * (h2 - h1) / (ATM.R * 216.65))
    assert np.isclose(p2 / p1, expected_ratio, rtol=1e-6)


def test_gradient_layer_follows_the_power_law():
    """0-11 km has a -6.5 K/km lapse rate; check the polytropic exponent."""
    h = 8000.0
    t_expected = 288.15 - 0.0065 * h
    p_expected = 101325.0 * (288.15 / t_expected) ** (ATM.G0 / (ATM.R * -0.0065))
    rho, p, T, _ = ATM.get_conditions(ATM.geometric_altitude(h))
    assert np.isclose(T, t_expected, rtol=1e-9)
    assert np.isclose(p, p_expected, rtol=1e-9)


def test_density_is_nonzero_where_the_old_model_returned_zero():
    """The specific regression: 25-86 km used to be a hard vacuum."""
    for z in [26000, 40000, 60000, 80000, 85000]:
        rho, p, _, _ = ATM.get_conditions(z)
        assert rho > 0.0, z
        assert p > 0.0, z


def test_upper_atmosphere_is_thin_but_present():
    for z, rho_ref in [(100e3, 5.604e-7), (150e3, 2.076e-9), (200e3, 2.541e-10)]:
        rho, _, _, _ = ATM.get_conditions(z)
        assert np.isclose(rho, rho_ref, rtol=1e-6), f"rho at {z}: {rho}"


def test_density_decreases_monotonically():
    z = np.linspace(0.0, 299000.0, 4000)
    rho = np.array([ATM.get_conditions(float(v))[0] for v in z])
    assert np.all(np.diff(rho) <= 1e-18), "density must not increase with altitude"


def test_pressure_is_continuous_across_every_layer_boundary():
    """No jumps at 11, 20, 32, 47, 51, 71 or 86 km."""
    for boundary in [11e3, 20e3, 32e3, 47e3, 51e3, 71e3, 86e3]:
        below = ATM.get_conditions(boundary - 1.0)[1]
        above = ATM.get_conditions(boundary + 1.0)[1]
        assert np.isclose(below, above, rtol=2e-3), f"pressure jump at {boundary} m"


def test_temperature_is_continuous_across_every_layer_boundary():
    for boundary in [11e3, 20e3, 32e3, 47e3, 51e3, 71e3, 86e3]:
        below = ATM.get_conditions(boundary - 1.0)[2]
        above = ATM.get_conditions(boundary + 1.0)[2]
        assert np.isclose(below, above, rtol=2e-3), f"temperature jump at {boundary} m"


def test_geopotential_conversion_round_trips():
    for z in [0.0, 1000.0, 30000.0, 86000.0]:
        h = ATM.geopotential_altitude(z)
        assert np.isclose(ATM.geometric_altitude(h), z, rtol=1e-9)


def test_geopotential_is_below_geometric():
    """The correction the old model skipped: H < Z, by ~1.4% at 86 km."""
    assert ATM.geopotential_altitude(86000.0) < 86000.0
    assert np.isclose(ATM.geopotential_altitude(86000.0), 84852.0, rtol=1e-3)


def test_above_the_model_ceiling_density_is_zero():
    rho, p, _, _ = ATM.get_conditions(400e3)
    assert rho == 0.0 and p == 0.0


def test_negative_altitude_is_treated_as_sea_level():
    assert ATM.get_conditions(-500.0) == ATM.get_conditions(0.0)


def test_speed_of_sound_tracks_temperature():
    for z in [0.0, 20000.0, 50000.0]:
        _, _, T, a = ATM.get_conditions(z)
        assert np.isclose(a, np.sqrt(1.4 * 287.05 * T))


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
