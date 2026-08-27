"""Winds aloft and Dryden turbulence.

Runs under pytest, and standalone via
``python trajectory/tests/test_wind_aloft.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory import simulation as tm  # noqa: E402
from trajectory.environment.wind import DrydenTurbulence, WindModel  # noqa: E402


def _south(speed: float) -> np.ndarray:
    return np.array([0.0, 0.0, -speed])


def _west(speed: float) -> np.ndarray:
    return np.array([-speed, 0.0, 0.0])


# --------------------------------------------------------------- aloft


def test_without_a_sounding_the_surface_profile_is_what_it_was():
    plain = WindModel(surface_wind=np.array([5.0, 0.0]))
    with_empty = WindModel(surface_wind=np.array([5.0, 0.0]), aloft=[])
    for h in (0.0, 3.0, 10.0, 100.0, 5000.0):
        assert np.allclose(plain.mean_wind(h), with_empty.mean_wind(h))


def test_levels_are_interpolated_as_vectors():
    """Five from the north at 10 m, ten from the east at 1,000 m: halfway
    up the wind is the vector mean, not a speed-and-bearing mean."""
    wind = WindModel(surface_wind=np.array([5.0, 0.0]), surface_dir=0.0,
                     aloft=[(1000.0, 10.0, 90.0)])
    assert np.allclose(wind.mean_wind(10.0), _south(5.0))
    assert np.allclose(wind.mean_wind(1000.0), _west(10.0))
    assert np.allclose(wind.mean_wind(505.0), 0.5 * (_south(5.0) + _west(10.0)))


def test_the_top_level_holds_and_the_surface_layer_rules_below():
    wind = WindModel(surface_wind=np.array([5.0, 0.0]), surface_dir=0.0,
                     aloft=[(500.0, 8.0, 0.0), (2000.0, 20.0, 45.0)])
    assert np.allclose(wind.mean_wind(9000.0), wind.mean_wind(2000.0))
    assert np.linalg.norm(wind.mean_wind(3.0)) == pytest.approx(
        np.linalg.norm(WindModel(surface_wind=np.array([5.0, 0.0])).mean_wind(3.0))
    )


def test_a_calm_surface_under_a_sounding_grows_from_nothing():
    wind = WindModel(aloft=[(1000.0, 10.0, 90.0)])
    assert np.allclose(wind.mean_wind(5.0), 0.0)
    assert np.allclose(wind.mean_wind(505.0), _west(5.0))


def test_levels_in_the_surface_layer_are_ignored():
    wind = WindModel(surface_wind=np.array([5.0, 0.0]), aloft=[(5.0, 40.0, 180.0), (1000.0, 10.0, 0.0)])
    assert wind.aloft == [(1000.0, 10.0, 0.0)]


# --------------------------------------------------------------- turbulence


def _flat(sigma: float, length: float) -> DrydenTurbulence:
    """A field with one length and one intensity everywhere, for statistics."""
    turbulence = DrydenTurbulence(sigma, seed=7, top_m=60_000.0)
    turbulence.scales = lambda h, w20: (length, length, length, sigma, sigma, sigma)
    return turbulence


def test_the_field_has_the_stated_intensity_and_no_mean():
    turbulence = _flat(3.0, 10.0)
    heights = np.arange(0.0, 60_000.0, 2.0)
    samples = np.array([turbulence.gust(h, 0.0, np.array([0.0, 0.0, -1.0])) for h in heights])
    for k in range(3):
        assert np.std(samples[:, k]) == pytest.approx(3.0, rel=0.05)
        assert abs(np.mean(samples[:, k])) < 0.15


def test_the_correlation_length_is_the_dryden_scale():
    """Autocorrelation ``exp(-lag / L)``: 1/e at one scale length."""
    length = 100.0
    turbulence = _flat(1.0, length)
    heights = np.arange(0.0, 60_000.0, 2.0)
    u = np.array([turbulence.gust(h, 0.0, np.array([0.0, 0.0, -1.0]))[2] for h in heights])
    u = u - u.mean()
    lag = int(length / 2.0)
    rho = float(np.dot(u[:-lag], u[lag:]) / np.dot(u, u))
    assert rho == pytest.approx(np.exp(-1.0), abs=0.06)


def test_the_field_is_the_seeds_and_continuous():
    a = DrydenTurbulence(2.0, seed=1)
    b = DrydenTurbulence(2.0, seed=1)
    c = DrydenTurbulence(2.0, seed=2)
    along = np.array([1.0, 0.0, 0.0])
    assert np.allclose(a.gust(3000.0, 5.0, along), b.gust(3000.0, 5.0, along))
    assert not np.allclose(a.gust(3000.0, 5.0, along), c.gust(3000.0, 5.0, along))
    assert np.allclose(a.gust(3000.0, 5.0, along), a.gust(3000.0 + 1e-6, 5.0, along), atol=1e-4)
    a.reseed(2)
    assert np.allclose(a.gust(3000.0, 5.0, along), c.gust(3000.0, 5.0, along))


def test_the_surface_layer_follows_the_wind_at_twenty_feet():
    """``sigma_w = 0.1 W20`` low down; the level's own intensity aloft."""
    turbulence = DrydenTurbulence.from_level("moderate", seed=3)
    _, _, _, _, _, s_w = turbulence.scales(100.0, wind_20ft_mps=20.0)
    assert s_w == pytest.approx(2.0)
    _, _, _, _, _, s_w_calm = turbulence.scales(100.0, wind_20ft_mps=0.0)
    assert s_w_calm == pytest.approx(0.1 * 15.4), "floored at the level's reference wind"
    l_u, l_v, l_w, s_u, s_v, s_w = turbulence.scales(5000.0, wind_20ft_mps=20.0)
    assert (l_u, l_v, l_w) == (533.4, 533.4, 533.4)
    assert (s_u, s_v, s_w) == (3.0, 3.0, 3.0)
    assert DrydenTurbulence.from_level("none") is None


def test_the_gust_lies_along_across_and_up():
    turbulence = _flat(1.0, 50.0)
    gust = turbulence.gust(100.0, 0.0, np.array([1.0, 0.0, 0.0]))
    unit = turbulence._unit[int(100.0 / turbulence.STEP_M)]
    assert np.allclose(gust, [unit[0], unit[2], unit[1]])


def test_a_flight_through_turbulence_lands_somewhere_else_each_seed():
    landings = []
    for seed in (1, 2):
        sim = tm.RocketSimulation()
        sim.wind = WindModel(surface_wind=np.array([3.0, 0.0]),
                             turbulence=DrydenTurbulence.from_level("severe", seed=seed))
        result = sim.run(launch_elevation=np.radians(85.0), dt=0.5)
        assert result.landed and np.all(np.isfinite(result.y))
        landings.append(result.y[[0, 2], -1])
    assert np.linalg.norm(landings[0] - landings[1]) > 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
