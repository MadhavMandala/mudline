"""Coupling the drag table to the trajectory that will fly it.

A table is built at sea level unless told otherwise. These tests pin the
two halves of the fix: the profile extractor reads (Mach, altitude) off a
flown trajectory in the units the engine's grid speaks, and a table built
along such a profile actually differs from the sea-level one in the
direction physics demands -- thinner air means lower Reynolds number means
more skin friction.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from parametric import aero, analysis
from parametric.standard import basic_rocket

FT_PER_M = 1.0 / 0.3048


def _fake_result(altitudes_m, speeds_mps):
    """A sim result with just enough state for the profile extractor."""
    n = len(altitudes_m)
    y = np.zeros((14, n))
    y[1, :] = altitudes_m
    y[4, :] = speeds_mps  # vertical velocity; speed is the norm of columns 3:6
    return SimpleNamespace(y=y, t=np.linspace(0.0, 60.0, n))


class TestMachAltProfile:
    def test_samples_ascent_in_feet_sorted_by_mach(self):
        alts = np.linspace(0.0, 12000.0, 400)
        speeds = np.concatenate([
            np.linspace(0.0, 600.0, 200),    # burn: accelerating
            np.linspace(600.0, 150.0, 200),  # coast: decelerating
        ])
        profile = analysis.mach_alt_profile(_fake_result(alts, speeds))

        assert profile, "an ascent this size must produce samples"
        machs = [m for m, _ in profile]
        assert machs == sorted(machs)
        assert max(machs) > 1.5

        # Every Mach the burn flew, the coast flies again in thinner air.
        # The burn carries the dynamic pressure, so the burn owns them all:
        # nothing is claimed above burnout, and Mach 1 sits where the burn
        # crossed it, a third of the way up, not where the coast did.
        burnout_ft = alts[199] * FT_PER_M
        assert max(h for _, h in profile) <= burnout_ft * 1.03
        at_mach_1 = dict(profile)[1.0]
        assert 2000.0 * FT_PER_M < at_mach_1 < 5000.0 * FT_PER_M
        # The top rung is the burn/coast junction, and it is in feet.
        assert profile[-1][1] == pytest.approx(burnout_ft, rel=0.03)

    def test_a_fast_burn_claims_every_rung(self):
        """A vehicle that climbs more than a rung's worth of Mach between
        two states must still claim every rung it passed. Sampling the
        states instead of interpolating the crossings let half the rungs
        fall through to the coast, 250,000 ft up, and which half changed
        with every pass of the coupling loop."""
        alts = np.concatenate([
            np.linspace(0.0, 6000.0, 30),        # 200 m and 0.06 Mach a step
            np.linspace(6200.0, 30000.0, 120),
        ])
        speeds = np.concatenate([
            np.linspace(0.0, 600.0, 30),
            np.linspace(590.0, 100.0, 120),
        ])
        profile = analysis.mach_alt_profile(_fake_result(alts, speeds))

        rungs = [round(m / 0.02) for m, _ in profile]
        assert rungs == list(range(rungs[0], rungs[-1] + 1)), "a rung fell through"
        altitudes = [h for _, h in profile]
        assert altitudes == sorted(altitudes), "a coast claim broke the burn's ladder"
        assert max(altitudes) <= 6000.0 * FT_PER_M * 1.03

    def test_descent_is_not_sampled(self):
        up = np.linspace(0.0, 5000.0, 100)
        down = np.linspace(5000.0, 0.0, 100)
        speeds = np.full(200, 200.0)
        profile = analysis.mach_alt_profile(
            _fake_result(np.concatenate([up, down]), speeds)
        )
        # One speed, flown at every altitude on the way up and again on the
        # way down: only the ascent's altitudes may appear, each once.
        assert profile
        assert len(profile) == len({h for _, h in profile})
        assert max(h for _, h in profile) <= 5000.0 * FT_PER_M

    def test_pad_altitude_shifts_to_msl(self):
        """Rungs are located on the flown trajectory; the pad lifts them."""
        from trajectory.environment.atmosphere import Atmosphere

        atmosphere = Atmosphere()
        pad = 1400.0
        alts = np.linspace(0.0, 3000.0, 50)
        ramp = np.linspace(0.5, 1.5, 50)
        # Speeds chosen so both flights hold the same Mach at the same
        # height above ground; only the air they do it in differs.
        sea_speeds = [m * atmosphere.get_conditions(h)[3]
                      for m, h in zip(ramp, alts)]
        high_speeds = [m * atmosphere.get_conditions(h + pad)[3]
                       for m, h in zip(ramp, alts)]
        sea = analysis.mach_alt_profile(_fake_result(alts, sea_speeds))
        high = analysis.mach_alt_profile(
            _fake_result(alts, high_speeds), pad_altitude_m=pad
        )
        assert [m for m, _ in sea] == [m for m, _ in high]
        for (_, h_sea), (_, h_high) in zip(sea, high):
            assert h_high == pytest.approx(h_sea + pad * FT_PER_M, abs=1e-6)

    def test_empty_and_stationary_results_yield_nothing(self):
        assert analysis.mach_alt_profile(
            SimpleNamespace(y=np.zeros((14, 1)), t=np.zeros(1))
        ) == []
        assert analysis.mach_alt_profile(
            _fake_result(np.zeros(50), np.zeros(50))
        ) == []


class TestCoupledTable:
    def test_altitude_grid_raises_supersonic_drag(self):
        """Thinner air, lower Reynolds, more friction: CD must go up."""
        model = basic_rocket()
        settings = aero.AeroSettings(
            mach_min=0.3, mach_max=3.0, mach_points=8,
            alpha_max_deg=4.0, alpha_points=2,
        )
        sea, _ = aero.run_analysis(model, settings)
        coupled, _ = aero.run_analysis(
            model, settings, mach_alt=[(0.0, 60000.0), (25.0, 60000.0)]
        )
        cd_sea = sea.lookup(2.0, 0.0).cd
        cd_alt = coupled.lookup(2.0, 0.0).cd
        assert cd_alt > cd_sea * 1.01, (cd_sea, cd_alt)

    def test_empty_profile_changes_nothing(self):
        model = basic_rocket()
        settings = aero.AeroSettings(
            mach_min=0.3, mach_max=2.0, mach_points=4,
            alpha_max_deg=4.0, alpha_points=2,
        )
        plain, _ = aero.run_analysis(model, settings)
        explicit, _ = aero.run_analysis(model, settings, mach_alt=[])
        assert plain.lookup(1.5, 0.0).cd == explicit.lookup(1.5, 0.0).cd
