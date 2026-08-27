"""The telemetry instrument, proven before it is trusted.

Two kinds of proof. The synthetic round trip manufactures a flight from a
known CD and demands the reconstruction hand it back -- if it cannot, the
algebra is wrong. The committed Qu8k card is then checked against the
flight's own published numbers: integrated apogee within half a percent of
121,478 ft is the accelerometer, the integration and the data file all
agreeing with the people who flew it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aeroengine.atmosphere import Atmosphere
from validation.telemetry import (
    G0_FPS2,
    TelemetryFlight,
    integrate_trace,
    load_flight,
    reconstruct_cd,
)

QU8K = Path(__file__).resolve().parent.parent / "data" / "qu8k"


def _synthetic_flight(cd: float, a_ref_in2: float, weight_lb: float):
    """Fly a point mass at constant CD and log what an RDAS would read."""
    atmos = Atmosphere()
    mass = weight_lb / G0_FPS2
    area = a_ref_in2 / 144.0

    dt = 0.02
    times, readings = [], []
    v, h = 0.0, 0.0
    for i in range(int(60.0 / dt)):
        t = i * dt
        rho = atmos.density(h)
        drag = 0.5 * rho * v * v * cd * area
        if t < 8.0:
            thrust = 12.0 * weight_lb          # a 12 G boost
            kinematic = (thrust - drag) / mass - G0_FPS2
        else:
            kinematic = -drag / mass - G0_FPS2
        times.append(t)
        readings.append((kinematic + G0_FPS2) / G0_FPS2)  # specific force, G
        v += kinematic * dt
        h += v * dt
        if v < 0.0:
            break

    return TelemetryFlight(
        name="synthetic", cdx1="none", burnout_weight_lb=weight_lb,
        site_elevation_ft=0.0, coast_start_s=9.0, coast_end_s=25.0,
        time_s=np.asarray(times), accel_g=np.asarray(readings),
    )


class TestRoundTrip:
    def test_known_cd_is_recovered(self):
        flight = _synthetic_flight(cd=0.40, a_ref_in2=50.0, weight_lb=150.0)
        bins = reconstruct_cd(flight, a_ref_in2=50.0)
        assert bins, "a Mach-3 coast must produce bins"
        assert max(b.mach for b in bins) > 2.0
        for b in bins:
            assert b.cd == pytest.approx(0.40, rel=0.03), (b.mach, b.cd)

    def test_reference_area_scales_out(self):
        flight = _synthetic_flight(cd=0.40, a_ref_in2=50.0, weight_lb=150.0)
        halved = reconstruct_cd(flight, a_ref_in2=25.0)
        assert halved[3].cd == pytest.approx(0.80, rel=0.03)


class TestQu8kCard:
    def test_card_loads(self):
        flight = load_flight(QU8K)
        assert flight.name == "Qu8k"
        assert len(flight.time_s) > 5000
        assert flight.burnout_weight_lb == pytest.approx(154.5)

    def test_integration_reproduces_published_apogee(self):
        flight = load_flight(QU8K)
        _, altitude = integrate_trace(flight.time_s, flight.accel_g)
        assert float(altitude.max()) == pytest.approx(121478.0, rel=0.005)

    def test_peak_mach_matches_the_flight(self):
        flight = load_flight(QU8K)
        velocity, altitude = integrate_trace(flight.time_s, flight.accel_g)
        atmos = Atmosphere()
        mach = [
            v / atmos.speed_of_sound(h + flight.site_elevation_ft)
            for v, h in zip(velocity, altitude)
        ]
        assert max(mach) == pytest.approx(3.2, abs=0.15)

    def test_reconstruction_spans_the_supersonic_coast(self):
        flight = load_flight(QU8K)
        bins = reconstruct_cd(flight, a_ref_in2=55.0)
        machs = [b.mach for b in bins]
        assert min(machs) < 2.1 and max(machs) > 2.8
        for b in bins:
            assert 0.1 < b.cd < 1.0
            assert b.count >= 1

    def test_card_cites_its_sources(self):
        card = json.loads((QU8K / "flight.json").read_text())
        assert card["sources"], "telemetry with no provenance is rumour"
