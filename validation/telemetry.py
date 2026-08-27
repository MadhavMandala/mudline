"""Reconstruct drag from flight telemetry, and hold the engine to it.

The scoreboard's truth is apogee: one scalar per flight, the *integral* of
drag, in which a wave-drag error and a base-drag error of opposite sign can
cancel without a trace. An accelerometer log has no such hiding place. In
coast there is no thrust, and an accelerometer does not feel gravity -- it
reads specific force -- so the axial channel *is* the drag, sampled every
few milliseconds through the whole supersonic deceleration::

    D = -m * f_axial          CD = D / (q(h) * A_ref)

One flight logged this way pins CD as a *function of Mach*, which is what
settles questions apogee scalars provably cannot -- the boattail experiment
found two flights whose apogees demand contradictory base-drag laws, and
attribution is exactly what this module exists to restore.

Assumptions, stated rather than buried: the flight is treated as vertical
and the accelerometer as axis-aligned, so velocity and altitude come from
integrating ``(raw - 1) * g0`` -- the same convention the flight's own
published reduction uses, and verified here against its published apogee.
Quantization (0.1 G on the reference flight) is beaten down by binning in
Mach; each bin reports its scatter so nobody mistakes the floor for signal.

Flight cards live in ``validation/data/<flight>/`` as ``flight.json`` plus
``trace.csv`` (time_s, acc_raw_G, ...). That directory is the format new
telemetry flights arrive in: add a card, cite the source, and the flight
joins the comparison.

Usage::

    python -m validation.telemetry [validation/data/qu8k] [--examples DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aeroengine.atmosphere import Atmosphere

G0_FPS2 = 32.174

__all__ = [
    "TelemetryFlight", "MachBin", "load_flight", "integrate_trace",
    "reconstruct_cd", "compare_against_engine",
]


# ---------------------------------------------------------------------------
# Flight cards
# ---------------------------------------------------------------------------


@dataclass
class TelemetryFlight:
    """One flight's telemetry and the facts needed to reduce it."""

    name: str
    cdx1: str
    burnout_weight_lb: float
    site_elevation_ft: float
    coast_start_s: float
    coast_end_s: float
    time_s: np.ndarray
    accel_g: np.ndarray
    measured_apogee_ft: float | None = None
    directory: Path | None = None


def load_flight(directory: str | Path) -> TelemetryFlight:
    """Read ``flight.json`` and its trace from a flight-card directory."""
    directory = Path(directory)
    card = json.loads((directory / "flight.json").read_text())

    trace_file = directory / card["trace"]["file"]
    times: list[float] = []
    accel: list[float] = []
    with trace_file.open() as f:
        for row in csv.DictReader(f):
            times.append(float(row["time_s"]))
            accel.append(float(row["acc_raw_G"]))

    return TelemetryFlight(
        name=card["name"],
        cdx1=card["cdx1"],
        burnout_weight_lb=float(card["burnout_weight_lb"]),
        site_elevation_ft=float(card["site_elevation_ft"]),
        coast_start_s=float(card["coast_start_s"]),
        coast_end_s=float(card["coast_end_s"]),
        time_s=np.asarray(times),
        accel_g=np.asarray(accel),
        measured_apogee_ft=card.get("measured_apogee_ft"),
        directory=directory,
    )


# ---------------------------------------------------------------------------
# Kinematics
# ---------------------------------------------------------------------------


def integrate_trace(time_s: np.ndarray, accel_g: np.ndarray):
    """Velocity (ft/s) and altitude (ft AGL) from the axial accelerometer.

    ``(raw - 1) * g0`` is the vertical kinematic acceleration under the
    vertical-flight assumption: at rest the accelerometer reads +1 G and the
    vehicle is not accelerating; in free coast it reads the drag alone while
    gravity, which it cannot feel, still acts. Trapezoid throughout, clocked
    from the first sample at or after t = 0.
    """
    time_s = np.asarray(time_s, dtype=float)
    accel_g = np.asarray(accel_g, dtype=float)
    kin = (accel_g - 1.0) * G0_FPS2
    kin = np.where(time_s < 0.0, 0.0, kin)

    velocity = np.concatenate([
        [0.0], np.cumsum(0.5 * (kin[1:] + kin[:-1]) * np.diff(time_s)),
    ])
    altitude = np.concatenate([
        [0.0],
        np.cumsum(0.5 * (velocity[1:] + velocity[:-1]) * np.diff(time_s)),
    ])
    return velocity, altitude


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


@dataclass
class MachBin:
    """Measured drag in one Mach bin: the mean, and how much it wobbles."""

    mach: float
    cd: float
    scatter: float
    count: int
    altitude_ft: float  # mean MSL altitude the bin was flown at
    samples: list[float] = field(default_factory=list, repr=False)


def reconstruct_cd(
    flight: TelemetryFlight,
    a_ref_in2: float,
    atmosphere: Atmosphere | None = None,
    bin_width: float = 0.05,
    min_quanta_g: float = 0.15,
) -> list[MachBin]:
    """CD(Mach) from the coast window, binned against quantization.

    ``min_quanta_g`` drops samples whose drag reading is within noise of the
    accelerometer's step size; below that the "measurement" is the
    quantizer, not the rocket.
    """
    atmosphere = atmosphere or Atmosphere()
    velocity, altitude = integrate_trace(flight.time_s, flight.accel_g)

    mass_slug = flight.burnout_weight_lb / G0_FPS2
    area_ft2 = a_ref_in2 / 144.0

    bins: dict[int, MachBin] = {}
    alt_sums: dict[int, float] = {}
    for t, raw, v, h in zip(flight.time_s, flight.accel_g, velocity, altitude):
        if not (flight.coast_start_s <= t <= flight.coast_end_s):
            continue
        if raw > -min_quanta_g or v <= 0.0:
            continue
        msl = h + flight.site_elevation_ft
        rho = atmosphere.density(msl)
        sound = atmosphere.speed_of_sound(msl)
        if rho <= 0.0 or sound <= 0.0:
            continue
        mach = v / sound
        q = 0.5 * rho * v * v
        cd = (-raw * G0_FPS2) * mass_slug / (q * area_ft2)

        key = int(mach / bin_width)
        entry = bins.get(key)
        if entry is None:
            entry = MachBin((key + 0.5) * bin_width, 0.0, 0.0, 0, 0.0)
            bins[key] = entry
            alt_sums[key] = 0.0
        entry.samples.append(cd)
        alt_sums[key] += msl

    out = []
    for key in sorted(bins):
        entry = bins[key]
        arr = np.asarray(entry.samples)
        entry.cd = float(arr.mean())
        entry.scatter = float(arr.std())
        entry.count = len(arr)
        entry.altitude_ft = alt_sums[key] / len(arr)
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def compare_against_engine(
    flight: TelemetryFlight,
    examples_dir: str | Path,
    models: tuple[str, ...] = ("rasaero", "corrected"),
) -> tuple[list[MachBin], dict[str, list[float]]]:
    """Measured CD(Mach) next to each engine model's power-off prediction.

    The engine is evaluated *along the flight*: its Mach/Alt grid is set
    from the reconstruction's own bins, so Reynolds number is taken at the
    altitude each Mach was actually flown, not at sea level.
    """
    from aeroengine.cdx1 import load as load_cdx1
    from aeroengine.solver import Engine

    design = load_cdx1(Path(examples_dir) / flight.cdx1)
    probe = Engine(design)
    bins = reconstruct_cd(flight, probe.cache.a_ref)
    design.mach_alt = [(b.mach, b.altitude_ft) for b in bins]

    predictions: dict[str, list[float]] = {}
    for model in models:
        engine = Engine(design, boattail_model=model)
        predictions[model] = [
            engine.solve(b.mach, 0.0).cd_off for b in bins
        ]
    return bins, predictions


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Telemetry drag reconstruction")
    ap.add_argument("flight", nargs="?",
                    default=Path(__file__).parent / "data" / "qu8k",
                    type=Path)
    ap.add_argument("--examples", type=Path,
                    default=Path.home() / "Documents" / "RASAero II" / "Examples")
    args = ap.parse_args(argv)

    flight = load_flight(args.flight)
    velocity, altitude = integrate_trace(flight.time_s, flight.accel_g)
    apogee = float(altitude.max())
    print(f"{flight.name}: integrated apogee {apogee:,.0f} ft AGL", end="")
    if flight.measured_apogee_ft:
        drift = 100.0 * (apogee - flight.measured_apogee_ft) / flight.measured_apogee_ft
        print(f"  (published {flight.measured_apogee_ft:,.0f}, {drift:+.2f}%)")
    else:
        print()

    bins, predictions = compare_against_engine(flight, args.examples)
    models = list(predictions)
    head = f"{'Mach':>6} {'alt kft':>8} {'n':>5} {'CD flight':>10} {'+/-':>6}"
    for model in models:
        head += f" {model:>10} {'d%':>7}"
    print(head)
    for i, b in enumerate(bins):
        line = (f"{b.mach:6.2f} {b.altitude_ft / 1000.0:8.1f} {b.count:5d} "
                f"{b.cd:10.3f} {b.scatter:6.3f}")
        for model in models:
            cd = predictions[model][i]
            line += f" {cd:10.3f} {100.0 * (cd - b.cd) / b.cd:+7.1f}"
        print(line)

    for model in models:
        err = [100.0 * (predictions[model][i] - b.cd) / b.cd
               for i, b in enumerate(bins)]
        arr = np.asarray(err)
        print(f"{model:>12}: mean {arr.mean():+6.1f}%   "
              f"mean|.| {np.abs(arr).mean():5.1f}%   "
              f"range {arr.min():+.1f}% .. {arr.max():+.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
