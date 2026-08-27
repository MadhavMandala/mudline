"""Compare a flight's rotational degrees of freedom against an IMU log.

What the tool can check on its own is that its attitude dynamics agree
with the equations they claim to integrate -- see ``validation.rotational``
and its tests. What it cannot check without a flight is that those
equations describe a real vehicle: the frequency the airframe actually
weathercocks at, how fast the oscillation actually dies, how fast it
actually rolls. A flight computer's gyros and accelerometers are the
measurement of exactly those things, and this module lines one up
against the simulation of the same flight.

The log is a CSV with a time column, three gyro columns and three
accelerometer columns. Column names, units and the IMU's axes are all
declared by the caller, since no two flight computers agree: the
``axes`` matrix maps the IMU's (x, y, z) onto the simulator's body frame,
+Y forward, so an IMU whose x points out the nose passes the matrix that
takes its x to y. Liftoff is found in both records from the axial
acceleration, and the two are aligned there; the simulation is then
interpolated onto the measured times and the difference reported per
channel, over the boost and over the whole record.

    python -m validation.imu flight.csv --vehicle vehicles/basic.json \
        --elevation 85 --rail 3 --gyro-units dps --accel-units g

flies the vehicle through the same path Run Flight uses and prints the
comparison. From a script, ``simulated_imu`` and ``compare`` do the same.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from trajectory.environment.gravity import G0

__all__ = ["ImuLog", "ImuComparison", "load_imu_csv", "simulated_imu", "compare"]


@dataclass
class ImuLog:
    """Body rates and specific force against time, in the simulator's body frame.

    ``gyro_radps`` is (n, 3) about body x, y, z; ``accel_mps2`` is (n, 3)
    specific force -- what an accelerometer reads, gravity included as
    the reaction it feels -- along body x, y, z, +Y forward.
    """

    time_s: np.ndarray
    gyro_radps: np.ndarray
    accel_mps2: np.ndarray
    source: str = ""

    def __len__(self) -> int:
        return len(self.time_s)

    @property
    def axial_g(self) -> np.ndarray:
        return self.accel_mps2[:, 1] / G0

    @property
    def roll_rate_radps(self) -> np.ndarray:
        return self.gyro_radps[:, 1]

    @property
    def transverse_rate_radps(self) -> np.ndarray:
        return np.hypot(self.gyro_radps[:, 0], self.gyro_radps[:, 2])

    def liftoff_index(self, threshold_g: float = 1.5) -> int:
        """First sample where the axial specific force exceeds the threshold."""
        above = np.flatnonzero(self.axial_g > threshold_g)
        return int(above[0]) if len(above) else 0

    def liftoff_time_s(self, threshold_g: float = 1.5) -> float:
        return float(self.time_s[self.liftoff_index(threshold_g)])

    def shifted(self, offset_s: float) -> "ImuLog":
        return ImuLog(self.time_s - offset_s, self.gyro_radps, self.accel_mps2, self.source)


def load_imu_csv(
    path: str | Path,
    time: str = "time_s",
    gyro: tuple[str, str, str] = ("gx", "gy", "gz"),
    accel: tuple[str, str, str] = ("ax", "ay", "az"),
    gyro_units: str = "rad/s",
    accel_units: str = "m/s2",
    axes: np.ndarray | None = None,
) -> ImuLog:
    """Read an IMU log.

    Args:
        gyro_units: ``"rad/s"`` or ``"dps"``.
        accel_units: ``"m/s2"`` or ``"g"``.
        axes: 3x3 matrix taking the IMU's (x, y, z) to the simulator's
            body (x, y, z), +Y forward. ``None`` means they already agree.
    """
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row:
                rows.append(row)
    if not rows:
        raise ValueError(f"{path} has no rows")
    for name in (time, *gyro, *accel):
        if name not in rows[0]:
            raise ValueError(f"{path} has no column {name!r}; it has {list(rows[0])}")
    t = np.array([float(r[time]) for r in rows])
    g = np.array([[float(r[c]) for c in gyro] for r in rows])
    a = np.array([[float(r[c]) for c in accel] for r in rows])
    if gyro_units == "dps":
        g = np.radians(g)
    elif gyro_units != "rad/s":
        raise ValueError(f"gyro_units must be 'rad/s' or 'dps', not {gyro_units!r}")
    if accel_units == "g":
        a = a * G0
    elif accel_units != "m/s2":
        raise ValueError(f"accel_units must be 'm/s2' or 'g', not {accel_units!r}")
    if axes is not None:
        m = np.asarray(axes, dtype=float)
        g = g @ m.T
        a = a @ m.T
    order = np.argsort(t, kind="stable")
    return ImuLog(t[order], g[order], a[order], source=str(path))


def simulated_imu(sim, result) -> ImuLog:
    """What an ideal IMU on the simulated vehicle would have read.

    Rates are the state's; the specific force is the total force less
    gravity, per unit mass, in body axes -- the same felt acceleration
    the flight log reports, as a vector.
    """
    from trajectory.analysis.flightlog import _arm_chute, _phase_at

    times = np.asarray(result.t, dtype=float)
    states = np.asarray(result.y, dtype=float).T
    phases = list(getattr(result, "phases", None) or [])
    gyro = states[:, 10:13].copy()
    accel = np.zeros((len(times), 3))
    saved = (sim._active_chute, sim._deploy_trigger_s)
    try:
        for i, (t, state) in enumerate(zip(times, states)):
            _arm_chute(sim, _phase_at(phases, float(t)))
            point = sim.evaluate(state, float(t))
            acceleration = point.force_inertial_n / point.mass_kg
            if point.rail_phase == "rail":
                acceleration = sim.launch_rail.constrain_acceleration(acceleration, state[3:6])
            elif point.rail_phase == "tipoff":
                acceleration, _ = sim.tipoff_accelerations(point, state)
            felt = acceleration - point.gravity_inertial_n / point.mass_kg
            accel[i] = point.dcm_b2i.T @ felt
    finally:
        sim._active_chute, sim._deploy_trigger_s = saved
    return ImuLog(times, gyro, accel, source="simulation")


@dataclass
class ImuComparison:
    """The measured record, the simulation on its times, and the differences."""

    measured: ImuLog
    simulated: ImuLog
    offset_s: float
    #: RMS of measured minus simulated, per channel, over the boost and overall.
    rms_boost: dict[str, float] = field(default_factory=dict)
    rms_all: dict[str, float] = field(default_factory=dict)
    peaks: dict[str, tuple[float, float]] = field(default_factory=dict)

    CHANNELS = ("axial_g", "roll_rate_dps", "transverse_rate_dps")

    def report(self) -> str:
        lines = [
            f"IMU comparison: {self.measured.source or 'measured'} vs simulation, "
            f"aligned at liftoff (measured record shifted by {self.offset_s:+.3f} s)",
            f"  {'channel':<20} {'RMS boost':>10} {'RMS all':>10} {'peak meas':>10} {'peak sim':>10}",
        ]
        for name in self.CHANNELS:
            meas, sim = self.peaks.get(name, (float("nan"), float("nan")))
            lines.append(
                f"  {name:<20} {self.rms_boost.get(name, float('nan')):10.3f} "
                f"{self.rms_all.get(name, float('nan')):10.3f} {meas:10.2f} {sim:10.2f}"
            )
        return "\n".join(lines)


def _channels(log: ImuLog) -> dict[str, np.ndarray]:
    return {
        "axial_g": log.axial_g,
        "roll_rate_dps": np.degrees(log.roll_rate_radps),
        "transverse_rate_dps": np.degrees(log.transverse_rate_radps),
    }


def compare(measured: ImuLog, simulated: ImuLog, boost_end_s: float | None = None,
            liftoff_threshold_g: float = 1.5) -> ImuComparison:
    """Line the two records up at liftoff and difference them channel by channel.

    ``boost_end_s`` is seconds after liftoff; ``None`` takes the end of
    the simulated record's thrust, read as the last time its axial
    specific force exceeds one g.
    """
    offset = measured.liftoff_time_s(liftoff_threshold_g) - simulated.liftoff_time_s(liftoff_threshold_g)
    aligned = measured.shifted(offset)
    sim_channels = _channels(simulated)
    meas_channels = _channels(aligned)
    t_lift = simulated.liftoff_time_s(liftoff_threshold_g)
    if boost_end_s is None:
        burning = np.flatnonzero(simulated.axial_g > 1.0)
        t_burn = float(simulated.time_s[burning[-1]]) if len(burning) else t_lift
    else:
        t_burn = t_lift + float(boost_end_s)
    window = (aligned.time_s >= simulated.time_s[0]) & (aligned.time_s <= simulated.time_s[-1])
    boost = window & (aligned.time_s >= t_lift) & (aligned.time_s <= t_burn)

    comparison = ImuComparison(aligned, simulated, offset)
    for name in ImuComparison.CHANNELS:
        on_measured_times = np.interp(aligned.time_s, simulated.time_s, sim_channels[name])
        diff = meas_channels[name] - on_measured_times
        comparison.rms_all[name] = float(np.sqrt(np.mean(diff[window] ** 2))) if np.any(window) else float("nan")
        comparison.rms_boost[name] = float(np.sqrt(np.mean(diff[boost] ** 2))) if np.any(boost) else float("nan")
        comparison.peaks[name] = (
            float(np.max(np.abs(meas_channels[name][window]))) if np.any(window) else float("nan"),
            float(np.max(np.abs(sim_channels[name]))),
        )
    return comparison


def write_imu_csv(log: ImuLog, path: str | Path) -> Path:
    """Write a log in the layout ``load_imu_csv`` reads by default (SI units)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "gx", "gy", "gz", "ax", "ay", "az"])
        for t, g, a in zip(log.time_s, log.gyro_radps, log.accel_mps2):
            writer.writerow([f"{t:.5f}", *(f"{v:.6g}" for v in g), *(f"{v:.6g}" for v in a)])
    return path


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compare an IMU log with the simulated flight")
    parser.add_argument("log", help="IMU CSV")
    parser.add_argument("--vehicle", required=True, help="vehicle JSON (a parametric model)")
    parser.add_argument("--elevation", type=float, default=85.0)
    parser.add_argument("--azimuth", type=float, default=0.0)
    parser.add_argument("--rail", type=float, default=3.0)
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--wind-from", type=float, default=0.0)
    parser.add_argument("--time", default="time_s")
    parser.add_argument("--gyro", nargs=3, default=("gx", "gy", "gz"))
    parser.add_argument("--accel", nargs=3, default=("ax", "ay", "az"))
    parser.add_argument("--gyro-units", default="rad/s", choices=("rad/s", "dps"))
    parser.add_argument("--accel-units", default="m/s2", choices=("m/s2", "g"))
    parser.add_argument("--axes", default=None,
                        help="nine numbers, row-major, taking IMU xyz to body xyz")
    args = parser.parse_args(argv)

    from parametric import aero
    from parametric.flight import FlightSettings, fly_model
    from parametric.model import VehicleModel

    with open(args.vehicle, "r", encoding="utf-8") as f:
        model = VehicleModel.from_dict(json.load(f))
    axes = None
    if args.axes:
        axes = np.array([float(v) for v in args.axes.replace(",", " ").split()]).reshape(3, 3)
    measured = load_imu_csv(args.log, args.time, tuple(args.gyro), tuple(args.accel),
                            args.gyro_units, args.accel_units, axes)
    table, _ = aero.run_analysis(model, aero.AeroSettings())
    outcome = fly_model(model, FlightSettings(
        elevation_deg=args.elevation, azimuth_deg=args.azimuth, rail_length_m=args.rail,
        wind_speed_mps=args.wind, wind_direction_deg=args.wind_from, dt_s=0.01,
    ), table)
    comparison = compare(measured, simulated_imu(outcome.simulation, outcome.result))
    print(comparison.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
