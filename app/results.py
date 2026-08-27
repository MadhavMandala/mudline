"""Analysis results, kept rather than thrown away.

Every analysis used to end in a modal dialog: you read six numbers, pressed OK,
and they were gone. That makes the one question a design tool exists to answer
unanswerable -- *did that change help?* You cannot compare a run to the run
before it if neither survives being read.

A ``Result`` is therefore a record: what ran, with what settings, against which
version of the model, what it produced, and the series worth plotting. Results
accumulate in a store, so two can be selected and differenced.

The model fingerprint travels with the result. That is what lets the panel say
"this run describes a vehicle you have since changed", instead of presenting a
stale number with the same confidence as a fresh one.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class Metric:
    """One reported number, with enough context to format and difference it."""

    label: str
    value: float
    unit: str = ""
    decimals: int = 3
    #: True when a larger value is the better outcome. Used only to colour a
    #: difference, never to decide anything.
    higher_is_better: bool | None = None

    def format(self) -> str:
        from app.units import UNITS

        return UNITS.format(self.value, self.unit, self.decimals)

    def format_like(self, other: "Metric") -> str:
        """Format in the unit ``other`` would be shown in, for a comparison."""
        from app.units import FEET_THRESHOLD_M, UNITS

        prefer_feet = (
            abs(other.value) >= FEET_THRESHOLD_M if self.unit == "m" else None
        )
        return UNITS.format(self.value, self.unit, self.decimals, prefer_feet=prefer_feet)

    def delta_text(self, other: "Metric") -> str:
        from app.units import FEET_THRESHOLD_M, UNITS

        # Both sides are converted with the *same* factor, chosen from this
        # metric's own value, so a comparison never mixes inches with feet.
        scale = UNITS.scale_for(self.value, self.unit) if self.unit else 1.0
        label = ""
        if self.unit:
            prefer = abs(self.value) >= FEET_THRESHOLD_M if self.unit == "m" else None
            label = " " + UNITS.unit_label(self.unit, prefer_feet=prefer)
        difference = (self.value - other.value) * scale
        if abs(other.value) > 1e-12:
            percent = 100.0 * (self.value - other.value) / abs(other.value)
            return f"{difference:+,.{self.decimals}f}{label} ({percent:+.1f}%)"
        return f"{difference:+,.{self.decimals}f}{label}"


def _build_provenance() -> dict[str, str]:
    """Version and revision of the running build, for :class:`Result.build`.

    Wrapped rather than used as the factory directly so that a checkout
    without git, or a frozen copy, records an empty provenance instead of
    failing to construct a result.
    """
    try:
        from app.version import provenance

        return provenance()
    except Exception:      # noqa: BLE001
        return {}


@dataclass
class Result:
    """One completed analysis run."""

    index: int
    kind: str
    label: str
    fingerprint: str
    timestamp: datetime = field(default_factory=datetime.now)
    settings: dict[str, str] = field(default_factory=dict)
    metrics: list[Metric] = field(default_factory=list)
    #: name -> (x, y, x_label, y_label), for plotting.
    series: dict[str, tuple] = field(default_factory=dict)
    #: Anything a consumer needs but the panel does not display.
    payload: dict = field(default_factory=dict)
    #: Which build produced this. Filled in automatically, because the run
    #: that most needs identifying is the one nobody thought to label -- a
    #: result file mailed to a colleague outlives the session that made it,
    #: and the model fingerprint above only answers "same vehicle?".
    build: dict[str, str] = field(default_factory=_build_provenance)

    @property
    def title(self) -> str:
        return f"#{self.index}  {self.label}"

    @property
    def clock(self) -> str:
        return self.timestamp.strftime("%H:%M:%S")

    def metric(self, label: str) -> Metric | None:
        for metric in self.metrics:
            if metric.label == label:
                return metric
        return None

    def is_current(self, fingerprint: str) -> bool:
        return self.fingerprint == fingerprint


class ResultStore:
    """The run history for a session."""

    def __init__(self):
        self._results: list[Result] = []
        self._counter = itertools.count(1)

    def __len__(self) -> int:
        return len(self._results)

    def __iter__(self):
        return iter(self._results)

    def add(self, kind: str, label: str, fingerprint: str, **kwargs) -> Result:
        result = Result(
            index=next(self._counter), kind=kind, label=label,
            fingerprint=fingerprint, **kwargs,
        )
        self._results.append(result)
        return result

    def restore(self, results) -> None:
        """Take back results read from a project file, keeping their numbers.

        The counter moves past them, so the next run is numbered after the
        last restored one. Appending straight to the list left the counter
        at 1, and the next run collided with a restored one: two rows
        titled "#1", a selection that grew into a comparison on refresh.
        """
        self._results.extend(results)
        if self._results:
            self._counter = itertools.count(max(r.index for r in self._results) + 1)

    def clear(self) -> None:
        self._results.clear()
        self._counter = itertools.count(1)

    def of_kind(self, kind: str) -> list[Result]:
        return [r for r in self._results if r.kind == kind]

    def latest(self, kind: str) -> Result | None:
        matches = self.of_kind(kind)
        return matches[-1] if matches else None

    def newest_first(self) -> list[Result]:
        return list(reversed(self._results))


# ----------------------------------------------------------------------
# Builders: turn an analysis output into a Result
# ----------------------------------------------------------------------


def mass_result(solved, model, fingerprint: str) -> dict:
    """Fields for a mass-properties run."""
    principal = np.diag(solved.inertia_kg_m2)
    analytic = model.mass_summary()
    return dict(
        kind="mass",
        label="Mass properties",
        fingerprint=fingerprint,
        settings={"source": "meshed CAD"},
        metrics=[
            Metric("Dry mass", solved.mass_kg, "kg", 3),
            Metric("CG station", solved.cg_station_m, "m", 4),
            Metric("Roll inertia", float(principal[2]), "kg·m²", 5),
            Metric("Pitch inertia", float(principal[0]), "kg·m²", 4),
            Metric("Analytic dry mass", analytic.dry_mass_kg, "kg", 3),
        ],
        payload={"solved": solved},
    )


def aero_result(database, geometry, settings, model, fingerprint: str,
                cg_station_m: float | None = None) -> dict:
    """Fields for an aerodynamic sweep, including CD and CP against Mach."""
    machs = np.array(sorted({row.mach for row in database.rows}))
    cd = np.array([database.lookup(float(m), 0.0).cd for m in machs])
    cp = np.array([database.lookup(float(m), 4.0).x_cp_m for m in machs])
    cn = np.array([database.lookup(float(m), 4.0).cn for m in machs])

    # The plume-filled base, when the table carries it: what the vehicle
    # actually flies against while the motor burns.
    from parametric.analysis import loaded_cg_station_m

    diameter = geometry.reference_diameter_m
    summary = model.mass_summary()
    cg_dry = summary.cg_station_m if cg_station_m is None else cg_station_m
    cg_wet = loaded_cg_station_m(model, cg_dry)

    # Rate derivatives, when the table carries them: pitch damping about
    # the loaded CG, and roll damping.
    rate_metrics: list[Metric] = []
    rate_series: dict[str, tuple] = {}
    if getattr(database, "has_damping", False):
        length = database.reference_length_m
        rows = [database.lookup(float(m), 0.0) for m in machs]
        cmq = np.array([
            -2.0 * (r.cna_x2_m2 - 2.0 * cg_wet * r.cna_x_m + cg_wet ** 2 * r.cna_sum)
            / length ** 2
            for r in rows
        ])
        rate_metrics.append(Metric(
            "Cmq at Mach 0.3 (loaded)", float(cmq[np.argmin(np.abs(machs - 0.3))]), "", 2,
        ))
        rate_series["Cmq vs Mach (loaded CG)"] = (machs, cmq, "Mach", "Cmq per rad")
    if getattr(database, "has_roll", False):
        clp = np.array([database.lookup(float(m), 0.0).clp for m in machs])
        rate_metrics.append(Metric(
            "Clp at Mach 0.3", float(clp[np.argmin(np.abs(machs - 0.3))]), "", 3,
        ))
        rate_series["Roll damping Clp vs Mach"] = (machs, clp, "Mach", "Clp")

    burning_metrics: list[Metric] = []
    burning_series: dict[str, tuple] = {}
    if getattr(database, "has_power_on", False):
        cd_on = np.array([database.lookup(float(m), 0.0).cd_power_on for m in machs])
        burning_metrics = [Metric(
            "CD at Mach 0.3 (burning)", float(cd_on[np.argmin(np.abs(machs - 0.3))]),
            "", 4, higher_is_better=False,
        )]
        burning_series = {"CD vs Mach (motor burning)": (machs, cd_on, "Mach", "CD")}

    # Stability is judged loaded, because that is the state the vehicle leaves
    # the rail in. Burnout is reported alongside it rather than instead of it:
    # a design can be fine loaded and over-stable empty, and only one of those
    # ends a flight early.
    margin = (cp - cg_wet) / diameter if diameter > 0 else np.zeros_like(cp)
    margin_dry = (cp - cg_dry) / diameter if diameter > 0 else np.zeros_like(cp)

    subsonic = float(cd[np.argmin(np.abs(machs - 0.3))])
    peak_index = int(np.argmax(cd))

    return dict(
        kind="aero",
        label="Aerodynamics",
        fingerprint=fingerprint,
        settings={
            "Mach": f"{settings.mach_min:.2f} – {settings.mach_max:.2f} "
                    f"({settings.mach_points} pts)",
            "alpha": f"0 – {settings.alpha_max_deg:.0f}° "
                     f"({settings.alpha_points} pts)",
            "altitude": f"{settings.altitude_m:,.0f} m",
            "roughness": f"{settings.roughness_m * 1e6:.0f} µm",
            "protuberances": str(len(geometry.protuberances)),
            "nose": "declared" if geometry.nose_declared else "inferred",
            "base": (
                "plume while burning" if getattr(settings, "power_on_base", False)
                else "power-off throughout"
            ),
        },
        metrics=[
            Metric("CD at Mach 0.3", subsonic, "", 4, higher_is_better=False),
            *burning_metrics,
            Metric("Peak CD", float(cd[peak_index]), "", 4, higher_is_better=False),
            Metric("Peak CD Mach", float(machs[peak_index]), "", 2),
            Metric("CP at Mach 0.3", float(cp[np.argmin(np.abs(machs - 0.3))]), "m", 4),
            Metric("Static margin (loaded)",
                   float(margin[np.argmin(np.abs(machs - 0.3))]),
                   "cal", 2, higher_is_better=True),
            Metric("Static margin (burnout)",
                   float(margin_dry[np.argmin(np.abs(machs - 0.3))]),
                   "cal", 2, higher_is_better=True),
            Metric("CG shift, loaded", cg_wet - cg_dry, "m", 4),
            *rate_metrics,
        ],
        series={
            "CD vs Mach": (machs, cd, "Mach", "CD"),
            **burning_series,
            **rate_series,
            "CN at 4° vs Mach": (machs, cn, "Mach", "CN"),
            "CP vs Mach": (machs, cp, "Mach", "CP m"),
            "Static margin vs Mach (loaded)": (
                machs, margin, "Mach", "calibres",
            ),
            "Static margin vs Mach (burnout)": (
                machs, margin_dry, "Mach", "calibres",
            ),
        },
        payload={"database": database, "geometry": geometry, "settings": settings},
    )


def flight_result(result, stats, peak, settings, fingerprint: str,
                  used_table: bool, log=None, caveats=None) -> dict:
    """Fields for a trajectory run, with the time histories worth plotting.

    ``log`` is the flight's ``FlightLog``; with it the run carries the
    quantities the integrator never kept -- felt acceleration, angle of
    attack, static margin, thrust, CG and CP against time.
    """
    from trajectory.analysis.export import derived_quantities

    times = np.asarray(result.t, dtype=float)
    states = np.asarray(result.y, dtype=float).T
    derived = derived_quantities(states)
    exit_state = getattr(result, "rail_exit", None)
    # The state is above sea level; a flight is quoted above its pad.
    pad = float(getattr(settings, "pad_altitude_m", 0.0))
    altitude = states[:, 1] - pad

    log_metrics: list[Metric] = []
    log_series: dict[str, tuple] = {}
    if log is not None:
        log_metrics.append(Metric("Max acceleration", log.max_acceleration_g, "g", 2))
        burnout = log.burnout_index
        if burnout is not None and np.isfinite(log.static_margin_cal[burnout]):
            log_metrics.append(Metric(
                "Static margin at burnout", float(log.static_margin_cal[burnout]),
                "cal", 2, higher_is_better=True,
            ))
        lowest = log.min_static_margin_cal()
        if np.isfinite(lowest):
            log_metrics.append(Metric(
                "Min static margin (boost)", lowest, "cal", 2, higher_is_better=True,
            ))
        hang = log.hang_angle_deg()
        if hang is not None:
            log_metrics.append(Metric("Hang angle at landing", hang, "deg", 1))
        log_series = {
            "Angle of attack vs time": (log.time_s, log.alpha_deg, "t s", "alpha deg"),
            "Acceleration vs time": (log.time_s, log.acceleration_g, "t s", "g"),
            "Thrust vs time": (log.time_s, log.thrust_n, "t s", "thrust N"),
            "CG station vs time": (log.time_s, log.cg_station_m, "t s", "CG m from nose"),
            "Roll rate vs time": (
                log.time_s, np.degrees(log.roll_rate_radps), "t s", "roll deg/s",
            ),
            "Wind vs altitude": (
                log.altitude_agl_m, log.wind_mps, "altitude above pad m", "wind m/s",
            ),
            "Attitude vs time": (
                log.time_s, log.axis_from_vertical_deg, "t s", "axis off vertical deg",
            ),
        }
        if np.any(np.isfinite(log.static_margin_cal)):
            log_series["Static margin vs time"] = (
                log.time_s, log.static_margin_cal, "t s", "calibres",
            )
            log_series["CP station vs time"] = (
                log.time_s, log.cp_station_m, "t s", "CP m from nose",
            )

    return dict(
        kind="flight",
        label="Flight",
        fingerprint=fingerprint,
        settings={
            "elevation": f"{settings.elevation_deg:.1f}°",
            "azimuth": f"{settings.azimuth_deg:.0f}°",
            "rail": (
                (settings.describe_rail()
                 if used_table or not getattr(settings, "rail_buttons", False)
                 else f"{settings.rail_length_m:.2f} m, CG constrained (tip-off needs the table)")
                if hasattr(settings, "describe_rail")
                else f"{settings.rail_length_m:.2f} m"
            ),
            "pad": f"{pad:.0f} m ASL",
            "latitude": (
                f"{settings.latitude_deg:.1f}° (Coriolis on)"
                if getattr(settings, "latitude_deg", None) is not None
                else "not set (no Coriolis)"
            ),
            "wind": (
                settings.describe_wind()
                if hasattr(settings, "describe_wind")
                else f"{settings.wind_speed_mps:.1f} m/s from {settings.wind_direction_deg:.0f}°"
            ),
            "recovery": (
                settings.describe_recovery()
                if hasattr(settings, "describe_recovery")
                else ("yes" if settings.use_recovery else "no")
            ),
            "aero": "coefficient table" if used_table else "fallback drag law",
            "imperfections": (
                settings.describe_imperfections()
                if hasattr(settings, "describe_imperfections") else "none"
            ),
            # What the tool knew to be doubtful about this run, kept with it.
            "caveats": " | ".join(caveats) if caveats else "none",
        },
        metrics=[
            Metric("Apogee", stats["max_altitude"] - pad, "m", 0,
                   higher_is_better=True),
            Metric("Apogee time", stats["apogee_time"], "s", 1),
            Metric("Max speed", stats["max_velocity"], "m/s", 0),
            Metric("Max-Q", peak["pressure_pa"] / 1000.0, "kPa", 0),
            Metric("Max-Q Mach", peak["mach"], "", 2),
            Metric("Rail exit", exit_state["velocity_mps"] if exit_state else 0.0,
                   "m/s", 1, higher_is_better=True),
            Metric("Rail exit alpha",
                   float(exit_state.get("alpha_deg", 0.0)) if exit_state else 0.0,
                   "deg", 1, higher_is_better=False),
            *([Metric("Tip-off rate", float(exit_state["pitch_rate_dps"]), "deg/s", 1,
                      higher_is_better=False),
               Metric("Tip-off angle", float(exit_state["tip_off_deg"]), "deg", 2,
                      higher_is_better=False)]
              if exit_state and "pitch_rate_dps" in exit_state else []),
            Metric("Downrange", stats["range"], "m", 0),
            Metric("Flight time", stats["flight_time"], "s", 0),
            *log_metrics,
        ],
        series={
            "Altitude vs time": (times, altitude, "t s", "altitude above pad m"),
            "Speed vs time": (times, derived["speed_mps"], "t s", "speed m/s"),
            # Speed is a magnitude and never reaches zero on a tilted launch:
            # at apogee the vehicle is still moving sideways. The vertical
            # component is what crosses zero there -- the signal the apogee
            # event fires on -- and the horizontal one is the drift.
            "Vertical velocity vs time": (
                times, states[:, 4], "t s", "vertical velocity m/s"
            ),
            "Horizontal speed vs time": (
                times, np.hypot(states[:, 3], states[:, 5]), "t s",
                "horizontal speed m/s",
            ),
            "Mach vs time": (times, derived["mach"], "t s", "Mach"),
            "Dynamic pressure vs time": (
                times, derived["dynamic_pressure_pa"] / 1000.0, "t s", "q kPa"
            ),
            "Altitude vs downrange": (
                np.hypot(states[:, 0], states[:, 2]), altitude,
                "downrange m", "altitude above pad m",
            ),
            **log_series,
        },
        payload={"result": result, "log": log},
    )


def dispersion_result(dispersion, fingerprint: str, cases: int, seed: int,
                      used_table: bool | None = None) -> dict:
    """Fields for a dispersion batch."""
    points = np.asarray(dispersion.landing_points, dtype=float)
    _, semi_major, semi_minor, _ = dispersion.ellipse
    altitudes = [c["max_altitude"] for c in dispersion.cases if c.get("success")]
    settings = {"cases": str(cases), "seed": str(seed)}
    if used_table is not None:
        settings["aero"] = "coefficient table" if used_table else "fallback drag law"

    return dict(
        kind="dispersion",
        label="Dispersion",
        fingerprint=fingerprint,
        settings=settings,
        metrics=[
            Metric("CEP", dispersion.cep_m, "m", 0, higher_is_better=False),
            Metric("95% semi-major", semi_major, "m", 0, higher_is_better=False),
            Metric("95% semi-minor", semi_minor, "m", 0, higher_is_better=False),
            Metric("Mean apogee", float(np.mean(altitudes)) if altitudes else 0.0,
                   "m", 0),
            Metric("Apogee spread (1σ)",
                   float(np.std(altitudes)) if altitudes else 0.0, "m", 0),
        ],
        series={
            "Landing points": (
                points[:, 0], points[:, 1], "east m", "north m",
            ),
        },
        payload={"dispersion": dispersion},
    )
