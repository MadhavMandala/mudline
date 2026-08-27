"""Design sweeps: vary a parameter and watch what it costs.

This is the reason the Parm layer exists. Every number in the model is already
named, bounded and addressable, so "vary the fin span from 60 to 140 mm and
show me apogee and static margin" needs no new plumbing -- only a loop that
sets a value, re-runs the chain, and records the answers.

It is also the thing that turns the tool from a calculator you drive by hand
into one that answers a design question. Running fifteen flights manually and
writing the numbers on paper is exactly the work a computer should be doing.

The model is restored afterwards. A sweep mutates the vehicle to evaluate each
point, and leaving it on the last one would quietly change the design the user
was working on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from parametric.flight import FlightSettings
from parametric.model import VehicleModel


@dataclass
class SweepVariable:
    """The parameter being varied, and over what range."""

    component_path: str
    parm: str
    start: float
    stop: float
    steps: int = 9

    @property
    def label(self) -> str:
        name = self.component_path.split("/")[-1]
        return f"{name}.{self.parm}"

    def values(self) -> np.ndarray:
        return np.linspace(self.start, self.stop, max(int(self.steps), 2))


@dataclass
class SweepSettings:
    """What to vary and how much analysis to run at each point."""

    variable: SweepVariable
    #: Flight is the expensive part -- a few seconds per point against a
    #: fraction of one for geometry and aerodynamics -- so it is optional.
    #: Mass, stability and drag alone already answer most shape questions.
    include_flight: bool = True
    include_aero: bool = True
    aero_settings: object | None = None
    #: The settings Run Flight last used, so a swept apogee is the apogee
    #: Run Flight would report for the same design.
    flight_settings: FlightSettings | None = None


@dataclass
class SweepPoint:
    """One evaluated design."""

    value: float
    metrics: dict[str, float] = field(default_factory=dict)
    failed: str = ""


@dataclass
class SweepResult:
    """A completed sweep."""

    variable: SweepVariable
    points: list[SweepPoint]
    unit: str = ""

    @property
    def values(self) -> np.ndarray:
        return np.array([p.value for p in self.points], dtype=float)

    def metric_names(self) -> list[str]:
        names: list[str] = []
        for point in self.points:
            for name in point.metrics:
                if name not in names:
                    names.append(name)
        return names

    def series(self, name: str) -> np.ndarray:
        return np.array(
            [p.metrics.get(name, np.nan) for p in self.points], dtype=float
        )

    def best(self, name: str, maximise: bool = True) -> SweepPoint | None:
        """The point with the highest or lowest value of a metric."""
        candidates = [p for p in self.points if name in p.metrics]
        if not candidates:
            return None
        key = (max if maximise else min)
        return key(candidates, key=lambda p: p.metrics[name])

    def report(self) -> str:
        names = self.metric_names()
        header = f"{self.variable.label:>16}" + "".join(f"{n:>14}" for n in names)
        lines = [f"Sweep: {self.variable.label} "
                 f"{self.variable.start:g} to {self.variable.stop:g} "
                 f"({len(self.points)} points)", "", header]
        for point in self.points:
            row = f"{point.value:>16.4g}"
            for name in names:
                value = point.metrics.get(name)
                row += f"{value:>14.4g}" if value is not None else f"{'-':>14}"
            if point.failed:
                row += f"   {point.failed}"
            lines.append(row)
        return "\n".join(lines)


# ----------------------------------------------------------------------


def sweepable_parms(model: VehicleModel) -> list[tuple[str, str, str]]:
    """Every parameter a sweep could drive: (path, parm, display label).

    Only bounded, designable parms are offered. An unbounded parameter has no
    sensible default range to propose, and one marked non-designable -- fin
    count, for instance -- is an integer choice rather than a continuum.
    Imported parts are skipped entirely: their geometry belongs to the CAD.
    """
    found: list[tuple[str, str, str]] = []
    for component in model.walk():
        # An imported part is CAD, not a design variable. Sweeping one would
        # produce results for a vehicle the STEP file does not describe.
        if getattr(component, "imported", False):
            continue
        for parm in component.parms():
            if not parm.designable:
                continue
            if not (np.isfinite(parm.minimum) and np.isfinite(parm.maximum)):
                continue
            found.append((
                component.path, parm.name,
                f"{component.name}.{parm.name}"
                + (f"  [{parm.unit}]" if parm.unit else ""),
            ))
    return found


def default_range(model: VehicleModel, path: str, parm_name: str,
                  span: float = 0.4) -> tuple[float, float]:
    """A sensible sweep range around the current value.

    Plus or minus 40% of the current value, clipped to the parm's own bounds,
    which is a better starting point than the full legal range: a fin span
    could legally be 100 m and nobody wants that as the first point.
    """
    component = _find(model, path)
    parm = component.parm(parm_name)
    value = parm.value
    if value == 0:
        width = max(abs(parm.maximum - parm.minimum) * 0.1, 1e-3)
        low, high = parm.minimum, parm.minimum + width
    else:
        low, high = value * (1.0 - span), value * (1.0 + span)
    return (
        float(max(low, parm.minimum)),
        float(min(high, parm.maximum)),
    )


def _find(model: VehicleModel, path: str):
    for component in model.walk():
        if component.path == path:
            return component
    raise KeyError(f"No component at {path!r}")


def run_sweep(
    model: VehicleModel,
    settings: SweepSettings,
    progress: Callable[[int, int, float], bool] | None = None,
) -> SweepResult:
    """Evaluate the model across the sweep range.

    Args:
        progress: called with (index, total, value) before each point; return
            False to stop early. A sweep with flight enabled takes seconds per
            point, so it has to be interruptible.
    """
    from parametric import aero, analysis

    variable = settings.variable
    values = variable.values()
    component = _find(model, variable.component_path)
    parm = component.parm(variable.parm)
    original = parm.value
    unit = parm.unit

    points: list[SweepPoint] = []
    try:
        for index, value in enumerate(values):
            if progress is not None and not progress(index, len(values), float(value)):
                break

            component.set(variable.parm, float(value))
            point = SweepPoint(value=float(value))

            try:
                summary = model.mass_summary()
                point.metrics["dry mass kg"] = summary.dry_mass_kg
                point.metrics["CG m"] = summary.wet_cg_station_m

                database = None
                if settings.include_aero or settings.include_flight:
                    database, geometry = aero.run_analysis(
                        model, settings.aero_settings
                    )
                    point.metrics["CD at M0.3"] = database.lookup(0.3, 0.0).cd
                    # Both, because a design can be stable loaded and
                    # over-stable empty, and only one of those is a problem.
                    point.metrics["static margin cal"] = analysis.static_margin(
                        model, loaded=True
                    )
                    point.metrics["margin burnout cal"] = analysis.static_margin(
                        model, loaded=False
                    )

                if settings.include_flight:
                    point.metrics.update(_fly(
                        model, database, settings.flight_settings,
                        settings.aero_settings,
                    ))
            except Exception as exc:  # noqa: BLE001
                # One infeasible design must not abandon the sweep; record why
                # and carry on, since the shape of the curve either side of a
                # failure is usually the interesting part.
                point.failed = str(exc)[:80]

            points.append(point)
    finally:
        component.set(variable.parm, original)

    return SweepResult(variable=variable, points=points, unit=unit)


def _fly(model: VehicleModel, database, flight_settings,
         aero_settings=None) -> dict[str, float]:
    """Fly one design and reduce it to the numbers a sweep cares about.

    Through ``fly_model``, with the settings Run Flight last used -- wind,
    azimuth, pad altitude, time step and the coupled-aero rebuild included.
    This used to be a launch sequence of its own that read three of the
    settings and defaulted the rest, so a swept apogee did not match the
    one Run Flight reported for the same design.
    """
    from parametric.flight import fly_model

    settings = flight_settings if flight_settings is not None else FlightSettings()
    outcome = fly_model(model, settings, database, aero_settings=aero_settings)
    peak = outcome.peak

    return {
        "apogee m": outcome.apogee_agl_m,
        "max speed m/s": outcome.stats["max_velocity"],
        "max-Q kPa": peak["pressure_pa"] / 1000.0,
        "rail exit m/s": outcome.rail_exit_mps,
    }
