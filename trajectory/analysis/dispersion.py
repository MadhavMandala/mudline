"""Monte Carlo dispersion studies.

Why this module exists
----------------------
``trajectory.analysis.monte_carlo.MonteCarlo`` and the CEP / landing-ellipse
routines in ``trajectory.analysis.statistics`` were both fully written and
neither was called from anywhere in the codebase. The sampler had no simulation
to drive and the statistics had no landing points to consume. This module is
the missing connection: it maps a dictionary of dispersed parameters onto a
configured ``RocketSimulation``, flies it, and returns the scalar outcomes the
statistics functions expect.

What gets dispersed
-------------------
The defaults below are representative rather than authoritative -- they are a
starting point to be replaced with a specific vehicle's measured or specified
tolerances. Each entry is ``(mean, std, low, high)`` and is sampled from a
truncated normal, so the bounds are hard limits rather than suggestions.

The parameters chosen are the ones that actually move a landing point:

* ``impulse_scale``   motor-to-motor total impulse variation. Scales the thrust
  curve and the propellant load together, so burn time is unchanged and
  delivered impulse moves with it. This is the dominant motor uncertainty and
  the one that behaves the way intuition expects: more impulse, higher apogee.
* ``thrust_scale``    thrust level at a *fixed* propellant load. Total impulse
  is unchanged; only burn duration moves. Note that raising it does not
  necessarily raise apogee: for a drag-dominated vehicle, burning the same
  propellant faster buys less against gravity than it pays in drag. The sample
  vehicle here loses roughly 15 km of apogee per 10% of thrust increase for
  exactly that reason. Left at 1.0 by default, since it is a burn-rate study
  parameter rather than a tolerance.
* ``dry_mass_kg``     build mass tolerance
* ``aero_scale``      spread between predicted and actual aerodynamic force
* ``launch_elevation_deg`` / ``launch_azimuth_deg``  rail setting and survey error
* ``wind_speed_mps`` / ``wind_direction_deg``  the usual dominant contributor
* ``thrust_tilt_x_deg`` / ``thrust_tilt_z_deg``  the thrust line off the axis,
  as two components so the magnitude is Rayleigh and the direction uniform
* ``cg_offset_x_m`` / ``cg_offset_z_m``  the dry CG off the centreline
* ``fin_cant_offset_deg``  a cant the fins were built with and the table
  was not; applied through the table's forcing per radian of cant

The last three are moments, and they need an aerodynamic restoring moment
to trim against; on a vehicle flying the fallback drag law -- no normal
force, no centre of pressure -- they only tumble it, so they are not in
the library defaults and the dialog leaves them at zero without a table.
Each adds to whatever the simulation already carries, so a nominal
misalignment set on the flight and a dispersed one compose.

Process model
-------------
``run_case`` is a module-level function taking a single dict, which is what
makes ``multiprocessing.Pool.map`` able to use it. A closure over a configured
simulation object would not pickle. A non-default vehicle is dispersed by
supplying a picklable case function -- see ``run_dispersion(case_fn=...)``
and ``parametric.dispersion.ModelCaseRunner``, which flies a parametric model
through the ``perturb_simulation`` and ``reduce_outcome`` below, so the
library's placeholder vehicle and a real one are perturbed by the same code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from trajectory.analysis.monte_carlo import MonteCarlo
from trajectory.analysis.statistics import (
    compute_cepxy,
    flight_statistics,
    landing_ellipse,
)


# (mean, std, low, high) per dispersed parameter.
DEFAULT_DISPERSIONS: dict[str, tuple[float, float, float, float]] = {
    "impulse_scale":        (1.00,  0.02,  0.92,  1.08),
    "dry_mass_kg":          (50.0,  1.5,   44.0,  56.0),
    "aero_scale":           (1.00,  0.08,  0.75,  1.25),
    "launch_elevation_deg": (85.0,  0.5,   82.0,  88.0),
    "launch_azimuth_deg":   (0.0,   2.0,  -8.0,   8.0),
    "wind_speed_mps":       (5.0,   2.5,   0.0,   15.0),
    "wind_direction_deg":   (0.0,  60.0, -180.0, 180.0),
}


@dataclass
class DispersionResult:
    """Outcome of a dispersion batch."""

    cases: list[dict]
    landing_points: np.ndarray      # (n, 2) as [East, North]
    summary: dict
    cep_m: float
    mean_landing_m: np.ndarray
    ellipse: tuple
    #: Cases that never reached the ground -- a cutoff, a failed integration
    #: -- and so contributed no landing point.
    discarded: int = 0

    @property
    def n_cases(self) -> int:
        return len(self.cases)

    def report(self) -> str:
        center, semi_major, semi_minor, orientation = self.ellipse
        lines = [
            f"Monte Carlo dispersion: {self.n_cases} cases"
            + (f" ({self.discarded} never landed, left out)" if self.discarded else ""),
            f"  CEP:            {self.cep_m:,.0f} m",
            f"  Mean landing:   E {self.mean_landing_m[0]:,.0f} m, "
            f"N {self.mean_landing_m[1]:,.0f} m",
            f"  95% ellipse:    {semi_major:,.0f} m x {semi_minor:,.0f} m "
            f"at {np.degrees(orientation):.0f} deg",
        ]
        for metric in ("max_altitude", "max_velocity", "flight_time"):
            if metric in self.summary:
                s = self.summary[metric]
                lines.append(
                    f"  {metric:<14} mean {s['mean']:,.1f}  "
                    f"sd {s['std']:,.1f}  "
                    f"[{s['min']:,.1f}, {s['max']:,.1f}]"
                )
        return "\n".join(lines)


def build_case_simulation(params: dict):
    """The library's default vehicle, perturbed by one sampled parameter set.

    Imported lazily so that a worker process pays the import cost once and this
    module stays importable without pulling in the whole simulation stack.
    """
    from trajectory import simulation as tm

    return perturb_simulation(tm.RocketSimulation(), params)


def perturb_simulation(sim, params: dict):
    """Apply one sampled parameter set to an already configured simulation.

    Works on any ``RocketSimulation`` -- the library default above, or one
    built from a parametric model -- so a real vehicle and the placeholder
    are dispersed by the same rules. Keys absent from ``params`` leave the
    simulation as it was.
    """
    from trajectory.environment.wind import WindModel

    # impulse_scale moves thrust and propellant together (burn time fixed,
    # total impulse scaled); thrust_scale moves thrust alone (total impulse
    # fixed, burn time scaled). They compose.
    impulse_scale = float(params.get("impulse_scale", 1.0))
    thrust_scale = float(params.get("thrust_scale", 1.0))
    curve_scale = impulse_scale * thrust_scale

    if curve_scale != 1.0:
        # Rebuild the engine rather than mutating the interpolator, so the
        # vacuum-referencing and mass-flow derivation stay consistent.
        from trajectory.vehicle.engine import Engine
        curve = np.asarray(sim.engine._vacuum_curve.y, dtype=float) * curve_scale
        times = np.asarray(sim.engine._vacuum_curve.x, dtype=float)
        sim.engine = Engine(
            thrust_curve=curve,
            time_points=times,
            isp_vac=sim.engine.isp_vac,
            isp_sl=sim.engine.isp_sl,
            # Exit area is geometry: it does not change because the motor
            # delivered a few percent more impulse than nominal.
            nozzle_area=sim.engine.nozzle_area,
            thrust_reference="vacuum",
        )

    if impulse_scale != 1.0:
        sim.mass_props.prop_mass *= impulse_scale

    if "dry_mass_kg" in params:
        sim.mass_props.dry_mass = float(params["dry_mass_kg"])

    # Recomputed unconditionally: either perturbation above invalidates it, and
    # a stale mass_0 silently corrupts the inertia scaling in at_propellant.
    sim.mass_props.mass_0 = sim.mass_props.dry_mass + sim.mass_props.prop_mass

    sim.aero_scale = float(params.get("aero_scale", 1.0))

    # Build imperfections, added to what the simulation already has.
    tilt = np.array([
        float(params.get("thrust_tilt_x_deg", 0.0)),
        float(params.get("thrust_tilt_z_deg", 0.0)),
    ])
    if np.any(tilt):
        sim.thrust_tilt_rad = np.asarray(sim.thrust_tilt_rad, dtype=float) + np.radians(tilt)
    offset = np.array([
        float(params.get("cg_offset_x_m", 0.0)), 0.0,
        float(params.get("cg_offset_z_m", 0.0)),
    ])
    if np.any(offset):
        sim.mass_props.cg_dry = np.asarray(sim.mass_props.cg_dry, dtype=float) + offset
    cant = float(params.get("fin_cant_offset_deg", 0.0))
    if cant != 0.0:
        sim.fin_cant_offset_rad = float(sim.fin_cant_offset_rad) + np.radians(cant)
        sim._rebuild_aero_model()

    # The surface wind is changed in place so winds aloft and turbulence
    # the simulation already carries stay with it; replacing the model
    # used to drop them.
    if "wind_speed_mps" in params or "wind_direction_deg" in params:
        wind = sim.wind if sim.wind is not None else WindModel()
        if "wind_speed_mps" in params:
            wind.surface_wind = np.array([max(float(params["wind_speed_mps"]), 0.0), 0.0])
        if "wind_direction_deg" in params:
            wind.surface_dir = np.radians(float(params["wind_direction_deg"]))
        sim.wind = wind
    if params.get("turbulence_seed") is not None and getattr(sim.wind, "turbulence", None) is not None:
        sim.wind.turbulence.reseed(int(round(float(params["turbulence_seed"]))))

    return sim


def run_case(params: dict) -> dict:
    """Fly one dispersed trajectory and reduce it to scalar outcomes.

    Module level and dict-argument by necessity: this is what gets pickled and
    shipped to a worker process.
    """
    from trajectory.vehicle.recovery import standard_recovery

    sim = build_case_simulation(params)
    recovery = standard_recovery(
        dry_mass_kg=sim.mass_props.dry_mass,
        main_deploy_altitude_m=float(params.get("main_deploy_altitude_m", 500.0)),
    )

    result = sim.run(
        launch_azimuth=np.radians(float(params.get("launch_azimuth_deg", 0.0))),
        launch_elevation=np.radians(float(params.get("launch_elevation_deg", 85.0))),
        t_max=(None if params.get("t_max") is None
               else float(params["t_max"])),
        dt=float(params.get("dt", 0.25)),
        recovery=recovery,
    )
    return reduce_outcome(result, params)


def reduce_outcome(result, params: dict, ground_m: float = 0.0) -> dict:
    """Reduce a flown trajectory to the scalars the statistics consume.

    ``ground_m`` is the pad altitude above sea level. Apogee is reported
    above the pad, which is how a flight is quoted; the landing point stays
    in the inertial frame the ellipse is drawn in.
    """
    states = result.y.T
    stats = flight_statistics(states, result.t)
    stats["max_altitude"] -= float(ground_m)
    landed = states[-1]
    outcome = {
        **stats,
        "landing_east_m": float(landed[0]),
        "landing_north_m": float(landed[2]),
        "landing_speed_mps": float(np.linalg.norm(landed[3:6])),
        "success": bool(result.success),
        # Whether the last state is on the ground at all. A flight cut off
        # by t_max used to be filed as a landing at wherever it stopped --
        # kilometres from where it would have come down.
        "landed": bool(getattr(result, "landed", True)),
        "params": dict(params),
    }
    exit_state = getattr(result, "rail_exit", None)
    outcome["rail_exit_mps"] = float(exit_state["velocity_mps"]) if exit_state else 0.0
    outcome["rail_exit_alpha_deg"] = (
        float(exit_state.get("alpha_deg", 0.0)) if exit_state else 0.0
    )
    return outcome


class _WithFixed:
    """A case function with constant parameters folded into every case.

    A class rather than a closure, so it can be sent to a worker process.
    """

    def __init__(self, runner: Callable[[dict], dict], fixed: dict):
        self.runner = runner
        self.fixed = dict(fixed)

    def __call__(self, params: dict) -> dict:
        return self.runner({**self.fixed, **params})


def run_dispersion(
    n_cases: int = 100,
    dispersions: dict | None = None,
    seed: int | None = 12345,
    n_processes: int = 1,
    case_fn: Callable[[dict], dict] | None = None,
    fixed: dict | None = None,
    require_landing: bool = True,
    progress: Callable[[int, int], bool] | None = None,
) -> DispersionResult:
    """Run a dispersion batch and reduce it to landing statistics.

    Args:
        n_cases: Number of trajectories to fly.
        dispersions: ``{name: (mean, std, low, high)}``; defaults to
            ``DEFAULT_DISPERSIONS``.
        seed: Sampling seed. Fixed by default so a study is reproducible --
            an unseeded dispersion result cannot be checked by anyone else.
        n_processes: Worker processes. Each case is independent, so this
            scales close to linearly.
        case_fn: Override the per-case function, for non-default vehicles.
        fixed: Parameters held constant across every case, merged into each
            sampled set. ``{"t_max": 900.0}`` caps every flight at 900 s;
            without it each case flies until it lands.
        require_landing: Only cases that reached the ground contribute a
            landing point; the rest are counted in ``discarded``. Off, a
            cut-off flight's last state is taken as its landing -- only
            sensible for a plumbing test that truncates every flight.
        progress: Called ``progress(done, total)`` as each case lands; return
            False to stop early and reduce the cases flown so far. A study of
            a few hundred cases takes minutes, so a caller with a user
            attached needs both halves of this: something to show, and a way
            out. The statistics of a stopped run are honest -- they are simply
            over fewer samples, and ``DispersionResult.cases`` says how many.
    """
    distributions = dict(DEFAULT_DISPERSIONS if dispersions is None else dispersions)
    fixed = dict(fixed or {})
    runner = case_fn or run_case

    if fixed:
        runner = _WithFixed(runner, fixed)

    monte_carlo = MonteCarlo(sim_func=runner, seed=seed)
    cases = monte_carlo.run_batch(
        n_samples=n_cases,
        param_distributions=distributions,
        n_processes=n_processes,
        progress=progress,
    )
    if not cases:
        raise RuntimeError("The dispersion study was stopped before any case flew.")

    flown = [
        c for c in cases
        if c.get("success") and (c.get("landed", True) or not require_landing)
    ]
    if not flown:
        raise RuntimeError(
            "No dispersion case reached the ground; nothing to draw a landing "
            "ellipse from."
        )

    landing_points = np.array(
        [[c["landing_east_m"], c["landing_north_m"]] for c in flown], dtype=float
    )
    cep, mean_pos, _, _ = compute_cepxy(landing_points)
    ellipse = landing_ellipse(landing_points)
    summary = monte_carlo.summarize(
        flown, ["max_altitude", "max_velocity", "flight_time", "landing_speed_mps"]
    )

    return DispersionResult(
        cases=cases,
        landing_points=landing_points,
        summary=summary,
        cep_m=float(cep),
        mean_landing_m=np.asarray(mean_pos, dtype=float),
        ellipse=ellipse,
        discarded=len(cases) - len(flown),
    )
