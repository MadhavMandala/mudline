"""Fly a parametric model under one set of flight settings.

Why this module exists
----------------------
Three things fly a vehicle -- the Run Flight action, the design sweep and
the dispersion study -- and each used to carry its own copy of the launch
sequence. The copies disagreed. The sweep ignored wind, azimuth and the pad
altitude and integrated at a coarser step, so a swept apogee was not the
apogee Run Flight reported for the same design. The dispersion study did not
build the vehicle at all: it flew the simulator's built-in placeholder and
filed the landing ellipse against the open model. One function that all
three call is the fix, and the reason the settings live here rather than in
the dialog that edits them.

The pad altitude is real here. The vehicle starts that high above sea
level, the atmosphere is evaluated at the altitude it is actually at, and
the ground -- and the altitude the main parachute triggers on -- is the pad.
The setting used to shift only the coupled-aero profile; every flight left
from sea level, which on a desert pad is about twelve percent too much air.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from parametric.model import VehicleModel

#: Coupled-aero passes: fly, rebuild the drag table along the altitudes
#: actually flown, re-fly. Two almost always suffice; the third is insurance.
COUPLED_PASSES = 3
#: The fixed point is declared reached when the apogee moves by less than
#: this fraction between passes.
COUPLED_TOLERANCE = 0.001
#: Below this exit speed the fins have not yet got enough air to hold the
#: vehicle once the rail lets go -- the usual rule of thumb, not a law.
MIN_RAIL_EXIT_MPS = 15.0
#: A tip-off rate worth remarking on: a few tens of degrees a second is
#: what a typical launch produces, and the fins take it out in a second.
MAX_TIPOFF_RATE_DPS = 30.0


@dataclass
class FlightSettings:
    """Everything a flight run depends on."""

    elevation_deg: float = 85.0
    #: From North, positive toward East.
    azimuth_deg: float = 0.0
    rail_length_m: float = 3.0
    #: Ride the rail on two buttons, with the tip-off phase between the
    #: forward one leaving and the aft one. ``rail_buttons_m`` names their
    #: stations from the nose tip; ``None`` puts the forward one at the
    #: loaded CG and the aft one near the tail, which is where they
    #: usually are. Off -- or without a coefficient table, since the
    #: fallback drag law has nothing to arrest a tip-off rate with -- the
    #: CG is constrained until it has travelled the rail's length.
    rail_buttons: bool = True
    rail_buttons_m: tuple[float, float] | None = None
    #: Rail foot above sea level. The flight starts and ends here, and the
    #: air is as thin as it really is up here.
    pad_altitude_m: float = 0.0
    #: Launch-site latitude for the Coriolis term, or ``None`` for a frame
    #: that does not rotate. Tens of metres on a high flight, a systematic
    #: bias rather than a spread, and so worth having right in an ellipse.
    latitude_deg: float | None = None

    wind_speed_mps: float = 0.0
    #: Meteorological: the bearing the wind blows *from*, clockwise from North.
    wind_direction_deg: float = 0.0
    #: Winds aloft as ``(altitude_m, speed_mps, from_deg)`` rows above the
    #: surface layer, interpolated as vectors; the top level holds above it.
    wind_aloft: list = field(default_factory=list)
    #: Dryden turbulence: "none", "light", "moderate" or "severe", and the
    #: seed of its frozen field. A dispersion draws a seed per case.
    turbulence: str = "none"
    turbulence_seed: int = 0

    use_recovery: bool = True
    drogue_descent_mps: float = 18.0
    main_descent_mps: float = 5.0
    #: Above the pad, as an altimeter measures it.
    main_deploy_altitude_m: float = 150.0
    #: The canopy pulls at the harness attachment rather than the CG, so
    #: the vehicle hangs from it. ``chute_attachment_station_m`` is that
    #: point as a station from the nose tip; ``None`` takes the nose
    #: shoulder, where a harness is usually anchored.
    chute_at_attachment: bool = True
    chute_attachment_station_m: float | None = None

    #: No flight-time cutoff: the flight runs until the vehicle is back on
    #: the ground. There used to be a "Max time" here, and a slow descent
    #: from a high apogee ran into it and stopped in mid-air.
    dt_s: float = 0.05
    #: Solver tolerances. ``None`` takes the integrator's defaults (1e-6,
    #: 1e-8), which resolve the attitude to a hundredth of a degree; the
    #: library's own 1e-3 is a fast preview.
    rtol: float | None = None
    atol: float | None = None
    use_aero_table: bool = True
    #: Rebuild the drag table along the flown trajectory and re-fly until the
    #: apogee stops moving. A table built at sea level overstates skin
    #: friction wherever the vehicle actually is; on a high flight the
    #: difference is worth hundreds to thousands of feet of apogee.
    couple_aero_altitude: bool = True

    #: Build imperfections the nominal vehicle carries. Each is a magnitude
    #: and a clock angle about the body axis, from body X toward body Z.
    #: The thrust line off the thrust axis; the dry CG off the centreline;
    #: a fin cant the table was not built with. All zero for a perfect
    #: build; a dispersion study adds its own spread on top.
    thrust_misalignment_deg: float = 0.0
    thrust_misalignment_clock_deg: float = 0.0
    cg_offset_m: float = 0.0
    cg_offset_clock_deg: float = 0.0
    fin_cant_offset_deg: float = 0.0

    def pad_position_m(self) -> np.ndarray:
        """Inertial position of the rail foot: the pad, at its altitude."""
        return np.array([0.0, float(self.pad_altitude_m), 0.0])

    def describe_rail(self) -> str:
        text = f"{self.rail_length_m:.2f} m"
        if not self.rail_buttons:
            return text + ", CG constrained"
        if self.rail_buttons_m is not None:
            forward, aft = sorted(self.rail_buttons_m)
            return text + f", buttons at {forward:.2f} and {aft:.2f} m"
        return text + ", buttons at the CG and the tail"

    def describe_recovery(self) -> str:
        if not self.use_recovery:
            return "no"
        if not self.chute_at_attachment:
            return "yes, canopy through the CG"
        where = (
            f"{self.chute_attachment_station_m:.2f} m from the nose"
            if self.chute_attachment_station_m is not None else "the nose shoulder"
        )
        return f"yes, hanging from {where}"

    def describe_wind(self) -> str:
        text = f"{self.wind_speed_mps:.1f} m/s from {self.wind_direction_deg:.0f}°"
        if self.wind_aloft:
            text += f", {len(self.wind_aloft)} level{'s' if len(self.wind_aloft) != 1 else ''} aloft"
        if self.turbulence != "none":
            text += f", {self.turbulence} turbulence (seed {self.turbulence_seed})"
        return text

    def imperfections(self) -> dict[str, float]:
        """The build imperfections as the components ``perturb_simulation`` takes."""
        params: dict[str, float] = {}
        if self.thrust_misalignment_deg:
            clock = np.radians(float(self.thrust_misalignment_clock_deg))
            params["thrust_tilt_x_deg"] = float(self.thrust_misalignment_deg) * np.cos(clock)
            params["thrust_tilt_z_deg"] = float(self.thrust_misalignment_deg) * np.sin(clock)
        if self.cg_offset_m:
            clock = np.radians(float(self.cg_offset_clock_deg))
            params["cg_offset_x_m"] = float(self.cg_offset_m) * np.cos(clock)
            params["cg_offset_z_m"] = float(self.cg_offset_m) * np.sin(clock)
        if self.fin_cant_offset_deg:
            params["fin_cant_offset_deg"] = float(self.fin_cant_offset_deg)
        return params

    def describe_imperfections(self) -> str:
        parts = []
        if self.thrust_misalignment_deg:
            parts.append(f"thrust {self.thrust_misalignment_deg:.2f}° "
                         f"at {self.thrust_misalignment_clock_deg:.0f}°")
        if self.cg_offset_m:
            parts.append(f"CG {self.cg_offset_m * 1000:.1f} mm "
                         f"at {self.cg_offset_clock_deg:.0f}°")
        if self.fin_cant_offset_deg:
            parts.append(f"fin cant {self.fin_cant_offset_deg:+.2f}°")
        return ", ".join(parts) if parts else "none"


@dataclass
class FlightOutcome:
    """A flown trajectory and what it was flown with."""

    result: Any
    settings: FlightSettings
    used_table: bool
    coupled_passes: int
    #: The table the final pass flew on -- the coupled one when coupling
    #: ran -- which is the table a dispersion of this flight should reuse.
    database: Any | None = None
    #: The simulation the final pass flew, kept so the flight log can
    #: re-evaluate the force model along the stored states.
    simulation: Any | None = field(default=None, repr=False)
    #: What the boattail caveat is keyed on: the vehicle's steepest boattail
    #: and the base-drag law the table was built with.
    boattail_half_angle_deg: float = 0.0
    boattail_model: str = "rasaero"
    _log: Any | None = field(default=None, repr=False, init=False)

    @property
    def log(self):
        """The flight's time history through the force model.

        ``None`` when the outcome carries no simulation -- a run restored
        from a project file, say. Built once and kept.
        """
        if self._log is None and self.simulation is not None:
            from trajectory.analysis.flightlog import FlightLog

            self._log = FlightLog.from_flight(self.simulation, self.result)
        return self._log

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self.result.t, dtype=float)

    @property
    def states(self) -> np.ndarray:
        return np.asarray(self.result.y, dtype=float).T

    @property
    def stats(self) -> dict:
        """``flight_statistics`` of the run. Altitudes are above sea level."""
        from trajectory.analysis.statistics import flight_statistics

        return flight_statistics(self.states, self.times)

    @property
    def apogee_agl_m(self) -> float:
        """Apogee above the pad, which is the number a flight is quoted by."""
        return self.stats["max_altitude"] - float(self.settings.pad_altitude_m)

    @property
    def peak(self) -> dict:
        from trajectory.analysis.export import max_q

        return max_q(self.result)

    @property
    def landed(self) -> bool:
        return bool(getattr(self.result, "landed", True))

    @property
    def rail_exit(self) -> dict | None:
        """The exact rail-exit state, or ``None`` without a rail."""
        return getattr(self.result, "rail_exit", None)

    @property
    def rail_exit_mps(self) -> float:
        exit_state = self.rail_exit
        return float(exit_state["velocity_mps"]) if exit_state else 0.0

    @property
    def rail_exit_alpha_deg(self) -> float:
        exit_state = self.rail_exit
        return float(exit_state.get("alpha_deg", 0.0)) if exit_state else 0.0

    def rail_check(self) -> list[str]:
        """What the first instant off the rail says about the launch.

        The rail zeroes every rotation, so the vehicle leaves it pointing
        where the rail pointed, into whatever relative wind the crosswind
        makes of its airspeed. That first angle of attack is usually the
        largest of the boost and the one the table has to cover; the exit
        speed is the go/no-go number the fins need. Each note is a finding,
        not a refusal -- the flight was flown either way.
        """
        exit_state = self.rail_exit
        if exit_state is None:
            return []
        notes: list[str] = []
        speed = float(exit_state["velocity_mps"])
        alpha = float(exit_state.get("alpha_deg", 0.0))
        if speed < MIN_RAIL_EXIT_MPS:
            notes.append(
                f"Rail exit at {speed:.1f} m/s is below the {MIN_RAIL_EXIT_MPS:.0f} m/s "
                f"rule of thumb; the fins may not hold the vehicle once the rail "
                f"lets go."
            )
        rate = float(exit_state.get("pitch_rate_dps", 0.0))
        if rate > MAX_TIPOFF_RATE_DPS:
            notes.append(
                f"Tip-off left the vehicle pitching at {rate:.0f} deg/s off the rail "
                f"({exit_state.get('tip_off_deg', 0.0):.1f} deg off the rail's line); "
                f"a longer rail or a lighter first second would calm it."
            )
        if self.used_table and self.database is not None:
            _, alpha_max = self.database.alpha_range_deg
            if alpha > alpha_max + 1e-9:
                notes.append(
                    f"Angle of attack off the rail is {alpha:.1f} deg in "
                    f"{exit_state.get('wind_mps', 0.0):.1f} m/s of wind; the table "
                    f"stops at {alpha_max:.0f} deg and holds its edge beyond it."
                )
        return notes

    @property
    def max_mach(self) -> float:
        log = self.log
        return float(np.max(log.mach)) if log is not None and len(log) else 0.0

    def caveats(self) -> list[str]:
        """Everything the tool knows to be doubtful about this flight.

        The rail check's findings, and -- on a table-flown vehicle with a
        boattail past RASAero's separation clamp that went supersonic --
        the known drag discrepancy. Each is a finding, not a refusal.
        """
        notes = self.rail_check()
        if self.used_table:
            from parametric.aero import boattail_caveat

            note = boattail_caveat(self.boattail_half_angle_deg, self.max_mach, self.boattail_model)
            if note:
                notes.append(note)
        return notes

    def positions_from_pad(self) -> np.ndarray:
        """The trajectory relative to the rail foot, for drawing beside the model."""
        return self.states[:, 0:3] - self.settings.pad_position_m()


def parse_wind_aloft(text: str) -> list[tuple[float, float, float]]:
    """Winds aloft from text: one level per line, ``altitude speed from``.

    Commas, semicolons and whitespace all separate; blank lines and lines
    starting with ``#`` are skipped. Raises ``ValueError`` naming the line
    that could not be read.
    """
    levels = []
    for number, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p for p in line.replace(",", " ").replace(";", " ").split() if p]
        if len(parts) != 3:
            raise ValueError(
                f"line {number}: expected 'altitude_m speed_mps from_deg', got {raw!r}"
            )
        try:
            altitude, speed, bearing = (float(p) for p in parts)
        except ValueError as exc:
            raise ValueError(f"line {number}: {raw!r} is not three numbers") from exc
        if speed < 0.0:
            raise ValueError(f"line {number}: a wind cannot blow at {speed} m/s")
        levels.append((altitude, speed, bearing))
    return sorted(levels)


def wind_model_for(settings: FlightSettings):
    """The wind the settings describe, or ``None`` for calm, smooth air."""
    from trajectory.environment.wind import DrydenTurbulence, WindModel

    turbulence = DrydenTurbulence.from_level(
        getattr(settings, "turbulence", "none"), int(getattr(settings, "turbulence_seed", 0))
    )
    aloft = list(getattr(settings, "wind_aloft", []) or [])
    if settings.wind_speed_mps <= 0.0 and not aloft and turbulence is None:
        return None
    return WindModel(
        surface_wind=np.array([max(float(settings.wind_speed_mps), 0.0), 0.0]),
        surface_dir=np.radians(float(settings.wind_direction_deg)),
        aloft=aloft or None,
        turbulence=turbulence,
    )


def configure_simulation(model: VehicleModel, settings: FlightSettings,
                         database=None, solved=None):
    """A simulation of the model, with the table and the wind the settings ask for."""
    from parametric import analysis

    sim = analysis.build_simulation(model, solved)
    if database is not None:
        sim.set_aero_database(database)
    sim.rtol, sim.atol = settings.rtol, settings.atol
    sim.latitude_rad = (
        np.radians(float(settings.latitude_deg))
        if settings.latitude_deg is not None else None
    )
    imperfections = settings.imperfections()
    if imperfections:
        # The same code path a dispersed case goes through, so a nominal
        # misalignment and a sampled one are applied by one rule.
        from trajectory.analysis.dispersion import perturb_simulation

        perturb_simulation(sim, imperfections)
    wind = wind_model_for(settings)
    if wind is not None:
        sim.wind = wind
    return sim


def default_rail_buttons_m(model: VehicleModel, solved=None) -> tuple[float, float]:
    """Where rail buttons usually are: one at the loaded CG, one near the tail.

    The aft button sits 5 cm ahead of the aft end so it is on the body,
    not the nozzle; if the CG is too close to it the forward one moves
    forward to keep the pair a tenth of the length apart.
    """
    from parametric.analysis import loaded_cg_station_m

    start, end = model.station_range_m()
    length = max(end - start, 0.1)
    aft = end - min(0.05, 0.1 * length)
    cg = loaded_cg_station_m(model, solved.cg_station_m if solved is not None else None)
    forward = min(float(cg), aft - 0.1 * length)
    return (float(forward), float(aft))


def default_attachment_station_m(model: VehicleModel) -> float:
    """Where a harness is usually anchored: the nose shoulder.

    The aft end of the first segment of the canonical model -- the nose
    cone's base, where its bulkhead sits. A model with no nose at all
    gives a fifth of its length.
    """
    from parametric import analysis

    try:
        canonical = analysis.to_canonical(model)
        nose = canonical.segments[0]
        if getattr(nose, "kind", "") == "nose":
            return float(nose.start_m + nose.length_m)
    except Exception:  # noqa: BLE001 -- a model too odd to canonicalise
        pass
    start, end = model.station_range_m()
    return float(start + 0.2 * (end - start))


def recovery_for(dry_mass_kg: float, settings: FlightSettings,
                 attachment_station_m: float | None = None):
    """The recovery system the settings describe, sized for this dry mass.

    ``attachment_station_m`` is the harness point to use when the settings
    ask for one and do not name it -- the model's nose shoulder.
    """
    if not settings.use_recovery:
        return None
    from trajectory.vehicle.recovery import standard_recovery

    station = None
    if getattr(settings, "chute_at_attachment", False):
        named = getattr(settings, "chute_attachment_station_m", None)
        station = float(named) if named is not None else attachment_station_m
    return standard_recovery(
        dry_mass_kg=float(dry_mass_kg),
        main_descent_mps=settings.main_descent_mps,
        drogue_descent_mps=settings.drogue_descent_mps,
        main_deploy_altitude_m=settings.main_deploy_altitude_m,
        attachment_station_m=station,
    )


def fly_model(
    model: VehicleModel,
    settings: FlightSettings,
    database=None,
    solved=None,
    aero_settings=None,
    perturb: Callable[[Any], None] | None = None,
    progress: Callable[[str], None] | None = None,
    t_max: float | None = None,
) -> FlightOutcome:
    """Fly the model once under the settings, coupling the drag table if asked.

    Args:
        database: The coefficient table. ``None`` -- or ``use_aero_table``
            off -- flies the simulator's fallback drag law, which has no
            normal force, so the vehicle cannot weathercock.
        solved: Meshed mass properties when a solve has been run; otherwise
            the analytic roll-up and a slender-rod inertia.
        aero_settings: How to rebuild the table when coupling it to altitude.
        perturb: Called with the configured simulation before it flies --
            the dispersion study's hook for dispersing mass and thrust.
            Called before every pass, because every pass builds a fresh
            simulation.
        progress: Told what is happening between coupled passes.
        t_max: A cutoff, for a truncated flight on purpose; ``None`` flies
            until the vehicle is back on the pad.
    """
    using_table = bool(settings.use_aero_table and database is not None)
    attachment = (
        default_attachment_station_m(model)
        if settings.use_recovery and getattr(settings, "chute_at_attachment", False)
        else None
    )
    # Tip-off only with a table. The fallback drag law has no normal force
    # and so no restoring moment: a rate the rail leaves the vehicle with
    # is never arrested, and it pitches over for the whole flight. With
    # the CG constrained instead the attitude is frozen, which is the
    # better proxy the fallback can offer.
    buttons = None
    if using_table and getattr(settings, "rail_buttons", False) and settings.rail_length_m > 0.0:
        named = getattr(settings, "rail_buttons_m", None)
        buttons = tuple(named) if named is not None else default_rail_buttons_m(model, solved)

    def fly(table):
        sim = configure_simulation(model, settings, table, solved)
        if perturb is not None:
            perturb(sim)
        # Sized for the mass the simulation will actually fly -- solved,
        # perturbed or otherwise -- rather than re-read from the model.
        recovery = recovery_for(sim.mass_props.dry_mass, settings, attachment)
        return sim, sim.run(
            launch_azimuth=np.radians(float(settings.azimuth_deg)),
            launch_elevation=np.radians(float(settings.elevation_deg)),
            rail_length_m=float(settings.rail_length_m),
            pad_position_m=settings.pad_position_m(),
            t_max=t_max,
            dt=float(settings.dt_s),
            recovery=recovery,
            rail_buttons_m=buttons,
        )

    def apogee_of(result) -> float:
        altitude = np.asarray(result.y, dtype=float)[1]
        return float(np.max(altitude)) - float(settings.pad_altitude_m)

    table = database if using_table else None
    sim, result = fly(table)
    passes = 0

    if using_table and settings.couple_aero_altitude:
        from parametric import aero, analysis

        aero_settings = copy.copy(aero_settings) if aero_settings is not None \
            else aero.AeroSettings()
        if getattr(aero_settings, "method", None) == "rasaero-app":
            # The out-of-process application cannot be re-run per pass; the
            # in-process engine is its validated equal and can.
            aero_settings.method = "rasaero"

        previous = apogee_of(result)
        for _ in range(COUPLED_PASSES):
            # The flown states are already above sea level, so the profile
            # needs no pad shift.
            samples = analysis.mach_alt_profile(result)
            if not samples:
                break
            if progress is not None:
                progress(f"Coupling drag table to trajectory (pass {passes + 1})...")
            table, _ = aero.run_analysis(model, aero_settings, mach_alt=samples)
            sim, result = fly(table)
            passes += 1
            apogee = apogee_of(result)
            if previous > 0 and abs(apogee - previous) < COUPLED_TOLERANCE * previous:
                break
            previous = apogee

    from parametric.aero import steepest_boattail_deg

    return FlightOutcome(
        result, settings, using_table, passes, table, simulation=sim,
        boattail_half_angle_deg=steepest_boattail_deg(model),
        boattail_model=getattr(aero_settings, "boattail_model", "rasaero"),
    )
