"""A flight's time history, recovered from the force model.

The integrator carries fourteen states and keeps nothing else. Everything
the force model computed on the way -- thrust, drag, angle of attack, the
centre of pressure, the static margin, the acceleration the airframe felt --
was evaluated at every step and discarded, so the numbers a flight is
actually judged by (max-g, margin at burnout, alpha off the rail) were never
available anywhere. This module re-evaluates the model at every stored
sample. The states are exact and the derived quantities are the same
function of them the integrator used, so nothing here approximates the
flight; it is the flight, looked at.

Recovery phases are replayed from ``result.phases`` so the canopy drag at
each sample is the one that was acting then -- which is where the largest
acceleration of many flights turns out to be.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

G0 = 9.80665


@dataclass
class FlightLog:
    """Derived quantities at every sample of a flown trajectory."""

    time_s: np.ndarray
    phase: list[str]
    altitude_msl_m: np.ndarray
    altitude_agl_m: np.ndarray
    airspeed_mps: np.ndarray
    mach: np.ndarray
    dynamic_pressure_pa: np.ndarray
    alpha_deg: np.ndarray
    thrust_n: np.ndarray
    #: Aerodynamic force along the body axis, opposing motion.
    drag_n: np.ndarray
    #: Aerodynamic force across the body axis.
    normal_force_n: np.ndarray
    chute_cda_m2: np.ndarray
    mass_kg: np.ndarray
    #: From the nose tip, aft positive -- the model's station convention.
    cg_station_m: np.ndarray
    #: NaN where no coefficient table placed one.
    cp_station_m: np.ndarray
    #: (CP - CG) / reference diameter; NaN without a table.
    static_margin_cal: np.ndarray
    #: Non-gravitational acceleration, what an accelerometer reads, in g.
    acceleration_g: np.ndarray
    #: Its component along the body axis, forward positive.
    axial_g: np.ndarray
    #: Body rate about the axis, and the magnitude of the rate across it.
    roll_rate_radps: np.ndarray
    pitch_rate_radps: np.ndarray
    #: The wind the vehicle was flying through, turbulence included.
    wind_mps: np.ndarray
    #: The airframe axis off the vertical: zero nose-up, 180 nose-down.
    axis_from_vertical_deg: np.ndarray
    on_rail: np.ndarray
    pad_altitude_m: float = 0.0
    reference_diameter_m: float = 0.0

    # ------------------------------------------------------------------

    @classmethod
    def from_flight(cls, sim, result, reference_diameter_m: float | None = None):
        """Replay the stored states through the simulation's force model.

        ``sim`` must be the simulation that flew ``result`` -- same engine,
        mass model, wind and coefficient table -- or the log describes a
        different vehicle than the trajectory.
        """
        times = np.asarray(result.t, dtype=float)
        states = np.asarray(result.y, dtype=float).T
        pad_position = getattr(result, "pad_position_m", None)
        pad = float(pad_position[1]) if pad_position is not None else 0.0
        diameter = (
            float(reference_diameter_m) if reference_diameter_m
            else float(np.sqrt(4.0 * float(sim.reference_area) / np.pi))
        )
        axis = np.asarray(sim.thrust_axis_body, dtype=float)
        phases = list(getattr(result, "phases", None) or [])

        n = len(times)
        columns = {
            name: np.zeros(n) for name in (
                "altitude_msl_m", "airspeed_mps", "mach", "dynamic_pressure_pa",
                "alpha_deg", "thrust_n", "drag_n", "normal_force_n",
                "chute_cda_m2", "mass_kg", "cg_station_m", "acceleration_g",
                "axial_g", "roll_rate_radps", "pitch_rate_radps", "wind_mps",
                "axis_from_vertical_deg",
            )
        }
        columns["cp_station_m"] = np.full(n, np.nan)
        columns["static_margin_cal"] = np.full(n, np.nan)
        on_rail = np.zeros(n, dtype=bool)
        phase_names: list[str] = []

        saved = (sim._active_chute, sim._deploy_trigger_s)
        try:
            for i, (t, state) in enumerate(zip(times, states)):
                phase = _phase_at(phases, float(t))
                phase_names.append(phase["name"] if phase else "flight")
                _arm_chute(sim, phase)

                point = sim.evaluate(state, float(t))
                acceleration = point.force_inertial_n / point.mass_kg
                if point.rail_phase == "rail":
                    acceleration = sim.launch_rail.constrain_acceleration(
                        acceleration, state[3:6]
                    )
                elif point.rail_phase == "tipoff":
                    acceleration, _ = sim.tipoff_accelerations(point, state)
                felt = acceleration - point.gravity_inertial_n / point.mass_kg
                axis_inertial = point.dcm_b2i @ axis
                aero_body = point.dcm_b2i.T @ point.aero_force_inertial_n
                along = float(np.dot(aero_body, axis))

                columns["altitude_msl_m"][i] = point.altitude_m
                columns["airspeed_mps"][i] = point.airspeed_mps
                columns["mach"][i] = point.mach
                columns["dynamic_pressure_pa"][i] = point.dynamic_pressure_pa
                columns["alpha_deg"][i] = point.alpha_deg
                columns["thrust_n"][i] = point.thrust_n
                columns["drag_n"][i] = -along
                columns["normal_force_n"][i] = float(np.linalg.norm(aero_body - along * axis))
                columns["chute_cda_m2"][i] = point.chute_cda_m2
                columns["mass_kg"][i] = point.mass_kg
                columns["cg_station_m"][i] = -float(np.dot(point.cg_body_m, axis))
                columns["acceleration_g"][i] = float(np.linalg.norm(felt)) / G0
                columns["axial_g"][i] = float(np.dot(felt, axis_inertial)) / G0
                omega = state[10:13]
                roll = float(np.dot(omega, axis))
                columns["roll_rate_radps"][i] = roll
                columns["pitch_rate_radps"][i] = float(np.linalg.norm(omega - roll * axis))
                columns["wind_mps"][i] = float(np.linalg.norm(point.wind_inertial_mps))
                columns["axis_from_vertical_deg"][i] = float(np.degrees(
                    np.arccos(np.clip(axis_inertial[1], -1.0, 1.0))
                ))
                on_rail[i] = point.on_rail

                aero = point.aero
                if aero is not None:
                    columns["static_margin_cal"][i] = (
                        float(aero.static_margin_m) / diameter if diameter > 0 else np.nan
                    )
                    if aero.cp_body_m is not None:
                        columns["cp_station_m"][i] = -float(np.dot(aero.cp_body_m, axis))
        finally:
            sim._active_chute, sim._deploy_trigger_s = saved

        return cls(
            time_s=times,
            phase=phase_names,
            altitude_agl_m=columns["altitude_msl_m"] - pad,
            on_rail=on_rail,
            pad_altitude_m=pad,
            reference_diameter_m=diameter,
            **columns,
        )

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.time_s)

    def index_at(self, t: float) -> int:
        """Index of the first sample at or after ``t``, clamped to the record."""
        return int(np.clip(np.searchsorted(self.time_s, float(t)), 0, len(self) - 1))

    @property
    def rail_exit_index(self) -> int | None:
        """First sample off the rail, or ``None`` if there was no rail."""
        if not np.any(self.on_rail):
            return None
        off = np.flatnonzero(~self.on_rail)
        off = off[off > int(np.flatnonzero(self.on_rail)[0])]
        return int(off[0]) if len(off) else None

    @property
    def burnout_index(self) -> int | None:
        """First sample after the motor has burned for the last time.

        The last burning sample plus one, so a curve with a gap in it -- a
        dual-thrust motor, a pause -- reports burnout at the end of the burn
        rather than at the first zero. ``None`` if the record ends with the
        motor still burning.
        """
        burning = np.flatnonzero(self.thrust_n > 0.0)
        if not len(burning):
            return None
        last = int(burning[-1])
        return last + 1 if last + 1 < len(self) else None

    @property
    def apogee_index(self) -> int:
        return int(np.argmax(self.altitude_msl_m))

    @property
    def max_acceleration_g(self) -> float:
        return float(np.max(self.acceleration_g)) if len(self) else 0.0

    @property
    def max_acceleration_time_s(self) -> float:
        return float(self.time_s[int(np.argmax(self.acceleration_g))]) if len(self) else 0.0

    def hang_angle_deg(self, window_s: float = 5.0) -> float | None:
        """Mean angle of the axis off the vertical over the last seconds
        of the descent, or ``None`` when the flight did not descend under
        a canopy. Near zero for a vehicle hanging nose-up from its harness."""
        phases = np.asarray(self.phase, dtype=object)
        under = np.isin(phases, ["drogue", "main"])
        if not np.any(under):
            return None
        t_end = float(self.time_s[under][-1])
        window = under & (self.time_s >= t_end - window_s)
        return float(np.mean(self.axis_from_vertical_deg[window]))

    def min_static_margin_cal(self) -> float:
        """Lowest margin flown free between the rail and apogee; NaN without a table."""
        start = self.rail_exit_index or 0
        window = self.static_margin_cal[start:self.apogee_index + 1]
        finite = window[np.isfinite(window)]
        return float(np.min(finite)) if len(finite) else float("nan")

    def columns(self) -> dict[str, np.ndarray]:
        """Every derived column by name, for a CSV; time is the caller's."""
        return {
            "phase": np.asarray(self.phase, dtype=object),
            "altitude_agl_m": self.altitude_agl_m,
            "airspeed_mps": self.airspeed_mps,
            "alpha_deg": self.alpha_deg,
            "thrust_n": self.thrust_n,
            "drag_n": self.drag_n,
            "normal_force_n": self.normal_force_n,
            "chute_cda_m2": self.chute_cda_m2,
            "mass_total_kg": self.mass_kg,
            "cg_station_m": self.cg_station_m,
            "cp_station_m": self.cp_station_m,
            "static_margin_cal": self.static_margin_cal,
            "acceleration_g": self.acceleration_g,
            "axial_g": self.axial_g,
            "roll_rate_radps": self.roll_rate_radps,
            "pitch_rate_radps": self.pitch_rate_radps,
            "wind_mps": self.wind_mps,
            "axis_from_vertical_deg": self.axis_from_vertical_deg,
            "on_rail": self.on_rail.astype(int),
        }

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = self.columns()
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_s", *columns])
            for i, t in enumerate(self.time_s):
                writer.writerow([f"{t:.4f}", *(csv_cell(columns[k][i]) for k in columns)])
        return path


def csv_cell(value) -> str:
    """A CSV cell: text as is, finite numbers compactly, NaN blank."""
    if isinstance(value, str):
        return value
    value = float(value)
    return f"{value:.6g}" if np.isfinite(value) else ""


def _phase_at(phases: list[dict], t: float) -> dict | None:
    """The phase in force at ``t``: the last one that had started."""
    current = None
    for phase in phases:
        if float(phase.get("t_start", 0.0)) <= t + 1e-9:
            current = phase
    return current


def _arm_chute(sim, phase: dict | None) -> None:
    """Put the simulation in the recovery state the phase was flown in."""
    recovery = getattr(sim, "recovery", None)
    name = phase["name"] if phase else ""
    chute = None
    if recovery is not None and name == "drogue":
        chute = recovery.drogue
    elif recovery is not None and name == "main":
        chute = recovery.main
    sim._active_chute = chute
    sim._deploy_trigger_s = float(phase["t_start"]) if (chute is not None and phase) else 0.0
