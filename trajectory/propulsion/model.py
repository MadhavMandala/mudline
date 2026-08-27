"""Saved propulsion model format for trajectory simulations."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from trajectory.vehicle.engine import (
    THRUST_REFERENCES,
    Engine,
    nozzle_area_for_isp_pair,
)


PROPULSION_FORMAT = "mudline.propulsion.v1"
#: Accepted on load as well: files written before the rename.
LEGACY_PROPULSION_FORMATS = ("rocket_model.propulsion.v1",)


@dataclass(frozen=True)
class PropulsionModel:
    """Trajectory-ready propulsion model."""

    name: str
    system_type: str
    time_s: np.ndarray
    thrust_n: np.ndarray
    isp_vac_s: float
    isp_sl_s: float
    nozzle_area_m2: float
    propellant_mass_kg: float
    # What ambient condition ``thrust_n`` was measured or computed at. Files
    # written before this field existed are read as "vacuum", which is what the
    # solver already assumed -- so old models keep their previous behaviour.
    thrust_reference: str = "vacuum"
    dry_mass_kg: float = 0.0
    fuel: str = ""
    oxidizer: str = ""
    pressurant: str = ""
    mixture_ratio: float = 0.0
    chamber_pressure_pa: float = 0.0
    exit_pressure_pa: float = 0.0
    tank_cg_m: np.ndarray = field(default_factory=lambda: np.array([0.0, -2.0, 0.0]))
    thrust_axis_body: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    thrust_position_body_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    max_gimbal_deg: float = 0.0
    notes: str = ""
    path: Path | None = None

    @property
    def burn_time_s(self) -> float:
        active = np.asarray(self.thrust_n) > 0.0
        if np.any(active):
            return float(np.max(np.asarray(self.time_s)[active]))
        return 0.0

    @property
    def peak_thrust_n(self) -> float:
        return float(np.max(self.thrust_n)) if len(self.thrust_n) else 0.0

    @property
    def total_impulse_ns(self) -> float:
        if len(self.time_s) < 2:
            return 0.0
        return float(np.trapezoid(self.thrust_n, self.time_s))

    @property
    def average_thrust_n(self) -> float:
        burn_time = self.burn_time_s
        return self.total_impulse_ns / burn_time if burn_time > 0 else 0.0

    @property
    def consistent_nozzle_area_m2(self) -> float:
        """Exit area implied by the declared vacuum/sea-level Isp pair."""
        return nozzle_area_for_isp_pair(
            self.peak_thrust_n, self.isp_vac_s, self.isp_sl_s
        )

    def nozzle_area_consistency(self) -> tuple[float, float, float]:
        """Return (declared, implied, relative_error) for the exit area.

        The model is over-specified: a vacuum thrust curve plus ``isp_vac`` and
        ``nozzle_area_m2`` already determine sea-level performance, so a
        separately declared ``isp_sl_s`` can disagree with them. A large error
        here means one of the three numbers is wrong.
        """
        declared = float(self.nozzle_area_m2)
        implied = self.consistent_nozzle_area_m2
        if implied <= 0:
            return declared, implied, 0.0
        return declared, implied, abs(declared - implied) / implied

    def to_engine(self) -> Engine:
        _, implied, error = self.nozzle_area_consistency()
        if implied > 0 and error > 0.25:
            warnings.warn(
                f"Nozzle exit area {self.nozzle_area_m2:.4f} m^2 disagrees with "
                f"the declared Isp pair ({self.isp_vac_s:.1f}/{self.isp_sl_s:.1f} s), "
                f"which implies {implied:.4f} m^2 -- a {error * 100:.0f}% error. "
                "Sea-level thrust will be wrong; check the exit area or the Isp values.",
                stacklevel=2,
            )

        engine = Engine(
            thrust_curve=np.asarray(self.thrust_n, dtype=float),
            time_points=np.asarray(self.time_s, dtype=float),
            isp_vac=float(self.isp_vac_s),
            isp_sl=float(self.isp_sl_s),
            nozzle_area=float(self.nozzle_area_m2),
            thrust_reference=str(self.thrust_reference),
        )
        engine.max_gimbal = np.radians(max(0.0, float(self.max_gimbal_deg)))
        return engine

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": PROPULSION_FORMAT,
            "name": self.name,
            "system_type": self.system_type,
            "propellants": {
                "fuel": self.fuel,
                "oxidizer": self.oxidizer,
                "pressurant": self.pressurant,
                "mixture_ratio": self.mixture_ratio,
                "propellant_mass_kg": self.propellant_mass_kg,
                "dry_mass_kg": self.dry_mass_kg,
            },
            "performance": {
                "isp_vac_s": self.isp_vac_s,
                "isp_sl_s": self.isp_sl_s,
                "nozzle_area_m2": self.nozzle_area_m2,
                "thrust_reference": self.thrust_reference,
                "chamber_pressure_pa": self.chamber_pressure_pa,
                "exit_pressure_pa": self.exit_pressure_pa,
                "max_gimbal_deg": self.max_gimbal_deg,
            },
            "geometry": {
                "tank_cg_m": np.asarray(self.tank_cg_m, dtype=float).tolist(),
                "thrust_axis_body": np.asarray(self.thrust_axis_body, dtype=float).tolist(),
                "thrust_position_body_m": np.asarray(self.thrust_position_body_m, dtype=float).tolist(),
            },
            "thrust_curve": [
                {"time_s": float(t), "thrust_n": float(thrust)}
                for t, thrust in zip(self.time_s, self.thrust_n)
            ],
            "notes": self.notes,
        }


def default_propulsion_model() -> PropulsionModel:
    return PropulsionModel(
        name="Default Solid Motor",
        system_type="Solid Rocket Motor",
        time_s=np.array([0.0, 30.0, 31.0, 120.0]),
        thrust_n=np.array([20000.0, 20000.0, 0.0, 0.0]),
        isp_vac_s=280.0,
        isp_sl_s=250.0,
        # ~0.0211 m^2 (164 mm exit), consistent with the Isp pair above. The
        # previous 0.15 m^2 default described a 437 mm exit on a 300 mm vehicle.
        nozzle_area_m2=nozzle_area_for_isp_pair(20000.0, 280.0, 250.0),
        propellant_mass_kg=150.0,
        thrust_reference="vacuum",
        dry_mass_kg=0.0,
        fuel="Composite propellant",
        tank_cg_m=np.array([0.0, -2.0, 0.0]),
    )


def load_propulsion_model(path: str | Path) -> PropulsionModel:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("format") not in (PROPULSION_FORMAT, *LEGACY_PROPULSION_FORMATS):
        raise ValueError("Not a supported propulsion model file.")

    propellants = data.get("propellants", {})
    performance = data.get("performance", {})
    geometry = data.get("geometry", {})
    curve = data.get("thrust_curve", [])
    if len(curve) < 2:
        raise ValueError("Propulsion model must include at least two thrust-curve points.")

    time_s = np.array([float(row["time_s"]) for row in curve], dtype=float)
    thrust_n = np.array([float(row["thrust_n"]) for row in curve], dtype=float)
    order = np.argsort(time_s)
    time_s = time_s[order]
    thrust_n = thrust_n[order]

    model = PropulsionModel(
        name=str(data.get("name") or path.stem),
        system_type=str(data.get("system_type") or "Other"),
        time_s=time_s,
        thrust_n=np.maximum(thrust_n, 0.0),
        isp_vac_s=float(performance.get("isp_vac_s", 0.0)),
        isp_sl_s=float(performance.get("isp_sl_s", 0.0)),
        nozzle_area_m2=float(performance.get("nozzle_area_m2", 0.0)),
        propellant_mass_kg=float(propellants.get("propellant_mass_kg", 0.0)),
        # Absent in files written before the field existed. "vacuum" is what the
        # solver already assumed, so those models are read back unchanged.
        thrust_reference=str(performance.get("thrust_reference", "vacuum")),
        dry_mass_kg=float(propellants.get("dry_mass_kg", 0.0)),
        fuel=str(propellants.get("fuel") or ""),
        oxidizer=str(propellants.get("oxidizer") or ""),
        pressurant=str(propellants.get("pressurant") or ""),
        mixture_ratio=float(propellants.get("mixture_ratio", 0.0)),
        chamber_pressure_pa=float(performance.get("chamber_pressure_pa", 0.0)),
        exit_pressure_pa=float(performance.get("exit_pressure_pa", 0.0)),
        tank_cg_m=np.array(geometry.get("tank_cg_m", [0.0, -2.0, 0.0]), dtype=float),
        thrust_axis_body=_normalize_axis(
            np.array(geometry.get("thrust_axis_body", [0.0, 1.0, 0.0]), dtype=float)
        ),
        thrust_position_body_m=np.array(
            geometry.get("thrust_position_body_m", [0.0, 0.0, 0.0]), dtype=float
        ),
        max_gimbal_deg=float(performance.get("max_gimbal_deg", 0.0)),
        notes=str(data.get("notes") or ""),
        path=path,
    )
    _validate_model(model)
    return model


def save_propulsion_model(model: PropulsionModel, path: str | Path) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".json":
        path = path.with_suffix(".propulsion.json")
    if path.name.endswith(".json") and not path.name.endswith(".propulsion.json"):
        path = path.with_name(f"{path.stem}.propulsion.json")

    _validate_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(model.to_dict(), f, indent=2)
    return path


def _validate_model(model: PropulsionModel) -> None:
    if len(model.time_s) != len(model.thrust_n) or len(model.time_s) < 2:
        raise ValueError("Thrust curve must contain at least two matching time/thrust points.")
    if np.any(np.diff(model.time_s) < 0):
        raise ValueError("Thrust curve time values must be sorted.")
    if model.isp_vac_s <= 0 or model.isp_sl_s <= 0:
        raise ValueError("Specific impulse values must be greater than zero.")
    if model.nozzle_area_m2 < 0:
        raise ValueError("Nozzle area cannot be negative.")
    if model.propellant_mass_kg < 0 or model.dry_mass_kg < 0:
        raise ValueError("Mass values cannot be negative.")
    if model.thrust_reference not in THRUST_REFERENCES:
        raise ValueError(
            f"thrust_reference must be one of {THRUST_REFERENCES}; "
            f"got {model.thrust_reference!r}"
        )
    if model.isp_sl_s > model.isp_vac_s:
        raise ValueError(
            f"Sea-level Isp ({model.isp_sl_s} s) cannot exceed vacuum Isp "
            f"({model.isp_vac_s} s)."
        )


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        return np.array([0.0, 1.0, 0.0])
    return axis / norm
