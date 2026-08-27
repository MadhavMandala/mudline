"""Analysis pipeline state.

The chain has a real order -- mass, then aerodynamics, then flight -- and
before this nothing said so. Worse, nothing tracked staleness: you could run
the aerodynamics, drag a fin span, run a flight, and it would fly with
coefficients computed for a *different vehicle* without a word. That is not a
usability wrinkle, it is silently wrong results.

The fix is a fingerprint. Each stage records what the model looked like when it
ran; if the model no longer matches, the stage is stale and says so. Downstream
stages depend on upstream ones, so a geometry edit invalidates everything.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum


class StageState(str, Enum):
    NOT_RUN = "not run"
    CURRENT = "current"
    STALE = "stale"

    @property
    def symbol(self) -> str:
        return {"not run": "·", "current": "✓", "stale": "!"}[self.value]


def model_fingerprint(model) -> str:
    """A hash of everything an analysis depends on.

    Deliberately covers roles and materials as well as dimensions: switching a
    component to *internal* removes it from the aerodynamics, and changing a
    material moves the centre of gravity. Both must invalidate a result even
    though neither moves a single vertex.
    """
    payload: list = []
    for component in model.walk():
        entry = {
            "path": component.path,
            "kind": component.kind,
            "material": component.material,
            "role": component.aero_role.value,
            "parms": {name: round(value, 10)
                      for name, value in component.parm_values().items()},
        }
        sections = getattr(component, "sections", None)
        if sections is not None:
            entry["sections"] = [
                [s.shape.value, round(s.station_m, 10),
                 round(s.width_m, 10), round(s.height_m, 10)]
                for s in sorted(sections, key=lambda x: x.station_m)
            ]
        curve = getattr(component, "curve", None)
        if curve is not None:
            entry["curve"] = [[round(t, 9), round(f, 6)] for t, f in curve]
        shape = getattr(component, "shape", None)
        if shape is not None and component.kind == "protuberance":
            entry["shape"] = shape.value
        payload.append(entry)

    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Stage:
    """One step of the analysis chain."""

    key: str
    label: str
    depends_on: tuple[str, ...] = ()
    fingerprint: str | None = None
    detail: str = ""

    def state(self, current: str, upstream_stale: bool) -> StageState:
        if self.fingerprint is None:
            return StageState.NOT_RUN
        if self.fingerprint != current or upstream_stale:
            return StageState.STALE
        return StageState.CURRENT


class Pipeline:
    """Tracks which analyses are current for the model in front of you."""

    def __init__(self):
        self.stages: dict[str, Stage] = {
            "mass": Stage("mass", "Mass properties"),
            "aero": Stage("aero", "Aerodynamics"),
            "flight": Stage("flight", "Flight", depends_on=("aero",)),
            "dispersion": Stage("dispersion", "Dispersion", depends_on=("aero",)),
        }

    def record(self, key: str, model, detail: str = "") -> None:
        stage = self.stages[key]
        stage.fingerprint = model_fingerprint(model)
        stage.detail = detail

    def clear(self) -> None:
        for stage in self.stages.values():
            stage.fingerprint = None
            stage.detail = ""

    def state(self, key: str, model) -> StageState:
        current = model_fingerprint(model)
        return self._state_with(key, current)

    def _state_with(self, key: str, current: str) -> StageState:
        stage = self.stages[key]
        upstream_stale = any(
            self._state_with(name, current) is not StageState.CURRENT
            and self.stages[name].fingerprint is not None
            for name in stage.depends_on
        )
        return stage.state(current, upstream_stale)

    def states(self, model) -> dict[str, StageState]:
        current = model_fingerprint(model)
        return {key: self._state_with(key, current) for key in self.stages}

    #: Short names for the status bar, where space is tight.
    SHORT = {"mass": "mass", "aero": "aero", "flight": "flight",
             "dispersion": "disp"}

    def summary(self, model) -> str:
        """One line for the status bar."""
        states = self.states(model)
        return "   ".join(
            f"{states[key].symbol} {self.SHORT[key]}"
            for key in ("mass", "aero", "flight")
        )

    def stale_warning(self, key: str, model) -> str | None:
        """A sentence to show before using a stale result, or None."""
        state = self.state(key, model)
        if state is StageState.CURRENT:
            return None
        if state is StageState.NOT_RUN:
            return f"{self.stages[key].label} has not been run for this vehicle."
        return (
            f"{self.stages[key].label} is stale: the model has changed since it "
            f"ran. Re-run it, or the result will describe a different vehicle."
        )
