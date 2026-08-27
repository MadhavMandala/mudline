"""Named, bounded parameters.

Every number in a parametric model is a ``Parm`` rather than a bare float. That
sounds like ceremony until you want any of the things it buys:

* **Bounds.** A fin span cannot go negative, a wall thickness cannot exceed the
  radius. Enforcing that where the value lives means no downstream consumer has
  to re-check it, and the GUI gets slider ranges for free.
* **Change tracking.** A component knows when one of its parms moved, so only
  the geometry that actually changed gets rebuilt. Lofting is the expensive
  step; rebuilding everything on every edit makes live dragging impossible.
* **Identity.** A parm has a stable path like ``airframe/nose/length``, which
  is what a design variable, a dispersion sweep or an optimiser needs to refer
  to. ``nlopt`` and ``casadi`` are already installed and unused; this is the
  layer that would let them drive the model.
* **Units and description.** The GUI can label and format a value without
  hardcoding knowledge of what each field means.

This mirrors OpenVSP's Parm/ParmContainer arrangement, which exists for the
same reasons.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator


@dataclass
class Parm:
    """A single named value with bounds."""

    name: str
    value: float
    minimum: float = float("-inf")
    maximum: float = float("inf")
    unit: str = ""
    description: str = ""
    #: Set false for values that must not be swept or optimised.
    designable: bool = True
    #: Which block of the editor this belongs under, e.g. "Geometry".
    #:
    #: Grouping lives on the parm rather than in the editor for the same reason
    #: units and bounds do: the editor is not allowed to know what a rocket is,
    #: so anything it needs in order to lay a component out has to travel with
    #: the value. Empty means one unnamed block, which is what every component
    #: got before this existed.
    group: str = ""

    #: Bounds that depend on other values, resolved against ``owner``.
    #:
    #: A fixed number cannot express "a wall cannot exceed half the diameter",
    #: which is the constraint that actually matters -- the flat 1.0 m cap that
    #: stood here instead was both uselessly loose on a 50 mm tube and wrong in
    #: kind, because the limit is not a length, it is a relationship. Each
    #: returns a bound, or ``None`` to mean "no opinion", which falls back to
    #: the static one. That fallback is what makes them safe on a part that is
    #: not built yet: a Stack with no sections has no diameter to halve.
    min_fn: Callable[[object], float | None] | None = None
    max_fn: Callable[[object], float | None] | None = None
    #: The container holding this parm. Set by ``add_parm``.
    owner: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ValueError(
                f"{self.name}: minimum {self.minimum} exceeds maximum {self.maximum}"
            )
        self.value = self.clamp(float(self.value))

    def _resolve(self, fn, fallback: float) -> float:
        if fn is None or self.owner is None:
            return fallback
        try:
            value = fn(self.owner)
        except Exception:
            return fallback
        if value is None:
            return fallback
        value = float(value)
        return value if value == value and abs(value) != float("inf") else fallback

    @property
    def lower(self) -> float:
        """The minimum in force now, linked bounds included."""
        return self._resolve(self.min_fn, self.minimum)

    @property
    def upper(self) -> float:
        """The maximum in force now, linked bounds included."""
        resolved = self._resolve(self.max_fn, self.maximum)
        # A linked bound that has collapsed below the floor would pin the value
        # rather than constrain it; the static bound is the safer answer.
        return resolved if resolved > self.lower else self.maximum

    def clamp(self, value: float) -> float:
        return min(max(float(value), self.lower), self.upper)

    def set(self, value: float) -> bool:
        """Set the value, clamped. Returns True if it actually changed."""
        new = self.clamp(value)
        if new == self.value:
            return False
        self.value = new
        return True

    @property
    def fraction(self) -> float:
        """Position within the bounds, 0 to 1. Used for slider mapping."""
        low, high = self.lower, self.upper
        if not (low > float("-inf") and high < float("inf")):
            return 0.0
        span = high - low
        return 0.0 if span <= 0 else (self.value - low) / span

    def format(self) -> str:
        text = f"{self.value:.6g}"
        return f"{text} {self.unit}".strip()


class ParmContainer:
    """A named group of parms, with change notification.

    Subclassed by every component. Assigning through ``set`` marks the
    container dirty, which is what lets the rebuild skip untouched geometry.
    """

    def __init__(self, name: str):
        self.name = name
        self._parms: dict[str, Parm] = {}
        self._dirty = True
        self._enforcing = False
        self._listeners: list[Callable[[ParmContainer, str], None]] = []

    # ------------------------------------------------------------------

    def add_parm(
        self,
        name: str,
        value: float,
        minimum: float = float("-inf"),
        maximum: float = float("inf"),
        unit: str = "",
        description: str = "",
        designable: bool = True,
        group: str = "",
        min_fn=None,
        max_fn=None,
    ) -> Parm:
        if name in self._parms:
            raise ValueError(f"{self.name}: duplicate parm {name!r}")
        parm = Parm(name, value, minimum, maximum, unit, description,
                    designable, group, min_fn, max_fn)
        # Set after construction: __post_init__ clamps, and a linked bound
        # cannot be resolved before the parm has an owner to resolve against.
        parm.owner = self
        self._parms[name] = parm
        return parm

    def parm_groups(self) -> list[tuple[str, list[Parm]]]:
        """Parms in declaration order, gathered under their group headings.

        Order follows first appearance rather than anything alphabetical, so a
        component controls how it reads by the order it declares its parms in.
        """
        groups: dict[str, list[Parm]] = {}
        for parm in self._parms.values():
            groups.setdefault(parm.group, []).append(parm)
        return list(groups.items())

    def parm(self, name: str) -> Parm:
        try:
            return self._parms[name]
        except KeyError:
            raise KeyError(
                f"{self.name} has no parm {name!r}. Available: "
                f"{', '.join(sorted(self._parms))}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._parms

    def __iter__(self) -> Iterator[Parm]:
        return iter(self._parms.values())

    def parms(self) -> list[Parm]:
        return list(self._parms.values())

    # ------------------------------------------------------------------

    def get(self, name: str) -> float:
        return self.parm(name).value

    def set(self, name: str, value: float) -> bool:
        """Set a parm and mark this container dirty if it moved."""
        changed = self.parm(name).set(value)
        if changed:
            self.mark_dirty(name)
        return changed

    def update(self, **values: float) -> bool:
        """Set several parms at once. Returns True if any changed."""
        changed = False
        for name, value in values.items():
            changed |= self.set(name, value)
        return changed

    # ------------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    def enforce_links(self) -> None:
        """Linked bounds constrain edits, not state: deliberately a no-op.

        This used to re-clamp every linked parm whenever anything on the
        container changed, and write the clamped value back. That is a
        ratchet. A stack's wall is bounded by half its diameter, and while
        a nose is being generated the first section is a 1.5 mm tip, so a
        3 mm wall was clamped to 0.75 mm on the first section and never
        recovered -- the shipped boattail demonstrator built with half its
        declared wall and half its airframe mass, and the loft agreed with
        the analytic number so nothing caught it. Shrink a tube and grow it
        back and the wall stayed shrunk.

        The bound still applies to what a user or a sweep *sets*, through
        ``Parm.set``, and still drives the editor's slider range. A value a
        later geometry change leaves outside its bound is left where it is:
        the geometry that consumes it already handles that case (a wall
        thicker than the radius simply makes a solid part).
        """
        return

    def mark_dirty(self, reason: str = "") -> None:
        self._dirty = True
        self.enforce_links()
        for listener in self._listeners:
            listener(self, reason)

    def mark_clean(self) -> None:
        self._dirty = False

    def on_change(self, listener: Callable[["ParmContainer", str], None]) -> None:
        self._listeners.append(listener)

    # ------------------------------------------------------------------

    def parm_values(self) -> dict[str, float]:
        return {name: parm.value for name, parm in self._parms.items()}

    def apply_values(self, values: dict[str, float]) -> bool:
        changed = False
        for name, value in values.items():
            if name in self._parms:
                changed |= self.set(name, value)
        return changed
