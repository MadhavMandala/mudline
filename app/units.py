"""Unit display.

Rocketry in the United States is inches and pounds, and a tool that only speaks
SI will be fought with rather than used. But converting *stored* values would
be a mistake: every solver, every geometry routine and every file on disk works
in SI, and a model whose numbers depend on a display preference is a model that
produces different answers depending on a menu.

So this converts at the boundary only. Values are stored, computed and saved in
SI always; the unit system changes what is shown and how typed input is
interpreted, and nothing else.

Lengths
-------
One subtlety worth stating. "m" maps to inches for component dimensions and to
feet for flight distances, because that is what people actually use: a fin span
in feet and an apogee in inches are both useless. The choice is made from the
magnitude, and a metric keeps one unit across a comparison so two runs stay
directly readable side by side.
"""

from __future__ import annotations

from enum import Enum


class UnitSystem(str, Enum):
    SI = "si"
    IMPERIAL = "imperial"

    @property
    def label(self) -> str:
        return "SI (m, kg, N)" if self is UnitSystem.SI else "Imperial (in, lb, psi)"


#: unit -> (imperial unit, factor from SI)
CONVERSIONS: dict[str, tuple[str, float]] = {
    "m": ("in", 39.37007874015748),
    "mm": ("in", 0.03937007874015748),
    "m²": ("in²", 1550.0031000062),
    "m2": ("in²", 1550.0031000062),
    "m³": ("in³", 61023.744094732),
    "m/s": ("ft/s", 3.280839895013123),
    "kg": ("lb", 2.2046226218487757),
    "N": ("lbf", 0.2248089430997105),
    "N·s": ("lbf·s", 0.2248089430997105),
    "N.s": ("lbf·s", 0.2248089430997105),
    "Pa": ("psi", 1.4503773800722e-4),
    "kPa": ("psi", 0.14503773800722),
    "kg/m³": ("lb/in³", 3.6127292000084e-5),
    "kg·m²": ("lb·in²", 3417.171893129),
    "N·m": ("lbf·ft", 0.7375621492772654),
}

#: Above this many metres, a length reads better in feet than inches.
FEET_THRESHOLD_M = 10.0
METRES_TO_FEET = 3.280839895013123


class Units:
    """The active unit system, and the conversions that follow from it."""

    def __init__(self, system: UnitSystem = UnitSystem.SI):
        self.system = system

    @property
    def imperial(self) -> bool:
        return self.system is UnitSystem.IMPERIAL

    # ------------------------------------------------------------------

    def display(self, value: float, unit: str,
                prefer_feet: bool | None = None) -> tuple[float, str]:
        """Convert an SI value for display. Returns (value, unit)."""
        if not self.imperial or not unit:
            return value, unit

        if unit == "m":
            use_feet = (
                abs(value) >= FEET_THRESHOLD_M if prefer_feet is None else prefer_feet
            )
            if use_feet:
                return value * METRES_TO_FEET, "ft"
            return value * CONVERSIONS["m"][1], "in"

        converted = CONVERSIONS.get(unit)
        if converted is None:
            return value, unit
        imperial_unit, factor = converted
        return value * factor, imperial_unit

    def to_si(self, value: float, unit: str,
              prefer_feet: bool | None = None) -> float:
        """Interpret a displayed value back into SI, for typed input."""
        if not self.imperial or not unit:
            return value

        if unit == "m":
            if prefer_feet:
                return value / METRES_TO_FEET
            return value / CONVERSIONS["m"][1]

        converted = CONVERSIONS.get(unit)
        if converted is None:
            return value
        return value / converted[1]

    def factor(self, unit: str, prefer_feet: bool | None = None) -> float:
        """Multiplier from SI to display, for scaling a whole series."""
        one, _ = self.display(1.0, unit, prefer_feet=prefer_feet)
        return one

    def scale_for(self, value: float, unit: str) -> float:
        """The factor consistent with how *this* value would be displayed.

        Necessary because the metre rule is magnitude-dependent. Asking
        ``factor("m")`` answers for 1.0 m, which is inches -- so a difference
        between two apogees would come out in inches while the apogees
        themselves were shown in feet.
        """
        prefer = (
            abs(value) >= FEET_THRESHOLD_M if unit == "m" else None
        )
        return self.factor(unit, prefer_feet=prefer)

    def unit_label(self, unit: str, prefer_feet: bool | None = None) -> str:
        return self.display(1.0, unit, prefer_feet=prefer_feet)[1]

    def format(self, value: float, unit: str, decimals: int = 3,
               prefer_feet: bool | None = None) -> str:
        shown, shown_unit = self.display(value, unit, prefer_feet=prefer_feet)
        return f"{shown:,.{decimals}f} {shown_unit}".strip()


#: One instance, shared. A unit preference is a property of the session rather
#: than of any one widget, and threading it through every constructor would add
#: a parameter to code that has no other reason to know about units.
UNITS = Units()


def set_system(system: UnitSystem) -> None:
    UNITS.system = system
