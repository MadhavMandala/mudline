from __future__ import annotations

import pint
from functools import lru_cache


@lru_cache(maxsize=1)
def get_ureg() -> pint.UnitRegistry:
    """Return a cached pint UnitRegistry to avoid repeated instantiation cost."""
    return pint.UnitRegistry()


def convert_to_internal(value: float, from_unit: str) -> float:
    """Convert a value from the given unit to the internal unit (inch or lbm)."""
    ureg = get_ureg()
    return ureg(value, from_unit).to(ureg.inch).magnitude


def convert_from_internal(value: float, to_unit: str) -> float:
    """Convert a value from the internal unit (inch or lbm) to the requested unit."""
    ureg = get_ureg()
    return ureg(value, ureg.inch).to(to_unit).magnitude


def convert_mass_to_internal(value: float, from_unit: str) -> float:
    ureg = get_ureg()
    return ureg(value, from_unit).to(ureg.pound).magnitude


def convert_mass_from_internal(value: float, to_unit: str) -> float:
    ureg = get_ureg()
    return ureg(value, ureg.pound).to(to_unit).magnitude


def convert_density_to_internal(value: float, from_unit: str) -> float:
    ureg = get_ureg()
    return ureg(value, from_unit).to(ureg.pound / ureg.inch**3).magnitude


def convert_density_from_internal(value: float, to_unit: str) -> float:
    ureg = get_ureg()
    return ureg(value, ureg.pound / ureg.inch**3).to(to_unit).magnitude


def convert_inertia_to_internal(value: float, from_unit: str) -> float:
    ureg = get_ureg()
    return ureg(value, from_unit).to(ureg.pound * ureg.inch**2).magnitude


def convert_inertia_from_internal(value: float, to_unit: str) -> float:
    ureg = get_ureg()
    return ureg(value, ureg.pound * ureg.inch**2).to(to_unit).magnitude


def scale_factor_to_inches(native_unit: str) -> float:
    """Return the multiplicative factor to convert native_unit to inches."""
    ureg = get_ureg()
    return ureg(native_unit).to(ureg.inch).magnitude
