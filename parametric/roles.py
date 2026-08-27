"""Aerodynamic roles, and Hoerner drag for the things Barrowman cannot describe.

Two problems, one mechanism.

**Inference is brittle.** The solver used to guess which part of the silhouette
was the nose -- "the first station reaching maximum radius". That works until a
vehicle has a payload bulge wider than its nose base, at which point it
silently picks the wrong nose and the centre of pressure goes with it. A
declared role cannot be guessed wrong.

**Barrowman knows four shapes.** Nose, body, transition, fin. Real vehicles
carry rail buttons, launch lugs, camera fairings, antenna covers, cable
conduit. Every one has drag, and a method with no term for them contributes
exactly zero -- which is not conservative, just wrong. Hoerner's *Fluid-Dynamic
Drag* (1965) is the standard catalogue for exactly these, and is what the
``PROTUBERANCE`` role uses.

The role also does the job OpenVSP's Sets do, in the form that matters here:
``INTERNAL`` means the component is real mass but no aerodynamics. A bulkhead,
an avionics sled or a motor casing inside the airframe should weigh something
and be invisible to the air, and there was previously no way to say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AeroRole(str, Enum):
    """What a component is, aerodynamically."""

    AUTO = "auto"
    """Infer from geometry, as before. The default, so nothing changes until a
    role is declared."""

    NOSE = "nose"
    BODY = "body"
    TRANSITION = "transition"
    FIN = "fin"

    PROTUBERANCE = "protuberance"
    """Something sticking into the flow that Barrowman has no term for.
    Drag comes from a Hoerner shape estimate."""

    INTERNAL = "internal"
    """Mass but no aerodynamics: bulkheads, sleds, anything inside the
    airframe."""

    @property
    def is_external(self) -> bool:
        return self is not AeroRole.INTERNAL

    @property
    def label(self) -> str:
        return {
            AeroRole.AUTO: "auto (infer)",
            AeroRole.NOSE: "nose",
            AeroRole.BODY: "body tube",
            AeroRole.TRANSITION: "transition / boattail",
            AeroRole.FIN: "fin set",
            AeroRole.PROTUBERANCE: "protuberance (Hoerner)",
            AeroRole.INTERNAL: "internal (mass only)",
        }[self]


class HoernerShape(str, Enum):
    """Protuberance shapes with a tabulated drag coefficient.

    Coefficients are referenced to the protuberance's own frontal area and are
    the standard values from Hoerner's *Fluid-Dynamic Drag*, chapters 3 and 13
    (drag of surface irregularities and of three-dimensional bodies).
    """

    STREAMLINED_FAIRING = "streamlined_fairing"
    HALF_ROUND = "half_round"
    ROUND_CYLINDER = "round_cylinder"
    SQUARE_BLOCK = "square_block"
    FLAT_PLATE_NORMAL = "flat_plate_normal"
    RAIL_BUTTON = "rail_button"
    LAUNCH_LUG = "launch_lug"
    ANTENNA_WIRE = "antenna_wire"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


#: Base drag coefficient on the protuberance's own frontal area, incompressible.
HOERNER_CD: dict[HoernerShape, float] = {
    # A well-faired body: mostly friction, little pressure drag.
    HoernerShape.STREAMLINED_FAIRING: 0.08,
    # Half-body on a surface; Hoerner gives ~0.4 for a hemisphere-like bump.
    HoernerShape.HALF_ROUND: 0.42,
    # Circular cylinder in crossflow at subcritical Reynolds.
    HoernerShape.ROUND_CYLINDER: 1.17,
    # Sharp-edged rectangular block.
    HoernerShape.SQUARE_BLOCK: 1.05,
    # Flat plate normal to the flow.
    HoernerShape.FLAT_PLATE_NORMAL: 1.28,
    # A rail button is a small, partly-faired stud.
    HoernerShape.RAIL_BUTTON: 0.75,
    # A launch lug is a short tube: open end plus cylinder in crossflow.
    HoernerShape.LAUNCH_LUG: 1.00,
    # Wire or thin rod, per Hoerner's slender-cylinder value.
    HoernerShape.ANTENNA_WIRE: 1.10,
}


@dataclass
class Protuberance:
    """A drag-producing item that is not part of the mould line."""

    name: str
    shape: HoernerShape
    frontal_area_m2: float
    station_m: float
    count: int = 1
    #: Fraction of the local boundary layer the item is buried in. Something
    #: sitting inside a thick boundary layer sees far less than free-stream
    #: dynamic pressure, which is why a rail button aft is cheaper than the
    #: same button near the nose.
    immersion: float = 1.0

    def drag_area_m2(self, mach: float = 0.0) -> float:
        """Effective CD·A, referenced to the protuberance's own frontal area."""
        cd = HOERNER_CD[self.shape]
        cd *= compressibility_factor(mach)
        return cd * self.frontal_area_m2 * self.count * max(self.immersion, 0.0)


def compressibility_factor(mach: float) -> float:
    """Growth of protuberance drag with Mach number.

    Below the critical Mach a protuberance behaves incompressibly. Through the
    transonic range its drag rises sharply as local flow chokes around it, then
    settles supersonically. The shape here follows Hoerner's discussion of
    excrescence drag rather than a Prandtl-Glauert scaling, which is a lift
    correction and does not apply to a bluff body.
    """
    if mach <= 0.7:
        return 1.0
    if mach <= 1.2:
        # Rise to roughly 1.6x through the transonic region.
        return float(1.0 + 0.6 * ((mach - 0.7) / 0.5) ** 2)
    return float(1.6 * (1.2 / mach) ** 0.35)


def boundary_layer_thickness_m(station_m: float, reynolds_per_m: float) -> float:
    """Turbulent boundary-layer thickness at a station, for immersion.

    The 1/7-power estimate: delta ~ 0.37 x / Re_x^0.2. Used to work out how
    much of a small protuberance is actually in the free stream.
    """
    if station_m <= 0 or reynolds_per_m <= 0:
        return 0.0
    reynolds_x = reynolds_per_m * station_m
    return float(0.37 * station_m / max(reynolds_x, 1.0) ** 0.2)


def immersion_factor(height_m: float, boundary_layer_m: float) -> float:
    """Fraction of free-stream dynamic pressure a protuberance of this height sees.

    A protuberance buried in the boundary layer sees a velocity profile, not
    free stream. Integrating the 1/7-power profile over the immersed part gives
    the effective dynamic-pressure fraction; anything taller than the layer sees
    close to free stream over most of its height.
    """
    if height_m <= 0:
        return 0.0
    if boundary_layer_m <= 0:
        return 1.0
    ratio = height_m / boundary_layer_m
    if ratio >= 1.0:
        # Partly immersed: the buried fraction contributes the profile average.
        buried = 1.0 / ratio
        return float((1.0 - buried) + buried * (7.0 / 9.0))
    # Fully immersed: mean of (y/delta)^(2/7) over the height.
    return float(ratio ** (2.0 / 7.0) * 7.0 / 9.0)


def protuberance_drag_coefficient(
    protuberances: list[Protuberance],
    reference_area_m2: float,
    mach: float,
    reynolds_per_m: float = 0.0,
) -> float:
    """Total protuberance drag, referenced to the vehicle's frontal area."""
    if reference_area_m2 <= 0 or not protuberances:
        return 0.0

    total = 0.0
    for item in protuberances:
        immersion = item.immersion
        if reynolds_per_m > 0 and item.frontal_area_m2 > 0:
            # Treat the item as roughly as tall as it is wide.
            height = float(np.sqrt(item.frontal_area_m2))
            delta = boundary_layer_thickness_m(item.station_m, reynolds_per_m)
            immersion = min(immersion, immersion_factor(height, delta))
        total += (
            HOERNER_CD[item.shape]
            * compressibility_factor(mach)
            * item.frontal_area_m2
            * item.count
            * max(immersion, 0.0)
        )
    return float(total / reference_area_m2)


def interference_drag_coefficient(fin_sets, reference_area_m2: float,
                                  mach: float = 0.0) -> float:
    """Drag from fin-body junctions.

    Hoerner gives junction interference as roughly 0.75 t^2 per junction on the
    fin thickness, which is small individually and not negligible across four
    fins. It is the term that makes a thick-rooted fin cost more than its
    planform suggests.
    """
    if reference_area_m2 <= 0:
        return 0.0
    total = 0.0
    for fins in fin_sets:
        thickness = fins.get("thickness")
        total += 0.75 * thickness ** 2 * fins.count
    return float(total * compressibility_factor(mach) / reference_area_m2)
