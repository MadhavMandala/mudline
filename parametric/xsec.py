"""Cross-sections: the primitive the whole geometry model is built from.

The OpenVSP idea, and the reason it is worth stealing: a body is not a nose
plus a tube plus a boattail. It is an ordered stack of cross-sections along a
spine, and the shapes people name -- ogive, cone, transition, boattail, payload
bulge -- are just particular distributions of section size. Modelling it that
way removes every special case. There is no "nose cone type" enumeration to
extend when someone wants a shape it does not contain; you place sections.

Each XSec carries:

* a **station** along the spine,
* a **shape** and its parameters, which need not be circular -- a rounded
  rectangle is how you describe an avionics bay or a rail-button fairing,
* a **tangent** strength controlling how the loft leaves the section, which is
  what distinguishes a smooth shoulder from a sharp break at the same station.

Analytic profiles are kept, but demoted from types to *generators*: a von
Karman nose is a function that emits sections. Once emitted they are ordinary,
individually editable sections, which is exactly OpenVSP's behaviour when you
switch a component to edit-curve mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from parametric.parm import ParmContainer


class XSecShape(str, Enum):
    """Section outline families."""

    POINT = "point"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    SUPER_ELLIPSE = "super_ellipse"
    ROUNDED_RECTANGLE = "rounded_rectangle"


class NoseProfile(str, Enum):
    """Analytic radius distributions used to generate section stacks."""

    CONICAL = "conical"
    OGIVE = "ogive"
    VON_KARMAN = "von_karman"
    ELLIPTICAL = "elliptical"
    POWER_HALF = "power_half"


#: A truly sharp tip has zero wall thickness and a degenerate loft face. Real
#: cones have a machined tip radius, and meshers need one too.
MIN_TIP_RADIUS_M = 0.0015


class XSec(ParmContainer):
    """One cross-section of a stack.

    The section lies in the plane normal to the spine, described in local
    (y, z) coordinates with the spine passing through the origin.
    """

    def __init__(
        self,
        station_m: float,
        shape: XSecShape = XSecShape.CIRCLE,
        width_m: float = 0.1,
        height_m: float | None = None,
        exponent: float = 2.0,
        corner_radius_m: float = 0.0,
        tangent: float = 1.0,
        name: str = "xsec",
    ):
        super().__init__(name)
        self.shape = shape
        # Negative is allowed: the origin is a datum, not the nose tip, and a
        # section forward of it is a legitimate thing to type.
        self.add_parm("station", station_m, -1e5, 1e5, "m",
                      "Position along the spine")
        self.add_parm("width", width_m, 0.0, 1e4, "m", "Full width of the section")
        self.add_parm(
            "height", width_m if height_m is None else height_m, 0.0, 1e4, "m",
            "Full height of the section",
        )
        self.add_parm(
            "exponent", exponent, 0.2, 20.0, "", "Super-ellipse exponent; 2 is an ellipse"
        )
        self.add_parm(
            "corner_radius", corner_radius_m, 0.0, 1e4, "m", "Rounded-rectangle corner",
            # A corner radius of half the width is a fully rounded end; beyond
            # that the outline stops being a rounded rectangle at all.
            max_fn=lambda s: min(s.get("width"), s.get("height")) * 0.5 or None,
        )
        self.add_parm(
            "tangent", tangent, 0.0, 5.0, "", "Loft tangent strength leaving this section"
        )

    # ------------------------------------------------------------------

    @property
    def station_m(self) -> float:
        return self.get("station")

    @property
    def width_m(self) -> float:
        return self.get("width")

    @property
    def height_m(self) -> float:
        return self.get("height")

    @property
    def is_point(self) -> bool:
        return self.shape is XSecShape.POINT or max(self.width_m, self.height_m) <= 0.0

    @property
    def equivalent_radius_m(self) -> float:
        """Radius of a circle with the same area, for aero reference sizing."""
        area = self.area_m2
        return float(np.sqrt(area / np.pi)) if area > 0 else 0.0

    @property
    def area_m2(self) -> float:
        """Enclosed area of the section."""
        a, b = 0.5 * self.width_m, 0.5 * self.height_m
        if a <= 0 or b <= 0:
            return 0.0
        if self.shape is XSecShape.CIRCLE or self.shape is XSecShape.ELLIPSE:
            return float(np.pi * a * b)
        if self.shape is XSecShape.SUPER_ELLIPSE:
            # Exact area of |y/a|^n + |z/b|^n = 1 via the beta function.
            from math import gamma

            n = self.get("exponent")
            return float(4.0 * a * b * gamma(1.0 + 1.0 / n) ** 2 / gamma(1.0 + 2.0 / n))
        if self.shape is XSecShape.ROUNDED_RECTANGLE:
            r = min(self.get("corner_radius"), a, b)
            return float(4.0 * a * b - (4.0 - np.pi) * r * r)
        return 0.0

    # ------------------------------------------------------------------

    def outline(self, points: int = 96) -> np.ndarray:
        """Return (n, 2) local (y, z) points tracing the section, closed.

        A point section returns a single coordinate; the loft turns that into a
        blunted tip rather than a degenerate face.
        """
        if self.is_point:
            return np.zeros((1, 2))

        a, b = 0.5 * self.width_m, 0.5 * self.height_m
        theta = np.linspace(0.0, 2.0 * np.pi, points, endpoint=False)

        if self.shape in (XSecShape.CIRCLE, XSecShape.ELLIPSE):
            return np.column_stack([a * np.cos(theta), b * np.sin(theta)])

        if self.shape is XSecShape.SUPER_ELLIPSE:
            n = self.get("exponent")
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            y = a * np.sign(cos_t) * np.abs(cos_t) ** (2.0 / n)
            z = b * np.sign(sin_t) * np.abs(sin_t) ** (2.0 / n)
            return np.column_stack([y, z])

        if self.shape is XSecShape.ROUNDED_RECTANGLE:
            return _rounded_rectangle(a, b, min(self.get("corner_radius"), a, b), points)

        raise ValueError(f"Unhandled section shape {self.shape!r}")

    def describe(self) -> str:
        if self.is_point:
            return f"{self.station_m:7.3f} m   point"
        if self.shape is XSecShape.CIRCLE:
            return f"{self.station_m:7.3f} m   circle  d={self.width_m:.3f}"
        return (
            f"{self.station_m:7.3f} m   {self.shape.value}  "
            f"{self.width_m:.3f} x {self.height_m:.3f}"
        )


def _rounded_rectangle(a: float, b: float, radius: float, points: int) -> np.ndarray:
    """Trace a rounded rectangle of half-width a, half-height b."""
    radius = max(0.0, min(radius, a, b))
    if radius <= 1e-12:
        corners = np.array([[a, b], [-a, b], [-a, -b], [a, -b]])
        return _resample_closed(corners, points)

    per_corner = max(3, points // 4)
    arcs = []
    centres = [(a - radius, b - radius), (-(a - radius), b - radius),
               (-(a - radius), -(b - radius)), (a - radius, -(b - radius))]
    starts = [0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    for (cy, cz), start in zip(centres, starts):
        angles = np.linspace(start, start + 0.5 * np.pi, per_corner)
        arcs.append(np.column_stack([cy + radius * np.cos(angles),
                                     cz + radius * np.sin(angles)]))
    return np.vstack(arcs)


def _resample_closed(polygon: np.ndarray, points: int) -> np.ndarray:
    """Evenly resample a closed polygon to a fixed point count."""
    closed = np.vstack([polygon, polygon[:1]])
    segments = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    lengths = np.concatenate([[0.0], np.cumsum(segments)])
    total = lengths[-1]
    if total <= 0:
        return np.repeat(polygon[:1], points, axis=0)
    targets = np.linspace(0.0, total, points, endpoint=False)
    y = np.interp(targets, lengths, closed[:, 0])
    z = np.interp(targets, lengths, closed[:, 1])
    return np.column_stack([y, z])


# ----------------------------------------------------------------------
# Analytic profiles, demoted to section generators
# ----------------------------------------------------------------------


def nose_radius(profile: NoseProfile, length: float, radius: float,
                stations: np.ndarray) -> np.ndarray:
    """Radius of an analytic nose profile at the given stations from the tip."""
    z = np.clip(np.asarray(stations, dtype=float), 0.0, length)
    if length <= 0 or radius <= 0:
        return np.zeros_like(z)
    fraction = z / length

    if profile is NoseProfile.CONICAL:
        return radius * fraction
    if profile is NoseProfile.ELLIPTICAL:
        return radius * np.sqrt(np.clip(1.0 - (1.0 - fraction) ** 2, 0.0, None))
    if profile is NoseProfile.POWER_HALF:
        return radius * np.sqrt(fraction)
    if profile is NoseProfile.VON_KARMAN:
        theta = np.arccos(np.clip(1.0 - 2.0 * fraction, -1.0, 1.0))
        return (radius / np.sqrt(np.pi)) * np.sqrt(
            np.clip(theta - 0.5 * np.sin(2.0 * theta), 0.0, None)
        )
    # Tangent ogive.
    rho = (radius ** 2 + length ** 2) / (2.0 * radius)
    inner = np.clip(rho ** 2 - (length - z) ** 2, 0.0, None)
    return np.sqrt(inner) + radius - rho


def generate_nose_sections(
    profile: NoseProfile,
    length_m: float,
    base_diameter_m: float,
    sections: int = 14,
    tip_radius_m: float = MIN_TIP_RADIUS_M,
    start_station_m: float = 0.0,
) -> list[XSec]:
    """Emit a stack of sections tracing an analytic nose profile.

    Sections are clustered toward the tip, where curvature is highest, using a
    cosine distribution. Uniform spacing wastes sections on the nearly
    cylindrical aft end and under-resolves the tip, which is exactly where the
    lofted surface departs from the intended profile.
    """
    if sections < 3:
        raise ValueError("A nose needs at least three sections.")

    # Cosine clustering: dense at the tip, sparse at the base.
    beta = np.linspace(0.0, 0.5 * np.pi, sections)
    fractions = 1.0 - np.cos(beta)
    stations = fractions * length_m

    radius = 0.5 * base_diameter_m
    radii = np.maximum(nose_radius(profile, length_m, radius, stations),
                       max(tip_radius_m, MIN_TIP_RADIUS_M))

    out: list[XSec] = []
    for index, (station, r) in enumerate(zip(stations, radii)):
        out.append(XSec(
            station_m=float(start_station_m + station),
            shape=XSecShape.CIRCLE,
            width_m=float(2.0 * r),
            name=f"nose_{index}",
        ))
    return out


def generate_tube_sections(
    length_m: float,
    diameter_m: float,
    start_station_m: float,
    name: str = "tube",
) -> list[XSec]:
    """Two sections describing a constant-diameter run."""
    return [
        XSec(start_station_m, XSecShape.CIRCLE, diameter_m, name=f"{name}_fwd"),
        XSec(start_station_m + length_m, XSecShape.CIRCLE, diameter_m, name=f"{name}_aft"),
    ]


def generate_transition_sections(
    length_m: float,
    front_diameter_m: float,
    rear_diameter_m: float,
    start_station_m: float,
    sections: int = 6,
    name: str = "transition",
) -> list[XSec]:
    """A conical transition or boattail as a short run of sections."""
    stations = np.linspace(0.0, length_m, sections)
    diameters = np.linspace(front_diameter_m, rear_diameter_m, sections)
    return [
        XSec(float(start_station_m + s), XSecShape.CIRCLE, float(d), name=f"{name}_{i}")
        for i, (s, d) in enumerate(zip(stations, diameters))
    ]
