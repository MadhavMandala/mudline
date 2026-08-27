"""Pass A: the geometry RASAero derives once per design and caches.

Transcription policy
--------------------
Every formula here is transcribed **literally** from the original --
same operation order, same constants, no algebraic simplification. Two reasons:

1. The mandate is bug-for-bug agreement. Simplifying ``a/b*b`` to ``a`` changes
   the last bits, and the acceptance bar is per-term agreement to RASAero's
   printed precision across 2500 Mach points.
2. Review is mechanical. A reader with ``i.cs`` open can check a line at a
   time, which matters because the failure mode in this port is not mistyped
   numbers -- it is *parameter identity*: which variable feeds which formula.

RASAero's two pis
-----------------
The binary uses the literal ``3.14159`` in most area work and ``Math.PI`` in
some of the same expressions. This is not a rounding artefact to tidy up: the
reference area itself is computed with ``3.14159`` (i.cs:1071) while the public
accessor for the same quantity uses ``Math.PI`` (i.cs:419), so the program
holds two values of A_ref that differ by 8.4e-7 relative. Both are reproduced,
and which one appears where is load-bearing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .parts import (
    Design,
    Fins,
    NoseShape,
    Part,
    PartType,
)

#: RASAero's literal pi. Distinct from math.pi on purpose.
PI_RA = 3.14159

#: Degrees-to-radians as RASAero computes it: 3.14159/180, not math.pi/180.
DEG_TO_RAD_RA = 0.017453277777777776
RAD_TO_DEG_RA = 57.295827908797776

#: The body-shaped part types. Fins and protuberances are excluded from the
#: reference-length sum and from the base-diameter search.
_BODY_TYPES = frozenset({
    PartType.NOSE_CONE,
    PartType.BODY_TUBE,
    PartType.EXPANSION,
    PartType.REDUCER,
})


@dataclass
class PartGeometry:
    """Per-part derived geometry (the ``g``, ``v``, ``u`` fields of ``aq``)."""

    wetted_over_aref: float = 0.0   # aq.g -- already normalised by A_ref
    planform_area: float = 0.0      # aq.v, in^2
    planform_centroid: float = 0.0  # aq.u, in from nose tip


@dataclass
class GeometryCache:
    """Everything Pass A produces. Invalidated by any design change."""

    l_body: float = 0.0        # be
    d_ref: float = 0.0         # cd
    a_ref: float = 0.0         # ce  (uses PI_RA)
    d_nose: float = 0.0        # cc
    d_base: float = 0.0        # cb
    boattail_angle_deg: float = 0.0   # db
    boattail_fwd_diameter: float = 0.0  # dc

    wetted_over_aref: float = 0.0     # cf
    planform_area: float = 0.0        # cg
    planform_centroid: float = 0.0    # ch

    per_part: dict[int, PartGeometry] = field(default_factory=dict)
    #: Equivalent-volume body diameter under each fin set, by part index (aq.ai).
    fin_equiv_diameter: dict[int, float] = field(default_factory=dict)
    #: Nose-shock standoff for each fin set, by part index (az.ag).
    fin_shock_standoff: dict[int, float] = field(default_factory=dict)
    #: Nose-lengths aft of the nose, by part index (az.af).
    fin_nose_offset: dict[int, float] = field(default_factory=dict)
    #: 1-based fin set ordinal, by part index (aq.al). Used only for messages.
    fin_ordinal: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# A.1 -- reference quantities
# ---------------------------------------------------------------------------


def _reference_quantities(design: Design, cache: GeometryCache) -> None:
    parts = design.parts

    # i.cs:1021-1038. The boattail angle and forward diameter are assigned
    # inside this loop and reset to zero by every non-Reducer body part, so
    # they describe the LAST body part only -- and are zero unless that part
    # is itself a Reducer. A reducer followed by a body tube leaves them zero.
    l_body = 0.0
    bt_angle = 0.0
    bt_fwd = 0.0
    for part in parts:
        if part.part_type in _BODY_TYPES:
            l_body += part.length
            if part.part_type is PartType.REDUCER:
                bt_fwd = part.d_fwd
                bt_angle = math.atan(
                    ((part.d_fwd - part.d_aft) / 2.0) / part.length
                ) * (180.0 / math.pi)
            else:
                bt_fwd = 0.0
                bt_angle = 0.0
    cache.l_body = l_body
    cache.boattail_angle_deg = bt_angle
    cache.boattail_fwd_diameter = bt_fwd

    # i.cs:1039-1050. Max over BOTH diameters of EVERY part, fins included --
    # a fin set carries the body diameters at its station.
    d_ref = 0.0
    for part in parts:
        if part.d_fwd > d_ref:
            d_ref = part.d_fwd
        if part.d_aft > d_ref:
            d_ref = part.d_aft

    # i.cs:1051-1063. Aft diameter of the last body-shaped part.
    d_base = 0.0
    for part in parts:
        if part.part_type in _BODY_TYPES:
            d_base = part.d_aft

    cache.d_nose = parts[0].d_aft          # cc, i.cs:1068
    cache.d_ref = d_ref
    cache.d_base = d_base
    cache.a_ref = PI_RA * (d_ref / 2.0) ** 2   # ce, i.cs:1071 -- literal pi


# ---------------------------------------------------------------------------
# A.2 -- equivalent-volume body diameter under each fin root
# ---------------------------------------------------------------------------


def _fin_equivalent_diameter(design: Design, cache: GeometryCache) -> None:
    """i.cs:1074-1101.

    Where a fin sits on changing diameter, RASAero replaces the local body
    with an equivalent cylinder of the same *volume* swept under the fin root,
    and it is that diameter -- not the local tube -- which feeds the fin-body
    interference factor.

    The integration window is the fin's ROOT CHORD, not its span. The two are
    easy to confuse because the source spells the root chord ``az2.b`` while
    ``az2.b()`` is the aft diameter of the same object.
    """
    parts = design.parts
    for index, part in enumerate(parts):
        if part.part_type is not PartType.FINS:
            continue
        fin: Fins = part  # type: ignore[assignment]

        if fin.d_fwd == fin.d_aft:
            cache.fin_equiv_diameter[index] = fin.d_fwd
            continue

        volume = 0.0
        x_prev = fin.x0
        d_prev = fin.d_fwd
        window_end = fin.x0 + fin.root_chord
        for other in parts:
            if other.part_type is PartType.FINS:
                continue
            if not (fin.x0 < other.x0 < window_end):
                continue
            volume += (other.x0 - x_prev) * (math.pi / 3.0) * (
                (d_prev / 2.0) ** 2
                + (d_prev / 2.0) * (other.d_fwd / 2.0)
                + (other.d_fwd / 2.0) ** 2
            )
            x_prev = other.x0
            d_prev = other.d_fwd

        volume += (fin.x0 + fin.root_chord - x_prev) * (math.pi / 3.0) * (
            (d_prev / 2.0) ** 2
            + (d_prev / 2.0) * (fin.d_aft / 2.0)
            + (fin.d_aft / 2.0) ** 2
        )
        cache.fin_equiv_diameter[index] = 2.0 * (
            volume / (math.pi * fin.root_chord)
        ) ** 0.5


# ---------------------------------------------------------------------------
# A.3 -- nose shock standoff
# ---------------------------------------------------------------------------


def _fin_shock_standoff(design: Design, cache: GeometryCache) -> None:
    """i.cs:1102-1127.

    A fixed one-degree shock half-angle, not a computed Mach angle, and the
    *distance* is capped at ten nose diameters rather than the ratio being
    capped at ten. Feeds the supersonic effective fin planform.
    """
    parts = design.parts
    nose = parts[0]
    ordinal = 0
    for index, part in enumerate(parts):
        if part.part_type is not PartType.FINS:
            continue
        ordinal += 1
        cache.fin_ordinal[index] = ordinal

        offset = (part.x0 - nose.length) / nose.d_aft
        cache.fin_nose_offset[index] = offset
        if offset > 10.0:
            standoff = math.tan(DEG_TO_RAD_RA) * (10.0 * nose.d_aft)
        else:
            standoff = math.tan(DEG_TO_RAD_RA) * (part.x0 - nose.length)
        cache.fin_shock_standoff[index] = standoff


# ---------------------------------------------------------------------------
# A.4 -- per-part wetted area, planform area, planform centroid
# ---------------------------------------------------------------------------


def _nose_geometry(part, a_ref: float) -> PartGeometry:
    """i.cs:1203-1263.

    Note which pi normalises which family: the ogive and conical branches
    finish with ``Math.PI``, the power-law, parabolic and elliptical branches
    with the literal ``3.14159``. Reproduced exactly.
    """
    d = part.d_aft
    length = part.length
    x0 = part.x0
    shape = part.shape

    if shape in (NoseShape.TANGENT_OGIVE, NoseShape.VON_KARMAN, NoseShape.LV_HAACK):
        rho = (d / 4.0) + length ** 2 / d
        g = length - (rho - d / 2.0) * math.asin(length / rho)
        g = g * ((2.0 * rho) / (d / 2.0) ** 2)
        g = g * (math.pi * (d / 2.0) ** 2 / a_ref)
        return PartGeometry(g, 2.0 / 3.0 * length * d, x0 + 0.625 * length)

    if shape is NoseShape.CONICAL:
        g = PI_RA * (d / 2.0) * ((d / 2.0) ** 2 + length ** 2) ** 0.5
        g = g / (PI_RA * (d / 2.0) ** 2)
        g = g * (math.pi * (d / 2.0) ** 2 / a_ref)
        return PartGeometry(g, 0.5 * length * d, x0 + 2.0 / 3.0 * length)

    if shape is NoseShape.POWER_LAW:
        n = part.power_law_n
        cone = PI_RA * (d / 2.0) * ((d / 2.0) ** 2 + length ** 2) ** 0.5
        if 0.0 <= n < 0.5:
            ak = cone * (2.0 + -1.3428 * n)
            g = ak / (PI_RA * (d / 2.0) ** 2)
            g = g * (PI_RA * (d / 2.0) ** 2 / a_ref)
            return PartGeometry(
                g,
                (1.0 + -0.6666666666666667 * n) * length * d,
                x0 + (0.5 + 0.19999999999999996 * n) * length,
            )
        if 0.5 <= n <= 1.0:
            ak = cone * (0.5 * n ** 2 - 1.4043 * n + 1.9057)
            g = ak / (PI_RA * (d / 2.0) ** 2)
            g = g * (PI_RA * (d / 2.0) ** 2 / a_ref)
            return PartGeometry(
                g,
                (2.0 / 3.0 + -0.33333333333333326 * (n - 0.5)) * length * d,
                x0 + (0.6 + 0.1333333333333333 * (n - 0.5)) * length,
            )
        # Outside 0..1 RASAero leaves every field at zero rather than
        # clamping, so a nonsense exponent yields a nose with no wetted area.
        return PartGeometry()

    if shape is NoseShape.PARABOLIC:
        ak = PI_RA * (d / 2.0) * ((d / 2.0) ** 2 + length ** 2) ** 0.5
        ak = 1.3286 * ak
        g = ak / (PI_RA * (d / 2.0) ** 2)
        g = g * (PI_RA * (d / 2.0) ** 2 / a_ref)
        return PartGeometry(g, 2.0 / 3.0 * length * d, x0 + 0.6 * length)

    if shape is NoseShape.ELLIPTICAL:
        e = (1.0 - d ** 2 / (4.0 * length ** 2)) ** 0.5
        g = math.asin(e)
        g = g * ((2.0 * (length / d)) / e)
        g = 1.0 + g
        g = g * (PI_RA * (d / 2.0) ** 2 / a_ref)
        return PartGeometry(
            g,
            PI_RA * (d / 2.0) * length / 2.0,
            x0 + (length - 4.0 * length / 9.424769999999999),
        )

    raise ValueError(f"unhandled nose shape {shape!r}")


def _tube_geometry(part: Part, a_ref: float) -> PartGeometry:
    """i.cs:1266-1275."""
    d = part.d_fwd
    g = 2.0 * (part.length / (d / 2.0))
    g = g * (PI_RA * (d / 2.0) ** 2 / a_ref)
    return PartGeometry(g, part.length * d, part.x0 + 0.5 * part.length)


def _transition_geometry(part: Part, a_ref: float) -> PartGeometry:
    """i.cs:1277-1299. Identical for Expansion and Reducer."""
    r_aft = part.d_aft / 2.0
    r_fwd = part.d_fwd / 2.0
    g = PI_RA * (r_aft + r_fwd) * (part.length ** 2 + (r_aft - r_fwd) ** 2) ** 0.5
    g = g / (PI_RA * r_fwd ** 2)
    g = g * (PI_RA * r_fwd ** 2 / a_ref)
    planform = part.length * (part.d_fwd + part.d_aft) / 2.0
    centroid = part.x0 + (
        part.length * (part.d_fwd + 2.0 * part.d_aft)
        / (3.0 * (part.d_fwd + part.d_aft))
    )
    return PartGeometry(g, planform, centroid)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(design: Design) -> GeometryCache:
    """Run Pass A. Cheap enough to call per design, far too slow per Mach."""
    if not design.parts:
        raise ValueError("design has no parts")
    if design.parts[0].part_type is not PartType.NOSE_CONE:
        raise ValueError(
            "the first part must be the nose cone: RASAero takes the reference "
            "body diameter from parts[0] (i.cs:1068)"
        )

    cache = GeometryCache()
    _reference_quantities(design, cache)
    _fin_equivalent_diameter(design, cache)
    _fin_shock_standoff(design, cache)

    for index, part in enumerate(design.parts):
        if part.part_type is PartType.NOSE_CONE:
            cache.per_part[index] = _nose_geometry(part, cache.a_ref)
        elif part.part_type is PartType.BODY_TUBE:
            cache.per_part[index] = _tube_geometry(part, cache.a_ref)
        elif part.part_type in (PartType.EXPANSION, PartType.REDUCER):
            cache.per_part[index] = _transition_geometry(part, cache.a_ref)
        else:
            cache.per_part[index] = PartGeometry()

    # A.5 -- totals (i.cs:1153-1168).
    cache.wetted_over_aref = sum(p.wetted_over_aref for p in cache.per_part.values())
    cache.planform_area = sum(p.planform_area for p in cache.per_part.values())
    moment = sum(
        p.planform_area * p.planform_centroid for p in cache.per_part.values()
    )
    cache.planform_centroid = moment / cache.planform_area if cache.planform_area else 0.0

    return cache
