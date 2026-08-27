"""Jorgensen's viscous crossflow terms: the eta factor and cylinder Cd_c.

Jorgensen's semi-empirical body normal force splits into a potential term and
a viscous crossflow term. RASAero assembles the viscous term as

    dCN = eta * Cd_c * (A_plan / A_ref) * sin^2(alpha)          (i.cs:3753)

and this module owns the two coefficients in front: ``eta`` (i.cs:874-877) and
``Cd_c`` (i.cs:1642-1696). The assembly itself, and the moment arm ``aw = ch``
that goes with it, belong to the caller.

Parameter identity
------------------
Both crossflow arguments are *component* quantities resolved normal to the
body axis, formed at i.cs:3749-3750 (and again at 4056-4057, 4098-4099):

    ay = M    * sin(alpha)      -- crossflow Mach,     field ``ay``
    a0 = Re_L * sin(alpha)      -- crossflow Reynolds, field ``a0``

``A_0`` at that call site is the freestream Mach number, not beta, and ``bc``
is the Reynolds number on the *body length* (i.cs:871, ``be``/12 in feet), not
on a diameter. That matters: the 200,000 branch threshold below is a classic
cylinder-diameter critical Reynolds number, but RASAero feeds it a
length-based one, so the drag-crisis branch trips at a body-length Reynolds of
200,000 -- three orders of magnitude off the physical crisis for any realistic
rocket. Reproduced as written; see ``crossflow_reynolds``.

The three ``az`` of i.cs
------------------------
The eta field is ``i.az``. Do not confuse it with the *class* ``az`` (the fin
part type, cast to at i.cs:1080, 1303, 2473, ...) or with ``aq.az``, a
per-part interpolation slot written at i.cs:915 and read at i.cs:3890. Three
unrelated things spelled the same way; only the first is eta.
"""

from __future__ import annotations

import math

from .geometry import GeometryCache

#: Branch threshold on the crossflow Reynolds number (i.cs:1644, 1668).
#: Nominally the subcritical/supercritical cylinder transition, but applied to
#: a body-length Reynolds number -- see the module docstring.
RE_CROSSFLOW_CRITICAL = 200000.0


# ---------------------------------------------------------------------------
# eta -- the crossflow drag proportionality factor
# ---------------------------------------------------------------------------


def eta(cache: GeometryCache) -> float:
    """Jorgensen's crossflow drag proportionality factor (i.cs:874-877).

    A finite-length cylinder at incidence sheds less than the two-dimensional
    crossflow drag; eta is the ratio, read off Jorgensen's chart against
    fineness ratio and fitted here as ``0.509 + 0.183 log10(L/d)``.

    The ratio is ``be / cd``, which is the pair to get right because three
    other diameters are cached alongside:

    * ``be`` = ``cache.l_body``, the summed length of the *body-shaped* parts
      only (NoseCone, BodyTube, Expansion, Reducer -- i.cs:1021-1038). Fins
      and protuberances contribute nothing, so this is body length, not
      overall length.
    * ``cd`` = ``cache.d_ref``, the maximum over *both* diameters of *every*
      part (i.cs:1039-1050). Not ``cb``/``d_base`` (aft diameter of the last
      body part), not ``cc``/``d_nose`` (aft diameter of parts[0]), and not
      the equivalent-volume fin diameter ``aq.ai``.

    Unguarded on purpose: a design with ``l_body < d_ref`` gives a negative
    log and an eta below 0.509, and ``l_body == 0`` raises. RASAero has no
    guard either -- ``Math.Log10(0)`` yields -Infinity and the surrounding
    ``new decimal(...)`` throws -- so nothing is clamped here.
    """
    return 0.509 + 0.183 * math.log10(cache.l_body / cache.d_ref)  # i.cs:876


# ---------------------------------------------------------------------------
# Crossflow conditions -- how the caller forms the arguments to Cd_c
# ---------------------------------------------------------------------------


def crossflow_mach(mach: float, alpha_rad: float) -> float:
    """Component of the freestream Mach normal to the body axis (i.cs:3749).

    ``alpha_rad`` is RASAero's ``m_g``, the angle of attack already in
    radians. Note that ``m_g`` is built with ``Math.PI / 180`` (i.cs:488)
    while the 4-degree reference incidence ``m_f`` is the hard-coded
    ``0.0698131111111111`` (i.cs:3815, 4097) -- which is 4 * ``DEG_TO_RAD_RA``,
    the *literal*-pi conversion. The two degree conversions disagree in the
    15th digit and the caller must keep them straight; this function takes
    radians so it stays out of that.
    """
    return mach * math.sin(alpha_rad)


def crossflow_reynolds(reynolds_body: float, alpha_rad: float) -> float:
    """Component of the body-length Reynolds number normal to the axis.

    i.cs:3750. ``reynolds_body`` is ``bc``, built from the *body length*
    ``be`` (i.cs:871, 3639), so this is a length-based Reynolds number being
    scaled by sin(alpha) and then compared against a diameter-based critical
    value in :func:`crossflow_drag_coefficient`. That mismatch is RASAero's,
    and it is what makes the supercritical ladder the one almost every real
    flight point actually takes.
    """
    return reynolds_body * math.sin(alpha_rad)


# ---------------------------------------------------------------------------
# Cd_c -- cylinder crossflow drag coefficient
# ---------------------------------------------------------------------------


def crossflow_drag_coefficient(
    mach_crossflow: float,
    reynolds_crossflow: float,
) -> float:
    """Crossflow drag coefficient of a circular cylinder (i.cs:1642-1696).

    Field ``ax``. Two different Mach ladders below Mc = 0.6 depending on which
    side of ``RE_CROSSFLOW_CRITICAL`` the crossflow Reynolds number falls, and
    a single shared ladder above it.

    The supercritical ladder carries the drag crisis: Cd_c collapses from 1.2
    to 0.25 between Mc = 0.1 and 0.25, sits at 0.25 through Mc = 0.4, then
    recovers. RASAero has indexed a Reynolds-number effect on Mach, which is
    why the plateau exists at all; the subcritical ladder simply holds 1.2 to
    Mc = 0.2 and joins the same quadratic.

    Both ladders are slightly discontinuous where they join, and both
    discontinuities are reproduced:

    * supercritical, at Mc = 0.5: the linear ramp ends at 1.3570 while the
      quadratic starts at 1.35725;
    * subcritical, at Mc = 0.2: the 1.2 plateau meets a quadratic worth
      1.20188.

    Above Mc = 1.6 the value is pinned at 1.2, which is *below* the 1.5 held
    just above Mc = 0.6 -- the cubic between them is a fit through the
    transonic hump and its endpoints are the ladder's, not the fit's.
    """
    # Guards are transcribed as written rather than merged into an if/elif
    # chain: the source tests the same `ay <= 0.6` twice and then its
    # complement, using non-short-circuit `&`. The three are exhaustive and
    # each ladder covers its whole interval, so `ax` is always assigned --
    # but `ax` is a persistent field, so had any ladder had a hole it would
    # have silently reused the previous Mach point's value.

    if (mach_crossflow <= 0.6) and (reynolds_crossflow >= RE_CROSSFLOW_CRITICAL):
        # Supercritical: post-drag-crisis. i.cs:1644-1667.
        if mach_crossflow <= 0.1:
            return 1.2  # i.cs:1649
        if mach_crossflow <= 0.25:
            # i.cs:1653. Written `1.2 + -0.95 * ...`; lands exactly on 0.25
            # at the upper end, so the plateau below is continuous.
            return 1.2 + -0.95 * ((mach_crossflow - 0.1) / 0.15)
        if mach_crossflow <= 0.4:
            return 0.25  # i.cs:1657
        if mach_crossflow <= 0.5:
            # i.cs:1661. The denominator is 0.5 - 0.4 evaluated in binary
            # floating point, not 0.1 -- keep it literal.
            return 0.25 + 1.107 * (
                (mach_crossflow - 0.4) / 0.09999999999999998
            )
        # mach_crossflow <= 0.6, i.cs:1665
        return (
            1.329
            - 1.097 * mach_crossflow
            + 2.307 * mach_crossflow ** 2.0
        )

    if (mach_crossflow <= 0.6) and (reynolds_crossflow < RE_CROSSFLOW_CRITICAL):
        # Subcritical: no drag crisis, so no 0.25 plateau. i.cs:1668-1679.
        if mach_crossflow <= 0.2:
            return 1.2  # i.cs:1673
        # mach_crossflow <= 0.6, i.cs:1677 -- same quadratic as the
        # supercritical branch, entered from a different Mach.
        return (
            1.329
            - 1.097 * mach_crossflow
            + 2.307 * mach_crossflow ** 2.0
        )

    # i.cs:1680. The source guards this ladder explicitly rather than letting
    # it fall through, and the guard is kept: with three independent `if`
    # blocks and no `else`, a non-finite crossflow Mach matches none of them
    # and RASAero leaves its `ax` field at whatever the previous call left
    # there. That staleness is not reproduced -- it would make this function
    # order-dependent, and the auditor confirmed the ladders are bit-identical
    # over the whole finite domain -- so the unmatched case returns NaN, which
    # propagates rather than silently borrowing a neighbouring answer.
    if not mach_crossflow > 0.6:
        return math.nan

    # Reynolds no longer enters above Mc = 0.6. i.cs:1680-1695.
    if mach_crossflow <= 0.8:
        return 1.5  # i.cs:1685
    if mach_crossflow <= 1.6:
        # i.cs:1689. Transonic hump fit; 1.4953 at Mc = 0.8 and 1.1967 at
        # Mc = 1.6, so it very nearly meets the constants either side.
        return (
            -0.5135 * mach_crossflow ** 3.0
            + 2.2825 * mach_crossflow ** 2.0
            - 3.5508 * mach_crossflow
            + 3.1381
        )
    return 1.2  # i.cs:1693
