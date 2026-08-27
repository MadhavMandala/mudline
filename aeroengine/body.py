"""Pass B, bodies: nose-cone and transition aerodynamics.

Wave drag, normal-force-curve slope and centre of pressure for the three
axisymmetric body parts that carry any -- the nose cone, the expansion and the
reducer -- in both regimes and both Barrowman modes. Body tubes carry none of
the three (``i.cs:3948-3951`` zeroes them outright); fins live elsewhere; and
the transonic band is not a third regime but a straight interpolation between
the subsonic and supersonic values these functions return (``i.cs:3884-3892``),
which is the solver's job, not this module's.

Conventions
-----------
* CN-alpha is per radian and **already divided by the reference area**: every
  routine ends with ``* (PI_RA * r**2 / a_ref)``. That factor is not always a
  no-op -- which radius appears in the numerator differs between routines, and
  in two of them it is the *nose* radius rather than the part's own.
* CP is returned as an absolute station in inches from the nose tip, i.e.
  ``part.x0 + arm``, matching ``aq.r``.
* Wave drag is a Cd referenced to A_ref.

Argument identity: read this before touching a formula
------------------------------------------------------
The supersonic solver calls the nose routines back to back with *different*
third arguments (traced at ``i.cs:3936-3946``)::

    g(nose, A_0)                       # wave drag  <- A_0 is MACH
    value = (A_0*A_0 - 1) ** 0.5       # i.cs:3924  -- this is beta
    a(num2, ref a1)                    # i.cs:1698  -- aft body length, inches
    c(nose, (float)a1, (float)value)   # CN-alpha   <- a1 and BETA
    a(nose, (float)a1, (float)value)   # CP         <- a1 and BETA

So exactly one nose routine sees Mach and two see beta, and the two that see
beta immediately divide it by the fineness ratio to index a chart. Feeding
Mach to those two produces a perfectly smooth, perfectly wrong curve
everywhere above M 1.05, which is why beta is a required positional argument
here rather than something derivable inside.

The same split holds for the transitions: ``c(expansion, A_0)`` gets Mach
(i.cs:3953) while ``b(expansion, a1, value)`` gets beta (i.cs:3955);
``d(reducer, value)`` gets beta (i.cs:3960) while the reducer's wave drag gets
Mach as its *sixth* argument (i.cs:3959).

Dead and degenerate code reproduced here
----------------------------------------
* Supersonic CN-alpha, Power Law: the n in [0.75, 1] blend at i.cs:1960-1966
  is the identity, because the switch above it routes Power Law into the
  *tangent ogive* case, so both ends of the blend are the same number.
* Supersonic CP, Power Law: the ``case "Power Law"`` label inside the switch
  at i.cs:2119 is unreachable -- the enclosing ``if`` at i.cs:2076 already
  excluded that string -- so the blend block at i.cs:2154-2166 never runs, and
  even if it did, both its endpoints are uninitialised zeros and ``num8`` is
  overwritten on the very next line.
* Two more write-only locals: i.cs:2199 (``num9 = num8``) and i.cs:2230 (the
  expansion's forward-truncation length).
* i.cs:2450-2453: a reducer with zero aft diameter computes its CP arm and
  never stores it, leaving the CP at the nose tip.
* Several chart branches leave the working value at its initial zero when a
  ratio and its reciprocal are both >= 1, which cannot happen in exact
  arithmetic. Noted at each site.

Single precision, where it is load-bearing
------------------------------------------
RASAero holds most intermediates in ``decimal`` but declares the chart-lookup
parameters as C# ``float``. That was originally dismissed here as a seventh-
significant-figure difference "that can only change a result by flipping a
chart branch for an input sitting within 1e-7 of a knot" -- which is exactly
what happens, and more often than that phrasing suggests, because the knots sit
at round numbers and test conditions sit on round numbers too.

A 3-calibre Von Karman nose at Mach 2.60 gives beta/(L/d) = 0.8 to within
1e-16. Single precision rounds below the 0.8 knot, double rounds above it, and
the two chart segments do not meet there: the centre of pressure steps by
0.016 in. So the lookup parameters are passed through ``_f32`` and everything
downstream stays in double, which is what the original does.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence

from .compressible import cp_stagnation
from .geometry import PI_RA
from .parts import Expansion, NoseCone, NoseShape, Part, PartType, Reducer

__all__ = [
    "conical_cna_chart",
    "conical_wave_drag_chart",
    "expansion_cna_subsonic",
    "expansion_cna_supersonic",
    "expansion_cp_subsonic",
    "expansion_cp_supersonic",
    "expansion_wave_drag",
    "nose_cna_subsonic",
    "nose_cna_supersonic",
    "nose_cp_subsonic",
    "nose_cp_supersonic",
    "nose_wave_drag",
    "reducer_cna_subsonic",
    "reducer_cna_supersonic",
    "reducer_cp_subsonic",
    "reducer_cp_supersonic",
    "reducer_wave_drag",
]


# ---------------------------------------------------------------------------
# Shared primitives
# ---------------------------------------------------------------------------


def _f32(value: float) -> float:
    """Round to IEEE single precision.

    RASAero declares the chart-lookup parameters as C# ``float``, not
    ``decimal`` or ``double`` (i.cs:1727, 1861-1862, 2070-2071, 2212, 2432).
    Everywhere else the difference is a part in 10^7 and invisible, but these
    values are *compared against chart breakpoints*, and the chart segments do
    not meet exactly at their seams. At a condition that lands on a boundary
    the two precisions take different branches and the answer steps.

    A real instance: a 3-calibre Von Karman nose at Mach 2.60 gives
    beta/(L/d) = 0.79999995 in single and 0.80000000000000016 in double. The
    first takes the ``<= 0.8`` branch, the second does not, and the centre of
    pressure moves 0.016 in. Small, but it is a discontinuity, and reproducing
    it is the difference between matching RASAero and nearly matching it.
    """
    return struct.unpack("f", struct.pack("f", value))[0]


def _le(value: float, threshold: float) -> bool:
    """``value <= threshold`` evaluated in single precision, as C# does.

    The chart branch tests in the original compare a ``float`` against a
    ``float`` literal -- ``num <= 0.8f`` -- so BOTH sides are single. Rounding
    only the left-hand side is worse than rounding neither: 0.8f is
    0.800000011920929, which is *larger* than double 0.8, so an x that C#
    accepts into the branch gets rejected here.

    That is not hypothetical. A 3-calibre Von Karman nose at Mach 2.60 lands
    on 0.8f exactly; C# takes the ``<= 0.8`` branch and double arithmetic does
    not, moving the centre of pressure 0.016 in.
    """
    return value <= _f32(threshold)


def _lt(value: float, threshold: float) -> bool:
    """Strict form of :func:`_le`, for the ``< 1f`` tests."""
    return value < _f32(threshold)


def _pow(base: float, exponent: float) -> float:
    """``Math.Pow`` semantics, which differ from Python's for a negative base.

    A boattail steeper than the 17.5-degree clamp can rebuild a *negative*
    effective aft diameter, and that ratio is then raised to a fractional
    power. .NET returns NaN; Python returns a complex number, which would
    propagate silently into the drag sum and only surface much later as a
    TypeError somewhere unrelated. NaN is the faithful result and it is also
    the one that fails loudly at the right place.
    """
    if base < 0.0 and exponent != int(exponent):
        return math.nan
    return base ** exponent


def _capped(value: float, cap: float) -> float:
    """RASAero's ``(!(v > cap)) ? v : cap``.

    Written with the negation rather than as ``v <= cap`` because the two
    differ on NaN -- the negated form propagates it, the direct comparison
    clamps it -- and the aft-body-length ratio is NaN for a part of zero
    diameter. The cap *value* is deliberately not a default: it differs
    between routines and between shape families within one routine, so every
    call site names its own.
    """
    return value if not (value > cap) else cap


def _cd_quadratic(a2: float, a1: float, a0: float, x: float, mach: float) -> float:
    """``(a2 x^2 + a1 x + a0) / (M^2 * 0.7)``.

    The form every nose wave-drag correlation takes. ``x`` is Mach over the
    fineness ratio; the 0.7 M^2 divisor converts a pressure-integral fitted
    against static pressure into a coefficient against dynamic pressure, since
    q = (gamma/2) p M^2 = 0.7 p M^2.

    The square term comes first here because that is the order the original
    writes these particular curves in, and float addition is not associative:
    these values are the endpoints of linear interpolations that must agree
    with the neighbouring branch's endpoint bit for bit.
    """
    return (a2 * x ** 2.0 + a1 * x + a0) / (mach ** 2.0 * 0.7)


def _cd_quadratic_constant_first(
    a0: float, a1: float, a2: float, x: float, mach: float
) -> float:
    """Same curve as ``_cd_quadratic``, summed constant-term first.

    The tangent-ogive, conical and Von Karman correlations (i.cs:1731, 1735,
    1739) are written ``c + b*x + a*x^2`` while every other curve in the same
    method is written ``a*x^2 + b*x + c``. Kept apart rather than unified,
    because the difference is real at the last bit and the conical curve in
    particular is an interpolation endpoint (i.cs:1804).
    """
    return (a0 + a1 * x + a2 * x ** 2.0) / (mach ** 2.0 * 0.7)


def conical_wave_drag_chart(x: float, mach: float) -> float:
    """Conical-nose wave drag, ``i.cs:1735``.

    Called from three places: a conical nose, the n = 1 anchor of the
    power-law table (i.cs:1804), and every expansion (i.cs:2213), which is
    treated as a slice of the cone it would complete.

    ``x`` is Mach divided by the fineness ratio -- written in the original as
    ``M * (d / L)``, not ``M / (L / d)``.
    """
    if _le(x, 1.4792):
        return _cd_quadratic_constant_first(0.016351, 0.11179, 0.38354, x, mach)
    return _cd_quadratic_constant_first(0.080555, 0.12364, 0.34619, x, mach)


def conical_cna_chart(
    l_over_d: float, beta: float, aft_body_diameters: float
) -> float:
    """Supersonic CN-alpha of a cone of fineness ``l_over_d`` (i.cs:1927-1958).

    Not folded into the nose code because the expansion evaluates it *twice*
    and differences the results (i.cs:2243-2311): once for the virtual cone
    completed from the aft diameter, once for the cone completed from the
    forward diameter. The difference is the normal force the frustum itself
    carries.

    ``aft_body_diameters`` is the length of unbroken body tube behind the part,
    in that part's diameters: a body tube carries no normal force of its own,
    so the chart hands the tube's share of the lift back to the part ahead of
    it. It is capped at 10 here; the CP charts cap the same quantity at 25 for
    every shape but the cone, which is why the cap is written at each call
    site rather than kept as one shared constant.

    Returns the slope per radian, *not* yet divided by the reference area.
    """
    # i.cs:1861 and 2247/2279: beta over fineness, the classic supersonic
    # slender-body similarity parameter.
    x = _f32(beta / l_over_d)                                 # C# float, i.cs:2212
    n3 = _capped(aft_body_diameters, 10.0)
    # i.cs:1864 -- the working value starts at zero and stays there if nothing
    # below matches. See the innermost `else`.
    value = 0.0

    if _le(x, 0.15):
        value = 2.0                                              # i.cs:1933
    elif _le(x, 1.0):
        value = 2.0 - 0.147 * (x - 0.15)                         # i.cs:1938
        value = value + (
            0.3 * n3 - 0.0283 * n3 ** 2.0 + 0.0008 * n3 ** 3.0
        ) * (1.323 * (x - 0.15))                                 # i.cs:1939
    else:
        # Past x = 1 the chart is re-indexed on the reciprocal, recomputed
        # from scratch rather than as 1/x (i.cs:1942).
        inv = l_over_d / beta
        if _le(inv, 0.35):
            value = 2.0 + (
                0.3 * n3 - 0.0283 * n3 ** 2.0 + 0.0008 * n3 ** 3.0
            ) * (3.443 * inv)                                    # i.cs:1946
        elif _le(inv, 0.7):
            value = 2.0 + (
                0.3 * n3 - 0.0283 * n3 ** 2.0 + 0.0008 * n3 ** 3.0
            ) * (
                -1.117 + 9.733 * inv - 9.312 * inv ** 2.0 + 1.305 * inv ** 3.0
            )                                                    # i.cs:1950
        elif _lt(inv, 1.0):
            value = 2.0 - 0.417 * (inv - 0.7)                    # i.cs:1954
            value = value + (
                0.3 * n3 - 0.0283 * n3 ** 2.0 + 0.0008 * n3 ** 3.0
            ) * (1.58 - 1.518 * (inv - 0.7))                     # i.cs:1955
        # else: no branch assigns, and the slope stays 0. Unreachable in exact
        # arithmetic -- inv is 1/x and we are here only because x > 1 -- but
        # RASAero holds the working value in a *field* (`cx`) for the expansion
        # case, so were it ever reached the previous part's chart value would
        # leak through. Left as a zero rather than a leak; the difference
        # cannot be observed.
    return value


# ---------------------------------------------------------------------------
# Nose cone
# ---------------------------------------------------------------------------


def _power_law_wave_drag(nose: NoseCone, x: float, mach: float) -> float:
    """The power-law nose wave-drag table, ``i.cs:1741-1807``.

    A ladder of eleven anchor curves fitted at n = 0, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.666, 0.75, 0.8, 0.9 and 1, linearly interpolated in n between whichever
    pair brackets the exponent. Two anchors are shared with other shapes and
    worth recognising: n = 0.5 is exactly the Parabolic curve (i.cs:1814) and
    n = 1 is exactly the Conical curve, cone drag being what a power-law body
    with n = 1 is.

    The n = 0 anchor is not a curve fit at all but a flat-faced cylinder:
    0.9 * Cp_stagnation, minus a Mach-dependent correction. The interval
    widths are transcribed as the original stores them -- ``0.09999999999999998``
    is not a typo for 0.1, it is the constant-folded ``0.3 - 0.2``.
    """
    n = nose.power_law_n

    if n <= 0.1:
        # i.cs:1746-1748 -- modified-Newtonian stagnation Cp for the blunt
        # n = 0 limit, then i.cs:1749: 0.9 of it, less 0.2 below M 2.67 and a
        # 1.43/M^2 relief above it.
        cp_stag = cp_stagnation(mach)
        if mach <= 2.67:
            a_0 = 0.9 * cp_stag - 0.2
        else:
            a_0 = 0.9 * cp_stag - 1.43 / mach ** 2.0
        a_01 = _cd_quadratic(5.977, 0.2237, 0.5396, x, mach)
        return a_0 + (a_01 - a_0) / 0.1 * n
    if n <= 0.2:
        a_01 = _cd_quadratic(5.977, 0.2237, 0.5396, x, mach)
        a_02 = _cd_quadratic(3.4669, -0.0915, 0.1349, x, mach)
        return a_01 + (a_02 - a_01) / 0.1 * (n - 0.1)
    if n <= 0.3:
        a_02 = _cd_quadratic(3.4669, -0.0915, 0.1349, x, mach)
        a_03 = _cd_quadratic(1.6917, 0.0982, 0.0221, x, mach)
        return a_02 + (a_03 - a_02) / 0.09999999999999998 * (n - 0.2)
    if n <= 0.4:
        a_03 = _cd_quadratic(1.6917, 0.0982, 0.0221, x, mach)
        a_04 = _cd_quadratic(0.8582, 0.108, 0.0106, x, mach)
        return a_03 + (a_04 - a_03) / 0.10000000000000003 * (n - 0.3)
    if n <= 0.5:
        a_04 = _cd_quadratic(0.8582, 0.108, 0.0106, x, mach)
        a_05 = _cd_quadratic(0.4955, 0.1139, 0.0097, x, mach)
        return a_04 + (a_05 - a_04) / 0.09999999999999998 * (n - 0.4)
    if n <= 0.666:
        a_05 = _cd_quadratic(0.4955, 0.1139, 0.0097, x, mach)
        a_0666 = _cd_quadratic(0.3994, -0.1705, 0.1332, x, mach)
        return a_05 + (a_0666 - a_05) / 0.16600000000000004 * (n - 0.5)
    if n <= 0.75:
        a_0666 = _cd_quadratic(0.3994, -0.1705, 0.1332, x, mach)
        a_075 = _cd_quadratic(0.3152, 0.092, 0.0126, x, mach)
        return a_0666 + (a_075 - a_0666) / 0.08399999999999996 * (n - 0.666)
    if n <= 0.8:
        a_075 = _cd_quadratic(0.3152, 0.092, 0.0126, x, mach)
        a_08 = _cd_quadratic(0.305, 0.1146, 0.0096, x, mach)
        return a_075 + (a_08 - a_075) / 0.050000000000000044 * (n - 0.75)
    if n <= 0.9:
        a_08 = _cd_quadratic(0.305, 0.1146, 0.0096, x, mach)
        a_09 = _cd_quadratic(0.3198, 0.1189, 0.0165, x, mach)
        return a_08 + (a_09 - a_08) / 0.09999999999999998 * (n - 0.8)
    # n <= 1.0. The caller has already established 0 <= n <= 1.
    a_09 = _cd_quadratic(0.3198, 0.1189, 0.0165, x, mach)
    a_10 = conical_wave_drag_chart(x, mach)                      # i.cs:1804
    return a_09 + (a_10 - a_09) / 0.09999999999999998 * (n - 0.9)


def nose_wave_drag(nose: NoseCone, mach: float, *, a_ref: float) -> float:
    """Nose wave drag, ``i.cs:1723-1837``. Supersonic pass only.

    ``mach`` here really is Mach -- this is the one nose routine that receives
    it (i.cs:3938); its neighbours receive beta. See the module docstring.

    A nose whose shape has no correlation, or a Power Law with an exponent
    outside [0, 1], contributes only its blunting term: the shape tests are a
    chain of independent ``if``s over a value initialised to zero, with no
    else and no fallback (i.cs:1728-1820).
    """
    # i.cs:1727. Mach over fineness, written as M * (d/L).
    x = _f32(mach * (nose.d_aft / nose.length))               # C# float, i.cs:1727
    shape = nose.shape

    cd = 0.0
    if shape is NoseShape.TANGENT_OGIVE:
        if _le(x, 1.1053):
            cd = _cd_quadratic_constant_first(0.013881, 0.016821, 0.57574, x, mach)
        else:
            cd = _cd_quadratic_constant_first(0.098737, 0.043618, 0.48204, x, mach)
    if shape is NoseShape.CONICAL:
        cd = conical_wave_drag_chart(x, mach)
    if shape is NoseShape.VON_KARMAN:
        if _le(x, 1.8203):
            cd = _cd_quadratic_constant_first(-0.0070281, 0.084717, 0.40673, x, mach)
        else:
            cd = _cd_quadratic_constant_first(0.1653, -0.045249, 0.42612, x, mach)
    if shape is NoseShape.POWER_LAW and 0.0 <= nose.power_law_n <= 1.0:
        cd = _power_law_wave_drag(nose, x, mach)
    if shape is NoseShape.LV_HAACK:
        if _le(x, 1.3725):
            cd = _cd_quadratic(0.5407, 0.1113, 0.0027, x, mach)
        else:
            cd = _cd_quadratic(0.5776, -0.1035, 0.228, x, mach)
    if shape is NoseShape.PARABOLIC:
        # i.cs:1814 -- the n = 0.5 power-law anchor, reused verbatim.
        cd = _cd_quadratic(0.4955, 0.1139, 0.0097, x, mach)
    if shape is NoseShape.ELLIPTICAL:
        cd = _cd_quadratic(0.5395, 0.0501, 0.1088, x, mach)

    if nose.blunt_radius > 0.0:
        # i.cs:1824-1827. A spherical cap of radius r_b carrying half its
        # stagnation Cp (the hemispherical average), scaled by the fraction of
        # the base area it occupies. Both pis appear in that one ratio: the
        # cap area uses the literal 3.14159 and the base area uses Math.PI, so
        # the ratio is not 1 when r_b = d/2 but 1 - 8.4e-7. Reproduced.
        blunt = (1.0 * (cp_stagnation(mach) / 2.0)) * (
            PI_RA * nose.blunt_radius ** 2.0
            / (math.pi * (nose.d_aft / 2.0) ** 2.0)
        )
    else:
        blunt = 0.0

    total = cd + blunt                                           # i.cs:1833
    return total * (PI_RA * (nose.d_aft / 2.0) ** 2.0 / a_ref)   # i.cs:1834


def nose_cna_subsonic(
    nose: NoseCone,
    aft_body_length: float,
    *,
    a_ref: float,
    modified_barrowman: bool,
) -> float:
    """Nose CN-alpha, subsonic, ``i.cs:1839-1855``.

    Classic Barrowman gives every nose a slope of exactly 2 per radian
    regardless of shape or fineness. Modified Barrowman multiplies that by
    ``((L + Lb) / L) ** 0.1``, where Lb is the length of unbroken body tube
    behind the nose *in inches* -- note the units: the supersonic routines
    take the same quantity in nose diameters.

    ``aft_body_length`` is ``a1``, computed by i.cs:1698-1721: the summed
    length of the BodyTube parts following this one, stopping at the first
    Expansion or Reducer.
    """
    if modified_barrowman:
        cna = 2.0 * (
            (nose.length + aft_body_length) / nose.length
        ) ** 0.1                                                 # i.cs:1846
    else:
        cna = 2.0                                                # i.cs:1851
    return cna * (PI_RA * (nose.d_aft / 2.0) ** 2.0 / a_ref)


def nose_cna_supersonic(
    nose: NoseCone,
    aft_body_length: float,
    beta: float,
    *,
    a_ref: float,
) -> float:
    """Nose CN-alpha, supersonic, ``i.cs:1857-1969``.

    ``beta`` is ``sqrt(M^2 - 1)`` (i.cs:3924), NOT Mach. ``aft_body_length``
    is a1 in inches; it is converted to diameters here and capped at 10 for
    every shape family -- unlike the CP routine, which caps at 25 for all
    shapes but the cone.

    Three charts: a tangent-ogive family (which Power Law, Parabolic and
    Elliptical all share), a Von Karman family shared with LV-Haack, and the
    cone. The Barrowman mode does not enter: RASAero has no modified-Barrowman
    supersonic nose slope.
    """
    # i.cs:1861-1862.
    x = _f32(beta / (nose.length / nose.d_aft))               # C# float, i.cs:1861/2070
    aft_diameters = aft_body_length / nose.d_aft
    shape = nose.shape
    value = 0.0

    if shape in (
        NoseShape.TANGENT_OGIVE,
        NoseShape.POWER_LAW,
        NoseShape.PARABOLIC,
        NoseShape.ELLIPTICAL,
    ):
        n3 = _capped(aft_diameters, 10.0)                        # i.cs:1872
        if _le(x, 0.5):
            value = 1.99 + 1.61 * x - 0.985 * x ** 2.0
        elif _le(x, 0.7):
            value = 1.99 + 1.61 * x - 0.985 * x ** 2.0
            value = value + (0.21 * n3 - 0.0111 * n3 ** 2.0) * (
                1.167 * (x - 0.5)
            )
        elif _le(x, 1.0):
            value = 2.635 - 0.704 * (x - 0.7)
            value = value + (0.21 * n3 - 0.0111 * n3 ** 2.0) * (
                0.233 + 2.3 * (x - 0.7)
            )
        else:
            inv = (nose.length / nose.d_aft) / beta               # i.cs:1890
            value = 1.735 + 0.689 * inv
            value = value + (0.21 * n3 - 0.0111 * n3 ** 2.0) * (
                2.804 * inv - 1.882 * inv ** 2.0
            )
    elif shape in (NoseShape.VON_KARMAN, NoseShape.LV_HAACK):
        n3 = _capped(aft_diameters, 10.0)                        # i.cs:1903
        if _le(x, 0.5):
            value = 1.91 + 1.513 * x - 1.24 * x ** 2.0
        elif _le(x, 0.8):
            value = 1.91 + 1.513 * x - 1.24 * x ** 2.0
            value = value + (0.227 * n3 - 0.0128 * n3 ** 2.0) * (
                2.142 * (x - 0.5)
            )
        elif _le(x, 1.0):
            value = 2.328 - 0.358 * (x - 0.8)
            value = value + (0.227 * n3 - 0.0128 * n3 ** 2.0) * (
                0.643 + 2.4 * (x - 0.8)
            )
        else:
            inv = (nose.length / nose.d_aft) / beta               # i.cs:1921
            value = 1.757 + 0.5 * inv
            value = value + (0.227 * n3 - 0.0128 * n3 ** 2.0) * (
                -0.247 + 2.593 * inv - 1.223 * inv ** 2.0
            )
    elif shape is NoseShape.CONICAL:
        value = conical_cna_chart(nose.length / nose.d_aft, beta, aft_diameters)

    if shape is NoseShape.POWER_LAW:
        # i.cs:1894-1897 and 1960-1966. The apparent intent is a blend from
        # the tangent-ogive chart to the conical chart over n in [0.75, 1],
        # mirroring the wave-drag table where n = 1 *is* the cone. It does not
        # execute: the switch above lists "Power Law" alongside "Tangent
        # Ogive", so `value` at both i.cs:1896 and i.cs:1962 is the same
        # tangent-ogive number, and the interpolation is the identity for
        # every n. A power-law nose therefore gets tangent-ogive lift at all
        # exponents, including n = 1 where it is geometrically a cone.
        tangent_end = value                                      # num8
        conical_end = value                                      # num10
        n = nose.power_law_n
        if 0.0 <= n <= 1.0:
            if n >= 0.75:
                value = tangent_end + (conical_end - tangent_end) / 0.25 * (
                    n - 0.75
                )
            else:
                value = tangent_end

    return value * (PI_RA * (nose.d_aft / 2.0) ** 2.0 / a_ref)   # i.cs:1968


def nose_cp_subsonic(
    nose: NoseCone,
    aft_body_length: float,
    *,
    modified_barrowman: bool,
) -> float:
    """Nose CP, subsonic, ``i.cs:1972-2064``. Station in inches from the tip.

    Classic Barrowman is the textbook table of centroid fractions of the nose
    length. Modified Barrowman shifts the CP aft by treating the nose plus a
    shape-dependent multiple of the afterbody as one lifting length::

        x_cp = -(1 - f) * (L + k * Lb) + L + Lb

    with f the same centroid fraction and k a per-shape lever (1.4 for an
    ellipse up to 2.8 for a cone). With Lb = 0 the two agree exactly.

    The Power Law branch derives f from the body's own volume rather than a
    table, and leaves the arm at zero -- CP at the nose tip -- when the
    exponent falls outside [0, 1] (i.cs:2000, 2043).
    """
    arm = 0.0
    shape = nose.shape

    if modified_barrowman:
        if shape is NoseShape.TANGENT_OGIVE:
            f = 0.466
            arm = (-1.0 * (1.0 - f)) * (
                nose.length + 1.75 * aft_body_length
            ) + nose.length + aft_body_length                    # i.cs:1984
        elif shape is NoseShape.CONICAL:
            f = 0.666666666666667
            arm = (-1.0 * (1.0 - f)) * (
                nose.length + 2.8 * aft_body_length
            ) + nose.length + aft_body_length                    # i.cs:1990
        elif shape is NoseShape.VON_KARMAN:
            f = 0.5
            arm = (-1.0 * (1.0 - f)) * (
                nose.length + 1.85 * aft_body_length
            ) + nose.length + aft_body_length                    # i.cs:1996
        elif shape is NoseShape.POWER_LAW:
            if 0.0 <= nose.power_law_n <= 1.0:
                # i.cs:2002-2005. Volume of the power-law solid, divided by
                # the base area to get an equivalent length, divided by the
                # length to get a fraction, then complemented. Note the volume
                # uses the full diameter squared, not the radius.
                f = (
                    PI_RA * nose.d_aft ** 2.0 * nose.length
                    / (4.0 * (2.0 * nose.power_law_n + 1.0))
                )
                f = f / (PI_RA * (nose.d_aft / 2.0) ** 2.0)
                f = f / nose.length
                f = 1.0 - f
                arm = (-1.0 * (1.0 - f)) * (
                    nose.length
                    + (1.9 * nose.power_law_n + 0.9) * aft_body_length
                ) + nose.length + aft_body_length                # i.cs:2006
        elif shape is NoseShape.LV_HAACK:
            f = 0.4415
            arm = (-1.0 * (1.0 - f)) * (
                nose.length + 1.65 * aft_body_length
            ) + nose.length + aft_body_length                    # i.cs:2012
        elif shape is NoseShape.PARABOLIC:
            f = 0.5
            arm = (-1.0 * (1.0 - f)) * (
                nose.length + 1.85 * aft_body_length
            ) + nose.length + aft_body_length                    # i.cs:2018
        elif shape is NoseShape.ELLIPTICAL:
            f = 0.333
            arm = (-1.0 * (1.0 - f)) * (
                nose.length + 1.4 * aft_body_length
            ) + nose.length + aft_body_length                    # i.cs:2024
    else:
        if shape is NoseShape.TANGENT_OGIVE:
            arm = 0.466 * nose.length
        elif shape is NoseShape.CONICAL:
            arm = 2.0 / 3.0 * nose.length
        elif shape is NoseShape.VON_KARMAN:
            arm = 0.5 * nose.length
        elif shape is NoseShape.POWER_LAW:
            if 0.0 <= nose.power_law_n <= 1.0:
                arm = (
                    PI_RA * nose.d_aft ** 2.0 * nose.length
                    / (4.0 * (2.0 * nose.power_law_n + 1.0))
                )                                                # i.cs:2045
                arm = arm / (PI_RA * (nose.d_aft / 2.0) ** 2.0)
                arm = arm / nose.length
                arm = (1.0 - arm) * nose.length
        elif shape is NoseShape.LV_HAACK:
            arm = 0.4415 * nose.length
        elif shape is NoseShape.PARABOLIC:
            arm = 0.5 * nose.length
        elif shape is NoseShape.ELLIPTICAL:
            arm = 0.333 * nose.length

    return nose.x0 + arm                                         # i.cs:2062


def nose_cp_supersonic(
    nose: NoseCone,
    aft_body_length: float,
    beta: float,
) -> float:
    """Nose CP, supersonic, ``i.cs:2066-2204``. Station in inches from the tip.

    ``beta`` is ``sqrt(M^2 - 1)``, not Mach (i.cs:3945).

    The control flow is worth stating plainly, because the shape routing is
    not what the ``switch`` appears to say. The enclosing test at i.cs:2076
    diverts Tangent Ogive *and Power Law* past the switch entirely, straight
    to the tangent-ogive block; Parabolic and Elliptical enter the switch only
    to ``break`` out of it and land in the same block. So:

    * tangent-ogive chart, afterbody capped at 25 diameters: Tangent Ogive,
      Power Law, Parabolic, Elliptical
    * Von Karman chart, capped at 25: Von Karman, LV-Haack
    * conical chart, capped at 10: Conical only

    The ``case "Power Law"`` sharing the conical label at i.cs:2119 is dead,
    and with it the blend block at i.cs:2154-2166 -- which was doubly dead,
    since both its endpoints were uninitialised zeros and its result is
    overwritten by the next statement.
    """
    # i.cs:2070-2071.
    x = _f32(beta / (nose.length / nose.d_aft))               # C# float, i.cs:1861/2070
    aft_diameters = aft_body_length / nose.d_aft
    shape = nose.shape
    fraction = 0.0

    if shape is not NoseShape.TANGENT_OGIVE and shape is not NoseShape.POWER_LAW:
        if shape in (NoseShape.VON_KARMAN, NoseShape.LV_HAACK):
            n3 = _capped(aft_diameters, 25.0)                    # i.cs:2086
            if _le(x, 0.5):
                fraction = 0.507 + 0.097 * x - 0.059 * x ** 2.0
            elif _le(x, 0.8):
                fraction = 0.507 + 0.097 * x - 0.059 * x ** 2.0
                fraction = fraction + (
                    0.1242 * n3 - 0.00235 * n3 ** 2.0
                ) * (0.75 * (x - 0.5))
            elif _le(x, 1.0):
                fraction = 0.547 + (
                    0.1242 * n3 - 0.00235 * n3 ** 2.0
                ) * (0.226 + 0.775 * (x - 0.8))
            else:
                inv = (nose.length / nose.d_aft) / beta           # i.cs:2103
                if _le(inv, 0.45):
                    fraction = 0.52 + (
                        0.1242 * n3 - 0.00235 * n3 ** 2.0
                    ) * (0.09 + 0.935 * inv - 0.619 * inv ** 2.0)
                elif _lt(inv, 1.0):
                    fraction = 0.52 + 0.049 * (inv - 0.45)
                    fraction = fraction + (
                        0.1242 * n3 - 0.00235 * n3 ** 2.0
                    ) * (0.11 + 0.89 * inv - 0.619 * inv ** 2.0)
                # else: fraction stays 0, putting the CP at the nose tip.
                # Unreachable: inv is the reciprocal of x, which is > 1 here.
            return nose.x0 + fraction * nose.length              # i.cs:2115

        if shape is NoseShape.CONICAL:
            n3 = _capped(aft_diameters, 10.0)                    # i.cs:2121
            if _le(x, 0.5):
                fraction = 0.666
            elif _le(x, 1.0):
                fraction = 0.666 + (
                    0.209 * n3 - 0.0109 * n3 ** 2.0
                ) * (0.468 * (x - 0.5))
            else:
                inv = (nose.length / nose.d_aft) / beta           # i.cs:2133
                if _le(inv, 0.25):
                    fraction = 0.695 - 0.058 * inv
                    fraction = fraction + (
                        0.209 * n3 - 0.0109 * n3 ** 2.0
                    ) * (0.305 + 0.795 * inv)
                elif _le(inv, 0.5):
                    fraction = 0.695 - 0.058 * inv
                    fraction = fraction + (
                        0.209 * n3 - 0.0109 * n3 ** 2.0
                    ) * (0.24 + 1.429 * inv - 1.503 * inv ** 2.0)
                elif _le(inv, 0.7):
                    fraction = 0.666 + (
                        0.209 * n3 - 0.0109 * n3 ** 2.0
                    ) * (0.269 + 1.371 * inv - 1.503 * inv ** 2.0)
                elif _lt(inv, 1.0):
                    fraction = 0.666 + (
                        0.209 * n3 - 0.0109 * n3 ** 2.0
                    ) * (1.1 - 0.862 * inv)
                # else: fraction stays 0. Unreachable, as above.
            # i.cs:2154-2166 sits here: a Power-Law blend inside a branch only
            # a Conical nose can reach, writing a value the next line discards.
            # Reproduced by omission -- there is nothing it could change.
            return nose.x0 + fraction * nose.length              # i.cs:2167

    # Tangent Ogive and Power Law arrive here without entering the switch;
    # Parabolic and Elliptical arrive by breaking out of it (i.cs:2080-2082).
    # The switch's `default` (which would leave the CP at the nose tip) is
    # unreachable: those four plus the three handled above are every shape.
    n3 = _capped(aft_diameters, 25.0)                            # i.cs:2174
    if _le(x, 0.5):
        fraction = 0.465 + 0.215 * x - 0.145 * x ** 2.0
    elif _le(x, 0.7):
        fraction = 0.465 + 0.215 * x - 0.145 * x ** 2.0
        fraction = fraction + (0.1236 * n3 - 0.00232 * n3 ** 2.0) * (
            0.698 * (x - 0.5)
        )
    elif _le(x, 1.0):
        fraction = 0.545 + 0.0052 * (x - 0.7)
        fraction = fraction + (0.1236 * n3 - 0.00232 * n3 ** 2.0) * (
            0.14 + 0.734 * (x - 0.7)
        )
    else:
        inv = (nose.length / nose.d_aft) / beta                  # i.cs:2192
        fraction = 0.464 + 0.082 * inv
        fraction = fraction + (0.1236 * n3 - 0.00232 * n3 ** 2.0) * (
            1.878 * inv - 2.305 * inv ** 2.0 + 0.787 * inv ** 3.0
        )
    # i.cs:2197-2200 copies the result into a local for Power Law and never
    # reads it. Nothing to reproduce.
    return nose.x0 + fraction * nose.length                      # i.cs:2196


# ---------------------------------------------------------------------------
# Expansion
# ---------------------------------------------------------------------------
#
# Every expansion routine works through the cone the frustum is a slice of:
# extend the sides forward to a point and you have a cone of half-angle
# atan(((d_aft - d_fwd)/2) / L) whose base is d_aft. Two lengths matter and
# the original writes each of them in more than one operation order, which is
# preserved here rather than factored:
#
#   L_cone  = (d_aft/2) * (L / ((d_aft - d_fwd)/2))      i.cs:2211, 2226
#   L/d aft = ((L / ((d_aft - d_fwd)/2)) * (d_aft/2)) / d_aft   i.cs:2247
#   L/d fwd = ((L / ((d_aft - d_fwd)/2)) * (d_fwd/2)) / d_fwd   i.cs:2279
#
# The first two are the same number by algebra and not always by arithmetic.


def expansion_wave_drag(part: Expansion, mach: float, *, a_ref: float) -> float:
    """Expansion wave drag, ``i.cs:2207-2218``. Supersonic pass only.

    ``mach`` is Mach (i.cs:3953). The frustum is charged the wave drag of the
    complete cone it belongs to, scaled from that cone's base area to the
    frustum's own annular frontal area by ``(r_aft/r_fwd)^2 - 1``, and then
    referenced to A_ref through the *forward* area. The two area factors do
    not cancel and are not cancelled.

    Stored in ``aq.an`` rather than ``aq.o``: the solver sums expansion and
    reducer wave drag separately from the nose's (i.cs:4026-4033).
    """
    l_cone = (part.d_aft / 2.0) * (
        part.length / ((part.d_aft - part.d_fwd) / 2.0)
    )                                                            # i.cs:2211
    x = _f32(mach * (part.d_aft / l_cone))                       # C# float, i.cs:2212
    cd = conical_wave_drag_chart(x, mach)
    cd = cd * (
        (part.d_aft / 2.0) ** 2.0 / (part.d_fwd / 2.0) ** 2.0 - 1.0
    )                                                            # i.cs:2214
    return cd * (PI_RA * (part.d_fwd / 2.0) ** 2.0 / a_ref)      # i.cs:2215


def expansion_cna_subsonic(
    part: Expansion,
    aft_body_length: float,
    *,
    a_ref: float,
    d_nose: float,
    modified_barrowman: bool,
) -> float:
    """Expansion CN-alpha, subsonic, ``i.cs:2220-2241``.

    The two modes are not variations on one formula; they are different
    derivations.

    Classic Barrowman is the textbook transition result
    ``2 * ((d_aft/d_ref)^2 - (d_fwd/d_ref)^2)`` -- except that RASAero
    normalises by the **nose** diameter ``cc`` and then multiplies by
    ``pi*(cc/2)^2 / A_ref``, so the nose diameter cancels and the true
    reference is the vehicle's maximum diameter. Transcribed uncancelled;
    ``d_nose`` is genuinely read here (i.cs:2237-2238) and a caller passing
    the part's own diameter would get the same answer only by luck.

    Modified Barrowman instead treats the expansion as the cone it completes,
    applies the same ``((L + Lb)/L)^0.1`` afterbody augmentation the nose gets,
    scales by the area ratio and subtracts the unaugmented 2 -- so the
    frustum's slope is the *excess* over a plain cone. ``aft_body_length`` is
    only read in this mode; the solver only bothers to recompute it in this
    mode too (i.cs:3674-3677), which is why a stale value is harmless.
    """
    if modified_barrowman:
        l_cone = (part.d_aft / 2.0) * (
            part.length / ((part.d_aft - part.d_fwd) / 2.0)
        )                                                        # i.cs:2226
        cna = 2.0 * ((l_cone + aft_body_length) / l_cone) ** 0.1  # i.cs:2228
        cna = cna * (
            (part.d_aft / 2.0) ** 2.0 / (part.d_fwd / 2.0) ** 2.0
        )                                                        # i.cs:2229
        # i.cs:2230 computes the forward truncation length
        # (d_fwd/2) * (L / ((d_aft - d_fwd)/2)) here and never uses it. Its
        # absence is the point: what gets subtracted below is a bare 2, so the
        # tip cone is removed *unaugmented* while the full cone was augmented,
        # and the frustum is credited with the tip's share of the afterbody
        # term. Using num2 would have made the two ends symmetric.
        cna = cna - 2.0                                          # i.cs:2232
        return cna * (PI_RA * (part.d_fwd / 2.0) ** 2.0 / a_ref)
    cna = 2.0 * (
        (part.d_aft / d_nose) ** 2.0 - (part.d_fwd / d_nose) ** 2.0
    )                                                            # i.cs:2237
    return cna * (PI_RA * (d_nose / 2.0) ** 2.0 / a_ref)         # i.cs:2238


def expansion_cna_supersonic(
    part: Expansion,
    aft_body_length: float,
    beta: float,
    *,
    a_ref: float,
) -> float:
    """Expansion CN-alpha, supersonic, ``i.cs:2243-2313``.

    ``beta`` is ``sqrt(M^2 - 1)`` (i.cs:3955). The frustum's slope is the
    difference of two conical-chart evaluations -- the cone completed from the
    aft diameter minus the cone completed from the forward diameter -- with
    the first scaled up by the area ratio so both are referenced to the same
    area.

    The two evaluations are not symmetric. The first sees the real afterbody
    length; the second is handed a hard zero (i.cs:2280) before the cap is
    applied, so the tip cone gets no afterbody credit at all. That asymmetry
    is what makes the difference finite for a long afterbody.
    """
    l_over_d_aft = (
        (part.length / ((part.d_aft - part.d_fwd) / 2.0)) * (part.d_aft / 2.0)
    ) / part.d_aft                                               # i.cs:2247
    aft_diameters = aft_body_length / part.d_aft                 # i.cs:2248
    cna_aft = conical_cna_chart(l_over_d_aft, beta, aft_diameters)

    l_over_d_fwd = (
        (part.length / ((part.d_aft - part.d_fwd) / 2.0)) * (part.d_fwd / 2.0)
    ) / part.d_fwd                                               # i.cs:2279
    # i.cs:2280 -- zero, not `aft_body_length / part.d_fwd`.
    cna_fwd = conical_cna_chart(l_over_d_fwd, beta, 0.0)

    cna = cna_aft * (
        (part.d_aft / 2.0) ** 2.0 / (part.d_fwd / 2.0) ** 2.0
    ) - cna_fwd                                                  # i.cs:2311
    return cna * (PI_RA * (part.d_fwd / 2.0) ** 2.0 / a_ref)     # i.cs:2312


def expansion_cp_subsonic(part: Expansion, *, modified_barrowman: bool) -> float:
    """Expansion CP, subsonic, ``i.cs:2316-2331``. Station in inches.

    Modified Barrowman puts the CP at the aft end of the transition, full
    stop. Classic Barrowman uses the textbook transition centroid, which for a
    growing frustum lands somewhere between L/3 and L/2 aft of its shoulder.
    """
    if modified_barrowman:
        return part.x0 + part.length                             # i.cs:2322
    arm = (part.length / 3.0) * (
        1.0
        + (1.0 - part.d_fwd / part.d_aft)
        / (1.0 - (part.d_fwd / part.d_aft) ** 2.0)
    )                                                            # i.cs:2327
    return part.x0 + arm


def expansion_cp_supersonic(part: Expansion) -> float:
    """Expansion CP, supersonic, ``i.cs:2333-2340``.

    The aft end, in both Barrowman modes -- supersonically the lift of an
    expansion is concentrated at the shoulder where the flow is turned.
    """
    return part.x0 + part.length


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------


def _has_transition_before(parts: Sequence[Part], index: int) -> bool:
    """Is there an Expansion or Reducer ahead of ``parts[index]``?

    i.cs:2370-2378. The scan starts at index 1, not 0, so it cannot see the
    nose -- harmless, since parts[0] is always the nose cone -- and stops one
    short of the part itself.
    """
    for other in parts[1:index]:
        if other.part_type in (PartType.EXPANSION, PartType.REDUCER):
            return True
    return False


def _nose_interference_ogive(nose_fineness: float, mach: float) -> float:
    """Boattail drag increment for an ogive-family nose, ``i.cs:2397``.

    Two factors: a fineness-ratio shape term normalised to unity at a nose
    L/d of 4 -- the quadratic evaluates to exactly the 0.0152 it is divided by
    at f = 4 -- times a Mach decay that is frozen at its M 1.25 value below
    M 1.25 rather than extrapolated.
    """
    if mach < 1.25:
        return (
            0.002 * nose_fineness ** 2.0 - 0.0218 * nose_fineness + 0.0704
        ) / 0.0152 * 0.01629
    return (
        0.002 * nose_fineness ** 2.0 - 0.0218 * nose_fineness + 0.0704
    ) / 0.0152 * (0.0226 * mach ** -1.4677)


def _nose_interference_conical(nose_fineness: float, mach: float) -> float:
    """As above for a conical nose, ``i.cs:2402``.

    Same construction, its own coefficients, and again exactly unity at a nose
    L/d of 4. The Mach decay is steeper (-1.95 against -1.47) and starts from
    twice the value, so a conical nose leaves roughly twice the interference
    just above M 1.25 and less of it by M 3.
    """
    if mach < 1.25:
        return (
            0.0016 * nose_fineness ** 2.0 - 0.0198 * nose_fineness + 0.0879
        ) / 0.0343 * 0.0332
    return (
        0.0016 * nose_fineness ** 2.0 - 0.0198 * nose_fineness + 0.0879
    ) / 0.0343 * (0.0513 * mach ** -1.9502)


def reducer_wave_drag(
    part: Reducer,
    mach: float,
    *,
    parts: Sequence[Part],
    index: int,
    nose: NoseCone,
    aft_body_length: float,
    boattail_angle_deg: float,
    a_ref: float,
) -> float:
    """Reducer (boattail) wave drag, ``i.cs:2351-2426``. Supersonic pass only.

    ``mach`` is Mach: it arrives as the routine's *sixth* argument at
    i.cs:3959, which is the one place in the supersonic loop where a Mach and
    a beta could be swapped without an obvious type error.

    Two terms are summed.

    The first is the boattail's own drag, a function of the boattail angle and
    the radius ratio raised to an exponent that ramps 4 -> 3 between M 3 and
    M 4. ``boattail_angle_deg`` is the cached ``db`` (geometry.py's
    ``boattail_angle_deg``), which -- read its comment -- describes the *last*
    body part only and is zero unless that part is itself a reducer. A reducer
    followed by a body tube therefore gets zero angle and, from
    ``0.001*0 + 0.00071*0``, zero drag. That is RASAero's behaviour, not an
    oversight here.

    Above 17.5 degrees the angle is clamped and, more surprisingly, the aft
    diameter is *rebuilt* from the clamped angle rather than used as given
    (i.cs:2360), so a steep boattail is charged the drag of the 17.5-degree
    boattail of the same length and forward diameter.

    The second term is a nose-interference increment that only applies when
    the reducer sees clean flow: any Expansion or Reducer ahead of it, or more
    than 2.5 forward-diameters of body tube between it and the nose, and the
    term is zero. It is the one place in this module where a part's drag
    depends on the *nose's* shape and fineness.

    ``aft_body_length`` is a1 in inches. The binary does not receive it
    directly -- it reconstructs it as ``A_2 - (nose.x0 + nose.length)`` from a
    station the nose case stashed (i.cs:2386, 3941) -- which is the same
    number up to the single-precision round trip.

    ``nose`` likewise stands in for two different routes in the original: the
    fineness ratio is read from ``m_a[0]`` directly (i.cs:2388) while the
    shape and exponent arrive as arguments A_3 and A_4, carried over from
    whichever nose the loop last visited (i.cs:3942-3943). With one nose per
    vehicle these are the same object; they are one parameter here.
    """
    # i.cs:2355. The radius-ratio exponent: 4 up to M 3, 3 from M 4, linear
    # between. Higher Mach -> lower exponent -> the ratio term bites sooner.
    if mach <= 3.0:
        exponent = 4.0
    elif not (mach < 4.0):
        exponent = 3.0
    else:
        exponent = 4.0 + -1.0 * (mach - 3.0)

    if boattail_angle_deg > 17.5:
        angle = 17.5
        # 0.30543236111111105 is the original's constant-folded 17.5 degrees in
        # radians, and equals 17.5 * DEG_TO_RAD_RA to the bit. The result is
        # the aft diameter a 17.5-degree boattail of this length and forward
        # diameter would have; it is strictly positive whenever this branch
        # runs, since exceeding 17.5 degrees requires exactly that.
        d_aft_capped = -1.0 * (
            2.0 * math.tan(0.30543236111111105) * part.length - part.d_fwd
        )                                                        # i.cs:2360
        cd = (0.001 * angle + 0.00071 * angle ** 2.0) / mach * (
            1.0 - _pow(d_aft_capped / 2.0 / (part.d_fwd / 2.0), exponent)
        )                                                        # i.cs:2361
    else:
        cd = (
            0.001 * boattail_angle_deg
            + 0.00071 * boattail_angle_deg ** 2.0
        ) / mach * (
            1.0
            - ((part.d_aft / 2.0) / (part.d_fwd / 2.0)) ** exponent
        )                                                        # i.cs:2365

    interference = 0.0
    if not _has_transition_before(parts, index):
        tube_diameters = aft_body_length / part.d_fwd            # i.cs:2387
        nose_fineness = nose.length / nose.d_aft                 # i.cs:2388
        if tube_diameters >= 2.5:
            interference = 0.0                                   # i.cs:2391
        else:
            # Faded out linearly over 2.5 diameters of separation. The three
            # tests below are independent ``if``s in the original; the shape
            # strings make them exclusive, and Power Law is deliberately not
            # in the first list.
            if nose.shape in (
                NoseShape.TANGENT_OGIVE,
                NoseShape.VON_KARMAN,
                NoseShape.LV_HAACK,
                NoseShape.PARABOLIC,
                NoseShape.ELLIPTICAL,
            ):
                k = _nose_interference_ogive(nose_fineness, mach)
                interference = k + (-1.0 * k) / 2.5 * tube_diameters
            if nose.shape is NoseShape.CONICAL:
                k = _nose_interference_conical(nose_fineness, mach)
                interference = k + (-1.0 * k) / 2.5 * tube_diameters
            if nose.shape is NoseShape.POWER_LAW:
                # Here the ogive-to-cone blend over n in [0.75, 1] that the
                # CN-alpha and CP routines only appear to do actually happens.
                k = _nose_interference_ogive(nose_fineness, mach)
                ogive_end = k + (-1.0 * k) / 2.5 * tube_diameters
                if 0.0 <= nose.power_law_n <= 0.75:
                    interference = ogive_end                     # i.cs:2411
                if 0.75 < nose.power_law_n <= 1.0:
                    k = _nose_interference_conical(nose_fineness, mach)
                    conical_end = k + (-1.0 * k) / 2.5 * tube_diameters
                    interference = ogive_end + (
                        conical_end - ogive_end
                    ) / 0.25 * (nose.power_law_n - 0.75)         # i.cs:2417
                # An exponent outside [0, 1] matches neither test and leaves
                # the increment at zero, discarding the ogive value computed
                # just above.

    total = cd + interference                                    # i.cs:2422
    return total * (PI_RA * (part.d_fwd / 2.0) ** 2.0 / a_ref)   # i.cs:2423


def reducer_cna_subsonic(part: Reducer, *, a_ref: float, d_nose: float) -> float:
    """Reducer CN-alpha, subsonic, ``i.cs:2342-2349``. Negative, as it should be.

    The classic Barrowman transition result, normalised by the nose diameter
    and renormalised to A_ref exactly as the expansion's classic branch is.
    There is no modified-Barrowman variant: i.cs:3682 calls this
    unconditionally, so the afterbody augmentation the nose and expansions
    receive is silently not applied to reducers.
    """
    cna = 2.0 * (
        (part.d_aft / d_nose) ** 2.0 - (part.d_fwd / d_nose) ** 2.0
    )                                                            # i.cs:2346
    return cna * (PI_RA * (d_nose / 2.0) ** 2.0 / a_ref)         # i.cs:2347


def reducer_cna_supersonic(part: Reducer, beta: float, *, a_ref: float) -> float:
    """Reducer CN-alpha, supersonic, ``i.cs:2428-2444``.

    ``beta`` is ``sqrt(M^2 - 1)`` (i.cs:3960), and the chart parameter is
    ``(L/d_fwd) / beta`` -- fineness over beta, the reciprocal of the grouping
    the nose and expansion charts use.

    Past 2.1491 the cubic is replaced by the flat slender-body value -2; that
    knot is where the cubic reaches 2, so the curve is continuous there.
    """
    x = _f32((part.length / part.d_fwd) / beta)                  # C# float, i.cs:2432
    if x >= 2.1491:
        cna = -2.0 * (1.0 - (part.d_aft / part.d_fwd) ** 2.0)    # i.cs:2435
    else:
        cna = (
            0.0084969
            + 2.0498 * x
            - 0.68455 * x ** 2.0
            + 0.075355 * x ** 3.0
        )                                                        # i.cs:2439
        cna = (-1.0 * cna) * (1.0 - (part.d_aft / part.d_fwd) ** 2.0)
    return cna * (PI_RA * (part.d_fwd / 2.0) ** 2.0 / a_ref)     # i.cs:2442


def reducer_cp_subsonic(part: Reducer) -> float:
    """Reducer CP, subsonic, ``i.cs:2446-2460``. Station in inches.

    BUG REPRODUCED. When the aft diameter is exactly zero -- a boattail closed
    to a point, which is how a tail cone is drawn -- i.cs:2450-2453 computes
    ``L/3`` into a local and never stores it. The CP field keeps the zero the
    per-Mach reset loop left there (i.cs:3914), i.e. the reducer's negative
    normal force is applied at the *nose tip*, dragging the vehicle CP
    forward. Returning 0.0 rather than raising, because 0.0 is exactly what
    the solver goes on to read.
    """
    if part.d_aft == 0.0:
        return 0.0
    arm = (part.length / 3.0) * (
        1.0
        + (1.0 - part.d_fwd / part.d_aft)
        / (1.0 - (part.d_fwd / part.d_aft) ** 2.0)
    )                                                            # i.cs:2456
    return part.x0 + arm


def reducer_cp_supersonic(part: Reducer) -> float:
    """Reducer CP, supersonic, ``i.cs:2462-2469``.

    A flat 55% of the length aft of the shoulder, with no Mach or shape
    dependence and no zero-diameter special case -- so the zero-aft-diameter
    reducer that loses its CP subsonically keeps a sane one supersonically,
    and the transonic interpolation between them (i.cs:3889) sweeps the CP
    from the nose tip to 0.55 L across M 0.9 to 1.05.
    """
    return part.x0 + 0.55 * part.length                          # i.cs:2466
