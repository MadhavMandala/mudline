"""Skin friction: the flat-plate correlation, the roughness cutoff, and the
compressibility correction.

RASAero computes mean skin friction twice per Mach point, from one routine
duplicated verbatim in the original: once for the body over the total body
length (``i.cs:1471-1548``, result in ``i.m_w``) and once per fin set over
that set's mean aerodynamic chord (``i.cs:1550-1640``, result in ``i.m_x``).
The two copies differ in exactly three places, all of which are parameter
plumbing rather than physics:

* the body is handed a Reynolds number directly, while the fin routine
  rescales the body's Reynolds number by ``MAC / L_body``;
* the body's Reynolds floor is a flat 10 000, while the fin's floor is
  ``10000 * MAC / L_body`` -- the floor is a floor on the *body* Reynolds
  number, expressed in fin units (see ``_skin_friction``);
* the body reads the global turbulence flag ``bi``, while each fin set reads
  its own ``aq.f``, which is seeded from the global flag but forced on for a
  Square airfoil (``i.cs:1382-1385``).

Everything below the parameter plumbing is shared, so it is shared here too.

Two cutoff mechanisms, not one
------------------------------
Surface roughness enters through a cutoff Reynolds number ``Re_cut``: above
it, the boundary layer is roughness-dominated and the friction coefficient
stops falling with Reynolds number. RASAero implements that idea twice, with
different behaviour, and which one runs depends on where ``Re_cut`` lands and
on the turbulence flag:

1. **A genuine cutoff.** ``Re`` is clamped to ``Re_cut`` and the ordinary
   correlations are evaluated there. This is what happens whenever the
   turbulence flag is set, whenever ``Re_cut`` exceeds 2 580 000, and (in the
   flag-clear case) whenever ``Re_cut`` is below 182 500.

2. **A discontinuous snap.** With the flag clear *and* ``Re_cut`` inside
   [182 500, 2 580 000], ``Re`` is not clamped to ``Re_cut`` at all. It is
   replaced by one of two fixed constants chosen by where ``Re`` itself sits:
   182 500 if ``Re`` is in [182 500, 500 000), and 2 580 000 if ``Re`` is at
   or above 500 000 -- *however far above*. ``Re`` below 182 500 is left
   alone.

The two constants are calibrated against each other: ``1.328/sqrt(182500)``
= 0.0031086 and the turbulent value at 2 580 000 is 0.0031084, so the snap is
smooth in ``Re`` across 500 000 and smooth in ``Re_cut`` across 182 500. It is
*not* smooth in ``Re_cut`` across 2 580 000, because mechanism 1 respects the
actual Reynolds number and mechanism 2 discards it. A vehicle at Re = 600 000
whose ``Re_cut`` drifts across 2 580 000 with Mach sees C_f jump from 0.00209
to 0.00311 -- a 48% step, and a visible step in C_D. That step is a defect,
and reproducing it is the point of this module.

Direction of the turbulence flag
--------------------------------
``bi == -1`` means *Turbulent Flow is on* (``i.cs:434-440``). With it set,
RASAero takes the fully-turbulent Prandtl-Schlichting curve; with it clear it
subtracts ``1700/Re``, the standard allowance for the laminar run ahead of
transition, and below Re = 500 000 it drops to Blasius laminar entirely. So
the flag *raises* friction, which is the sanity check on the branch: at
Re = 2.58e6 the flag-set value is 0.00377 against 0.00311 flag-clear, 21%
higher.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .parts import Airfoil, Design, Fins

# ---------------------------------------------------------------------------
# The magic Reynolds numbers. All four are literals in the original; they are
# named here only because each appears two or three times and a transposed
# digit in one copy would be invisible.
# ---------------------------------------------------------------------------

#: Lower edge of the snap band, and the low snap target (i.cs:1492-1500).
RE_SNAP_LOW = 182500.0

#: Upper edge of the snap band, and the high snap target (i.cs:1483, 1496,
#: 1504). Also the Reynolds number at which the roughness-limited turbulent
#: C_f is evaluated, which is what makes the snap continuous at RE_SNAP_LOW.
RE_SNAP_HIGH = 2580000.0

#: Laminar/turbulent switch (i.cs:1509). The snap at i.cs:1502 reuses the same
#: number, so a snapped-up Reynolds number always lands in the turbulent
#: branch -- the two uses are numerically identical and conceptually distinct.
RE_TURBULENT = 500000.0

#: Reynolds floor, applied before anything else (i.cs:1473, 1553). Guards the
#: log10 and the 1/sqrt at zero airspeed rather than expressing any physics.
RE_FLOOR = 10000.0


def _f32(x: float) -> float:
    """Round to IEEE single precision.

    The Reynolds floor lives in a C# ``float`` and is compared against with
    ``Convert.ToSingle`` (i.cs:1473-1478, 1553-1561), so it carries about
    seven significant digits. Every other quantity in these two routines is
    ``decimal`` or ``double``; only the floor narrows, and only the fin floor
    narrows visibly because it is a product rather than a literal.
    """
    return struct.unpack("<f", struct.pack("<f", x))[0]


@dataclass(frozen=True)
class SkinFriction:
    """One evaluation of the skin-friction routine.

    ``cf`` is the only thing that feeds C_D; the rest is returned because it
    is what makes a disagreement with RASAero diagnosable. RASAero itself
    keeps only ``cf`` (``i.m_w`` / ``i.m_x``).
    """

    cf: float
    #: Reynolds number the correlation was actually evaluated at, after the
    #: floor, the cutoff clamp and the snap.
    re_effective: float
    #: The roughness cutoff Reynolds number. Zero means "not computed": with
    #: zero roughness the whole cutoff block is skipped (i.cs:1479, 1562), so
    #: zero here is the absence of a cutoff, not a cutoff of zero.
    re_cutoff: float
    #: Whether the turbulent branch was taken. Drives RASAero's "TRANSITIONING
    #: TO TURBULENT FLOW" console message (i.cs:1519-1527, 1602-1618) and
    #: nothing else -- the ``bh`` / ``aq.ab`` state flag those lines latch is
    #: write-only as far as the aerodynamics are concerned.
    turbulent: bool


def cutoff_reynolds(ref_length_in: float, roughness_in: float, mach: float) -> float:
    """Roughness cutoff Reynolds number for a surface (i.cs:1482 / 1565).

    ``Re_cut = f(M) * (L/k)^1.056`` where ``L/k`` is the reference length over
    the equivalent sand-grain roughness height, both in inches, and ``f(M)``
    is a four-segment fit rising from 34.8 at M = 0 to a plateau of 180 at
    M >= 3. The segments are continuous at M = 1, 2 and 3 (42.0, 89.6, 180.0
    from either side), so the Mach dependence introduces no step of its own.

    Transcribed in the source's own test order: the M >= 3 plateau is tested
    *before* the 2 <= M < 3 ramp, which is why the ramp is the fallthrough.
    """
    value = ref_length_in / roughness_in
    if mach < 1.0:
        return (34.8 + 7.2 * mach) * value ** 1.056
    if mach < 2.0:
        return (42.0 + 47.6 * (mach - 1.0)) * value ** 1.056
    if mach >= 3.0:
        return 180.0 * value ** 1.056
    return (89.6 + 90.4 * (mach - 2.0)) * value ** 1.056


def compressibility_correction(cf: float, mach: float) -> float:
    """Scale an incompressible C_f for Mach number (i.cs:1533-1547 / 1624-1638).

    Three disjoint segments above M = 1.05 and no correction at or below it.
    The internal joins are smooth to four figures -- 0.744000 against 0.744027
    at M = 2, and 0.169088 against the 0.1691 tail constant at M = 10, that
    constant being just the M = 10 value of the middle segment. The *entry* is
    not: RASAero applies the linear segment from its full value at M = 1.05
    rather than fairing it in from unity at M = 1, so C_f drops 1.28%
    discontinuously as Mach crosses 1.05.

    The segments are written as three independent ``if``s in the source rather
    than a chain. They cannot both fire, but the shape is kept.
    """
    if not (mach < 1.05):  # negated <, as written -- kept for the NaN path
        if mach >= 1.05 and mach <= 2.0:
            cf = cf * (1.0 - 0.256 * (mach - 1.0))
        if mach > 2.0 and mach <= 10.0:
            cf = cf / (1.0 + 0.144 * mach * mach) ** 0.65
        if mach > 10.0:
            cf = cf * 0.1691
    return cf


def _skin_friction(
    re: float,
    re_floor: float,
    ref_length_in: float,
    roughness_in: float,
    mach: float,
    turbulent: bool,
) -> SkinFriction:
    """The routine both callers share (i.cs:1471-1548, i.cs:1550-1640).

    ``turbulent`` is the flag's *meaning*, not its encoding: RASAero stores
    -1 for on and 0 for off, and tests ``== -1`` at i.cs:1483/1509/1511 for
    the body and ``az3.f() == -1`` at i.cs:1566/1592/1594 for a fin. Note
    ``az3.f()`` there is the base-class ``aq`` accessor returning the numeric
    turbulence flag -- ``az3.f`` without the parentheses is the *airfoil
    section name*, a string that ``az`` redeclares with ``new``.

    ``ref_length_in`` feeds only the cutoff, and it is each caller's own
    reference length: the body length for the body, the fin set's mean
    aerodynamic chord for a fin set.
    """
    # i.cs:1475 / 1558. The comparison is made in single precision because the
    # floor is a C# float; the floor value itself is already single-precision.
    if _f32(re) < re_floor:
        re = re_floor

    re_cut = 0.0
    if roughness_in > 0.0:  # i.cs:1479 / 1562 -- no roughness, no cutoff at all
        re_cut = cutoff_reynolds(ref_length_in, roughness_in, mach)

        if re_cut > RE_SNAP_HIGH or turbulent:
            # Mechanism 1, the genuine cutoff: evaluate the correlation at
            # Re_cut instead of Re. Note this is the *only* path taken when
            # the turbulence flag is set -- the snap is unreachable then.
            if re >= re_cut:  # i.cs:1485 / 1568
                re = re_cut
        else:
            # Flag clear and Re_cut <= 2 580 000.
            if re_cut < RE_SNAP_LOW and re >= re_cut:  # i.cs:1492 / 1575
                re = re_cut
            # The second half of this test is redundant: the enclosing else
            # already established Re_cut <= 2 580 000. Reproduced as written.
            if re_cut >= RE_SNAP_LOW and re_cut <= RE_SNAP_HIGH:  # i.cs:1496 / 1579
                # Mechanism 2, the snap. Re_cut has served its purpose as a
                # band test and is now discarded: the replacement value
                # depends only on which side of RE_TURBULENT *Re* sits. A
                # vehicle at Re = 10^8 is snapped down to 2.58e6 just as
                # readily as one at Re = 5e5 is snapped up to it.
                if re >= RE_SNAP_LOW and re < RE_TURBULENT:  # i.cs:1498 / 1581
                    re = RE_SNAP_LOW
                if re >= RE_TURBULENT:  # i.cs:1502 / 1585
                    re = RE_SNAP_HIGH

    if turbulent or re >= RE_TURBULENT:  # i.cs:1509 / 1592
        if turbulent:
            # Prandtl-Schlichting, fully turbulent from the leading edge.
            #
            # re == 0 is reachable: a zero-length surface with a non-zero
            # roughness drives the cutoff to zero and clamps Re to it. .NET
            # does not fault there -- Math.Log10(0) is -Infinity, and
            # Math.Pow(-Infinity, 2.58) is +Infinity because 2.58 is not an
            # odd integer -- so RASAero returns C_f = 0. Python would raise
            # ValueError instead, which is a different behaviour, not a
            # stricter one. The flag-clear branch below is left to raise
            # because .NET overflows there too (decimal cannot hold Infinity).
            cf = 0.0 if re == 0.0 else 0.455 / math.log10(re) ** 2.58  # i.cs:1513 / 1596
            was_turbulent = True
        else:
            # Same curve less 1700/Re, the classical allowance for the laminar
            # run ahead of transition. Only reachable at Re >= 500 000, which
            # is why the subtraction never drives C_f negative here.
            cf = 0.455 / math.log10(re) ** 2.58 - 1700.0 / re  # i.cs:1517 / 1600
            was_turbulent = True
    else:
        cf = 1.328 / re ** 0.5  # Blasius laminar flat plate, i.cs:1531 / 1622
        was_turbulent = False

    cf = compressibility_correction(cf, mach)
    return SkinFriction(
        cf=cf, re_effective=re, re_cutoff=re_cut, turbulent=was_turbulent
    )


def body_skin_friction(
    re_body: float,
    l_body_in: float,
    mach: float,
    *,
    roughness_in: float,
    turbulent: bool,
) -> SkinFriction:
    """Body mean skin friction (``i.m_w``, i.cs:1471-1548).

    Called as ``c(bc, be, A_0)`` at i.cs:3640 and 3925, so the arguments are,
    in order, the Reynolds number on the body length, the body length itself
    and the *Mach number* -- not beta. ``bc`` is built at i.cs:3639 from the
    same body length, so ``re_body`` and ``l_body_in`` are consistent by
    construction; pass ``geometry.GeometryCache.l_body`` and
    ``atmosphere.reynolds(mach, cache.l_body, alt)``.

    The caller multiplies the result by the total wetted-area ratio ``cf``
    to get the friction drag coefficient (i.cs:3641).
    """
    return _skin_friction(
        re=re_body,
        re_floor=RE_FLOOR,
        ref_length_in=l_body_in,
        roughness_in=roughness_in,
        mach=mach,
        turbulent=turbulent,
    )


def fin_skin_friction(
    re_body: float,
    l_body_in: float,
    mac_in: float,
    mach: float,
    *,
    roughness_in: float,
    turbulent: bool,
) -> SkinFriction:
    """One fin set's mean skin friction (``i.m_x``, i.cs:1550-1640).

    Called as ``a(this.m_a[num2], bc, be, A_0)`` at i.cs:3686 and 3965: the
    fin part, the *body* Reynolds number, the *body* length, and Mach. The
    routine derives the fin's own Reynolds number at i.cs:1555 by rescaling,
    ``Re_body * (MAC / L_body)``, rather than recomputing it from the
    atmosphere -- identical in exact arithmetic, and the route by which the
    body length reaches a fin calculation at all.

    ``mac_in`` is ``aq.aj``, set at i.cs:1356 and 1363. Read it off the base
    class: ``az`` redeclares ``aj`` as a float holding something else
    entirely, which is why the source writes ``((aq)az3).aj``.
    """
    # i.cs:1557. The floor is scaled by the same chord ratio as the Reynolds
    # number, so it is a floor of 10 000 on the *body* Reynolds number, not on
    # the fin's. A fin whose MAC is a tenth of the body length therefore floors
    # at Re = 1000, well below anything the correlations were fitted over.
    # Computed in single precision because the source stores it in a float.
    re_floor = _f32(_f32(RE_FLOOR) * _f32(mac_in / l_body_in))
    return _skin_friction(
        re=re_body * (mac_in / l_body_in),
        re_floor=re_floor,
        ref_length_in=mac_in,
        roughness_in=roughness_in,
        mach=mach,
        turbulent=turbulent,
    )


def fin_turbulence_flag(fin: Fins, design: Design) -> bool:
    """The turbulence flag a fin set actually reads.

    Every part is seeded from the vehicle-wide flag at i.cs:1066
    (``item4.b(bk)``, and ``bk`` tracks ``bi`` exactly -- both are set at
    i.cs:434-440 and nowhere else). A Square airfoil then overrides its own
    copy to -1 at i.cs:1382-1385, so a square-sectioned fin set runs fully
    turbulent even with Turbulent Flow switched off, while the body and every
    other fin set on the same vehicle do not.
    """
    return design.turbulence or fin.airfoil is Airfoil.SQUARE
