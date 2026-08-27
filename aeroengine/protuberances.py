"""Protuberance drag: four hand-fitted curves and two types that carry none.

Curves at ``i.cs:3458-3559``; the scaled types at ``i.cs:3725-3741`` (subsonic)
and ``i.cs:4034-4049`` (supersonic).

Every curve here is piecewise in Mach with a subsonic branch ending at 0.90, a
supersonic branch starting at 1.05, and *nothing in between*: the routine's
accumulator is left at its C# default and zero is written out. That gap is not
an oversight. The dispatcher at ``i.cs:946-969`` never calls either drag
buildup inside (0.90, 1.05) -- it calls the transonic fairing, which re-runs
the whole buildup at exactly 0.90 and at exactly 1.05 and blends between the
two anchors. So the gap is unreachable, and these functions return 0.0 in it
to match what the source would produce.

Where the result lands
----------------------
RASAero returns nothing from these routines; each writes a per-part slot, and
the assembly sums ``k + l + m + a3`` over every part (i.cs:3722, i.cs:4030):

===========  =======  ==========
part         slot     written at
===========  =======  ==========
RailGuide    ``k``    i.cs:3484
LaunchLug    ``l``    i.cs:3508
LaunchShoe   ``m``    i.cs:3526
Plate        ``a3``   i.cs:3557
===========  =======  ==========

Three of them also write a second, *shadowed* field of the same name family
(``bm.c`` through the property setter at i.cs:3483, ``an.b`` at i.cs:3496 and
3506, ``q.b`` at i.cs:3525). Two are dead stores. The third, ``an.b``, is a
latent bug -- see :func:`launch_lug_cd`.

``a_ref`` throughout is ``GeometryCache.a_ref``, which is built with the
literal 3.14159 (``geometry.py``, i.cs:1071). The launch-lug frontal area
above it is built with ``Math.PI``. That mismatch is RASAero's and is
load-bearing; do not unify the two.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from .basedrag import base_pressure_coefficient
from .compressible import cp_stagnation
from .geometry import DEG_TO_RAD_RA
from .parts import (
    LaunchLug,
    LaunchShoe,
    Part,
    PartType,
    Plate,
    RailGuide,
)


def _div(numerator: float, denominator: float) -> float:
    """Divide with C# ``double`` semantics: infinity or NaN, never a raise.

    Needed exactly once, for the inclined plate's division by ``sin(theta)``
    (i.cs:3549). RASAero accepts a plate entered at zero degrees without
    complaint; there the division yields +/-Infinity (or NaN for 0/0), the
    subsequent product collapses to NaN, and NaN then propagates through
    CD_prot into the vehicle total. Python would raise ZeroDivisionError
    instead, which is a *different* failure, so the IEEE-754 result is
    produced explicitly here. This reproduces the defect; it does not repair
    it, and callers must not treat it as a validated input.
    """
    if denominator == 0.0:
        # NaN/0 is NaN under IEEE-754, not a signed infinity. Testing for zero
        # first would fall through to copysign(inf, nan), which yields an
        # infinity and quietly turns a propagating NaN into a finite-looking
        # sign. Ordering matters here.
        if math.isnan(numerator) or numerator == 0.0:
            return math.nan
        return math.copysign(math.inf, numerator) * math.copysign(1.0, denominator)
    return numerator / denominator


# ---------------------------------------------------------------------------
# Base pressure behind a bluff protuberance
# ---------------------------------------------------------------------------


def cp_base_plate(mach: float) -> float:
    """Base pressure coefficient used by the inclined plate (i.cs:3551).

    A *different* ladder from ``basedrag.base_pressure_coefficient``
    (i.cs:3437), and the two are easy to confuse because both
    are three-piece polynomial fits in Mach terminating in a hypersonic
    asymptote. This one starts at -0.46 rather than -0.21, its first
    breakpoint is a strict ``<`` at 1.19 rather than ``<=`` at 1.15, and it
    reaches its -1.43/M^2 asymptote by Mach 5 instead of Mach 7.

    Byte-identical constants appear at i.cs:2737 for fin trailing-edge base
    drag, so ``fins.py`` should import this rather than retype it.
    """
    # Source is one nested ternary; the branch order below is that nesting,
    # unflattened, so a reviewer can follow it against the single line.
    if mach < 1.19:
        return -0.46
    if mach <= 3.0:
        return -1.0 * (
            1.2622
            - 0.98895 * mach
            + 0.30409 * mach ** 2.0
            - 0.033885 * mach ** 3.0
        )
    if mach > 5.0:
        return -1.0 * (1.43 / mach ** 2.0)
    return -1.0 * (0.117 + -0.06 * ((mach - 3.0) / 2.0))


# ---------------------------------------------------------------------------
# The four curves
# ---------------------------------------------------------------------------


def rail_guide_cd(part: RailGuide, mach: float, a_ref: float) -> float:
    """CD of one rail guide on A_ref (i.cs:3458-3486).

    Traced from the caller: ``b(this.m_a[num2], A_0)`` at i.cs:3692 (subsonic)
    and i.cs:3993 (supersonic) both pass ``A_0``, which is Mach in both
    buildups -- not beta, and not the Reynolds number computed a few lines
    earlier in the same scope.

    The frontal area is diameter x height (i.cs:3462): the guide is charged as
    a rectangular block, not as a round rod. The two dimensions are swapped
    twice on the way in -- a6.cs:259 calls ``d(height, diameter)`` and
    i.cs:703 calls ``bm.a(A_1, A_0)`` -- so ``bm.a`` is the diameter and
    ``bm.b`` the height. Their product is order-independent, but the swap is
    worth knowing before trusting either field alone.
    """
    area = part.guide_diameter * part.guide_height          # i.cs:3462
    cd = 0.0                                                # C# default(float)

    if mach <= 0.9:
        if mach <= 0.7:
            # 1.12 is a fixed bluff-body Cp inflated by Prandtl-Glauert; the
            # 4.5 is a fitted scale carried by all three subsonic branches.
            cd = (
                4.5
                * (1.12 / (1.0 - mach ** 2.0) ** 0.5)
                * (area / a_ref)
            )                                               # i.cs:3468
        elif mach <= 0.8:
            cd = 4.5 * (1.57 - 0.7 * (mach - 0.7)) * (area / a_ref)   # i.cs:3472
        elif mach <= 0.9:
            # Redundant against the outer test, and reproduced because it is
            # not quite redundant in the source: the outer test runs on the
            # Mach converted to double, this one on the decimal. A value
            # between the two representations would fall through and leave CD
            # at zero. No Mach the sweep generates can do that.
            cd = 4.5 * (1.5 + 6.0 * (mach - 0.8)) * (area / a_ref)    # i.cs:3476

    # Second independent `if`, not an `else` -- as written at i.cs:3479.
    if mach >= 1.05:
        if mach <= 1.8:
            cd = (
                4.5
                * (4.715 - 3.7 * mach + 0.986 * mach * mach)
                * (area / a_ref)
            )                                               # i.cs:3481
        else:
            cd = 5.625 * (area / a_ref)                     # i.cs:3481
    return cd


def launch_lug_cd(part: LaunchLug, mach: float, a_ref: float) -> float:
    """CD of one launch lug on A_ref (i.cs:3488-3510).

    ``Math.PI`` at i.cs:3492, not the literal 3.14159 -- the lug frontal area
    is one of the places the original switches pi mid-buildup, and the A_ref it
    is divided by was built with 3.14159. Both are reproduced as written.

    Only the lug *diameter* ever reaches the solver. a6.cs:250 passes
    ``au2.c`` (LaunchLugDiameter) and nothing else; ``au2.d``
    (LaunchLugLength) round-trips through the CDX1 file and the dialog but
    feeds no coefficient anywhere. ``LaunchLug.lug_length`` is therefore inert
    here, deliberately -- a longer lug costs nothing in RASAero.

    Two further quirks:

    * The supersonic branch *adds to* the subsonic 5.75 term rather than
      replacing it (i.cs:3505 reuses the local ``value``), so the curve steps
      up discontinuously across the transonic gap.
    * In the gap itself neither branch assigns, and the field that would have
      been assigned -- ``an.b``, a shadow of ``aq.b`` declared with ``new`` --
      is cleared by neither reset loop (i.cs:3626-3637, i.cs:3903-3921).
      ``an3.l = an3.b`` at i.cs:3508 would therefore republish the *previous*
      call's CD instead of zero. Unreachable through the dispatcher, so this
      returns 0.0; the staleness is recorded rather than emulated, because a
      pure function has nowhere to keep it.
    """
    area = math.pi * (part.lug_diameter / 2.0) ** 2.0       # i.cs:3492 -- Math.PI
    cd_sub = 5.75 * (area / a_ref)                          # i.cs:3493
    cd = 0.0

    if mach <= 0.9:
        cd = cd_sub                                         # i.cs:3496

    if mach >= 1.05:
        # i.cs:3500-3502 is cp_stagnation written out longhand, including the
        # "- 1.4 + 1.0" that is not folded to -0.4 anywhere in the original.
        cp_stag = cp_stagnation(mach)
        half_stag = 1.0 * (cp_stag / 2.0)                   # i.cs:3503
        # i.cs:3504 is character-for-character the vehicle base-pressure
        # ladder of i.cs:3437, so it is imported rather than retyped.
        cp_b = base_pressure_coefficient(mach)              # i.cs:3504
        cd = cd_sub + 3.0 * (half_stag + -1.0 * cp_b) * (area / a_ref)  # i.cs:3505
    return cd


def launch_shoe_cd(part: LaunchShoe, mach: float, a_ref: float) -> float:
    """CD of one launch shoe on A_ref (i.cs:3512-3528).

    The shoe has no shape at all: the user types a frontal area in square
    inches (a6.cs:254 passes ``au2.e``, LaunchShoeArea) and the curves are
    pure functions of Mach.

    Both branches are written in the source as ternaries with a negated test
    -- ``(!(M <= 0.5)) ? poly : 9.0`` -- so the branch direction below is the
    same one, unnegated: Mach at or below 0.5 takes the constant.
    """
    cd = 0.0

    if mach <= 0.9:
        if mach <= 0.5:
            cd = 9.0 * (part.area / a_ref)                  # i.cs:3519
        else:
            cd = (
                4.5
                * (2.446 - 1.864 * mach + 1.917 * mach ** 2.0)
                * (part.area / a_ref)
            )                                               # i.cs:3519

    if mach >= 1.05:
        if mach <= 1.83:
            cd = (
                4.5
                * (
                    7.146
                    - 7.709 * mach
                    + 3.604 * mach ** 2.0
                    - 0.546 * mach ** 3.0
                )
                * (part.area / a_ref)
            )                                               # i.cs:3523
        else:
            cd = 7.92 * (part.area / a_ref)                 # i.cs:3523
    return cd


def plate_cd(part: Plate, mach: float, a_ref: float) -> float:
    """CD of one inclined flat plate on A_ref (i.cs:3530-3559).

    Field identity, because the constructor swaps its arguments: a6.cs:274
    calls ``f(frontal_area, angle)``, i.cs:718 forwards to ``r.a(A_0, A_1)``,
    and ``r.a(A_0, A_1)`` assigns ``this.a = A_1`` and ``b = A_0``. So
    **``r.a`` is the angle in degrees and ``r.b`` is the frontal area**, the
    reverse of the call. Reading them the other way round silently produces a
    plausible-looking curve.

    The angle is used in *both* units and it matters which goes where: the
    subsonic fit is a polynomial in DEGREES (i.cs:3538 reads ``r3.a``
    directly), while the radian conversion at i.cs:3534 is consumed only by
    the supersonic Newtonian term.

    Supersonically this is Newtonian impact theory -- Cp = Cp_stag * sin^2(th)
    on a surface inclined at th -- with the ``4 sin^3 th`` form arising from
    resolving that pressure along the axis over the plate's inclined area. The
    0.75 that multiplies both branches is a fixed empirical knockdown with no
    identified published source; the original is the only reference for it.

    Unlike the launch lug, the supersonic branch here *overwrites* the
    subsonic value rather than adding to it (i.cs:3550).
    """
    theta_rad = part.angle_deg * DEG_TO_RAD_RA              # i.cs:3534
    cd = 0.0

    if mach <= 0.9:
        # Degrees, not radians -- see the docstring.
        cd = (
            -0.0001 * part.angle_deg * part.angle_deg
            + 0.0261 * part.angle_deg
            + 0.5856
        )                                                   # i.cs:3538
        cd = 0.75 * cd                                      # i.cs:3539
        cd = cd * (part.area / a_ref)                       # i.cs:3540

    if mach >= 1.05:
        cp_stag = cp_stagnation(mach)                       # i.cs:3544-3546
        newtonian = 4.0 * math.sin(theta_rad) ** 3.0        # i.cs:3547
        cn = cp_stag / 2.0 * newtonian                      # i.cs:3548
        # (A / sin th) / A -- the plate's inclined (wetted) area referenced
        # back to its own frontal area. Algebraically this is 1/sin th and
        # nothing else, and it is left uncancelled because the source leaves
        # it uncancelled: cancelling changes the last bits, and it also hides
        # that a zero plate angle divides by zero here. See _div.
        wetted_over_frontal = _div(part.area, math.sin(theta_rad))   # i.cs:3549
        cn = _div(cn * wetted_over_frontal, part.area)      # i.cs:3549
        cd = cn                                             # i.cs:3550
        cp_b = cp_base_plate(mach)                          # i.cs:3551
        cd = cd + -1.0 * cp_b                               # i.cs:3552-3553
        cd = 0.75 * cd                                      # i.cs:3554
        cd = cd * (part.area / a_ref)                       # i.cs:3555
    return cd


def protuberance_cd(part: Part, mach: float, a_ref: float) -> float:
    """Dispatch one part to its curve, per i.cs:3691-3703 and i.cs:3992-4004.

    The switch in the source has cases for exactly four type names. Anything
    else -- body parts, fins, and both streamlined types -- falls through with
    no assignment, hence 0.0 here. The streamlined types are not forgotten:
    they have no curve at all and are charged from the assembled body terms by
    :func:`streamlined_cd_subsonic` / :func:`streamlined_cd_supersonic`.
    """
    if part.part_type is PartType.RAIL_GUIDE:
        return rail_guide_cd(part, mach, a_ref)     # type: ignore[arg-type]
    if part.part_type is PartType.LAUNCH_LUG:
        return launch_lug_cd(part, mach, a_ref)     # type: ignore[arg-type]
    if part.part_type is PartType.LAUNCH_SHOE:
        return launch_shoe_cd(part, mach, a_ref)    # type: ignore[arg-type]
    if part.part_type is PartType.PLATE:
        return plate_cd(part, mach, a_ref)          # type: ignore[arg-type]
    return 0.0


# ---------------------------------------------------------------------------
# The two streamlined types, which have no curve of their own
# ---------------------------------------------------------------------------


def streamlined_areas(parts: Iterable[Part]) -> tuple[float, float]:
    """``(A_s1, A_s2)``: summed areas of the two streamlined types, in^2.

    i.cs:3720-3721 (subsonic) and i.cs:4028-4029 (supersonic). RASAero
    accumulates these into ``object`` locals seeded with the *integer* 0,
    which is why the guards downstream compare against ``0`` and not ``0m``:
    an accumulator that never saw a part is still an int. The practical
    consequence is nil -- a design carrying a streamlined part of zero area is
    indistinguishable from one carrying none, and both give zero -- but the
    guard is reproduced in shape below so the two match line for line.
    """
    area_no_base_drag = 0.0
    area_with_base_drag = 0.0
    for part in parts:
        if part.part_type is PartType.STREAMLINED_NO_BASE_DRAG:
            area_no_base_drag += part.area          # type: ignore[attr-defined]
        elif part.part_type is PartType.STREAMLINED_WITH_BASE_DRAG:
            area_with_base_drag += part.area        # type: ignore[attr-defined]
    return area_no_base_drag, area_with_base_drag


def streamlined_cd_subsonic(
    area_no_base_drag: float,
    area_with_base_drag: float,
    *,
    cd_friction: float,
    cd_form: float,
    cd_base: float,
    a_ref: float,
) -> tuple[float, float]:
    """Subsonic streamlined-protuberance drag (i.cs:3725-3741).

    A streamlined protuberance is charged a share of the *body's own* drag in
    proportion to its area: RASAero assumes a fairing behaves like a small
    copy of the rocket it is bolted to, rather than fitting it a curve.

    Term identity, traced at i.cs:3641-3643 where they are assigned:
    ``cd_friction`` is ``m_s`` (skin friction on wetted area), ``cd_form`` is
    ``m_t`` (body form drag), ``cd_base`` is ``m_r`` (base drag). The
    with-base-drag variant is the same sum plus base drag -- that single term
    is the entire difference between the two part types.

    **This term set is not the supersonic one.** Supersonically ``cd_form``
    does not exist and two other terms take its place; see
    :func:`streamlined_cd_supersonic`. Crossing the two is the plausible
    mistake here, and it would show up only above Mach 1.05.

    Returns ``(CD_s1, CD_s2)``, the ``df`` and ``dg`` of the source. Both are
    added into CD_prot at i.cs:3741.
    """
    # The `1.0 *` is RASAero's own `decimal.Multiply(1m, ...)`, kept because
    # the house rule is literal transcription, not because it does anything.
    if area_no_base_drag == 0:
        cd_no_base = 0.0                                    # i.cs:3727
    else:
        cd_no_base = (
            1.0 * (cd_friction + cd_form) * (area_no_base_drag / a_ref)
        )                                                   # i.cs:3731

    if area_with_base_drag == 0:
        cd_with_base = 0.0                                  # i.cs:3735
    else:
        cd_with_base = (
            1.0
            * ((cd_friction + cd_form) + cd_base)
            * (area_with_base_drag / a_ref)
        )                                                   # i.cs:3739
    return cd_no_base, cd_with_base


def streamlined_cd_supersonic(
    area_no_base_drag: float,
    area_with_base_drag: float,
    *,
    cd_friction: float,
    cd_wave_nose: float,
    cd_base: float,
    cd_transition_wave: float,
    a_ref: float,
) -> tuple[float, float]:
    """Supersonic streamlined-protuberance drag (i.cs:4034-4049).

    Same idea as the subsonic case, different term set -- and the difference
    is not cosmetic. Supersonically there is no form-drag term at all; nose
    wave drag replaces it, and transition wave drag joins as a new term that
    has no subsonic counterpart.

    Term identity, traced where each is assigned or summed:

    * ``cd_friction`` -- ``m_s``, i.cs:3926.
    * ``cd_wave_nose`` -- ``cy``, i.cs:1833-1834. Set inside the nose
      wave-drag routine, so it is the *nose's* wave drag alone, already
      normalised on A_ref.
    * ``cd_base`` -- ``m_r``, i.cs:3438-3453 (supersonic base drag, including
      the boattail scaling).
    * ``cd_transition_wave`` -- ``c2``, i.cs:4026-4027 and 4033: the sum of
      every part's ``an`` (Expansion wave drag) plus every part's ``ao``
      (Reducer wave drag), added together *before* reaching here.

    Note the summation order in the with-base-drag variant: base drag is added
    in the middle of the chain, ``((fric + wave) + base) + trans``, not at the
    end. Preserved.

    Returns ``(CD_s1, CD_s2)``, added into CD_prot at i.cs:4050.
    """
    if area_no_base_drag == 0:
        cd_no_base = 0.0                                    # i.cs:4036
    else:
        cd_no_base = (
            1.0
            * ((cd_friction + cd_wave_nose) + cd_transition_wave)
            * (area_no_base_drag / a_ref)
        )                                                   # i.cs:4040

    if area_with_base_drag == 0:
        cd_with_base = 0.0                                  # i.cs:4044
    else:
        cd_with_base = (
            1.0
            * (((cd_friction + cd_wave_nose) + cd_base) + cd_transition_wave)
            * (area_with_base_drag / a_ref)
        )                                                   # i.cs:4048
    return cd_no_base, cd_with_base
