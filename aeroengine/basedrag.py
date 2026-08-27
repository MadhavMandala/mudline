"""Base drag, both regimes, and the power-on credit.

Three unrelated pieces of physics that happen to share one output slot
(``i.m_r`` for base drag, ``i.m_i`` for the credit):

* **Subsonic** (``i.cs:3643``) is an empirical correlation on the *total*
  forward drag. Thicker boundary layers fill the wake and raise base
  pressure, so base drag falls as friction and form drag rise -- hence the
  reciprocal square root, and hence this function cannot be evaluated before
  its own siblings. It is written inline in the subsonic solver rather than
  in a method of its own; there is no ``i`` method to point at.

* **Supersonic** (``i.cs:3435-3456``) is a four-branch fit of the base
  *pressure coefficient*, with the top branch a hypersonic correlation
  referenced to post-shock rather than freestream conditions, followed by a
  boattail area scaling with a separation-angle clamp.

* **The power-on credit** (``i.cs:3561-3617``) is the only difference between
  power-off and power-on drag anywhere in RASAero. Everything else --
  friction, wave drag, fins, protuberances -- is identical between the two
  columns of a Run Test dump. It is a piecewise ladder in Mach applied over
  the *nozzle exit* area: the plume repressurises the part of the base it
  covers, and nothing else changes.

Sign conventions differ between the pieces and are easy to get backwards.
``base_pressure_coefficient`` returns Cp, which is negative. Both
``base_drag_*`` functions return a positive drag increment.
``power_on_credit`` returns a *negative* number, added to CD as-is
(``i.cs:3745``: ``m_u = m_n + m_i``).
"""

from __future__ import annotations

import math

from .compressible import normal_shock_pressure_ratio
from .geometry import DEG_TO_RAD_RA, PI_RA, GeometryCache
from .parts import Design

#: Boattail half-angle beyond which RASAero assumes the flow has separated
#: and stops believing the geometry. In the source this appears twice, once
#: as the literal 17.5 degrees compared against ``db`` and once already in
#: radians as 0.30543236111111105 (i.cs:3444, 3446). The radian literal is
#: bit-for-bit ``17.5 * DEG_TO_RAD_RA``, i.e. built from RASAero's pi, while
#: ``db`` itself was built from ``Math.PI`` at i.cs:1029. The mismatch is
#: real and both halves are reproduced as written.
_SEPARATION_ANGLE_DEG = 17.5
_SEPARATION_ANGLE_RAD = 17.5 * DEG_TO_RAD_RA
#: The clamp, for callers outside the engine that need to know where the
#: law stops believing the geometry -- the boattail caveat is keyed on it.
SEPARATION_ANGLE_DEG = _SEPARATION_ANGLE_DEG


# ---------------------------------------------------------------------------
# Supersonic base pressure coefficient
# ---------------------------------------------------------------------------


def base_pressure_coefficient(mach: float) -> float:
    """Cp on a blunt base, supersonic. Negative. i.cs:3437.

    The identical four-branch expression appears again at i.cs:3504 for the
    forward-facing step of an Expansion, so this is genuinely shared code in
    the original and not a coincidence of constants.

    Branch order is load-bearing and is *not* monotone in Mach: the ``<= 6``
    test is evaluated before the ``>= 7`` test, so the 6 < M < 7 window falls
    through to the final linear segment. Rewriting the chain as ascending
    thresholds gives the same answer here only because the segments were
    fitted to be continuous; the ordering is preserved anyway.
    """
    if mach <= 1.15:
        # Flat below the fit's range. The transonic solver never calls this
        # (base drag below M = 1.05 comes from the subsonic correlation), so
        # the constant only ever applies over 1.05 <= M <= 1.15.
        return -0.21

    if mach <= 6.0:
        return -1.0 * (
            0.34464
            - 0.14358 * mach
            + 0.024909 * mach ** 2.0
            - 0.0015993 * mach ** 3.0
        )

    if mach >= 7.0:
        # p_base / p_infinity = 0.7747 * M^-2.8 * (p2/p1), i.e. the base
        # pressure is correlated against conditions BEHIND the bow shock, not
        # against freestream. Cp = 2/(gamma M^2) * (p_base/p_inf - 1).
        # The coefficient is tuned so this meets the linear segment below at
        # M = 7 (both give -0.02363).
        return (
            2.0
            / (1.4 * mach ** 2.0)
            * (
                0.7747226539790469
                * (1.0 / mach) ** 2.8
                * normal_shock_pressure_ratio(mach)
                - 1.0
            )
        )

    # 6 < M < 7: a straight bridge between the cubic fit and the hypersonic
    # correlation, which do not otherwise meet.
    return -1.0 * (0.0344 + -0.0108 * (mach - 6.0))


# ---------------------------------------------------------------------------
# Subsonic
# ---------------------------------------------------------------------------


def base_drag_subsonic(
    cache: GeometryCache,
    cd_friction: float,
    cd_form: float,
) -> float:
    """Subsonic base drag, referenced to A_ref. i.cs:3643.

    ``cd_friction`` is ``i.m_s`` (skin friction coefficient times wetted area
    over A_ref) and ``cd_form`` is ``i.m_t`` (the body fineness-ratio form
    drag), both already normalised by A_ref. Passing them in rather than
    recomputing them is not a convenience: the correlation is
    ``0.029 (d_base/d_ref)^3 / sqrt(CD_friction + CD_form)`` and the coupling
    is the whole point of the formula.

    A trap for the caller: ``i.m_t`` is reused as scratch storage a few lines
    later (i.cs:3657, 3661) to hold a Power Law nose blending factor, so by
    the time the subsonic solver sums its terms at i.cs:3742 the value has
    been clobbered for power-law noses. Base drag is computed *before* that
    happens, so what belongs here is the fineness-ratio value from i.cs:3642.

    No guard on the denominator. A vehicle with zero friction and zero form
    drag divides by zero here, exactly as the original does.
    """
    return (
        0.029
        * (cache.d_base / cache.d_ref) ** 3.0
        / (cd_friction + cd_form) ** 0.5
        * (PI_RA * (cache.d_ref / 2.0) ** 2.0 / cache.a_ref)
    )


# ---------------------------------------------------------------------------
# Supersonic
# ---------------------------------------------------------------------------


def base_drag_supersonic(
    design: Design,
    cache: GeometryCache,
    mach: float,
) -> float:
    """Supersonic base drag, referenced to A_ref. i.cs:3435-3456.

    Two stages. First -Cp times the base-area ratio. Then, if the vehicle
    ends in something narrower than it is wide, a boattail scaling that
    *replaces* the first result rather than refining it.

    ``mach`` is the freestream Mach number: the only caller is i.cs:3927,
    inside the supersonic solver, which passes its own ``A_0``. It is not
    beta, and it is not the post-shock Mach.
    """
    cp_base = base_pressure_coefficient(mach)

    cd_base = -1.0 * cp_base
    cd_base = cd_base * (
        math.pi * (cache.d_base / 2.0) ** 2.0 / cache.a_ref
    )  # i.cs:3439 -- Math.PI here, literal pi at i.cs:3453 below

    # The comparison is against the LAST PART IN THE LIST, not the last
    # body-shaped part, and RASAero never sorts the list -- it appends in
    # file order (i.cs:652-734). A fin set or launch lug listed after the
    # boattail therefore supplies h() here, and a fin set carries the body
    # diameter at its own station, so the test can fire on a vehicle with no
    # boattail at all. Reproduced: parts[-1], whatever it is.
    last = design.parts[-1]
    if cache.d_base < last.d_fwd:  # i.cs:3442
        if cache.boattail_angle_deg > _SEPARATION_ANGLE_DEG:  # i.cs:3444
            # Steeper than the separation angle, so the geometry aft of the
            # separation point does no work. RASAero substitutes the aft
            # diameter a boattail of the SAME LENGTH would have had at
            # exactly 17.5 degrees. Nothing clamps this at zero: a long
            # enough boattail drives d_effective negative and the cube below
            # then makes the base drag negative.
            d_effective = -1.0 * (
                2.0 * math.tan(_SEPARATION_ANGLE_RAD) * last.length - last.d_fwd
            )  # i.cs:3446
            cd_base = (-1.0 * cp_base) * (
                d_effective / 2.0 / (last.d_fwd / 2.0)
            ) ** 3.0  # i.cs:3447
        else:
            cd_base = (-1.0 * cp_base) * (
                (cache.d_base / 2.0) / (last.d_fwd / 2.0)
            ) ** 3.0  # i.cs:3451

        # Both branches assign from -Cp again, so the base-area scaling
        # applied at i.cs:3439 is discarded whenever this block runs. Dead
        # work in the original; kept because removing it would mean deciding
        # the original meant something else.
        #
        # Note also which diameter closes the expression: dc, the boattail's
        # FORWARD diameter (cache.boattail_fwd_diameter), not d_base. And
        # note the pi: literal 3.14159 here against Math.PI at i.cs:3439.
        # dc is zero unless the last body-shaped part is a Reducer, so a
        # vehicle that trips the d_base < last.d_fwd test on a fin set gets
        # zero base drag rather than a wrong one.
        cd_base = cd_base * (
            PI_RA * (cache.boattail_fwd_diameter / 2.0) ** 2.0 / cache.a_ref
        )  # i.cs:3453

    return cd_base


def base_drag_supersonic_corrected(
    design: Design,
    cache: GeometryCache,
    mach: float,
) -> float:
    """A provisional replacement for the boattail branch. NOT RASAero.

    Same cylindrical-afterbody base pressure; the boattail's credit is the
    cube of the diameter ratio -- Hoerner's law, which the original also
    uses -- but closed with the geometric *base* area instead of the
    boattail's forward area, and with no separation clamp. Three defects of
    the original disappear by construction: the drag cannot go negative
    (the clamp's unfloored effective diameter could), a short steep boattail
    is not silently treated as a near-cylinder (the clamp charged the two
    highest flights on the scoreboard 72-77% of full-cylinder base drag,
    which flight data contradict by 8-13% of apogee), and a fin set listed
    after the boattail degrades to the plain area law instead of to zero.

    Provisional means provisional: the exponent and the absent clamp are
    the best fit to two measured flights, which is not a validation. The
    telemetry reconstruction in ``validation.telemetry`` is the instrument
    that will confirm or replace this form. Select it with
    ``Engine(design, boattail_model="corrected")``; the default everywhere
    remains the faithful port, and the oracle tests always run the port.
    """
    cp_base = base_pressure_coefficient(mach)
    a_base = math.pi * (cache.d_base / 2.0) ** 2.0 / cache.a_ref

    last = design.parts[-1]
    if 0.0 < cache.d_base < last.d_fwd:
        return (-1.0 * cp_base) * (cache.d_base / last.d_fwd) ** 3.0 * a_base
    return -1.0 * cp_base * a_base


# ---------------------------------------------------------------------------
# Power-on credit
# ---------------------------------------------------------------------------


def power_on_base_pressure_ladder(mach: float) -> float:
    """The bare piecewise ladder of i.cs:3563-3609. Positive.

    This is ``i.m_i`` as it stands before the area scaling, and RASAero holds
    it with the sign already flipped: it is -Cp_base, so it comes out
    positive (0.0547 to 0.1607 over the tabulated range). The caller negates
    it again at i.cs:3615 to turn it back into a credit.

    It is *not* the same fit as :func:`base_pressure_coefficient`. Different
    breakpoints (4.387 and 7.5 rather than 6 and 7), different tabulated
    values, and eight straight segments below M = 4.387 where the power-off
    fit uses a single cubic. Do not unify them.

    The segments are continuous by construction, including across the two
    hypersonic branches: the blend factor is 1.4175568973525314 at M = 4.387,
    which lands the vacuum-limit correlation exactly on the 0.09-at-3.85
    segment, and decays to exactly 1.0 at M = 7.5.
    """
    if mach >= 7.5:
        value = (
            2.0
            / (1.4 * mach ** 2.0)
            * (
                0.7747226539790469
                * (1.0 / mach) ** 2.8
                * normal_shock_pressure_ratio(mach)
                - 1.0
            )
        )  # i.cs:3565
        return -1.0 * value

    if mach >= 4.387:
        value = (
            2.0
            / (1.4 * mach ** 2.0)
            * (
                0.7747226539790469
                * (1.0 / mach) ** 2.8
                * normal_shock_pressure_ratio(mach)
                - 1.0
            )
        )  # i.cs:3570
        value = -1.0 * value
        # Linear ramp from 1.41756 down to 1.0, stitching the correlation to
        # the tabulated curve at 4.387 and releasing it at 7.5.
        return (
            -0.13413327894395483 * (mach - 4.387) + 1.4175568973525314
        ) * value  # i.cs:3572

    if mach >= 3.85:
        return 0.09 + (mach - 3.85) * -0.025512104        # i.cs:3576
    if mach >= 3.0:
        return 0.1236 + (mach - 3.0) * -0.039529412       # i.cs:3580
    if mach >= 2.45:
        return 0.166 + (mach - 2.45) * -0.077090909       # i.cs:3584
    if mach >= 1.97:
        return 0.1478 + (mach - 1.97) * 0.037916667       # i.cs:3588
    if mach >= 1.1935:
        return 0.1478                                     # i.cs:3592
    if mach >= 1.0:
        return 0.1607 + (mach - 1.0) * -0.066666667       # i.cs:3596
    if mach >= 0.925:
        return 0.0931 + (mach - 0.925) * 0.901333333      # i.cs:3600
    if mach >= 0.58:
        return 0.0547 + (mach - 0.58) * 0.111304348       # i.cs:3604
    return 0.0547                                         # i.cs:3608


def power_on_credit(
    design: Design,
    cache: GeometryCache,
    mach: float,
) -> float:
    """The power-on drag credit, referenced to A_ref. Negative. i.cs:3561-3617.

    Added to power-off CD unchanged (i.cs:3745, 4054:
    ``m_u = m_n + m_i``) and it is the *entire* difference between the two
    drag columns. If this returns zero, power-on and power-off CD are
    identical to the last digit.

    ``mach`` is freestream Mach; both callers (i.cs:3744 subsonic, i.cs:4053
    supersonic) pass their solver's own ``A_0``. The same ladder therefore
    serves both regimes -- unlike base drag itself, the credit is not
    regime-switched, which is why the ladder has to cover M < 1 at all.

    The credit is evaluated unconditionally, whether or not the user asked
    for power-on output. It goes to zero on its own when
    ``design.nozzle_diameter`` is zero, since that is the area it acts over.
    """
    ladder = power_on_base_pressure_ladder(mach)

    # i.cs:3610 -- the ONLY guard in the routine, and it is on the base
    # diameter, not the nozzle. A nozzle wider than the base is accepted and
    # yields a credit larger in magnitude than the base drag it is crediting
    # against; a nozzle on a vehicle with no base short-circuits to zero.
    if cache.d_base == 0.0:
        return 0.0

    # Nozzle exit area over base area, then base area over A_ref. The base
    # area very nearly cancels -- but not exactly, because the first ratio's
    # denominator uses the literal 3.14159 while both numerators use Math.PI.
    # Left uncancelled deliberately: the residue is the pi mismatch, and the
    # program's A_ref (cache.a_ref) is itself built from the literal pi, so
    # the surviving factor is not 1.
    credit = (-1.0 * ladder) * (
        math.pi * (design.nozzle_diameter / 2.0) ** 2.0
        / (PI_RA * (cache.d_base / 2.0) ** 2.0)
    )  # i.cs:3615
    credit = credit * (
        math.pi * (cache.d_base / 2.0) ** 2.0 / cache.a_ref
    )  # i.cs:3616
    return credit
