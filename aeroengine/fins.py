"""Pass B: the fin set.

This is the largest single piece of RASAero's aerodynamics and the one with the
most opportunity for a silent transcription error, so read the field map below
before reading any formula.

Three geometry sets
-------------------
RASAero carries a fin panel three times over, and almost every formula in the
module picks one of the three. Getting the wrong one is not a rounding error:
the sets differ by the body radius under the root, which is often comparable to
the exposed span.

``panel``  -- the five dimensions as entered (root chord, tip chord, sweep,
    span, station). Used for the *virtual* planform that extends the fin to the
    body centreline, for the leading-edge sweep angle, for the section stations,
    and by a handful of drag terms that test ``d_fwd >= d_aft`` rather than
    ``d_fwd == d_aft`` (see below).

``primed`` (``p_*``) -- the panel corrected for a body whose diameter changes
    under the root. The exposed span is grown by ``(d_fwd - d_equiv)/2`` --
    where ``d_equiv`` is the equal-volume diameter from the geometry cache --
    and the root chord, sweep and station are slid along the leading and
    trailing edges to suit. When ``d_fwd == d_aft`` this set is copied verbatim
    from the panel, so the primed set is the one to reach for by default.

``double primed`` (``pp_*``) -- the primed set with the nose-shock standoff cut
    off the root: supersonically the inboard part of the fin sits inside the
    nose shock and does not see freestream. Used **only** by the supersonic
    CN-alpha and CP routines.

Field map (the shadowing trap)
------------------------------
``az`` redeclares most of ``aq``'s fields with ``new``, so ``az3.ag`` and
``((aq)az3).ag`` are different variables that appear *in the same expression*
at i.cs:1366. The bindings actually in force inside class ``i``:

===============  ==========================  ==================================
source           meaning                     here
===============  ==========================  ==================================
``az.b/c/d/e``   root, tip, sweep, span      ``root_chord`` ...
``i() c() h()``  station, length, d_fwd      ``fin.x0``, --, ``fin.d_fwd``
``b()``          d_aft                       ``fin.d_aft``
``f()``          per-part turbulence flag    ``turbulent_flag``
``az.k``/``l``   panel area / virtual area   ``panel_area`` / ``virtual_area``
``az.m``/``n``   LE sweep / max-t sweep      ``le_sweep`` / ``max_thickness_sweep``
``az.ae``        panel mean chord            ``mean_chord``      (**base**)
``az.o``/``p``   aspect / taper ratio        ``p_aspect_ratio`` / ``p_taper_ratio``
``az.r``         thickness / primed MAC      ``thickness_ratio``
``az.s/t/u``     three section stations      ``x_le`` / ``x_mid`` / ``x_te``
``az.x``         wave-drag shape factor K    ``wave_drag_k``
``aq.ad``        primed station              ``p_x0``
``aq.ae/af``     primed root / tip chord     ``p_root_chord`` / ``p_tip_chord``
``aq.ag/ah``     primed sweep / span         ``p_sweep`` / ``p_span``
``aq.ai``        equal-volume body diameter  ``equivalent_diameter``
``aq.aj``        primed mean chord           ``p_mean_chord``
``aq.am``        primed area                 ``p_area``
``az.ag``        nose-shock standoff         ``shock_standoff``  (**float, shadows aq.ag**)
``az.ah..ap``    the double-primed set       ``pp_*``
===============  ==========================  ==================================

``az.y``, ``az.z``, ``az.aa``, ``az.ab``, ``az.ac`` are declared ``private new``
in ``az`` and are therefore *inaccessible* from class ``i``; every ``az3.y`` in
the source binds to the inherited ``aq.y``. That is why the assembly loop can
read them back through an ``aq``-typed reference at i.cs:3973.

Precision
---------
RASAero mixes ``decimal``, ``double`` and ``float`` freely, and this module
follows the convention already set by :mod:`.geometry`: everything is a Python
float. The single-precision truncations (i.cs:1317-1323, 2553, 3080-3110, and
the whole of the supersonic CP routine) are therefore *not* reproduced; they
bite at the seventh significant figure, below RASAero's printed precision, but
they are the first place to look if a fin term disagrees in the last digit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .compressible import beta, cp_stagnation
from .geometry import DEG_TO_RAD_RA, PI_RA, RAD_TO_DEG_RA, GeometryCache
from .parts import WEDGE_SECTIONS, Airfoil, Fins

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: Sections whose aft face is cut off square, so they carry a base drag and an
#: edge drag that the closed sections do not. i.cs:2555, 2735.
_BLUNT_BASE_SECTIONS = frozenset({
    Airfoil.SINGLE_WEDGE,
    Airfoil.HEXAGONAL_BLUNT_BASE,
    Airfoil.SQUARE,
    Airfoil.ROUNDED,
})

#: Sections with no analytic wave-drag solution, treated as empirical drag
#: coefficients rather than 2*Cf*form_factor. i.cs:2508, 2530, 2577.
_EMPIRICAL_SECTIONS = frozenset({Airfoil.ROUNDED, Airfoil.SQUARE})

#: Ratio of the normal force of N fins at incidence to that of 4 fins, summed
#: over roll orientation. 4 fins is unity because RASAero's charts are built
#: for a cruciform tail. i.cs:2843-2863 and repeated at 3082-3111, 3293-3370,
#: 3389-3412 -- the same six numbers written out four times.
_FIN_COUNT_FACTOR: dict[int, float] = {
    3: 0.866,
    4: 1.0,
    5: 1.5388,
    6: 1.7321,
    7: 2.1906,
    8: 2.4142,
}

#: 5.33333333333333 as RASAero spells 16/3, the linear-theory wave-drag shape
#: factor of a biconvex (parabolic-arc) section. Written with fourteen 3s, not
#: sixteen, and the truncation is visible in the third decimal of Cd_wave at
#: high thickness ratio, so it is kept exactly as typed.
_K_BICONVEX = 5.33333333333333

#: The incidence RASAero evaluates every modified-Newtonian derivative at:
#: four degrees, in radians, via its own 3.14159/180. i.cs:3284, m_f at 4097.
_ALPHA_REF_RAD = 0.0698131111111111

#: The same four degrees as a single-precision literal. It is compared against
#: the wedge half-angle at i.cs:3282 and differs from _ALPHA_REF_RAD in the
#: eighth digit; both appear within three lines of each other.
_ALPHA_REF_F32 = 0.06981311

#: ...and a third spelling, six digits, used inside sin() at i.cs:3292/3387.
_ALPHA_REF_SIN = 0.069813


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FinGeometry:
    """Everything i.cs:1301-1448 derives once per design for one fin set.

    Field names carry the set they belong to: no prefix is the panel as
    entered, ``p_`` is the primed (equivalent-diameter) set, ``pp_`` is the
    double-primed (shock-corrected) set. See the module docstring.
    """

    # --- panel, as entered -------------------------------------------------
    root_chord: float           # az.b
    tip_chord: float            # az.c
    sweep: float                # az.d -- LE axial sweep distance, not an angle
    span: float                 # az.e -- exposed semispan of one panel
    x0: float                   # i()
    te_sweep: float             # local at i.cs:1305 -- the TE's axial sweep
    panel_area: float           # az.k
    virtual_area: float         # az.l -- panel extended to the body centreline
    mean_chord: float           # az.ae -- ALWAYS the base panel's, see below
    le_sweep: float             # az.m, rad, from the spanwise direction
    max_thickness_sweep: float  # az.n, rad

    # --- primed: corrected to the equal-volume body diameter ---------------
    p_root_chord: float         # aq.ae
    p_tip_chord: float          # aq.af
    p_sweep: float              # aq.ag
    p_span: float               # aq.ah
    p_x0: float                 # aq.ad
    p_area: float               # aq.am
    p_mean_chord: float         # aq.aj
    p_aspect_ratio: float       # az.o -- of the two-panel wing, (2b)^2/(2S)
    p_taper_ratio: float        # az.p -- NOT clamped, unlike pp_taper_ratio

    # --- double primed: further corrected for the nose shock standoff ------
    pp_root_chord: float        # az.ah
    pp_tip_chord: float         # az.ai
    pp_sweep: float             # az.aj
    pp_span: float              # az.ak
    pp_x0: float                # az.al
    pp_area: float              # az.am
    pp_aspect_ratio: float      # az.an
    pp_taper_ratio: float       # az.ao, clamped to <= 1
    pp_mean_chord: float        # az.ap

    # --- section and flags -------------------------------------------------
    thickness_ratio: float      # az.r -- thickness / PRIMED mean chord
    x_le: float                 # az.s -- LE to max thickness
    x_mid: float                # az.t -- constant-thickness run
    x_te: float                 # az.u -- max thickness to TE
    wave_drag_k: float          # az.x
    turbulent_flag: float       # aq.f -- -1 means "skip laminar", see derive()
    equivalent_diameter: float  # aq.ai
    shock_standoff: float       # az.ag


@dataclass(frozen=True)
class FinDrag:
    """Zero-lift drag of one fin set, all referenced to ``cache.a_ref``."""

    cd: float               # aq.n subsonically, aq.i supersonically
    cd_interference: float  # aq.h -- the buried planform inside the body
    cd_base: float          # aq.j -- 0 except for the square-based sections


@dataclass(frozen=True)
class FinNormalForce:
    """CN-alpha of one fin set, per radian, on ``cache.a_ref``."""

    cn_alpha: float       # aq.p -- panel force incl. wing-body interference
    cn_alpha_body: float  # aq.q -- body carryover; 0 in classical Barrowman


@dataclass(frozen=True)
class FinCenterOfPressure:
    """Centres of pressure for the two CN-alpha terms, inches from the tip."""

    cp: float       # aq.s
    cp_body: float  # aq.t


@dataclass(frozen=True)
class FinNewtonian:
    """The modified-Newtonian alternative to the linear-theory pair."""

    cn_alpha: float  # aq.ap
    cp: float        # aq.aq


@dataclass(frozen=True)
class FinWaveDragLinear:
    """Linear-theory wave drag, plus the Mach it was bracketed against."""

    cd_wave: float        # aq.y
    le_sonic_mach: float  # aq.aa -- sec(sweep): where the LE goes supersonic


# ---------------------------------------------------------------------------
# B.0 -- derived geometry and the airfoil section (i.cs:1301-1448)
# ---------------------------------------------------------------------------


def derive(
    fin: Fins,
    cache: GeometryCache,
    index: int,
    *,
    turbulent_flow: bool = False,
) -> FinGeometry:
    """Run i.cs:1301-1448 for the fin set at ``index`` in the part list.

    ``index`` keys the two per-fin-set values Pass A already computed: the
    equal-volume body diameter under the root (``aq.ai``) and the nose-shock
    standoff (``az.ag``).
    """
    root_chord = fin.root_chord      # az.b
    tip_chord = fin.tip_chord        # az.c
    sweep = fin.sweep                # az.d, clamped to 0.01 at construction
    span = fin.span                  # az.e
    d_fwd = fin.d_fwd                # h()
    d_aft = fin.d_aft                # b()
    d_equiv = cache.fin_equiv_diameter[index]     # aq.ai
    standoff = cache.fin_shock_standoff[index]    # az.ag

    # The trailing edge's own axial sweep: how far aft of the root TE the tip TE
    # sits. Negative -- a forward-swept TE -- whenever the tip chord plus the LE
    # sweep falls short of the root chord, which is the usual clipped delta with
    # a straight TE. i.cs:1305.
    te_sweep = sweep + tip_chord - root_chord

    # num3 is declared outside the branch at i.cs:1306 and read again at 1332,
    # so it has to survive a constant-diameter fin set as zero.
    num3 = 0.0

    if d_fwd == d_aft:
        # i.cs:1307-1314. Constant diameter: the primed set is the panel.
        p_root_chord = root_chord
        p_tip_chord = tip_chord
        p_sweep = sweep
        p_span = span
        p_x0 = fin.x0
    else:
        # i.cs:1317-1323. Grow the panel inboard (or trim it) by the difference
        # between the local forward diameter and the equal-volume diameter,
        # sliding the root along the LE and TE lines to keep both straight.
        # num2/num3 are the axial distances the LE and TE move.
        if span == 0.0 or sweep == 0.0:
            num2 = 0.0
        else:
            num2 = ((d_fwd - d_equiv) / 2.0) / (span / sweep)
        if span == 0.0 or te_sweep == 0.0:
            num3 = 0.0
        else:
            num3 = ((d_fwd - d_equiv) / 2.0) / (span / te_sweep)
        p_root_chord = num2 + root_chord - num3
        p_tip_chord = tip_chord
        p_sweep = num2 + sweep
        p_span = span + (d_fwd - d_equiv) / 2.0
        p_x0 = fin.x0 - num2

    # i.cs:1325. Algebraically (Cr + Ct)/2 * b, but written the long way round
    # via the TE sweep and left that way.
    panel_area = (
        (sweep / 2.0) + (te_sweep / 2.0) + (tip_chord - te_sweep)
    ) * span

    if d_fwd == d_aft:
        p_area = panel_area                                  # i.cs:1328
    else:
        # i.cs:1332. Same construction on the primed dimensions; the primed
        # TE sweep is num3 + te_sweep.
        p_area = (
            (p_sweep / 2.0)
            + (num3 + te_sweep) / 2.0
            + (p_tip_chord - (num3 + te_sweep))
        ) * p_span

    # i.cs:1334. The panel continued inboard to the body centreline, on the
    # *forward* diameter and the *base* sweeps -- this is the area the
    # fin-body interference drag is charged against, and it is the term whose
    # being zero disables the whole fin set downstream.
    virtual_area = panel_area + (
        ((sweep / span) * (d_fwd / 2.0)) / 2.0
        - ((te_sweep / span) * (d_fwd / 2.0)) / 2.0
        + root_chord
    ) * (d_fwd / 2.0)

    le_sweep = math.atan(sweep / span)                       # az.m, i.cs:1335

    # Chordwise position of maximum thickness as a fraction of the *average*
    # chord, which is what sets the sweep of the max-thickness line. i.cs:1337.
    if fin.airfoil in WEDGE_SECTIONS:
        thickness_fraction = fin.fx1 / ((root_chord + tip_chord) / 2.0)
    else:
        thickness_fraction = 0.0
    if fin.airfoil is Airfoil.SUBSONIC_NACA:
        thickness_fraction = 0.25                            # i.cs:1343
    if fin.airfoil in _EMPIRICAL_SECTIONS:
        thickness_fraction = 0.0                             # i.cs:1347

    # i.cs:1349. On a forward-swept panel the sweep term is negated but the
    # thickness terms are not, so the max-thickness line is NOT the mirror of
    # the aft-swept case. Reproduced; it feeds cos() in the subsonic form
    # factor, where the sign does not matter, and the supersonic drag-rise
    # Mach, where it does.
    if le_sweep < 0.0:
        max_thickness_tan = (
            (-1.0 * sweep)
            + thickness_fraction * tip_chord
            - thickness_fraction * root_chord
        ) / span
    else:
        max_thickness_tan = (
            sweep
            + thickness_fraction * tip_chord
            - thickness_fraction * root_chord
        ) / span
    max_thickness_sweep = math.atan(max_thickness_tan)       # az.n

    if d_fwd == d_aft:
        # i.cs:1353-1356.
        p_aspect_ratio = (2.0 * span) ** 2.0 / (2.0 * panel_area)
        p_taper_ratio = tip_chord / root_chord
        mean_chord = panel_area / span
        p_mean_chord = mean_chord
    else:
        # i.cs:1360-1363. Note az.ae (mean_chord) is the *panel's* mean chord
        # in both branches -- only aq.aj gets the primed value. Every airfoil
        # station below is therefore laid out on the uncorrected chord.
        p_aspect_ratio = (2.0 * p_span) ** 2.0 / (2.0 * p_area)
        p_taper_ratio = p_tip_chord / p_root_chord
        mean_chord = panel_area / span
        p_mean_chord = p_area / p_span

    # --- double primed: cut the nose-shock standoff off the root -----------
    # i.cs:1365-1367. num5/num6 are the axial distances the LE and TE move when
    # the root is raised by the standoff. The `standoff /` here is az.ag, the
    # float shadow field, divided by a ratio of aq.ah and aq.ag -- three
    # different "ag/ah" in one expression, which is the reason for the field
    # map in the module docstring.
    p_te_sweep = p_sweep + p_tip_chord - p_root_chord
    if p_span == 0.0 or p_sweep == 0.0:
        num5 = 0.0
    else:
        num5 = standoff / (p_span / p_sweep)
    if p_span == 0.0 or p_te_sweep == 0.0:
        num6 = 0.0
    else:
        num6 = standoff / (p_span / p_te_sweep)

    pp_root_chord = p_root_chord - num5 + num6               # i.cs:1368
    pp_tip_chord = p_tip_chord                               # i.cs:1369
    pp_sweep = p_sweep - num5                                # i.cs:1370
    pp_span = p_span - standoff                              # i.cs:1371
    pp_x0 = p_x0 + num5                                      # i.cs:1372
    # (p_te_sweep - num6) is the double-primed TE sweep, the same construction
    # as the primed area above.
    pp_area = (
        pp_sweep / 2.0
        + (p_te_sweep - num6) / 2.0
        + (pp_tip_chord - (p_te_sweep - num6))
    ) * pp_span                                              # i.cs:1373
    pp_aspect_ratio = (2.0 * pp_span) ** 2.0 / (2.0 * pp_area)  # i.cs:1374
    pp_taper_ratio = pp_tip_chord / pp_root_chord            # i.cs:1375
    pp_mean_chord = pp_area / pp_span                        # i.cs:1376
    if pp_taper_ratio > 1.0:
        # i.cs:1377-1380. Only the double-primed taper ratio is clamped; the
        # primed one at az.p is left free. The clamp makes the `> 1` arms of
        # the supersonic CN-alpha chart (i.cs:3016, 3041) unreachable.
        pp_taper_ratio = 1.0

    thickness_ratio = fin.thickness / p_mean_chord           # az.r, i.cs:1381

    # i.cs:1066 seeds every part's flag from the global Turbulent Flow setting;
    # i.cs:1384 then forces it on for a square section regardless. That makes a
    # square fin take the roughness penalty supersonically even with turbulence
    # switched off -- see supersonic_friction_drag().
    turbulent_flag = -1.0 if turbulent_flow else 0.0
    if fin.airfoil is Airfoil.SQUARE:
        turbulent_flag = -1.0

    # --- the section: three chordwise stations and the shape factor K ------
    # i.cs:1386-1446. x_le/x_te start as the user's FX1/FX3 and are overwritten
    # for every section that defines them geometrically; x_mid and K start at
    # zero because they are plain fields on a fresh object.
    x_le = fin.fx1
    x_te = fin.fx3
    x_mid = 0.0
    wave_drag_k = 0.0

    if fin.airfoil is Airfoil.DOUBLE_WEDGE:
        if x_le == mean_chord:
            # Guards against K's 1 - x_le/c denominator vanishing.
            x_le = mean_chord - 0.01                         # i.cs:1393
        if x_le == 0.0:
            # A file that never touched FX1 lands here, and K = (c/0.01) /
            # (1 - 0.01/c) is then enormous rather than absent. Not a fix:
            # RASAero's own dialog defaults FX1 to 0.0000.
            x_le = 0.01                                      # i.cs:1397
        x_te = mean_chord - x_le
        x_mid = 0.0
        wave_drag_k = (mean_chord / x_le) / (1.0 - x_le / mean_chord)
    elif fin.airfoil is Airfoil.BICONVEX:
        wave_drag_k = _K_BICONVEX                            # i.cs:1404
        x_le = mean_chord / 2.0
        x_te = x_le
        x_mid = 0.0
    elif fin.airfoil is Airfoil.SUBSONIC_NACA:
        wave_drag_k = _K_BICONVEX                            # i.cs:1410
    elif fin.airfoil is Airfoil.HEXAGONAL:
        if x_le == 0.0:
            x_le = 0.01                                      # i.cs:1415
        if x_te == 0.0:
            x_te = 0.01                                      # i.cs:1419
        x_mid = mean_chord - (x_le + x_te)
        wave_drag_k = (mean_chord * (mean_chord - x_mid)) / (x_le * x_te)
    elif fin.airfoil is Airfoil.SINGLE_WEDGE:
        wave_drag_k = 1.0                                    # i.cs:1425
        x_le = mean_chord
        x_te = 0.0
        x_mid = 0.0
    elif fin.airfoil is Airfoil.HEXAGONAL_BLUNT_BASE:
        if x_le == 0.0:
            x_le = 0.01                                      # i.cs:1433
        x_mid = mean_chord - 2.0 * x_le
        wave_drag_k = (mean_chord * (mean_chord - x_mid)) / (x_le * x_le)
        x_te = 0.0
        # x_mid is reassigned after K has consumed it: K is computed on the
        # symmetric hexagon, then the section is squared off at the base and
        # x_mid becomes everything aft of the forward wedge. i.cs:1435/1438.
        x_mid = mean_chord - x_le
    elif fin.airfoil is Airfoil.ROUNDED:
        wave_drag_k = _K_BICONVEX                            # i.cs:1441
    elif fin.airfoil is Airfoil.SQUARE:
        wave_drag_k = _K_BICONVEX                            # i.cs:1444

    return FinGeometry(
        root_chord=root_chord,
        tip_chord=tip_chord,
        sweep=sweep,
        span=span,
        x0=fin.x0,
        te_sweep=te_sweep,
        panel_area=panel_area,
        virtual_area=virtual_area,
        mean_chord=mean_chord,
        le_sweep=le_sweep,
        max_thickness_sweep=max_thickness_sweep,
        p_root_chord=p_root_chord,
        p_tip_chord=p_tip_chord,
        p_sweep=p_sweep,
        p_span=p_span,
        p_x0=p_x0,
        p_area=p_area,
        p_mean_chord=p_mean_chord,
        p_aspect_ratio=p_aspect_ratio,
        p_taper_ratio=p_taper_ratio,
        pp_root_chord=pp_root_chord,
        pp_tip_chord=pp_tip_chord,
        pp_sweep=pp_sweep,
        pp_span=pp_span,
        pp_x0=pp_x0,
        pp_area=pp_area,
        pp_aspect_ratio=pp_aspect_ratio,
        pp_taper_ratio=pp_taper_ratio,
        pp_mean_chord=pp_mean_chord,
        thickness_ratio=thickness_ratio,
        x_le=x_le,
        x_mid=x_mid,
        x_te=x_te,
        wave_drag_k=wave_drag_k,
        turbulent_flag=turbulent_flag,
        equivalent_diameter=d_equiv,
        shock_standoff=standoff,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _degenerate(fin: Fins, geom: FinGeometry) -> bool:
    """The guard repeated at the head of all ten fin routines.

    i.cs:2475 etc. Tests the *areas*, not the entered dimensions, so a fin set
    with a root chord and a span but zero of both diameters (hence zero virtual
    area) is silently zeroed too.
    """
    return (
        fin.count == 0
        or geom.panel_area == 0.0
        or geom.virtual_area == 0.0
    )


def _roughness_factor(
    reynolds_body: float, p_mean_chord: float, length_body: float
) -> float:
    """Surface-roughness multiplier on an empirical section's drag.

    i.cs:2513-2528, and written out three more times at 2535, 2584, 2606. The
    2.7x at low Reynolds number decays linearly to 1.0 at Re = 2.58e6, which is
    the cutoff Reynolds number the body's own friction routine uses.

    The floor at i.cs:2514 is ``10000 * (chord/L_body)``, not 10000: it scales
    the intended floor by the chord ratio a second time, so for a fin much
    shorter than the body it never binds. Reproduced.
    """
    re_chord = reynolds_body * (p_mean_chord / length_body)
    floor = 10000.0 * (p_mean_chord / length_body)
    if re_chord < floor:
        re_chord = floor
    if re_chord <= 500000.0:
        return 2.7
    if re_chord < 2580000.0:
        return 2.7 + -8.173076923076924e-07 * (re_chord - 500000.0)
    return 1.0


def _empirical_rounded_cd(geom: FinGeometry, fin: Fins) -> float:
    """Rounded-section pressure drag. i.cs:2509-2510 and 2581.

    Linear in thickness ratio about 4%, less the base drag, which is added back
    as a separate term so it can be Mach-corrected on its own.
    """
    cd = 0.02 + 0.02 * ((geom.thickness_ratio - 0.04) / 0.04)
    return cd - 0.29 * (fin.thickness * geom.p_span / geom.p_area)


def _empirical_square_cd(geom: FinGeometry, fin: Fins) -> float:
    """Square-section pressure drag. i.cs:2531-2532 and 2603."""
    cd = 0.028 + 0.06 * ((geom.thickness_ratio - 0.032) / 0.068)
    return cd - 0.29 * (fin.thickness * geom.p_span / geom.p_area)


def _le_bluntness_cd(
    mach: float,
    le_sweep: float,
    diameter: float,
    span: float,
    area: float,
    count: int,
    a_ref: float,
) -> float:
    """Drag of a blunt leading edge of diameter ``diameter``.

    i.cs:2701-2703, repeated verbatim six more times. The 1.28 M^3 cos^6 /
    (1 + M^3 cos^3) group is a cylinder's crossflow drag faired from the
    incompressible 1.28 to the Newtonian limit, on the sweep-normal Mach.

    ``span`` and ``area`` are the caller's choice of the base or primed set --
    the source picks differently between the linear-theory and the
    shock-expansion routines, so it is not safe to fix one here.
    """
    cd = (
        1.28
        * mach ** 3.0
        * math.cos(le_sweep) ** 6.0
        / (1.0 + mach ** 3.0 * math.cos(le_sweep) ** 3.0)
    )
    cd = cd * (
        diameter * (2.0 * span / math.cos(le_sweep)) / (2.0 * area)
    )
    return cd * ((2.0 * area / a_ref) * (count / 2.0))


# ---------------------------------------------------------------------------
# B.1 -- subsonic fin drag (i.cs:2471-2566)
# ---------------------------------------------------------------------------


def subsonic_drag(
    fin: Fins,
    geom: FinGeometry,
    cache: GeometryCache,
    *,
    mach: float,
    cf_fin: float,
    reynolds_body: float,
    length_body: float,
    turbulent_flow: bool,
) -> FinDrag:
    """i.cs:2471-2566.

    ``cf_fin`` is ``i.m_x``, the fin's flat-plate skin friction coefficient
    computed on the *primed* mean chord by the routine at i.cs:1550.
    ``reynolds_body`` and ``length_body`` are the caller's ``bc`` and ``be``:
    the vehicle Reynolds number and the body length, from which the fin's own
    Reynolds number is rebuilt by chord ratio.

    ``turbulent_flow`` is the *global* setting, not the per-part flag. The
    supersonic routine tests the per-part flag instead; the difference is
    load-bearing for a square section (see supersonic_friction_drag).
    """
    if _degenerate(fin, geom):
        return FinDrag(0.0, 0.0, 0.0)

    # i.cs:2483. Subsonic form/interference factor: flat 1.044 below M 0.25,
    # rising through transonic, and identically 1.0 above M 0.9 -- note the
    # M > 0.9 arm is tested *before* the 0.6-0.9 arm, so 0.9 itself takes the
    # interpolation, not the flat value.
    cos_n = math.cos(geom.max_thickness_sweep)
    if mach <= 0.25:
        form = 1.044 * cos_n ** 0.28
    elif mach <= 0.6:
        form = (1.044 + 0.231 * (mach - 0.25)) * cos_n ** 0.28
    elif mach > 0.9:
        form = 1.0
    else:
        form = (1.125 + 0.634 * (mach - 0.6)) * cos_n ** 0.28

    r = geom.thickness_ratio
    cd = 0.0

    if fin.airfoil in (
        Airfoil.DOUBLE_WEDGE,
        Airfoil.HEXAGONAL,
        Airfoil.HEXAGONAL_BLUNT_BASE,
    ):
        # i.cs:2490. The switch is on the *entered* FX1 over the base mean
        # chord, not on the clamped station x_le -- so a hexagonal section
        # whose FX1 was defaulted to zero takes the forward-max-thickness
        # (1 + 2t/c) form factor even though derive() moved its station to
        # 0.01 in.
        if (fin.fx1 / geom.mean_chord) < 0.3:
            cd = 2.0 * cf_fin * ((1.0 + 2.0 * r) + 100.0 * r ** 4.0)
        else:
            cd = 2.0 * cf_fin * (1.0 + 1.2 * r + 100.0 * r ** 4.0)
    elif fin.airfoil is Airfoil.SUBSONIC_NACA:
        cd = 2.0 * cf_fin * ((1.0 + 2.0 * r) + 100.0 * r ** 4.0)   # i.cs:2502
    elif fin.airfoil in (Airfoil.BICONVEX, Airfoil.SINGLE_WEDGE):
        cd = 2.0 * cf_fin * (1.0 + 1.2 * r + 100.0 * r ** 4.0)     # i.cs:2506
    elif fin.airfoil is Airfoil.ROUNDED:
        cd = _empirical_rounded_cd(geom, fin)
        if turbulent_flow:                                          # i.cs:2511
            cd = cd * _roughness_factor(
                reynolds_body, geom.p_mean_chord, length_body
            )
    elif fin.airfoil is Airfoil.SQUARE:
        cd = _empirical_square_cd(geom, fin)
        if turbulent_flow:                                          # i.cs:2533
            cd = cd * _roughness_factor(
                reynolds_body, geom.p_mean_chord, length_body
            )

    # i.cs:2553. Referenced to A_ref on the primed panel area, then the form
    # factor is applied *after* the area ratio, so it scales the referenced
    # coefficient rather than the sectional one. Same number either way, but
    # the source truncates to single precision between the two steps.
    cd = (cd * (geom.p_area * fin.count / cache.a_ref)) * form

    # i.cs:2554. The planform buried inside the body, charged at the same
    # coefficient. The fin count cancels; it is written out both times.
    cd_interference = cd * (
        (geom.virtual_area - geom.p_area) * fin.count
    ) / (geom.p_area * fin.count)

    if fin.airfoil in _BLUNT_BASE_SECTIONS:
        # i.cs:2557-2558. A base pressure coefficient of -0.29 on the trailing
        # edge area, negated to a drag. This is the term subtracted out of the
        # empirical Rounded/Square coefficients above, added back here so it is
        # reported separately -- and added for Single Wedge and Hexagonal Blunt
        # Base too, which never subtracted it.
        cd_base = -1.0 * (-0.29) * (
            (fin.thickness * geom.p_span) * fin.count
        ) / cache.a_ref
    else:
        cd_base = 0.0

    return FinDrag(cd, cd_interference, cd_base)


# ---------------------------------------------------------------------------
# B.2 -- supersonic fin skin friction (i.cs:2568-2644)
# ---------------------------------------------------------------------------


def supersonic_friction_drag(
    fin: Fins,
    geom: FinGeometry,
    cache: GeometryCache,
    *,
    mach: float,
    cf_fin: float,
    reynolds_body: float,
    length_body: float,
) -> FinDrag:
    """i.cs:2568-2644. ``cd_base`` is always 0 here; see supersonic_edge_drag.

    The Rounded and Square sections do not use ``cf_fin`` at all: their drag is
    the empirical pressure coefficient, Mach-corrected with the same
    compressibility factors the flat-plate friction uses.
    """
    if _degenerate(fin, geom):
        return FinDrag(0.0, 0.0, 0.0)

    if fin.airfoil in _EMPIRICAL_SECTIONS:
        if fin.airfoil is Airfoil.ROUNDED:
            cd = _empirical_rounded_cd(geom, fin)                  # i.cs:2581
            if geom.turbulent_flag == -1.0:                        # i.cs:2582
                cd = cd * _roughness_factor(
                    reynolds_body, geom.p_mean_chord, length_body
                )
        else:
            cd = _empirical_square_cd(geom, fin)                   # i.cs:2603
            # This test is on the PER-PART flag, which derive() forced to -1
            # for every square section at i.cs:1384. A square fin therefore
            # always takes the 2.7x roughness penalty supersonically, even with
            # Turbulent Flow switched off -- unlike subsonically, where the
            # same block tests the global setting.
            if geom.turbulent_flag == -1.0:                        # i.cs:2604
                cd = cd * _roughness_factor(
                    reynolds_body, geom.p_mean_chord, length_body
                )

        # i.cs:2623-2634. The same three compressibility corrections the body's
        # friction routine applies at i.cs:1626-1637. The 1.05 <= M <= 2 arm is
        # evaluated with `(1 - 0.256*(M - 1))`, i.e. referenced to M = 1, not
        # to the 1.05 the branch starts at, so there is a step at M = 1.05.
        if mach >= 1.05 and mach <= 2.0:
            cd = cd * (1.0 - 0.256 * (mach - 1.0))
        if mach > 2.0 and mach <= 10.0:
            cd = cd / (1.0 + 0.144 * mach * mach) ** 0.65
        if mach > 10.0:
            cd = cd * 0.1691

        cd = cd * (geom.p_area * fin.count / cache.a_ref)          # i.cs:2635
        cd_interference = cd * (
            (geom.virtual_area - geom.p_area) * fin.count
        ) / (geom.p_area * fin.count)                              # i.cs:2636
    else:
        # i.cs:2640-2641. Two wetted sides at the flat-plate coefficient. Note
        # the interference term is referenced straight to A_ref here rather
        # than being scaled off cd as it is everywhere else; the result is the
        # same, the operation order is not.
        cd = 2.0 * cf_fin * (geom.p_area * fin.count / cache.a_ref)
        cd_interference = 2.0 * cf_fin * (
            (geom.virtual_area - geom.p_area) * fin.count / cache.a_ref
        )

    return FinDrag(cd, cd_interference, 0.0)


# ---------------------------------------------------------------------------
# B.3 -- supersonic wave drag, linear theory (i.cs:2646-2725)
# ---------------------------------------------------------------------------


def supersonic_wave_drag_linear(
    fin: Fins,
    geom: FinGeometry,
    cache: GeometryCache,
    *,
    mach: float,
    mach_beta: float,
) -> FinWaveDragLinear:
    """i.cs:2646-2725. ``mach_beta`` is sqrt(M^2 - 1), the caller's ``value``.

    Also returns the leading-edge sonic Mach ``sec(sweep)``, because the
    assembly loop needs it later to arbitrate between this result and the
    shock-expansion one, and because a degenerate fin set never computes it.
    """
    if _degenerate(fin, geom):
        # i.cs:2652 zeroes the drag but leaves aq.aa untouched, which on a
        # fresh part is zero. Returning zero matches the first evaluation; a
        # re-used part object would carry the previous Mach's value.
        return FinWaveDragLinear(0.0, 0.0)

    # i.cs:2656. Drag-divergence Mach of the section, 0.9 raised by the sweep
    # of the max-thickness line. The guard replaces a max-thickness line at
    # right angles to the flow with 1.569 rad, clamping the factor at ~74
    # rather than letting cos() reach zero.
    if geom.max_thickness_sweep >= 1.57 or geom.max_thickness_sweep <= -1.57:
        m_divergence = 0.9 / math.cos(1.569) ** 0.5
    else:
        m_divergence = 0.9 / math.cos(geom.max_thickness_sweep) ** 0.5

    if geom.le_sweep == 0.0:
        # Unreachable: az.d is clamped to 0.01 at construction, so le_sweep is
        # never exactly zero. sec(0) = 1 anyway.
        le_sonic_mach = 1.0                                        # i.cs:2659
    else:
        # i.cs:2663-2664. tan^2(sweep), then sqrt(1 + that) = |sec(sweep)|:
        # the Mach at which the sweep-normal component reaches 1.
        le_sonic_mach = (
            1.0 / (math.cos(geom.le_sweep) / math.sin(geom.le_sweep))
        ) ** 2.0
        le_sonic_mach = (le_sonic_mach + 1.0) ** 0.5

    cd_wave = 0.0
    if mach <= m_divergence:
        cd_wave = 0.0                                              # i.cs:2668
    if mach > m_divergence and mach < le_sonic_mach:
        # Subsonic leading edge. The supersonic-LE result evaluated at the
        # sonic-LE Mach -- where beta = tan(sweep), so (sec^2 - 1)^0.5 -- faired
        # back to zero at the divergence Mach by a 3/2 power. i.cs:2672-2673.
        cd_wave = (
            geom.wave_drag_k
            * geom.thickness_ratio ** 2.0
            * (geom.p_area * fin.count / cache.a_ref)
            / (le_sonic_mach ** 2.0 - 1.0) ** 0.5
        )
        cd_wave = cd_wave * (
            (mach ** 2.0 - m_divergence ** 2.0) ** 1.5
            / (le_sonic_mach ** 2.0 - m_divergence ** 2.0) ** 1.5
        )
    if mach >= le_sonic_mach:
        # Supersonic leading edge: the textbook K (t/c)^2 / beta. i.cs:2696.
        cd_wave = (
            (geom.wave_drag_k / mach_beta)
            * geom.thickness_ratio ** 2.0
            * (geom.p_area * fin.count / cache.a_ref)
        )

    # --- leading-edge bluntness, added on top -----------------------------
    bluntness = 0.0
    if fin.airfoil in (Airfoil.SUBSONIC_NACA, Airfoil.ROUNDED):
        # i.cs:2701-2703. The rounded LE's diameter is taken as the section
        # thickness: 2*(t/2). Written that way, and kept.
        bluntness = _le_bluntness_cd(
            mach, geom.le_sweep,
            2.0 * (fin.thickness / 2.0),
            geom.p_span, geom.p_area, fin.count, cache.a_ref,
        )
    if fin.airfoil in WEDGE_SECTIONS:
        if fin.le_radius > 0.0:
            # i.cs:2709-2711. Diameter = 2 * the entered leading-edge radius.
            bluntness = _le_bluntness_cd(
                mach, geom.le_sweep,
                2.0 * fin.le_radius,
                geom.p_span, geom.p_area, fin.count, cache.a_ref,
            )
        else:
            bluntness = 0.0
    if fin.airfoil is Airfoil.SQUARE:
        # i.cs:2720. Diameter = 2 * the thickness, i.e. twice the blunt face
        # actually presented. Every other section passes 2*(t/2) or 2*r_le.
        bluntness = _le_bluntness_cd(
            mach, geom.le_sweep,
            2.0 * fin.thickness,
            geom.p_span, geom.p_area, fin.count, cache.a_ref,
        )

    return FinWaveDragLinear(cd_wave + bluntness, le_sonic_mach)


# ---------------------------------------------------------------------------
# B.4 -- fin base/edge drag, supersonic (i.cs:2727-2745)
# ---------------------------------------------------------------------------


def supersonic_edge_drag(
    fin: Fins,
    geom: FinGeometry,
    cache: GeometryCache,
    *,
    mach: float,
) -> float:
    """i.cs:2727-2745. Returns aq.j -- the square trailing edge's base drag.

    Only the four sections with a cut-off base carry it. The base pressure
    coefficient is flat at -0.46 below M 1.19, then a cubic fit, then a linear
    ramp, then the 1.43/M^2 hypersonic limit.
    """
    if _degenerate(fin, geom):
        return 0.0
    if fin.airfoil not in _BLUNT_BASE_SECTIONS:
        return 0.0

    # i.cs:2737. Note the ordering: the M > 5 arm is tested before the 3-to-5
    # arm, so 3 < M <= 5 falls through to the linear ramp.
    if mach < 1.19:
        cp_base = -0.46
    elif mach <= 3.0:
        cp_base = -1.0 * (
            1.2622
            - 0.98895 * mach
            + 0.30409 * mach ** 2.0
            - 0.033885 * mach ** 3.0
        )
    elif mach > 5.0:
        cp_base = -1.0 * (1.43 / mach ** 2.0)
    else:
        cp_base = -1.0 * (0.117 + -0.06 * ((mach - 3.0) / 2.0))

    return -1.0 * cp_base * (
        (fin.thickness * geom.p_span) * fin.count
    ) / cache.a_ref                                                # i.cs:2738


# ---------------------------------------------------------------------------
# B.5 -- supersonic wave drag, shock-expansion (i.cs:2747-2827)
# ---------------------------------------------------------------------------


def supersonic_wave_drag_shock_expansion(
    fin: Fins,
    geom: FinGeometry,
    cache: GeometryCache,
    *,
    mach: float,
) -> float:
    """i.cs:2747-2827. Returns aq.z.

    The alternative to linear theory at high Mach, where K (t/c)^2 / beta
    understates the compression on the forward wedge. Modified-Newtonian
    pressure on the wedge face rather than a linearised one.

    Every branch here tests ``d_fwd >= d_aft`` and uses the *base* panel when
    it holds, where derive() branches on ``d_fwd == d_aft``. A fin on a
    reducer, where d_fwd > d_aft, therefore gets primed geometry from derive()
    and base geometry here. Reproduced.
    """
    if _degenerate(fin, geom):
        return 0.0

    cp_max = cp_stagnation(mach)
    base_geometry = fin.d_fwd >= fin.d_aft
    cd = 0.0

    if fin.airfoil in WEDGE_SECTIONS:
        # i.cs:2762-2764. Half-angle of the forward wedge, then Newtonian
        # 4 sin^3 on it. The sin^3 (not sin^2) absorbs the projection of the
        # wedge face onto the drag direction.
        half_angle = math.atan(0.5 * fin.thickness / geom.x_le)
        cd = cp_max / 2.0 * (4.0 * math.sin(half_angle) ** 3.0)

        if base_geometry:
            cd = cd * (geom.x_le * geom.span * fin.count) / cache.a_ref
        else:
            cd = cd * (geom.x_le * geom.p_span * fin.count) / cache.a_ref

        if fin.le_radius > 0.0:
            if base_geometry:
                # i.cs:2777-2779 uses the BASE span and BASE panel area, where
                # the linear-theory routine always used the primed pair.
                bluntness = _le_bluntness_cd(
                    mach, geom.le_sweep, 2.0 * fin.le_radius,
                    geom.span, geom.panel_area, fin.count, cache.a_ref,
                )
            else:
                bluntness = _le_bluntness_cd(
                    mach, geom.le_sweep, 2.0 * fin.le_radius,
                    geom.p_span, geom.p_area, fin.count, cache.a_ref,
                )
        else:
            bluntness = 0.0
        cd = cd + bluntness                                        # i.cs:2794

    if fin.airfoil in (Airfoil.SUBSONIC_NACA, Airfoil.ROUNDED):
        # i.cs:2800-2810. No wedge face to compress: the whole shock-expansion
        # answer for a round-nosed section is its leading-edge bluntness drag,
        # and it *overwrites* rather than adds.
        if base_geometry:
            cd = _le_bluntness_cd(
                mach, geom.le_sweep, 2.0 * (fin.thickness / 2.0),
                geom.span, geom.panel_area, fin.count, cache.a_ref,
            )
        else:
            cd = _le_bluntness_cd(
                mach, geom.le_sweep, 2.0 * (fin.thickness / 2.0),
                geom.p_span, geom.p_area, fin.count, cache.a_ref,
            )

    if fin.airfoil is Airfoil.SQUARE:
        # i.cs:2815. The bluntness term is built on the PRIMED span and area in
        # both diameter cases -- the base/primed test below applies only to the
        # second factor.
        cd = _le_bluntness_cd(
            mach, geom.le_sweep, 2.0 * fin.thickness,
            geom.p_span, geom.p_area, fin.count, cache.a_ref,
        )
        # i.cs:2818/2822. A drag coefficient multiplied by a second, different
        # area ratio: _le_bluntness_cd already referenced its answer to A_ref,
        # and this multiplies it by (t * span * N)/(A_ref cos(sweep)) again.
        # Dimensionally wrong and left alone -- it is what RASAero prints.
        if base_geometry:
            cd = cd * (
                (fin.thickness * geom.span * fin.count)
                / (cache.a_ref * math.cos(geom.le_sweep))
            )
        else:
            cd = cd * (
                (fin.thickness * geom.p_span * fin.count)
                / (cache.a_ref * math.cos(geom.le_sweep))
            )

    return cd


# ---------------------------------------------------------------------------
# B.6 -- fin CN-alpha, subsonic (i.cs:2829-2881)
# ---------------------------------------------------------------------------


def subsonic_cn_alpha(
    fin: Fins,
    geom: FinGeometry,
    cache: GeometryCache,
    *,
    modified_barrowman: bool,
) -> FinNormalForce:
    """i.cs:2829-2881. Barrowman's fin CN-alpha, in both of RASAero's modes.

    The two modes differ in four ways and all four matter:

    1. Classical builds the panel term as ``4 N (s/d)^2``; modified builds it
       as ``16 (s/d)^2`` -- four fins' worth -- and then applies a fin-count
       factor that is not linear in N.
    2. The wing-body interference factor is linear in the radius ratio in
       classical (``1 + k r/(s+r)`` with k = 1 for 3 or 4 fins, 0.5 otherwise)
       and quadratic in modified.
    3. Only modified carries the body-carryover term ``aq.q``.
    4. The reference-area conversion uses the literal 3.14159 in modified and
       Math.PI in classical, from the same nose diameter. The two differ by
       8.4e-7 relative and both are reproduced.

    ``s/d`` is the *primed* span over the NOSE diameter ``cache.d_nose``, not
    over the reference diameter -- which is why the conversion to A_ref that
    follows is written on the nose area.
    """
    if _degenerate(fin, geom):
        return FinNormalForce(0.0, 0.0)

    d_nose = cache.d_nose

    if modified_barrowman:
        # i.cs:2840. Barrowman's l_f: the length of the mid-chord line.
        mid_chord_line = (
            (geom.p_sweep + 0.5 * geom.p_tip_chord - 0.5 * geom.p_root_chord)
            ** 2.0
            + geom.p_span ** 2.0
        ) ** 0.5
        value = 16.0 * (geom.p_span / d_nose) ** 2.0               # i.cs:2841
        value = value / (
            1.0
            + (
                1.0
                + (
                    2.0 * mid_chord_line
                    / (geom.p_root_chord + geom.p_tip_chord)
                ) ** 2.0
            ) ** 0.5
        )
        # i.cs:2843-2863. An unhandled fin count leaves value alone, i.e.
        # is treated as four fins.
        value = value * _FIN_COUNT_FACTOR.get(fin.count, 1.0)

        # i.cs:2864-2865. r/(r + s) on the equal-volume body radius and the
        # primed span. b3 is the *carryover*: the total (1 + r/(r+s))^2 of
        # slender-body theory less the fin-on-body part a8.
        ratio = (geom.equivalent_diameter / 2.0) / (
            geom.equivalent_diameter / 2.0 + geom.p_span
        )
        k_fin_body = 1.0 + (0.803 * ratio + 0.201 * ratio ** 2.0)
        k_body_fin = (ratio + 1.0) ** 2.0 - k_fin_body

        cn_alpha = (k_fin_body * value) * (
            PI_RA * (d_nose / 2.0) ** 2.0 / cache.a_ref
        )                                                          # i.cs:2866
        cn_alpha_body = (k_body_fin * value) * (
            PI_RA * (d_nose / 2.0) ** 2.0 / cache.a_ref
        )                                                          # i.cs:2867
        return FinNormalForce(cn_alpha, cn_alpha_body)

    # --- classical Barrowman ----------------------------------------------
    mid_chord_line = (
        (geom.p_sweep + 0.5 * geom.p_tip_chord - 0.5 * geom.p_root_chord)
        ** 2.0
        + geom.p_span * geom.p_span
    ) ** 0.5                                                       # i.cs:2871
    value = float(4 * fin.count) * (geom.p_span / d_nose) ** 2.0   # i.cs:2872
    value = value / (
        1.0
        + (
            1.0
            + (2.0 * mid_chord_line / (geom.p_root_chord + geom.p_tip_chord))
            ** 2.0
        ) ** 0.5
    )
    # i.cs:2874. Written as "not (N < 2.5 or N > 4.5)", so only 3 and 4 fins
    # get the full Barrowman interference factor; everything else, including
    # 1 and 2 fins, is halved.
    k = 1.0 if not (fin.count < 2.5 or fin.count > 4.5) else 0.5
    k_fin_body = 1.0 + (k * (geom.equivalent_diameter / 2.0)) / (
        geom.p_span + geom.equivalent_diameter / 2.0
    )                                                              # i.cs:2875
    cn_alpha = (k_fin_body * value) * (
        math.pi * (d_nose / 2.0) ** 2.0 / cache.a_ref
    )                                                              # i.cs:2877
    return FinNormalForce(cn_alpha, 0.0)


# ---------------------------------------------------------------------------
# B.7 -- fin CN-alpha, supersonic (i.cs:2883-3114)
# ---------------------------------------------------------------------------


def _supersonic_le_chart(tan_sweep_over_beta: float, ar_tan_sweep: float) -> float:
    """beta * CN-alpha for a SUPERSONIC leading edge. i.cs:2926-2973.

    The chart is a family of cubics in ``tan(sweep)/beta`` -- which runs 0 to 1
    over the supersonic-LE range -- interpolated linearly across
    ``AR tan(sweep)``. The transcription check is the first curve at zero
    sweep: 4.025/beta, which is 2D linear theory's 4/beta.
    """
    x = tan_sweep_over_beta
    c0 = 4.025 - 10.145 * x + 9.709 * x ** 2.0 - 3.147 * x ** 3.0
    c1 = 4.019 - 2.855 * x - 3.73 * x ** 2.0 + 3.498 * x ** 3.0
    c2 = 3.997 - 0.721 * x - 2.793 * x ** 2.0 + 1.468 * x ** 3.0
    c3 = 3.991 - 0.375 * x - 0.193 * x ** 2.0 - 0.409 * x ** 3.0
    c4 = 3.992 - 0.227 * x + 1.131 * x ** 2.0 - 1.361 * x ** 3.0
    c5 = 4.007 - 0.754 * x + 3.099 * x ** 2.0 - 2.41 * x ** 3.0
    c6 = 4.007 - 0.595 * x + 3.111 * x ** 2.0 - 2.185 * x ** 3.0
    c7 = 4.017 - 0.95 * x + 4.661 * x ** 2.0 - 3.241 * x ** 3.0

    m = ar_tan_sweep
    if m <= 0.25:
        return c0
    if m <= 0.5:
        return c0 + (c1 - c0) * ((m - 0.25) / 0.25)
    if m <= 1.0:
        return c1 + (c2 - c1) * ((m - 0.5) / 0.5)
    if m <= 2.0:
        return c2 + (c3 - c2) * ((m - 1.0) / 1.0)
    if m <= 3.0:
        return c3 + (c4 - c3) * ((m - 2.0) / 1.0)
    if m <= 4.0:
        return c4 + (c5 - c4) * ((m - 3.0) / 1.0)
    if m <= 5.0:
        return c5 + (c6 - c5) * ((m - 4.0) / 1.0)
    if m <= 6.0:
        return c6 + (c7 - c6) * ((m - 5.0) / 1.0)
    return c7


def _subsonic_le_curve(beta_cot_sweep: float) -> float:
    """The AR tan(sweep) = 2 curve of the subsonic-LE chart. i.cs:3003.

    Runs 3.137 to 3.750 as beta cot(sweep) goes 0 to 0.4, then back down to
    3.014 at 1.0 (the sonic leading edge). Written as "if NOT (0.4 < x <= 1)"
    so the rising limb also catches x > 1 -- unreachable, because this chart is
    only entered when x < 1.
    """
    if not (beta_cot_sweep > 0.4 and beta_cot_sweep <= 1.0):
        return 3.137 + 0.613 * (beta_cot_sweep / 0.4)
    return 3.75 + -0.7360000000000002 * ((beta_cot_sweep - 0.4) / 0.6)


def _subsonic_le_taper(beta_cot_sweep: float, taper: float) -> float:
    """Interpolate the AR tan(sweep) = 2 value across taper ratio.

    i.cs:3000-3019, repeated at 3025-3044. Both ends of the taper range use the
    same 3.014, so the curve only bulges in the middle.
    """
    if taper < 0.2:
        return 3.014 + (_subsonic_le_curve(beta_cot_sweep) - 3.014) * (
            taper / 0.2
        )
    if taper >= 0.2 and taper <= 0.5:
        return _subsonic_le_curve(beta_cot_sweep)
    if taper > 0.5 and taper <= 1.0:
        edge = _subsonic_le_curve(beta_cot_sweep)
        return edge + (3.014 - edge) * ((taper - 0.5) / 0.5)
    # Unreachable: derive() clamps pp_taper_ratio at 1. i.cs:3016.
    return 3.014


def _subsonic_le_capped(
    beta_cot_sweep: float, cap: float, intercept: float
) -> float:
    """A capped-then-linear arm of the subsonic-LE chart. i.cs:3046 etc.

    Each curve rises with beta cot(sweep) only until the chart's own knee at
    ``cap``, a polynomial in taper ratio; past it the value is frozen. Written
    as a slope of -0.445 off ``1 - x`` so that x = 1 gives the intercept.
    """
    if beta_cot_sweep > cap:
        return intercept + (1.0 - beta_cot_sweep) * 0.445
    return intercept + (1.0 - cap) * 0.445


def supersonic_cn_alpha(
    fin: Fins,
    geom: FinGeometry,
    cache: GeometryCache,
    *,
    mach_beta: float,
) -> FinNormalForce:
    """i.cs:2883-3114. ``mach_beta`` is sqrt(M^2 - 1).

    Which side of the leading-edge test is which
    --------------------------------------------
    ``az.m`` is atan(sweep distance / span), i.e. the leading-edge sweep
    measured from the *spanwise* direction, so an unswept fin has m = 0. The
    Mach line lies at (90 deg - mu) from that same spanwise direction, and
    tan(90 - mu) = cot(mu) = beta. So:

        tan(sweep) > beta  ->  the LE lies behind the Mach cone  ->  SUBSONIC LE
        tan(sweep) < beta  ->  the LE is ahead of it             ->  SUPERSONIC LE

    The source's test is ``num2 <= 1`` with num2 = tan(sweep)/beta, which is
    therefore the SUPERSONIC-leading-edge branch. Two independent checks agree:
    that branch's chart bottoms out at 4.025/beta, which is 2D supersonic
    theory's 4/beta; and each branch's chart parameter (num2 there, num = beta
    cot(sweep) here) is confined to [0, 1] exactly on its own side of the test.

    Beware: the supersonic *CP* routine names these two the other way round --
    there ``num2`` is beta cot(sweep) and ``num2 <= 1`` selects the SUBSONIC
    leading edge. The names are swapped between the two routines.

    Note also that unlike the subsonic routine, this one has no Barrowman-mode
    switch: it always uses the modified-Barrowman quadratic interference factor
    and always produces a body-carryover term.
    """
    if _degenerate(fin, geom):
        return FinNormalForce(0.0, 0.0)

    # i.cs:2896-2910. Magnitudes: a forward-swept panel is treated as its
    # mirror image. The m == 0 arm is unreachable (sweep is clamped to 0.01).
    if geom.le_sweep == 0.0:
        beta_cot_sweep = mach_beta / math.tan(0.01)
        tan_sweep_over_beta = math.tan(0.01) / mach_beta
    elif geom.le_sweep > 0.0:
        beta_cot_sweep = mach_beta / math.tan(geom.le_sweep)
        tan_sweep_over_beta = math.tan(geom.le_sweep) / mach_beta
    else:
        beta_cot_sweep = -1.0 * (mach_beta / math.tan(geom.le_sweep))
        tan_sweep_over_beta = -1.0 * (math.tan(geom.le_sweep) / mach_beta)

    # i.cs:2913-2917. The chart abscissa, on the DOUBLE-PRIMED aspect ratio.
    if geom.le_sweep >= 0.0:
        ar_tan_sweep = geom.pp_aspect_ratio * math.tan(geom.le_sweep)
    else:
        ar_tan_sweep = -1.0 * (geom.pp_aspect_ratio * math.tan(geom.le_sweep))

    if tan_sweep_over_beta <= 1.0:
        # Supersonic leading edge.
        panel_cn_alpha = (
            _supersonic_le_chart(tan_sweep_over_beta, ar_tan_sweep)
            / mach_beta
        )                                                          # i.cs:2974
    else:
        # Subsonic leading edge. The chart is CN-alpha * tan(sweep) against
        # AR tan(sweep), so the result is divided by tan(sweep) rather than by
        # beta. i.cs:2980-3078.
        taper = geom.pp_taper_ratio
        m = ar_tan_sweep
        if m <= 0.25:
            value = 0.442
        elif m <= 0.5:
            # 0.442 and 0.932 are assigned to dead locals at i.cs:2986-2987;
            # the interpolation uses their difference as a literal.
            value = 0.442 + 0.49000000000000005 * ((m - 0.25) / 0.25)
        elif m <= 1.0:
            value = 0.932 + 1.0190000000000001 * ((m - 0.5) / 0.5)
        elif m <= 2.0:
            hi = _subsonic_le_taper(beta_cot_sweep, taper)
            value = 1.951 + (hi - 1.951) * ((m - 1.0) / 1.0)
        elif m <= 3.0:
            lo = _subsonic_le_taper(beta_cot_sweep, taper)
            hi = _subsonic_le_capped(
                beta_cot_sweep, 0.017 + 0.996 * taper, 3.535
            )
            value = lo + (hi - lo) * ((m - 2.0) / 1.0)
        elif m <= 4.0:
            lo = _subsonic_le_capped(
                beta_cot_sweep, 0.017 + 0.996 * taper, 3.535
            )
            hi = _subsonic_le_capped(
                beta_cot_sweep,
                0.059 + 1.49 * taper - 0.552 * taper ** 2.0,
                3.942,
            )
            value = lo + (hi - lo) * ((m - 3.0) / 1.0)
        elif m <= 5.0:
            lo = _subsonic_le_capped(
                beta_cot_sweep,
                0.059 + 1.49 * taper - 0.552 * taper ** 2.0,
                3.942,
            )
            hi = _subsonic_le_capped(
                beta_cot_sweep,
                0.21 + 1.348 * taper - 0.561 * taper ** 2.0,
                4.338,
            )
            value = lo + (hi - lo) * ((m - 4.0) / 1.0)
        elif m <= 6.0:
            lo = _subsonic_le_capped(
                beta_cot_sweep,
                0.21 + 1.348 * taper - 0.561 * taper ** 2.0,
                4.338,
            )
            hi = _subsonic_le_capped(
                beta_cot_sweep,
                0.353 + 1.049 * taper - 0.403 * taper ** 2.0,
                4.487,
            )
            value = lo + (hi - lo) * ((m - 5.0) / 1.0)
        else:
            value = _subsonic_le_capped(
                beta_cot_sweep,
                0.353 + 1.049 * taper - 0.403 * taper ** 2.0,
                4.487,
            )

        # i.cs:3078. Same sign convention as above.
        if geom.le_sweep == 0.0:
            panel_cn_alpha = value / math.tan(0.01)
        elif geom.le_sweep <= 0.0:
            panel_cn_alpha = -1.0 * (value / math.tan(geom.le_sweep))
        else:
            panel_cn_alpha = value / math.tan(geom.le_sweep)

    # i.cs:3080-3081. The modified-Barrowman interference pair again, but on
    # the DOUBLE-PRIMED span: supersonically the shock-shadowed root is gone.
    ratio = (geom.equivalent_diameter / 2.0) / (
        geom.equivalent_diameter / 2.0 + geom.pp_span
    )
    k_fin_body = 1.0 + (0.803 * ratio + 0.201 * ratio ** 2.0)
    k_body_fin = (ratio + 1.0) ** 2.0 - k_fin_body

    # i.cs:3082-3111. Two panels' worth of double-primed area, times the
    # fin-count factor. An unhandled count leaves aq.p and aq.q at whatever
    # the previous Mach step wrote; a pure function cannot carry that, so
    # this returns zero -- the value a fresh part would have had.
    if fin.count not in _FIN_COUNT_FACTOR:
        return FinNormalForce(0.0, 0.0)
    factor = _FIN_COUNT_FACTOR[fin.count]
    area_ratio = 2.0 * geom.pp_area / cache.a_ref
    return FinNormalForce(
        (k_fin_body * panel_cn_alpha) * area_ratio * factor,
        (k_body_fin * panel_cn_alpha) * area_ratio * factor,
    )


# ---------------------------------------------------------------------------
# B.8 -- fin centre of pressure, subsonic (i.cs:3116-3140)
# ---------------------------------------------------------------------------


def subsonic_cp(
    fin: Fins,
    geom: FinGeometry,
    *,
    modified_barrowman: bool,
) -> FinCenterOfPressure:
    """i.cs:3116-3140. Barrowman's fin CP, inches aft of the nose tip.

    The two Barrowman modes compute the panel CP identically -- the source
    duplicates the three lines verbatim in both arms of the `if` -- and differ
    only in whether the body-carryover CP exists. It is placed at the quarter
    chord of the primed root, which is where slender-body theory puts the
    carryover load.
    """
    if _degenerate(fin, geom):
        return FinCenterOfPressure(0.0, 0.0)

    # i.cs:3127-3128. Barrowman's X_f, on the primed panel.
    value = (
        geom.p_sweep * (geom.p_root_chord + 2.0 * geom.p_tip_chord)
    ) / (3.0 * (geom.p_root_chord + geom.p_tip_chord))
    value = value + 1.0 / 6.0 * (
        (geom.p_root_chord + geom.p_tip_chord)
        - (geom.p_root_chord * geom.p_tip_chord)
        / (geom.p_root_chord + geom.p_tip_chord)
    )
    cp = geom.p_x0 + value

    if modified_barrowman:
        return FinCenterOfPressure(cp, geom.p_x0 + 0.25 * geom.p_root_chord)
    return FinCenterOfPressure(cp, 0.0)


# ---------------------------------------------------------------------------
# B.9 -- fin centre of pressure, supersonic (i.cs:3142-3254)
# ---------------------------------------------------------------------------


def _supersonic_cp_fraction(
    ar_tan_sweep: float,
    taper: float,
    coefficients: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
) -> float:
    """Interpolate one of the two CP charts across taper ratio.

    i.cs:3181-3209 and 3212-3240 have identical structure and differ only in
    their ten constants, so the shape is factored out. Each ``(a, b)`` is the
    line ``a * AR tan(sweep) + b`` at taper ratio 0, 0.25, 0.33, 0.5 and 1.
    """
    lines = [a * ar_tan_sweep + b for a, b in coefficients]
    if taper <= 0.25:
        return lines[0] + (lines[1] - lines[0]) / 0.25 * (taper - 0.0)
    if taper <= 0.33:
        return lines[1] + (lines[2] - lines[1]) / 0.08000000000000002 * (
            taper - 0.25
        )
    if taper <= 0.5:
        return lines[2] + (lines[3] - lines[2]) / 0.16999999999999998 * (
            taper - 0.33
        )
    if taper <= 1.0:
        return lines[3] + (lines[4] - lines[3]) / 0.5 * (taper - 0.5)
    # Unreachable: derive() clamps pp_taper_ratio at 1. i.cs:3205, 3236.
    return lines[4]


#: The subsonic-leading-edge CP chart, as a fraction of the double-primed root
#: chord. i.cs:3183-3202.
_CP_SUBSONIC_LE = (
    (0.1635, 0.0109),
    (0.2049, 0.0171),
    (0.2014, 0.0637),
    (0.2286, 0.0388),
    (0.2659, 0.1148),
)

#: The supersonic-leading-edge CP chart. Approaches 0.5 -- the 2D flat-plate
#: mid-chord -- at high taper ratio, which is the transcription check.
#: i.cs:3214-3233.
_CP_SUPERSONIC_LE = (
    (0.0832, 0.3256),
    (0.1246, 0.3380),
    (0.1366, 0.3588),
    (0.1651, 0.3800),
    (0.2476, 0.4945),
)


def supersonic_cp(
    fin: Fins,
    geom: FinGeometry,
    *,
    mach: float,
) -> FinCenterOfPressure:
    """i.cs:3142-3254. Takes MACH, not beta -- it forms beta itself at 3144.

    The two charts are blended on the leading-edge parameter. Careful: the
    source's ``num2`` here is beta cot(sweep), the *reciprocal* of the ``num2``
    in supersonic_cn_alpha, so ``num2 <= 1`` selects the SUBSONIC leading edge
    in this routine and the SUPERSONIC one in that. Continuity at the sonic
    leading edge confirms the reading: both arms give the midpoint of the two
    charts when the parameter is 1.

    The body-carryover CP is set equal to the panel CP supersonically -- there
    is no separate quarter-chord station as there is subsonically.
    """
    mach_beta = beta(mach)

    if _degenerate(fin, geom):
        return FinCenterOfPressure(0.0, 0.0)

    # i.cs:3156-3169. Same magnitude convention as the CN-alpha routine.
    if geom.le_sweep == 0.0:
        beta_cot_sweep = mach_beta / math.tan(0.01)
        tan_sweep_over_beta = math.tan(0.01) / mach_beta
    elif geom.le_sweep > 0.0:
        beta_cot_sweep = mach_beta / math.tan(geom.le_sweep)
        tan_sweep_over_beta = math.tan(geom.le_sweep) / mach_beta
    else:
        beta_cot_sweep = -1.0 * (mach_beta / math.tan(geom.le_sweep))
        tan_sweep_over_beta = -1.0 * (math.tan(geom.le_sweep) / mach_beta)

    if geom.le_sweep >= 0.0:
        ar_tan_sweep = geom.pp_aspect_ratio * math.tan(geom.le_sweep)
    else:
        ar_tan_sweep = -1.0 * (geom.pp_aspect_ratio * math.tan(geom.le_sweep))

    subsonic_le = _supersonic_cp_fraction(
        ar_tan_sweep, geom.pp_taper_ratio, _CP_SUBSONIC_LE
    )
    supersonic_le = _supersonic_cp_fraction(
        ar_tan_sweep, geom.pp_taper_ratio, _CP_SUPERSONIC_LE
    )

    # i.cs:3241. Below a sonic LE the blend runs 0 -> 1/2 as beta cot(sweep)
    # goes 0 -> 1; above it, it runs 1/2 -> 1 as tan(sweep)/beta falls 1 -> 0.
    # Continuous at the sonic leading edge, where both give 1/2.
    if beta_cot_sweep <= 1.0:
        fraction = subsonic_le + (supersonic_le - subsonic_le) / 2.0 * (
            beta_cot_sweep
        )
    else:
        fraction = subsonic_le + (supersonic_le - subsonic_le) / 2.0 * (
            1.0 + (1.0 - tan_sweep_over_beta)
        )

    if geom.le_sweep >= 0.0:
        cp = geom.pp_x0 + fraction * geom.pp_root_chord             # i.cs:3244
    else:
        # Forward-swept: the charts do not apply, so RASAero falls back to the
        # midpoint of the double-primed mean aerodynamic chord. i.cs:3249.
        cp = geom.pp_x0 + geom.pp_sweep / 2.0 + geom.pp_mean_chord / 2.0

    return FinCenterOfPressure(cp, cp)


# ---------------------------------------------------------------------------
# B.10 -- modified Newtonian fin CN-alpha and CP (i.cs:3256-3433)
# ---------------------------------------------------------------------------


def newtonian_cn_alpha_cp(
    fin: Fins,
    geom: FinGeometry,
    cache: GeometryCache,
    *,
    mach: float,
) -> FinNewtonian:
    """i.cs:3256-3433. The hypersonic alternative to the linear-theory pair.

    RASAero does not differentiate analytically: it evaluates the Newtonian
    pressure at plus and minus four degrees about the local surface angle and
    divides by the four-degree increment, which is why 4 deg in radians appears
    three times over with three different numbers of digits.

    Two contributions, at two different stations:

    * ``d2`` -- the forward wedge face, present only on the wedge family. It
      acts at the midpoint of the forward wedge run.
    * ``num5`` -- the constant-thickness run, a flat plate at incidence. It
      acts at the midpoint of that run, and is non-zero only for the two
      hexagonal sections: every other wedge section has ``x_mid`` = 0.

    The third term (``num6``/``d5``) is dead: both are hard zero in every arm.
    """
    if _degenerate(fin, geom):
        return FinNewtonian(0.0, 0.0)

    cp_max = cp_stagnation(mach)
    base_geometry = fin.d_fwd >= fin.d_aft

    if fin.airfoil in WEDGE_SECTIONS:
        # i.cs:3278-3292.
        half_angle = math.atan(0.5 * fin.thickness / geom.x_le)
        plus = (half_angle * RAD_TO_DEG_RA + 4.0) * DEG_TO_RAD_RA
        minus = (half_angle * RAD_TO_DEG_RA - 4.0) * DEG_TO_RAD_RA

        if half_angle <= _ALPHA_REF_F32:
            # A wedge shallower than the incidence increment: the lower surface
            # would be at a negative angle, so RASAero takes a forward
            # difference over the full increment instead of a central one.
            wedge = (
                (cp_max / 2.0 * 2.0)
                * math.sin(plus) ** 2.0
                * math.cos(half_angle)
                / _ALPHA_REF_RAD
            )                                                      # i.cs:3284
        else:
            wedge = (
                (cp_max / 2.0 * 2.0)
                * math.sin(plus) ** 2.0
                * math.cos(half_angle)
            )
            wedge = wedge - (cp_max / 2.0 * 2.0) * math.sin(
                minus
            ) ** 2.0 * math.cos(half_angle)
            wedge = wedge / _ALPHA_REF_RAD                         # i.cs:3290

        flat = (
            (cp_max / 2.0 * 2.0)
            * math.sin(_ALPHA_REF_SIN) ** 2.0
            / _ALPHA_REF_RAD
        )                                                          # i.cs:3292

        # i.cs:3293-3370. Referenced on the wedge run and the mid run
        # respectively, two panels each. An unhandled fin count skips the
        # normalisation entirely, leaving raw pressure coefficients -- there is
        # no default arm here, unlike the non-wedge branch below.
        if fin.count in _FIN_COUNT_FACTOR:
            factor = _FIN_COUNT_FACTOR[fin.count]
            span = geom.span if base_geometry else geom.p_span
            wedge = wedge * (geom.x_le * span * 2.0 / cache.a_ref) * factor
            flat = flat * (geom.x_mid * span * 2.0 / cache.a_ref) * factor

        # i.cs:3374-3382. The two arms are CROSS-WIRED relative to every other
        # routine in this module: the constant-diameter arm uses the PRIMED
        # station with the BASE sweep, and the varying-diameter arm uses the
        # BASE station with the PRIMED sweep. When the diameters are equal the
        # two sets coincide so the first arm is harmless; the second is not.
        # Compare the non-wedge branch below, which pairs them the other way.
        if base_geometry:
            x_wedge = geom.p_x0 + geom.sweep / 2.0 + geom.x_le / 2.0
            x_flat = (
                geom.p_x0 + geom.sweep / 2.0 + geom.x_le + geom.x_mid / 2.0
            )
        else:
            x_wedge = geom.x0 + geom.p_sweep / 2.0 + geom.x_le / 2.0
            x_flat = (
                geom.x0 + geom.p_sweep / 2.0 + geom.x_le + geom.x_mid / 2.0
            )
    else:
        # Subsonic NACA, Rounded, Square: no wedge face, so the whole panel is
        # treated as a flat plate at incidence. i.cs:3387-3388. Note the
        # operation order differs from the wedge branch's identical quantity.
        wedge = 0.0
        x_wedge = 0.0
        flat = (cp_max / 2.0) * (
            2.0 * math.sin(_ALPHA_REF_SIN) ** 2.0
        ) / _ALPHA_REF_RAD
        if base_geometry:
            flat = flat * (geom.panel_area * 2.0 / cache.a_ref)
        else:
            flat = flat * (geom.p_area * 2.0 / cache.a_ref)
        # i.cs:3389-3412. Here an unhandled count pops a message box and
        # leaves the value unscaled; the pure port just leaves it unscaled.
        flat = flat * _FIN_COUNT_FACTOR.get(fin.count, 1.0)

        # i.cs:3415-3426. Paired the natural way round -- base station with
        # base sweep, primed station with primed sweep -- and both on the BASE
        # mean chord.
        if base_geometry:
            x_flat = geom.x0 + geom.sweep / 2.0 + geom.mean_chord / 2.0
        else:
            x_flat = geom.p_x0 + geom.p_sweep / 2.0 + geom.mean_chord / 2.0

    cn_alpha = wedge + flat + 0.0                                  # i.cs:3428
    # i.cs:3429-3430. Moment about the nose tip divided by the total: an
    # unguarded division, which is exactly what RASAero does.
    cp = (wedge * x_wedge + flat * x_flat + 0.0 * 0.0) / cn_alpha
    return FinNewtonian(cn_alpha, cp)


# ---------------------------------------------------------------------------
# B.11 -- the supersonic selection rules (i.cs:3973-3989)
# ---------------------------------------------------------------------------


def select_wave_drag(
    linear: FinWaveDragLinear,
    shock_expansion: float,
    *,
    mach: float,
) -> float:
    """Pick between the two supersonic fin wave-drag models. i.cs:3973-3980.

    Shock-expansion wins only when it is the *larger* of the two and the
    leading edge is already supersonic. Below the sonic-LE Mach linear theory
    is kept even where shock-expansion would give more, because the
    shock-expansion model has no swept-leading-edge relief in it at all.
    """
    if linear.cd_wave < shock_expansion and mach >= linear.le_sonic_mach:
        return shock_expansion
    return linear.cd_wave


def select_normal_force(
    linear: FinNormalForce,
    linear_cp: FinCenterOfPressure,
    newtonian: FinNewtonian,
    *,
    mach: float,
    le_sonic_mach: float,
) -> tuple[FinNormalForce, FinCenterOfPressure]:
    """Pick between linear theory and modified Newtonian. i.cs:3981-3989.

    Three things about this rule are easy to get wrong and all three matter:

    1. It compares **moments about the nose tip**, not coefficients. The
       linear-theory moment sums both of its terms, ``p*s + q*t``; the
       Newtonian one is the single product ``ap*aq``.
    2. When it fires it does not merely swap the panel term -- it **zeroes the
       body-carryover term** as well. Modified Newtonian has no carryover
       model, so the carryover load simply disappears above the crossover.
    3. It is gated on the leading edge already being supersonic, the same gate
       the wave-drag selection uses.
    """
    linear_moment = (
        linear.cn_alpha * linear_cp.cp
        + linear.cn_alpha_body * linear_cp.cp_body
    )
    newtonian_moment = newtonian.cn_alpha * newtonian.cp
    if newtonian_moment > linear_moment and mach >= le_sonic_mach:
        return (
            FinNormalForce(newtonian.cn_alpha, 0.0),
            FinCenterOfPressure(newtonian.cp, 0.0),
        )
    return linear, linear_cp
