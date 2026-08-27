"""The engine proper: dispatch, assembly, and the transonic fairing.

Everything else in this package computes one term. This is what turns them
into a coefficient, and it is where RASAero's three unrelated solvers get
stitched together.

Three facts drive the design:

* **Drag and normal force are not separable.** One solve fills the friction,
  form/wave, base and fin drag terms *and* every component's CN-alpha and its
  station. Splitting them would mean solving twice.
* **The engine is stateful.** The geometry pass is cached per design, and the
  mode flags are engine state rather than per-query arguments -- changing any
  of them invalidates the cache, exactly as ``ca = false`` does at i.cs:413.
* **A transonic query costs two full solves.** The 0.90-1.05 window is not
  computed; it is interpolated between the subsonic solve at exactly 0.90 and
  the supersonic solve at exactly 1.05. RASAero re-runs both on every
  transonic call. They are cached here per (alpha, altitude), which is the one
  place this implementation is deliberately faster than the original -- it
  changes no result, only how many times the same result is computed.

The per-term breakdown on ``AeroResult`` is not a diagnostic nicety. It is the
interface the oracle validates against: RASAero's own Tools > Run Test emits
the same breakdown, and a total that agrees while two terms cancel is not a
correct engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import basedrag, body, crossflow, fins, friction, protuberances
from .atmosphere import Atmosphere, STANDARD, reynolds
from .compressible import beta as beta_of
from .geometry import GeometryCache, build as build_geometry
from .parts import Design, Fins, NoseCone, NoseShape, PartType

#: Four degrees in radians, using RASAero's pi. The CN-alpha and CP columns
#: labelled "(0 to 4 deg)" are always evaluated here regardless of the alpha
#: actually requested, and it is this pair the flight simulator consumes.
ALPHA_REF_RAD = 0.0698131111111111

#: Regime boundaries (i.cs:946-969). Inclusive/exclusive exactly as written.
MACH_MIN = 0.01
MACH_SUBSONIC_MAX = 0.90
MACH_SUPERSONIC_MIN = 1.05

#: The transonic drag peak: +4.25% over the subsonic anchor, reached at
#: M = 0.925 by a parabola whose coefficient (68) is chosen to land exactly
#: there -- 68 * 0.025^2 = 0.0425.
TRANSONIC_PEAK_FACTOR = 1.0425
TRANSONIC_PEAK_MACH = 0.925
TRANSONIC_PARABOLA = 68.0


@dataclass
class PartTerms:
    """Per-part working storage (``o/p/q`` and ``r/s/t`` on ``aq``).

    Three normal-force contributions with three independent stations. For a
    body part only the first is used; a fin set uses the second for the
    fin-on-body term and the third for the body-carryover term, which exists
    only in Modified Barrowman mode.
    """

    cna_own: float = 0.0
    cna_fin_body: float = 0.0
    cna_body_fin: float = 0.0
    cp_own: float = 0.0
    cp_fin_body: float = 0.0
    cp_body_fin: float = 0.0


@dataclass
class AeroResult:
    """One solve. Field names match the oracle's parsed columns."""

    mach: float
    alpha_deg: float
    regime: str                      # "sub" | "trans" | "sup"
    reynolds: float = 0.0

    # --- drag buildup, power-off -------------------------------------
    cd_friction: float = 0.0
    cd_form: float = 0.0             # subsonic only
    cd_wave_nose: float = 0.0        # supersonic only
    cd_base: float = 0.0
    fin_profile: float = 0.0         # subsonic: thickness + friction combined
    fin_friction: float = 0.0        # supersonic
    fin_wave: float = 0.0            # supersonic, after the linear/Newtonian pick
    fin_interference: float = 0.0
    fin_edge: float = 0.0
    transition_wave: float = 0.0     # supersonic
    cd_protuberance: float = 0.0

    # --- assembled ----------------------------------------------------
    cd_off: float = 0.0              # the raw axial buildup
    cd_on: float = 0.0
    cn_potential: float = 0.0
    cn_viscous: float = 0.0
    cn: float = 0.0
    cp: float = 0.0
    cn_alpha_0to4: float = 0.0
    cp_0to4: float = 0.0
    ca_off: float = 0.0
    ca_on: float = 0.0
    cl_off: float = 0.0
    cl_on: float = 0.0
    cd_off_wind: float = 0.0
    cd_on_wind: float = 0.0

    per_part: dict[int, PartTerms] = field(default_factory=dict)


def _aft_body_length(design: Design, index: int) -> float:
    """Length of body tube aft of ``index`` (i.cs:1698-1721).

    Walks forward accumulating BodyTube lengths. A Fins part is stepped over
    without contributing and without stopping the walk; an Expansion or
    Reducer latches a flag that suppresses all *further* accumulation but does
    not break the loop. Protuberances are inert. Reproduced literally because
    the latch-rather-than-break shape is observable when a transition is
    followed by more tube.
    """
    total = 0.0
    stopped = False
    for i in range(index, len(design.parts) - 1):
        nxt = design.parts[i + 1]
        if nxt.part_type in (PartType.BODY_TUBE, PartType.FINS):
            if not stopped and nxt.part_type is not PartType.FINS:
                total += nxt.length
        elif nxt.part_type in (PartType.EXPANSION, PartType.REDUCER):
            stopped = True
    return total


class Engine:
    """A design plus its cached geometry. Reusable across Mach and alpha."""

    def __init__(self, design: Design, atmosphere: Atmosphere = STANDARD,
                 boattail_model: str = "rasaero"):
        self.design = design
        self.atmosphere = atmosphere
        self.cache: GeometryCache = build_geometry(design)
        self._anchors: dict[tuple[float, float], tuple[AeroResult, AeroResult]] = {}
        # Which supersonic base-drag law to run. "rasaero" is the faithful
        # port and the default everywhere -- the oracle stands on it.
        # "corrected" is the provisional boattail replacement; see
        # basedrag.base_drag_supersonic_corrected for what it claims and,
        # more importantly, what it does not claim yet.
        if boattail_model == "rasaero":
            self._base_supersonic = basedrag.base_drag_supersonic
        elif boattail_model == "corrected":
            self._base_supersonic = basedrag.base_drag_supersonic_corrected
        else:
            raise ValueError(
                f"unknown boattail_model {boattail_model!r}; "
                "expected 'rasaero' or 'corrected'"
            )
        self.boattail_model = boattail_model

    # ------------------------------------------------------------------

    def solve(self, mach: float, alpha_deg: float = 0.0) -> AeroResult:
        """Coefficients at one flight condition.

        Alpha's sign is carried separately and reapplied on the way out; the
        entire solve runs on the magnitude (i.cs:478-494). The model is exactly
        mirror-symmetric, so this is a faithful shortcut rather than a
        simplification.
        """
        mach = max(mach, MACH_MIN)
        sign = -1.0 if alpha_deg < 0.0 else 1.0
        alpha_mag_deg = abs(alpha_deg)
        alpha_rad = alpha_mag_deg * (math.pi / 180.0)
        altitude = self.design.altitude_for(mach)

        if mach <= MACH_SUBSONIC_MAX:
            result = self._subsonic(mach, altitude, alpha_rad)
        elif mach < MACH_SUPERSONIC_MIN:
            result = self._transonic(mach, altitude, alpha_rad)
        else:
            result = self._supersonic(mach, altitude, alpha_rad)

        result.alpha_deg = alpha_mag_deg * sign
        for name in ("cn", "cn_potential", "cn_viscous", "cn_alpha_0to4",
                     "cl_off", "cl_on", "cd_off_wind", "cd_on_wind"):
            setattr(result, name, getattr(result, name) * sign)
        return result

    # ------------------------------------------------------------------
    # Subsonic
    # ------------------------------------------------------------------

    def _subsonic(self, mach: float, altitude: float, alpha_rad: float) -> AeroResult:
        design, cache = self.design, self.cache
        r = AeroResult(mach=mach, alpha_deg=0.0, regime="sub")

        re = reynolds(mach, cache.l_body, altitude, self.atmosphere)
        r.reynolds = re
        cf = friction.body_skin_friction(
            re, cache.l_body, mach,
            roughness_in=design.roughness, turbulent=design.turbulence,
        )

        r.cd_friction = cf.cf * cache.wetted_over_aref
        fineness = cache.l_body / cache.d_ref
        r.cd_form = r.cd_friction * (
            60.0 / fineness ** 3.0 + 0.0025 * fineness
        )                                                    # i.cs:3642

        # Base drag is computed HERE, at i.cs:3643 -- before the per-part loop
        # that begins on the next line. That ordering matters: a Power-Law
        # nose REPLACES the form drag inside that loop (i.cs:3657), and base
        # drag has already been taken from the original value by then. Applying
        # the override first, which is the intuitive reading, moves base drag
        # by 0.06 on a blunt power-law nose.
        r.cd_base = basedrag.base_drag_subsonic(cache, r.cd_friction, r.cd_form)

        nose = design.parts[0]
        if isinstance(nose, NoseCone) and nose.shape is NoseShape.POWER_LAW:
            n = nose.power_law_n
            if 0.0 <= n <= 1.0:
                # Two independent `if`s in the source, not a chain; the
                # conditions are disjoint so the behaviour is the same.
                if n <= 0.1:
                    r.cd_form = 0.8 + -30.000000000000007 * (n - 0.1)
                elif n <= 0.3:
                    r.cd_form = 0.2 + (r.cd_form - 0.2) / 0.19999999999999998 * (n - 0.1)

        self._per_part_subsonic(r, mach, re, alpha_rad)
        self._assemble(r, mach, alpha_rad, subsonic=True)
        return r

    def _per_part_subsonic(
        self, r: AeroResult, mach: float, re: float, alpha_rad: float
    ) -> None:
        design, cache = self.design, self.cache
        a_ref = cache.a_ref

        for index, part in enumerate(design.parts):
            terms = PartTerms()
            r.per_part[index] = terms
            kind = part.part_type

            if kind is PartType.NOSE_CONE:
                aft = _aft_body_length(design, index)
                terms.cna_own = body.nose_cna_subsonic(
                    part, aft, a_ref=a_ref,
                    modified_barrowman=design.modified_barrowman,
                )
                terms.cp_own = body.nose_cp_subsonic(
                    part, aft, modified_barrowman=design.modified_barrowman
                )

            elif kind is PartType.EXPANSION:
                aft = _aft_body_length(design, index)
                terms.cna_own = body.expansion_cna_subsonic(
                    part, aft, a_ref=a_ref, d_nose=cache.d_nose,
                    modified_barrowman=design.modified_barrowman,
                )
                terms.cp_own = body.expansion_cp_subsonic(
                    part, modified_barrowman=design.modified_barrowman
                )

            elif kind is PartType.REDUCER:
                terms.cna_own = body.reducer_cna_subsonic(
                    part, a_ref=a_ref, d_nose=cache.d_nose
                )
                terms.cp_own = body.reducer_cp_subsonic(part)

            elif kind is PartType.FINS:
                fin: Fins = part  # type: ignore[assignment]
                if fin.is_degenerate:
                    continue
                geom = fins.derive(
                    fin, cache, index,
                    turbulent_flow=friction.fin_turbulence_flag(fin, design),
                )
                # Skin friction reads the fin's OWN flag (i.cs:1592, 1594),
                # which a Square section forces to turbulent for itself.
                cf_fin = friction.fin_skin_friction(
                    re, cache.l_body, geom.p_mean_chord, mach,
                    roughness_in=design.roughness, turbulent=geom.turbulent_flag,
                ).cf
                # The Rounded/Square Reynolds multiplier does NOT: subsonically
                # it is guarded by the vehicle-wide flag (`if (a3)`,
                # i.cs:2511), while its supersonic twin uses the per-part flag
                # (i.cs:2582). So a square-finned vehicle with Turbulent Flow
                # off gets the 2.7x multiplier above Mach 1.05 and not below
                # it. Passing one flag to both inflates subsonic fin drag by
                # exactly 2.7.
                drag = fins.subsonic_drag(
                    fin, geom, cache, mach=mach, cf_fin=cf_fin,
                    reynolds_body=re, length_body=cache.l_body,
                    turbulent_flow=design.turbulence,
                )
                r.fin_profile += drag.cd
                r.fin_interference += drag.cd_interference
                r.fin_edge += drag.cd_base

                nf = fins.subsonic_cn_alpha(
                    fin, geom, cache,
                    modified_barrowman=design.modified_barrowman,
                )
                cp = fins.subsonic_cp(
                    fin, geom, modified_barrowman=design.modified_barrowman
                )
                terms.cna_fin_body = nf.cn_alpha
                terms.cna_body_fin = nf.cn_alpha_body
                terms.cp_fin_body = cp.cp
                terms.cp_body_fin = cp.cp_body

            elif kind in (
                PartType.RAIL_GUIDE, PartType.LAUNCH_LUG,
                PartType.LAUNCH_SHOE, PartType.PLATE,
            ):
                r.cd_protuberance += protuberances.protuberance_cd(part, mach, a_ref)

        area_a, area_b = protuberances.streamlined_areas(design.parts)
        s1, s2 = protuberances.streamlined_cd_subsonic(
            area_a, area_b,
            cd_friction=r.cd_friction, cd_form=r.cd_form, cd_base=r.cd_base,
            a_ref=a_ref,
        )
        r.cd_protuberance += s1 + s2

    # ------------------------------------------------------------------
    # Supersonic
    # ------------------------------------------------------------------

    def _supersonic(self, mach: float, altitude: float, alpha_rad: float) -> AeroResult:
        design, cache = self.design, self.cache
        r = AeroResult(mach=mach, alpha_deg=0.0, regime="sup")

        re = reynolds(mach, cache.l_body, altitude, self.atmosphere)
        r.reynolds = re
        beta = beta_of(mach)

        cf = friction.body_skin_friction(
            re, cache.l_body, mach,
            roughness_in=design.roughness, turbulent=design.turbulence,
        )
        r.cd_friction = cf.cf * cache.wetted_over_aref
        # No form-drag term above M 1.05: it is displaced by nose wave drag.
        r.cd_base = self._base_supersonic(design, cache, mach)

        self._per_part_supersonic(r, mach, beta, re, alpha_rad)
        self._assemble(r, mach, alpha_rad, subsonic=False)
        return r

    def _per_part_supersonic(
        self, r: AeroResult, mach: float, beta: float, re: float, alpha_rad: float
    ) -> None:
        design, cache = self.design, self.cache
        a_ref = cache.a_ref
        nose = design.parts[0]

        for index, part in enumerate(design.parts):
            terms = PartTerms()
            r.per_part[index] = terms
            kind = part.part_type

            if kind is PartType.NOSE_CONE:
                aft = _aft_body_length(design, index)
                r.cd_wave_nose += body.nose_wave_drag(part, mach, a_ref=a_ref)
                terms.cna_own = body.nose_cna_supersonic(part, aft, beta, a_ref=a_ref)
                terms.cp_own = body.nose_cp_supersonic(part, aft, beta)

            elif kind is PartType.BODY_TUBE:
                # A cylinder carries no normal force in either mode.
                pass

            elif kind is PartType.EXPANSION:
                aft = _aft_body_length(design, index)
                r.transition_wave += body.expansion_wave_drag(part, mach, a_ref=a_ref)
                terms.cna_own = body.expansion_cna_supersonic(part, aft, beta, a_ref=a_ref)
                terms.cp_own = body.expansion_cp_supersonic(part)

            elif kind is PartType.REDUCER:
                aft = _aft_body_length(design, 0)
                r.transition_wave += body.reducer_wave_drag(
                    part, mach,
                    parts=design.parts, index=index, nose=nose,
                    aft_body_length=aft,
                    boattail_angle_deg=cache.boattail_angle_deg,
                    a_ref=a_ref,
                )
                terms.cna_own = body.reducer_cna_supersonic(part, beta, a_ref=a_ref)
                terms.cp_own = body.reducer_cp_supersonic(part)

            elif kind is PartType.FINS:
                fin: Fins = part  # type: ignore[assignment]
                if fin.is_degenerate:
                    continue
                geom = fins.derive(
                    fin, cache, index,
                    turbulent_flow=friction.fin_turbulence_flag(fin, design),
                )
                cf_fin = friction.fin_skin_friction(
                    re, cache.l_body, geom.p_mean_chord, mach,
                    roughness_in=design.roughness, turbulent=geom.turbulent_flag,
                ).cf
                fr = fins.supersonic_friction_drag(
                    fin, geom, cache, mach=mach, cf_fin=cf_fin,
                    reynolds_body=re, length_body=cache.l_body,
                )
                r.fin_friction += fr.cd
                r.fin_interference += fr.cd_interference

                linear = fins.supersonic_wave_drag_linear(
                    fin, geom, cache, mach=mach, mach_beta=beta
                )
                shock = fins.supersonic_wave_drag_shock_expansion(
                    fin, geom, cache, mach=mach
                )
                r.fin_wave += fins.select_wave_drag(linear, shock, mach=mach)
                r.fin_edge += fins.supersonic_edge_drag(fin, geom, cache, mach=mach)

                nf = fins.supersonic_cn_alpha(fin, geom, cache, mach_beta=beta)
                cp = fins.supersonic_cp(fin, geom, mach=mach)
                newt = fins.newtonian_cn_alpha_cp(fin, geom, cache, mach=mach)
                nf, cp = fins.select_normal_force(
                    nf, cp, newt, mach=mach, le_sonic_mach=linear.le_sonic_mach
                )
                terms.cna_fin_body = nf.cn_alpha
                terms.cna_body_fin = nf.cn_alpha_body
                terms.cp_fin_body = cp.cp
                terms.cp_body_fin = cp.cp_body

            elif kind in (
                PartType.RAIL_GUIDE, PartType.LAUNCH_LUG,
                PartType.LAUNCH_SHOE, PartType.PLATE,
            ):
                r.cd_protuberance += protuberances.protuberance_cd(part, mach, a_ref)

        area_a, area_b = protuberances.streamlined_areas(design.parts)
        s1, s2 = protuberances.streamlined_cd_supersonic(
            area_a, area_b,
            cd_friction=r.cd_friction, cd_wave_nose=r.cd_wave_nose,
            cd_base=r.cd_base, cd_transition_wave=r.transition_wave,
            a_ref=a_ref,
        )
        r.cd_protuberance += s1 + s2

    # ------------------------------------------------------------------
    # Assembly, shared
    # ------------------------------------------------------------------

    def _crossflow(self, mach: float, re: float, alpha_rad: float) -> float:
        """Jorgensen's viscous term. Zero at zero incidence."""
        if alpha_rad == 0.0:
            return 0.0
        cache = self.cache
        cdc = crossflow.crossflow_drag_coefficient(
            crossflow.crossflow_mach(mach, alpha_rad),
            crossflow.crossflow_reynolds(re, alpha_rad),
        )
        return (
            crossflow.eta(cache) * cdc
            * (cache.planform_area / cache.a_ref)
            * math.sin(alpha_rad) ** 2.0
        )

    def _assemble(
        self, r: AeroResult, mach: float, alpha_rad: float, *, subsonic: bool
    ) -> None:
        cache = self.cache

        if subsonic:
            r.cd_off = (
                r.cd_friction + r.cd_form + r.cd_base
                + r.fin_profile + r.fin_interference + r.fin_edge
                + r.cd_protuberance
            )
        else:
            r.cd_off = (
                r.cd_friction + r.cd_wave_nose + r.cd_base
                + r.fin_friction + r.fin_wave + r.fin_interference + r.fin_edge
                + r.transition_wave + r.cd_protuberance
            )
        r.cd_on = r.cd_off + basedrag.power_on_credit(self.design, cache, mach)

        # Viscous crossflow is gated on Modified Barrowman below M 0.9
        # (i.cs:3747) but applied UNCONDITIONALLY above M 1.05 (i.cs:4056,
        # which has no flag test). A classic-Barrowman vehicle therefore has
        # no body lift subsonically and body lift supersonically, and the
        # transonic fairing interpolates between the two.
        use_crossflow = (not subsonic) or self.design.modified_barrowman
        r.cn_viscous = self._crossflow(mach, r.reynolds, alpha_rad) if use_crossflow else 0.0

        cna_total = sum(
            t.cna_own + t.cna_fin_body + t.cna_body_fin for t in r.per_part.values()
        )
        r.cn_potential = cna_total * alpha_rad
        r.cn = r.cn_potential + r.cn_viscous

        if alpha_rad == 0.0:
            moment = sum(
                t.cna_own * t.cp_own
                + t.cna_fin_body * t.cp_fin_body
                + t.cna_body_fin * t.cp_body_fin
                for t in r.per_part.values()
            )
            r.cp = moment / cna_total if cna_total else 0.0
        else:
            moment = sum(
                t.cna_own * alpha_rad * t.cp_own
                + t.cna_fin_body * alpha_rad * t.cp_fin_body
                + t.cna_body_fin * alpha_rad * t.cp_body_fin
                for t in r.per_part.values()
            )
            moment += r.cn_viscous * cache.planform_centroid
            r.cp = moment / r.cn if r.cn else 0.0

        # The "(0 to 4 deg)" pair, always evaluated at 4 degrees whatever
        # alpha was asked for. This is what the flight simulator reads.
        visc4 = self._crossflow(mach, r.reynolds, ALPHA_REF_RAD) if use_crossflow else 0.0
        cn4 = cna_total * ALPHA_REF_RAD + visc4
        r.cn_alpha_0to4 = cn4 / ALPHA_REF_RAD
        moment4 = sum(
            t.cna_own * ALPHA_REF_RAD * t.cp_own
            + t.cna_fin_body * ALPHA_REF_RAD * t.cp_fin_body
            + t.cna_body_fin * ALPHA_REF_RAD * t.cp_body_fin
            for t in r.per_part.values()
        )
        moment4 += visc4 * cache.planform_centroid
        r.cp_0to4 = moment4 / cn4 if cn4 else 0.0

        self._to_wind_axes(r, alpha_rad)

    @staticmethod
    def _to_wind_axes(r: AeroResult, alpha_rad: float) -> None:
        """Body axes to wind axes (i.cs:3831-3840).

        The buildup is an AXIAL coefficient, not a drag coefficient. It is
        inflated by 1/cos^2(alpha) -- an empirical incidence correction with no
        stated source -- to give CA, and only then rotated. At alpha = 0 all
        four outputs collapse back to the buildup.
        """
        c, s = math.cos(alpha_rad), math.sin(alpha_rad)
        r.ca_off = r.cd_off / c ** 2.0
        r.ca_on = r.cd_on / c ** 2.0
        r.cl_off = r.cn * c - r.ca_off * s
        r.cd_off_wind = r.cn * s + r.ca_off * c
        r.cl_on = r.cn * c - r.ca_on * s
        r.cd_on_wind = r.cn * s + r.ca_on * c

    # ------------------------------------------------------------------
    # Transonic
    # ------------------------------------------------------------------

    def _transonic(self, mach: float, altitude: float, alpha_rad: float) -> AeroResult:
        """Interpolate. Nothing here is derived from the vehicle's shape
        beyond the two anchor solves.
        """
        # Each anchor is a complete solve AT ITS OWN MACH, so it takes the
        # altitude the Mach/Alt table gives for 0.90 and 1.05 -- not the one
        # for the Mach being asked about. The Reynolds helper (i.cs:847-871)
        # overwrites whatever altitude it is handed with the table lookup for
        # the Mach it is solving, so the anchors are pinned to their own
        # conditions. With an empty table every altitude is zero and the
        # distinction disappears, which is why this only shows up on a vehicle
        # that actually has a table.
        key = (alpha_rad,)
        if key not in self._anchors:
            self._anchors[key] = (
                self._subsonic(
                    MACH_SUBSONIC_MAX,
                    self.design.altitude_for(MACH_SUBSONIC_MAX),
                    alpha_rad,
                ),
                self._supersonic(
                    MACH_SUPERSONIC_MIN,
                    self.design.altitude_for(MACH_SUPERSONIC_MIN),
                    alpha_rad,
                ),
            )
        sub, sup = self._anchors[key]

        r = AeroResult(mach=mach, alpha_deg=0.0, regime="trans")
        r.reynolds = reynolds(mach, self.cache.l_body, altitude, self.atmosphere)

        u = (mach - MACH_SUBSONIC_MAX) / 0.15000000000000002

        def lerp(a: float, b: float) -> float:
            return a + (b - a) * u

        # The drag family takes the peak treatment; everything else is a
        # straight blend. The branch test is made on the POWER-OFF buildup and
        # then applied to all four drag quantities (i.cs:3848).
        if sup.cd_off < sub.cd_off * TRANSONIC_PEAK_FACTOR:
            r.cd_off = lerp(sub.cd_off, sup.cd_off)
            r.cd_on = lerp(sub.cd_on, sup.cd_on)
            r.ca_off = lerp(sub.ca_off, sup.ca_off)
            r.ca_on = lerp(sub.ca_on, sup.ca_on)
        else:
            def rise(sub_v: float, sup_v: float) -> float:
                if mach <= TRANSONIC_PEAK_MACH:
                    return sub_v * (
                        1.0 + TRANSONIC_PARABOLA * (mach - MACH_SUBSONIC_MAX) ** 2.0
                    )
                peak = sub_v * TRANSONIC_PEAK_FACTOR
                return peak + (sup_v - peak) * (
                    (mach - TRANSONIC_PEAK_MACH) / 0.125
                )
            r.cd_off = rise(sub.cd_off, sup.cd_off)
            r.cd_on = rise(sub.cd_on, sup.cd_on)
            r.ca_off = rise(sub.ca_off, sup.ca_off)
            r.ca_on = rise(sub.ca_on, sup.ca_on)

        r.cn_alpha_0to4 = lerp(sub.cn_alpha_0to4, sup.cn_alpha_0to4)
        r.cp_0to4 = lerp(sub.cp_0to4, sup.cp_0to4)
        r.cn_potential = lerp(sub.cn_potential, sup.cn_potential)
        r.cn_viscous = lerp(sub.cn_viscous, sup.cn_viscous)
        r.cn = lerp(sub.cn, sup.cn)
        r.cp = lerp(sub.cp, sup.cp)

        for index in sub.per_part:
            a, b = sub.per_part[index], sup.per_part.get(index, PartTerms())
            r.per_part[index] = PartTerms(
                cna_own=lerp(a.cna_own, b.cna_own),
                cna_fin_body=lerp(a.cna_fin_body, b.cna_fin_body),
                cna_body_fin=lerp(a.cna_body_fin, b.cna_body_fin),
                cp_own=lerp(a.cp_own, b.cp_own),
                cp_fin_body=lerp(a.cp_fin_body, b.cp_fin_body),
                cp_body_fin=lerp(a.cp_body_fin, b.cp_body_fin),
            )

        # Re-derive the wind-axis pair from the blended CA and CN rather than
        # blending CL and CD directly (i.cs:3880-3883).
        c, s = math.cos(alpha_rad), math.sin(alpha_rad)
        r.cl_off = r.cn * c - r.ca_off * s
        r.cd_off_wind = r.cn * s + r.ca_off * c
        r.cl_on = r.cn * c - r.ca_on * s
        r.cd_on_wind = r.cn * s + r.ca_on * c
        return r
