"""Runtime aerodynamic force/moment model backed by a coefficient database.

Damping
-------
A static coefficient table gives the moment produced by an angle of attack, but
not the moment produced by a *rotation rate*. Without the latter there is
nothing in the equations of motion that removes rotational energy: the restoring
moment from the CP-CG offset makes the vehicle a pendulum with no friction, so
any disturbance -- a gust, a launch transient, a thrust misalignment -- produces
a pitch oscillation that persists undiminished for the whole flight. Real
vehicles damp such an oscillation out within a few cycles.

The physical source is straightforward. When the vehicle pitches at rate ``q``,
a station a distance ``x`` ahead of the CG sees an extra local angle of attack
``q*x/V``. Integrating the resulting normal force over the body gives a moment
that opposes ``q``. Lumping the whole normal force at the centre of pressure --
the same approximation the force model already makes -- reduces the integral to

    Cmq = -2 * CN_alpha * ((x_cp - x_cg) / L_ref)^2

which is the standard single-surface result. It needs no data the table does not
already carry: ``CN_alpha`` is the slope of the tabulated normal force and
``x_cp - x_cg`` is the static margin the force model already computes. The sign
is negative for any stable vehicle and the term is quadratic in the static
margin, so a more stable rocket is also a better-damped one -- as observed.

The damping moment is then

    M = q_dyn * S_ref * L_ref * (L_ref / (2V)) * Cmq * omega_transverse

The apparent 1/V singularity is not real: ``q_dyn`` carries a factor of V^2, so
the product is linear in V and goes to zero with airspeed, as it must.

That single-surface estimate is the fallback. A table that carries its own
damping moments -- the normal-force slope of every lifting part with its
first and second moments about the nose, as ``aeroengine``-built tables do
-- gives Cmq summed over the parts about whatever CG the vehicle has at the
moment. The difference is not small: lumping the slope at the total CP
cancels the nose's arm against the fins' on a nose-and-fins vehicle, and
understates the damping several-fold.

Roll follows the same pattern. A table with ``clp`` and ``cl_roll`` --
strip-theory damping over the fin span and the forcing moment from fin cant
-- gives a rolling vehicle a rate to settle at; without them roll is
undamped and undriven, which is what it was. Both are referenced to the
diameter and the rate ``p d / 2V``, so that the steady roll rate is the
closed form ``p = -2 V Cl / (d Clp)``.

Jet damping and the inertia-rate term of a body losing mass live in the
simulation's force model and Euler's equations, not here.

Axial force, not wind-axis drag
-------------------------------
The table's drag column is the wind-axis ``CD = CA cos a + CN sin a`` -- the
engine assembles it that way and so does RASAero's export. The force the
body feels along its axis is ``CA``, and it is recovered from the table
exactly, ``CA = (CD - CN sin a) / cos a``. This model used to apply ``CD``
along the axis, which counted the induced part of the drag twice: once
inside ``CD`` and again through the normal force's own wind-axis component.
At 16 degrees on the basic rocket that was about 40% too much drag; at 90
degrees it would have been meaningless.

Beyond the table
----------------
A RASAero-class table is a small-angle method, and the flight leaves its
range: a gusty rail exit at 20-30 degrees, a descent under drogue at 90, a
tumble. Holding the edge value there gave a normal force that stopped
growing and an axial force that never shrank. Past the edge the table is
extrapolated on its own slope to the stall onset (15 degrees, or the edge
if later), then blended over the next 15 degrees into the empirical
high-alpha model every comparable tool uses:

* the body -- Jorgensen's form of Allen-Perkins: slender-body potential
  lift ``(A_base/A_ref) sin 2a cos(a/2)`` at the nose, plus viscous crossflow
  ``eta Cd_c (A_plan/A_ref) sin^2 a`` at the planform centroid, with
  ``Cd_c`` the crossflow drag of a cylinder against crossflow Mach and
  ``eta`` the finite-length factor against fineness;
* the fins -- their attached lift ``CNa_fin sin a cos a`` plus a plate's
  crossflow ``1.17 sin^2 a``, on their planform area at their centroid.
  Rocket fins are low-aspect-ratio surfaces that keep lifting to forty
  degrees and beyond; taken as fully stalled plates from the blend on,
  they lost half their lift by twenty degrees and a stable vehicle was
  called unstable there. Broadside the sum is the plate's ``1.17``;
* the axial force falling as ``cos a`` to nothing broadside.

The centre of pressure is the moment-weighted blend, so the moment itself
is continuous. This is a correlation good to a few tens of percent, not a
computation; its job is a bounded force of the right size and sign where
the table has nothing to say, not accuracy. The planform, fins and nose
length come from the parametric model with the table; a table without them
is treated as an ogive-nosed cylinder of its reference length and diameter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .aero_database import AeroDatabase, AeroCoefficients

#: Where fin stall is taken to begin, or the table's edge if that is later.
STALL_ONSET_DEG = 15.0
#: Width of the blend from the extrapolated table into the high-alpha model.
HIGH_ALPHA_BLEND_DEG = 15.0
#: Normal-force coefficient of a finite flat plate broadside (Hoerner).
FLAT_PLATE_CN_90 = 1.17


def crossflow_drag_coefficient(mach_crossflow: float) -> float:
    """Drag coefficient of a circular cylinder against the crossflow Mach.

    Subsonic 1.2, rising through the transonic crossflow to about 1.8 near
    Mach 1 and settling around 1.4 supersonically -- Jorgensen's curve,
    piecewise linear.
    """
    return float(np.interp(
        abs(float(mach_crossflow)),
        [0.0, 0.4, 1.0, 1.6, 3.0],
        [1.2, 1.2, 1.8, 1.45, 1.35],
    ))


def crossflow_proportionality(fineness: float) -> float:
    """Allen-Perkins finite-length factor ``eta`` against length over diameter.

    A finite cylinder sheds less crossflow drag than an infinite one; the
    factor rises from about 0.6 at L/d 4 toward 1 for a very long body.
    """
    return float(np.interp(
        float(fineness),
        [4.0, 8.0, 12.0, 16.0, 20.0, 40.0],
        [0.60, 0.66, 0.72, 0.78, 0.82, 0.90],
    ))


def _smoothstep(t: float) -> float:
    t = min(max(float(t), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class HighAlphaGeometry:
    """What the high-alpha extension needs to know about the vehicle.

    Stations are from the nose tip, aft positive, like the table's CP.
    """

    length_m: float
    diameter_m: float
    #: Side-projected area of the body and where it is centred.
    planform_area_m2: float
    planform_centroid_m: float
    nose_length_m: float
    #: Exposed planform of every fin panel together, and its centroid.
    fin_area_m2: float = 0.0
    fin_centroid_m: float = 0.0

    @property
    def base_area_m2(self) -> float:
        return float(np.pi * (0.5 * self.diameter_m) ** 2)

    @classmethod
    def cylinder(cls, length_m: float, diameter_m: float) -> "HighAlphaGeometry":
        """An ogive-nosed cylinder: the fallback for a table without geometry."""
        nose = 0.2 * length_m
        planform = diameter_m * (length_m - nose) + (2.0 / 3.0) * diameter_m * nose
        return cls(
            length_m=float(length_m), diameter_m=float(diameter_m),
            planform_area_m2=float(planform), planform_centroid_m=0.53 * float(length_m),
            nose_length_m=float(nose),
        )


@dataclass
class AeroForces:
    force_body_n: np.ndarray
    moment_body_nm: np.ndarray
    coefficients: AeroCoefficients
    mach: float
    alpha_deg: float
    # Split out so callers can inspect how much of the moment is damping.
    static_moment_body_nm: np.ndarray | None = None
    damping_moment_body_nm: np.ndarray | None = None
    cmq: float = 0.0
    #: CP minus CG along the body axis, measured aft: positive when the
    #: centre of pressure sits behind the centre of gravity, i.e. for a
    #: statically stable vehicle. The same sign the status bar quotes.
    static_margin_m: float = 0.0
    #: Where the centre of pressure was placed, in the body frame.
    cp_body_m: np.ndarray | None = None
    #: Roll damping and cant forcing actually applied, per the module docstring.
    clp: float = 0.0
    cl_roll: float = 0.0
    #: The axial and normal coefficients and CP station actually applied --
    #: the table's inside its range, the extension's beyond it.
    ca_applied: float = 0.0
    cn_applied: float = 0.0
    x_cp_applied_m: float = 0.0


class RasaeroAeroModel:
    """Compute body-frame aero loads from RASAero-derived coefficients."""

    def __init__(
        self,
        database: AeroDatabase,
        reference_area_m2: float,
        reference_length_m: float,
        body_axis: np.ndarray | None = None,
        nose_position_body_m: np.ndarray | None = None,
        cmq: float | None = None,
        clp: float = 0.0,
        high_alpha: HighAlphaGeometry | None = None,
        cant_offset_rad: float = 0.0,
    ):
        """
        Args:
            cmq: Pitch/yaw damping derivative [1/rad]. ``None`` takes the
                table's per-part damping moments when it has them, else
                estimates from the normal-force slope and the static margin;
                see the module docstring. An explicit (negative) value
                overrides both.
            clp: Roll damping derivative per ``p d / 2V``, on the reference
                diameter. Zero takes the table's; a non-zero value overrides.
            high_alpha: Planform, fins and nose for the extension beyond the
                table's alpha range. ``None`` takes the table's own, and a
                table without one is treated as an ogive-nosed cylinder.
            cant_offset_rad: Fin cant the vehicle has that the table was not
                built with -- a build error. Applied through the table's
                ``cl_cant``, the forcing per radian; a table without that
                column cannot apply it and the offset does nothing.
        """
        self.database = database
        self.reference_area_m2 = float(reference_area_m2)
        self.reference_length_m = max(float(reference_length_m), 1e-9)
        # The reference area is the body cross-section, so its diameter is
        # the roll reference the fins are measured against.
        self.reference_diameter_m = float(np.sqrt(4.0 * self.reference_area_m2 / np.pi))
        known = high_alpha or getattr(database, "high_alpha", None)
        #: Whether the planform is the vehicle's own. Without it the
        #: extension still grows the force but leaves the centre of pressure
        #: where the table put it: a finless cylinder's planform centroid is
        #: forward of any finned rocket's CG, and blending toward it turned a
        #: stable vehicle divergent from 25 degrees on.
        self.geometry_known = known is not None
        self.high_alpha = known or HighAlphaGeometry.cylinder(
            self.reference_length_m, self.reference_diameter_m,
        )
        self.body_axis = _unit(np.array([0.0, 1.0, 0.0]) if body_axis is None else body_axis)
        self.nose_position_body_m = (
            np.zeros(3) if nose_position_body_m is None else np.asarray(nose_position_body_m, dtype=float)
        )
        self.cmq_override = None if cmq is None else float(cmq)
        self.clp = float(clp)
        self.cant_offset_rad = float(cant_offset_rad)

    def estimate_cmq(self, mach: float, alpha_deg: float, static_margin_m: float) -> float:
        """Pitch/yaw damping derivative [1/rad], negative for a stable vehicle.

        ``static_margin_m`` is the signed distance from CG to CP along the body
        axis; it is squared here, so a vehicle that is unstable in the static
        sense still damps rotation rather than amplifying it, which is correct
        -- static instability and rotational damping are independent effects.
        """
        if self.cmq_override is not None:
            return self.cmq_override
        cn_alpha = self.database.cn_alpha_per_rad(mach, alpha_deg)
        arm = static_margin_m / self.reference_length_m
        return -2.0 * cn_alpha * arm * arm

    def coefficients_at(self, mach: float, alpha_deg: float,
                        power_on: bool = False) -> tuple[float, float, float]:
        """Axial coefficient, normal coefficient and CP station at (Mach, alpha).

        Inside the table's alpha range the table rules, with the axial
        force recovered from its wind-axis drag. Beyond it, the extension
        described in the module docstring. Alpha is taken as a magnitude.
        """
        alpha = abs(float(alpha_deg))
        _, edge = self.database.alpha_range_deg
        queried = min(alpha, edge)
        coeffs = self.database.lookup(mach, queried)
        cd = (
            coeffs.cd_power_on
            if power_on and coeffs.cd_power_on is not None
            else coeffs.cd
        )
        a = np.radians(queried)
        ca_edge = (cd - coeffs.cn * np.sin(a)) / max(np.cos(a), 1e-6)
        if alpha <= edge + 1e-9 or edge >= 89.0:
            return float(ca_edge), float(coeffs.cn), float(coeffs.x_cp_m)
        return self._extend(mach, alpha, edge, float(ca_edge), float(coeffs.cn),
                            float(coeffs.x_cp_m))

    def _extend(self, mach: float, alpha: float, edge: float,
                ca_edge: float, cn_edge: float, x_cp_edge: float) -> tuple[float, float, float]:
        """The table extrapolated to stall onset, then blended into the high-alpha model."""
        geometry = self.high_alpha
        a_ref = self.reference_area_m2
        a = np.radians(alpha)
        s, c = np.sin(a), np.cos(a)

        # The table on its own slope past its edge. A four-degree export
        # would otherwise be "stalled" from five degrees on.
        slope = self.database.cn_alpha_per_rad(mach, edge)
        cn_linear = cn_edge + slope * np.radians(alpha - edge)

        # The body: slender-body potential lift at the nose, viscous
        # crossflow at the planform centroid. The fins: stalled plates.
        diameter_ratio = geometry.diameter_m / max(self.reference_diameter_m, 1e-9)
        eta = crossflow_proportionality(geometry.length_m / max(geometry.diameter_m, 1e-9))
        cd_c = crossflow_drag_coefficient(mach * s)
        cn_pot = diameter_ratio ** 2 * np.sin(2.0 * a) * np.cos(0.5 * a)
        cn_cross = eta * cd_c * (geometry.planform_area_m2 / a_ref) * s * s
        # The fins' own slope: the table's, less the slender body's
        # ``2 A_base / A_ref``, per unit of their planform. Attached lift
        # falls as cos a and the plate's crossflow takes over, so the sum
        # is the plate's 1.17 broadside and the table's slope at the edge.
        fin_ratio = geometry.fin_area_m2 / a_ref
        cn_alpha_fins = max(slope - 2.0 * diameter_ratio ** 2, 0.0) / fin_ratio if fin_ratio > 0 else 0.0
        cn_fins = fin_ratio * (cn_alpha_fins * s * c + FLAT_PLATE_CN_90 * s * s)
        cn_high = cn_pot + cn_cross + cn_fins
        if self.geometry_known:
            moment_high = (
                cn_pot * 0.5 * geometry.nose_length_m
                + cn_cross * geometry.planform_centroid_m
                + cn_fins * geometry.fin_centroid_m
            )
        else:
            # No planform to place the force on: keep it where the table
            # says, and only let it grow.
            moment_high = cn_high * x_cp_edge

        onset = max(edge, STALL_ONSET_DEG)
        w = _smoothstep((alpha - onset) / HIGH_ALPHA_BLEND_DEG)
        cn = (1.0 - w) * cn_linear + w * cn_high
        # Blend the moment, not the station, so the moment is continuous.
        moment = (1.0 - w) * cn_linear * x_cp_edge + w * moment_high
        x_cp = moment / cn if abs(cn) > 1e-12 else x_cp_edge
        ca = ca_edge * c / max(np.cos(np.radians(edge)), 1e-6)
        return float(ca), float(cn), float(x_cp)

    def cmq_from_table(self, coeffs: AeroCoefficients, cg_body_m: np.ndarray) -> float | None:
        """Pitch/yaw damping summed over the table's lifting parts, about this CG.

        The table stores the normal-force slope's zeroth, first and second
        moments about the nose, so the sum of ``CN_alpha_i (x_i - x_cg)^2``
        is exact for whatever CG the vehicle has right now -- it moves as
        propellant drains, and a column tabulated about one CG would not.
        ``None`` when the table carries no moments.
        """
        if coeffs.cna_sum is None or coeffs.cna_x_m is None or coeffs.cna_x2_m2 is None:
            return None
        cg = np.asarray(cg_body_m, dtype=float) - self.nose_position_body_m
        x_cg = -float(np.dot(cg, self.body_axis))       # station from the nose, aft positive
        second = coeffs.cna_x2_m2 - 2.0 * x_cg * coeffs.cna_x_m + x_cg * x_cg * coeffs.cna_sum
        return -2.0 * second / (self.reference_length_m ** 2)

    def forces_and_moments(
        self,
        velocity_body_mps: np.ndarray,
        rho_kg_m3: float,
        speed_of_sound_mps: float,
        cg_body_m: np.ndarray,
        omega_body_radps: np.ndarray | None = None,
        power_on: bool = False,
    ) -> AeroForces:
        """Body-frame aerodynamic force and moment.

        Args:
            omega_body_radps: Body angular rate [rad/s]. Omitting it drops the
                damping term, leaving the undamped behaviour this model had
                before -- useful for isolating damping in tests, wrong for
                flight.
            power_on: The motor is burning, so the base is filled by the
                plume. Uses the table's ``cd_power_on`` column when it has
                one; a single-column table is used as is either way.
        """
        v_body = np.asarray(velocity_body_mps, dtype=float)
        speed = float(np.linalg.norm(v_body))
        if speed < 1e-9 or rho_kg_m3 <= 0:
            coeffs = self.database.lookup(0.0, 0.0)
            zero = np.zeros(3)
            return AeroForces(zero, zero, coeffs, 0.0, 0.0, zero, zero, 0.0)

        axial_speed = float(np.dot(v_body, self.body_axis))
        lateral = v_body - axial_speed * self.body_axis
        lateral_speed = float(np.linalg.norm(lateral))
        alpha_deg = float(np.degrees(np.arctan2(lateral_speed, max(abs(axial_speed), 1e-9))))
        mach = speed / speed_of_sound_mps if speed_of_sound_mps > 1e-9 else 0.0
        coeffs = self.database.lookup(mach, alpha_deg)
        q_dyn = 0.5 * rho_kg_m3 * speed * speed

        # The table's inside its alpha range, the extension's beyond it;
        # the axial force recovered from the wind-axis drag either way.
        ca, cn, x_cp = self.coefficients_at(mach, alpha_deg, power_on)

        axial_dir = -np.sign(axial_speed or 1.0) * self.body_axis
        drag_body = q_dyn * self.reference_area_m2 * ca * axial_dir

        if lateral_speed > 1e-9:
            normal_dir = -lateral / lateral_speed
            normal_body = q_dyn * self.reference_area_m2 * cn * normal_dir
        else:
            normal_body = np.zeros(3)

        force_body = drag_body + normal_body
        cg_body = np.asarray(cg_body_m, dtype=float)

        # Two equivalent descriptions of the static pitching moment. A table
        # with centre-of-pressure data gives it as a moment arm, which is
        # preferred because it stays correct as the CG migrates during the
        # burn. Some exports instead zero x_cp and describe the moment through
        # the cm coefficient alone -- which used to be read from the CSV,
        # stored, and never used, silently leaving such a vehicle with no
        # restoring moment at all.
        cp_body = None
        if self.database.has_x_cp or not self.database.has_cm:
            # Stations run aft from the nose and the body axis points
            # forward, so the CP is x_cp *against* the axis -- the mapping
            # ``frames.station_to_body`` applies to the CG. It used to be
            # placed along the axis, x_cp ahead of the nose. The arm then
            # pointed the wrong way and every stable vehicle's restoring
            # moment diverged; the damping estimate, squared on an arm of
            # x_cp + x_cg instead of x_cp - x_cg, was some hundreds of times
            # too strong, froze the attitude, and hid the divergence -- a
            # thrusting rocket with a frozen attitude still flies along its
            # own axis, so alpha collapsed to zero and nothing looked wrong.
            cp_body = self.nose_position_body_m - self.body_axis * x_cp
            static_moment = np.cross(cp_body - cg_body, force_body)
            arm_m = float(np.dot(cp_body - cg_body, self.body_axis))
        else:
            # cm is referenced about the table's own moment reference point,
            # not about the current CG, so it cannot track CG migration. Used
            # only when there is nothing better. It is a coefficient on the
            # *table's* reference length, not this model's.
            moment_magnitude = (
                q_dyn * self.reference_area_m2 * self.database.reference_length_m * coeffs.cm
            )
            if lateral_speed > 1e-9:
                pitch_axis = np.cross(self.body_axis, lateral / lateral_speed)
                norm = np.linalg.norm(pitch_axis)
                pitch_axis = pitch_axis / norm if norm > 1e-12 else np.zeros(3)
            else:
                pitch_axis = np.zeros(3)
            # The standard convention: a negative Cm is nose-down at positive
            # alpha, i.e. restoring. ``pitch_axis`` is body x flow, which
            # points the other way, so the sign flips here. It used to be
            # applied unflipped, and a table with the standard sign flew as
            # an unstable vehicle.
            static_moment = -moment_magnitude * pitch_axis
            # Recover an equivalent arm so the damping estimate still has one:
            # the moment equals arm x N, so the arm is Cm L / CN, negative
            # (aft) for a restoring Cm.
            normal_magnitude = q_dyn * self.reference_area_m2 * coeffs.cn
            arm_m = (
                moment_magnitude / normal_magnitude
                if abs(normal_magnitude) > 1e-9
                else 0.0
            )

        # The arm is measured forward; the margin is quoted aft, positive
        # for a stable vehicle, the way every other part of the tool does.
        static_margin = -arm_m

        # Damping. The table's per-part moments when it has them; otherwise
        # the single-surface estimate on the static margin -- the CP-CG
        # offset along the body axis, the transverse offset contributing
        # nothing to pitch damping.
        if self.cmq_override is not None:
            cmq = self.cmq_override
        else:
            cmq = self.cmq_from_table(coeffs, cg_body)
            if cmq is None:
                cmq = self.estimate_cmq(mach, alpha_deg, static_margin)

        # Roll: an explicit clp overrides, else the table's, else nothing.
        clp = self.clp if self.clp != 0.0 else float(coeffs.clp or 0.0)
        cl_roll = float(coeffs.cl_roll or 0.0)
        if self.cant_offset_rad != 0.0 and coeffs.cl_cant is not None:
            cl_roll += float(coeffs.cl_cant) * self.cant_offset_rad
        diameter = self.reference_diameter_m

        # Fin cant drives roll whether or not the vehicle is rolling yet.
        if cl_roll != 0.0:
            static_moment = static_moment + (
                q_dyn * self.reference_area_m2 * diameter * cl_roll * self.body_axis
            )

        damping_moment = np.zeros(3)
        if omega_body_radps is not None:
            omega = np.asarray(omega_body_radps, dtype=float)
            roll_rate = float(np.dot(omega, self.body_axis))
            omega_axial = roll_rate * self.body_axis
            omega_transverse = omega - omega_axial

            # q_dyn * (L / 2V) == 0.25 * rho * V * L, so this is linear in
            # airspeed and has no singularity as V -> 0. Pitch and yaw are
            # referenced to the length, roll to the diameter.
            pitch_scale = (
                0.25 * rho_kg_m3 * speed
                * self.reference_area_m2
                * self.reference_length_m ** 2
            )
            roll_scale = 0.25 * rho_kg_m3 * speed * self.reference_area_m2 * diameter ** 2
            damping_moment = (
                pitch_scale * cmq * omega_transverse + roll_scale * clp * omega_axial
            )

        moment_body = static_moment + damping_moment
        return AeroForces(
            force_body,
            moment_body,
            coeffs,
            mach,
            alpha_deg,
            static_moment,
            damping_moment,
            cmq,
            static_margin,
            cp_body,
            clp,
            cl_roll,
            ca,
            cn,
            x_cp,
        )


def _unit(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=float)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        raise ValueError("Axis vector must be non-zero.")
    return value / norm
