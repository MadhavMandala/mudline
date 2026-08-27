"""Rocket Trajectory Simulator - Main Entry Point"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from trajectory.vehicle import (
    AeroDatabase,
    Engine,
    MassProperties,
    RasaeroAeroModel,
    VehicleCadModel,
    load_saved_vehicle,
    load_vehicle_cad,
)
from trajectory.environment import Atmosphere
from trajectory.environment.gravity import coriolis_acceleration, gravity_simple
from trajectory.environment.wind import WindModel
from trajectory.eom import TranslationalEOM, quat_to_dcm, quaternion_derivative
from trajectory.frames import station_to_body
from trajectory.eom.rotational import RotationalEOM
from trajectory.propulsion import PropulsionModel, load_propulsion_model
from trajectory.sim import (
    EventDetector,
    LaunchRail,
    TrajectoryIntegrator,
)
from trajectory.vehicle.recovery import Parachute, RecoverySystem, standard_recovery
from trajectory.analysis.statistics import flight_statistics
from trajectory.vehicle.engine import nozzle_area_for_isp_pair


# State layout: [x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r, prop_mass]
#                0  1  2   3   4   5   6   7   8   9  10 11 12     13
# Propellant mass is carried as a state so it is integrated from mdot rather
# than reconstructed from elapsed time. Indices 0-12 are unchanged, so
# events.py, integrator.state_to_dict and statistics.py are unaffected.
PROP_IDX = 13
STATE_SIZE = 14

#: The airframe's own axis in the body frame: +Y forward. The aerodynamics
#: and the angle of attack are referenced to it. The thrust axis is a
#: separate thing -- it coincides with this until a nozzle is canted, and a
#: canted nozzle must not tilt the aerodynamics with it.
BODY_AXIS = np.array([0.0, 1.0, 0.0])


@dataclass
class FlightPoint:
    """Everything the force model knows at one instant.

    Built on every derivative evaluation and returned by ``evaluate``. The
    integrator keeps only the state; the flight log re-evaluates the stored
    states through this afterwards to recover what was computed on the way
    and thrown away -- thrust, drag, angle of attack, the centre of
    pressure, the static margin, the acceleration the airframe felt.
    """

    t: float
    altitude_m: float
    rho_kg_m3: float
    pressure_pa: float
    speed_of_sound_mps: float
    mass_kg: float
    cg_body_m: np.ndarray
    inertia_kg_m2: np.ndarray
    inertia_inv: np.ndarray
    #: dI/dt, for Euler's equations on a body losing mass.
    inertia_rate: np.ndarray
    mass_flow_kgps: float
    thrust_n: float
    thrust_body_n: np.ndarray
    thrust_inertial_n: np.ndarray
    dcm_b2i: np.ndarray
    wind_inertial_mps: np.ndarray
    v_rel_body_mps: np.ndarray
    airspeed_mps: float
    mach: float
    dynamic_pressure_pa: float
    alpha_deg: float
    #: The aero model's own report, or ``None`` on the fallback drag law.
    aero: object | None
    aero_force_inertial_n: np.ndarray
    aero_moment_body_nm: np.ndarray
    chute_cda_m2: float
    chute_force_inertial_n: np.ndarray
    gravity_inertial_n: np.ndarray
    force_inertial_n: np.ndarray
    #: Total body moment, jet damping included.
    moment_body_nm: np.ndarray
    jet_damping_moment_body_nm: np.ndarray
    on_rail: bool
    #: The Coriolis pseudo-force, zero unless the launch site has a latitude.
    coriolis_inertial_n: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: The canopy's moment about the CG, zero unless it pulls at a harness point.
    chute_moment_body_nm: np.ndarray = field(default_factory=lambda: np.zeros(3))
    #: ``"rail"``, ``"tipoff"`` or ``"free"``; ``on_rail`` is the first two.
    rail_phase: str = "free"

    def as_tuple(self) -> tuple:
        """The ``compute_forces_moments`` return, for callers that unpack it."""
        return (
            self.force_inertial_n, self.moment_body_nm, self.mass_kg,
            self.inertia_kg_m2, self.inertia_inv, self.mass_flow_kgps,
        )


def jet_damping_moment(mass_flow_kgps: float, r_exit_body_m: np.ndarray,
                       omega_body_radps: np.ndarray) -> np.ndarray:
    """Moment from the exhaust carrying the vehicle's rotation away with it.

    Mass leaving at the nozzle exit, a vector ``r_e`` from the CG, leaves
    with the velocity ``w x r_e`` the rotation gave it; the angular momentum
    it removes is ``mdot r_e x (w x r_e)`` per second, and the vehicle feels
    the negative of that. For an exit on the body axis the roll component
    vanishes and the transverse rates see ``-mdot |r_e|^2 w``, the textbook
    jet-damping term. It is what damps a slender vehicle in the first second
    off the rail, before there is enough airspeed for the fins to.
    """
    if mass_flow_kgps <= 0.0:
        return np.zeros(3)
    r = np.asarray(r_exit_body_m, dtype=float)
    w = np.asarray(omega_body_radps, dtype=float)
    return -float(mass_flow_kgps) * np.cross(r, np.cross(w, r))


def transverse_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to ``axis`` and to each other.

    For the body axis +Y they are +X and +Z, so that a tilt "toward X"
    means what it says; for any other axis the first is X projected off it.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    seed = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(seed, axis))) > 0.9:
        seed = np.array([0.0, 0.0, 1.0])
    e1 = seed - float(np.dot(seed, axis)) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, axis)
    return e1, e2


class RocketSimulation:
    """Full 6-DOF rocket trajectory simulation."""

    def __init__(self):
        # Environment
        self.atm = Atmosphere()
        self.wind = WindModel()

        # Vehicle components (initialized with defaults). In the simulator's
        # body frame -- +Y forward, nose at the origin -- so the dry CG is a
        # metre and a half *behind* the nose and the small inertia is roll,
        # on Y. These used to be written in the model's axes (CG at +0.5,
        # roll on Z), which put the CG ahead of the nose tip and a 5:1 split
        # between the two transverse inertias of a body of revolution.
        self.mass_props = MassProperties(
            dry_mass=50.0,         # kg
            prop_mass=150.0,       # kg
            cg_dry=np.array([0.0, -1.5, 0.0]),
            i_tensor_dry=np.diag([10.0, 2.0, 10.0])  # kg-m^2, roll on Y
        )

        # Simple linear thrust curve, quoted as vacuum thrust.
        thrust_times = np.array([0, 30, 31, 120])
        thrust_values = np.array([20000, 20000, 0, 0])
        # Exit area consistent with the declared Isp pair (~0.0211 m^2, a 164 mm
        # exit). The previous default of 0.15 m^2 was a 437 mm exit -- larger
        # than the 300 mm vehicle it was bolted to -- and it ate 76% of
        # sea-level thrust.
        self.engine = Engine(
            thrust_curve=thrust_values,
            time_points=thrust_times,
            isp_vac=280.0,
            isp_sl=250.0,
            nozzle_area=nozzle_area_for_isp_pair(20000.0, 280.0, 250.0),
            thrust_reference="vacuum",
        )

        self.reference_area = np.pi * (0.15**2)
        self.vehicle_cad: VehicleCadModel | None = None
        self.aero_database: AeroDatabase | None = None
        self.aero_model: RasaeroAeroModel | None = None
        self.propulsion_model: PropulsionModel | None = None
        self.thrust_axis_body = np.array([0.0, 1.0, 0.0])
        #: Where the thrust line actually points, as small angles off the
        #: thrust axis toward body X and body Z: a misaligned nozzle, a bent
        #: motor mount. Nominally zero; a dispersion study spreads it.
        self.thrust_tilt_rad = np.zeros(2)
        #: Fin cant the vehicle was built with that the table was not --
        #: applied through the table's forcing per radian of cant.
        self.fin_cant_offset_rad: float = 0.0
        # The nozzle exit, at the tail of a four-metre placeholder. It used
        # to sit at the nose tip, which put the jet-damping arm through zero
        # mid-burn.
        self.thrust_position_body_m = np.array([0.0, -4.0, 0.0])
        self.integrator: TrajectoryIntegrator | None = None
        self.launch_rail: LaunchRail | None = None
        self.aero_scale = 1.0

        # Recovery. ``_active_chute`` and ``_deploy_trigger_s`` are set by the
        # phase machinery in run(); they are not user-facing.
        self.recovery: RecoverySystem | None = None
        self._active_chute: Parachute | None = None
        self._deploy_trigger_s: float = 0.0

        # A flight runs until it is over, not until a clock says so. Each
        # phase is integrated in windows of this many seconds until its
        # terminal event fires -- ground impact is armed in every phase --
        # and the windows are joined seamlessly. The runaway guard is the
        # only clock left, and it is there for a vehicle that never comes
        # down (an escape trajectory, a chute sized to hover); it is hours,
        # not minutes, so no real flight ever meets it.
        self.phase_window_s: float = 300.0
        self.runaway_s: float = 4.0 * 3600.0
        #: Solver tolerances and the largest step it may take; ``None``
        #: takes the integrator's defaults. They used to be SciPy's own
        #: loose defaults and there was no way to change them.
        self.rtol: float | None = None
        self.atol: float | None = None
        self.max_step_s: float = 0.5
        #: Launch-site latitude for the Coriolis term; ``None`` leaves the
        #: frame non-rotating, as it always was.
        self.latitude_rad: float | None = None
        #: (time, state) at the rail-exit root of the last run, when it fired.
        self._rail_exit_record: tuple[float, np.ndarray] | None = None
        #: (time, state) when the forward button left and the tip-off began.
        self._tipoff_record: tuple[float, np.ndarray] | None = None
        self._rail_events: list = []

    def _deployed_drag_area(self, t: float) -> float:
        """Effective parachute drag area at time ``t`` [m^2]."""
        if self._active_chute is None:
            return 0.0
        return self._active_chute.drag_area_at(t - self._deploy_trigger_s)

    def _chute_attachment_body(self) -> np.ndarray | None:
        """Where the active canopy pulls, in the body frame; ``None`` for the CG."""
        chute = self._active_chute
        station = getattr(chute, "attachment_station_m", None) if chute is not None else None
        if station is None:
            return None
        return station_to_body(float(station))

    def import_vehicle_cad(
        self,
        filepath: str | Path,
        density_lbm_per_in3: float = 0.098,
    ) -> VehicleCadModel:
        """Load vehicle CAD and apply its dry mass properties to the simulation."""
        cad = load_vehicle_cad(filepath, density_lbm_per_in3=density_lbm_per_in3)
        return self.apply_vehicle_model(cad)

    def import_saved_vehicle(self, filepath: str | Path) -> VehicleCadModel:
        """Load a saved vehicle package and apply it to the simulation."""
        cad = load_saved_vehicle(filepath)
        return self.apply_vehicle_model(cad)

    def apply_vehicle_model(self, cad: VehicleCadModel) -> VehicleCadModel:
        """Apply vehicle mass properties to the simulation."""
        prop_mass = self.mass_props.prop_mass

        self.mass_props = MassProperties(
            dry_mass=cad.dry_mass_kg,
            prop_mass=prop_mass,
            cg_dry=cad.cg_m,
            i_tensor_dry=cad.inertia_kg_m2,
        )
        self.reference_area = cad.reference_area_m2
        self.vehicle_cad = cad
        if self.aero_database is not None:
            self._rebuild_aero_model()
        return cad

    def set_aero_database(self, database) -> None:
        """Adopt a coefficient table and rebuild the aero model on it.

        The public seam for callers that build their own ``AeroDatabase`` --
        the application's flight stage, sweeps, dispersion -- which used to
        assign the attribute and poke ``_rebuild_aero_model`` by hand.
        """
        self.aero_database = database
        self._rebuild_aero_model()

    def import_aero_database(self, filepath: str | Path) -> AeroDatabase:
        """Load a RASAero-derived aerodynamic coefficient table."""
        reference_length = self._reference_length_m()
        database = AeroDatabase.from_csv(filepath, reference_length_m=reference_length)
        self.aero_database = database
        self._rebuild_aero_model()
        return database

    def import_saved_propulsion(self, filepath: str | Path) -> PropulsionModel:
        """Load a saved propulsion package and apply it to the simulation."""
        propulsion = load_propulsion_model(filepath)
        return self.apply_propulsion_model(propulsion)

    def apply_propulsion_model(self, propulsion: PropulsionModel) -> PropulsionModel:
        """Apply propulsion performance and propellant mass to the simulation."""
        self.engine = propulsion.to_engine()
        self.mass_props.prop_mass = propulsion.propellant_mass_kg
        self.mass_props.mass_0 = self.mass_props.dry_mass + self.mass_props.prop_mass
        self.mass_props.cg_prop_full = propulsion.tank_cg_m
        self.thrust_axis_body = propulsion.thrust_axis_body
        self.thrust_position_body_m = propulsion.thrust_position_body_m
        self.propulsion_model = propulsion
        return propulsion

    def _rebuild_aero_model(self) -> None:
        if self.aero_database is None:
            self.aero_model = None
            return
        # The table's own reference length, so the Cmq the flight applies is
        # the Cmq the aero results report. It used to be _reference_length_m,
        # which is 1.0 without a CAD model -- the applied moment is
        # L-independent, but the number on screen was 3.4x off.
        self.aero_model = RasaeroAeroModel(
            self.aero_database,
            reference_area_m2=self.reference_area,
            reference_length_m=self.aero_database.reference_length_m,
            body_axis=BODY_AXIS,
            cant_offset_rad=float(self.fin_cant_offset_rad),
        )

    def thrust_direction_body(self) -> np.ndarray:
        """Unit vector the thrust acts along, in the body frame.

        The thrust axis tilted by ``thrust_tilt_rad`` toward body X and
        body Z. The two transverse directions are fixed by the axis: for
        the usual +Y axis they are +X and +Z themselves.
        """
        axis = np.asarray(self.thrust_axis_body, dtype=float)
        axis = axis / np.linalg.norm(axis)
        tilt = np.asarray(self.thrust_tilt_rad, dtype=float).reshape(2)
        if not np.any(tilt):
            return axis
        e1, e2 = transverse_basis(axis)
        direction = axis + np.tan(tilt[0]) * e1 + np.tan(tilt[1]) * e2
        return direction / np.linalg.norm(direction)

    def _reference_length_m(self) -> float:
        if self.vehicle_cad is None:
            return 1.0
        spans = [
            hi - lo
            for lo, hi in self.vehicle_cad.bounds_m.values()
            if np.isfinite(hi - lo) and hi > lo
        ]
        return max(spans) if spans else 1.0

    def compute_forces_moments(self, state: np.ndarray, t: float) -> tuple:
        """Total inertial force, body moment, mass, inertia and mass flow."""
        return self.evaluate(state, t).as_tuple()

    def evaluate(self, state: np.ndarray, t: float) -> FlightPoint:
        """Evaluate the force model at one state, keeping everything it knows."""
        pos = state[0:3]
        vel = state[3:6]
        q = state[6:10]
        omega = state[10:13]
        prop_remaining = max(0.0, float(state[PROP_IDX]))

        altitude = max(0, pos[1])

        # Environment. The atmosphere wants altitude above sea level; the
        # wind profile wants height above the ground it grows from. Handing
        # it the sea-level altitude doubled a 10 m/s surface wind on a
        # 2,000 m pad.
        rho, p, T, a = self.atm.get_conditions(altitude)
        wind = self.wind.total_wind(max(0.0, pos[1] - self._ground_m()), t)

        # Thrust and mass flow. One call: previously thrust_at was invoked
        # twice per derivative evaluation with the same arguments.
        thrust_mag, mass_flow = self.engine.thrust_at(t, p)

        # Propellant exhaustion is governed by what is left in the tank, not
        # only by the shape of the thrust curve. Without this an over-long
        # curve would keep producing thrust from an empty vehicle.
        if prop_remaining <= 0.0:
            thrust_mag = 0.0
            mass_flow = 0.0

        # Mass properties follow the integrated propellant state.
        mass, cg, i_tensor = self.mass_props.at_propellant(prop_remaining)
        i_inv = np.linalg.inv(i_tensor)

        # dI/dt for Euler's equations on a body losing mass: how the tensor
        # changes per kilogram burned -- the mass model knows where each
        # kilogram leaves from -- times the rate it burns.
        inertia_rate = np.zeros((3, 3))
        if mass_flow > 0.0 and prop_remaining > 0.0:
            step = min(prop_remaining, max(1e-3 * self.mass_props.prop_mass, 1e-6))
            _, _, i_less = self.mass_props.at_propellant(prop_remaining - step)
            inertia_rate = (i_tensor - i_less) / step * (-mass_flow)

        # Attitude. quat_to_dcm is body-to-inertial, the standard reading of
        # the attitude quaternion and the one its derivative propagates; the
        # transpose that used to sit here rotated the vehicle backwards.
        dcm_b2i = quat_to_dcm(q)

        # Aerodynamics
        v_rel = vel - wind
        v_mag = float(np.linalg.norm(v_rel))
        v_rel_body = dcm_b2i.T @ v_rel
        # Angle of attack from the geometry alone, so the log has it whether
        # or not a coefficient table is loaded. Against the airframe axis,
        # not the thrust axis.
        axial = float(np.dot(v_rel_body, BODY_AXIS))
        lateral = float(np.linalg.norm(v_rel_body - axial * BODY_AXIS))
        alpha_deg = (
            float(np.degrees(np.arctan2(lateral, abs(axial)))) if v_mag > 0.1 else 0.0
        )
        mach = v_mag / a if a > 1e-9 else 0.0
        q_dyn = 0.5 * rho * v_mag * v_mag
        aero = None
        aero_force_inertial = np.zeros(3)
        aero_moment_body = np.zeros(3)
        if v_mag > 0.1 and self.aero_model is not None:
            # omega is required for the damping term; without it the attitude
            # oscillation from any disturbance never decays. The base is
            # filled by the plume exactly while thrust is on: the table's
            # power-on drag column during the burn, power-off after it.
            aero = self.aero_model.forces_and_moments(
                v_rel_body, rho, a, cg, omega, power_on=thrust_mag > 0.0,
            )
            aero_force_inertial = dcm_b2i @ aero.force_body_n
            aero_moment_body = aero.moment_body_nm
        elif v_mag > 0.1:
            drag_dir = -v_rel / v_mag
            mach = v_mag / a
            cd = 0.3 + 0.1 * mach if mach < 1 else 0.5  # Simplified fallback
            aero_force_inertial = 0.5 * rho * v_mag**2 * cd * self.reference_area * drag_dir

        # Aerodynamic uncertainty multiplier. Nominally 1.0; dispersion studies
        # perturb it to represent the spread between a predicted coefficient
        # set and the real vehicle, which is the dominant aero uncertainty.
        if self.aero_scale != 1.0:
            aero_force_inertial = aero_force_inertial * self.aero_scale
            aero_moment_body = aero_moment_body * self.aero_scale

        # Thrust along the line it actually acts on, which is the thrust
        # axis unless the nozzle is misaligned.
        thrust_body = self.thrust_direction_body() * thrust_mag
        thrust = dcm_b2i @ thrust_body

        # Gravity, and the Coriolis pseudo-force of a frame fixed to a
        # rotating Earth when the site has a latitude.
        gravity = np.array([0.0, -gravity_simple(altitude), 0.0]) * mass
        coriolis = (
            mass * coriolis_acceleration(vel, self.latitude_rad)
            if self.latitude_rad is not None else np.zeros(3)
        )

        # Parachute drag, if a canopy is deployed and inflating/inflated.
        # At the harness attachment when the canopy has one: the drag
        # opposes that point's own motion through the air, v + w x r, and
        # its moment about the CG is what swings the body to hang nose-up
        # and what damps the swing. Through the CG otherwise -- see
        # trajectory.vehicle.recovery.
        chute_force = np.zeros(3)
        chute_moment_body = np.zeros(3)
        cda = self._deployed_drag_area(t)
        if cda > 0.0:
            attachment = self._chute_attachment_body()
            if attachment is None:
                if v_mag > 1e-6:
                    chute_force = -0.5 * rho * v_mag * cda * v_rel
            else:
                r_attach = attachment - cg
                v_point = v_rel + dcm_b2i @ np.cross(omega, r_attach)
                point_speed = float(np.linalg.norm(v_point))
                if point_speed > 1e-6:
                    chute_force = -0.5 * rho * point_speed * cda * v_point
                    chute_moment_body = np.cross(r_attach, dcm_b2i.T @ chute_force)

        # Total forces
        forces = gravity + aero_force_inertial + thrust + chute_force + coriolis

        r_thrust = self.thrust_position_body_m - cg
        # The exhaust leaves from the nozzle with the rotation the vehicle
        # gave it, and takes that angular momentum away.
        jet_damping = jet_damping_moment(mass_flow, r_thrust, omega)
        moments_body = (
            np.cross(r_thrust, thrust_body) + aero_moment_body + jet_damping
            + chute_moment_body
        )

        rail_phase = self.launch_rail.phase_of(state, cg) if self.launch_rail is not None else "free"
        on_rail = rail_phase != "free"
        return FlightPoint(
            t=float(t), altitude_m=float(altitude), rho_kg_m3=float(rho),
            pressure_pa=float(p), speed_of_sound_mps=float(a),
            mass_kg=float(mass), cg_body_m=cg, inertia_kg_m2=i_tensor,
            inertia_inv=i_inv, inertia_rate=inertia_rate,
            mass_flow_kgps=float(mass_flow),
            thrust_n=float(thrust_mag), thrust_body_n=thrust_body,
            thrust_inertial_n=thrust, dcm_b2i=dcm_b2i,
            wind_inertial_mps=wind, v_rel_body_mps=v_rel_body,
            airspeed_mps=v_mag, mach=float(mach), dynamic_pressure_pa=float(q_dyn),
            alpha_deg=alpha_deg, aero=aero,
            aero_force_inertial_n=aero_force_inertial,
            aero_moment_body_nm=aero_moment_body,
            chute_cda_m2=float(cda), chute_force_inertial_n=chute_force,
            gravity_inertial_n=gravity, force_inertial_n=forces,
            moment_body_nm=moments_body, jet_damping_moment_body_nm=jet_damping,
            on_rail=bool(on_rail), coriolis_inertial_n=coriolis,
            chute_moment_body_nm=chute_moment_body, rail_phase=rail_phase,
        )

    def tipoff_accelerations(self, point: FlightPoint, state: np.ndarray) -> tuple:
        """CG acceleration and angular acceleration with the aft button pinned.

        The button is a point of the body constrained to the rail's line:
        two lateral constraint forces at it, and a torque about the body
        axis so it does not turn in the slot. Three unknowns, three
        conditions -- the button's lateral acceleration is zero in both
        directions across the rail, the roll acceleration is zero -- and
        the map from unknowns to residuals is affine, so it is solved
        exactly from four evaluations of the free equations rather than
        by penalty or projection. The constraint forces do no work: the
        button moves along the rail and they act across it.
        """
        rail = self.launch_rail
        n1, n2 = transverse_basis(rail.direction)
        dcm = point.dcm_b2i
        mass, inertia, inertia_inv = point.mass_kg, point.inertia_kg_m2, point.inertia_inv
        omega = np.asarray(state[10:13], dtype=float)
        r_pivot = rail.button_bodies()[1] - point.cg_body_m
        centripetal = dcm @ np.cross(omega, np.cross(omega, r_pivot))

        def accelerations(x):
            constraint = x[0] * n1 + x[1] * n2
            a_cg = (point.force_inertial_n + constraint) / mass
            moment = (
                point.moment_body_nm + np.cross(r_pivot, dcm.T @ constraint)
                + x[2] * BODY_AXIS
            )
            alpha = RotationalEOM(omega, moment, inertia, inertia_inv, point.inertia_rate)
            return a_cg, alpha

        def residual(x):
            a_cg, alpha = accelerations(x)
            a_pivot = a_cg + dcm @ np.cross(alpha, r_pivot) + centripetal
            return np.array([a_pivot @ n1, a_pivot @ n2, alpha @ BODY_AXIS])

        r0 = residual(np.zeros(3))
        columns = [residual(unit) - r0 for unit in np.eye(3)]
        x = np.linalg.solve(np.column_stack(columns), -r0)
        return accelerations(x)

    def state_derivative(self, t: float, state: np.ndarray) -> np.ndarray:
        """Compute state derivative for integration."""
        point = self.evaluate(state, t)
        forces, mass, mass_flow = (
            point.force_inertial_n, point.mass_kg, point.mass_flow_kgps
        )

        # Translational
        deriv = TranslationalEOM(state, forces, mass)

        # Rotational: Euler's equations for a body losing mass. The
        # inertia-rate term rides with the jet-damping moment already in
        # the point's total; see trajectory.eom.rotational for why both.
        omega = state[10:13]
        omega_dot = RotationalEOM(
            omega, point.moment_body_nm, point.inertia_kg_m2, point.inertia_inv,
            point.inertia_rate,
        )

        # Quaternion propagation
        q = state[6:10]
        q = q / np.linalg.norm(q)  # Normalize

        # While the vehicle is on the rail it is not a free body: the rail
        # reacts every transverse force and every moment. Applying the
        # constraint to the derivative keeps it exact under the adaptive
        # integrator -- clamping the state after each step instead would let
        # the solver's error estimate see a discontinuity that is not real.
        # Pivoting on the aft button, the constraint is the button's, and
        # the vehicle turns about it.
        if point.rail_phase == "rail":
            deriv[3:6] = self.launch_rail.constrain_acceleration(
                deriv[3:6], state[3:6]
            )
            omega = np.zeros(3)
            omega_dot = np.zeros(3)
        elif point.rail_phase == "tipoff":
            deriv[3:6], omega_dot = self.tipoff_accelerations(point, state)

        result = np.zeros(STATE_SIZE)
        result[0:6] = deriv  # Position and velocity
        # The exact derivative, 0.5*Omega(w)q. This used to be computed as
        # propagate_quaternion(q, omega, 1.0) - q, on the stated assumption that
        # a unit first-order step is the derivative. It is not: that function
        # renormalises before returning, so the difference carried a stray
        # norm-correction term worth 3.7% at 0.15 rad/s and 41% at 2 rad/s,
        # always directed along -q and so quietly shrinking the quaternion.
        result[6:10] = quaternion_derivative(q, omega)
        result[10:13] = omega_dot  # Angular acceleration

        # Propellant depletion, integrated rather than reconstructed from t.
        result[PROP_IDX] = -mass_flow

        return result

    def initial_state(self, rail: LaunchRail) -> np.ndarray:
        """Build the state vector at ignition, sitting at rest on the rail."""
        state0 = np.zeros(STATE_SIZE)
        # With buttons, the aft button sits at the rail's foot and the CG
        # is up the rail from it; without, the CG starts at the foot.
        _, cg, _ = self.mass_props.at_propellant(self.mass_props.prop_mass)
        offset = rail.start_offset_m(cg, BODY_AXIS)
        state0[0:3] = rail.position_m + offset * rail.direction
        state0[3:6] = np.zeros(3)                     # at rest on the pad
        state0[6:10] = rail.initial_quaternion(self.thrust_axis_body)
        state0[PROP_IDX] = self.mass_props.prop_mass  # full tank
        return state0

    def run(
        self,
        launch_azimuth: float = 0.0,
        launch_elevation: float = np.radians(89),
        rail_length_m: float = 5.0,
        pad_position_m: np.ndarray | None = None,
        t_max: float | None = None,
        dt: float = 0.1,
        events: list | None = None,
        recovery: RecoverySystem | None = None,
        rail_buttons_m: tuple[float, float] | None = None,
    ) -> dict:
        """
        Run trajectory from launch until it is back on the ground.

        Args:
            launch_azimuth: Launch direction from North, positive toward East [rad]
            launch_elevation: Launch angle up from horizontal [rad]; pi/2 is vertical
            rail_length_m: Rail travel before the vehicle flies freely. 0 disables.
            pad_position_m: Inertial position of the rail foot [m]
            t_max: Optional cutoff [s]. ``None`` -- the default -- flies
                until the vehicle is back at the altitude it launched from.
                A cutoff exists for callers that want a truncated flight on
                purpose (a quick test, an ascent-only comparison); it is not
                how a flight normally ends. A hard stop used to be the only
                ending there was, and a slow descent from a high apogee
                simply stopped in mid-air with the main never deployed.
            dt: Output sample interval [s]
            events: Extra event functions, appended to ground impact
            recovery: Parachute configuration. When supplied the flight is
                integrated in phases with a restart at each deployment.
            rail_buttons_m: Stations of the two rail buttons from the nose
                tip, for the tip-off phase; ``None`` constrains the CG.

        Returns:
            Integration result, with ``rail_exit``, ``phases`` and ``landed``
            attached. ``landed`` is False only when a cutoff or the runaway
            guard ended the flight before the ground did.

        Both launch angles are now honoured. They were previously accepted and
        discarded, and the vehicle began every flight already moving at 50 m/s.
        """
        self.launch_rail = LaunchRail(
            azimuth_rad=float(launch_azimuth),
            elevation_rad=float(launch_elevation),
            length_m=float(rail_length_m),
            position_m=np.zeros(3) if pad_position_m is None else pad_position_m,
            buttons_m=rail_buttons_m,
        )
        self.recovery = recovery
        self._active_chute = None
        self._deploy_trigger_s = 0.0

        state0 = self.initial_state(self.launch_rail)
        extra = list(events or [])

        # The rail exit is a non-terminal event of the first phase, appended
        # after the caller's so no other index moves. Its root is the exact
        # state the exit speed and the off-the-rail angle of attack are read
        # from; the grid scan it replaces was good to one output step.
        self._rail_exit_record = None
        self._tipoff_record = None
        rail_event = (
            EventDetector.rail_exit(self.launch_rail, self._cg_of)
            if self.launch_rail.length_m > 0.0 else None
        )
        button_event = (
            EventDetector.forward_button_exit(self.launch_rail, self._cg_of)
            if rail_event is not None and self.launch_rail.has_buttons else None
        )
        self._rail_events = [e for e in (rail_event, button_event) if e is not None]
        first_extra = extra + self._rail_events

        if recovery is None or not recovery.enabled:
            result = self._integrate_phase(state0, 0.0, t_max, dt, first_extra)
            self._note_rail_exit(result, rail_event)
            result.phases = [{
                "name": "flight", "t_start": 0.0, "t_end": self._segment_end(result)[0],
            }]
        else:
            result = self._run_with_recovery(
                state0, t_max, dt, extra, recovery,
                ascent_extra=first_extra, rail_event=rail_event,
            )

        # Where the flight left from, so a consumer can draw or quote the
        # trajectory relative to the pad rather than to sea level.
        result.pad_position_m = self.launch_rail.position_m.copy()
        result.rail_exit = self._rail_exit_state(result)
        # Ground impact is event 0 of every phase, so the last phase's event
        # record says whether the flight ended on the ground or on a clock.
        result.landed = self._event_fired(result, index=0)
        if result.landed:
            # The sample grid stops up to one dt short of the event; the
            # event state is exact. End the record on the ground, not above it.
            t_end, y_end = self._segment_end(result)
            if t_end > float(result.t[-1]):
                result.t = np.append(result.t, t_end)
                result.y = np.column_stack([result.y, y_end])
        return result

    def _cg_of(self, state: np.ndarray) -> np.ndarray:
        """The CG in the body frame at this state's propellant load."""
        return self.mass_props.at_propellant(max(0.0, float(state[PROP_IDX])))[1]

    def _ground_m(self) -> float:
        """Altitude of the rail foot: where the flight starts and ends."""
        if self.launch_rail is None:
            return 0.0
        return float(self.launch_rail.position_m[1])

    def _event_root(self, segment, event) -> tuple[float, np.ndarray] | None:
        """The first root of a non-terminal event in a phase, if it fired."""
        if event is None or self.integrator is None:
            return None
        events = list(getattr(self.integrator, "events", []))
        if event not in events:
            return None
        k = events.index(event)
        times = getattr(segment, "t_events", None)
        states = getattr(segment, "y_events", None)
        if times is None or states is None or k >= len(times) or not len(times[k]):
            return None
        return float(times[k][0]), np.asarray(states[k][0], dtype=float).copy()

    def _note_rail_exit(self, segment, rail_event) -> None:
        """Record the rail-exit and forward-button roots of a phase, if they fired."""
        root = self._event_root(segment, rail_event)
        if root is not None:
            self._rail_exit_record = root
        if len(self._rail_events) > 1:
            root = self._event_root(segment, self._rail_events[1])
            if root is not None:
                self._tipoff_record = root

    def _rail_exit_state(self, result) -> dict | None:
        """The exit, evaluated through the force model at its exact state.

        Speed and position as before, plus what the first free instant
        looks like aerodynamically: airspeed, Mach, the wind, and the angle
        of attack the crosswind makes of a vehicle still pointing where the
        rail pointed. Falls back to the grid scan when the root never fired
        -- a flight cut off before the exit.
        """
        rail = self.launch_rail
        if rail is None or rail.length_m <= 0.0:
            return None
        exact = self._rail_exit_record is not None
        if exact:
            t_exit, y_exit = self._rail_exit_record
            y_exit = y_exit.copy()
            if not rail.has_buttons:
                # The root is read from the integrator's dense-output
                # interpolant, which knows nothing of the constraint and
                # lets a few hundredths of a metre per second leak across
                # the rail. On the rail the velocity is along it by
                # construction; put it back there. Pivoting on a button
                # the CG does move across the rail, so this only applies
                # to the CG-constrained rail.
                along = float(np.dot(y_exit[3:6], rail.direction))
                y_exit[3:6] = rail.direction * along
        else:
            sampled = rail.exit_state(result.t, result.y.T, self._cg_of)
            if sampled is None:
                return None
            i = int(np.searchsorted(result.t, sampled["time_s"]))
            t_exit, y_exit = float(result.t[i]), np.asarray(result.y[:, i], dtype=float)

        # No canopy at the rail; the run may have left one armed.
        saved = (self._active_chute, self._deploy_trigger_s)
        self._active_chute = None
        try:
            point = self.evaluate(y_exit, t_exit)
        finally:
            self._active_chute, self._deploy_trigger_s = saved

        record = {
            "time_s": t_exit,
            "velocity_mps": float(np.linalg.norm(y_exit[3:6])),
            "position_m": y_exit[0:3].copy(),
            "airspeed_mps": float(point.airspeed_mps),
            "mach": float(point.mach),
            "alpha_deg": float(point.alpha_deg),
            "wind_mps": float(np.linalg.norm(point.wind_inertial_mps)),
            "exact": exact,
        }
        if rail.has_buttons:
            # What the tip-off left the vehicle with: its axis off the
            # rail's line, and the transverse rate it flies off at.
            axis = point.dcm_b2i @ BODY_AXIS
            omega = y_exit[10:13]
            transverse = omega - float(np.dot(omega, BODY_AXIS)) * BODY_AXIS
            record["tip_off_deg"] = float(np.degrees(np.arccos(
                np.clip(np.dot(axis, rail.direction), -1.0, 1.0)
            )))
            record["pitch_rate_dps"] = float(np.degrees(np.linalg.norm(transverse)))
            record["tipoff_time_s"] = (
                t_exit - self._tipoff_record[0] if self._tipoff_record is not None else 0.0
            )
        return record

    def _integrate_phase(self, state0, t_start, t_end, dt, extra_events):
        """Integrate one continuous-dynamics segment.

        ``t_end`` of ``None`` means until a terminal event fires. Ground
        impact is always the first event, so every phase has one.
        """
        self.integrator = TrajectoryIntegrator(
            derivative_fn=lambda t, y: self.state_derivative(t, y),
            events=[EventDetector.ground(ground_m=self._ground_m()), *extra_events],
            rtol=self.rtol, atol=self.atol, max_step=self.max_step_s,
        )
        if t_end is not None:
            return self.integrator.integrate(
                state0, t_span=(float(t_start), float(t_end)), dt=float(dt),
                method="RK45",
            )

        # Open-ended: integrate window by window until the solver stops on
        # an event (status 1) or fails (status -1). A window that runs to its
        # end (status 0) hands its exact end state -- from the dense output,
        # not the last sample, which sits up to one dt short -- to the next.
        windows = []
        t0, state = float(t_start), np.asarray(state0, dtype=float)
        window = max(float(self.phase_window_s), 2.0 * float(dt))
        while True:
            tf = t0 + window
            seg = self.integrator.integrate(
                state, t_span=(t0, tf), dt=float(dt), method="RK45"
            )
            windows.append(seg)
            if seg.status != 0 or tf - float(t_start) >= self.runaway_s:
                break
            t0, state = tf, np.asarray(seg.sol(tf), dtype=float)
        return self._join_windows(windows)

    @staticmethod
    def _join_windows(windows):
        """One segment from consecutive windows of the same phase.

        Times and states concatenate; the event records merge index by
        index, so a non-terminal event that fired in an early window is not
        lost when a later one carries the terminal event.
        """
        if len(windows) == 1:
            return windows[0]
        combined = windows[-1]
        # A window's grid can land its last sample exactly on its end time,
        # which is where the next window starts; drop the repeat, as
        # _concatenate does for phases.
        times, states = [windows[0].t], [windows[0].y]
        for window in windows[1:]:
            keep = window.t > times[-1][-1]
            times.append(window.t[keep])
            states.append(window.y[:, keep])
        combined.t = np.concatenate(times)
        combined.y = np.concatenate(states, axis=1)
        combined.success = all(w.success for w in windows)
        first = windows[0]
        if getattr(first, "t_events", None) is not None:
            n = len(first.t_events)
            combined.t_events = [
                np.concatenate([np.asarray(w.t_events[i], dtype=float)
                                for w in windows])
                for i in range(n)
            ]
            combined.y_events = [
                np.concatenate([np.asarray(w.y_events[i], dtype=float)
                                .reshape(-1, windows[0].y.shape[0])
                                for w in windows])
                for i in range(n)
            ]
        return combined

    @staticmethod
    def _before_cutoff(t_now: float, t_max: float | None) -> bool:
        return t_max is None or t_now < t_max

    def _run_with_recovery(self, state0, t_max, dt, extra, recovery,
                           ascent_extra=None, rail_event=None):
        """Integrate ascent, drogue descent and main descent as separate phases.

        A restart at each deployment is what keeps the right-hand side
        continuous within every phase. Deploying via a flag inside the
        derivative would put a two-orders-of-magnitude jump in drag area inside
        an integration step, which an adaptive solver cannot integrate across.
        """
        segments = []
        phases = []
        t_now = 0.0
        state = state0

        # --- Phase 1: powered ascent and coast to apogee -------------------
        apogee = EventDetector.apogee(ground_m=self._ground_m())
        ascent_events = extra if ascent_extra is None else ascent_extra
        seg = self._integrate_phase(state, t_now, t_max, dt, [apogee, *ascent_events])
        self._note_rail_exit(seg, rail_event)
        segments.append(seg)
        # A phase ends at its event's root, which is where the next one
        # starts -- not at the last sample before it, up to one dt earlier.
        t_next, state_next = self._segment_end(seg)
        phases.append({"name": "ascent", "t_start": t_now, "t_end": t_next})
        t_now, state = t_next, state_next

        reached_apogee = self._event_fired(seg, index=1)

        # --- Phase 2: drogue descent, or a free fall to the main's height --
        # The height trigger used to live inside the drogue branch, so a
        # main-only system skipped straight to the main at apogee and its
        # deployment height meant nothing.
        deploy_alt = recovery.main_deploy_altitude_m
        falls_to_main = recovery.main is not None and deploy_alt is not None
        if (reached_apogee and (recovery.drogue is not None or falls_to_main)
                and self._before_cutoff(t_now, t_max)):
            self._active_chute = recovery.drogue          # None: a free fall
            self._deploy_trigger_s = t_now
            phase_events = list(extra)
            if deploy_alt is not None:
                # The deployment altitude is above the pad, as an altimeter
                # reads it; the state is above sea level. Compared raw, a
                # main set for 150 m never fired from a pad at 1,400 m.
                phase_events.append(
                    EventDetector.descending_through(self._ground_m() + deploy_alt)
                )
            seg = self._integrate_phase(state, t_now, t_max, dt, phase_events)
            segments.append(seg)
            t_next, state_next = self._segment_end(seg)
            phases.append({
                "name": "drogue" if recovery.drogue is not None else "freefall",
                "t_start": t_now, "t_end": t_next,
                "cda_m2": recovery.drogue.cda_m2 if recovery.drogue is not None else 0.0,
            })
            t_now, state = t_next, state_next

        # --- Phase 3: main descent ----------------------------------------
        airborne = state[1] > self._ground_m()
        if (reached_apogee and recovery.main is not None and airborne
                and self._before_cutoff(t_now, t_max)):
            self._active_chute = recovery.main
            self._deploy_trigger_s = t_now
            seg = self._integrate_phase(state, t_now, t_max, dt, extra)
            segments.append(seg)
            phases.append({
                "name": "main", "t_start": t_now, "t_end": self._segment_end(seg)[0],
                "cda_m2": recovery.main.cda_m2,
            })

        result = self._concatenate(segments)
        result.phases = phases
        return result

    @staticmethod
    def _event_fired(segment, index: int) -> bool:
        events = getattr(segment, "t_events", None)
        return bool(events is not None and len(events) > index and len(events[index]))

    def _segment_end(self, segment) -> tuple:
        """Final time and state of a segment, preferring the event state.

        ``t_eval`` samples on a fixed grid, so the last sampled point generally
        sits *before* the terminal event. Restarting from it would replay a
        fraction of a second of the previous phase's dynamics. The event state
        is exact, so use it when there is one.

        Only a *terminal* event ends a segment. A non-terminal one -- the
        rail exit -- fires early in the ascent and is passed over; taking
        the first non-empty record regardless would have ended the ascent
        on the rail and deployed the drogue there.
        """
        events = list(getattr(self.integrator, "events", []) or []) if self.integrator else []
        for i, (times, states) in enumerate(zip(
            getattr(segment, "t_events", []) or [],
            getattr(segment, "y_events", []) or [],
        )):
            if i < len(events) and not getattr(events[i], "terminal", True):
                continue
            if len(times):
                return float(times[-1]), np.asarray(states[-1], dtype=float).copy()
        return float(segment.t[-1]), segment.y[:, -1].copy()

    @staticmethod
    def _concatenate(segments):
        """Join phase results into one trajectory."""
        kept = [s for s in segments if len(s.t)]
        if not kept:
            return segments[0]

        times = [kept[0].t]
        states = [kept[0].y]
        for seg in kept[1:]:
            # Drop leading samples that duplicate or precede the previous
            # phase's end, which the fixed t_eval grid can produce.
            mask = seg.t > times[-1][-1]
            if not np.any(mask):
                continue
            times.append(seg.t[mask])
            states.append(seg.y[:, mask])

        combined = kept[-1]
        combined.t = np.concatenate(times)
        combined.y = np.concatenate(states, axis=1)
        combined.success = all(s.success for s in kept)
        return combined


def main(argv: list[str] | None = None):
    """Run a sample trajectory, optionally exporting the results."""
    import argparse

    parser = argparse.ArgumentParser(description="Rocket trajectory simulator")
    parser.add_argument("--elevation", type=float, default=89.0,
                        help="Launch elevation from horizontal [deg]")
    parser.add_argument("--azimuth", type=float, default=0.0,
                        help="Launch azimuth from North [deg]")
    parser.add_argument("--rail", type=float, default=5.0,
                        help="Launch rail length [m]")
    parser.add_argument("--recovery", action="store_true",
                        help="Fly with a dual-deploy recovery system")
    parser.add_argument("--t-max", type=float, default=None,
                        help="Integration cutoff [s]")
    parser.add_argument("--export", metavar="DIR",
                        help="Write a CSV time history and summary plot here")
    parser.add_argument("--dispersion", type=int, metavar="N",
                        help="Instead of one flight, run an N-case Monte Carlo")
    parser.add_argument("--processes", type=int, default=1,
                        help="Worker processes for --dispersion")
    args = parser.parse_args(argv)

    print("Rocket Trajectory Simulator")
    print("=" * 40)

    if args.dispersion:
        from trajectory.analysis.dispersion import run_dispersion

        dispersion = run_dispersion(
            n_cases=args.dispersion, n_processes=args.processes
        )
        print()
        print(dispersion.report())
        if args.export:
            from trajectory.analysis.export import (
                plot_dispersion,
                write_dispersion_csv,
            )
            out = Path(args.export)
            print(f"\n  CSV:  {write_dispersion_csv(dispersion, out / 'dispersion.csv')}")
            print(f"  Plot: {plot_dispersion(dispersion, out / 'dispersion.png')}")
        return dispersion

    recovery = None
    if args.recovery:
        recovery = standard_recovery(dry_mass_kg=50.0, main_deploy_altitude_m=500.0)
        print(f"Recovery: {recovery.describe(50.0)}")

    sim = RocketSimulation()
    result = sim.run(
        launch_azimuth=np.radians(args.azimuth),
        launch_elevation=np.radians(args.elevation),
        rail_length_m=args.rail,
        t_max=args.t_max,
        recovery=recovery,
    )

    if not result.success:
        print("Integration failed")
        return result

    print(f"\nTrajectory completed in {result.t[-1]:.1f}s")

    stats = flight_statistics(result.y.T, result.t)
    print("\nFlight Statistics:")
    print(f"  Max Altitude: {stats['max_altitude']:.1f} m")
    print(f"  Max Velocity: {stats['max_velocity']:.1f} m/s")
    print(f"  Apogee Time:  {stats['apogee_time']:.1f} s")
    print(f"  Downrange:    {stats['range']:.1f} m")

    exit_state = getattr(result, "rail_exit", None)
    if exit_state:
        print(
            f"  Rail Exit:    {exit_state['velocity_mps']:.1f} m/s "
            f"at t={exit_state['time_s']:.2f} s"
        )

    from trajectory.analysis.export import max_q
    peak = max_q(result)
    print(
        f"  Max-Q:        {peak['pressure_pa'] / 1000:.1f} kPa at "
        f"{peak['altitude_m'] / 1000:.1f} km, Mach {peak['mach']:.2f}"
    )

    for phase in getattr(result, "phases", []) or []:
        print(f"  Phase {phase['name']:<7} {phase['t_start']:7.1f} -> {phase['t_end']:7.1f} s")

    if args.export:
        from trajectory.analysis.export import export_all
        written = export_all(result, args.export, dry_mass_kg=sim.mass_props.dry_mass)
        print(f"\n  CSV:  {written['csv']}")
        print(f"  Plot: {written['plot']}")

    return result


if __name__ == "__main__":
    main()
