"""Verification of the rotational degrees of freedom against closed forms.

There is no flight data for attitude in the repository -- the Qu8k card
is altitude and axial acceleration -- so the rotational dynamics are held
to what can be computed instead: conservation laws a torque-free body
must obey whatever the integrator does, the exact kinematics of a steady
spin, and the linearised pitch dynamics a table-flown vehicle must match
when its disturbance is small. Each function here returns the numbers a
test then judges; ``validation.imu`` is the path to a real measurement
when a flight computer's log is available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from trajectory import simulation as tm
from trajectory.eom import quat_to_dcm
from trajectory.vehicle.engine import Engine

__all__ = ["TumbleRecord", "torque_free_tumble", "PitchFit", "fit_pitch_oscillation",
           "linear_pitch_prediction"]


def free_body(inertia: np.ndarray, mass_kg: float = 50.0) -> tm.RocketSimulation:
    """A body with nothing acting on it but gravity: no thrust, no air."""
    sim = tm.RocketSimulation()
    sim.engine = Engine(
        thrust_curve=np.array([0.0, 0.0]), time_points=np.array([0.0, 1.0]),
        isp_vac=250.0, isp_sl=250.0, nozzle_area=0.0, thrust_reference="vacuum",
    )
    sim.mass_props = tm.MassProperties(
        dry_mass=mass_kg, prop_mass=0.0, cg_dry=np.array([0.0, -1.5, 0.0]),
        i_tensor_dry=np.asarray(inertia, dtype=float),
    )
    sim.reference_area = 0.0
    sim.launch_rail = None
    sim.rtol, sim.atol = 1e-10, 1e-12
    return sim


@dataclass
class TumbleRecord:
    time_s: np.ndarray
    #: Angular momentum in the inertial frame, per sample.
    momentum: np.ndarray
    #: Rotational kinetic energy per sample.
    energy: np.ndarray
    quaternion_norm: np.ndarray
    dcm: list


def torque_free_tumble(inertia, omega0, duration_s: float = 10.0, dt: float = 0.01) -> TumbleRecord:
    """Integrate a torque-free body from a rate and record what should be conserved."""
    sim = free_body(inertia)
    state = np.zeros(tm.STATE_SIZE)
    state[1] = 100_000.0                 # high enough that it never lands
    state[6] = 1.0
    state[10:13] = omega0
    result = sim._integrate_phase(state, 0.0, duration_s, dt, [])
    inertia = np.asarray(inertia, dtype=float)
    momentum, energy, norms, dcms = [], [], [], []
    for y in result.y.T:
        q = y[6:10]
        dcm = quat_to_dcm(q / np.linalg.norm(q))
        w = y[10:13]
        momentum.append(dcm @ (inertia @ w))
        energy.append(0.5 * w @ inertia @ w)
        norms.append(np.linalg.norm(q))
        dcms.append(dcm)
    return TumbleRecord(result.t, np.array(momentum), np.array(energy), np.array(norms), dcms)


@dataclass
class PitchFit:
    """What a damped oscillation in the angle of attack measured as."""

    damped_frequency_radps: float
    damping_ratio: float
    natural_frequency_radps: float
    peaks: np.ndarray
    peak_times: np.ndarray


def fit_pitch_oscillation(time_s: np.ndarray, alpha_deg: np.ndarray, start_s: float = 0.0) -> PitchFit:
    """Frequency and damping from the peaks of an unsigned angle of attack.

    The log's alpha is a magnitude, so successive peaks are alternate
    half-cycles: the damped period is twice their spacing, and the
    damping ratio follows from the ratio of successive peaks over half a
    period.
    """
    t = np.asarray(time_s, dtype=float)
    a = np.asarray(alpha_deg, dtype=float)
    keep = t >= start_s
    t, a = t[keep], a[keep]
    interior = np.flatnonzero((a[1:-1] > a[:-2]) & (a[1:-1] >= a[2:])) + 1
    interior = interior[a[interior] > 0.05]
    if len(interior) < 3:
        raise ValueError("fewer than three peaks to fit")
    peaks, times = a[interior], t[interior]
    half_period = float(np.mean(np.diff(times[:4])))
    omega_d = np.pi / half_period
    ratios = peaks[1:4] / peaks[:3]
    decrement = -np.log(float(np.mean(ratios)))          # per half period
    zeta_over = decrement / np.pi                        # zeta / sqrt(1 - zeta^2)
    zeta = zeta_over / np.sqrt(1.0 + zeta_over ** 2)
    omega_n = omega_d / np.sqrt(1.0 - zeta ** 2)
    return PitchFit(float(omega_d), float(zeta), float(omega_n), peaks, times)


def linear_pitch_prediction(point, aero, table, inertia_transverse: float) -> tuple[float, float]:
    """``(omega_n, zeta)`` of the linearised short-period mode at this point.

    Two degrees of freedom, angle of attack and pitch rate. The normal
    force turns the flight path, which eats angle of attack at the rate
    ``a = q S CNa / (m V)``; the damping derivative resists the pitch
    rate at ``b = -(rho V S L^2 / 4) Cmq / I``; the static margin
    restores at ``k = q S CNa (x_cp - x_cg) / I``::

        s^2 + (a + b) s + (k + a b) = 0

    so ``omega_n^2 = k + a b`` and ``2 zeta omega_n = a + b``. The heave
    term is worth as much as the damping derivative on a light vehicle:
    leaving it out predicts about half the damping the flight shows.
    """
    from math import sqrt

    model = point.aero
    cn_alpha = table.cn_alpha_per_rad(point.mach, max(point.alpha_deg, 1.0))
    q = float(point.dynamic_pressure_pa)
    area = float(aero.reference_area_m2)
    length = float(aero.reference_length_m)
    k = q * area * cn_alpha * float(model.static_margin_m) / inertia_transverse
    b = -0.25 * point.rho_kg_m3 * point.airspeed_mps * area * length ** 2 * float(model.cmq) / inertia_transverse
    a = q * area * cn_alpha / (point.mass_kg * point.airspeed_mps)
    omega_n = sqrt(k + a * b)
    zeta = (a + b) / (2.0 * omega_n)
    return omega_n, zeta
