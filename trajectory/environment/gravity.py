"""Gravity models."""

import numpy as np


MU_EARTH = 3.986004418e14  # m^3/s^2
R_EARTH = 6371000.0        # m


def gravity_earth(position: np.ndarray, j2: bool = False) -> np.ndarray:
    """
    Compute gravitational acceleration at position.

    Args:
        position: ECI/EF position vector [m]
        j2: Include J2 oblateness perturbation

    Returns:
        Acceleration vector [m/s^2]
    """
    r = np.linalg.norm(position)
    r_hat = position / r

    # Central gravity
    g = -MU_EARTH / r**2 * r_hat

    if j2:
        # J2 perturbation (zonal harmonic)
        J2 = 1.08263e-3
        z = position[2]
        factor = 1.5 * J2 * (R_EARTH / r)**2
        g[0] *= (1 - factor * (5 * (z/r)**2 - 1))
        g[1] *= (1 - factor * (5 * (z/r)**2 - 1))
        g[2] *= (1 - factor * (5 * (z/r)**2 - 3))

    return g


#: Standard gravity, the g0 every Isp in the tool is defined against.
G0 = 9.80665


def gravity_simple(altitude: float) -> float:
    """Gravity magnitude at altitude: g0 at the surface, falling as 1/r^2.

    ``MU_EARTH / (R_EARTH + h)^2`` with the mean radius gave 9.8203 m/s^2 at
    the surface -- 0.14% high, a constant bias on every gravity loss, and a
    different g from the one the engine's Isp turns into mass flow.
    """
    r = R_EARTH + altitude
    return G0 * (R_EARTH / r) ** 2


#: Earth's sidereal rotation rate [rad/s].
OMEGA_EARTH_RADPS = 7.292115e-5


def earth_rotation_enu(latitude_rad: float) -> np.ndarray:
    """Earth's angular velocity in the local ENU frame [East, Up, North].

    The axis points at the celestial pole: up by the sine of the latitude,
    north by its cosine, with no east component.
    """
    return OMEGA_EARTH_RADPS * np.array([0.0, np.sin(latitude_rad), np.cos(latitude_rad)])


def coriolis_acceleration(velocity_eun: np.ndarray, latitude_rad: float) -> np.ndarray:
    """The Coriolis acceleration ``-2 Omega x v`` felt in the rotating frame.

    Centrifugal acceleration is not added: standard gravity already contains
    it, which is why ``g0`` is 9.80665 rather than the 9.82 of gravitation
    alone. What is left is the velocity-dependent term. On a vertical
    launch it deflects a rising vehicle west and a falling one east -- the
    Eotvos effect -- and on a long ballistic arc it turns the track to the
    right in the northern hemisphere. Tens of metres on a 70 km flight:
    small, and a systematic bias a landing ellipse should not carry.

    The simulator's inertial triple is [East, Up, North], which as a
    physical frame is left-handed: East x Up is South. Every other cross
    product in the simulator is taken within one frame and is consistent
    with itself, so the mirror never shows. Earth's rotation is a physical
    pseudo-vector, though, and the product has to be formed in a
    right-handed frame -- [East, North, Up] here -- and mapped back.
    """
    east, up, north = np.asarray(velocity_eun, dtype=float)
    v_enu = np.array([east, north, up])
    omega_enu = OMEGA_EARTH_RADPS * np.array([0.0, np.cos(latitude_rad), np.sin(latitude_rad)])
    a_enu = -2.0 * np.cross(omega_enu, v_enu)
    return np.array([a_enu[0], a_enu[2], a_enu[1]])
