"""Rotational equations of motion: Euler's equations, for a body losing mass."""

import numpy as np


def RotationalEOM(omega: np.ndarray, moments: np.ndarray,
                  inertia: np.ndarray, inertia_inv: np.ndarray,
                  inertia_rate: np.ndarray | None = None) -> np.ndarray:
    """Angular acceleration from Euler's equations.

        I dw/dt + w x (I w) + dI/dt w = M

    Args:
        omega: Angular velocity in the body frame [p, q, r] [rad/s]
        moments: Total moment in the body frame [N-m], jet damping included
        inertia: Inertia tensor [kg-m^2]
        inertia_inv: Its inverse
        inertia_rate: dI/dt [kg-m^2/s] for a body losing mass; ``None``
            means rigid, which is what this equation used to assume.

    The dI/dt term is the part of d(Iw)/dt a variable-mass body carries. As
    propellant leaves, the inertia about the CG falls and, with nothing
    acting, the rate rises to conserve the angular momentum the vehicle
    keeps. Together with the jet-damping moment -- the angular momentum the
    exhaust takes with it, applied by the force model -- this is Thomson's
    result: a net damping of mdot (l_e^2 - rho_p^2) w, positive when the
    nozzle is farther from the CG than the propellant being burned (every
    rocket that matters) and negative for the other kind. Neither term on
    its own is that result, which is why both are here.

    Returns:
        Angular acceleration [dp/dt, dq/dt, dr/dt] [rad/s^2]
    """
    i_omega = inertia @ omega
    gyro = np.cross(omega, i_omega)
    total = moments - gyro
    if inertia_rate is not None:
        total = total - inertia_rate @ omega
    return inertia_inv @ total
