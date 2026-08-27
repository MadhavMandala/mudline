"""The rotation between the model's axes and the simulator's.

The parametric model is +Z aft with the nose tip at the origin, so a
station is a Z. The simulator's body frame is +Y forward: the nose tip is
the origin and everything else has negative y. The mapping is a *rotation*,
not a relabelling, and the difference bites on the inertia tensor: it
transforms as ``I_sim = R I_def R^T``. For an axisymmetric vehicle whose
model inertia is ``diag(Ixx, Iyy, Izz)`` with ``Izz`` the roll term, the
simulator sees ``diag(Ixx, Izz, Iyy)`` -- roll lands on the *y* axis, which
is exactly where the simulator's body axis is. A plain permutation that put
roll anywhere else would leave the vehicle with a pitch inertia about its
roll axis, some two orders of magnitude wrong.
"""

from __future__ import annotations

import numpy as np

#: Rotation from model axes (+Z aft) to simulator body axes (+Y forward).
DEF_TO_SIM = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
])


def station_to_body(station_m: float, radial_offset_m: float = 0.0) -> np.ndarray:
    """Position of a station in the simulator's body frame [m]."""
    return DEF_TO_SIM @ np.array([radial_offset_m, 0.0, station_m])


def inertia_to_body(inertia_def: np.ndarray) -> np.ndarray:
    """Rotate an inertia tensor from model axes into body axes."""
    inertia_def = np.asarray(inertia_def, dtype=float)
    return DEF_TO_SIM @ inertia_def @ DEF_TO_SIM.T
