"""Engine thrust and mass flow models.

Thrust reference convention
---------------------------
A thrust curve is meaningless without knowing what ambient pressure it was
measured at. Published motor data is usually **sea-level** static-fire thrust;
analytical models usually quote **vacuum** thrust. Subtracting the ambient
pressure term from a sea-level curve double-counts the loss, so the convention
is now declared explicitly via ``thrust_reference`` and the curve is normalised
to vacuum thrust internally.

Vacuum thrust already contains the exit-pressure term::

    F_vac = mdot * Ve + Pe * Ae

so the altitude correction is the single-sided

    F(Pa) = F_vac - Pa * Ae

which is exact, not an approximation, once the curve is known to be vacuum
referenced.

Mass flow
---------
Mass flow is set by chamber pressure and throat area. It does **not** depend on
ambient pressure -- a motor does not burn propellant more slowly at sea level.
It is therefore derived once from the vacuum curve::

    mdot(t) = F_vac(t) / (Isp_vac * g0)

Deriving it from the pressure-corrected thrust and an altitude-blended Isp (as
this module previously did) applies the altitude effect twice and made burn rate
vary by a factor of ~3.7 between sea level and vacuum.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

G0 = 9.80665              # standard gravity [m/s^2]
P_STD = 101325.0          # standard sea-level pressure [Pa]

THRUST_REFERENCES = ("vacuum", "sea_level")


class Engine:
    """Engine with thrust curve and gimbal capability."""

    def __init__(self, thrust_curve: np.ndarray, time_points: np.ndarray,
                 isp_vac: float, isp_sl: float, nozzle_area: float,
                 thrust_reference: str = "vacuum"):
        """
        Args:
            thrust_curve: Thrust vs time [N], at the ambient condition named by
                ``thrust_reference``.
            time_points: Time array [s]
            isp_vac: Vacuum specific impulse [s]
            isp_sl: Sea level specific impulse [s]
            nozzle_area: Nozzle exit area [m^2]
            thrust_reference: "vacuum" or "sea_level" -- what ``thrust_curve``
                was measured or computed at.
        """
        if thrust_reference not in THRUST_REFERENCES:
            raise ValueError(
                f"thrust_reference must be one of {THRUST_REFERENCES}; "
                f"got {thrust_reference!r}"
            )

        self.isp_vac = isp_vac
        self.isp_sl = isp_sl
        self.nozzle_area = nozzle_area
        self.thrust_reference = thrust_reference

        # Normalise the supplied curve to vacuum thrust so everything
        # downstream works from one unambiguous reference.
        curve = np.asarray(thrust_curve, dtype=float)
        if thrust_reference == "sea_level":
            curve = curve + P_STD * nozzle_area

        self._vacuum_curve = interp1d(
            time_points, curve, bounds_error=False, fill_value=0.0
        )

        # Gimbal limits [rad]
        self.max_gimbal = np.radians(8.0)

    def vacuum_thrust_at(self, t: float) -> float:
        """Vacuum thrust at time t [N]."""
        return max(0.0, float(self._vacuum_curve(t)))

    def mass_flow_at(self, t: float) -> float:
        """Propellant mass flow at time t [kg/s].

        Independent of altitude by construction -- see the module docstring.
        """
        if self.isp_vac <= 0:
            return 0.0
        return self.vacuum_thrust_at(t) / (self.isp_vac * G0)

    def thrust_at(self, t: float, p_ambient: float = 0.0) -> tuple:
        """Return (thrust_mag, mass_flow) at time and ambient pressure.

        Thrust is clamped at zero, but mass flow is taken from the vacuum curve
        rather than the clamped thrust. Propellant keeps flowing even where the
        pressure term drives net thrust to zero, and a clamped thrust must never
        be able to produce a negative -- previously it could, which made the
        vehicle gain mass during coast.
        """
        vacuum_thrust = self.vacuum_thrust_at(t)
        thrust = vacuum_thrust - p_ambient * self.nozzle_area
        return max(0.0, thrust), self.mass_flow_at(t)

    def effective_isp(self, t: float, p_ambient: float) -> float:
        """Delivered specific impulse at this time and ambient pressure [s].

        Derived from the thrust and mass flow rather than interpolated, so it
        is consistent with them by construction.
        """
        thrust, mass_flow = self.thrust_at(t, p_ambient)
        if mass_flow <= 0:
            return 0.0
        return thrust / (mass_flow * G0)

    def implied_isp_sl(self, t: float | None = None) -> float:
        """Sea-level Isp implied by ``isp_vac`` and ``nozzle_area``.

        The model is over-specified: given a vacuum thrust curve, isp_vac and
        nozzle area fully determine sea-level performance, so a separately
        declared ``isp_sl`` is a claim that can disagree. Compare the two to
        catch an inconsistent nozzle area.

        Uses the peak of the curve when ``t`` is not given, since that is where
        the ratio is best conditioned.
        """
        if t is None:
            times = np.asarray(self._vacuum_curve.x, dtype=float)
            thrusts = np.asarray(self._vacuum_curve.y, dtype=float)
            t = float(times[int(np.argmax(thrusts))])

        mass_flow = self.mass_flow_at(t)
        if mass_flow <= 0:
            return 0.0
        return (self.vacuum_thrust_at(t) - P_STD * self.nozzle_area) / (mass_flow * G0)

    def gimbal_transform(self, pitch: float, yaw: float) -> np.ndarray:
        """Return thrust direction vector in body frame given gimbal angles."""
        # Clamp to limits
        pitch = np.clip(pitch, -self.max_gimbal, self.max_gimbal)
        yaw = np.clip(yaw, -self.max_gimbal, self.max_gimbal)

        # Rotation about Y (pitch) then Z (yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([sy, -sp * cy, cp * cy])  # Body frame


def nozzle_area_for_isp_pair(
    vacuum_thrust: float, isp_vac: float, isp_sl: float
) -> float:
    """Nozzle exit area consistent with a declared vacuum/sea-level Isp pair.

    From ``Isp_sl = Isp_vac - P_std * Ae / (mdot * g0)`` with
    ``mdot = F_vac / (Isp_vac * g0)``::

        Ae = (Isp_vac - Isp_sl) * F_vac / (Isp_vac * P_std)

    Useful for setting a physically sensible default instead of guessing.
    """
    if isp_vac <= 0 or vacuum_thrust <= 0:
        return 0.0
    return max(0.0, (isp_vac - isp_sl) * vacuum_thrust / (isp_vac * P_STD))
