"""US Standard Atmosphere 1976.

Coverage
--------
The previous implementation modelled two layers and returned ``rho = 0, p = 0``
above 25 km. A sounding-rocket flight to 150 km therefore spent roughly 83% of
its trajectory in a hard vacuum by construction: no drag on the way up, none on
the way down, and a Mach number pinned to a constant speed of sound.

This version implements the full seven-layer geopotential stack to 86 km
exactly as specified, then continues to 300 km by log-density interpolation of
the standard's own upper-atmosphere table.

Structure below 86 km
---------------------
Each layer has a constant molecular-scale temperature gradient ``L``. Within a
layer, temperature is linear in geopotential altitude and pressure follows

    L != 0:  P = P_b * (T_b / (T_b + L * (H - H_b))) ** (g0 * M / (R* * L))
    L == 0:  P = P_b * exp(-g0 * M * (H - H_b) / (R* * T_b))

Geopotential altitude ``H`` and geometric altitude ``Z`` are related by

    H = r0 * Z / (r0 + Z)

with ``r0 = 6356766 m``, the effective Earth radius the standard defines. Using
geometric altitude directly instead is a ~0.5% error in H at 30 km and ~1.4% at
86 km; it is cheap to do correctly.

Above 86 km
-----------
The standard stops using a single mean molecular weight above 86 km, because
diffusive separation and dissociation make the composition altitude-dependent.
Reproducing that machinery is not worth it here, so the tabulated values are
carried directly and density is interpolated logarithmically -- density falls
by roughly a decade per 15 km, so linear interpolation of the raw value would
be badly wrong between points.

Above 300 km density is returned as zero. It is below 1e-11 kg/m^3 there, some
nine orders of magnitude under the drag threshold that matters for a launch
vehicle, and this is not an orbit-lifetime tool.

The speed of sound above 86 km is reported from the tabulated kinetic
temperature. It is not physically meaningful in free-molecular flow, but the
trajectory code needs *some* Mach number to index the aero table, and a
continuous value is better than a discontinuity at the boundary.
"""

from __future__ import annotations

import numpy as np


# Layer bases below 86 km: (geopotential height [m], base temperature [K],
# lapse rate [K/m], base pressure [Pa]).
_LAYERS = (
    (0.0,     288.15, -0.0065,  101325.0),
    (11000.0, 216.65,  0.0,      22632.06),
    (20000.0, 216.65,  0.001,     5474.889),
    (32000.0, 228.65,  0.0028,     868.0187),
    (47000.0, 270.65,  0.0,        110.9063),
    (51000.0, 270.65, -0.0028,      66.93887),
    (71000.0, 214.65, -0.002,        3.956420),
)
_H_TOP = 84852.0          # geopotential altitude of the 86 km geometric top [m]
_P_TOP = 0.3733836        # pressure there [Pa]
_T_TOP = 186.87           # temperature there [K]

# US Standard Atmosphere 1976 upper table, geometric altitude [m],
# temperature [K], density [kg/m^3]. Used above 86 km.
_UPPER_Z = np.array([
    86e3, 90e3, 95e3, 100e3, 110e3, 120e3, 130e3, 140e3,
    150e3, 160e3, 180e3, 200e3, 250e3, 300e3,
])
_UPPER_T = np.array([
    186.87, 186.87, 188.42, 195.08, 240.00, 360.00, 469.27, 559.63,
    634.39, 696.29, 790.07, 854.56, 941.33, 976.01,
])
_UPPER_RHO = np.array([
    6.958e-6, 3.416e-6, 1.393e-6, 5.604e-7, 9.708e-8, 2.222e-8, 8.152e-9,
    3.831e-9, 2.076e-9, 1.233e-9, 5.194e-10, 2.541e-10, 6.073e-11, 1.916e-11,
])
_UPPER_LOG_RHO = np.log(_UPPER_RHO)


class Atmosphere:
    """US Standard Atmosphere 1976, valid to 300 km."""

    # Sea level constants
    SL_PRESSURE = 101325.0      # Pa
    SL_TEMP = 288.15            # K
    SL_DENSITY = 1.225          # kg/m^3
    G0 = 9.80665                # m/s^2
    R = 287.05                  # J/(kg-K), specific gas constant for air
    GAMMA = 1.4
    R_EARTH_EFFECTIVE = 6356766.0   # m, the standard's geopotential radius
    Z_MAX = 300000.0            # m, above which density is reported as zero

    def __init__(self):
        self.tropopause = 11000.0   # m
        self.stratopause = 47000.0  # m
        self.mesopause = 86000.0    # m

    # ------------------------------------------------------------------
    # Altitude conversions
    # ------------------------------------------------------------------

    def geopotential_altitude(self, z_geometric: float) -> float:
        """Convert geometric altitude [m] to geopotential altitude [m]."""
        r = self.R_EARTH_EFFECTIVE
        return r * z_geometric / (r + z_geometric)

    def geometric_altitude(self, h_geopotential: float) -> float:
        """Convert geopotential altitude [m] to geometric altitude [m]."""
        r = self.R_EARTH_EFFECTIVE
        return r * h_geopotential / (r - h_geopotential)

    # ------------------------------------------------------------------
    # Conditions
    # ------------------------------------------------------------------

    def get_conditions(self, altitude: float) -> tuple:
        """Return (density, pressure, temperature, speed_of_sound) at altitude.

        Args:
            altitude: Geometric altitude [m]. Negative values are treated as
                sea level.

        Returns:
            Tuple of (rho [kg/m^3], p [Pa], T [K], a [m/s]).
        """
        z = max(0.0, float(altitude))

        if z <= self.mesopause:
            T, p = self._lower_conditions(self.geopotential_altitude(z))
            rho = p / (self.R * T)
        else:
            T, rho, p = self._upper_conditions(z)

        a = np.sqrt(self.GAMMA * self.R * T) if T > 0 else 0.0
        return rho, p, T, a

    def _lower_conditions(self, h: float) -> tuple:
        """Temperature and pressure at geopotential altitude ``h`` [m]."""
        if h >= _H_TOP:
            return _T_TOP, _P_TOP

        # Walk down to the layer containing h. Seven layers, so a scan is
        # cheaper than anything cleverer.
        h_b, t_b, lapse, p_b = _LAYERS[0]
        for base in _LAYERS:
            if h >= base[0]:
                h_b, t_b, lapse, p_b = base
            else:
                break

        dh = h - h_b
        t = t_b + lapse * dh
        if lapse == 0.0:
            p = p_b * np.exp(-self.G0 * dh / (self.R * t_b))
        else:
            p = p_b * (t_b / t) ** (self.G0 / (self.R * lapse))
        return t, p

    def _upper_conditions(self, z: float) -> tuple:
        """Temperature, density and pressure above 86 km geometric."""
        if z >= self.Z_MAX:
            return float(_UPPER_T[-1]), 0.0, 0.0

        t = float(np.interp(z, _UPPER_Z, _UPPER_T))
        rho = float(np.exp(np.interp(z, _UPPER_Z, _UPPER_LOG_RHO)))
        # The ideal gas law still holds; only the composition assumption that
        # fixes R breaks down, and R varies by well under an order of magnitude
        # across this band while density varies by nine.
        p = rho * self.R * t
        return t, rho, p

    def density(self, altitude: float) -> float:
        """Density at geometric altitude [kg/m^3]."""
        return self.get_conditions(altitude)[0]

    def dynamic_viscosity(self, T: float) -> float:
        """Sutherland's formula for dynamic viscosity [Pa-s]."""
        return 1.458e-6 * T**1.5 / (T + 110.4)

    def mach(self, velocity: float, altitude: float) -> float:
        """Compute Mach number."""
        _, _, _, a = self.get_conditions(altitude)
        return velocity / a if a > 0 else 0.0
