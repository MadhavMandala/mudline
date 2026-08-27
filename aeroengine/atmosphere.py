"""RASAero II's atmosphere, reimplemented.

Piecewise curve fits of the 1962 US Standard Atmosphere in English units,
valid to 300,000 ft. Reproduced exactly, and validated against RASAero's own Run Test output
(``ab.cs`` in the reference notes).

The launch-site model is unusual and worth understanding before touching it.
Rather than perturbing the profiles, RASAero converts the site's temperature
and pressure into **three independent altitude offsets** and then evaluates
the standard curves at a *different shifted altitude for each property*:
density at ``h + dh_rho``, temperature at ``h + dh_T``, pressure at
``h + dh_P``. Kinematic viscosity rides on the density offset, not the
temperature one. There is no literature source for this construction; it is
RASAero's own, and reproducing it is required to match.

One consequence that matters for validation: RASAero's Tools > Run Test path
never applies the launch site at all. It leaves this object at its
constructor defaults -- every offset zero -- which is exactly standard sea
level. Every Run Test dump is therefore standard-day referenced no matter
what the CDX1's LaunchSite block says, and the Mach/Alt table is the only
route by which altitude enters a dump.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants, exactly as they appear in the original. Several are near-duplicates
# of each other (two different tropospheric lapse constants appear within four
# lines of one another in ab.cs); they are kept distinct on purpose.
# ---------------------------------------------------------------------------

RHO_SL = 0.002378          # slug/ft^3
T_SL_R = 518.69            # degR
P_SL_PSIA = 14.6958        # psia
GAMMA_R_G = 2403.07606     # gamma * R * g_c, for a = sqrt(GAMMA_R_G * T_R)
R_AIR = 1716.4829          # ft-lbf / (slug degR)
G0 = 32.174                # ft/s^2
EARTH_RADIUS_FT = 20_925_738

_INHG_TO_PSIA = 0.49117
_F_TO_R = 459.69

_LAPSE_PRESSURE = 6.87535e-6   # used for the site pressure-altitude inversion
_LAPSE_SITE = 6.8753e-6        # used for standard site pressure
_LAPSE_PROFILE = 6.8735e-6     # used for the pressure profile
_LAPSE_TEMP = 6.944e-6         # used for the temperature and density profiles


@dataclass(frozen=True)
class Atmosphere:
    """Standard atmosphere with RASAero's three-offset site model.

    The default instance is standard sea level with all offsets zero, which
    is the state Run Test evaluates in.
    """

    dh_density: float = 0.0      # ab.cs m_e
    dh_temperature: float = 0.0  # ab.cs m_f
    dh_pressure: float = 0.0     # ab.cs m_g

    # ------------------------------------------------------------------

    @classmethod
    def from_site(
        cls,
        elevation_ft: float = 0.0,
        pressure_inhg: float = 0.0,
        temperature_f: float | None = None,
    ) -> "Atmosphere":
        """Build the offsets from launch-site conditions (``ab.cs:44-88``).

        ``pressure_inhg = 0`` means "standard pressure for this elevation",
        not vacuum -- it is the RASAero UI's sentinel for "not entered".
        ``temperature_f = None`` likewise leaves the temperature offset at
        zero, matching a dialog field that was never filled in.
        """
        p_site = pressure_inhg * _INHG_TO_PSIA

        if p_site == 0.0:
            p_site = P_SL_PSIA * (1.0 - _LAPSE_SITE * elevation_ft) ** 5.2561
            # Not zero: with no pressure given the site's own elevation *is*
            # its pressure altitude, and the profile is read from there.
            dh_pressure = elevation_ft
        else:
            dh_pressure = elevation_ft + (
                (p_site / P_SL_PSIA) ** 0.19026 - 1.0
            ) / -_LAPSE_PRESSURE
            p_site = P_SL_PSIA * (1.0 - _LAPSE_PRESSURE * dh_pressure) ** 5.2561

        if temperature_f is None:
            return cls(0.0, 0.0, dh_pressure)

        t_site = temperature_f + _F_TO_R
        rho_site = p_site * 144.0 / (R_AIR * t_site)
        return cls(
            dh_density=(rho_site / RHO_SL - 1.0) * -40_000.0,
            dh_temperature=(t_site / T_SL_R - 1.0) * -144_000.0,
            dh_pressure=dh_pressure,
        )

    # ------------------------------------------------------------------

    def temperature_r(self, altitude_ft: float) -> float:
        h = self.dh_temperature + altitude_ft
        if h < 36_000:
            return T_SL_R * (1.0 - h * _LAPSE_TEMP)
        if h <= 65_800:
            return 389.97
        if h < 105_000:
            return 389.97 + 0.000543852 * (h - 65_800)
        if h < 155_500:
            return 411.289 + 0.001502594 * (h - 105_000)
        if h <= 172_000:
            return 487.17
        if h < 200_000:
            return 487.17 - 0.0010775 * (h - 172_000)
        if h < 262_500:
            return 457.0 - 0.00210928 * (h - 200_000)
        if h <= 295_000:
            return 325.17
        if h >= 300_000:
            return 332.9
        return 325.17 + 0.001546 * (h - 295_000)

    def density(self, altitude_ft: float) -> float:
        h = self.dh_density + altitude_ft
        if h <= 16_000:
            return RHO_SL * (1.0 - h * 2.5e-5)
        if h <= 36_000:
            return RHO_SL * (1.0 - h * _LAPSE_TEMP) ** 4.2561
        if h < 46_000:
            return RHO_SL * (0.3 - 1.1579e-5 * (h - 36_000))
        return 0.00046227 * math.exp(-4.7823e-5 * (h - 45_000))

    def pressure_psia(self, altitude_ft: float) -> float:
        h = self.dh_pressure + altitude_ft
        if h <= 36_089:
            return P_SL_PSIA * (1.0 - _LAPSE_PROFILE * h) ** 5.2561
        if h <= 82_021:
            return 3.2824244964 * math.exp(-4.80634e-5 * (h - 36_089.24))
        if h <= 160_000:
            return 3.2824244964 * math.exp(
                -4.80634e-5 * (h - 36_089.24 - 0.128239654 * (h - 82_021))
            )
        return 3.2824244964 * math.exp(-4.80634e-5 * (h - 46_098.24))

    def speed_of_sound(self, altitude_ft: float) -> float:
        return math.sqrt(GAMMA_R_G * self.temperature_r(altitude_ft))

    def kinematic_viscosity(self, altitude_ft: float) -> float:
        """ft^2/s. Rides on the *density* offset, not the temperature one."""
        h = self.dh_density + altitude_ft
        if h <= 29_500:
            return 0.00015723 + 5.7908e-9 * h
        if h < 80_000:
            return 8.2492e-5 * 10.0 ** (2.0265e-5 * h)
        if h < 146_000:
            return 7.4809e-5 * 10.0 ** (2.0971e-5 * h)
        return 0.00039234 * 10.0 ** (1.6057e-5 * h)

    def gravity(self, altitude_ft: float, site_elevation_ft: float = 0.0) -> float:
        r = EARTH_RADIUS_FT
        return G0 * (r / (r + altitude_ft + site_elevation_ft)) ** 2


#: The state Run Test evaluates in, and the reference for oracle comparison.
STANDARD = Atmosphere()


def reynolds(
    mach: float,
    length_in: float,
    altitude_ft: float = 0.0,
    atmosphere: Atmosphere = STANDARD,
) -> float:
    """Reynolds number on a reference length given in **inches** (i.cs:871)."""
    a = atmosphere.speed_of_sound(altitude_ft)
    nu = atmosphere.kinematic_viscosity(altitude_ft)
    return mach * a * (length_in / 12.0) / nu
