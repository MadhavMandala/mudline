"""Wind: a surface profile, winds aloft, a discrete gust, and turbulence.

The surface wind is a speed and a bearing at the 10 m reference height,
grown from the ground by a power law or the log law (the power law follows
the log law below the reference height). Winds aloft are a sounding: a
table of altitude, speed and bearing above the surface layer, interpolated
as vectors so that a veer between two levels turns the wind through the
shorter way rather than swinging it through zero speed. Above the top
level the wind holds.

Turbulence is the Dryden model of MIL-F-8785C, generated once as a frozen
field along altitude rather than filtered in time. A rocket climbs
through the field at a speed that changes by two orders of magnitude
during the flight, so a temporal filter would have to change its own
time constant with the airspeed at every step; the spatial form with
Taylor's frozen-turbulence hypothesis has one shape and the vehicle
samples it wherever it is. The scale lengths and intensities are the
specification's: below 300 m the lengths grow with height and the
intensity follows the wind at 20 ft, above 600 m the lengths are 533 m and
the intensity is the chosen level's, blended between. The first-order
longitudinal spectral form is used for all three components -- the
lateral form's extra numerator term changes the spectrum's shape, not its
energy or its correlation length, which are what matter here. The field is
deterministic in its seed, so a flight is reproducible and a dispersion
study draws a seed per case.
"""

from __future__ import annotations

import numpy as np

#: Root-mean-square gust [m/s] above the surface layer for each named
#: level -- light, moderate and severe in MIL-F-8785C's terms -- and the
#: 20 ft wind each is associated with there [m/s], which sets the
#: turbulence near the ground when the actual surface wind is lighter.
TURBULENCE_LEVELS: dict[str, tuple[float, float]] = {
    "none": (0.0, 0.0),
    "light": (1.5, 7.7),
    "moderate": (3.0, 15.4),
    "severe": (6.0, 23.1),
}
FT_PER_M = 1.0 / 0.3048


class DrydenTurbulence:
    """Continuous turbulence as a frozen field in altitude.

    ``sigma_mps`` is the intensity above the surface layer; ``reference_wind_mps``
    floors the 20 ft wind the surface-layer intensity is taken from.
    """

    #: Below this height the surface-layer scales apply, above ``HIGH_M``
    #: the high-altitude ones; between them a linear blend.
    LOW_M = 300.0
    HIGH_M = 600.0
    HIGH_LENGTH_M = 533.4          # 1750 ft
    MIN_LENGTH_M = 5.0
    STEP_M = 2.0

    def __init__(self, sigma_mps: float, seed: int = 0,
                 reference_wind_mps: float = 0.0, top_m: float = 120_000.0):
        self.sigma_mps = float(sigma_mps)
        self.seed = int(seed)
        self.reference_wind_mps = float(reference_wind_mps)
        self.top_m = float(top_m)
        self._heights: np.ndarray | None = None
        self._unit: np.ndarray | None = None

    @classmethod
    def from_level(cls, level: str, seed: int = 0) -> "DrydenTurbulence | None":
        """A named level, or ``None`` for "none"."""
        sigma, reference = TURBULENCE_LEVELS[level]
        if sigma <= 0.0:
            return None
        return cls(sigma, seed=seed, reference_wind_mps=reference)

    def reseed(self, seed: int) -> None:
        self.seed = int(seed)
        self._heights = self._unit = None

    # ------------------------------------------------------------ scales

    def scales(self, height_m: float, wind_20ft_mps: float) -> tuple:
        """``(L_u, L_v, L_w, s_u, s_v, s_w)`` at this height [m, m/s].

        MIL-F-8785C: in the surface layer ``L_w = h``,
        ``L_u = L_v = h / (0.177 + 0.000823 h)^1.2`` (h in feet),
        ``s_w = 0.1 W20``, ``s_u = s_v = s_w / (0.177 + 0.000823 h)^0.4``;
        aloft every length is 1750 ft and every intensity the level's.
        """
        h = max(float(height_m), 1.0)
        w20 = max(float(wind_20ft_mps), self.reference_wind_mps)
        h_ft = h * FT_PER_M
        f = 0.177 + 0.000823 * min(h_ft, self.LOW_M * FT_PER_M)
        low = (
            max(h / f ** 1.2, self.MIN_LENGTH_M), max(h / f ** 1.2, self.MIN_LENGTH_M),
            max(h, self.MIN_LENGTH_M),
            0.1 * w20 / f ** 0.4, 0.1 * w20 / f ** 0.4, 0.1 * w20,
        )
        high = (self.HIGH_LENGTH_M,) * 3 + (self.sigma_mps,) * 3
        if h <= self.LOW_M:
            return low
        if h >= self.HIGH_M:
            return high
        t = (h - self.LOW_M) / (self.HIGH_M - self.LOW_M)
        return tuple((1.0 - t) * a + t * b for a, b in zip(low, high))

    # ------------------------------------------------------------ the field

    def _generate(self, wind_20ft_mps: float) -> None:
        """Three unit-variance first-order fields along altitude.

        Each step ``x[i] = a x[i-1] + sqrt(1 - a^2) n`` with ``a = exp(-dh / L)``
        keeps the variance at one whatever ``L`` does, so the intensity can
        be applied afterwards as a function of height.
        """
        rng = np.random.default_rng(self.seed)
        heights = np.arange(0.0, self.top_m + self.STEP_M, self.STEP_M)
        n = len(heights)
        noise = rng.standard_normal((n, 3))
        unit = np.zeros((n, 3))
        lengths = np.array([self.scales(h, wind_20ft_mps)[:3] for h in heights])
        a = np.exp(-self.STEP_M / lengths)
        b = np.sqrt(1.0 - a * a)
        unit[0] = noise[0]
        for i in range(1, n):
            unit[i] = a[i] * unit[i - 1] + b[i] * noise[i]
        self._heights, self._unit = heights, unit

    def gust(self, height_m: float, wind_20ft_mps: float,
             along_enu: np.ndarray) -> np.ndarray:
        """The gust at this height in ENU [East, Up, North].

        ``along_enu`` is the horizontal unit vector the mean wind blows
        toward; the longitudinal component lies along it, the lateral one
        across, the vertical one up.
        """
        if self.sigma_mps <= 0.0:
            return np.zeros(3)
        if self._unit is None:
            self._generate(wind_20ft_mps)
        h = min(max(float(height_m), 0.0), float(self._heights[-1]))
        unit = np.array([np.interp(h, self._heights, self._unit[:, k]) for k in range(3)])
        _, _, _, s_u, s_v, s_w = self.scales(h, wind_20ft_mps)
        along = np.asarray(along_enu, dtype=float)
        across = np.array([-along[2], 0.0, along[0]])          # rotate the horizontal by 90 deg
        return s_u * unit[0] * along + s_v * unit[1] * across + s_w * unit[2] * np.array([0.0, 1.0, 0.0])


class WindModel:
    """Atmospheric wind: surface profile, winds aloft, gust and turbulence."""

    def __init__(self, surface_wind: np.ndarray = None,
                 surface_dir: float = 0.0,     # From N [rad]
                 profile_type: str = "power_law",
                 aloft: list | None = None,
                 turbulence: DrydenTurbulence | None = None):
        """
        Args:
            surface_wind: [u, v] components [m/s], or None for calm
            surface_dir: Wind direction (meteorological: from direction)
            profile_type: "power_law" or "log"
            aloft: Winds aloft as ``(altitude_m, speed_mps, from_deg)``
                rows -- a sounding, so the bearing is in degrees as it is
                written. Levels at or below the reference height are
                ignored; the surface profile rules there.
            turbulence: A ``DrydenTurbulence``, or ``None`` for smooth air.
        """
        if surface_wind is None:
            surface_wind = np.array([0.0, 0.0])
        self.surface_wind = np.asarray(surface_wind, dtype=float)
        self.surface_dir = surface_dir
        self.profile_type = profile_type
        self.z_ref = 10.0   # Reference height [m]
        self.alpha = 0.14    # Power law exponent (typical for neutral stability)
        self.turbulence = turbulence
        self.aloft: list[tuple[float, float, float]] = []
        if aloft:
            self.aloft = sorted(
                (float(h), float(v), float(b)) for h, v, b in aloft if float(h) > self.z_ref
            )

        # Discrete gust parameters
        self.gust_active = False
        self.gust_amp = 0.0
        self.gust_freq = 0.0
        self.gust_phase = 0.0

    @staticmethod
    def toward(from_deg: float) -> np.ndarray:
        """Unit vector a wind from this bearing blows toward, in ENU."""
        bearing = np.radians(float(from_deg))
        return np.array([-np.sin(bearing), 0.0, -np.cos(bearing)])

    def surface_speed(self) -> float:
        return float(np.linalg.norm(self.surface_wind))

    def _surface_factor(self, height: float) -> float:
        """The profile's multiplier on the surface value at this height."""
        if self.profile_type == "power_law" and height >= self.z_ref:
            return (height / self.z_ref) ** self.alpha
        # Logarithmic profile -- and the power law's own behaviour below
        # the reference height. The power law is a fit to this above
        # z_ref; below it it used to be floored at the full surface
        # value, so a rail at 3 m saw the 10 m wind, which the log law
        # puts at 77% of it. Both give exactly the surface value at
        # z_ref, so the join is seamless.
        #
        # Both guards below are load-bearing: the profile is only
        # defined above the roughness length z0. At exactly ground level
        # log(0) is -inf, which propagates NaN through every force in the
        # simulation; between 0 and z0 the logarithm is negative, which
        # would reverse the wind direction near the pad. Clamping to z0
        # gives the physically right answer -- zero wind at the surface,
        # rising from there.
        z0 = 0.05  # Roughness length
        factor = np.log(max(height, z0) / z0) / np.log(self.z_ref / z0)
        return max(0.0, float(factor))

    def mean_wind(self, altitude: float) -> np.ndarray:
        """Mean wind vector in inertial (ENU) frame at this height above ground."""
        height = max(0.0, float(altitude))
        surface = self.surface_speed()

        if not self.aloft or height <= self.z_ref:
            if surface == 0.0:
                return np.zeros(3)
            return surface * self._surface_factor(height) * self.blows_toward()

        # Winds aloft: the surface value at the reference height, then the
        # sounding, as vectors. Above the top level the wind holds.
        heights = [self.z_ref] + [h for h, _, _ in self.aloft]
        vectors = [surface * self.blows_toward()] + [
            speed * self.toward(bearing) for _, speed, bearing in self.aloft
        ]
        vectors = np.array(vectors)
        return np.array([np.interp(height, heights, vectors[:, k]) for k in range(3)])

    def blows_toward(self) -> np.ndarray:
        """Unit vector the wind blows *toward*, in ENU [East, Up, North].

        ``surface_dir`` is meteorological -- the bearing the wind comes
        *from*, clockwise from North -- so a wind from the north (0) blows
        south and a wind from the east (90 deg) blows west. The conversion
        used to drop both minus signs, so every wind blew toward the bearing
        it was said to come from: 180 degrees wrong, in the landing ellipse
        and in every weathercock.
        """
        return np.array([-np.sin(self.surface_dir), 0.0, -np.cos(self.surface_dir)])

    def gust(self, t: float) -> np.ndarray:
        """Sinusoidal gust along the wind's bearing."""
        if not self.gust_active:
            return np.array([0.0, 0.0, 0.0])

        # Along the wind, not along East: a gust is the wind blowing harder
        # for a moment, whichever way it was blowing.
        amplitude = self.gust_amp * np.sin(
            2 * np.pi * self.gust_freq * t + self.gust_phase
        )
        return amplitude * self.blows_toward()

    def turbulent_gust(self, altitude: float) -> np.ndarray:
        """The turbulence field's gust at this height above ground, or zero."""
        if self.turbulence is None:
            return np.zeros(3)
        # The 20 ft wind the surface-layer intensity follows.
        w20 = float(np.linalg.norm(self.mean_wind(20.0 / FT_PER_M)))
        along = self.blows_toward() if self.surface_speed() > 0.0 else np.array([0.0, 0.0, -1.0])
        return self.turbulence.gust(altitude, w20, along)

    def total_wind(self, altitude: float, t: float) -> np.ndarray:
        """Total wind vector at this height above ground."""
        return self.mean_wind(altitude) + self.gust(t) + self.turbulent_gust(altitude)

    def set_gust(self, amplitude: float, frequency: float, phase: float = 0.0):
        """Configure a discrete sinusoidal gust."""
        self.gust_active = True
        self.gust_amp = amplitude
        self.gust_freq = frequency
        self.gust_phase = phase

    def clear_gust(self):
        """Disable gust model."""
        self.gust_active = False
