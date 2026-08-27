"""A point-mass trajectory, matched in complexity to what it is measuring.

The scoreboard exists to score the *aerodynamics* against real flights, so the
flight model around it is deliberately the simplest one that does not itself
become the error term: a two-degree-of-freedom point mass in the vertical
plane, thrust and drag along the velocity vector once the rocket is off the
rod, gravity falling off with altitude. That is the same model RASAero flies,
which is what makes the comparison fair -- a six-degree-of-freedom simulation
here would fold its own attitude dynamics into a number meant to isolate drag.

Two conventions carry real weight.

*Propellant is burned by impulse, not by time.* See ``motors.Motor``.

*Drag switches at burnout.* The base of a flying rocket is a separated wake
whose pressure depends on whether a jet is filling it, so the buildup produces
two numbers -- power-on and power-off -- differing by the nozzle-to-base area
credit. Using either one for the whole flight is worth several percent of
apogee on a high-thrust vehicle, and gets worse the larger the nozzle.

The altitude the aerodynamics is evaluated at is left to the caller, through
``cd_off``/``cd_on``. That is the whole point of ``mode`` in ``scoreboard``:
RASAero freezes the drag table at whatever altitudes its Mach/Alt grid names
-- sea level when the grid is empty, which is the default -- and then flies
that frozen table by Mach alone. Passing a table built along the trajectory's
own altitude history instead is the fix, and the difference between the two
runs is what that defect is worth in feet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from aeroengine.atmosphere import G0, Atmosphere

__all__ = ["FlightResult", "fly"]


@dataclass
class FlightResult:
    """What the vehicle did, in the units the flight cards use."""

    apogee_ft: float = 0.0            # above the launch site
    time_to_apogee_s: float = 0.0
    max_velocity_fps: float = 0.0
    max_mach: float = 0.0
    burnout_ft: float = 0.0
    burnout_velocity_fps: float = 0.0
    burnout_mach: float = 0.0
    #: (Mach, altitude) along the ascent, for re-tabulating the drag.
    mach_alt: list[tuple[float, float]] = field(default_factory=list)
    steps: int = 0


def fly(
    card,
    motor,
    a_ref_in2: float,
    cd_off: Callable[[float], float],
    cd_on: Callable[[float], float],
    atmos: Atmosphere,
    *,
    dt: float = 0.005,
    max_time: float = 600.0,
) -> FlightResult:
    """Integrate to apogee with RK4 and return the summary.

    ``a_ref_in2`` is the engine's own reference area, not a recomputed one --
    the drag coefficients are normalised by it, so any other value silently
    rescales every force.
    """
    a_ref_ft2 = a_ref_in2 / 144.0
    site = card.site_altitude_ft
    delay = card.ignition_delay_s
    rod_rad = math.radians(card.rod_angle_deg)
    rod_dir = (math.sin(rod_rad), math.cos(rod_rad))

    def derivs(t: float, s: tuple[float, float, float, float]):
        x, h, vx, vh = s
        h_eval = max(h, 0.0)
        speed = math.hypot(vx, vh)
        rho = atmos.density(h_eval)
        sound = atmos.speed_of_sound(h_eval)
        mach = speed / sound if sound > 0.0 else 0.0

        te = t - delay
        thrust = motor.thrust_at(te)
        weight = card.launch_weight_lb - motor.propellant_burned(te)
        mass = max(weight, 1e-6) / G0

        # On the rod the vehicle is constrained to the rail; after that it
        # weathercocks onto the relative wind, so body axis == velocity.
        travelled = math.hypot(x, h - 0.0)
        if travelled < card.rod_length_ft or speed < 1e-6:
            ux, uy = rod_dir
        else:
            ux, uy = vx / speed, vh / speed

        cd = cd_on(mach) if thrust > 0.0 else cd_off(mach)
        drag = 0.5 * rho * speed * speed * a_ref_ft2 * cd
        axial = thrust - drag

        g = atmos.gravity(h_eval, site)
        return (vx, vh, axial * ux / mass, axial * uy / mass - g)

    state = (0.0, 0.0, 0.0, 0.0)
    t = 0.0
    out = FlightResult()
    burnout_seen = False
    thrusting = False
    last_sample = -1.0

    while t < max_time:
        k1 = derivs(t, state)
        k2 = derivs(t + dt / 2, tuple(state[i] + dt / 2 * k1[i] for i in range(4)))
        k3 = derivs(t + dt / 2, tuple(state[i] + dt / 2 * k2[i] for i in range(4)))
        k4 = derivs(t + dt, tuple(state[i] + dt * k3[i] for i in range(4)))
        new = tuple(
            state[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i])
            for i in range(4)
        )

        t += dt
        out.steps += 1
        x, h, vx, vh = new
        speed = math.hypot(vx, vh)
        sound = atmos.speed_of_sound(max(h, 0.0))
        mach = speed / sound if sound > 0.0 else 0.0

        if speed > out.max_velocity_fps:
            out.max_velocity_fps = speed
        if mach > out.max_mach:
            out.max_mach = mach

        te = t - delay
        if te >= motor.times[0]:
            thrusting = True
        if not burnout_seen and thrusting and motor.thrust_at(te) <= 0.0:
            burnout_seen = True
            out.burnout_ft = h
            out.burnout_velocity_fps = speed
            out.burnout_mach = mach

        # A coarse ascent sample is enough to re-tabulate drag against; the
        # table is interpolated by Mach and the profile is monotonic in it
        # until well after burnout.
        if h > last_sample + 250.0:
            out.mach_alt.append((mach, h))
            last_sample = h

        # Apogee: interpolate the zero crossing of vertical velocity rather
        # than taking the last step above it, which would quantise apogee to
        # whatever dt happens to be.
        if new[3] <= 0.0 < state[3]:
            frac = state[3] / (state[3] - new[3])
            out.apogee_ft = state[1] + frac * (new[1] - state[1])
            out.time_to_apogee_s = t - dt + frac * dt
            state = new
            break
        if t > 1.0 and h < 0.0:
            out.apogee_ft = max(out.apogee_ft, state[1])
            break

        state = new

    if out.apogee_ft == 0.0:
        out.apogee_ft = state[1]
    return out
