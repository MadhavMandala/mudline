"""Launch rail geometry and the on-rail constraint.

Why this exists
---------------
``RocketSimulation.run`` accepted ``launch_azimuth`` and ``launch_elevation``
and used neither. Every flight started perfectly vertical, already moving at a
hardcoded 50 m/s, with the vehicle free to rotate from the first instant. That
is wrong in three separate ways:

* A launch cannot be aimed. Azimuth and elevation are the two parameters that
  matter most for where a rocket lands, and they did nothing.
* The 50 m/s head start is roughly 125 kJ of free kinetic energy on a 100 kg
  vehicle, invented to get the vehicle moving fast enough that the aero model's
  ``v_mag > 0.1`` guard would engage and the attitude would not tumble at t=0.
* Real vehicles are constrained by the rail until they clear it. That
  constraint is what makes the early trajectory insensitive to wind, and rail
  exit velocity is the number that decides whether a launch is safe: too slow
  and there is not enough airspeed for the fins to stabilise the vehicle before
  it is free.

Frame convention
----------------
The simulator's inertial frame is ENU-like: ``x`` East, ``y`` Up, ``z`` North,
matching ``WindModel.mean_wind``. Azimuth is measured from North toward East,
the meteorological/survey convention; elevation is measured up from horizontal,
so 90 degrees is vertical.

Tip-off
-------
A vehicle rides the rail on two buttons. While both are in the slot it can
only slide; when the forward one leaves the top, the vehicle is held at
the aft button alone and is free to pitch and yaw about it until that
button leaves too. Through that interval gravity's component across the
rail turns the nose down, the wind's normal force turns it wherever the
wind says, and the vehicle's own acceleration -- pushing a rod from its
tail -- amplifies whatever angle has opened. The rate it has when the aft
button leaves is the rate it flies off with, and on a slow or heavy
vehicle it is the largest attitude disturbance of the boost.

With ``buttons_m`` set the rail is modelled that way: the aft button
starts at the rail's foot, the phase is read from where the buttons are,
and the tip-off phase is integrated with the aft button constrained to
the rail's line -- the two lateral constraint forces and the roll
constraint torque solved for exactly at every evaluation, so the button
neither leaves the line nor turns in the slot. Without buttons the rail
is what it was: the CG constrained until it has travelled the length.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trajectory.eom import quat_to_dcm
from trajectory.frames import station_to_body


def rail_direction(azimuth_rad: float, elevation_rad: float) -> np.ndarray:
    """Unit vector along the rail in the inertial ENU frame [East, Up, North]."""
    ce = np.cos(elevation_rad)
    return np.array([
        np.sin(azimuth_rad) * ce,      # East
        np.sin(elevation_rad),         # Up
        np.cos(azimuth_rad) * ce,      # North
    ])


def alignment_quaternion(body_axis: np.ndarray, target_inertial: np.ndarray) -> np.ndarray:
    """Attitude quaternion [w, x, y, z] that points ``body_axis`` along ``target_inertial``.

    Used to aim the vehicle's thrust axis down the rail at ignition, rather than
    assuming the body axis and "up" coincide.

    Convention: ``quat_to_dcm`` is body-to-inertial, so the returned
    quaternion satisfies ``quat_to_dcm(q) @ body_axis == target`` -- the
    plain rotation taking ``a`` to ``b``. It used to return the conjugate to
    suit a transposed reading of the matrix, which was the same wrong reading
    the simulator made, so the two agreed with each other and disagreed with
    the quaternion kinematics. Getting this backwards points the thrust the
    wrong way, and the rail's at-rest floor then holds the vehicle on the
    pad forever.
    """
    a = np.asarray(body_axis, dtype=float)
    b = np.asarray(target_inertial, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if dot < -1.0 + 1e-12:
        # Antiparallel: any perpendicular axis is a valid 180-degree rotation,
        # and a half-turn is its own conjugate up to sign.
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            fallback = np.array([0.0, 0.0, 1.0])
        axis = np.cross(a, fallback)
        axis /= np.linalg.norm(axis)
        return np.array([0.0, *axis])

    axis = np.cross(a, b)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(dot)
    return np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])


@dataclass
class LaunchRail:
    """A straight rail the vehicle slides along before it flies freely.

    Args:
        azimuth_rad: Direction from North, positive toward East.
        elevation_rad: Angle up from horizontal; pi/2 is vertical.
        length_m: Distance travelled along the rail before the vehicle is free.
            Zero disables the constraint entirely.
        position_m: Inertial position of the rail's foot.
        buttons_m: Stations from the nose tip of the two rail buttons, in
            either order; ``None`` constrains the CG instead and has no
            tip-off phase.
    """

    azimuth_rad: float = 0.0
    elevation_rad: float = np.radians(89.0)
    length_m: float = 5.0
    position_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    buttons_m: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        self.position_m = np.asarray(self.position_m, dtype=float)
        self.direction = rail_direction(self.azimuth_rad, self.elevation_rad)
        if self.buttons_m is not None:
            forward, aft = sorted(float(s) for s in self.buttons_m)
            if aft - forward < 1e-6:
                raise ValueError("the two rail buttons must be at different stations")
            self.buttons_m = (forward, aft)

    # ------------------------------------------------------------ buttons

    @property
    def has_buttons(self) -> bool:
        return self.buttons_m is not None

    def button_bodies(self) -> tuple[np.ndarray, np.ndarray]:
        """The forward and aft buttons in the body frame, on the axis."""
        forward, aft = self.buttons_m
        return station_to_body(forward), station_to_body(aft)

    def travel_of(self, state: np.ndarray, r_body: np.ndarray,
                  cg_body_m: np.ndarray) -> float:
        """How far a body-fixed point has travelled up the rail [m].

        The state carries the CG; a body point is ``r_body - cg`` from it,
        since the body frame's origin is the nose tip and not the CG.
        """
        dcm = quat_to_dcm(state[6:10])
        arm = np.asarray(r_body, dtype=float) - np.asarray(cg_body_m, dtype=float)
        return self.distance_along(np.asarray(state[0:3], dtype=float) + dcm @ arm)

    def phase_of(self, state: np.ndarray, cg_body_m: np.ndarray | None = None) -> str:
        """``"rail"`` (sliding), ``"tipoff"`` (pivoting on the aft button) or ``"free"``."""
        if self.length_m <= 0.0:
            return "free"
        if self.buttons_m is None:
            return "rail" if self.distance_along(state[0:3]) < self.length_m else "free"
        forward, aft = self.button_bodies()
        if self.travel_of(state, aft, cg_body_m) >= self.length_m:
            return "free"
        if self.travel_of(state, forward, cg_body_m) >= self.length_m:
            return "tipoff"
        return "rail"

    def exit_measure(self, state: np.ndarray, cg_body_m: np.ndarray | None = None) -> float:
        """Zero at the instant the vehicle is free: the aft button, or the CG, at the top."""
        if self.buttons_m is None:
            return self.distance_along(state[0:3]) - self.length_m
        return self.travel_of(state, self.button_bodies()[1], cg_body_m) - self.length_m

    def forward_button_measure(self, state: np.ndarray, cg_body_m: np.ndarray) -> float:
        """Zero when the forward button reaches the top and the tip-off begins."""
        return self.travel_of(state, self.button_bodies()[0], cg_body_m) - self.length_m

    def start_offset_m(self, cg_body_m: np.ndarray, body_axis: np.ndarray) -> float:
        """How far up the rail the CG sits at ignition: the aft button is at the foot."""
        if self.buttons_m is None:
            return 0.0
        aft = self.button_bodies()[1]
        return float(np.dot(np.asarray(cg_body_m, dtype=float) - aft, body_axis))

    def distance_along(self, position_m: np.ndarray) -> float:
        """How far the vehicle has travelled up the rail [m]."""
        return float(np.dot(np.asarray(position_m, dtype=float) - self.position_m,
                            self.direction))

    def is_on_rail(self, position_m: np.ndarray) -> bool:
        if self.length_m <= 0.0:
            return False
        return self.distance_along(position_m) < self.length_m

    def constrain_acceleration(
        self, acceleration: np.ndarray, velocity: np.ndarray
    ) -> np.ndarray:
        """Project acceleration onto the rail while the vehicle is still on it.

        The rail carries any transverse load, so only the along-rail component
        survives. The along-rail component is additionally floored at zero while
        the vehicle is at rest: before thrust exceeds weight the rail and pad
        hold the vehicle up, and without this floor a vehicle whose motor has
        not yet built to full thrust would slide backwards down the rail and
        underground.
        """
        along = float(np.dot(acceleration, self.direction))
        speed_along = float(np.dot(velocity, self.direction))
        if along < 0.0 and speed_along <= 0.0:
            along = 0.0
        return along * self.direction

    def initial_quaternion(self, body_axis: np.ndarray) -> np.ndarray:
        """Attitude at ignition: the vehicle's body axis lies along the rail."""
        return alignment_quaternion(body_axis, self.direction)

    def exit_state(self, times: np.ndarray, states: np.ndarray, cg_of=None) -> dict | None:
        """Find the sample where the vehicle cleared the rail.

        Rail exit velocity is the standard go/no-go number for a launch: below
        roughly 15-20 m/s there is not enough airspeed for fins to stabilise the
        vehicle before the rail stops constraining it.

        This is the grid scan, good to one output step. The simulation
        finds the exit as an event root and uses this only as the fallback
        for a flight cut off before the root fired.
        """
        if self.length_m <= 0.0:
            return None
        distances = np.array([
            self.exit_measure(s, cg_of(s) if cg_of is not None else None) for s in states
        ])
        cleared = np.flatnonzero(distances >= 0.0)
        if len(cleared) == 0:
            return None
        idx = int(cleared[0])
        velocity = states[idx, 3:6]
        return {
            "time_s": float(times[idx]),
            "velocity_mps": float(np.linalg.norm(velocity)),
            "position_m": states[idx, 0:3].copy(),
        }
