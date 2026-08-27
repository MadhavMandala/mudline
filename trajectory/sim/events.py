"""Event detection functions for trajectory monitoring."""

import numpy as np


class EventDetector:
    """Collection of common trajectory event functions."""

    @staticmethod
    def apogee(arm_altitude_m: float = 10.0, ground_m: float = 0.0):
        """Event triggered at vertical velocity = 0 (apogee).

        Armed only once the vehicle has climbed ``arm_altitude_m`` above the
        ground, which is ``ground_m`` above sea level.

        Without that guard the event fires at t = 0: a vehicle sitting on the
        rail has exactly zero vertical velocity at ignition, which is a
        legitimate root of vy = 0. The consequence is not subtle -- the ascent
        phase terminates immediately and the entire flight is then integrated
        with the drogue deployed. This was invisible while the simulator
        started every flight at a hardcoded 50 m/s.

        The arming height is above the *ground* for the same reason. Measured
        from sea level, a pad at 2,000 m sat above it already, the event was
        armed at ignition and fired on the pad, and the flight went under
        the drogue with the motor still burning.

        The guard returns a positive constant below the arming altitude. That
        is a discontinuity, but not one the root finder can trip over: the
        switch happens while the vehicle is climbing, so the event value goes
        from +1 to a positive vy without crossing zero.
        """
        def event(t, state):
            if state[1] - ground_m < arm_altitude_m:
                return 1.0
            return state[4]  # vy = 0
        event.terminal = True
        event.direction = -1  # Trigger when going down
        return event

    @staticmethod
    def ground(altitude_idx: int = 1, tolerance: float = 0.1,
               ground_m: float = 0.0, arm_m: float = 1.0):
        """Event triggered at impact with the ground, exactly at the pad.

        ``ground_m`` is the altitude the flight started at. The ground is
        wherever the rail foot stood, not the inertial origin: a pad placed
        at 250 m must not have its vehicle fall 250 m past it.

        Armed by the state: within ``arm_m`` of the pad and not descending
        -- at rest before liftoff, or climbing out -- the event returns a
        positive constant, so a vehicle sitting on the pad is not a root at
        t = 0. Descending, it is the height itself, and the root is the
        ground. A 0.1 m offset used to do the arming instead, at the cost of
        every flight landing 0.1 m under the pad; ``tolerance`` is kept for
        callers that pass it and no longer moves the root.
        """
        def event(t, state):
            height = state[altitude_idx] - ground_m
            if height < arm_m and state[4] >= 0.0:
                return arm_m
            return height
        event.terminal = True
        event.direction = -1
        return event

    @staticmethod
    def rail_exit(rail, cg_of=None):
        """Non-terminal event at the instant the vehicle clears the rail.

        The exit used to be found by scanning the output grid for the first
        sample past the rail's length, so the exit speed -- the go/no-go
        number for a launch -- was accurate only to one output step. A root
        is exact, and the state at the root is what the off-the-rail angle
        of attack is read from.

        ``cg_of(state)`` gives the CG in the body frame, which a rail with
        buttons needs to place them from the state.
        """
        def event(t, state):
            return rail.exit_measure(state, cg_of(state) if cg_of is not None else None)
        event.terminal = False
        event.direction = 1
        return event

    @staticmethod
    def forward_button_exit(rail, cg_of):
        """Non-terminal event at the instant the forward button leaves the rail.

        The start of the tip-off phase, recorded so its duration and the
        rate it produced can be quoted.
        """
        def event(t, state):
            return rail.forward_button_measure(state, cg_of(state))
        event.terminal = False
        event.direction = 1
        return event

    @staticmethod
    def descending_through(altitude_m: float):
        """Event triggered when descending past a given altitude.

        The main-parachute trigger. Directional, so a vehicle still climbing
        through this altitude on the way up does not fire it.
        """
        def event(t, state):
            return state[1] - altitude_m
        event.terminal = True
        event.direction = -1
        return event

    @staticmethod
    def max_altitude(max_alt: float):
        """Event triggered when exceeding max altitude."""
        def event(t, state):
            return max_alt - state[1]
        event.terminal = True
        event.direction = -1
        return event

    @staticmethod
    def max_range(max_range: float):
        """Event triggered when exceeding max downrange."""
        def event(t, state):
            return np.linalg.norm(state[[0, 2]]) - max_range
        event.terminal = True
        event.direction = 1
        return event

    @staticmethod
    def burnout(thrust_func: callable, threshold: float = 1.0):
        """Event triggered when thrust drops below threshold."""
        def event(t, state):
            return thrust_func(t) - threshold
        event.terminal = False  # Don't stop, just detect
        event.direction = -1
        return event


# Convenience instances
apogee_event = EventDetector.apogee()
ground_impact = EventDetector.ground()
