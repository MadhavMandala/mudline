"""Main simulation integrator using solve_ivp."""

import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional


class TrajectoryIntegrator:
    """
    State vector integrator for 6-DOF trajectory.

    State: [x, y, z, vx, vy, vz, q_w, q_x, q_y, q_z, p, q, r]
           0  1  2   3  4   5   6    7    8    9   10 11 12
    """

    #: The solver's relative and absolute tolerances. SciPy's own defaults
    #: (1e-3, 1e-6) are loose for a flight: the attitude states are unit
    #: quaternion components and rates of a fraction of a radian per second,
    #: and a relative tolerance of a thousandth on those is a tenth of a
    #: degree per step. These were never settable and never set; the
    #: simulation ran at the library default throughout.
    DEFAULT_RTOL = 1e-6
    DEFAULT_ATOL = 1e-8

    def __init__(self, derivative_fn: Callable, events: list = None,
                 rtol: float | None = None, atol: float | None = None,
                 max_step: float = 0.5):
        """
        Args:
            derivative_fn: Function(state, t) -> state_derivative
            events: List of event functions for termination/detection
            rtol, atol: Solver tolerances; ``None`` takes the defaults above.
            max_step: Largest step the solver may take [s], so a quiet coast
                cannot stride over a deployment trigger.
        """
        self.derivative_fn = derivative_fn
        self.events = events or []
        self.stop_on_event = True
        self.rtol = float(self.DEFAULT_RTOL if rtol is None else rtol)
        self.atol = float(self.DEFAULT_ATOL if atol is None else atol)
        self.max_step = float(max_step)

    def integrate(self, state0: np.ndarray, t_span: tuple,
                  dt: float = 0.1, method: str = "RK45") -> dict:
        """
        Run integration.

        Args:
            state0: Initial state vector (13 elements)
            t_span: (t0, tf) time range
            dt: Maximum output time step
            method: Integration method (RK45, RK23, DOP853, etc.)

        Returns:
            SciPy OdeResult with added 'trajectory' key containing
            full state history
        """
        # arange in floats can land its last point a rounding error past
        # the stop value, which solve_ivp rejects as outside t_span. Clip.
        t_eval = np.arange(t_span[0], t_span[1], dt)
        t_eval = t_eval[(t_eval >= t_span[0]) & (t_eval <= t_span[1])]

        sol = solve_ivp(
            self.derivative_fn,
            t_span,
            state0,
            method=method,
            t_eval=t_eval,
            events=self.events,
            dense_output=True,
            max_step=self.max_step,
            rtol=self.rtol,
            atol=self.atol,
        )

        return sol

    @staticmethod
    def state_to_dict(t: float, state: np.ndarray) -> dict:
        """Convert raw state vector to structured dict."""
        return {
            "t": t,
            "pos": state[0:3],      # [x, y, z]
            "vel": state[3:6],      # [vx, vy, vz]
            "quat": state[6:10],    # [w, x, y, z] - normalized
            "omega": state[10:13]   # [p, q, r] body rates
        }
