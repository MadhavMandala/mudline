"""The table-flown vehicle's pitch mode against the linearised short period.

Runs under pytest, and standalone via
``python parametric/tests/test_pitch_dynamics.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.flight import FlightSettings, configure_simulation  # noqa: E402
from parametric.standard import basic_rocket  # noqa: E402
from trajectory import simulation as tm  # noqa: E402
from trajectory.analysis.flightlog import FlightLog  # noqa: E402
from validation.rotational import fit_pitch_oscillation, linear_pitch_prediction  # noqa: E402


def _table(model):
    from parametric import aero

    table, _ = aero.run_analysis(model, aero.AeroSettings(
        mach_min=0.05, mach_max=1.5, mach_points=8, alpha_max_deg=8.0, alpha_points=3,
    ))
    return table


def test_the_coasting_pitch_mode_matches_the_linear_short_period(monkeypatch):
    """Coasting level at 150 m/s with gravity switched off, nudged three
    degrees: the frequency of the weathercock and its decay follow the
    two-degree-of-freedom short period -- the static margin's stiffness,
    the damping derivative, and the heave of the flight path under the
    normal force, which is worth as much as the derivative here. To two
    percent in frequency and ten in damping, measured from a log
    decrement over two cycles of a slowly decelerating vehicle."""
    monkeypatch.setattr(tm, "gravity_simple", lambda h: 0.0)
    model = basic_rocket()
    table = _table(model)
    sim = configure_simulation(model, FlightSettings(couple_aero_altitude=False), table)
    sim.launch_rail = None
    sim.rtol, sim.atol = 1e-9, 1e-11
    # Burnt out, so there is no thrust and no jet damping in the mode.
    from trajectory.vehicle.engine import Engine
    sim.engine = Engine(
        thrust_curve=np.array([0.0, 0.0]), time_points=np.array([0.0, 1.0]),
        isp_vac=250.0, isp_sl=250.0, nozzle_area=0.0, thrust_reference="vacuum",
    )
    state = np.zeros(tm.STATE_SIZE)
    state[1] = 1000.0
    state[5] = 150.0                                 # flying North
    # Axis along North, then pitched three degrees about body x.
    from trajectory.eom import propagate_quaternion
    from trajectory.sim.launch_rail import alignment_quaternion
    aim = alignment_quaternion(tm.BODY_AXIS, np.array([0.0, 0.0, 1.0]))
    state[6:10] = propagate_quaternion(aim, np.array([np.radians(3.0), 0.0, 0.0]), 1.0)
    state[tm.PROP_IDX] = 0.0
    result = sim._integrate_phase(state, 0.0, 2.0, 0.002, [])
    result.phases = []
    log = FlightLog.from_flight(sim, result)
    fit = fit_pitch_oscillation(log.time_s, log.alpha_deg)

    i = log.index_at(fit.peak_times[1])
    point = sim.evaluate(result.y[:, i], float(result.t[i]))
    omega_n, zeta = linear_pitch_prediction(point, sim.aero_model, table, point.inertia_kg_m2[0, 0])
    assert fit.natural_frequency_radps == pytest.approx(omega_n, rel=0.02), (fit, omega_n)
    assert fit.damping_ratio == pytest.approx(zeta, rel=0.10), (fit.damping_ratio, zeta)
    assert 0.05 < zeta < 0.3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
