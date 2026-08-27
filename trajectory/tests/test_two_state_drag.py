"""Two drag states: the base filled by the plume, and the base after burnout.

Base drag is a large share of the total on a blunt-based vehicle and it
largely disappears behind a plume, so the same rocket decelerates far harder
after burnout than during the burn. The table used to carry one column,
chosen once for the whole flight, so a coasting vehicle either flew with its
base still "filled" or a burning one flew with the full base drag.

Runs under pytest, and standalone via
``python trajectory/tests/test_two_state_drag.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.environment.gravity import gravity_simple  # noqa: E402
from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase  # noqa: E402
from trajectory.vehicle.aero_model import RasaeroAeroModel  # noqa: E402

BODY_AXIS = np.array([0.0, 1.0, 0.0])
CD_OFF, CD_ON = 0.5, 0.3


def _database() -> AeroDatabase:
    rows = [
        AeroCoefficients(mach=m, alpha_deg=a, cd=CD_OFF, cn=0.09 * a, cm=0.0,
                         x_cp_m=2.0, cd_power_on=CD_ON)
        for m in (0.0, 1.0, 3.0) for a in (0.0, 5.0, 10.0)
    ]
    return AeroDatabase(rows, reference_length_m=3.0)


def _model() -> RasaeroAeroModel:
    return RasaeroAeroModel(
        _database(), reference_area_m2=0.07, reference_length_m=3.0,
        body_axis=BODY_AXIS,
    )


def test_the_model_uses_the_burning_column_only_when_told():
    flow = np.array([0.0, 200.0, 0.0])
    cg = np.array([0.0, -1.5, 0.0])
    off = _model().forces_and_moments(flow, 1.0, 340.0, cg)
    on = _model().forces_and_moments(flow, 1.0, 340.0, cg, power_on=True)
    ratio = np.linalg.norm(on.force_body_n) / np.linalg.norm(off.force_body_n)
    assert np.isclose(ratio, CD_ON / CD_OFF)


def test_a_single_column_table_is_used_as_is_either_way():
    rows = [
        AeroCoefficients(mach=m, alpha_deg=a, cd=CD_OFF, cn=0.09 * a, cm=0.0, x_cp_m=2.0)
        for m in (0.0, 1.0, 3.0) for a in (0.0, 5.0, 10.0)
    ]
    model = RasaeroAeroModel(
        AeroDatabase(rows, reference_length_m=3.0),
        reference_area_m2=0.07, reference_length_m=3.0, body_axis=BODY_AXIS,
    )
    flow = np.array([0.0, 200.0, 0.0])
    cg = np.array([0.0, -1.5, 0.0])
    off = model.forces_and_moments(flow, 1.0, 340.0, cg)
    on = model.forces_and_moments(flow, 1.0, 340.0, cg, power_on=True)
    assert np.allclose(on.force_body_n, off.force_body_n)


def test_the_simulation_switches_columns_at_burnout():
    """Burning: power-on drag. Tank empty: power-off. Nothing else differs."""
    from trajectory import simulation as tm

    sim = tm.RocketSimulation()
    sim.set_aero_database(_database())

    altitude, speed = 1000.0, 100.0
    state = np.zeros(tm.STATE_SIZE)
    state[1] = altitude
    state[4] = speed                       # straight up, along body +Y
    state[6] = 1.0                         # identity attitude: body +Y is up
    g = gravity_simple(altitude)
    pressure = sim.atm.get_conditions(altitude)[1]

    burning = state.copy()
    burning[tm.PROP_IDX] = sim.mass_props.prop_mass
    forces_on, _, mass_on, *_ = sim.compute_forces_moments(burning, 5.0)
    thrust, _ = sim.engine.thrust_at(5.0, pressure)
    assert thrust > 0.0
    aero_on = forces_on - np.array([0.0, -g * mass_on, 0.0]) - np.array([0.0, thrust, 0.0])

    empty = state.copy()
    empty[tm.PROP_IDX] = 0.0
    forces_off, _, mass_off, *_ = sim.compute_forces_moments(empty, 5.0)
    aero_off = forces_off - np.array([0.0, -g * mass_off, 0.0])

    assert aero_on[1] < 0.0 and aero_off[1] < 0.0, "drag opposes the climb"
    assert np.isclose(aero_on[1] / aero_off[1], CD_ON / CD_OFF, rtol=1e-6)


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
