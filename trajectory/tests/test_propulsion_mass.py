"""Tests for propellant-as-state, thrust reference, and altitude-invariant mdot.

Each test names the specific defect it pins down. Runs under pytest, and
standalone via ``python trajectory/tests/test_propulsion_mass.py``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.propulsion.model import (  # noqa: E402
    PropulsionModel,
    default_propulsion_model,
)
from trajectory.vehicle.engine import (  # noqa: E402
    G0,
    P_STD,
    Engine,
    nozzle_area_for_isp_pair,
)
from trajectory.vehicle.mass_properties import MassProperties  # noqa: E402
from trajectory.simulation import PROP_IDX, STATE_SIZE, RocketSimulation  # noqa: E402


def make_engine(**kwargs) -> Engine:
    params = dict(
        thrust_curve=np.array([20000.0, 20000.0, 0.0, 0.0]),
        time_points=np.array([0.0, 30.0, 31.0, 120.0]),
        isp_vac=280.0,
        isp_sl=250.0,
        nozzle_area=nozzle_area_for_isp_pair(20000.0, 280.0, 250.0),
    )
    params.update(kwargs)
    return Engine(**params)


# --------------------------------------------------------------------------
# Mass flow no longer depends on altitude
# --------------------------------------------------------------------------


def test_mass_flow_is_altitude_invariant():
    """Was 1.958 kg/s at sea level vs 7.279 in vacuum -- a 3.7x swing."""
    eng = make_engine()
    flows = [eng.thrust_at(10.0, p)[1] for p in (0.0, 100.0, 50_000.0, P_STD)]
    assert np.allclose(flows, flows[0], rtol=1e-12), flows

    expected = 20000.0 / (280.0 * G0)
    assert np.isclose(flows[0], expected, rtol=1e-12)


def test_mass_flow_never_negative():
    """After burnout the pressure term used to drive mdot to -6.199 kg/s."""
    eng = make_engine()
    for t in (31.0, 32.0, 60.0, 100.0, 500.0):
        thrust, mdot = eng.thrust_at(t, P_STD)
        assert mdot >= 0.0, (t, mdot)
        assert thrust >= 0.0, (t, thrust)


def test_mass_flow_is_zero_after_burnout():
    eng = make_engine()
    assert eng.thrust_at(60.0, P_STD)[1] == 0.0
    assert eng.thrust_at(60.0, 0.0)[1] == 0.0


# --------------------------------------------------------------------------
# Thrust reference convention
# --------------------------------------------------------------------------


def test_vacuum_reference_loses_only_pressure_term_at_sea_level():
    eng = make_engine(thrust_reference="vacuum")
    thrust, _ = eng.thrust_at(10.0, P_STD)
    assert np.isclose(thrust, 20000.0 - P_STD * eng.nozzle_area)


def test_sea_level_reference_returns_its_own_curve_at_sea_level():
    """A sea-level curve must not have the ambient term subtracted twice."""
    area = nozzle_area_for_isp_pair(20000.0, 280.0, 250.0)
    eng = make_engine(nozzle_area=area, thrust_reference="sea_level")
    thrust, _ = eng.thrust_at(10.0, P_STD)
    assert np.isclose(thrust, 20000.0, rtol=1e-9)

    # ...and gains the full pressure term in vacuum.
    vac, _ = eng.thrust_at(10.0, 0.0)
    assert np.isclose(vac, 20000.0 + P_STD * area, rtol=1e-9)


def test_unknown_thrust_reference_rejected():
    try:
        make_engine(thrust_reference="mars")
    except ValueError as exc:
        assert "thrust_reference" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unknown reference")


def test_effective_isp_brackets_the_declared_pair():
    eng = make_engine()
    assert np.isclose(eng.effective_isp(10.0, 0.0), 280.0, rtol=1e-9)
    assert np.isclose(eng.effective_isp(10.0, P_STD), 250.0, rtol=1e-6)


# --------------------------------------------------------------------------
# Nozzle area consistency
# --------------------------------------------------------------------------


def test_nozzle_area_for_isp_pair_is_self_consistent():
    area = nozzle_area_for_isp_pair(20000.0, 280.0, 250.0)
    assert 0.02 < area < 0.022, area
    eng = make_engine(nozzle_area=area)
    assert np.isclose(eng.implied_isp_sl(), 250.0, rtol=1e-6)


def test_old_default_area_is_flagged_as_inconsistent():
    """0.15 m^2 was a 437 mm exit on a 300 mm vehicle."""
    model = default_propulsion_model()
    bad = PropulsionModel(**{**model.__dict__, "nozzle_area_m2": 0.15})
    _, implied, error = bad.nozzle_area_consistency()
    assert implied < 0.03
    assert error > 5.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bad.to_engine()
    assert any("disagrees" in str(w.message) for w in caught)


def test_consistent_model_does_not_warn():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        default_propulsion_model().to_engine()
    assert not [w for w in caught if "disagrees" in str(w.message)]


def test_thrust_reference_round_trips(tmp_path=None):
    import json
    import tempfile

    from trajectory.propulsion.model import load_propulsion_model, save_propulsion_model

    model = PropulsionModel(**{
        **default_propulsion_model().__dict__,
        "thrust_reference": "sea_level",
    })
    with tempfile.TemporaryDirectory() as d:
        path = save_propulsion_model(model, Path(d) / "m.propulsion.json")
        assert json.loads(path.read_text())["performance"]["thrust_reference"] == "sea_level"
        assert load_propulsion_model(path).thrust_reference == "sea_level"


def test_legacy_file_without_reference_reads_as_vacuum():
    import json
    import tempfile

    from trajectory.propulsion.model import load_propulsion_model

    payload = default_propulsion_model().to_dict()
    del payload["performance"]["thrust_reference"]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "legacy.propulsion.json"
        path.write_text(json.dumps(payload))
        assert load_propulsion_model(path).thrust_reference == "vacuum"


# --------------------------------------------------------------------------
# Mass properties driven by propellant state
# --------------------------------------------------------------------------


def test_mass_tracks_remaining_propellant():
    mp = MassProperties(50.0, 150.0, np.zeros(3), np.diag([10.0, 10.0, 2.0]))
    assert mp.at_propellant(150.0)[0] == 200.0
    assert mp.at_propellant(75.0)[0] == 125.0
    assert mp.at_propellant(0.0)[0] == 50.0


def test_empty_tank_cannot_go_below_dry_mass():
    mp = MassProperties(50.0, 150.0, np.zeros(3), np.diag([10.0, 10.0, 2.0]))
    assert mp.at_propellant(-20.0)[0] == 50.0


# --------------------------------------------------------------------------
# End-to-end: the integrated mass history
# --------------------------------------------------------------------------


def test_state_vector_carries_propellant():
    sim = RocketSimulation()
    assert STATE_SIZE == 14 and PROP_IDX == 13

    state = np.zeros(STATE_SIZE)
    state[6] = 1.0
    state[PROP_IDX] = sim.mass_props.prop_mass
    deriv = sim.state_derivative(0.0, state)

    assert len(deriv) == STATE_SIZE
    assert deriv[PROP_IDX] < 0.0                       # burning
    assert np.isclose(deriv[PROP_IDX], -sim.engine.mass_flow_at(0.0))


def test_mass_never_increases_over_a_full_flight():
    """The headline regression: mass used to snap back to 200 kg at burnout."""
    sim = RocketSimulation()
    result = sim.run()
    assert result.success

    prop = result.y[PROP_IDX]
    masses = np.array([sim.mass_props.at_propellant(p)[0] for p in prop])

    assert np.all(np.diff(masses) <= 1e-9), "mass increased during flight"
    assert masses[0] <= sim.mass_props.mass_0 + 1e-9
    assert np.isclose(masses[-1], sim.mass_props.dry_mass, atol=1e-6)

    # The raw state may dip a hair below zero at exhaustion -- RK45 lands on the
    # boundary to its own tolerance, not exactly. Measured overshoot is ~3e-6 kg
    # on a 150 kg load. What matters is that it is solver noise rather than a
    # step's worth of flow (~3.6 kg here), and that at_propellant clamps it so
    # no consumer ever sees a negative propellant mass.
    one_step_of_flow = sim.engine.mass_flow_at(0.0) * 0.5
    assert prop.min() > -1e-3, prop.min()
    assert abs(prop.min()) < 0.01 * one_step_of_flow


def test_burned_propellant_matches_impulse():
    """Integrated mdot must equal total impulse / (Isp_vac * g0)."""
    sim = RocketSimulation()
    result = sim.run()
    burned = sim.mass_props.prop_mass - result.y[PROP_IDX][-1]

    eng = sim.engine
    times = np.linspace(0.0, 40.0, 40_001)
    expected = np.trapezoid([eng.mass_flow_at(t) for t in times], times)
    expected = min(expected, sim.mass_props.prop_mass)

    assert np.isclose(burned, expected, rtol=2e-3), (burned, expected)


def test_thrust_stops_when_tank_is_empty():
    """A curve longer than the propellant load must not keep pushing."""
    sim = RocketSimulation()
    state = np.zeros(STATE_SIZE)
    state[6] = 1.0
    state[PROP_IDX] = 0.0

    forces, moments, mass, _, _, mass_flow = sim.compute_forces_moments(state, 10.0)
    assert mass_flow == 0.0
    assert np.isclose(mass, sim.mass_props.dry_mass)
    # Only gravity remains (no velocity, so no aero).
    assert forces[1] < 0.0


# --------------------------------------------------------------------------


def _run_standalone() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001 - standalone runner
            failures.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
