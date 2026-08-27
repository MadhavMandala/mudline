"""Tests for parachute recovery and phased integration.

Pins the absence of any descent model: flights had no recovery, so a vehicle
returned to the ground at whatever speed ballistic drag allowed, and the
landing-dispersion statistics in trajectory.analysis had nothing to consume.

Runs under pytest, and standalone via
``python trajectory/tests/test_recovery.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.vehicle.recovery import (  # noqa: E402
    Parachute,
    RecoverySystem,
    standard_recovery,
)


# ------------------------------------------------------------- canopy model


def test_drag_area_is_zero_before_the_trigger():
    chute = Parachute(cda_m2=10.0, inflation_time_s=1.0)
    assert chute.drag_area_at(-1.0) == 0.0
    assert chute.drag_area_at(0.0) == 0.0


def test_drag_area_ramps_during_inflation():
    chute = Parachute(cda_m2=10.0, inflation_time_s=2.0)
    assert np.isclose(chute.drag_area_at(0.5), 2.5)
    assert np.isclose(chute.drag_area_at(1.0), 5.0)
    assert np.isclose(chute.drag_area_at(2.0), 10.0)


def test_drag_area_saturates_at_full_area():
    chute = Parachute(cda_m2=10.0, inflation_time_s=2.0)
    assert np.isclose(chute.drag_area_at(50.0), 10.0)


def test_inflation_is_continuous():
    """No jump in drag area, so the derivative stays integrable."""
    chute = Parachute(cda_m2=25.0, inflation_time_s=1.5)
    t = np.linspace(-1.0, 4.0, 2001)
    areas = np.array([chute.drag_area_at(v) for v in t])
    assert np.abs(np.diff(areas)).max() < 0.1


def test_deploy_delay_shifts_the_ramp():
    chute = Parachute(cda_m2=10.0, inflation_time_s=1.0, deploy_delay_s=2.0)
    assert chute.drag_area_at(1.5) == 0.0
    assert np.isclose(chute.drag_area_at(2.5), 5.0)
    assert np.isclose(chute.drag_area_at(3.0), 10.0)


def test_terminal_velocity_matches_the_force_balance():
    chute = Parachute(cda_m2=20.0)
    v = chute.terminal_velocity(mass_kg=50.0)
    drag = 0.5 * 1.225 * v * v * 20.0
    assert np.isclose(drag, 50.0 * 9.80665, rtol=1e-9)


def test_zero_area_canopy_never_slows_anything():
    assert Parachute(cda_m2=0.0).terminal_velocity(50.0) == float("inf")


# ------------------------------------------------------------------ sizing


def test_standard_recovery_hits_its_target_descent_rates():
    rec = standard_recovery(dry_mass_kg=50.0, main_descent_mps=6.0,
                            drogue_descent_mps=25.0)
    assert np.isclose(rec.main.terminal_velocity(50.0), 6.0, rtol=1e-9)
    assert np.isclose(rec.drogue.terminal_velocity(50.0), 25.0, rtol=1e-9)


def test_main_is_much_larger_than_drogue():
    rec = standard_recovery(dry_mass_kg=50.0)
    assert rec.main.cda_m2 > 10 * rec.drogue.cda_m2


def test_empty_recovery_is_disabled():
    assert not RecoverySystem().enabled
    assert RecoverySystem(main=Parachute(5.0)).enabled


# ------------------------------------------------------- phased integration


def _sim():
    from trajectory import simulation as tm
    return tm.RocketSimulation()


def _flight(**kwargs):
    rec = standard_recovery(dry_mass_kg=50.0, main_deploy_altitude_m=500.0)
    defaults = dict(launch_elevation=np.radians(85), t_max=1500.0, recovery=rec)
    defaults.update(kwargs)
    sim = _sim()
    return sim, sim.run(**defaults)


def test_flight_runs_all_three_phases():
    _, result = _flight()
    assert [p["name"] for p in result.phases] == ["ascent", "drogue", "main"]


def test_phases_are_contiguous_in_time():
    _, result = _flight()
    for earlier, later in zip(result.phases, result.phases[1:]):
        assert later["t_start"] >= earlier["t_end"] - 1e-6
        assert later["t_start"] - earlier["t_end"] < 0.2


def test_time_is_monotonic_across_the_join():
    """Concatenating phases must not produce a time series that goes backwards."""
    _, result = _flight()
    assert np.all(np.diff(result.t) > 0.0)


def test_vehicle_lands():
    _, result = _flight()
    assert result.y[1, -1] < 1.0
    assert result.t[-1] < 1500.0


def test_the_main_opens_at_the_set_height_above_a_raised_pad():
    """The trigger is an altimeter reading -- above the pad, not the sea.

    Compared against the raw state, a main set for 500 m never fired from
    a pad at 1,000 m: the vehicle was on the ground before it got there.
    """
    pad = 1000.0
    _, result = _flight(pad_position_m=np.array([0.0, pad, 0.0]))
    main = next(p for p in result.phases if p["name"] == "main")
    i = int(np.searchsorted(result.t, main["t_start"]))
    assert abs(result.y[1, i] - (pad + 500.0)) < 5.0
    assert abs(result.y[1, -1] - pad) < 1.0


def test_landing_speed_matches_the_main_canopy():
    """The design target: 6 m/s under the main."""
    _, result = _flight()
    descent_rate = abs(result.y[4, -1])
    assert 5.0 < descent_rate < 7.5, descent_rate


def test_recovery_slows_the_landing_dramatically():
    """Without a canopy the same vehicle arrives far faster."""
    _, with_chute = _flight()
    sim = _sim()
    without = sim.run(launch_elevation=np.radians(85), t_max=1500.0)
    assert abs(without.y[4, -1]) > 5 * abs(with_chute.y[4, -1])


def test_main_deploys_at_the_configured_altitude():
    _, result = _flight()
    main_phase = next(p for p in result.phases if p["name"] == "main")
    idx = int(np.searchsorted(result.t, main_phase["t_start"]))
    altitude = result.y[1, min(idx, result.y.shape[1] - 1)]
    assert abs(altitude - 500.0) < 25.0, altitude


def test_drogue_descent_is_faster_than_main_descent():
    _, result = _flight()
    drogue = next(p for p in result.phases if p["name"] == "drogue")
    main = next(p for p in result.phases if p["name"] == "main")
    # Sample each phase near its end, once the canopy is fully inflated.
    d_idx = int(np.searchsorted(result.t, drogue["t_end"])) - 2
    m_idx = -1
    assert abs(result.y[4, d_idx]) > abs(result.y[4, m_idx])


def test_no_recovery_still_produces_a_single_phase():
    sim = _sim()
    result = sim.run(launch_elevation=np.radians(85), t_max=400.0)
    assert [p["name"] for p in result.phases] == ["flight"]


def test_drogue_only_system_works():
    sim = _sim()
    rec = RecoverySystem(drogue=Parachute(cda_m2=1.3, inflation_time_s=0.4))
    result = sim.run(launch_elevation=np.radians(85), t_max=1500.0, recovery=rec)
    assert [p["name"] for p in result.phases] == ["ascent", "drogue"]
    assert result.y[1, -1] < 1.0


if __name__ == "__main__":
    failures = 0
    names = sorted(n for n in globals() if n.startswith("test_"))
    for name in names:
        try:
            globals()[name]()
            print(f"  PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(names) - failures}/{len(names)} passed")
    raise SystemExit(1 if failures else 0)
