"""The tool says what it knows to be wrong: supersonic drag on a steep boattail.

Runs under pytest, and standalone via
``python parametric/tests/test_boattail_caveat.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.aero import boattail_caveat, steepest_boattail_deg  # noqa: E402
from parametric.components import FinSet, Motor, PointMass, Stack  # noqa: E402
from parametric.model import VehicleModel  # noqa: E402
from parametric.standard import basic_rocket, boattailed_rocket  # noqa: E402
from parametric.xsec import NoseProfile  # noqa: E402


def _steep() -> VehicleModel:
    """The demonstrator with a Proteus-class boattail: 32 degrees over 4 cm."""
    model = VehicleModel("Steep", "A")
    airframe = Stack("airframe", wall_thickness_m=0.003)
    airframe.add_nose(NoseProfile.VON_KARMAN, 0.55, 0.10, sections=16)
    airframe.add_tube(0.30, 0.10, name="forward")
    airframe.add_tube(1.10, 0.10, name="motor")
    airframe.add_transition(0.04, 0.05, name="boattail")
    model.add(airframe)
    airframe.add(FinSet("fins", count=4, root_chord_m=0.22, tip_chord_m=0.09, span_m=0.10,
                        sweep_m=0.11, thickness_m=0.004, station_m=1.70))
    motor = Motor("motor", propellant_mass_kg=2.5, station_m=1.25, length_m=0.70)
    motor.update(isp_vac=210.0, isp_sl=185.0)
    motor.curve_from_impulse(2.5 * 210.0 * 9.80665, burn_time_s=3.4)
    airframe.add(motor)
    model.add(PointMass("avionics", 0.5, 0.95))
    return model


def test_the_steepest_boattail_is_measured():
    assert steepest_boattail_deg(basic_rocket()) == 0.0
    assert steepest_boattail_deg(boattailed_rocket()) == pytest.approx(4.76, abs=0.1)
    assert steepest_boattail_deg(_steep()) == pytest.approx(32.0, abs=0.5)


def test_the_caveat_needs_both_the_angle_and_the_mach():
    assert boattail_caveat(32.0, 0.9) is None
    assert boattail_caveat(10.0, 3.0) is None
    note = boattail_caveat(32.0, 3.0)
    assert note is not None and "not trusted" in note and "32.0" in note
    assert "provisional" in boattail_caveat(32.0, 3.0, "corrected")


def test_the_report_carries_the_caveat_only_when_it_applies():
    from parametric import aero

    model = _steep()
    fast = aero.AeroSettings(mach_min=0.1, mach_max=3.0, mach_points=6, alpha_max_deg=4.0, alpha_points=2)
    table, geometry = aero.run_analysis(model, fast)
    assert "CAVEAT" in aero.analysis_report(model, table, geometry, None, settings=fast)
    slow = aero.AeroSettings(mach_min=0.1, mach_max=0.8, mach_points=4, alpha_max_deg=4.0, alpha_points=2)
    table, geometry = aero.run_analysis(model, slow)
    assert "CAVEAT" not in aero.analysis_report(model, table, geometry, None, settings=slow)
    gentle = boattailed_rocket()
    table, geometry = aero.run_analysis(gentle, fast)
    assert "CAVEAT" not in aero.analysis_report(gentle, table, geometry, None, settings=fast)


def test_a_supersonic_flight_of_a_steep_boattail_says_so():
    from parametric import aero
    from parametric.flight import FlightSettings, fly_model

    model = _steep()
    settings = aero.AeroSettings(mach_min=0.1, mach_max=3.0, mach_points=6, alpha_max_deg=4.0, alpha_points=2)
    table, _ = aero.run_analysis(model, settings)
    outcome = fly_model(model, FlightSettings(couple_aero_altitude=False, dt_s=0.05), table,
                        aero_settings=settings)
    assert outcome.max_mach > 1.2
    assert outcome.boattail_half_angle_deg == pytest.approx(32.0, abs=0.5)
    assert any("not trusted" in note for note in outcome.caveats())
    outcome.boattail_model = "corrected"
    assert any("provisional" in note for note in outcome.caveats())


def test_the_fallback_law_has_no_boattail_caveat():
    from parametric.flight import FlightSettings, fly_model

    outcome = fly_model(_steep(), FlightSettings(use_aero_table=False, dt_s=0.1))
    assert not any("boattail" in note.lower() for note in outcome.caveats())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
