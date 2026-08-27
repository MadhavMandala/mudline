"""Build imperfections: a tilted thrust line, an off-centre CG, a fin cant
the table was not built with -- on a nominal flight and in a dispersion.

Each is held to a closed form: a tilted thrust trims at the moment
balance, an off-centre CG is the same moment as the equivalent tilt, and
a cant offset rolls at the rate a built-in cant would. Runs under pytest,
and standalone via ``python parametric/tests/test_imperfections.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from parametric.dispersion import dispersions_about  # noqa: E402
from parametric.flight import FlightSettings, configure_simulation, fly_model  # noqa: E402
from parametric.standard import basic_rocket  # noqa: E402
from trajectory.analysis.dispersion import perturb_simulation  # noqa: E402

_TABLES: dict[float, object] = {}


def _table(cant_deg: float = 0.0):
    """One table per cant, built once for the module."""
    if cant_deg not in _TABLES:
        from parametric import aero

        model = basic_rocket()
        model.fin_sets[0].set("cant", cant_deg)
        _TABLES[cant_deg], _ = aero.run_analysis(model, aero.AeroSettings(
            mach_min=0.05, mach_max=1.5, mach_points=8, alpha_max_deg=8.0,
            alpha_points=3,
        ))
    return _TABLES[cant_deg]


def _settings(**overrides) -> FlightSettings:
    base = dict(use_aero_table=True, use_recovery=False, dt_s=0.02,
                couple_aero_altitude=False, elevation_deg=90.0)
    base.update(overrides)
    return FlightSettings(**base)


def _fly(cant_table_deg: float = 0.0, **overrides):
    model = basic_rocket()
    return fly_model(model, _settings(**overrides), _table(cant_table_deg))


def _mean_alpha(log, t0: float, t1: float) -> float:
    t = np.asarray(log.time_s)
    mask = (t >= t0) & (t <= t1)
    return float(np.mean(np.asarray(log.alpha_deg)[mask]))


# --------------------------------------------------------- the thrust line


def test_a_perfect_build_flies_straight_up():
    outcome = _fly()
    assert _mean_alpha(outcome.log, 1.0, 2.0) < 0.05


def test_a_tilted_thrust_trims_at_the_moment_balance():
    """``T l sin(d) = q S CNa a (x_cp - x_cg)``, late in the burn where the
    trim changes slowly enough for the vehicle to follow it. Early on it
    cannot: at 75 m/s the same tilt asks for ten times the angle, and the
    vehicle lags a trim that is collapsing as q builds."""
    tilt = 0.5
    outcome = _fly(thrust_misalignment_deg=tilt, thrust_misalignment_clock_deg=0.0)
    log, sim, table = outcome.log, outcome.simulation, _table()
    model = basic_rocket()
    nozzle = model.station_range_m()[1]
    t = np.asarray(log.time_s)
    mask = (t >= 2.0) & (t <= 2.5)
    expected = []
    for i in np.flatnonzero(mask):
        arm = nozzle - float(log.cg_station_m[i])
        margin = float(log.static_margin_cal[i]) * log.reference_diameter_m
        cn_alpha = table.cn_alpha_per_rad(float(log.mach[i]), 2.0)
        restoring = float(log.dynamic_pressure_pa[i]) * sim.reference_area * cn_alpha * margin
        expected.append(np.degrees(
            float(log.thrust_n[i]) * arm * np.sin(np.radians(tilt)) / restoring
        ))
    expected = float(np.mean(expected))
    measured = _mean_alpha(log, 2.0, 2.5)
    assert 0.05 < expected < 6.0, expected
    assert measured == pytest.approx(expected, rel=0.3), (measured, expected)


def test_the_tilt_is_a_direction_not_a_scalar():
    east = _fly(thrust_misalignment_deg=0.5, thrust_misalignment_clock_deg=0.0)
    north = _fly(thrust_misalignment_deg=0.5, thrust_misalignment_clock_deg=90.0)
    a, b = east.states[-1, [0, 2]], north.states[-1, [0, 2]]
    assert np.linalg.norm(a) > 20.0 and np.linalg.norm(b) > 20.0
    # Same distance, a quarter turn apart.
    assert np.linalg.norm(a) == pytest.approx(np.linalg.norm(b), rel=0.05)
    assert abs(np.dot(a, b)) < 0.2 * np.linalg.norm(a) * np.linalg.norm(b)


# --------------------------------------------------------- the CG


def test_a_cg_offset_is_a_thrust_moment():
    """Thrust along the axis, the CG a millimetre off it: a moment of ``T d``."""
    model = basic_rocket()
    nominal = configure_simulation(model, _settings(), _table())
    shifted = configure_simulation(model, _settings(cg_offset_m=0.002, cg_offset_clock_deg=90.0), _table())
    assert np.allclose(shifted.mass_props.cg_dry - nominal.mass_props.cg_dry, [0.0, 0.0, 0.002])
    state = nominal.initial_state(nominal.launch_rail or __import__("trajectory.sim", fromlist=["LaunchRail"]).LaunchRail())
    for sim in (nominal, shifted):
        sim.launch_rail = None
    a = nominal.evaluate(state, 0.5)
    b = shifted.evaluate(state, 0.5)
    assert a.thrust_n > 100.0
    moment = b.moment_body_nm - a.moment_body_nm
    # The dry offset is diluted by the propellant sitting on the axis.
    dilution = nominal.mass_props.dry_mass / a.mass_kg
    assert np.linalg.norm(moment) == pytest.approx(a.thrust_n * 0.002 * dilution, rel=1e-6)


def test_a_cg_offset_flies_like_the_equivalent_tilt():
    """A dry offset ``d`` is a tilt of ``d_wet / l`` at the nozzle."""
    offset = 0.005
    model = basic_rocket()
    sim = configure_simulation(model, _settings(), _table())
    mass, cg, _ = sim.mass_props.at_propellant(0.5 * sim.mass_props.prop_mass)
    wet_offset = offset * sim.mass_props.dry_mass / mass
    arm = model.station_range_m()[1] + float(cg[1])          # nozzle behind the CG
    tilt = np.degrees(wet_offset / arm)
    by_offset = _fly(cg_offset_m=offset, cg_offset_clock_deg=0.0)
    by_tilt = _fly(thrust_misalignment_deg=tilt, thrust_misalignment_clock_deg=0.0)
    a = _mean_alpha(by_offset.log, 1.0, 2.0)
    b = _mean_alpha(by_tilt.log, 1.0, 2.0)
    assert a > 0.1 and b > 0.1, (a, b)
    assert a == pytest.approx(b, rel=0.15), (a, b)


# --------------------------------------------------------- the cant


def test_the_table_carries_the_forcing_per_radian():
    canted, straight = _table(2.0), _table(0.0)
    row_c, row_s = canted.lookup(0.5, 0.0), straight.lookup(0.5, 0.0)
    assert row_s.cl_cant is not None and row_s.cl_cant > 0.0
    assert row_s.cl_roll == pytest.approx(0.0, abs=1e-12)
    assert row_c.cl_roll == pytest.approx(row_c.cl_cant * np.radians(2.0), rel=1e-9)
    assert row_c.cl_cant == pytest.approx(row_s.cl_cant, rel=1e-9)


def test_the_forcing_per_radian_survives_a_csv_round_trip(tmp_path):
    from trajectory.vehicle.aero_database import AeroDatabase

    table = _table(0.0)
    reloaded = AeroDatabase.from_csv(table.to_csv(tmp_path / "t.csv"), reference_length_m=2.0)
    assert reloaded.lookup(0.5, 0.0).cl_cant == pytest.approx(table.lookup(0.5, 0.0).cl_cant)


def test_a_cant_offset_rolls_like_a_built_in_cant():
    """An uncanted table plus a 2 degree offset against a table built at 2."""
    built_in = _fly(cant_table_deg=2.0)
    offset = _fly(cant_table_deg=0.0, fin_cant_offset_deg=2.0)
    i = built_in.log.index_at(1.2)
    a = float(built_in.log.roll_rate_radps[i])
    b = float(offset.log.roll_rate_radps[offset.log.index_at(1.2)])
    assert abs(a) > 1.0
    assert b == pytest.approx(a, rel=0.05), (a, b)


# --------------------------------------------------------- the dispersion


def test_a_dispersed_case_adds_to_the_nominal_build():
    model = basic_rocket()
    sim = configure_simulation(
        model, _settings(thrust_misalignment_deg=0.2, thrust_misalignment_clock_deg=0.0,
                         fin_cant_offset_deg=0.1), _table(),
    )
    perturb_simulation(sim, {"thrust_tilt_x_deg": 0.1, "thrust_tilt_z_deg": -0.05,
                             "cg_offset_z_m": 0.001, "fin_cant_offset_deg": 0.05})
    assert np.allclose(np.degrees(sim.thrust_tilt_rad), [0.3, -0.05])
    assert sim.mass_props.cg_dry[2] == pytest.approx(0.001)
    assert np.degrees(sim.fin_cant_offset_rad) == pytest.approx(0.15)
    assert sim.aero_model.cant_offset_rad == pytest.approx(np.radians(0.15))


def test_the_spreads_are_centred_on_zero_and_optional():
    with_them = dispersions_about(
        _settings(), 5.0, thrust_misalignment_sd_deg=0.1, cg_offset_sd_m=0.001,
        fin_cant_sd_deg=0.2,
    )
    assert with_them["thrust_tilt_x_deg"] == (0.0, 0.1, -0.4, 0.4)
    assert with_them["thrust_tilt_z_deg"] == with_them["thrust_tilt_x_deg"]
    assert with_them["cg_offset_z_m"] == (0.0, 0.001, -0.004, 0.004)
    assert with_them["fin_cant_offset_deg"] == (0.0, 0.2, -0.8, 0.8)
    without = dispersions_about(_settings(), 5.0)
    assert not any(k.startswith(("thrust_tilt", "cg_offset", "fin_cant")) for k in without)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
