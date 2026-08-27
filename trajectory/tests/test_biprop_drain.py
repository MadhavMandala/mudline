"""Tests for multi-tank propellant draining -- the biprop case.

One engine, one integrated propellant state, several columns feeding it.
The single-column model put a liquid's whole load at the motor and drained
it there, so the simulator flew a vehicle with half its wet mass teleported
to the tail. These check the split by mixture ratio, the waterfall when one
side runs dry, and that the vehicle the simulator flies is the vehicle the
mass summary describes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.vehicle.mass_properties import (  # noqa: E402
    LIQUID,
    RADIAL,
    MassProperties,
    PropellantLoad,
)

# Body frame: +Y forward, so a station is a negative y.
DRY_MASS = 40.0
CG_DRY = np.array([0.0, -1.5, 0.0])

FUEL = dict(mass_kg=60.0, forward=np.array([0.0, -1.0, 0.0]),
            aft=np.array([0.0, -2.0, 0.0]))
OX = dict(mass_kg=120.0, forward=np.array([0.0, -2.2, 0.0]),
          aft=np.array([0.0, -3.0, 0.0]))
MIXTURE_RATIO = 2.0        # sized to the loads: 120 / 60


def build(fuel_mass=FUEL["mass_kg"], ox_mass=OX["mass_kg"],
          ratio=MIXTURE_RATIO) -> MassProperties:
    props = MassProperties(
        dry_mass=DRY_MASS,
        prop_mass=fuel_mass + ox_mass,
        cg_dry=CG_DRY.copy(),
        i_tensor_dry=np.diag([50.0, 1.0, 50.0]),
        roll_axis=1,
    )
    props.set_propellant_loads([
        PropellantLoad(mass_kg=fuel_mass, forward=FUEL["forward"],
                       aft=FUEL["aft"], burn_geometry=LIQUID,
                       radius_m=0.14, drain_share=1.0 / (1.0 + ratio)),
        PropellantLoad(mass_kg=ox_mass, forward=OX["forward"],
                       aft=OX["aft"], burn_geometry=LIQUID,
                       radius_m=0.14, drain_share=ratio / (1.0 + ratio)),
    ])
    return props


def station(vector) -> float:
    return abs(float(vector[1]))


def expected_cg(fuel_left: float, ox_left: float,
                fuel_full: float = FUEL["mass_kg"],
                ox_full: float = OX["mass_kg"]) -> np.ndarray:
    """Hand-built CG: dry mass plus each tank's settled column."""
    def settled(load, left, full):
        centre = 0.5 * (load["forward"] + load["aft"])
        fraction = left / full if full > 0 else 0.0
        return load["aft"] + (centre - load["aft"]) * fraction

    total = DRY_MASS + fuel_left + ox_left
    moment = (CG_DRY * DRY_MASS
              + settled(FUEL, fuel_left, fuel_full) * fuel_left
              + settled(OX, ox_left, ox_full) * ox_left)
    return moment / total


# ---------------------------------------------------------------- draining


def test_the_full_vehicle_balances_where_the_hand_sum_says():
    props = build()
    _, cg, _ = props.at_propellant(180.0)
    assert np.allclose(cg, expected_cg(60.0, 120.0))


def test_consumption_splits_by_mixture_ratio():
    """30 kg burned at O/F 2 is 10 kg of fuel and 20 kg of oxidizer."""
    props = build()
    mass, cg, _ = props.at_propellant(150.0)
    assert np.isclose(mass, DRY_MASS + 150.0)
    assert np.allclose(cg, expected_cg(50.0, 100.0))


def test_a_dry_tank_hands_its_share_to_the_wet_one():
    """Undersized oxidizer: it exhausts first and fuel supplies the rest."""
    props = build(fuel_mass=60.0, ox_mass=30.0, ratio=2.0)
    # Burn 60 kg. The first 45 kg drain 15 fuel + 30 ox (the oxidizer's 2/3
    # share caps at what it holds); the last 15 kg are all fuel.
    mass, cg, _ = props.at_propellant(30.0)
    assert np.isclose(mass, DRY_MASS + 30.0)
    assert np.allclose(cg, expected_cg(30.0, 0.0, fuel_full=60.0, ox_full=30.0))


def test_mass_is_conserved_at_every_level():
    props = build()
    for remaining in np.linspace(0.0, 180.0, 13):
        mass, cg, inertia = props.at_propellant(remaining)
        assert np.isclose(mass, DRY_MASS + remaining)
        assert np.all(np.isfinite(cg))
        assert np.all(np.isfinite(inertia))


def test_the_ratio_split_is_a_state_a_lump_cannot_hold():
    """Equal tanks at O/F 2: halfway through, the tanks are NOT half full.

    The oxidizer side has supplied two thirds of everything burned, so its
    tank is emptier than the fuel's. A single lumped column has one fill
    fraction and cannot represent the asymmetry.
    """
    props = build(fuel_mass=90.0, ox_mass=90.0, ratio=2.0)
    _, cg, _ = props.at_propellant(90.0)     # 90 of 180 burned
    # fuel down 30 (60 left), ox down 60 (30 left)
    assert np.allclose(cg, expected_cg(60.0, 30.0, fuel_full=90.0, ox_full=90.0))


# ------------------------------------------------------------------- limits


def test_empty_reduces_to_the_dry_vehicle():
    props = build()
    mass, cg, inertia = props.at_propellant(0.0)
    assert np.isclose(mass, DRY_MASS)
    assert np.allclose(cg, CG_DRY)
    assert np.allclose(inertia, props.i_tensor_dry)


def test_transverse_axes_stay_equal():
    """Axisymmetric loads on the roll axis cannot split pitch from yaw."""
    props = build()
    for remaining in (180.0, 100.0, 30.0, 0.0):
        _, _, inertia = props.at_propellant(remaining)
        diagonal = np.diag(inertia)
        assert np.isclose(diagonal[0], diagonal[2], rtol=1e-12)
        assert np.allclose(inertia, inertia.T)


def test_the_tanks_carry_their_parallel_axis_terms():
    """Two tanks off the CG must contribute m*d^2 each, not a lump's worth."""
    props = build()
    _, cg, inertia = props.at_propellant(180.0)
    floor = props.i_tensor_dry[0, 0]
    for load in props.loads:
        offset = station(load.centroid(1.0)) - station(cg)
        floor += load.mass_kg * offset ** 2 * 0.5
    assert inertia[0, 0] > floor


def test_a_propellant_state_above_the_loads_total_is_rescaled():
    """An inconsistent config keeps the mass positioned, not lost.

    If a propulsion file's load disagrees with the tanks', the integrator's
    state wins for the total and the tanks carry proportionally more, so the
    CG stays a mass-weighted average rather than drifting to the origin.
    """
    props = build()
    mass, cg, _ = props.at_propellant(180.0 * 1.2)
    assert np.isclose(mass, DRY_MASS + 216.0)
    total = DRY_MASS + 216.0
    moment = (CG_DRY * DRY_MASS
              + 0.5 * (FUEL["forward"] + FUEL["aft"]) * 72.0
              + 0.5 * (OX["forward"] + OX["aft"]) * 144.0)
    assert np.allclose(cg, moment / total)


def test_single_column_geometry_clears_the_loads():
    """Last caller wins: the two descriptions never mix."""
    props = build()
    props.set_propellant_geometry(FUEL["forward"], FUEL["aft"], LIQUID,
                                  radius_m=0.03)
    assert props.loads == []
    _, cg, _ = props.at_propellant(180.0)
    single = (CG_DRY * DRY_MASS
              + 0.5 * (FUEL["forward"] + FUEL["aft"]) * 180.0) / (DRY_MASS + 180.0)
    assert np.allclose(cg, single)


def test_a_load_with_a_nonsense_geometry_is_rejected():
    with pytest.raises(ValueError, match="burn_geometry must be"):
        PropellantLoad(mass_kg=1.0, forward=np.zeros(3), aft=np.ones(3),
                       burn_geometry="sideways")


# --------------------------------------------------------- through the model


def biprop_model(ratio: float = MIXTURE_RATIO, label: bool = True):
    from parametric.components import Motor, Tank
    from parametric.model import VehicleModel
    from parametric.xsec import NoseProfile

    from parametric.components import Stack

    model = VehicleModel("biprop")
    nose = Stack("nose")
    nose.add_nose(NoseProfile.OGIVE, 0.4, 0.3)
    model.add(nose)

    fuel = Tank("fuel", station_m=0.5, propellant_mass_kg=30.0,
                propellant_density_kg_m3=810.0)
    model.add(fuel)
    lox = Tank("lox", station_m=1.8, propellant_mass_kg=60.0,
               propellant_density_kg_m3=1140.0)
    if label:
        lox.contents = "oxidizer"
    model.add(lox)

    engine = Motor("engine", station_m=3.2, length_m=0.4)
    engine.feed = "tanks"
    engine.set("mixture_ratio", ratio)
    engine.flat_curve(5000.0, 20.0)
    model.add(engine)
    return model


def test_the_simulator_flies_the_vehicle_the_status_bar_shows():
    """The bug this replaces: 90 kg of tank propellant living at the engine."""
    from parametric import analysis
    from trajectory.frames import station_to_body

    model = biprop_model()
    sim = analysis.build_simulation(model)
    summary = model.mass_summary()

    mass, cg, _ = sim.mass_props.at_propellant(sim.mass_props.prop_mass)
    assert np.isclose(mass, summary.wet_mass_kg)
    assert np.allclose(cg, station_to_body(summary.wet_cg_station_m))


def test_the_drain_shares_follow_the_mixture_ratio():
    from parametric import analysis

    sim = analysis.build_simulation(biprop_model(ratio=2.0))
    by_mass = {load.mass_kg: load.drain_share for load in sim.mass_props.loads}
    assert np.isclose(by_mass[30.0], 1.0 / 3.0)      # fuel: 1/(1+MR)
    assert np.isclose(by_mass[60.0], 2.0 / 3.0)      # lox: MR/(1+MR)


def test_unlabeled_tanks_fall_back_to_proportional():
    """Both tanks read as fuel, so the ratio has nothing to split between."""
    from parametric import analysis

    sim = analysis.build_simulation(biprop_model(ratio=2.0, label=False))
    by_mass = {load.mass_kg: load.drain_share for load in sim.mass_props.loads}
    assert np.isclose(by_mass[30.0], 30.0 / 90.0)
    assert np.isclose(by_mass[60.0], 60.0 / 90.0)


def test_a_grain_motor_keeps_loaded_tanks_placed_but_undrained():
    from parametric import analysis
    from parametric.standard import basic_rocket
    from parametric.components import Tank

    model = basic_rocket()
    tank = Tank("ballast", station_m=0.5, propellant_mass_kg=10.0)
    model.add(tank)
    sim = analysis.build_simulation(model)

    shares = sorted(load.drain_share for load in sim.mass_props.loads)
    assert shares == [0.0, 1.0]        # the grain drains, the tank rides
    geometries = {load.burn_geometry for load in sim.mass_props.loads}
    assert geometries == {RADIAL, LIQUID}


def test_a_plain_solid_still_takes_the_single_column_path():
    from parametric import analysis
    from parametric.standard import basic_rocket

    props = analysis.build_simulation(basic_rocket()).mass_props
    assert props.loads == []
    assert props.cg_prop_empty is not None


# ------------------------------------------------------------ save and load


def test_contents_and_mixture_ratio_survive_a_save(tmp_path):
    from parametric.model import VehicleModel

    model = biprop_model(ratio=2.3)
    path = model.save(tmp_path / "biprop.json")
    loaded = VehicleModel.load(path)
    assert {t.name: t.contents for t in loaded.tanks} == {
        "fuel": "fuel", "lox": "oxidizer",
    }
    assert np.isclose(loaded.motors[0].get("mixture_ratio"), 2.3)


def test_an_older_file_without_contents_reads_as_fuel(tmp_path):
    import json

    from parametric.model import VehicleModel

    path = tmp_path / "old.json"
    biprop_model().save(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    def strip(node):
        node.pop("contents", None)
        for child in node.get("children", []):
            strip(child)

    strip(data["tree"])
    path.write_text(json.dumps(data), encoding="utf-8")
    loaded = VehicleModel.load(path)
    assert all(t.contents == "fuel" for t in loaded.tanks)


# --------------------------------------------------------------- validation


def test_validate_reports_a_stranding_imbalance():
    """Tanks sized against the mixture ratio strand the difference."""
    model = biprop_model(ratio=2.0)
    model.tanks[0].set("propellant_mass", 40.0)      # fuel
    model.tanks[1].set("propellant_mass", 40.0)      # lox; O/F 2 wants 80
    problems = model.validate()
    assert any("stranding" in p and "fuel" in p for p in problems)


def test_validate_reports_a_ratio_with_nothing_to_split():
    model = biprop_model(ratio=2.0, label=False)
    problems = model.validate()
    assert any("labelled" in p for p in problems)


def test_matched_tanks_validate_clean_of_drain_problems():
    problems = biprop_model(ratio=2.0).validate()
    assert not any("stranding" in p or "labelled" in p for p in problems)
