"""Tests for where the propellant sits as it is consumed.

The old model held one fixed centroid, which is right for a solid core burner
and wrong for everything else. These check that each geometry moves the way the
physics says, and that the vehicle stays a body of revolution while it does.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.vehicle.mass_properties import (  # noqa: E402
    END_BURNER,
    LIQUID,
    RADIAL,
    MassProperties,
)

FORWARD = np.array([0.0, -1.0, 0.0])      # body +Y is forward, station grows aft
AFT = np.array([0.0, -2.0, 0.0])


def build(geometry: str, dry_mass: float = 4.0, prop_mass: float = 3.0):
    props = MassProperties(
        dry_mass=dry_mass,
        prop_mass=prop_mass,
        cg_dry=np.array([0.0, -1.2, 0.0]),
        i_tensor_dry=np.diag([1.0, 0.01, 1.0]),
        roll_axis=1,
    )
    props.set_propellant_geometry(FORWARD, AFT, geometry, radius_m=0.03)
    return props


def station(vector) -> float:
    """Distance along the body axis, as a positive station."""
    return abs(float(vector[1]))


# ------------------------------------------------------------------ centroid


def test_a_core_burner_keeps_its_centroid():
    """The annulus burns outward, so the axial centroid does not move."""
    props = build(RADIAL)
    stations = [station(props.propellant_centroid(f)) for f in (1.0, 0.5, 0.0)]
    assert len(set(np.round(stations, 9))) == 1
    assert np.isclose(stations[0], 1.5)


def test_a_liquid_tank_drains_towards_the_aft_bulkhead():
    props = build(LIQUID)
    full = station(props.propellant_centroid(1.0))
    half = station(props.propellant_centroid(0.5))
    empty = station(props.propellant_centroid(0.0))
    assert np.isclose(full, 1.5)          # centre of the tank
    assert np.isclose(empty, 2.0)         # the last drop, on the bulkhead
    assert full < half < empty            # migrates aft as it empties


def test_an_end_burner_regresses_forwards():
    props = build(END_BURNER)
    assert np.isclose(station(props.propellant_centroid(1.0)), 1.5)
    assert np.isclose(station(props.propellant_centroid(0.0)), 1.0)
    assert station(props.propellant_centroid(0.25)) < 1.5


def test_the_centroid_is_exactly_linear_in_fill():
    """For a constant cross-section this is exact, not an approximation."""
    props = build(LIQUID)
    quarter = station(props.propellant_centroid(0.25))
    half = station(props.propellant_centroid(0.5))
    three = station(props.propellant_centroid(0.75))
    assert np.isclose(half - quarter, three - half)


def test_fill_fraction_is_clamped():
    props = build(LIQUID)
    assert np.allclose(props.propellant_centroid(-1.0),
                       props.propellant_centroid(0.0))
    assert np.allclose(props.propellant_centroid(5.0),
                       props.propellant_centroid(1.0))


def test_an_unset_geometry_behaves_as_it_always_did():
    """Backwards compatibility: a fixed centroid unless told otherwise."""
    props = MassProperties(4.0, 3.0, np.array([0.0, -1.2, 0.0]), np.eye(3))
    props.cg_prop_full = np.array([0.0, -1.5, 0.0])
    assert np.allclose(props.propellant_centroid(1.0),
                       props.propellant_centroid(0.0))


def test_an_unknown_geometry_is_rejected():
    props = build(RADIAL)
    with pytest.raises(ValueError, match="burn_geometry must be"):
        props.set_propellant_geometry(FORWARD, AFT, "sideways")


# ----------------------------------------------------------------- vehicle CG


def test_a_liquid_moves_the_vehicle_cg_aft_before_forward():
    """The excursion a fixed centroid cannot produce.

    The tank centre is behind the dry CG, so early draining pushes the balance
    point further aft before the shrinking load finally pulls it forward. The
    minimum static margin is therefore partway into the burn, not on the rail.
    """
    props = build(LIQUID)
    stations = [
        station(props.at_propellant(props.prop_mass * f)[1])
        for f in (1.0, 0.75, 0.5, 0.25, 0.0)
    ]
    assert stations[1] > stations[0]      # moves aft first
    assert stations[-1] < stations[0]     # ends forward of where it began
    assert max(stations) > stations[0]


def test_a_core_burner_moves_the_cg_monotonically_forward():
    props = build(RADIAL)
    stations = [
        station(props.at_propellant(props.prop_mass * f)[1])
        for f in (1.0, 0.75, 0.5, 0.25, 0.0)
    ]
    assert stations == sorted(stations, reverse=True)


def test_every_geometry_ends_at_the_dry_cg():
    """With no propellant left there is only the structure."""
    for geometry in (RADIAL, LIQUID, END_BURNER):
        props = build(geometry)
        _, cg, _ = props.at_propellant(0.0)
        assert np.allclose(cg, props.cg_dry)


def test_mass_is_dry_plus_what_is_left():
    props = build(RADIAL)
    for remaining in (3.0, 1.5, 0.0):
        mass, _, _ = props.at_propellant(remaining)
        assert np.isclose(mass, 4.0 + remaining)


def test_negative_propellant_is_treated_as_empty():
    props = build(LIQUID)
    assert np.isclose(props.at_propellant(-5.0)[0], props.dry_mass)


# -------------------------------------------------------------------- inertia


def test_a_body_of_revolution_keeps_equal_transverse_inertia():
    """The bug this replaced: a roll term applied to a transverse axis.

    ``i_tensor[2, 2] *= ...`` assumed the vehicle rolled about Z. The simulator
    flies +Y forward, so it scaled pitch or yaw instead and split the two by
    40% at burnout, which no axisymmetric body can do.
    """
    for geometry in (RADIAL, LIQUID, END_BURNER):
        props = build(geometry)
        for fill in (1.0, 0.5, 0.0):
            _, _, inertia = props.at_propellant(props.prop_mass * fill)
            diagonal = np.diag(inertia)
            assert np.isclose(diagonal[0], diagonal[2], rtol=1e-12), (
                f"{geometry} at {fill}: transverse axes disagree"
            )


def test_roll_inertia_is_far_smaller_than_pitch():
    props = build(RADIAL)
    _, _, inertia = props.at_propellant(props.prop_mass)
    diagonal = np.diag(inertia)
    assert diagonal[1] < 0.1 * diagonal[0]


def test_propellant_adds_inertia_rather_than_scaling_it_away():
    """A loaded vehicle has more pitch inertia than an empty one."""
    props = build(RADIAL)
    _, _, full = props.at_propellant(props.prop_mass)
    _, _, empty = props.at_propellant(0.0)
    assert full[0, 0] > empty[0, 0]


def test_the_parallel_axis_term_is_actually_there():
    """Propellant offset from the CG must contribute m*d^2, not nothing."""
    props = build(RADIAL)
    _, cg, inertia = props.at_propellant(props.prop_mass)
    offset = station(props.propellant_centroid(1.0)) - station(cg)
    assert offset > 0.1
    lower_bound = props.i_tensor_dry[0, 0] + props.prop_mass * offset ** 2
    assert inertia[0, 0] > lower_bound * 0.5


def test_an_empty_vehicle_reduces_to_its_dry_tensor():
    props = build(RADIAL)
    _, _, inertia = props.at_propellant(0.0)
    assert np.allclose(inertia, props.i_tensor_dry)


def test_inertia_is_symmetric():
    props = build(LIQUID)
    _, _, inertia = props.at_propellant(props.prop_mass * 0.5)
    assert np.allclose(inertia, inertia.T)


def test_a_zero_mass_vehicle_does_not_divide_by_nothing():
    props = MassProperties(0.0, 0.0, np.zeros(3), np.eye(3))
    mass, cg, inertia = props.at_propellant(0.0)
    assert mass == 0.0
    assert np.all(np.isfinite(cg))
    assert np.all(np.isfinite(inertia))


# --------------------------------------------------------- through the model


def test_the_motor_carries_its_geometry_to_the_simulator():
    from parametric import analysis
    from parametric.standard import basic_rocket

    model = basic_rocket()
    model.motors[0].burn_geometry = LIQUID
    props = analysis.build_simulation(model).mass_props
    assert props.burn_geometry == LIQUID
    assert props.cg_prop_empty is not None


def test_burn_geometry_survives_a_save(tmp_path):
    from parametric.model import VehicleModel
    from parametric.standard import basic_rocket

    model = basic_rocket()
    model.motors[0].burn_geometry = LIQUID
    path = model.save(tmp_path / "liquid.json")
    assert VehicleModel.load(path).motors[0].burn_geometry == LIQUID


def test_an_older_file_without_the_field_reads_as_a_solid(tmp_path):
    import json

    from parametric.model import VehicleModel
    from parametric.standard import basic_rocket

    path = tmp_path / "old.json"
    basic_rocket().save(path)
    data = json.loads(path.read_text(encoding="utf-8"))

    def strip(node):
        node.pop("burn_geometry", None)
        for child in node.get("children", []):
            strip(child)

    strip(data["tree"])
    path.write_text(json.dumps(data), encoding="utf-8")
    assert VehicleModel.load(path).motors[0].burn_geometry == RADIAL


def test_the_propellant_radius_follows_the_load():
    from parametric.standard import basic_rocket

    motor = basic_rocket().motors[0]
    before = motor.propellant_radius_m
    motor.set("propellant_mass", motor.get("propellant_mass") * 4.0)
    assert motor.propellant_radius_m > before
    # Four times the volume in the same length is twice the radius.
    assert np.isclose(motor.propellant_radius_m, before * 2.0, rtol=1e-9)


def test_a_grain_with_no_length_reports_no_radius():
    from parametric.standard import basic_rocket

    motor = basic_rocket().motors[0]
    motor.set("length", 0.0)
    assert motor.propellant_radius_m == 0.0
