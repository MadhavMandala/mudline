"""Tests for launch rail geometry and the on-rail constraint.

Pins the defect where ``run()`` accepted launch_azimuth and launch_elevation
and used neither, starting every flight perfectly vertical at a hardcoded
50 m/s with no rail constraint.

Runs under pytest, and standalone via
``python trajectory/tests/test_launch_rail.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from trajectory.eom import quat_to_dcm  # noqa: E402
from trajectory.sim.launch_rail import (  # noqa: E402
    LaunchRail,
    alignment_quaternion,
    rail_direction,
)

UP = np.array([0.0, 1.0, 0.0])
EAST = np.array([1.0, 0.0, 0.0])
NORTH = np.array([0.0, 0.0, 1.0])


# ---------------------------------------------------------------- geometry


def test_vertical_rail_points_up():
    assert np.allclose(rail_direction(0.0, np.pi / 2), UP, atol=1e-12)


def test_azimuth_zero_is_north():
    assert np.allclose(rail_direction(0.0, 0.0), NORTH, atol=1e-12)


def test_azimuth_ninety_is_east():
    assert np.allclose(rail_direction(np.pi / 2, 0.0), EAST, atol=1e-12)


def test_direction_is_always_a_unit_vector():
    for az in np.linspace(0, 2 * np.pi, 13):
        for el in np.linspace(0, np.pi / 2, 7):
            assert np.isclose(np.linalg.norm(rail_direction(az, el)), 1.0)


def test_elevation_sets_the_vertical_component():
    for el_deg in [0, 30, 45, 60, 89, 90]:
        d = rail_direction(np.radians(45.0), np.radians(el_deg))
        assert np.isclose(d[1], np.sin(np.radians(el_deg)))


# ------------------------------------------------------------- attitude


def test_alignment_quaternion_rotates_body_axis_onto_the_rail():
    for az in [0.0, 1.0, 2.5]:
        for el in [np.radians(45.0), np.radians(80.0), np.pi / 2]:
            target = rail_direction(az, el)
            q = alignment_quaternion(UP, target)
            dcm_b2i = quat_to_dcm(q)
            assert np.allclose(dcm_b2i @ UP, target, atol=1e-12), (az, el)


def test_alignment_quaternion_is_normalised():
    q = alignment_quaternion(UP, rail_direction(1.1, 0.7))
    assert np.isclose(np.linalg.norm(q), 1.0)


def test_alignment_of_parallel_vectors_is_identity():
    assert np.allclose(alignment_quaternion(UP, UP), [1.0, 0.0, 0.0, 0.0])


def test_alignment_of_antiparallel_vectors_is_a_half_turn():
    q = alignment_quaternion(UP, -UP)
    assert np.allclose(quat_to_dcm(q) @ UP, -UP, atol=1e-12)


def test_non_vertical_body_axis_is_handled():
    """A vehicle whose thrust axis is not the body 'up' vector still aims right."""
    body_axis = np.array([0.0, 0.0, 1.0])
    target = rail_direction(0.6, np.radians(70.0))
    q = alignment_quaternion(body_axis, target)
    assert np.allclose(quat_to_dcm(q) @ body_axis, target, atol=1e-12)


# --------------------------------------------------------------- constraint


def test_distance_along_tracks_travel():
    rail = LaunchRail(elevation_rad=np.pi / 2, length_m=5.0)
    assert np.isclose(rail.distance_along(np.array([0.0, 3.0, 0.0])), 3.0)


def test_on_rail_until_the_length_is_cleared():
    rail = LaunchRail(elevation_rad=np.pi / 2, length_m=5.0)
    assert rail.is_on_rail(np.array([0.0, 0.0, 0.0]))
    assert rail.is_on_rail(np.array([0.0, 4.9, 0.0]))
    assert not rail.is_on_rail(np.array([0.0, 5.1, 0.0]))


def test_zero_length_rail_never_constrains():
    assert not LaunchRail(length_m=0.0).is_on_rail(np.zeros(3))


def test_transverse_acceleration_is_reacted_by_the_rail():
    """A crosswind load must not move the vehicle sideways while on the rail."""
    rail = LaunchRail(elevation_rad=np.pi / 2, length_m=5.0)
    acc = np.array([30.0, 20.0, -12.0])          # large transverse components
    out = rail.constrain_acceleration(acc, np.array([0.0, 5.0, 0.0]))
    assert np.allclose(out, [0.0, 20.0, 0.0])


def test_constrained_acceleration_stays_parallel_to_the_rail():
    rail = LaunchRail(azimuth_rad=0.9, elevation_rad=np.radians(75.0), length_m=4.0)
    out = rail.constrain_acceleration(np.array([5.0, 9.0, -3.0]), np.array([0.0, 2.0, 0.0]))
    cross = np.cross(out, rail.direction)
    assert np.allclose(cross, 0.0, atol=1e-12)


def test_vehicle_at_rest_does_not_slide_backwards():
    """Before thrust exceeds weight the pad holds the vehicle up."""
    rail = LaunchRail(elevation_rad=np.pi / 2, length_m=5.0)
    out = rail.constrain_acceleration(np.array([0.0, -9.81, 0.0]), np.zeros(3))
    assert np.allclose(out, 0.0)


def test_moving_vehicle_can_still_decelerate():
    """The floor applies only at rest; a coasting vehicle must decelerate."""
    rail = LaunchRail(elevation_rad=np.pi / 2, length_m=5.0)
    out = rail.constrain_acceleration(np.array([0.0, -9.81, 0.0]), np.array([0.0, 3.0, 0.0]))
    assert out[1] < 0.0


# ------------------------------------------------------------- rail exit


def test_exit_state_reports_first_crossing():
    rail = LaunchRail(elevation_rad=np.pi / 2, length_m=5.0)
    times = np.array([0.0, 1.0, 2.0, 3.0])
    states = np.zeros((4, 14))
    states[:, 1] = [0.0, 2.0, 6.0, 12.0]     # clears 5 m between t=1 and t=2
    states[:, 4] = [0.0, 4.0, 8.0, 12.0]
    exit_state = rail.exit_state(times, states)
    assert exit_state is not None
    assert np.isclose(exit_state["time_s"], 2.0)
    assert np.isclose(exit_state["velocity_mps"], 8.0)


def test_exit_state_is_none_if_never_cleared():
    rail = LaunchRail(elevation_rad=np.pi / 2, length_m=50.0)
    states = np.zeros((3, 14))
    states[:, 1] = [0.0, 1.0, 2.0]
    assert rail.exit_state(np.array([0.0, 1.0, 2.0]), states) is None


# --------------------------------------------------- end-to-end integration


def _sim():
    from trajectory import simulation as tm
    return tm.RocketSimulation()


def test_launch_starts_at_rest_not_at_fifty_mps():
    sim = _sim()
    state0 = sim.initial_state(LaunchRail(length_m=5.0))
    assert np.allclose(state0[3:6], 0.0)


def test_vertical_launch_has_no_downrange():
    sim = _sim()
    result = sim.run(launch_elevation=np.pi / 2, t_max=60.0)
    downrange = np.linalg.norm(result.y[[0, 2], -1])
    assert downrange < 1.0, downrange


def test_azimuth_steers_the_flight():
    """North and East launches must go to different places."""
    north = _sim().run(launch_azimuth=0.0, launch_elevation=np.radians(70), t_max=60.0)
    east = _sim().run(launch_azimuth=np.pi / 2, launch_elevation=np.radians(70), t_max=60.0)
    assert north.y[2, -1] > 100.0     # travelled North
    assert abs(north.y[0, -1]) < 1.0  # but not East
    assert east.y[0, -1] > 100.0      # travelled East
    assert abs(east.y[2, -1]) < 1.0   # but not North


def test_lower_elevation_gives_more_downrange():
    far = _sim().run(launch_elevation=np.radians(45), t_max=120.0)
    near = _sim().run(launch_elevation=np.radians(85), t_max=120.0)
    far_range = np.linalg.norm(far.y[[0, 2], -1])
    near_range = np.linalg.norm(near.y[[0, 2], -1])
    assert far_range > near_range


def test_rail_exit_velocity_matches_the_closed_form():
    """v = sqrt(2 * a * L) for constant net acceleration along a 5 m rail."""
    sim = _sim()
    result = sim.run(launch_elevation=np.pi / 2, rail_length_m=5.0, t_max=30.0)
    exit_state = result.rail_exit
    assert exit_state is not None

    thrust, _ = sim.engine.thrust_at(0.0, 101325.0)
    mass = sim.mass_props.mass_0
    acc = thrust / mass - 9.80665
    expected = np.sqrt(2 * acc * 5.0)
    # Sampled at 0.1 s, so the reported crossing overshoots slightly.
    assert 0.9 * expected < exit_state["velocity_mps"] < 1.25 * expected, (
        exit_state["velocity_mps"], expected
    )


def test_attitude_is_frozen_while_on_the_rail():
    sim = _sim()
    result = sim.run(launch_elevation=np.radians(80), rail_length_m=20.0, t_max=30.0)
    rail = sim.launch_rail
    on_rail = [i for i, s in enumerate(result.y.T) if rail.is_on_rail(s[0:3])]
    omegas = result.y[10:13, on_rail]
    assert np.allclose(omegas, 0.0, atol=1e-9), np.abs(omegas).max()


def test_disabling_the_rail_still_flies():
    result = _sim().run(rail_length_m=0.0, t_max=60.0)
    assert result.success
    assert result.rail_exit is None


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
