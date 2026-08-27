"""Tests for viewport scene maths.

Camera framing, grid scaling and the display transform are pure geometry, so
they are tested without a window. The previous viewer had none of this
separable -- every calculation lived inside a widget that needed a GL context
to exist at all, so none of it could be checked.

Runs under pytest, and standalone via ``python app/tests/test_scene.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.scene import (  # noqa: E402
    OrbitCamera,
    build_axis_triad,
    build_grid,
    display_transform,
    nice_spacing,
)


# ------------------------------------------------------------------ camera


def test_camera_looks_at_its_target():
    camera = OrbitCamera()
    camera.target = np.array([1.0, 2.0, 3.0])
    camera.distance = 5.0
    assert np.isclose(np.linalg.norm(camera.eye() - camera.target), 5.0)


def test_view_matrix_puts_the_target_on_the_axis():
    camera = OrbitCamera()
    camera.target = np.array([2.0, 1.0, -3.0])
    camera.distance = 7.0
    view = camera.view_matrix()
    seen = view @ np.append(camera.target, 1.0)
    assert np.allclose(seen[:2], 0.0, atol=1e-5)
    assert np.isclose(seen[2], -7.0, atol=1e-5)


def test_near_and_far_scale_with_distance():
    """A fixed near plane throws away depth precision at every other scale."""
    close, far = OrbitCamera(), OrbitCamera()
    close.distance = 0.2
    far.distance = 50_000.0
    assert close.near < far.near
    assert far.far > close.far
    for camera in (close, far):
        assert camera.far / camera.near < 1e7, "depth precision would collapse"


def test_projection_maps_the_near_plane_to_minus_one():
    camera = OrbitCamera()
    camera.distance = 10.0
    proj = camera.projection_matrix()
    point = np.array([0.0, 0.0, -camera.near, 1.0])
    clip = proj @ point
    assert np.isclose(clip[2] / clip[3], -1.0, atol=1e-4)


def test_projection_maps_the_far_plane_to_plus_one():
    camera = OrbitCamera()
    camera.distance = 10.0
    proj = camera.projection_matrix()
    clip = proj @ np.array([0.0, 0.0, -camera.far, 1.0])
    assert np.isclose(clip[2] / clip[3], 1.0, atol=1e-4)


def test_framing_fits_the_box():
    camera = OrbitCamera()
    camera.frame(np.array([-1.0, 0.0, -1.0]), np.array([1.0, 4.0, 1.0]))
    assert np.allclose(camera.target, [0.0, 2.0, 0.0])
    assert camera.distance > 2.0


def test_zoom_and_orbit_stay_in_range():
    camera = OrbitCamera()
    for _ in range(200):
        camera.zoom(1.0)
    assert camera.distance > 0.0
    for _ in range(200):
        camera.orbit(0.0, 100.0)
    assert abs(camera.elevation) < np.pi / 2


def test_pan_moves_the_target_not_the_distance():
    camera = OrbitCamera()
    before = camera.distance
    camera.pan(40.0, 20.0)
    assert np.isclose(camera.distance, before)
    assert not np.allclose(camera.target, 0.0)


# -------------------------------------------------------------------- grid


def test_nice_spacing_rounds_to_1_2_5():
    for value, expected in [
        (0.9, 1.0), (1.4, 2.0), (3.0, 5.0), (7.0, 10.0),
        (0.03, 0.05), (240.0, 500.0), (1400.0, 2000.0),
    ]:
        assert np.isclose(nice_spacing(value), expected), value


def test_grid_scales_with_the_view():
    """A 0.2 m part and a 100 km trajectory both need a readable grid."""
    close = build_grid(0.5)
    far = build_grid(100_000.0)
    assert close.spacing < far.spacing
    assert close.spacing <= 0.5
    assert far.spacing >= 1000.0


def test_grid_is_line_pairs_with_matching_colours():
    grid = build_grid(10.0)
    assert grid.positions.shape[0] % 2 == 0
    assert grid.positions.shape[0] == grid.colors.shape[0]
    assert np.allclose(grid.positions[:, 1], 0.0), "grid must lie on the ground"


def test_grid_follows_the_camera_target():
    near_origin = build_grid(10.0, np.zeros(3))
    far_away = build_grid(10.0, np.array([500.0, 0.0, 500.0]))
    assert far_away.positions[:, 0].mean() > near_origin.positions[:, 0].mean()


def test_axis_triad_has_three_coloured_lines():
    positions, colors = build_axis_triad(2.0)
    assert positions.shape == (6, 3)
    assert colors.shape == (6, 3)
    assert np.isclose(positions[1, 0], 2.0)   # X
    assert np.isclose(positions[3, 1], 2.0)   # Y
    assert np.isclose(positions[5, 2], 2.0)   # Z


# -------------------------------------------------------- display transform


def test_display_stands_the_vehicle_upright():
    """Model +Z is aft; on screen the nose must point up."""
    length = 2.0
    matrix = display_transform(length)
    nose = matrix @ np.array([0.0, 0.0, 0.0, 1.0])      # model nose tip
    tail = matrix @ np.array([0.0, 0.0, length, 1.0])   # model tail
    assert np.isclose(nose[1], length), "nose should be at the top"
    assert np.isclose(tail[1], 0.0), "tail should sit on the ground"


def test_display_transform_is_rigid():
    """No scale or shear, which is what lets the shader use mat3(model)."""
    rotation = display_transform(3.0)[:3, :3]
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert np.isclose(abs(np.linalg.det(rotation)), 1.0)


def test_display_preserves_radius():
    matrix = display_transform(2.0)
    point = matrix @ np.array([0.15, 0.0, 1.0, 1.0])
    assert np.isclose(np.hypot(point[0], point[2]), 0.15)


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
