"""Tests for the mesh mass-property solver and interior sampler.

Validated against closed-form results for a box, a sphere and a cylinder rather
than against another library, so a bug in trimesh (or in our use of it) cannot
make these pass.

Runs under pytest, and standalone via ``python tests/test_mesh_solver.py`` for
environments without pytest installed.
"""

from __future__ import annotations

import numpy as np

from massprops.solver import (
    contains,
    inspect,
    is_consistently_wound,
    is_watertight,
    mass_properties,
    point_cloud_mass_properties,
    repair,
    sample_interior,
    seed_for_matching,
    signed_volume,
    winding_numbers,
)

# --------------------------------------------------------------------------
# Mesh fixtures. Winding is hand-checked outward; test_box_is_outward_wound
# verifies that independently of the code under test.
# --------------------------------------------------------------------------


def box_mesh(lower=(0.0, 0.0, 0.0), upper=(1.0, 1.0, 1.0)):
    x0, y0, z0 = lower
    x1, y1, z1 = upper
    vertices = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 3, 2], [0, 2, 1],      # z0, normal -z
            [4, 5, 6], [4, 6, 7],      # z1, normal +z
            [0, 1, 5], [0, 5, 4],      # y0, normal -y
            [3, 7, 6], [3, 6, 2],      # y1, normal +y
            [0, 4, 7], [0, 7, 3],      # x0, normal -x
            [1, 2, 6], [1, 6, 5],      # x1, normal +x
        ],
        dtype=int,
    )
    return vertices, faces


def icosphere_mesh(radius=1.0, subdivisions=3, center=(0.0, 0.0, 0.0)):
    """Subdivided icosahedron, projected onto a sphere."""
    t = (1.0 + 5.0**0.5) / 2.0
    verts = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=int,
    )

    for _ in range(subdivisions):
        midpoint: dict[tuple[int, int], int] = {}
        new_faces = []
        verts = list(verts)

        def mid(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in midpoint:
                midpoint[key] = len(verts)
                verts.append((np.asarray(verts[i]) + np.asarray(verts[j])) / 2.0)
            return midpoint[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]

        verts = np.array(verts, dtype=float)
        faces = np.array(new_faces, dtype=int)

    verts = verts / np.linalg.norm(verts, axis=1)[:, None] * radius
    return verts + np.asarray(center, dtype=float), faces


def cylinder_mesh(radius=1.0, height=2.0, segments=96):
    """Z-aligned cylinder centred on the origin, with capped ends."""
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    ring = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)

    bottom = np.hstack([ring, np.full((segments, 1), -height / 2.0)])
    top = np.hstack([ring, np.full((segments, 1), height / 2.0)])
    centre_bottom = np.array([[0.0, 0.0, -height / 2.0]])
    centre_top = np.array([[0.0, 0.0, height / 2.0]])

    vertices = np.vstack([bottom, top, centre_bottom, centre_top])
    ib, it = 2 * segments, 2 * segments + 1

    faces = []
    for i in range(segments):
        j = (i + 1) % segments
        faces.append([i, j, segments + j])
        faces.append([i, segments + j, segments + i])
        faces.append([ib, j, i])                       # bottom cap, -z
        faces.append([it, segments + i, segments + j])  # top cap, +z
    return vertices, np.array(faces, dtype=int)


def hollow_box_mesh(outer=1.0, inner=0.5):
    """Cube with a concentric cubic void: inner shell wound inward."""
    ov, of = box_mesh((-outer / 2,) * 3, (outer / 2,) * 3)
    iv, if_ = box_mesh((-inner / 2,) * 3, (inner / 2,) * 3)
    vertices = np.vstack([ov, iv])
    faces = np.vstack([of, if_[:, ::-1] + len(ov)])
    return vertices, faces


# --------------------------------------------------------------------------
# Exact mass properties
# --------------------------------------------------------------------------


def test_box_is_outward_wound():
    """Face normals point away from the centre, checked without the solver."""
    v, f = box_mesh((-1, -1, -1), (1, 1, 1))
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    normals = np.cross(b - a, c - a)
    face_centres = (a + b + c) / 3.0
    assert np.all(np.einsum("ij,ij->i", normals, face_centres) > 0)


def test_box_volume_cg_inertia():
    v, f = box_mesh((0, 0, 0), (2, 3, 4))
    props = mass_properties(v, f, mass=60.0)

    assert np.isclose(props.volume, 24.0)
    assert np.allclose(props.centroid, [1.0, 1.5, 2.0])

    m, (a, b, c) = 60.0, (2.0, 3.0, 4.0)
    expected = np.diag([
        m * (b**2 + c**2) / 12.0,
        m * (a**2 + c**2) / 12.0,
        m * (a**2 + b**2) / 12.0,
    ])
    assert np.allclose(props.inertia, expected)


def test_box_inertia_is_origin_independent():
    """Same shape far from the origin gives the same inertia about its own CG."""
    near = mass_properties(*box_mesh((0, 0, 0), (2, 3, 4)), mass=60.0)
    far = mass_properties(*box_mesh((100, -50, 7), (102, -47, 11)), mass=60.0)
    assert np.allclose(near.inertia, far.inertia)
    assert np.allclose(far.centroid, [101.0, -48.5, 9.0])


def test_sphere_converges_to_analytic():
    radius, mass = 1.5, 10.0
    exact_volume = 4.0 / 3.0 * np.pi * radius**3
    exact_inertia = 2.0 / 5.0 * mass * radius**2

    errors = []
    for level in (1, 2, 3, 4):
        props = mass_properties(
            *icosphere_mesh(radius, subdivisions=level), mass=mass
        )
        errors.append(abs(props.volume - exact_volume) / exact_volume)
        assert np.allclose(props.centroid, 0.0, atol=1e-9)
        assert props.volume < exact_volume   # inscribed polyhedron

    # Halving the facet size should quarter the error. Requiring at least 3x
    # per refinement pins the solver to second-order convergence: a first-order
    # bug would still shrink the error, just far more slowly, and a plain
    # monotonic check would not catch it.
    ratios = [before / after for before, after in zip(errors, errors[1:])]
    assert all(ratio > 3.0 for ratio in ratios), ratios
    assert errors[-1] < 0.005, errors

    props = mass_properties(*icosphere_mesh(radius, subdivisions=4), mass=mass)
    assert np.allclose(props.inertia, np.eye(3) * exact_inertia, rtol=0.01)


def test_cylinder_matches_analytic():
    radius, height, mass = 0.75, 2.5, 12.0
    v, f = cylinder_mesh(radius, height, segments=256)
    props = mass_properties(v, f, mass=mass)

    assert np.isclose(props.volume, np.pi * radius**2 * height, rtol=2e-3)
    assert np.allclose(props.centroid, 0.0, atol=1e-9)

    axial = 0.5 * mass * radius**2
    transverse = mass * (3.0 * radius**2 + height**2) / 12.0
    assert np.isclose(props.inertia[2, 2], axial, rtol=5e-3)
    assert np.isclose(props.inertia[0, 0], transverse, rtol=5e-3)
    assert np.isclose(props.inertia[1, 1], transverse, rtol=5e-3)


def test_hollow_box_subtracts_the_void():
    v, f = hollow_box_mesh(outer=2.0, inner=1.0)
    props = mass_properties(v, f, density=1.0)
    assert np.isclose(props.volume, 8.0 - 1.0)
    assert np.allclose(props.centroid, 0.0, atol=1e-12)


def test_density_and_mass_agree():
    v, f = box_mesh((0, 0, 0), (2, 2, 2))
    by_density = mass_properties(v, f, density=2.5)
    by_mass = mass_properties(v, f, mass=8.0 * 2.5)
    assert np.isclose(by_density.mass, by_mass.mass)
    assert np.allclose(by_density.inertia, by_mass.inertia)


def test_inward_wound_mesh_is_rejected():
    v, f = box_mesh()
    try:
        mass_properties(v, f[:, ::-1])
    except ValueError as exc:
        assert "signed volume" in str(exc)
    else:
        raise AssertionError("expected a ValueError for an inward-wound mesh")


# --------------------------------------------------------------------------
# Quality checks and repair
# --------------------------------------------------------------------------


def test_watertight_detection():
    v, f = box_mesh()
    assert is_watertight(f)
    assert is_consistently_wound(f)
    assert is_watertight(f[:-1]) is False       # a hole where a triangle was


def test_repair_fixes_flipped_faces():
    v, f = box_mesh((0, 0, 0), (2, 3, 4))
    damaged = f.copy()
    damaged[[1, 4, 7]] = damaged[[1, 4, 7]][:, ::-1]
    assert not is_consistently_wound(damaged)

    v2, f2, before, after = repair(v, damaged)
    assert not before.consistently_wound
    assert after.consistently_wound
    assert after.watertight
    assert np.isclose(signed_volume(v2, f2), 24.0)


def test_repair_drops_degenerate_faces():
    v, f = box_mesh()
    padded = np.vstack([f, [[0, 0, 1], [2, 2, 2]]])
    _, cleaned, before, after = repair(v, padded)
    assert before.degenerate_faces == 2
    assert after.degenerate_faces == 0
    assert len(cleaned) == len(f)


def test_inspect_reports_holes():
    v, f = box_mesh()
    report = inspect(v, f[:-2])
    assert not report.watertight
    assert report.boundary_edges > 0
    assert "NOT watertight" in report.summary()


# --------------------------------------------------------------------------
# Containment
# --------------------------------------------------------------------------


def test_containment_on_box():
    v, f = box_mesh((0, 0, 0), (1, 1, 1))
    points = np.array(
        [
            [0.5, 0.5, 0.5],      # centre
            [0.01, 0.01, 0.01],   # just inside a corner
            [0.99, 0.5, 0.5],     # near a face
            [1.5, 0.5, 0.5],      # outside
            [-0.1, 0.5, 0.5],     # outside
            [0.5, 0.5, 2.0],      # outside
        ]
    )
    expected = np.array([True, True, True, False, False, False])
    for method in ("winding", "raycast", "auto"):
        assert np.array_equal(
            contains(v, f, points, method=method), expected
        ), method


def test_containment_handles_axis_aligned_degeneracies():
    """Points sharing coordinates with vertices are where naive rays fail."""
    v, f = box_mesh((0, 0, 0), (1, 1, 1))
    grid = np.array(np.meshgrid(*[[0.25, 0.5, 0.75]] * 3, indexing="ij"))
    inside = grid.reshape(3, -1).T
    outside = inside + np.array([2.0, 0.0, 0.0])

    assert contains(v, f, inside, method="raycast").all()
    assert not contains(v, f, outside, method="raycast").any()


def test_winding_number_is_near_one_inside():
    v, f = icosphere_mesh(1.0, subdivisions=2)
    w = winding_numbers(v, f, np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]))
    assert np.isclose(w[0], 1.0, atol=1e-6)
    assert np.isclose(w[1], 0.0, atol=1e-6)


def test_containment_respects_internal_void():
    v, f = hollow_box_mesh(outer=2.0, inner=1.0)
    points = np.array([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])   # void, then wall
    assert np.array_equal(
        contains(v, f, points, method="winding"), [False, True]
    )


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def test_sampled_points_are_all_inside():
    v, f = icosphere_mesh(2.0, subdivisions=3)
    result = sample_interior(v, f, 400, seed=1)
    assert len(result.points) == 400
    assert contains(v, f, result.points).all()
    assert np.isclose(result.fill_ratio, np.pi / 6.0, rtol=0.05)


def test_sample_never_lands_in_a_void():
    v, f = hollow_box_mesh(outer=2.0, inner=1.0)
    result = sample_interior(v, f, 300, method="winding", seed=3)
    inside_void = np.all(np.abs(result.points) < 0.5 - 1e-9, axis=1)
    assert not inside_void.any()


def test_sampled_cloud_reproduces_box_inertia():
    """A uniform lattice converges on the continuum result.

    Cell-centre sampling under-reports the second moment by h^2/12 per axis
    (Sheppard's correction), so the error falls as (cell/extent)^2 -- a few
    thousand points is already well inside a percent.
    """
    v, f = box_mesh((0, 0, 0), (2, 3, 4))
    mass = 60.0
    exact = mass_properties(v, f, mass=mass)

    result = sample_interior(v, f, 8000, layout="lattice", seed=5, exact=False)
    masses = np.full(len(result.points), mass / len(result.points))
    cloud = point_cloud_mass_properties(result.points, masses)

    assert np.allclose(cloud.centroid, exact.centroid, atol=0.02)
    assert np.allclose(np.diag(cloud.inertia), np.diag(exact.inertia), rtol=0.01)


def test_sampled_cloud_reproduces_sphere_inertia():
    radius, mass = 1.5, 10.0
    v, f = icosphere_mesh(radius, subdivisions=3)
    exact = mass_properties(v, f, mass=mass)

    result = sample_interior(v, f, 6000, layout="lattice", seed=7, exact=False)
    masses = np.full(len(result.points), mass / len(result.points))
    cloud = point_cloud_mass_properties(result.points, masses)

    assert np.allclose(cloud.centroid, 0.0, atol=0.02)
    assert np.allclose(
        np.diag(cloud.inertia), np.diag(exact.inertia), rtol=0.02
    )


def test_layouts_all_produce_interior_points():
    v, f = cylinder_mesh(1.0, 3.0, segments=64)
    for layout in ("lattice", "stratified", "random"):
        result = sample_interior(v, f, 250, layout=layout, seed=11)
        assert len(result.points) == 250, layout
        assert contains(v, f, result.points).all(), layout


def test_sampling_is_reproducible():
    v, f = box_mesh()
    a = sample_interior(v, f, 100, layout="stratified", seed=42)
    b = sample_interior(v, f, 100, layout="stratified", seed=42)
    assert np.array_equal(a.points, b.points)


def test_seed_for_matching_targets_the_exact_solid():
    """The KKT bridge: cloud in, exact targets alongside it."""
    radius, mass = 1.0, 25.0
    v, f = icosphere_mesh(radius, subdivisions=3)
    seeded = seed_for_matching(v, f, 500, mass=mass, seed=13)

    assert len(seeded.points) == 500
    assert np.isclose(seeded.masses.sum(), mass)
    assert np.isclose(seeded.target.mass, mass)
    assert contains(v, f, seeded.points).all()

    # The uniform starting guess is already close; the KKT solve closes the gap.
    start = point_cloud_mass_properties(seeded.points, seeded.masses)
    assert np.allclose(start.centroid, seeded.target.centroid, atol=0.05)


def test_thin_geometry_still_yields_points():
    """A thin plate has a poor fill ratio; the sampler must still converge."""
    v, f = box_mesh((0, 0, 0), (10, 10, 0.05))
    result = sample_interior(v, f, 200, seed=17)
    assert len(result.points) == 200
    assert contains(v, f, result.points).all()


def test_zero_points_rejected():
    v, f = box_mesh()
    try:
        sample_interior(v, f, 0)
    except ValueError as exc:
        assert "n_points" in str(exc)
    else:
        raise AssertionError("expected a ValueError for n_points=0")


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
            failures.append((name, exc))
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
