"""Interior point sampling for arbitrary closed triangle meshes.

This is the piece that unlocks inertial matching on real CAD geometry. On the
massman side, ``override_distribute`` seeds a point-mass grid with
``generate_points_volumetric``, which maps a structured grid through a hex
Jacobian and is therefore locked to an 8-corner bounding box. Everything
downstream of the seeding -- ``build_constraints``, ``build_axis_constraints``,
``solve_kkt``, ``optimise_masses``, ``compute_mass_props`` -- already operates on
plain ``(points, masses)`` arrays and has no idea what shape it is inside.

Replace the seeder and the entire KKT matching capability, both the 10-constraint
tensor mode and the 5-constraint hinge-axis mode, works on a meshed solid
unchanged.

Containment
-----------
Two tests are provided:

``winding``
    Generalised winding number via the Van Oosterom-Strackee solid angle. Exact
    for a closed mesh and degrades gracefully on one with small holes -- a point
    deep inside a slightly leaky mesh still scores near 1. Cost is O(points x
    faces) with no acceleration structure, which is fine at seeding sizes
    (hundreds to low thousands of points).

``raycast``
    Moller-Trumbore parity count along a random direction. Faster, but assumes a
    genuinely closed surface and can be fooled when a ray grazes an edge or
    vertex. Degenerate hits are detected and the direction is re-rolled.

``auto`` picks ``raycast`` for a mesh that is watertight and consistently wound,
``winding`` otherwise.
"""

from __future__ import annotations

from typing import Literal, NamedTuple, Optional

import numpy as np

from massprops.solver.mesh_props import MeshMassProps, mass_properties
from massprops.solver.mesh_quality import is_consistently_wound, is_watertight

Method = Literal["auto", "winding", "raycast"]
Layout = Literal["lattice", "stratified", "random"]

# Cap on the (points x faces x 3) working array, in float64 elements. Keeps peak
# memory near 200 MB regardless of how many points the caller asks for.
_MAX_WORKING_ELEMENTS = 25_000_000


class SampleResult(NamedTuple):
    """Points sampled inside a mesh, plus what it took to get them."""

    points: np.ndarray          # (k, 3)
    fill_ratio: float           # mesh volume / bounding-box volume
    n_candidates: int           # points tested
    n_inside: int               # points accepted before any trimming
    rounds: int                 # how many generate-and-test passes ran
    layout: str
    method: str

    def summary(self) -> str:
        return (
            f"{len(self.points)} pts via {self.layout}/{self.method}, "
            f"fill={self.fill_ratio:.3f}, tested={self.n_candidates}, "
            f"accepted={self.n_inside}, rounds={self.rounds}"
        )


def _chunk_size(n_faces: int) -> int:
    if n_faces <= 0:
        return 1
    return max(1, _MAX_WORKING_ELEMENTS // (n_faces * 3))


def winding_numbers(
    vertices: np.ndarray, faces: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Generalised winding number of each point with respect to the surface.

    Returns values near 1.0 inside a closed outward-wound mesh and near 0.0
    outside. Nested or self-overlapping shells give higher integers.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    p = np.atleast_2d(np.asarray(points, dtype=float))

    tri = v[f]                                  # (m, 3, 3)
    out = np.empty(len(p), dtype=float)
    step = _chunk_size(len(f))

    for start in range(0, len(p), step):
        block = p[start : start + step]                     # (k, 3)
        rel = tri[None, :, :, :] - block[:, None, None, :]  # (k, m, 3, 3)
        a, b, c = rel[:, :, 0, :], rel[:, :, 1, :], rel[:, :, 2, :]

        la = np.linalg.norm(a, axis=2)
        lb = np.linalg.norm(b, axis=2)
        lc = np.linalg.norm(c, axis=2)

        numerator = np.einsum("kmi,kmi->km", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("kmi,kmi->km", a, b) * lc
            + np.einsum("kmi,kmi->km", a, c) * lb
            + np.einsum("kmi,kmi->km", b, c) * la
        )
        out[start : start + step] = (
            2.0 * np.arctan2(numerator, denominator)
        ).sum(axis=1) / (4.0 * np.pi)

    return out


def _raycast_inside(
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 4,
    eps: float = 1e-9,
) -> np.ndarray:
    """Odd-even parity test along a random ray. Re-rolls on degenerate hits."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    p = np.atleast_2d(np.asarray(points, dtype=float))

    tri = v[f]
    a = tri[:, 0, :]
    e1 = tri[:, 1, :] - a
    e2 = tri[:, 2, :] - a

    inside = np.zeros(len(p), dtype=bool)
    unresolved = np.ones(len(p), dtype=bool)
    step = _chunk_size(len(f))

    for _ in range(max_attempts):
        if not unresolved.any():
            break

        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        h = np.cross(direction, e2)                 # (m, 3)
        det = np.einsum("mi,mi->m", e1, h)          # (m,)
        parallel = np.abs(det) < eps
        safe_det = np.where(parallel, 1.0, det)
        inv_det = 1.0 / safe_det

        idx = np.flatnonzero(unresolved)
        for start in range(0, len(idx), step):
            sel = idx[start : start + step]
            block = p[sel]                                   # (k, 3)

            s = block[:, None, :] - a[None, :, :]            # (k, m, 3)
            u = np.einsum("kmi,mi->km", s, h) * inv_det[None, :]
            q = np.cross(s, e1[None, :, :])                  # (k, m, 3)
            w = np.einsum("kmi,i->km", q, direction) * inv_det[None, :]
            t = np.einsum("kmi,mi->km", q, e2) * inv_det[None, :]

            hit = (
                (~parallel)[None, :]
                & (u >= 0.0)
                & (w >= 0.0)
                & (u + w <= 1.0)
                & (t > eps)
            )

            # A hit sitting on a triangle edge, or a ray starting on the
            # surface, makes the parity count unreliable for that point.
            grazing = hit & (
                (u < 1e-7)
                | (w < 1e-7)
                | (u + w > 1.0 - 1e-7)
                | (np.abs(t) < 1e-7)
            )
            suspect = grazing.any(axis=1)

            parity = (hit.sum(axis=1) % 2).astype(bool)
            inside[sel] = np.where(suspect, inside[sel], parity)
            unresolved[sel] = suspect

    # Anything still ambiguous after every re-roll falls back to the robust test.
    if unresolved.any():
        idx = np.flatnonzero(unresolved)
        inside[idx] = winding_numbers(v, f, p[idx]) > 0.5

    return inside


def contains(
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    method: Method = "auto",
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Boolean mask of which points lie inside the mesh."""
    p = np.atleast_2d(np.asarray(points, dtype=float))
    if p.size == 0:
        return np.zeros(0, dtype=bool)

    if method == "auto":
        method = (
            "raycast"
            if is_watertight(faces) and is_consistently_wound(faces)
            else "winding"
        )

    if method == "winding":
        return winding_numbers(vertices, faces, p) > 0.5
    if method == "raycast":
        return _raycast_inside(
            vertices, faces, p, np.random.default_rng(seed)
        )
    raise ValueError(f"unknown containment method: {method!r}")


def _candidate_points(
    lower: np.ndarray,
    upper: np.ndarray,
    n_target: int,
    layout: Layout,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate ``~n_target`` candidates in the box, by the chosen layout."""
    extent = upper - lower
    box_volume = float(np.prod(extent))
    if box_volume <= 0:
        raise ValueError("mesh bounding box has zero extent")

    if layout == "random":
        return lower + rng.random((n_target, 3)) * extent

    # Cell edge that puts about n_target cells in the box, then a per-axis count
    # proportional to that axis's extent so cells stay near-cubic. Near-cubic
    # cells matter: an anisotropic grid biases the inertia of the seeded cloud
    # along the stretched axis before the KKT solve ever runs.
    edge = (box_volume / max(n_target, 1)) ** (1.0 / 3.0)
    counts = np.maximum(1, np.round(extent / edge).astype(int))

    axes = [
        (np.arange(counts[i]) + 0.5) / counts[i] * extent[i] + lower[i]
        for i in range(3)
    ]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)

    if layout == "stratified":
        cell = extent / counts
        grid = grid + (rng.random(grid.shape) - 0.5) * cell

    return grid


def sample_interior(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    layout: Layout = "lattice",
    method: Method = "auto",
    seed: Optional[int] = 0,
    exact: bool = True,
    max_rounds: int = 6,
) -> SampleResult:
    """Sample points uniformly inside a closed mesh.

    Args:
        vertices: (n, 3) coordinates.
        faces: (m, 3) triangle indices, wound outward.
        n_points: How many interior points are wanted.
        layout: ``lattice`` for a regular grid (best conditioned for the KKT
            solve), ``stratified`` for a jittered grid, ``random`` for uniform.
        method: Containment test -- see module docstring.
        seed: RNG seed, for reproducible clouds.
        exact: Trim to exactly ``n_points``. When False, every accepted point is
            returned, which keeps the sample strictly uniform.
        max_rounds: Cap on generate-and-test passes before giving up.

    Returns:
        SampleResult. Raises if no interior points could be found at all.
    """
    if n_points <= 0:
        raise ValueError(f"n_points must be positive; got {n_points}")

    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    rng = np.random.default_rng(seed)

    lower, upper = v.min(axis=0), v.max(axis=0)
    box_volume = float(np.prod(upper - lower))
    exact_volume = mass_properties(v, f).volume
    fill_ratio = exact_volume / box_volume if box_volume > 0 else 0.0

    if method == "auto":
        method = (
            "raycast"
            if is_watertight(f) and is_consistently_wound(f)
            else "winding"
        )

    # Ask for enough candidates that the expected yield clears n_points with
    # headroom, since the fill ratio only predicts the mean.
    demand = int(np.ceil(n_points / max(fill_ratio, 1e-6) * 1.25)) + 8

    accepted: list[np.ndarray] = []
    n_found = 0
    n_candidates = 0
    rounds = 0

    for _ in range(max_rounds):
        rounds += 1
        candidates = _candidate_points(lower, upper, demand, layout, rng)
        n_candidates += len(candidates)

        mask = contains(v, f, candidates, method=method, seed=seed)
        hit = candidates[mask]
        if len(hit):
            accepted.append(hit)
            n_found += len(hit)

        if n_found >= n_points:
            break

        # A lattice is deterministic, so repeating it verbatim yields nothing
        # new -- grow the demand instead of resampling the same points.
        demand = int(demand * 2.0) + 8

    if n_found == 0:
        raise ValueError(
            "no interior points found. The mesh may be inside-out, degenerate, "
            "or too thin for this point count -- inspect it with "
            "mesh_quality.inspect() and raise n_points."
        )

    points = np.vstack(accepted)

    if exact and len(points) > n_points:
        keep = rng.choice(len(points), size=n_points, replace=False)
        points = points[np.sort(keep)]

    return SampleResult(
        points=points,
        fill_ratio=fill_ratio,
        n_candidates=n_candidates,
        n_inside=n_found,
        rounds=rounds,
        layout=layout,
        method=method,
    )


class SeedForMatching(NamedTuple):
    """A seeded cloud plus the exact targets the KKT solve should reproduce."""

    points: np.ndarray          # (k, 3)
    masses: np.ndarray          # (k,) uniform starting guess
    target: MeshMassProps       # exact properties of the meshed solid
    sample: SampleResult


def seed_for_matching(
    vertices: np.ndarray,
    faces: np.ndarray,
    n_points: int,
    mass: float,
    layout: Layout = "lattice",
    method: Method = "auto",
    seed: Optional[int] = 0,
) -> SeedForMatching:
    """Seed a point cloud inside a meshed solid, ready for inertial matching.

    Drop-in replacement for ``override_distribute.generate_points_volumetric``
    plus ``initial_masses``, but taking a triangle mesh instead of an 8-corner
    bounding box. Feed ``points`` and ``masses`` straight into
    ``build_constraints`` / ``solve_kkt``.

    ``target`` carries the exact mass, CG and inertia of the solid itself, which
    is what you match against when no measured values are available -- and is
    the yardstick for how well a given ``n_points`` represents the shape.
    """
    result = sample_interior(
        vertices, faces, n_points, layout=layout, method=method, seed=seed
    )
    target = mass_properties(vertices, faces, mass=mass)
    masses = np.full(len(result.points), mass / len(result.points), dtype=float)
    return SeedForMatching(
        points=result.points, masses=masses, target=target, sample=result
    )
