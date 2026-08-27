"""Exact mass properties of a closed triangle mesh.

Pure NumPy, no third-party mesh library. This is the first-party replacement for
the ``trimesh.moment_inertia`` call in ``massprops.mesh.mesher`` -- same results,
but testable against closed-form shapes and free of a black-box dependency.

Method
------
The solid is decomposed into signed tetrahedra, each formed by the origin and
one boundary triangle. Because the mesh is closed, the signed contributions of
tetrahedra outside the solid cancel exactly, so the sum over all faces is the
true integral regardless of where the origin sits.

For each tetrahedron (0, a, b, c) with ``A = [a b c]`` (columns):

    6 * V_tet = det(A)
    integral of x x^T over the tet = det(A) * A @ C0 @ A^T

where ``C0`` is the canonical-tetrahedron covariance. Summing gives the second
moment matrix ``C`` about the origin, and the inertia tensor follows from

    I = trace(C) * I3 - C

Orientation
-----------
Faces must be wound consistently outward, which makes the total signed volume
positive. ``mass_properties`` raises on a negative total rather than silently
returning ``abs()`` -- an inward-wound mesh is a real defect and hiding it here
would produce a correct-looking inertia with an inverted geometry upstream. Use
``massprops.solver.mesh_quality.repair`` to fix winding first.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

# Integral of x x^T over the canonical tetrahedron (0,0,0), (1,0,0), (0,1,0),
# (0,0,1). Diagonal terms are 1/60, off-diagonal 1/120.
_CANONICAL_TET_COVARIANCE = np.array(
    [[2.0, 1.0, 1.0],
     [1.0, 2.0, 1.0],
     [1.0, 1.0, 2.0]]
) / 120.0


class MeshMassProps(NamedTuple):
    """Mass properties of a closed mesh.

    ``inertia`` is taken about ``centroid``, matching the convention of
    ``massprops.model.models.MassProperties`` and of
    ``consolidate_points.MassComponent.inertia_cg`` on the massman side.
    """

    volume: float
    centroid: np.ndarray      # (3,)
    inertia: np.ndarray       # (3, 3) about the centroid
    mass: float


def _triangle_corners(vertices: np.ndarray, faces: np.ndarray):
    """Return the three corner-coordinate arrays of every face, each (m, 3)."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must be (n, 3); got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (m, 3); got {f.shape}")
    if f.size and (f.max() >= len(v) or f.min() < 0):
        raise ValueError("faces reference vertex indices outside the vertex array")
    return v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]


def _tet_determinants(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Six times the signed volume of each tetrahedron (origin, a, b, c)."""
    return np.einsum("ij,ij->i", a, np.cross(b, c))


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed mesh. Positive when wound outward."""
    a, b, c = _triangle_corners(vertices, faces)
    if len(a) == 0:
        return 0.0
    return float(_tet_determinants(a, b, c).sum() / 6.0)


def mass_properties(
    vertices: np.ndarray,
    faces: np.ndarray,
    density: float = 1.0,
    mass: Optional[float] = None,
) -> MeshMassProps:
    """Exact volume, centroid and inertia tensor of a closed triangle mesh.

    Args:
        vertices: (n, 3) coordinates.
        faces: (m, 3) triangle vertex indices, wound outward.
        density: Mass per unit volume. Ignored when ``mass`` is given.
        mass: Total mass. When supplied, overrides ``density`` -- the usual case
            for a part whose weight is known but whose material is not.

    Returns:
        MeshMassProps with inertia taken about the centroid.
    """
    a, b, c = _triangle_corners(vertices, faces)
    if len(a) == 0:
        raise ValueError("mesh has no faces")

    det = _tet_determinants(a, b, c)
    volume = float(det.sum() / 6.0)

    if volume <= 0.0:
        raise ValueError(
            f"mesh signed volume is {volume:.6g}; expected positive. The mesh is "
            "either inward-wound or not closed -- run mesh_quality.repair first."
        )

    total_det = det.sum()
    centroid = ((a + b + c) / 4.0 * det[:, None]).sum(axis=0) / total_det

    # Second moment matrix about the origin: sum_n det_n * A_n @ C0 @ A_n^T,
    # where the columns of A_n are the tet's three non-origin corners.
    corners = np.stack([a, b, c], axis=2)            # (m, 3, 3), columns a|b|c
    covariance = np.einsum(
        "n,nij,jk,nlk->il", det, corners, _CANONICAL_TET_COVARIANCE, corners
    )

    if mass is not None:
        if mass < 0:
            raise ValueError(f"mass must be non-negative; got {mass}")
        scale = mass / volume
        total_mass = float(mass)
    else:
        scale = density
        total_mass = float(density * volume)

    covariance = covariance * scale

    inertia_origin = np.trace(covariance) * np.eye(3) - covariance

    # Parallel axis, origin -> centroid.
    d_sq = float(centroid @ centroid)
    inertia_centroid = inertia_origin - total_mass * (
        d_sq * np.eye(3) - np.outer(centroid, centroid)
    )

    return MeshMassProps(
        volume=volume,
        centroid=centroid,
        inertia=inertia_centroid,
        mass=total_mass,
    )


def point_cloud_mass_properties(
    points: np.ndarray, masses: np.ndarray
) -> MeshMassProps:
    """Mass properties of a discrete point cloud, about its own CG.

    Mirrors ``override_distribute.compute_mass_props`` so a seeded cloud can be
    compared directly against the exact mesh result it is meant to represent.
    Point masses have no volume, so ``volume`` is reported as 0.
    """
    p = np.asarray(points, dtype=float)
    m = np.asarray(masses, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (n, 3); got {p.shape}")
    if m.shape != (len(p),):
        raise ValueError(f"masses must be ({len(p)},); got {m.shape}")

    total = float(m.sum())
    if total <= 0:
        raise ValueError(f"total mass must be positive; got {total}")

    cg = (p * m[:, None]).sum(axis=0) / total

    rel = p - cg
    d_sq = np.einsum("ij,ij->i", rel, rel)
    inertia = np.einsum("i,i,jk->jk", m, d_sq, np.eye(3)) - np.einsum(
        "i,ij,ik->jk", m, rel, rel
    )

    return MeshMassProps(volume=0.0, centroid=cg, inertia=inertia, mass=total)
