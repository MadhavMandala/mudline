"""Watertightness checks and winding repair for triangle meshes.

``massprops.mesh.mesher`` already produces watertight meshes by construction --
Gmsh meshes the STEP B-rep directly rather than tessellating to a tolerance, so
the topology is sound before it reaches Python. This module exists for the cases
that bypass that path: meshes loaded from STL, meshes stitched from several
sources, and anything handed over by a tool whose guarantees are unknown.

The checks are cheap and the failure they guard against is expensive:
``mesh_props.mass_properties`` and the interior sampler both assume a closed,
outward-wound surface, and a leaky mesh yields a plausible-looking wrong number
rather than an error.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class QualityReport(NamedTuple):
    """Topological state of a mesh."""

    n_vertices: int
    n_faces: int
    watertight: bool
    consistently_wound: bool
    boundary_edges: int          # edges used by exactly one face (holes)
    nonmanifold_edges: int       # edges used by three or more faces
    degenerate_faces: int        # zero-area or repeated-index faces
    duplicate_vertices: int
    signed_volume: float

    def summary(self) -> str:
        state = "watertight" if self.watertight else "NOT watertight"
        wound = "consistent" if self.consistently_wound else "INCONSISTENT"
        return (
            f"{self.n_vertices} verts, {self.n_faces} faces, {state}, "
            f"winding {wound}, boundary={self.boundary_edges}, "
            f"nonmanifold={self.nonmanifold_edges}, "
            f"degenerate={self.degenerate_faces}, volume={self.signed_volume:.6g}"
        )


def _undirected_edges(faces: np.ndarray) -> np.ndarray:
    """All (m*3, 2) undirected edges, each row sorted ascending."""
    f = np.asarray(faces, dtype=int)
    edges = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    return np.sort(edges, axis=1)


def _directed_edges(faces: np.ndarray) -> np.ndarray:
    """All (m*3, 2) directed edges in winding order."""
    f = np.asarray(faces, dtype=int)
    return np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)


def edge_counts(faces: np.ndarray) -> tuple[int, int]:
    """Return (boundary_edge_count, nonmanifold_edge_count)."""
    if len(faces) == 0:
        return 0, 0
    edges = _undirected_edges(faces)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int((counts == 1).sum()), int((counts >= 3).sum())


def is_watertight(faces: np.ndarray) -> bool:
    """True when every edge is shared by exactly two faces."""
    boundary, nonmanifold = edge_counts(faces)
    return boundary == 0 and nonmanifold == 0


def is_consistently_wound(faces: np.ndarray) -> bool:
    """True when each shared edge is traversed in opposite directions.

    A consistently wound closed mesh visits every undirected edge exactly twice,
    once in each direction, so the set of directed edges contains no duplicates.
    """
    if len(faces) == 0:
        return True
    directed = _directed_edges(faces)
    _, counts = np.unique(directed, axis=0, return_counts=True)
    return bool((counts == 1).all())


def find_degenerate_faces(
    vertices: np.ndarray, faces: np.ndarray, area_tol: float = 1e-12
) -> np.ndarray:
    """Boolean mask of faces that repeat a vertex index or have ~zero area."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    if len(f) == 0:
        return np.zeros(0, dtype=bool)

    repeated = (
        (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 0] == f[:, 2])
    )
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    twice_area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return repeated | (twice_area <= area_tol)


def merge_duplicate_vertices(
    vertices: np.ndarray, faces: np.ndarray, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray, int]:
    """Weld vertices closer than ``tol``, reindexing faces.

    Rounds to a lattice of size ``tol`` rather than doing a true radius query --
    fast, deterministic, and sufficient for welding the split seams that cause
    false boundary edges. Returns (vertices, faces, n_merged).
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    if len(v) == 0:
        return v, f, 0

    keys = np.round(v / tol).astype(np.int64)
    _, first_index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True
    )
    inverse = inverse.reshape(-1)

    # Keep the coordinates of the first occurrence of each welded group, in the
    # order np.unique produced, so face indices line up with `inverse`.
    new_vertices = v[first_index]
    order = np.argsort(first_index)
    remap = np.empty(len(first_index), dtype=int)
    remap[order] = np.arange(len(first_index))
    new_vertices = new_vertices[order]
    new_faces = remap[inverse][f]

    return new_vertices, new_faces, int(len(v) - len(new_vertices))


def orient_consistently(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, int]:
    """Flood-fill face orientation so neighbours traverse shared edges oppositely.

    Then flips the whole surface if the resulting signed volume is negative, so
    the result is outward-wound. Disconnected shells are each visited.

    Returns (faces, n_flipped).
    """
    f = np.asarray(faces, dtype=int).copy()
    if len(f) == 0:
        return f, 0

    # Undirected edge -> the faces that use it. A dict keeps the flood fill
    # readable; the mesh sizes this runs on (a part, not an assembly) make the
    # Python-level loop a non-issue next to Gmsh meshing itself.
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for face_index, tri in enumerate(f):
        for u, w in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_to_faces.setdefault((min(u, w), max(u, w)), []).append(face_index)

    visited = np.zeros(len(f), dtype=bool)
    flipped = 0

    def traverses(face_index: int, u: int, w: int) -> bool:
        """True if this face walks the edge u->w in its own winding order."""
        tri = f[face_index]
        return bool(
            (tri[0] == u and tri[1] == w)
            or (tri[1] == u and tri[2] == w)
            or (tri[2] == u and tri[0] == w)
        )

    for seed in range(len(f)):
        if visited[seed]:
            continue
        visited[seed] = True
        stack = [seed]
        while stack:
            current = stack.pop()
            tri = f[current]
            for u, w in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                for neighbour in edge_to_faces.get((min(u, w), max(u, w)), ()):
                    if neighbour == current or visited[neighbour]:
                        continue
                    # A correctly oriented neighbour walks this edge the other
                    # way (w->u). Walking it the same way means it is flipped.
                    if traverses(neighbour, u, w):
                        f[neighbour] = f[neighbour][::-1]
                        flipped += 1
                    visited[neighbour] = True
                    stack.append(neighbour)

    # Make the surface outward-wound overall.
    from massprops.solver.mesh_props import signed_volume

    if signed_volume(vertices, f) < 0:
        f = f[:, ::-1]
        flipped = len(f) - flipped

    return f, flipped


def inspect(vertices: np.ndarray, faces: np.ndarray) -> QualityReport:
    """Report a mesh's topological state without modifying it."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    boundary, nonmanifold = edge_counts(f)

    keys = np.round(v / 1e-9).astype(np.int64) if len(v) else np.zeros((0, 3), np.int64)
    n_unique = len(np.unique(keys, axis=0)) if len(v) else 0

    try:
        volume = float(
            np.einsum(
                "ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])
            ).sum()
            / 6.0
        ) if len(f) else 0.0
    except (IndexError, ValueError):
        volume = float("nan")

    return QualityReport(
        n_vertices=len(v),
        n_faces=len(f),
        watertight=(boundary == 0 and nonmanifold == 0),
        consistently_wound=is_consistently_wound(f),
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        degenerate_faces=int(find_degenerate_faces(v, f).sum()),
        duplicate_vertices=int(len(v) - n_unique),
        signed_volume=volume,
    )


def repair(
    vertices: np.ndarray,
    faces: np.ndarray,
    weld_tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, QualityReport, QualityReport]:
    """Weld duplicate vertices, drop degenerate faces, fix winding.

    Does not fill holes -- a hole is a genuine modelling defect and closing it
    silently would fabricate geometry. The returned report says whether the mesh
    is watertight; the caller decides what to do about it.

    Returns (vertices, faces, report_before, report_after).
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    before = inspect(v, f)

    v, f, _ = merge_duplicate_vertices(v, f, tol=weld_tol)

    keep = ~find_degenerate_faces(v, f)
    f = f[keep]

    if len(f):
        f, _ = orient_consistently(v, f)

    after = inspect(v, f)
    return v, f, before, after
