"""Vehicle CAD import helpers for trajectory simulations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trajectory.vehicle.stp_reader import load_stp


IN_TO_M = 0.0254
LBM_TO_KG = 0.45359237
LBIN2_TO_KGM2 = LBM_TO_KG * IN_TO_M**2
IN3_TO_M3 = IN_TO_M**3
IN2_TO_M2 = IN_TO_M**2


@dataclass(frozen=True)
class VehicleCadModel:
    """CAD-derived vehicle properties in SI units."""

    path: Path
    name: str
    density_lbm_per_in3: float
    dry_mass_kg: float
    cg_m: np.ndarray
    inertia_kg_m2: np.ndarray
    volume_m3: float
    surface_area_m2: float
    reference_area_m2: float
    bounds_m: dict[str, tuple[float, float]]
    mesh_vertices_m: np.ndarray
    mesh_normals: np.ndarray

    @property
    def triangle_count(self) -> int:
        return int(len(self.mesh_vertices_m))


def load_vehicle_cad(
    filepath: str | Path,
    density_lbm_per_in3: float = 0.098,
    tessellation_tolerance: float = 0.5,
) -> VehicleCadModel:
    """Load a STEP vehicle and return simulation-ready SI mass properties."""
    if density_lbm_per_in3 <= 0:
        raise ValueError("Density must be greater than zero to compute vehicle mass.")

    path = Path(filepath)
    reader = load_stp(
        str(path),
        density=density_lbm_per_in3,
        tessellation_tolerance=tessellation_tolerance,
    )

    dry_mass_kg = float(reader.get_mass() * LBM_TO_KG)
    if dry_mass_kg <= 0:
        raise ValueError("CAD import produced zero mass. Check density and model volume.")

    inertia_kg_m2 = np.asarray(reader.get_inertia_tensor(), dtype=float) * LBIN2_TO_KGM2
    inertia_kg_m2 = _ensure_positive_inertia(inertia_kg_m2)

    bounds_in = reader.get_bounds()
    bounds_m = {
        axis: (float(lo * IN_TO_M), float(hi * IN_TO_M))
        for axis, (lo, hi) in bounds_in.items()
    }
    spans_m = np.array([bounds_m[axis][1] - bounds_m[axis][0] for axis in ("x", "y", "z")])

    return VehicleCadModel(
        path=path,
        name=path.name,
        density_lbm_per_in3=float(density_lbm_per_in3),
        dry_mass_kg=dry_mass_kg,
        cg_m=np.asarray(reader.get_cg(), dtype=float) * IN_TO_M,
        inertia_kg_m2=inertia_kg_m2,
        volume_m3=float(reader.get_volume() * IN3_TO_M3),
        surface_area_m2=float(reader.get_surface_area() * IN2_TO_M2),
        reference_area_m2=_estimate_reference_area(spans_m),
        bounds_m=bounds_m,
        mesh_vertices_m=np.asarray(reader.vertices, dtype=float) * IN_TO_M,
        mesh_normals=np.asarray(reader.normals, dtype=float),
    )


def load_saved_vehicle(filepath: str | Path) -> VehicleCadModel:
    """Load a massprops-saved vehicle package."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The pre-rename identifier is still accepted; a manifest on disk
    # outlives the name the program went by when it wrote it.
    if data.get("format") not in ("mudline.vehicle.v1", "rocket_model.vehicle.v1"):
        raise ValueError("Not a supported saved vehicle file.")

    data_dir = _resolve_manifest_path(path.parent, data.get("data_dir", path.parent))
    mesh_path = _resolve_manifest_path(data_dir, data["mesh_npz"])
    props = data["mass_properties"]

    mesh = np.load(mesh_path)
    vertices_in = np.asarray(mesh["vertices"], dtype=float)
    faces = np.asarray(mesh["faces"], dtype=int)
    if len(vertices_in) == 0 or len(faces) == 0:
        raise ValueError("Saved vehicle does not contain an exterior shell mesh.")

    triangles_in = vertices_in[faces]
    triangles_m = triangles_in * IN_TO_M
    normals = _triangle_normals(triangles_m)
    bounds_m = _bounds_from_vertices(vertices_in * IN_TO_M)

    inertia_kg_m2 = np.asarray(props["inertia_lbm_in2"], dtype=float) * LBIN2_TO_KGM2
    inertia_kg_m2 = _ensure_positive_inertia(inertia_kg_m2)

    reference_area_m2 = float(
        data.get("reference_area_m2")
        or data.get("reference_area_in2", 0.0) * IN2_TO_M2
    )
    if reference_area_m2 <= 0:
        spans_m = np.array([bounds_m[axis][1] - bounds_m[axis][0] for axis in ("x", "y", "z")])
        reference_area_m2 = _estimate_reference_area(spans_m)

    return VehicleCadModel(
        path=path,
        name=data.get("name") or path.stem,
        density_lbm_per_in3=float(data.get("density_lbm_per_in3") or 0.0),
        dry_mass_kg=float(props["mass_lbm"] * LBM_TO_KG),
        cg_m=np.asarray(props["cg_in"], dtype=float) * IN_TO_M,
        inertia_kg_m2=inertia_kg_m2,
        volume_m3=float(props["volume_in3"] * IN3_TO_M3),
        surface_area_m2=float(data.get("surface_area_in2", 0.0) * IN2_TO_M2),
        reference_area_m2=reference_area_m2,
        bounds_m=bounds_m,
        mesh_vertices_m=triangles_m,
        mesh_normals=normals,
    )


def _estimate_reference_area(spans_m: np.ndarray) -> float:
    """Estimate frontal area from the two axes perpendicular to the longest axis."""
    valid_spans = spans_m[np.isfinite(spans_m) & (spans_m > 0)]
    if len(valid_spans) < 2:
        return float(np.pi * 0.15**2)

    longest = int(np.argmax(spans_m))
    cross_spans = np.delete(spans_m, longest)
    radius = max(float(np.max(cross_spans)) * 0.5, 0.01)
    return float(np.pi * radius**2)


def _resolve_manifest_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base_dir / path


def _bounds_from_vertices(vertices_m: np.ndarray) -> dict[str, tuple[float, float]]:
    mins = np.min(vertices_m, axis=0)
    maxs = np.max(vertices_m, axis=0)
    return {
        "x": (float(mins[0]), float(maxs[0])),
        "y": (float(mins[1]), float(maxs[1])),
        "z": (float(mins[2]), float(maxs[2])),
    }


def _triangle_normals(triangles_m: np.ndarray) -> np.ndarray:
    edge1 = triangles_m[:, 1] - triangles_m[:, 0]
    edge2 = triangles_m[:, 2] - triangles_m[:, 0]
    normals = np.cross(edge1, edge2)
    lens = np.linalg.norm(normals, axis=1)
    valid = lens > 1e-12
    normals[valid] /= lens[valid, None]
    normals[~valid] = np.array([0.0, 1.0, 0.0])
    return normals


def _ensure_positive_inertia(inertia: np.ndarray) -> np.ndarray:
    """Keep the integrator from seeing a singular CAD inertia tensor."""
    inertia = np.asarray(inertia, dtype=float)
    if inertia.shape != (3, 3):
        raise ValueError("CAD inertia tensor must be 3x3.")

    inertia = 0.5 * (inertia + inertia.T)
    eigvals = np.linalg.eigvalsh(inertia)
    min_eig = float(np.min(eigvals))
    if min_eig <= 1e-9:
        inertia = inertia + np.eye(3) * (1e-9 - min_eig)
    return inertia
