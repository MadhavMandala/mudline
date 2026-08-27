from __future__ import annotations

import gmsh
import numpy as np
import trimesh
from pathlib import Path
from typing import Optional, Tuple

from massprops.model.models import MassProperties
from massprops.io.step_utils import detect_step_unit
from massprops.utils.units import scale_factor_to_inches


def generate_watertight_mesh(
    file_path: Path | str,
    mesh_size: Optional[float] = None,
    mesh_size_factor: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a watertight surface mesh from a STEP file using Gmsh.

    Returns (vertices, faces) where vertices is Nx3 and faces is Mx3.
    Coordinates are converted to inches based on the STEP file's declared unit.
    """
    file_path = Path(file_path)
    native_unit = detect_step_unit(file_path)
    scale = scale_factor_to_inches(native_unit)

    gmsh.initialize()
    if not verbose:
        gmsh.option.setNumber("General.Terminal", 0)

    try:
        gmsh.model.occ.importShapes(str(file_path))
        gmsh.model.occ.synchronize()

        if mesh_size is not None:
            gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)
            gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
        if mesh_size_factor is not None:
            gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)

        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise ValueError("No mesh generated.")

        node_to_index = {tag: idx for idx, tag in enumerate(node_tags)}

        vertices = np.array(node_coords).reshape(-1, 3) * scale

        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        faces = []
        for e_type, e_nodes in zip(elem_types, elem_node_tags):
            if e_type == 2:  # first-order triangles
                e_nodes = np.array(e_nodes)
                tri_indices = np.array([node_to_index[t] for t in e_nodes])
                tri_indices = tri_indices.reshape(-1, 3)
                faces.append(tri_indices)
        if faces:
            faces = np.vstack(faces)
        else:
            faces = np.zeros((0, 3), dtype=int)

        if len(faces) == 0:
            raise ValueError("Mesh generated 0 faces. Try a smaller mesh-size-factor or omit it.")
    finally:
        gmsh.finalize()

    return vertices, faces


def compute_mass_properties(
    vertices: np.ndarray,
    faces: np.ndarray,
    density: float,
) -> MassProperties:
    """Compute mass properties from a mesh and density using trimesh.

    Args:
        vertices: Nx3 array in inches
        faces: Mx3 array of vertex indices
        density: in lbm/in^3

    Returns:
        MassProperties dataclass
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

    if not mesh.is_watertight:
        # Try to repair minor gaps
        mesh.fill_holes()

    if mesh.volume <= 0:
        raise ValueError(f"Mesh volume is {mesh.volume}. Is it a valid solid?")

    mesh.density = density

    props = MassProperties(
        mass=float(mesh.mass),
        cg=mesh.center_mass.copy(),
        inertia=mesh.moment_inertia.copy(),
        volume=float(mesh.volume),
    )
    return props


def mesh_and_compute(
    file_path: Path | str,
    density: float,
    mesh_size: Optional[float] = None,
    mesh_size_factor: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, MassProperties]:
    """Convenience: mesh a STEP file and compute mass properties in one call.

    Returns (vertices, faces, mass_properties).
    """
    vertices, faces = generate_watertight_mesh(
        file_path, mesh_size=mesh_size, mesh_size_factor=mesh_size_factor, verbose=verbose
    )
    props = compute_mass_properties(vertices, faces, density)
    return vertices, faces, props
