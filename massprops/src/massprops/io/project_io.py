from __future__ import annotations

import json
import numpy as np
import trimesh
from pathlib import Path
from typing import Any

from massprops.model.models import Component, Assembly, MassProperties


def _component_to_dict(comp: Component) -> dict[str, Any]:
    """Serialize a component tree to a dict."""
    data: dict[str, Any] = {
        "name": comp.name,
        "type": "assembly" if isinstance(comp, Assembly) else "component",
        "instance_transform": comp.instance_transform.tolist(),
        "density": comp.density,
        "step_product_def_id": comp.step_product_def_id,
        "step_metadata": comp.step_metadata,
    }
    if comp.source_step is not None:
        data["source_step"] = str(comp.source_step)
    if comp.overridden_props is not None:
        data["overridden_props"] = {
            "mass": comp.overridden_props.mass,
            "cg": comp.overridden_props.cg.tolist(),
            "inertia": comp.overridden_props.inertia.tolist(),
            "volume": comp.overridden_props.volume,
        }
        data["override_fields"] = list(comp.override_fields)
    if comp.mass_scale != 1.0:
        data["mass_scale"] = comp.mass_scale
    if comp.computed_props is not None:
        data["computed_props"] = {
            "mass": comp.computed_props.mass,
            "cg": comp.computed_props.cg.tolist(),
            "inertia": comp.computed_props.inertia.tolist(),
            "volume": comp.computed_props.volume,
        }
    data["children"] = [_component_to_dict(c) for c in comp.children]
    return data


def _dict_to_component(data: dict[str, Any]) -> Component:
    """Deserialize a component tree from a dict."""
    is_assembly = data.get("type") == "assembly"
    comp = Assembly(name=data["name"]) if is_assembly else Component(name=data["name"])
    comp.instance_transform = np.array(data.get("instance_transform", np.eye(4).tolist()))
    comp.density = data.get("density")
    comp.step_product_def_id = data.get("step_product_def_id")
    comp.step_metadata = data.get("step_metadata", {})
    if "source_step" in data:
        comp.source_step = Path(data["source_step"])
    if "overridden_props" in data:
        p = data["overridden_props"]
        comp.overridden_props = MassProperties(
            mass=p["mass"],
            cg=np.array(p["cg"]),
            inertia=np.array(p["inertia"]),
            volume=p["volume"],
        )
        comp.override_fields = set(data.get("override_fields", []))
    comp.mass_scale = data.get("mass_scale", 1.0)
    if "computed_props" in data:
        p = data["computed_props"]
        comp.computed_props = MassProperties(
            mass=p["mass"],
            cg=np.array(p["cg"]),
            inertia=np.array(p["inertia"]),
            volume=p["volume"],
        )
    for child_data in data.get("children", []):
        comp.children.append(_dict_to_component(child_data))
    return comp


def save_project(root: Component, project_path: Path, data_dir: Path | None = None) -> None:
    """Save project JSON and export meshes/STLs."""
    project_path = Path(project_path)
    if data_dir is None:
        data_dir = project_path.parent / (project_path.stem + "_data")
    data_dir.mkdir(parents=True, exist_ok=True)

    tree_dict = _component_to_dict(root)

    def export_node(node: Component, prefix: str = "") -> None:
        safe_name = "".join(c if c.isalnum() else "_" for c in node.name) or "part"
        file_stem = f"{prefix}{safe_name}"

        if node.mesh_vertices is not None and node.mesh_faces is not None:
            # Save mesh NPZ
            np.savez(
                data_dir / f"{file_stem}_mesh.npz",
                vertices=node.mesh_vertices,
                faces=node.mesh_faces,
            )
            # Export STL
            mesh = trimesh.Trimesh(vertices=node.mesh_vertices, faces=node.mesh_faces)
            mesh.export(data_dir / f"{file_stem}.stl")

        if node.computed_props is not None or node.overridden_props is not None:
            props = node.effective_props()
            props_dict = {
                "mass": props.mass,
                "cg": props.cg.tolist(),
                "inertia": props.inertia.tolist(),
                "volume": props.volume,
            }
            with open(data_dir / f"{file_stem}_props.json", "w") as f:
                json.dump(props_dict, f, indent=2)

        if node.overridden_props is not None and node.kkt_point_cloud is not None:
            points, masses = node.kkt_point_cloud
            csv_path = data_dir / f"{file_stem}_points.csv"
            rows = np.column_stack([points, masses])
            np.savetxt(csv_path, rows, delimiter=",", header="x,y,z,mass", comments="")

        for idx, child in enumerate(node.children):
            export_node(child, f"{prefix}{idx}_")

    export_node(root)

    project_data = {
        "tree": tree_dict,
        "data_dir": str(data_dir),
    }
    with open(project_path, "w") as f:
        json.dump(project_data, f, indent=2)


def save_vehicle(root: Component, vehicle_path: Path, data_dir: Path | None = None) -> Path:
    """Save a trajectory-ready vehicle package.

    The package is a small manifest JSON plus a data folder containing:
    - one combined exterior shell mesh as NPZ and STL
    - aggregate mass properties
    - the full massprops project export for later inspection
    """
    vehicle_path = Path(vehicle_path)
    if vehicle_path.suffix.lower() != ".json":
        vehicle_path = vehicle_path.with_suffix(".vehicle.json")
    if vehicle_path.name.endswith(".json") and not vehicle_path.name.endswith(".vehicle.json"):
        vehicle_path = vehicle_path.with_name(f"{vehicle_path.stem}.vehicle.json")

    if data_dir is None:
        data_dir = vehicle_path.parent / f"{_vehicle_stem(vehicle_path)}_vehicle_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    project_path = data_dir / "massprops_project.json"
    save_project(root, project_path, data_dir=data_dir / "parts")

    vertices, faces = _combined_mesh(root)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("No mesh data found. Open/mesh a STEP vehicle before saving.")

    mesh_npz = data_dir / "vehicle_shell_mesh.npz"
    mesh_stl = data_dir / "vehicle_shell.stl"
    np.savez(mesh_npz, vertices=vertices, faces=faces)
    trimesh.Trimesh(vertices=vertices, faces=faces, process=False).export(mesh_stl)

    props = root.effective_props()
    props_data = {
        "mass_lbm": float(props.mass),
        "cg_in": np.asarray(props.cg, dtype=float).tolist(),
        "inertia_lbm_in2": np.asarray(props.inertia, dtype=float).tolist(),
        "volume_in3": float(props.volume),
    }
    props_path = data_dir / "vehicle_mass_properties.json"
    with open(props_path, "w", encoding="utf-8") as f:
        json.dump(props_data, f, indent=2)

    bounds = _mesh_bounds(vertices)
    manifest = {
        "format": "mudline.vehicle.v1",
        "name": root.name or vehicle_path.stem,
        "data_dir": data_dir.name,
        "mesh_npz": mesh_npz.name,
        "mesh_stl": mesh_stl.name,
        "mass_properties_json": props_path.name,
        "massprops_project": str(project_path.relative_to(data_dir)),
        "mass_properties": props_data,
        "bounds_in": bounds,
        "surface_area_in2": _surface_area(vertices, faces),
        "reference_area_in2": _estimate_reference_area_in2(bounds),
    }

    with open(vehicle_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return vehicle_path


def _vehicle_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".vehicle.json"):
        return name[: -len(".vehicle.json")]
    return path.stem


def _combined_mesh(root: Component) -> tuple[np.ndarray, np.ndarray]:
    vertices_out: list[np.ndarray] = []
    faces_out: list[np.ndarray] = []

    def walk(node: Component, parent_transform: np.ndarray) -> None:
        world = parent_transform @ node.instance_transform

        if node.mesh_vertices is not None and node.mesh_faces is not None:
            verts = np.asarray(node.mesh_vertices, dtype=float)
            faces = np.asarray(node.mesh_faces, dtype=int)
            if len(verts) > 0 and len(faces) > 0:
                ones = np.ones((verts.shape[0], 1))
                verts_world = (np.hstack([verts, ones]) @ world.T)[:, :3]
                offset = sum(len(v) for v in vertices_out)
                vertices_out.append(verts_world)
                faces_out.append(faces + offset)

        for child in node.children:
            walk(child, world)

    walk(root, np.eye(4))
    if not vertices_out or not faces_out:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.vstack(vertices_out), np.vstack(faces_out)


def _mesh_bounds(vertices: np.ndarray) -> dict[str, list[float]]:
    mins = np.min(vertices, axis=0)
    maxs = np.max(vertices, axis=0)
    return {
        "x": [float(mins[0]), float(maxs[0])],
        "y": [float(mins[1]), float(maxs[1])],
        "z": [float(mins[2]), float(maxs[2])],
    }


def _surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    return float(0.5 * np.linalg.norm(np.cross(edge1, edge2), axis=1).sum())


def _estimate_reference_area_in2(bounds: dict[str, list[float]]) -> float:
    spans = np.array([
        bounds["x"][1] - bounds["x"][0],
        bounds["y"][1] - bounds["y"][0],
        bounds["z"][1] - bounds["z"][0],
    ])
    valid_spans = spans[np.isfinite(spans) & (spans > 0)]
    if len(valid_spans) < 2:
        return float(np.pi * 5.9**2)

    longest = int(np.argmax(spans))
    cross_spans = np.delete(spans, longest)
    radius = max(float(np.max(cross_spans)) * 0.5, 0.01)
    return float(np.pi * radius**2)


def load_project(project_path: Path) -> Component:
    """Load a project from JSON and restore meshes if available."""
    project_path = Path(project_path)
    with open(project_path, "r") as f:
        project_data = json.load(f)

    data_dir = Path(project_data.get("data_dir", project_path.parent / (project_path.stem + "_data")))
    root = _dict_to_component(project_data["tree"])

    def restore_node(node: Component, prefix: str = "") -> None:
        safe_name = "".join(c if c.isalnum() else "_" for c in node.name) or "part"
        file_stem = f"{prefix}{safe_name}"

        mesh_path = data_dir / f"{file_stem}_mesh.npz"
        if mesh_path.exists():
            data = np.load(mesh_path)
            node.mesh_vertices = data["vertices"]
            node.mesh_faces = data["faces"]

        for idx, child in enumerate(node.children):
            restore_node(child, f"{prefix}{idx}_")

    restore_node(root)
    return root
