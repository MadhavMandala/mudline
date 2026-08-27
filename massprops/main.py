from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path

# Ensure src is on path when running directly
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))


def _run_qt(
    step_path: Path | None = None,
    vehicle_handoff_path: Path | None = None,
) -> int:
    from PySide6.QtWidgets import QApplication 
    from massprops.gui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow(vehicle_handoff_path=vehicle_handoff_path)

    if step_path is not None and step_path.exists():
        window._load_step(step_path)

    window.show()
    return app.exec()


def _mesh_tree(
    node,
    cache,
    mesh_size_factor: float | None = None,
    verbose: bool = False,
    compute_props: bool = True,
) -> None:
    """Recursively mesh every component that has a source STEP file."""
    from massprops.mesh.mesh_cache import MeshCache

    if node.source_step and Path(node.source_step).exists():
        if not cache.load_cached(node):
            try:
                if verbose:
                    print(f"  Meshing {node.name} ...", end=" ", flush=True)
                MeshCache.mesh_component(
                    node,
                    mesh_size_factor=mesh_size_factor,
                    compute_props=compute_props,
                )
                if verbose:
                    tris = len(node.mesh_faces) if node.mesh_faces is not None else 0
                    print(f"({tris:,} triangles)")
                cache.save_cached(node)
            except Exception as exc:
                if verbose:
                    print(f"FAILED — {exc}")
                else:
                    print(f"Warning: could not mesh {node.name}: {exc}")
    for child in node.children:
        _mesh_tree(child, cache, mesh_size_factor, verbose, compute_props)


def _count_tris(node) -> int:
    total = len(node.mesh_faces) if node.mesh_faces is not None else 0
    for child in node.children:
        total += _count_tris(child)
    return total


def _run_pyvista_full(
    step_path: Path,
    mesh_size_factor: float | None = None,
    verbose: bool = False,
) -> int:
    """Full assembly path: parses STEP hierarchy, transforms, materials, mass props."""
    from massprops.io.assembly_loader import load_assembly, load_from_folder
    from massprops.io.material_extractor import apply_materials_to_tree
    from massprops.io.step_parser import StepParser
    from massprops.mesh.mesh_cache import MeshCache
    from massprops.gui.pyvista_viewer import quick_view

    if step_path.is_dir():
        root, master_path = load_from_folder(step_path)
    else:
        root = load_assembly(step_path)
        master_path = step_path

    apply_materials_to_tree(root, StepParser(master_path))

    cache = MeshCache(project_root / "data")
    if verbose:
        print("Meshing assembly (full mode) ...")
    _mesh_tree(root, cache, mesh_size_factor=mesh_size_factor, verbose=verbose, compute_props=True)

    total_tris = _count_tris(root)
    if verbose:
        print(f"Total triangles: {total_tris:,}")

    if total_tris > 500_000:
        print(
            f"WARNING: {total_tris:,} triangles is very large for the viewer. "
            "The viewer may be slow or unresponsive."
        )
        print("Tip: use --mesh-size-factor 10.0 (or larger) for a coarser mesh.")
    elif total_tris == 0:
        print("WARNING: No mesh data found.")

    quick_view(root, title=f"Assembly Viewer — {master_path.name}")
    return 0


_PART_COLORS = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]


def _vname_to_display(vname: str) -> str:
    """'X1-D551101-01_-@2' → 'X1-D551101-01_2',  'X1-D551071-01_-' → 'X1-D551071-01'"""
    if "@" in vname:
        base, idx = vname.rsplit("@", 1)
        import re
        return re.sub(r"_.*$", "", base) + f"_{idx}"
    import re
    return re.sub(r"_.*$", "", vname)


def _build_assembly_tree(folder_path: Path | str):
    """Parses NAUO entities in each STEP file to build the assembly hierarchy."""
    import re

    children = {}
    name_to_file = {}
    all_original_child_names = set()

    for file_path in glob.glob(os.path.join(folder_path, "*.stp")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        nauo_names = re.findall(
            r"NEXT_ASSEMBLY_USAGE_OCCURRENCE\s*\(\s*'([^']+)'", content
        )
        if not nauo_names:
            continue

        all_products = list(dict.fromkeys(
            re.findall(r"=PRODUCT\s*\(\s*'([^']+)'", content)
        ))
        child_set = set(nauo_names)
        parent = next((p for p in all_products if p not in child_set), None)
        if parent is None:
            continue

        counts = {}
        for n in nauo_names:
            counts[n] = counts.get(n, 0) + 1

        seen = {}
        virtual_children = []
        for n in nauo_names:
            if counts[n] > 1:
                seen[n] = seen.get(n, 0) + 1
                vname = f"{n}@{seen[n]}"
            else:
                vname = n
            virtual_children.append(vname)
            name_to_file[vname] = n

        name_to_file[parent] = parent
        children[parent] = virtual_children
        all_original_child_names.update(nauo_names)

    roots = set(children) - all_original_child_names
    root = roots.pop() if roots else next(iter(children), None)
    return root, children, name_to_file


def _build_parent_map(node, children_dict, parent=None, result=None):
    if result is None:
        result = {}
    result[node] = parent
    for child in children_dict.get(node, []):
        _build_parent_map(child, children_dict, node, result)
    return result


def _dfs_order(node, children, depth=0):
    result = [(node, depth)]
    for child in children.get(node, []):
        result.extend(_dfs_order(child, children, depth + 1))
    return result


def _run_pyvista_fast(
    step_path: Path,
    mesh_size_factor: float | None = None,
    verbose: bool = False,
) -> int:
    """Fast path: directly mesh file(s) with Gmsh, skip STEP parsing and trimesh."""
    from massprops.mesh.mesher import generate_watertight_mesh
    import pyvista as pv
    import numpy as np

    # Gather file list
    if step_path.is_dir():
        stp_files = sorted(
            glob.glob(str(step_path / "*.stp")) + glob.glob(str(step_path / "*.step"))
        )
        if not stp_files:
            print(f"ERROR: No .stp/.step files found in: {step_path}")
            return 1
    else:
        if not step_path.exists():
            print(f"ERROR: File not found: {step_path}")
            return 1
        stp_files = [str(step_path)]

    # Fast mesh every file independently
    parts: list[dict] = []
    for fp in stp_files:
        name = Path(fp).stem
        if verbose:
            print(f"Meshing {name} ...", end=" ", flush=True)
        try:
            verts, faces = generate_watertight_mesh(
                fp, mesh_size_factor=mesh_size_factor
            )
            if verbose:
                print(f"({len(faces):,} triangles)")
        except Exception as exc:
            if mesh_size_factor is not None:
                if verbose:
                    print(f"FAILED with factor={mesh_size_factor}, retrying default ...", end=" ", flush=True)
                try:
                    verts, faces = generate_watertight_mesh(fp)
                    if verbose:
                        print(f"({len(faces):,} triangles)")
                except Exception as exc2:
                    if verbose:
                        print(f"FAILED — {exc2}")
                    continue
            else:
                if verbose:
                    print(f"FAILED — {exc}")
                continue

        parts.append({
            "name": name,
            "verts": verts,
            "faces": faces,
        })

    if not parts:
        print("ERROR: No parts could be meshed.")
        return 1

    # Try to build an assembly tree for ordering (same-folder only)
    tree = None
    if step_path.is_dir():
        try:
            tree = _build_assembly_tree(step_path)
        except Exception:
            tree = None

    part_by_name = {p["name"]: p for p in parts}

    if tree is not None and tree[0] is not None:
        root, children_dict, name_to_file = tree
        ordered = _dfs_order(root, children_dict)
        parent_map = _build_parent_map(root, children_dict)
        referenced = set(name_to_file.values())
        for name, part in part_by_name.items():
            if name not in referenced:
                ordered.append((name, 0))
                parent_map[name] = None
                name_to_file[name] = name
    else:
        ordered = [(p["name"], 0) for p in parts]
        name_to_file = {p["name"]: p["name"] for p in parts}
        parent_map = {p["name"]: None for p in parts}

    plotter = pv.Plotter()
    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.1)

    for color_idx, (vname, depth) in enumerate(ordered):
        file_stem = name_to_file.get(vname, vname)
        part = part_by_name.get(file_stem)
        if part is None:
            continue

        verts = part["verts"]
        faces = part["faces"]

        # Decimate massive meshes for GPU safety
        max_tris = 50_000
        if len(faces) > max_tris:
            step = max(1, len(faces) // max_tris)
            faces = faces[::step]

        faces_pv = np.column_stack(
            [np.full(len(faces), 3, dtype=np.int64), faces]
        ).ravel()
        mesh = pv.PolyData(verts, faces_pv)

        color_hex = _PART_COLORS[color_idx % len(_PART_COLORS)]
        r = int(color_hex[1:3], 16) / 255.0
        g = int(color_hex[3:5], 16) / 255.0
        b = int(color_hex[5:7], 16) / 255.0

        plotter.add_mesh(
            mesh,
            color=(r, g, b),
            opacity=0.7,
            show_edges=True,
            edge_color="black",
            name=_vname_to_display(vname),
            pickable=False,
            smooth_shading=False,
        )

    plotter.add_axes()
    plotter.show_bounds(
        grid="back",
        location="outer",
        all_edges=False,
        xtitle="X (in)",
        ytitle="Y (in)",
        ztitle="Z (in)",
    )
    plotter.show(title=f"Assembly Viewer — {step_path.name}")
    return 0


def _pick_path() -> Path | None:
    """Show a GUI picker dialog for a file or folder. Returns None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    # Ask user whether they want a file or a folder
    choice = tk.messagebox.askyesnocancel(
        "MassProp Viewer",
        "Do you want to open a folder of STEP files?\n\n"
        "Yes = Pick a folder\n"
        "No = Pick a single file\n"
        "Cancel = Exit",
    )
    if choice is None:
        return None
    if choice:
        folder = filedialog.askdirectory(title="Select folder containing .stp files")
        root.destroy()
        return Path(folder) if folder else None
    else:
        file_path = filedialog.askopenfilename(
            title="Select STEP file",
            filetypes=[("STEP files", "*.stp *.step"), ("All files", "*.*")],
        )
        root.destroy()
        return Path(file_path) if file_path else None


def main() -> int:
    parser = argparse.ArgumentParser(description="MassProp — STEP assembly viewer")
    parser.add_argument(
        "path",
        nargs="?",
        help="Path to .stp/.step file or folder containing .stp files",
    )
    parser.add_argument(
        "--plotly",
        action="store_true",
        help="Open in PyVista viewer instead of Qt GUI",
    )
    parser.add_argument(
        "--full-assembly",
        action="store_true",
        help="Use full assembly loader with transforms and mass properties (slower, more accurate)",
    )
    parser.add_argument(
        "--mesh-size-factor",
        type=float,
        default=None,
        help="Mesh coarseness factor for visualization (default: None = Gmsh default). "
        "1.0 = default Gmsh size, 2.0 = coarser, 5.0 = very coarse, 0.5 = finer.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress during meshing and assembly loading",
    )
    parser.add_argument(
        "--vehicle-handoff",
        default=None,
        help="Path where Save Vehicle writes the saved vehicle manifest for the trajectory app.",
    )
    args = parser.parse_args()

    step_path = Path(args.path) if args.path else None

    if args.plotly:
        if step_path is None:
            step_path = _pick_path()
            if step_path is None:
                print("No path selected. Exiting.")
                return 1
        if not step_path.exists():
            print(f"Error: Path does not exist: {step_path}")
            return 1
        if args.full_assembly:
            return _run_pyvista_full(
                step_path,
                mesh_size_factor=args.mesh_size_factor,
                verbose=args.verbose,
            )
        return _run_pyvista_fast(
            step_path,
            mesh_size_factor=args.mesh_size_factor,
            verbose=args.verbose,
        )

    handoff_path = Path(args.vehicle_handoff) if args.vehicle_handoff else None
    return _run_qt(step_path, vehicle_handoff_path=handoff_path)


if __name__ == "__main__":
    sys.exit(main())
