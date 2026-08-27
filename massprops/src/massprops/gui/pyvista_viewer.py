from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np
import pyvista as pv

from massprops.model.models import Component


_PART_COLORS = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]


def _hash_color(name: str) -> str:
    """Generate a consistent pastel color from a name."""
    h = hashlib.md5(name.encode()).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    # Lighten to pastel
    r = int(0.5 * r + 0.5 * 255)
    g = int(0.5 * g + 0.5 * 255)
    b = int(0.5 * b + 0.5 * 255)
    return f"#{r:02x}{g:02x}{b:02x}"


def _color_from_palette(idx: int) -> str:
    return _PART_COLORS[idx % len(_PART_COLORS)]


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    """Convert #RRGGBB to (R, G, B) floats in [0, 1]."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def populate_plotter(
    plotter: pv.Plotter,
    root: Component,
    selected: Optional[Component] = None,
    show_edges: bool = True,
    max_tris_per_component: Optional[int] = 50_000,
) -> None:
    """Add Component tree meshes to an existing PyVista plotter.

    Args:
        plotter: PyVista Plotter or QtInteractor to add meshes to.
        root: Root component of the assembly tree.
        selected: If provided, highlights this component in red.
        show_edges: Whether to show mesh edges.
        max_tris_per_component: If a mesh exceeds this many triangles, it is
            decimated by taking every Nth face so the renderer doesn't hang.
            Set to None to disable decimation.
    """
    trace_idx = 0

    def walk(
        node: Component,
        parent_transform: np.ndarray,
    ) -> None:
        nonlocal trace_idx
        world = parent_transform @ node.instance_transform

        if node.mesh_vertices is not None and node.mesh_faces is not None:
            verts = node.mesh_vertices.copy()
            faces = node.mesh_faces.copy()

            # Decimate massive meshes so the GPU doesn't choke
            if max_tris_per_component is not None and len(faces) > max_tris_per_component:
                step = max(1, len(faces) // max_tris_per_component)
                faces = faces[::step]

            # Transform vertices to world space
            ones = np.ones((verts.shape[0], 1))
            verts_h = np.hstack([verts, ones])
            verts_world = (verts_h @ world.T)[:, :3]

            color_hex = _color_from_palette(trace_idx)
            opacity = 0.7
            if selected is not None and node is selected:
                color_hex = "#FF5050"
                opacity = 1.0

            # Build PyVista faces array: [3, i, j, k, 3, i, j, k, ...]
            faces_pv = np.column_stack(
                [np.full(len(faces), 3, dtype=np.int64), faces]
            ).ravel()

            mesh = pv.PolyData(verts_world, faces_pv)
            plotter.add_mesh(
                mesh,
                color=_hex_to_rgb(color_hex),
                opacity=opacity,
                show_edges=show_edges,
                edge_color="black",
                name=f"{node.name}_{trace_idx}",
                pickable=False,
                smooth_shading=False,
            )
            trace_idx += 1

        for child in node.children:
            walk(child, world)

    walk(root, np.eye(4))


def build_plotter(
    root: Component,
    selected: Optional[Component] = None,
    show_edges: bool = True,
    use_legend_groups: bool = False,
    max_tris_per_component: Optional[int] = 50_000,
) -> pv.Plotter:
    """Build a standalone PyVista Plotter from a component tree.

    Args:
        root: Root component of the assembly tree.
        selected: If provided, highlights this component in red.
        show_edges: Whether to show mesh edges.
        use_legend_groups: Ignored (kept for API compatibility).
        max_tris_per_component: Decimation limit per mesh.

    Returns:
        A PyVista Plotter ready to be shown.
    """
    plotter = pv.Plotter()
    plotter.enable_depth_peeling(number_of_peels=8, occlusion_ratio=0.1)
    populate_plotter(plotter, root, selected, show_edges, max_tris_per_component)
    plotter.add_axes()
    plotter.show_bounds(
        grid="back",
        location="outer",
        all_edges=False,
        xtitle="X (in)",
        ytitle="Y (in)",
        ztitle="Z (in)",
    )
    return plotter


def quick_view(
    root: Component,
    title: str = "Assembly Viewer",
    use_legend_groups: bool = False,
) -> None:
    """Build a PyVista plotter from a Component tree and show it."""
    plotter = build_plotter(root, use_legend_groups=use_legend_groups)
    plotter.show(title=title)
