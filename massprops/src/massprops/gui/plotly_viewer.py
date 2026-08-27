from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from typing import Optional

import numpy as np
import plotly.graph_objects as go

from massprops.model.models import Component, Assembly


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
    return f"rgb({r},{g},{b})"


def _color_from_palette(idx: int) -> str:
    return _PART_COLORS[idx % len(_PART_COLORS)]


def build_figure(
    root: Component,
    selected: Optional[Component] = None,
    show_edges: bool = True,
    use_legend_groups: bool = False,
    max_tris_per_component: Optional[int] = 50_000,
) -> go.Figure:
    """Build a Plotly figure from a component tree.

    Each leaf component with a mesh becomes a Mesh3d trace.
    When use_legend_groups is True, child components are grouped under their
    parent's name in the legend (collapsible in Plotly).

    Args:
        max_tris_per_component: If a mesh exceeds this many triangles, it is
            decimated by taking every Nth face so the browser doesn't hang.
            Set to None to disable decimation.
    """
    traces = []
    labels = []

    def walk(
        node: Component,
        parent_transform: np.ndarray,
        parent_name: Optional[str] = None,
        depth: int = 0,
    ):
        world = parent_transform @ node.instance_transform
        group_name = parent_name if (use_legend_groups and parent_name is not None) else node.name

        if node.mesh_vertices is not None and node.mesh_faces is not None:
            verts = node.mesh_vertices.copy()
            faces = node.mesh_faces.copy()

            # Decimate massive meshes so the browser doesn't hang
            if max_tris_per_component is not None and len(faces) > max_tris_per_component:
                step = max(1, len(faces) // max_tris_per_component)
                faces = faces[::step]

            # Transform vertices to world space
            ones = np.ones((verts.shape[0], 1))
            verts_h = np.hstack([verts, ones])
            verts_world = (verts_h @ world.T)[:, :3]

            color = _color_from_palette(len(traces))
            opacity = 0.7
            if selected is not None and node is selected:
                color = "rgb(255, 80, 80)"
                opacity = 1.0

            trace = go.Mesh3d(
                x=verts_world[:, 0],
                y=verts_world[:, 1],
                z=verts_world[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name=node.name,
                color=color,
                opacity=opacity,
                flatshading=show_edges,
                lighting=dict(
                    ambient=0.4,
                    diffuse=0.8,
                    roughness=0.5,
                    specular=0.4,
                    fresnel=0.2,
                ),
                lightposition=dict(x=100, y=200, z=500),
                showscale=False,
                hoverinfo="name",
                showlegend=False,
            )
            traces.append(trace)
            labels.append(node.name)

        for child in node.children:
            walk(child, world, node.name, depth + 1)

    walk(root, np.eye(4))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Assembly Viewer",
        scene=dict(
            aspectmode="data",
            xaxis_title="X (in)",
            yaxis_title="Y (in)",
            zaxis_title="Z (in)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
    )
    return fig


def figure_to_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to a full HTML string for QWebEngineView."""
    return fig.to_html(include_plotlyjs="cdn", full_html=True)


def show_figure_browser(fig: go.Figure, title: str = "Assembly Viewer") -> None:
    """Open a Plotly figure in a dedicated standalone browser app window."""
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".html", delete=False, encoding="utf-8"
    ) as f:
        f.write(html)
        path = f.name

    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\BraveSoftware\Brave-Browser\Application\brave.exe",
    ]
    browser = next((c for c in candidates if os.path.exists(c)), None)

    url = f"file:///{path.replace(os.sep, '/')}"
    if browser:
        subprocess.Popen(
            [
                browser,
                f"--app={url}",
                "--window-size=1280,900",
                "--window-position=100,50",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        import webbrowser

        webbrowser.open(url)


def quick_view(root: Component, title: str = "Assembly Viewer", use_legend_groups: bool = False) -> None:
    """Build a figure from a Component tree and open it in the default browser."""
    fig = build_figure(root, use_legend_groups=use_legend_groups)
    show_figure_browser(fig, title=title)
