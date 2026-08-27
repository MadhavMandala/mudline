"""STL file reader with Plotly visualization for debugging."""

from __future__ import annotations

import struct
import numpy as np


class STLReader:
    """Reads binary and ASCII STL files, returns mesh data with visualization."""

    def __init__(self):
        self.vertices = np.array([])  # Shape (N, 3, 3) - N triangles, 3 verts each
        self.normals = np.array([])  # Shape (N, 3) - facet normals
        self.triangles = None  # Legacy alias for vertices

    @classmethod
    def load(cls, filepath: str) -> "STLReader":
        """Load an STL file (auto-detects binary or ASCII)."""
        reader = cls()

        with open(filepath, 'rb') as f:
            header = f.read(80)

            # Binary STL: first 80 bytes are header, then 4-byte triangle count
            # ASCII STL: starts with "solid "
            if header.startswith(b'solid '):
                f.seek(0)
                try:
                    reader._read_ascii(f)
                except Exception:
                    # Some binary files incorrectly have "solid" in header
                    f.seek(0)
                    reader._read_binary(f)
            else:
                f.seek(0)
                reader._read_binary(f)

        reader.triangles = reader.vertices  # Backward compatibility alias
        return reader

    def _read_binary(self, f):
        """Read binary STL format."""
        f.read(80)  # Skip header

        num_triangles = struct.unpack('<I', f.read(4))[0]

        all_vertices = []
        all_normals = []

        for _ in range(num_triangles):
            # Each triangle: normal (3 floats), 3 vertices (9 floats), attribute (1 short)
            data = struct.unpack('<12fH', f.read(50))

            normal = np.array(data[0:3])
            v0 = np.array(data[3:6])
            v1 = np.array(data[6:9])
            v2 = np.array(data[9:12])

            all_normals.append(normal)
            all_vertices.append([v0, v1, v2])

        self.vertices = np.array(all_vertices)
        self.normals = np.array(all_normals)

    def _read_ascii(self, f):
        """Read ASCII STL format."""
        import re

        content = f.read().decode('utf-8', errors='ignore')

        all_vertices = []
        all_normals = []

        # Find all facet...endfacet blocks
        facet_pattern = re.compile(
            r'facet normal\s+([\d\-.eE]+)\s+([\d\-.eE]+)\s+([\d\-.eE]+)\s+'
            r'outer loop\s+'
            r'vertex\s+([\d\-.eE]+)\s+([\d\-.eE]+)\s+([\d\-.eE]+)\s+'
            r'vertex\s+([\d\-.eE]+)\s+([\d\-.eE]+)\s+([\d\-.eE]+)\s+'
            r'vertex\s+([\d\-.eE]+)\s+([\d\-.eE]+)\s+([\d\-.eE]+)\s+'
            r'endloop\s+'
            r'endfacet',
            re.IGNORECASE
        )

        for match in facet_pattern.finditer(content):
            vals = [float(m) for m in match.groups()]
            normal = np.array(vals[0:3])
            v0 = np.array(vals[3:6])
            v1 = np.array(vals[6:9])
            v2 = np.array(vals[9:12])

            all_normals.append(normal)
            all_vertices.append([v0, v1, v2])

        self.vertices = np.array(all_vertices)
        self.normals = np.array(all_normals)

    def get_surface_area(self) -> float:
        """Calculate total surface area."""
        if len(self.vertices) == 0:
            return 0.0

        # Cross product of two edges gives area of parallelogram, half is triangle area
        edges1 = self.vertices[:, 1] - self.vertices[:, 0]
        edges2 = self.vertices[:, 2] - self.vertices[:, 0]
        cross_products = np.cross(edges1, edges2)
        areas = 0.5 * np.linalg.norm(cross_products, axis=1)
        return np.sum(areas)

    def get_bounds(self) -> dict:
        """Get bounding box."""
        if len(self.vertices) == 0:
            return {'x': (0, 0), 'y': (0, 0), 'z': (0, 0)}

        flat = self.vertices.reshape(-1, 3)
        mins = np.min(flat, axis=0)
        maxs = np.max(flat, axis=0)

        return {
            'x': (mins[0], maxs[0]),
            'y': (mins[1], maxs[1]),
            'z': (mins[2], maxs[2])
        }

    def visualize(self, show_normals: bool = False, auto_show: bool = True) -> go.Figure:
        """
        Plotly 3D visualization for debugging.

        Args:
            show_normals: Draw surface normals as arrows
            auto_show: If True, calls fig.show() automatically

        Returns:
            Plotly figure object
        """
        import plotly.graph_objects as go
        if len(self.vertices) == 0:
            raise ValueError("No mesh data loaded. Call load() first.")

        # Flatten vertices for mesh3d
        flat_verts = self.vertices.reshape(-1, 3)

        # Create indices for triangles
        num_tris = len(self.vertices)
        i = np.arange(0, num_tris * 3, 3)
        j = np.arange(1, num_tris * 3, 3)
        k = np.arange(2, num_tris * 3, 3)

        fig = go.Figure(data=[go.Mesh3d(
            x=flat_verts[:, 0],
            y=flat_verts[:, 1],
            z=flat_verts[:, 2],
            i=i, j=j, k=k,
            opacity=0.7,
            colorscale='Viridis',
            intensity=flat_verts[:, 2],  # Color by Z height for depth cue
            showscale=False,
            name='Surface'
        )])

        # Add wireframe overlay (triangle edges)
        edge_traces = []
        for tri in self.vertices:
            # Close the triangle
            x = [tri[0, 0], tri[1, 0], tri[2, 0], tri[0, 0], None]
            y = [tri[0, 1], tri[1, 1], tri[2, 1], tri[0, 1], None]
            z = [tri[0, 2], tri[1, 2], tri[2, 2], tri[0, 2], None]

            edge_traces.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines',
                line=dict(color='black', width=1),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add normals
        if show_normals:
            centroids = np.mean(self.vertices, axis=1)
            scale = self.get_bounds()
            diag = np.sqrt(sum((scale[axis][1] - scale[axis][0])**2 for axis in ['x', 'y', 'z']))
            normal_scale = diag * 0.05  # 5% of bounding diagonal

            for centroid, normal in zip(centroids, self.normals):
                normal = normal / (np.linalg.norm(normal) + 1e-10)
                end = centroid + normal * normal_scale

                edge_traces.append(go.Scatter3d(
                    x=[centroid[0], end[0]],
                    y=[centroid[1], end[1]],
                    z=[centroid[2], end[2]],
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ))

        for trace in edge_traces:
            fig.add_trace(trace)

        # Equal aspect ratio
        bounds = self.get_bounds()
        max_range = max(
            bounds['x'][1] - bounds['x'][0],
            bounds['y'][1] - bounds['y'][0],
            bounds['z'][1] - bounds['z'][0]
        ) * 0.5

        center_x = (bounds['x'][0] + bounds['x'][1]) / 2
        center_y = (bounds['y'][0] + bounds['y'][1]) / 2
        center_z = (bounds['z'][0] + bounds['z'][1]) / 2

        fig.update_layout(
            title=f"STL Mesh ({num_tris:,} triangles)",
            scene=dict(
                xaxis=dict(range=[center_x - max_range, center_x + max_range], title='X'),
                yaxis=dict(range=[center_y - max_range, center_y + max_range], title='Y'),
                zaxis=dict(range=[center_z - max_range, center_z + max_range], title='Z'),
                aspectmode='cube'
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, b=0, t=30)
        )

        if auto_show:
            fig.show()

        return fig


def load_stl(filepath: str) -> STLReader:
    """Convenience function: load_stl('file.stl') -> STLReader instance."""
    return STLReader.load(filepath)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        script_dir = Path(__file__).parent
        filepath = script_dir / "test_cube(1).stl"
        print(f"No file specified, loading default: {filepath}")

    reader = load_stl(filepath)
    print(f"Loaded {len(reader.vertices)} triangles")
    print(f"Surface area: {reader.get_surface_area():.4f}")
    bounds = reader.get_bounds()
    print(f"Bounds: {bounds}")
    reader.visualize()
