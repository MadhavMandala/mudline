"""STEP file reader with unit conversion, CG, and inertia tensor calculation."""

from __future__ import annotations

import numpy as np
from pathlib import Path

# Initialize unit registry globally
_UREG = None


def _unit_registry():
    """pint, loaded on first use: it is cad-extra, and only units need it."""
    global _UREG
    if _UREG is None:
        from pint import UnitRegistry
        _UREG = UnitRegistry()
    return _UREG


class STPReader:
    """Reads STEP (.stp/.step) files, returns mesh data in inches with mass properties."""

    def __init__(self):
        self.vertices = np.array([])  # Shape (N, 3, 3) - N triangles, 3 verts each
        self.normals = np.array([])   # Shape (N, 3) - facet normals
        self.triangles = None         # Legacy alias for vertices
        self.original_unit = 'MM'     # Default assumption
        self.scale_factor = 1.0       # Applied to convert to inches
        self.cg = np.array([0.0, 0.0, 0.0])  # Center of gravity in inches
        self.volume = 0.0             # Volume in cubic inches
        self.mass = 0.0               # Mass (requires density assumption)
        self.inertia_tensor = np.zeros((3, 3))  # About CG, in lb-in^2 (assuming density)
        self.density = 0.0            # lb/in^3 - set by user
        self._cq_shape = None         # Store cadquery shape for additional processing

    @classmethod
    def load(
        cls,
        filepath: str,
        density: float = None,
        tessellation_tolerance: float = 0.5,
    ) -> "STPReader":
        """
        Load a STEP file and tessellate geometry using CadQuery.

        Args:
            filepath: Path to .stp or .step file
            density: Material density in lb/in^3 (optional, for mass calculations)
                    Common values: Aluminum ~0.098, Steel ~0.283, Titanium ~0.163
            tessellation_tolerance: OpenCascade mesh tolerance in millimeters.
        """
        import cadquery as cq
        from cadquery.occ_impl.shapes import Shape
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp

        reader = cls()
        reader.density = density if density is not None else 0.0

        # Load STEP file using CadQuery
        try:
            result = cq.importers.importStep(str(filepath))
            reader._cq_shape = result
        except Exception as e:
            raise ValueError(f"Failed to read STEP file: {filepath} - {e}")

        # Get the underlying OCCT shape
        shape = result.val().wrapped

        if shape.IsNull():
            raise ValueError("No valid geometry found in STEP file")

        # CadQuery/OpenCascade normalizes imported STEP geometry to millimeters.
        # Convert the OCCT-native values to the reader's inch-based API.
        reader._set_scale_factor('MM')

        # Calculate mass properties using OpenCASCADE VolumeProperties
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)

        if props.Mass() <= 0:
            raise ValueError("Shape has zero or negative volume - may be a surface/shell only")

        # Get center of gravity (in original units - meters)
        cg_point = props.CentreOfMass()
        reader.cg_original = np.array([cg_point.X(), cg_point.Y(), cg_point.Z()])

        # Get volume in original units (meters cubed)
        reader.volume_original = props.Mass()

        # Get inertia matrix (about the CG)
        # Returns: Ixx, Iyy, Izz, Ixy, Ixz, Iyz
        ixx = props.MatrixOfInertia().Value(1, 1)
        iyy = props.MatrixOfInertia().Value(2, 2)
        izz = props.MatrixOfInertia().Value(3, 3)
        ixy = -props.MatrixOfInertia().Value(1, 2)
        ixz = -props.MatrixOfInertia().Value(1, 3)
        iyz = -props.MatrixOfInertia().Value(2, 3)

        reader.inertia_original = np.array([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz]
        ])

        # Convert to inches
        reader._convert_to_inches()

        # Tessellate the shape into triangles using CadQuery
        reader._tessellate(result, tolerance=tessellation_tolerance)

        reader.triangles = reader.vertices  # Backward compatibility alias

        return reader

    def _set_scale_factor(self, unit_name: str):
        """Set scale factor using Pint unit conversion to inches."""
        unit_name_upper = unit_name.upper()
        ureg = _unit_registry()

        # Map unit names to Pint units
        unit_map = {
            'INCH': ureg.inch,
            'IN': ureg.inch,
            'MM': ureg.mm,
            'MILLIMETER': ureg.mm,
            'M': ureg.meter,
            'METER': ureg.meter,
            'CM': ureg.cm,
            'CENTIMETER': ureg.cm,
            'FT': ureg.ft,
            'FOOT': ureg.ft,
            'YD': ureg.yard,
            'YARD': ureg.yard,
        }

        if unit_name_upper in unit_map:
            source_unit = unit_map[unit_name_upper]
            # Convert 1 unit to inches and extract magnitude
            conversion = (1.0 * source_unit).to(ureg.inch)
            self.scale_factor = conversion.magnitude
            self.original_unit = unit_name_upper
        else:
            # Unknown unit - default to METER
            conversion = (1.0 * ureg.meter).to(ureg.inch)
            self.scale_factor = conversion.magnitude
            self.original_unit = 'METER'

    def _convert_to_inches(self):
        """Convert all measurements to inches."""
        scale = self.scale_factor
        scale_sq = scale * scale
        scale_cube = scale_sq * scale
        scale_fifth = scale_cube * scale_sq

        # Convert CG
        self.cg = self.cg_original * scale

        # Convert volume
        self.volume = self.volume_original * scale_cube

        # Convert volume inertia integral from source length^5 to inches^5.
        self.inertia_tensor = self.inertia_original * scale_fifth

        # If density provided, calculate actual mass and scale inertia properly
        if self.density > 0:
            self.mass = self.volume * self.density
            self.inertia_tensor = self.inertia_tensor * self.density

    def _tessellate(self, cq_workplane, tolerance: float = 0.5):
        """Convert CadQuery shape to triangular mesh."""
        def get_mesh(obj, tolerance=0.5):
            """Extract vertices and triangles from CadQuery shape."""
            shape = obj.val().wrapped

            from OCP.BRepMesh import BRepMesh_IncrementalMesh
            from OCP.BRep import BRep_Tool
            from OCP.TopoDS import TopoDS
            from OCP.TopExp import TopExp_Explorer
            from OCP.TopAbs import TopAbs_FACE
            from OCP.Poly import Poly_Triangulation

            # Mesh the shape
            mesh = BRepMesh_IncrementalMesh(shape, tolerance)
            mesh.Perform()

            all_vertices = []
            all_normals = []

            # Iterate over all faces and extract triangles
            face_exp = TopExp_Explorer(shape, TopAbs_FACE)

            while face_exp.More():
                face = TopoDS.Face_s(face_exp.Current())

                # Get triangulation for this face
                from OCP.TopLoc import TopLoc_Location
                location = TopLoc_Location()
                triangulation = BRep_Tool.Triangulation_s(face, location)

                if triangulation is not None and triangulation.NbTriangles() > 0:
                    transform = location.Transformation()
                    for i in range(1, triangulation.NbTriangles() + 1):
                        tri = triangulation.Triangle(i)

                        # Get triangle vertices (1-indexed in OpenCASCADE)
                        p1 = triangulation.Node(tri.Value(1)).Transformed(transform)
                        p2 = triangulation.Node(tri.Value(2)).Transformed(transform)
                        p3 = triangulation.Node(tri.Value(3)).Transformed(transform)

                        # Convert to numpy arrays and apply scale
                        v0 = np.array([p1.X(), p1.Y(), p1.Z()]) * self.scale_factor
                        v1 = np.array([p2.X(), p2.Y(), p2.Z()]) * self.scale_factor
                        v2 = np.array([p3.X(), p3.Y(), p3.Z()]) * self.scale_factor

                        all_vertices.append([v0, v1, v2])

                        # Compute triangle normal
                        edge1 = v1 - v0
                        edge2 = v2 - v0
                        normal = np.cross(edge1, edge2)
                        norm = np.linalg.norm(normal)
                        if norm > 0:
                            normal = normal / norm
                        all_normals.append(normal)

                face_exp.Next()

            return np.array(all_vertices), np.array(all_normals)

        self.vertices, self.normals = get_mesh(cq_workplane, tolerance=tolerance)

        if len(self.vertices) == 0:
            raise ValueError("No triangulable geometry found in STEP file")

    def get_surface_area(self) -> float:
        """Calculate total surface area in square inches."""
        if len(self.vertices) == 0:
            return 0.0

        # Cross product of two edges gives area of parallelogram, half is triangle area
        edges1 = self.vertices[:, 1] - self.vertices[:, 0]
        edges2 = self.vertices[:, 2] - self.vertices[:, 0]
        cross_products = np.cross(edges1, edges2)
        areas = 0.5 * np.linalg.norm(cross_products, axis=1)
        return np.sum(areas)

    def get_bounds(self) -> dict:
        """Get bounding box in inches."""
        if len(self.vertices) == 0:
            return {'x': (0, 0), 'y': (0, 0), 'z': (0, 0)}

        flat = self.vertices.reshape(-1, 3)
        mins = np.min(flat, axis=0)
        maxs = np.max(flat, axis=0)

        return {
            'x': (float(mins[0]), float(maxs[0])),
            'y': (float(mins[1]), float(maxs[1])),
            'z': (float(mins[2]), float(maxs[2]))
        }

    def get_cg(self) -> np.ndarray:
        """Get center of gravity coordinates in inches."""
        return self.cg.copy()

    def get_volume(self) -> float:
        """Get volume in cubic inches."""
        return self.volume

    def get_mass(self) -> float:
        """Get mass in pounds (if density was provided)."""
        return self.mass

    def get_inertia_tensor(self) -> np.ndarray:
        """
        Get inertia tensor about the CG.

        Returns 3x3 numpy array in units of lb-in^2 (if density provided)
        or in^4 (if no density - treating density as 1).

        The inertia tensor I is defined such that:
        I = [[Ixx, Ixy, Ixz],
             [Ixy, Iyy, Iyz],
             [Ixz, Iyz, Izz]]
        """
        return self.inertia_tensor.copy()

    def get_principal_moments(self) -> tuple:
        """
        Get principal moments of inertia and principal axes.

        Returns:
            (eigenvalues, eigenvectors) where eigenvalues are the principal
            moments of inertia and eigenvectors are the corresponding axes
        """
        eigenvalues, eigenvectors = np.linalg.eigh(self.inertia_tensor)
        return eigenvalues, eigenvectors

    def visualize(self, show_cg: bool = True, show_normals: bool = False,
                  auto_show: bool = True) -> go.Figure:
        """
        Plotly 3D visualization for debugging.

        Args:
            show_cg: Draw center of gravity marker
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

        # Add CG marker
        if show_cg:
            edge_traces.append(go.Scatter3d(
                x=[self.cg[0]],
                y=[self.cg[1]],
                z=[self.cg[2]],
                mode='markers+text',
                marker=dict(size=8, color='red', symbol='diamond'),
                text=['CG'],
                textposition='top center',
                name='Center of Gravity',
                showlegend=True
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
            title=f"STEP Mesh ({num_tris:,} triangles, {self.original_unit} → inches)",
            scene=dict(
                xaxis=dict(range=[center_x - max_range, center_x + max_range], title='X (in)'),
                yaxis=dict(range=[center_y - max_range, center_y + max_range], title='Y (in)'),
                zaxis=dict(range=[center_z - max_range, center_z + max_range], title='Z (in)'),
                aspectmode='cube'
            ),
            width=800,
            height=600,
            margin=dict(l=0, r=0, b=0, t=30)
        )

        if auto_show:
            fig.show()

        return fig

    def __repr__(self):
        """String representation with key data."""
        lines = [
            f"<STPReader: {len(self.vertices)} triangles>",
            f"  Units: {self.original_unit} (converted to inches)",
            f"  Volume: {self.volume:.4f} in³",
            f"  Surface Area: {self.get_surface_area():.4f} in²",
            f"  CG: ({self.cg[0]:.4f}, {self.cg[1]:.4f}, {self.cg[2]:.4f}) in",
        ]
        if self.density > 0:
            lines.append(f"  Mass: {self.mass:.4f} lb")
            lines.append(f"  Inertia tensor: [{self.inertia_tensor[0,0]:.4f}, {self.inertia_tensor[0,1]:.4f}, {self.inertia_tensor[0,2]:.4f}]")
            lines.append(f"                  [{self.inertia_tensor[1,0]:.4f}, {self.inertia_tensor[1,1]:.4f}, {self.inertia_tensor[1,2]:.4f}]")
            lines.append(f"                  [{self.inertia_tensor[2,0]:.4f}, {self.inertia_tensor[2,1]:.4f}, {self.inertia_tensor[2,2]:.4f}] lb-in²")
        return "\n".join(lines)


def load_stp(
    filepath: str,
    density: float = None,
    tessellation_tolerance: float = 0.5,
) -> STPReader:
    """
    Convenience function: load_stp('file.stp') -> STPReader instance.

    Args:
        filepath: Path to .stp or .step file
        density: Material density in lb/in^3 for mass/inertia calculations
        tessellation_tolerance: OpenCascade mesh tolerance in millimeters.
    """
    return STPReader.load(filepath, density, tessellation_tolerance)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Look for test files in the same directory
        script_dir = Path(__file__).parent
        test_files = list(script_dir.glob("*.stp")) + list(script_dir.glob("*.step"))
        if test_files:
            filepath = test_files[0]
            print(f"No file specified, loading: {filepath}")
        else:
            print("Usage: python stp_reader.py <filepath> [density_lb_per_in3]")
            print("Example: python stp_reader.py part.stp 0.098  # Aluminum")
            sys.exit(1)

    # Optional density argument
    density = None
    if len(sys.argv) > 2:
        density = float(sys.argv[2])

    reader = load_stp(filepath, density)
    print(reader)
    reader.visualize()
