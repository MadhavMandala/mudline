"""Scene construction: camera, grid, axes, and turning solids into meshes.

Pure geometry and numpy; no Qt and no GL objects. Keeping it separate means the
scale logic and the camera can be tested without a window, which the previous
viewer could not do at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# Camera
# ----------------------------------------------------------------------


class OrbitCamera:
    """Turntable camera: orbits a target, with framing that fits a bounding box."""

    def __init__(self, aspect: float = 16 / 9):
        self.target = np.zeros(3)
        self.distance = 3.0
        self.azimuth = np.radians(35.0)
        self.elevation = np.radians(18.0)
        self.fov_y = np.radians(45.0)
        self.aspect = max(aspect, 1e-6)
        #: Near and far are derived from distance rather than fixed. A fixed
        #: 0.3 m near plane with a 200 km far plane throws away almost all
        #: depth precision; scaling both with the view keeps the ratio sane at
        #: any zoom level, from a 0.1 m fin to a 100 km trajectory.
        self.near_factor = 0.002
        self.far_factor = 40.0

    # ------------------------------------------------------------------

    @property
    def near(self) -> float:
        return max(self.distance * self.near_factor, 1e-4)

    @property
    def far(self) -> float:
        return max(self.distance * self.far_factor, self.near * 100.0)

    def eye(self) -> np.ndarray:
        ce = np.cos(self.elevation)
        return self.target + self.distance * np.array([
            ce * np.sin(self.azimuth),
            np.sin(self.elevation),
            ce * np.cos(self.azimuth),
        ])

    def view_matrix(self) -> np.ndarray:
        eye = self.eye()
        forward = self.target - eye
        forward = forward / max(np.linalg.norm(forward), 1e-12)
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(forward, world_up)) > 0.999:
            world_up = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, world_up)
        right /= max(np.linalg.norm(right), 1e-12)
        up = np.cross(right, forward)

        view = np.eye(4, dtype=np.float32)
        view[0, :3], view[1, :3], view[2, :3] = right, up, -forward
        view[0, 3] = -float(np.dot(right, eye))
        view[1, 3] = -float(np.dot(up, eye))
        view[2, 3] = float(np.dot(forward, eye))
        return view

    def projection_matrix(self) -> np.ndarray:
        f = 1.0 / np.tan(self.fov_y * 0.5)
        near, far = self.near, self.far
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = (2.0 * far * near) / (near - far)
        proj[3, 2] = -1.0
        return proj

    # ------------------------------------------------------------------

    def orbit(self, dx: float, dy: float) -> None:
        self.azimuth -= dx * 0.008
        self.elevation = float(np.clip(self.elevation + dy * 0.008,
                                       np.radians(-89.0), np.radians(89.0)))

    def pan(self, dx: float, dy: float) -> None:
        right, up = self.basis()
        scale = self.distance * 0.0015
        self.target = self.target - right * dx * scale + up * dy * scale

    def zoom(self, steps: float) -> None:
        self.distance = float(np.clip(self.distance * (0.88 ** steps), 1e-3, 1e9))

    def basis(self) -> tuple[np.ndarray, np.ndarray]:
        eye = self.eye()
        forward = self.target - eye
        forward /= max(np.linalg.norm(forward), 1e-12)
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= max(np.linalg.norm(right), 1e-12)
        return right, np.cross(right, forward)

    def frame(self, low: np.ndarray, high: np.ndarray, margin: float = 1.35) -> None:
        """Fit a bounding box in view."""
        low = np.asarray(low, dtype=float)
        high = np.asarray(high, dtype=float)
        self.target = 0.5 * (low + high)
        extent = float(np.linalg.norm(high - low))
        if extent <= 0:
            extent = 1.0
        self.distance = max(extent * margin / (2.0 * np.tan(self.fov_y * 0.5)), 1e-3)


# ----------------------------------------------------------------------
# Grid
# ----------------------------------------------------------------------


def nice_spacing(target: float) -> float:
    """Round a spacing to the nearest 1, 2 or 5 times a power of ten.

    Keeps grid squares readable at any zoom: a 0.2 m rocket gets a 5 cm grid
    and a 100 km trajectory gets a 20 km one, without either turning into
    unreadable hatching or a single square filling the screen.
    """
    if target <= 0:
        return 1.0
    exponent = np.floor(np.log10(target))
    base = target / (10.0 ** exponent)
    for step in (1.0, 2.0, 5.0):
        if base <= step:
            return float(step * 10.0 ** exponent)
    return float(10.0 ** (exponent + 1))


@dataclass
class GridMesh:
    """Line geometry for the ground grid."""

    positions: np.ndarray
    colors: np.ndarray
    spacing: float
    extent: float


def build_grid(camera_distance: float, centre: np.ndarray | None = None,
               divisions: int = 24) -> GridMesh:
    """A ground grid sized to the current view, with emphasised major lines."""
    centre = np.zeros(3) if centre is None else np.asarray(centre, dtype=float)
    spacing = nice_spacing(camera_distance * 2.2 / divisions)
    extent = spacing * divisions

    # Snap the grid origin so lines stay put while panning instead of crawling.
    cx = np.round(centre[0] / spacing) * spacing
    cz = np.round(centre[2] / spacing) * spacing

    minor = np.array([0.26, 0.29, 0.34], dtype=np.float32)
    major = np.array([0.38, 0.43, 0.50], dtype=np.float32)
    axis_x = np.array([0.62, 0.31, 0.31], dtype=np.float32)
    axis_z = np.array([0.31, 0.48, 0.62], dtype=np.float32)

    positions: list[list[float]] = []
    colors: list[np.ndarray] = []

    for i in range(-divisions, divisions + 1):
        offset = i * spacing
        x = cx + offset
        z = cz + offset
        is_major = (i % 5) == 0

        colour_x = axis_z if abs(x) < spacing * 0.01 else (major if is_major else minor)
        positions += [[x, 0.0, cz - extent], [x, 0.0, cz + extent]]
        colors += [colour_x, colour_x]

        colour_z = axis_x if abs(z) < spacing * 0.01 else (major if is_major else minor)
        positions += [[cx - extent, 0.0, z], [cx + extent, 0.0, z]]
        colors += [colour_z, colour_z]

    return GridMesh(
        positions=np.array(positions, dtype=np.float32),
        colors=np.array(colors, dtype=np.float32),
        spacing=spacing,
        extent=extent,
    )


def build_axis_triad(length: float) -> tuple[np.ndarray, np.ndarray]:
    """Three coloured lines from the origin: X red, Y green, Z blue."""
    positions = np.array([
        [0, 0, 0], [length, 0, 0],
        [0, 0, 0], [0, length, 0],
        [0, 0, 0], [0, 0, length],
    ], dtype=np.float32)
    colors = np.array([
        [0.85, 0.30, 0.30], [0.85, 0.30, 0.30],
        [0.35, 0.78, 0.40], [0.35, 0.78, 0.40],
        [0.34, 0.55, 0.90], [0.34, 0.55, 0.90],
    ], dtype=np.float32)
    return positions, colors


# ----------------------------------------------------------------------
# Solids to renderable meshes
# ----------------------------------------------------------------------


#: The model's axis is +Z aft with the nose at the origin. On screen a rocket
#: should stand upright with its nose up, so display coordinates put the
#: vehicle axis on +Y and flip it: world = (x, length - z, y).
def display_transform(length_m: float) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = 1.0        # model x -> world x
    matrix[1, 2] = -1.0       # model z -> world -y
    matrix[1, 3] = length_m   # ... offset so the tail sits at y = 0
    matrix[2, 1] = 1.0        # model y -> world z
    matrix[3, 3] = 1.0
    return matrix


@dataclass
class RenderMesh:
    """Triangles ready for a vertex buffer."""

    name: str
    vertices: np.ndarray          # (n, 6): position + normal, interleaved
    triangle_count: int
    bounds_low: np.ndarray
    bounds_high: np.ndarray
    color: tuple[float, float, float] = (0.72, 0.76, 0.82)


MM_PER_M = 1000.0


def tessellate_solid(name: str, solid, tolerance_mm: float = 1.2,
                     color=(0.72, 0.76, 0.82)) -> RenderMesh | None:
    """Turn an OCC solid into an interleaved position/normal triangle array."""
    vertices, faces = solid.tessellate(tolerance_mm)
    if not faces:
        return None

    points = np.array([[v.x, v.y, v.z] for v in vertices], dtype=np.float32) / MM_PER_M
    triangles = points[np.array(faces, dtype=np.int64)]        # (m, 3, 3)

    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge1, edge2)
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-12
    normals[good] /= lengths[good, None]
    normals[~good] = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    flat_positions = triangles.reshape(-1, 3)
    flat_normals = np.repeat(normals, 3, axis=0)
    interleaved = np.hstack([flat_positions, flat_normals]).astype(np.float32)

    return RenderMesh(
        name=name,
        vertices=interleaved,
        triangle_count=len(triangles),
        bounds_low=flat_positions.min(axis=0),
        bounds_high=flat_positions.max(axis=0),
        color=color,
    )
