from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Callable, Optional

from massprops.model.models import Component, MassProperties
from massprops.mesh.mesher import generate_watertight_mesh, compute_mass_properties


# Optional QThreadPool support
try:
    from PySide6.QtCore import QRunnable, QThreadPool
    _HAS_QT = True
except Exception:
    _HAS_QT = False


class _MeshTask(QRunnable if _HAS_QT else object):
    """Background task for meshing a component."""
    def __init__(
        self,
        component: Component,
        mesh_size: Optional[float],
        mesh_size_factor: Optional[float],
        on_done: Callable[[Component], None],
    ):
        super().__init__()
        self.component = component
        self.mesh_size = mesh_size
        self.mesh_size_factor = mesh_size_factor
        self.on_done = on_done

    def run(self):
        try:
            MeshCache.mesh_component(
                self.component, self.mesh_size, self.mesh_size_factor, compute_props=True
            )
        except Exception as exc:
            self.component.step_metadata["mesh_error"] = str(exc)
        finally:
            self.on_done(self.component)


class MeshCache:
    """Thread-safe mesh cache with disk persistence.
    
    Cache files are stored as `{component_name}_mesh.npz` in the cache directory.
    """

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._pool = QThreadPool.globalInstance() if _HAS_QT else None

    def _cache_path(self, component: Component) -> Path:
        safe_name = "".join(c if c.isalnum() else "_" for c in component.name) or "mesh"
        return self.cache_dir / f"{safe_name}_{id(component)}_mesh.npz"

    def load_cached(self, component: Component) -> bool:
        """Try to load mesh from cache. Returns True on success."""
        path = self._cache_path(component)
        if not path.exists():
            return False
        try:
            data = np.load(path)
            component.mesh_vertices = data["vertices"]
            component.mesh_faces = data["faces"]
            if "mass" in data:
                component.computed_props = MassProperties(
                    mass=float(data["mass"]),
                    cg=data["cg"],
                    inertia=data["inertia"],
                    volume=float(data["volume"]),
                )
            return True
        except Exception:
            return False

    def save_cached(self, component: Component) -> None:
        """Save component mesh and properties to cache."""
        if component.mesh_vertices is None or component.mesh_faces is None:
            return
        path = self._cache_path(component)
        props = component.computed_props
        np.savez(
            path,
            vertices=component.mesh_vertices,
            faces=component.mesh_faces,
            mass=props.mass if props else 0.0,
            cg=props.cg if props else np.zeros(3),
            inertia=props.inertia if props else np.zeros((3, 3)),
            volume=props.volume if props else 0.0,
        )

    @staticmethod
    def mesh_component(
        component: Component,
        mesh_size: Optional[float] = None,
        mesh_size_factor: Optional[float] = None,
        compute_props: bool = True,
    ) -> None:
        """Mesh a single component synchronously."""
        if component.source_step is None:
            return
        if not Path(component.source_step).exists():
            return
        vertices, faces = generate_watertight_mesh(
            component.source_step,
            mesh_size=mesh_size,
            mesh_size_factor=mesh_size_factor,
        )
        component.mesh_vertices = vertices
        component.mesh_faces = faces
        if compute_props:
            density = component.density or 0.1
            component.computed_props = compute_mass_properties(vertices, faces, density)

    def queue_mesh(
        self,
        component: Component,
        mesh_size: Optional[float] = None,
        mesh_size_factor: Optional[float] = None,
        on_done: Optional[Callable[[Component], None]] = None,
    ) -> None:
        """Queue a component for meshing (background if Qt is available)."""
        if self.load_cached(component):
            if on_done:
                on_done(component)
            return

        if _HAS_QT and self._pool:
            task = _MeshTask(component, mesh_size, mesh_size_factor, on_done or (lambda c: None))
            self._pool.start(task)
        else:
            self.mesh_component(component, mesh_size, mesh_size_factor, compute_props=True)
            self.save_cached(component)
            if on_done:
                on_done(component)
