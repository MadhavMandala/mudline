"""Build a synthetic assembly from separate STEP parts and launch the GUI.

This demonstrates multi-part 3D viewing because each child loads its own
STEP file and gets its own color + transform in the viewer.
"""
import sys
from pathlib import Path
import numpy as np

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from PySide6.QtWidgets import QApplication
from massprops.model.models import Assembly, Component
from massprops.mesh.mesh_cache import MeshCache
from massprops.model.assembly import aggregate_properties
from massprops.gui.main_window import MainWindow


def build_demo_assembly() -> Assembly:
    """Create an assembly with three distinct parts positioned in space."""
    root = Assembly(name="DemoAssembly")

    # Part 1: 50mm cube at origin
    cube = Component(name="BaseCube")
    cube.source_step = Path(__file__).parent / "cube_50mm.step"
    cube.instance_transform = np.eye(4)
    cube.density = 0.1
    root.children.append(cube)

    # Part 2: Cylinder on top of cube
    cyl = Component(name="TopCylinder")
    cyl.source_step = Path(__file__).parent / "cylinder_5x20mm.step"
    # Place on top of 50mm cube: translate Z by 50mm = 1.9685 in
    cyl.instance_transform = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 50.0 / 25.4],
        [0, 0, 0, 1],
    ])
    cyl.density = 0.1
    root.children.append(cyl)

    # Part 3: Small cube offset to the side
    small = Component(name="SideCube")
    small.source_step = Path(__file__).parent / "cube_10mm.step"
    small.instance_transform = np.array([
        [1, 0, 0, 60.0 / 25.4],
        [0, 1, 0, 0],
        [0, 0, 1, 10.0 / 25.4],
        [0, 0, 0, 1],
    ])
    small.density = 0.1
    root.children.append(small)

    # Mesh each part
    cache = MeshCache(project_root / "data")
    for child in root.children:
        if not cache.load_cached(child):
            MeshCache.mesh_component(child, mesh_size=1.0)
            cache.save_cached(child)

    aggregate_properties(root)
    return root


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    root = build_demo_assembly()
    window.set_root(root)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
