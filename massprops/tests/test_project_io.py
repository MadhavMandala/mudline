import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

pytest.importorskip("gmsh", reason="massprops meshing needs the cad extra")
pytest.importorskip("trimesh", reason="massprops meshing needs the cad extra")

from massprops.io.project_io import save_project, load_project
from massprops.model.models import Component, MassProperties


def test_save_load():
    root = Component(name="TestAssembly")
    child = Component(name="ChildPart")
    child.computed_props = MassProperties(mass=5.0, cg=np.array([1,2,3]), inertia=np.eye(3)*10, volume=10.0)
    child.mesh_vertices = np.array([[0,0,0],[1,0,0],[0,1,0]])
    child.mesh_faces = np.array([[0,1,2]])
    root.children.append(child)

    project_path = Path("massprops/data/test_project.json")
    save_project(root, project_path)
    print("Saved to", project_path)

    loaded = load_project(project_path)
    print("Loaded:", loaded.name)
    assert loaded.name == "TestAssembly"
    assert len(loaded.children) == 1
    assert loaded.children[0].name == "ChildPart"
    assert loaded.children[0].computed_props.mass == 5.0
    assert loaded.children[0].mesh_vertices is not None
    print("Project IO OK")


if __name__ == "__main__":
    test_save_load()
