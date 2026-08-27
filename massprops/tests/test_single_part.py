import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

pytest.importorskip("gmsh", reason="massprops meshing needs the cad extra")
pytest.importorskip("trimesh", reason="massprops meshing needs the cad extra")

from massprops.io.step_parser import StepParser
from massprops.io.assembly_loader import load_assembly
from massprops.io.material_extractor import apply_materials_to_tree
from massprops.mesh.mesh_cache import MeshCache
from massprops.model.assembly import aggregate_properties


def test_sphere():
    step_file = Path("trajectory/vehicle/test_sphere.step")
    root = load_assembly(step_file)
    assert root is not None
    print(f"Root: {root.name}, children: {len(root.children)}")

    apply_materials_to_tree(root, StepParser(step_file))
    print(f"Density: {root.density}")

    cache = MeshCache("massprops/data")
    if not cache.load_cached(root):
        MeshCache.mesh_component(root, mesh_size=1.0)
        cache.save_cached(root)
    print(f"Meshed: {root.mesh_vertices.shape}")

    props = aggregate_properties(root)
    print(f"Mass: {props.mass:.4f} lbm")
    print(f"CG: {props.cg}")
    print(f"Volume: {props.volume:.4f} in³")


if __name__ == "__main__":
    test_sphere()
