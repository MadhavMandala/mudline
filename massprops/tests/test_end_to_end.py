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


def test_assembly_load():
    step_file = Path(".venv/share/doc/gmsh/examples/api/as1-tu-203.stp")
    root = load_assembly(step_file)
    assert root is not None
    assert len(root.children) > 0
    print("Assembly tree loaded OK")
    print(f"Root: {root.name}, children: {len(root.children)}")

    apply_materials_to_tree(root, StepParser(step_file))
    print("Materials applied")

    # Mesh the whole file for now
    cache = MeshCache("massprops/data")
    if not cache.load_cached(root):
        MeshCache.mesh_component(root, mesh_size=2.0)
        cache.save_cached(root)
    print(f"Meshed: {root.mesh_vertices.shape if root.mesh_vertices is not None else None}")

    props = aggregate_properties(root)
    print(f"Aggregated mass: {props.mass:.4f} lbm")
    print(f"Aggregated CG: {props.cg}")
    print(f"Aggregated volume: {props.volume:.4f} in³")

    assert props.mass > 0
    assert props.volume > 0


if __name__ == "__main__":
    test_assembly_load()
