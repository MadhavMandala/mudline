# MassProp

Agent-assisted development of a CAD mass properties tool.

## What was built

- **Custom ISO 10303-21 STEP parser** (`massprops/io/step_parser.py`) with AP242 assembly extraction
- **Assembly loader** (`massprops/io/assembly_loader.py`) – builds Component/Assembly trees with transforms
- **Material extractor** (`massprops/io/material_extractor.py`) – pulls density/mass/volume from STEP or falls back to default
- **Gmsh + Trimesh mesher** (`massprops/mesh/mesher.py`) – watertight surface meshing and mass property computation
- **Mesh cache** (`massprops/mesh/mesh_cache.py`) – disk cache (`.npz`) with optional QThreadPool background meshing
- **Assembly aggregation** (`massprops/model/assembly.py`) – parallel axis theorem for combined CG and inertia
- **PySide6 GUI skeleton** – tree view, Plotly 3D viewer, property panel with override UI
- **Project save/load** (`massprops/io/project_io.py`) – JSON tree + STL/CSV/JSON/ NPZ exports

## Known limitations / MVP gaps

- **Task 7 (KKT solver)** was skipped per plan.
- **Per-part geometry in assemblies**: the MVP meshes the whole STEP file and attaches it to the root. Splitting meshes to individual components requires a full B-rep kernel (pythonocc-core) or bounding-box heuristics.
- **Transform/geometry unit consistency**: Gmsh/OCC returns coordinates in mm regardless of STEP declared unit. The mesher scales by `1/25.4` to inches. Assembly transforms from the parser are scaled by the STEP declared unit. For meter-based files with non-identity transforms, there may be a mismatch.
- **STEP parser coverage**: complex entity instances are stored but not fully parsed. External file references use common patterns but may miss edge cases.

## Run

```bash
# From project root (mudline/)
python massprops/main.py [optional_step_file.stp]
```

## Environment

Uses the existing `.venv` in the parent directory (Python 3.12). Key packages:
- `gmsh` (STEP import + meshing)
- `trimesh` (mass properties)
- `pint` (unit conversion)
- `plotly` + `PySide6` (GUI)
