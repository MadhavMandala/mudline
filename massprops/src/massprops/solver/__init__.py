"""First-party mass-properties solvers for triangle meshes.

Pure NumPy. Nothing here imports trimesh, gmsh or any other mesh library, so the
package can move into a shared core later without dragging dependencies along.

    mesh_props     exact volume / CG / inertia of a closed mesh
    mesh_quality   watertightness checks and winding repair
    mesh_sampler   interior point sampling, for inertial matching on real CAD
"""

from massprops.solver.mesh_props import (
    MeshMassProps,
    mass_properties,
    point_cloud_mass_properties,
    signed_volume,
)
from massprops.solver.mesh_quality import (
    QualityReport,
    inspect,
    is_consistently_wound,
    is_watertight,
    repair,
)
from massprops.solver.mesh_sampler import (
    SampleResult,
    SeedForMatching,
    contains,
    sample_interior,
    seed_for_matching,
    winding_numbers,
)

__all__ = [
    "MeshMassProps",
    "QualityReport",
    "SampleResult",
    "SeedForMatching",
    "contains",
    "inspect",
    "is_consistently_wound",
    "is_watertight",
    "mass_properties",
    "point_cloud_mass_properties",
    "repair",
    "sample_interior",
    "seed_for_matching",
    "signed_volume",
    "winding_numbers",
]
