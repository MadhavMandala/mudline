from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from pathlib import Path
from typing import Optional


@dataclass
class MassProperties:
    mass: float = 0.0               # lbm
    cg: np.ndarray = field(default_factory=lambda: np.zeros(3))  # in
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))  # lbm·in²
    volume: float = 0.0             # in³


@dataclass
class Component:
    name: str = ""
    source_step: Optional[Path] = None
    instance_transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    mesh_vertices: Optional[np.ndarray] = None  # Nx3 inches
    mesh_faces: Optional[np.ndarray] = None      # Mx3 ints
    density: Optional[float] = None              # lbm/in³
    computed_props: Optional[MassProperties] = None
    overridden_props: Optional[MassProperties] = None
    kkt_point_cloud: Optional[tuple[np.ndarray, np.ndarray]] = None  # (points, masses)
    children: list['Component'] = field(default_factory=list)
    # Metadata from STEP parser
    step_product_def_id: Optional[int] = None
    step_metadata: dict = field(default_factory=dict)
    # Which fields in overridden_props are actually active (e.g. {'mass', 'cg'})
    override_fields: set[str] = field(default_factory=set)
    # Assembly-distributed mass scale factor (applied after overrides, leaf-only)
    mass_scale: float = 1.0

    def effective_props(self) -> MassProperties:
        # Start with base properties (computed for leaf, aggregated for assembly)
        if self.children:
            from massprops.model.assembly import aggregate_properties
            base = aggregate_properties(self)
        else:
            base = self.computed_props or MassProperties()

        # Apply field overrides on top of base
        if self.overridden_props is not None and self.override_fields:
            mass = self.overridden_props.mass if 'mass' in self.override_fields else base.mass
            cg = self.overridden_props.cg if 'cg' in self.override_fields else base.cg
            inertia = self.overridden_props.inertia if 'inertia' in self.override_fields else base.inertia
            volume = self.overridden_props.volume if 'volume' in self.override_fields else base.volume
            base = MassProperties(mass=mass, cg=cg, inertia=inertia, volume=volume)

        # Apply assembly-distributed mass scale (leaf nodes only; assemblies
        # pick up the scale naturally through aggregation of scaled children)
        if not self.children and self.mass_scale != 1.0 and base.mass > 0:
            base = MassProperties(
                mass=base.mass * self.mass_scale,
                cg=base.cg,
                inertia=base.inertia * self.mass_scale,
                volume=base.volume,
            )

        return base


class Assembly(Component):
    pass
