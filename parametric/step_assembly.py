"""Read a STEP file as the assembly it is, rather than one silhouette.

``cad_import`` answers a different question. It takes a solid, slices it along
its axis and fits a Stack -- one radius per station. That is the right answer
for a single turned part and the wrong one for a vehicle: fins are not a value
of r(x), so they come back as a body bulge of the wrong diameter, and three
tanks and two intertanks come back as one tube.

A STEP file already carries the decomposition. An assembly written by any CAD
system names its components, and even a flat multi-solid export keeps the
solids apart -- the four fins on the test article arrive as four solids whether
or not anybody named them. This module surfaces that structure and measures
each piece, so the application can *ask* what each one is instead of guessing.

Nothing here decides what a component is. That is the point: ``roles.py``
argues that inference is brittle and a declared role cannot be guessed wrong,
and the same argument applies with more force to a whole vehicle. What this
returns is evidence -- name, volume, extent, where it sits, how round it is --
for a person to assign against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

MM_PER_M = 1000.0


@dataclass
class StepComponent:
    """One solid out of the file, measured but not interpreted."""

    index: int
    name: str
    volume_m3: float
    #: Axis-aligned bounds in metres, in file order (x, y, z).
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    centroid_m: tuple[float, float, float]
    #: The OCC shape, kept so a measurement can go back to the geometry.
    shape: object = field(default=None, repr=False)

    # -- derived, against a chosen axis -------------------------------

    def extent_m(self, axis: int) -> tuple[float, float]:
        return (self.bounds_min_m[axis], self.bounds_max_m[axis])

    def length_m(self, axis: int) -> float:
        low, high = self.extent_m(axis)
        return high - low

    def lateral_axes(self, axis: int) -> tuple[int, int]:
        pair = [i for i in range(3) if i != axis]
        return (pair[0], pair[1])

    def max_radius_m(self, axis: int) -> float:
        """Half the larger lateral extent: how far it reaches from the line."""
        a, b = self.lateral_axes(axis)
        return 0.5 * max(
            self.bounds_max_m[a] - self.bounds_min_m[a],
            self.bounds_max_m[b] - self.bounds_min_m[b],
        )

    def offset_m(self, axis: int) -> float:
        """Radial distance of the centroid from the axis line.

        A body sits on the centreline and reads ~0; a fin sits out at its own
        mean radius. This is the cheapest thing that separates the two, and it
        needs no assumption about what either one is.
        """
        a, b = self.lateral_axes(axis)
        return math.hypot(self.centroid_m[a], self.centroid_m[b])

    def solidity(self, axis: int) -> float:
        """Volume over the cylinder its lateral extent sweeps.

        Near 1 for a solid billet, well under it for a thin blade. This is the
        same outer-versus-material comparison ``cad_import`` uses to spot a
        hollow shell, applied per component instead of per station.
        """
        radius = self.max_radius_m(axis)
        length = self.length_m(axis)
        swept = math.pi * radius * radius * length
        return self.volume_m3 / swept if swept > 1e-15 else 0.0


@dataclass
class AssemblyRead:
    """Everything the file gave up."""

    source: Path
    components: list[StepComponent]
    #: 0/1/2 for x/y/z: the direction of greatest overall extent.
    axis: int
    named: bool
    notes: list[str] = field(default_factory=list)

    @property
    def axis_letter(self) -> str:
        return "XYZ"[self.axis]

    @property
    def total_length_m(self) -> float:
        if not self.components:
            return 0.0
        low = min(c.bounds_min_m[self.axis] for c in self.components)
        high = max(c.bounds_max_m[self.axis] for c in self.components)
        return high - low

    def origin_m(self) -> float:
        """Axial coordinate of the aft-most point, which becomes station 0."""
        if not self.components:
            return 0.0
        return min(c.bounds_min_m[self.axis] for c in self.components)

    def text(self) -> str:
        suffix = "" if self.named else "  (unnamed; the file carries no component names)"
        lines = [
            f"Read {self.source.name}",
            f"  axis            {self.axis_letter}",
            f"  length          {self.total_length_m:.4f} m",
            f"  components      {len(self.components)}{suffix}",
        ]
        lines += [f"  ! {note}" for note in self.notes]
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------


def read_assembly(path: str | Path) -> AssemblyRead:
    """Enumerate the solids in a STEP file, with names where it has them.

    Goes through XCAF -- the same reader ``step_metadata`` uses for material --
    because it is the one that carries the product structure. A plain
    ``STEPControl_Reader`` hands back a compound with the names discarded.
    """
    path = Path(path)
    notes: list[str] = []

    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPCAFControl import STEPCAFControl_Reader
    from OCP.TCollection import TCollection_ExtendedString
    from OCP.TDataStd import TDataStd_Name
    from OCP.TDF import TDF_Label, TDF_LabelSequence
    from OCP.TDocStd import TDocStd_Document
    from OCP.XCAFApp import XCAFApp_Application
    from OCP.XCAFDoc import XCAFDoc_DocumentTool

    application = XCAFApp_Application.GetApplication_s()
    document = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))
    application.NewDocument(TCollection_ExtendedString("MDTV-CAF"), document)

    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"{path.name}: OpenCascade could not read the file.")
    if not reader.Transfer(document):
        raise ValueError(f"{path.name}: nothing transferred out of the file.")

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(document.Main())

    def label_name(label) -> str:
        attribute = TDataStd_Name()
        if label.FindAttribute(TDataStd_Name.GetID_s(), attribute):
            return str(attribute.Get().ToExtString()).strip()
        return ""

    collected: list[tuple[str, object]] = []

    def visit(label, inherited: str = "") -> None:
        children = TDF_LabelSequence()
        shape_tool.GetComponents_s(label, children)
        if children.Length() == 0:
            collected.append(
                (inherited or label_name(label), shape_tool.GetShape_s(label))
            )
            return
        for i in range(1, children.Length() + 1):
            child = children.Value(i)
            referred = TDF_Label()
            target = referred if shape_tool.GetReferredShape_s(child, referred) else child
            name = label_name(child) or label_name(target) or inherited
            grandchildren = TDF_LabelSequence()
            shape_tool.GetComponents_s(target, grandchildren)
            if grandchildren.Length() == 0:
                # Take the shape off the *component* label so the assembly's
                # placement transform is included; the referred label holds the
                # part at its own origin.
                collected.append((name, shape_tool.GetShape_s(child)))
            else:
                visit(target, name)

    roots = TDF_LabelSequence()
    shape_tool.GetFreeShapes(roots)
    for i in range(1, roots.Length() + 1):
        visit(roots.Value(i))

    if not collected:
        raise ValueError(f"{path.name}: no solids found.")

    measured = [
        _measure(index, name, shape) for index, (name, shape) in enumerate(collected)
    ]
    components = [c for c in measured if c.volume_m3 > 0.0]
    if not components:
        raise ValueError(f"{path.name}: every solid measured zero volume.")
    if len(components) < len(measured):
        notes.append(
            f"{len(measured) - len(components)} of {len(measured)} solids had no "
            "volume and were dropped."
        )

    named = any(c.name and not c.name.isdigit() for c in components)
    if not named:
        notes.append(
            "The file names no components, so they are listed by size and "
            "position. Assignments cannot be matched by name on re-import."
        )

    axis = _detect_axis(components)
    return AssemblyRead(path, components, axis, named, notes)


def _measure(index: int, name: str, shape) -> StepComponent:
    """Volume, bounds and centroid of one solid, in metres."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    volume_mm3 = props.Mass()
    centre = props.CentreOfMass()

    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()

    return StepComponent(
        index=index,
        name=name or str(index + 1),
        volume_m3=volume_mm3 / (MM_PER_M ** 3),
        bounds_min_m=(xmin / MM_PER_M, ymin / MM_PER_M, zmin / MM_PER_M),
        bounds_max_m=(xmax / MM_PER_M, ymax / MM_PER_M, zmax / MM_PER_M),
        centroid_m=(
            centre.X() / MM_PER_M,
            centre.Y() / MM_PER_M,
            centre.Z() / MM_PER_M,
        ),
        shape=shape,
    )


def _detect_axis(components: list[StepComponent]) -> int:
    """The direction of greatest overall extent.

    Taken across the whole assembly rather than any one part, because the
    longest single component of a finned vehicle can easily be a tank while the
    vehicle runs the other way on a short stubby stage.
    """
    spans = []
    for i in range(3):
        low = min(c.bounds_min_m[i] for c in components)
        high = max(c.bounds_max_m[i] for c in components)
        spans.append(high - low)
    return max(range(3), key=lambda i: spans[i])
