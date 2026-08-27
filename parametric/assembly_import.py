"""Turn an assigned STEP assembly into a vehicle model.

The division of labour matters. ``step_assembly`` reads the file and measures
what is in it; this module measures each group *against the type it has been
told it is*, and builds the model. Nothing here guesses what a component is.

That is the whole difference from ``cad_import``. Measuring a solid you have
been told is a fin is a well-posed problem -- cut it at its root and at its tip
and read off the two chords. Deciding *which* solid is a fin from a radius
profile is not, and the old importer answered it by fitting the fins into the
body as a bulge of the wrong diameter, quietly, with a residual that read as a
clean import.

Imported parts carry their CAD volume as a mass override rather than letting
the geometry imply one. That is not a small thing: ``Stack.volume_m3`` integrates
section *area* by trapezoid, and area goes as r^2, so a taper fitted with few
sections is overestimated -- 50% on a cone described by its two ends. The solid
in the file has an exact volume, so an import should use it.

Station convention: 0 at the forward tip, increasing aft, matching the rest of
the model. Which end that is comes from the declared nose, not from a guess.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from parametric.components import FinSet, PointMass, Protuberance, Stack, Tank
from parametric.model import VehicleModel
from parametric.roles import AeroRole
from parametric.step_assembly import AssemblyRead, StepComponent, read_assembly
from parametric.xsec import NoseProfile, XSec, XSecShape, nose_radius

MM_PER_M = 1000.0

#: How little of its own swept cylinder a solid must fill to read as a blade.
#: A trapezoidal fin comes in near 0.05 and a cone at exactly 1/3, so the line
#: sits between them with room on both sides.
BLADE_SOLIDITY = 0.20


class PartType(str, Enum):
    """What a person says a group of solids is.

    ``TANK`` and ``INTERTANK`` are the same thing to the air -- both are body,
    both carry the outer mould line -- and different everywhere else. A tank
    holds propellant that drains and moves the CG through the burn; an
    intertank is structure and therefore only a mass. Keeping them apart in the
    tree is what lets a propellant load be assigned to the right part later.
    """

    NOSE = "nose"
    BODY = "body"
    TANK = "tank"
    INTERTANK = "intertank"
    FINSET = "finset"
    PROTUBERANCE = "protuberance"
    INTERNAL = "internal"
    IGNORE = "ignore"

    @property
    def label(self) -> str:
        return {
            PartType.NOSE: "Nose cone",
            PartType.BODY: "Body section",
            PartType.TANK: "Tank (holds propellant)",
            PartType.INTERTANK: "Intertank / skirt (structure)",
            PartType.FINSET: "Fin set",
            PartType.PROTUBERANCE: "Protuberance",
            PartType.INTERNAL: "Internal mass",
            PartType.IGNORE: "Ignore",
        }[self]

    @property
    def aero_role(self) -> AeroRole:
        return {
            PartType.NOSE: AeroRole.NOSE,
            PartType.BODY: AeroRole.BODY,
            PartType.TANK: AeroRole.BODY,
            PartType.INTERTANK: AeroRole.BODY,
            PartType.FINSET: AeroRole.FIN,
            PartType.PROTUBERANCE: AeroRole.PROTUBERANCE,
            PartType.INTERNAL: AeroRole.INTERNAL,
            PartType.IGNORE: AeroRole.INTERNAL,
        }[self]

    @property
    def groups_solids(self) -> bool:
        """Whether several solids naturally become one part."""
        return self in (PartType.FINSET, PartType.TANK, PartType.BODY)


@dataclass
class Assignment:
    """One part, and the solids that make it up."""

    part_type: PartType
    name: str
    indices: list[int] = field(default_factory=list)

    def components(self, read: AssemblyRead) -> list[StepComponent]:
        by_index = {c.index: c for c in read.components}
        return [by_index[i] for i in self.indices if i in by_index]


@dataclass
class BuildReport:
    """What the build made of the assignments."""

    source: Path
    parts: int
    solids: int
    length_m: float
    dry_mass_kg: float
    notes: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        out = [
            f"Imported {self.source.name}",
            f"  parts           {self.parts} from {self.solids} solids",
            f"  length          {self.length_m:.4f} m",
            f"  dry mass        {self.dry_mass_kg:.3f} kg",
            "",
        ]
        out += self.lines
        if self.notes:
            out.append("")
            out += [f"  ! {note}" for note in self.notes]
        return "\n".join(out)


# ----------------------------------------------------------------------
# Suggestion -- a starting point for the dialog, never a decision
# ----------------------------------------------------------------------


def suggest(read: AssemblyRead) -> list[Assignment]:
    """Pre-fill the assignment table so the common case is a glance and an OK.

    Two signals separate a blade from a body, and *both* are required, because
    neither is sufficient alone: a fin's centroid sits off the centreline, and
    a fin fills almost none of the cylinder its extent sweeps. A cone is the
    reason both are needed -- it fills exactly 1/3 of its own cylinder, close
    enough to a blade's 0.05 to be uncomfortable, and it is only ever ruled out
    by sitting on the centreline. Everything else is ordered along the axis and
    offered as body. The nose is the piece at one end whose solidity says it
    tapers.

    Every one of these is editable in the dialog. They are offered because a
    twelve-part assembly is tedious to assign from scratch, not because the
    guess is trusted -- which is the distinction ``roles.py`` draws.
    """
    axis = read.axis
    bodies: list[StepComponent] = []
    blades: list[StepComponent] = []
    for component in read.components:
        offset = component.offset_m(axis)
        radius = component.max_radius_m(axis)
        off_axis = radius > 1e-9 and offset > 0.35 * max(radius, 1e-9)
        # Well below a cone's 1/3, and well above a plate's 0.05.
        if off_axis and component.solidity(axis) < BLADE_SOLIDITY:
            blades.append(component)
        else:
            bodies.append(component)

    bodies.sort(key=lambda c: c.bounds_min_m[axis])
    assignments: list[Assignment] = []

    # The nose: an end piece that tapers. Solidity of a cone about its own
    # extent is 1/3, a cylinder is 1, so the test is loose on purpose.
    nose_index = None
    if bodies:
        for candidate in (bodies[0], bodies[-1]):
            if candidate.solidity(axis) < 0.75:
                nose_index = candidate.index
                break

    for component in bodies:
        if component.index == nose_index:
            assignments.append(Assignment(PartType.NOSE, component.name, [component.index]))
        else:
            assignments.append(Assignment(PartType.BODY, component.name, [component.index]))

    if blades:
        blades.sort(key=lambda c: c.bounds_min_m[axis])
        # Fins at the same station are one set. Anything at a different station
        # is a second set, which is common enough (canards) to be worth doing.
        groups: list[list[StepComponent]] = []
        for blade in blades:
            placed = False
            for group in groups:
                reference = group[0]
                same_station = abs(
                    blade.bounds_min_m[axis] - reference.bounds_min_m[axis]
                ) < 0.01
                same_size = abs(blade.volume_m3 - reference.volume_m3) < 0.05 * max(
                    reference.volume_m3, 1e-12
                )
                if same_station and same_size:
                    group.append(blade)
                    placed = True
                    break
            if not placed:
                groups.append([blade])
        for number, group in enumerate(groups, start=1):
            name = "fins" if len(groups) == 1 else f"fins_{number}"
            assignments.append(
                Assignment(PartType.FINSET, name, [c.index for c in group])
            )

    return assignments


# ----------------------------------------------------------------------
# Measurement -- against a declared type
# ----------------------------------------------------------------------


def _slab_bounds(shape, axis: int, low_m: float, high_m: float):
    """Bounding box of the part of ``shape`` between two axial planes.

    Returns metres, or None where the cut is empty. Cutting with a box rather
    than sectioning with a plane keeps this to one well-conditioned boolean;
    a planar section hands back a wire whose faces have to be sorted out.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    whole = Bnd_Box()
    BRepBndLib.Add_s(shape, whole)
    xmin, ymin, zmin, xmax, ymax, zmax = whole.Get()
    lows = [xmin, ymin, zmin]
    highs = [xmax, ymax, zmax]

    pad = 1.0 + max(highs[i] - lows[i] for i in range(3))
    lows = [value - pad for value in lows]
    highs = [value + pad for value in highs]
    lows[axis] = low_m * MM_PER_M
    highs[axis] = high_m * MM_PER_M

    box = BRepPrimAPI_MakeBox(
        gp_Pnt(lows[0], lows[1], lows[2]), gp_Pnt(highs[0], highs[1], highs[2])
    ).Shape()
    common = BRepAlgoAPI_Common(shape, box)
    common.Build()
    if not common.IsDone():
        return None
    cut = common.Shape()

    result = Bnd_Box()
    BRepBndLib.Add_s(cut, result)
    if result.IsVoid():
        return None
    a, b, c, d, e, f = result.Get()
    return (
        (a / MM_PER_M, b / MM_PER_M, c / MM_PER_M),
        (d / MM_PER_M, e / MM_PER_M, f / MM_PER_M),
    )


def _axial_profile(component: StepComponent, axis: int, samples: int = 24):
    """Radius against axial position, sampled from the solid itself."""
    low, high = component.extent_m(axis)
    span = high - low
    if span <= 0:
        return np.zeros(0), np.zeros(0)
    step = span / samples
    # The slab is cut much thinner than the spacing. A slab reports the largest
    # radius anywhere inside it, so through a taper its own thickness becomes a
    # systematic overestimate -- on a 250 mm cone sampled 24 times a full-width
    # slab reads 2.1 mm high, and a perfect cone stops looking like one.
    # Keeping the cut thin costs nothing and removes the bias.
    thickness = 0.12 * step
    a, b = component.lateral_axes(axis)
    stations, radii = [], []
    for index in range(samples):
        centre = low + (index + 0.5) * step
        bounds = _slab_bounds(
            component.shape, axis, centre - 0.5 * thickness, centre + 0.5 * thickness
        )
        if bounds is None:
            continue
        lo, hi = bounds
        radius = 0.25 * ((hi[a] - lo[a]) + (hi[b] - lo[b]))
        stations.append(centre)
        radii.append(radius)
    return np.array(stations), np.array(radii)


def _rotate_about_axis(shape, axis: int, angle_rad: float):
    """Spin a shape about the vehicle axis through the origin."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf

    direction = [0.0, 0.0, 0.0]
    direction[axis] = 1.0
    transform = gp_Trsf()
    transform.SetRotation(
        gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(direction[0], direction[1], direction[2])),
        angle_rad,
    )
    return BRepBuilderAPI_Transform(shape, transform, True).Shape()


def _lateral_principal_azimuth(shape, axis: int) -> float:
    """Azimuth of the plane a blade lies in.

    Not the centroid's azimuth. A fin is often modelled to one side of the
    radial plane rather than straddling it -- an offset of a few millimetres on
    a blade a hundred out is under two degrees of centroid azimuth, and
    rotating by that tilts the blade so its own thickness reads almost twice
    what it is. The blade's *shape* is unambiguous where its position is not:
    the direction of greatest spread in the lateral plane is the way it points.
    """
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    a, b = [i for i in range(3) if i != axis]
    points = []
    explorer = TopExp_Explorer(shape, TopAbs_VERTEX)
    while explorer.More():
        point = BRep_Tool.Pnt_s(TopoDS.Vertex_s(explorer.Current()))
        coords = (point.X(), point.Y(), point.Z())
        points.append((coords[a], coords[b]))
        explorer.Next()
    if len(points) < 2:
        return 0.0

    data = np.array(points, dtype=float)
    centred = data - data.mean(axis=0)
    # Principal direction of the lateral footprint.
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    direction = vectors[0]
    # Point it outward: the blade extends away from the axis, so the far end
    # should project positive.
    projections = data @ direction
    if abs(projections.min()) > abs(projections.max()):
        direction = -direction
    return float(math.atan2(direction[1], direction[0]))


def measure_finset(components: list[StepComponent], axis: int) -> dict:
    """Root chord, tip chord, span, sweep and thickness from the fin solids.

    The blade is spun about the vehicle axis until it lies in the plane the
    first lateral axis spans, which turns "measure a fin at some azimuth" into
    "read a bounding box".

    The two chords are not read at the root and the tip. They are read at two
    stations *inside* the span and extrapolated outward, because a trapezoidal
    fin -- which is what ``FinSet`` models -- has a chord and a leading edge
    both linear in radius, so two interior cuts determine all four numbers
    exactly while avoiding the root fillet and the tip edge, where a real
    blade stops being trapezoidal and a cut is least reliable.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Section
    from OCP.BRepBndLib import BRepBndLib
    from OCP.gp import gp_Dir, gp_Pln, gp_Pnt

    reference = components[0]
    a, b = reference.lateral_axes(axis)
    azimuth = _lateral_principal_azimuth(reference.shape, axis)
    aligned = _rotate_about_axis(reference.shape, axis, -azimuth)

    box = Bnd_Box()
    BRepBndLib.Add_s(aligned, box)
    raw = box.Get()
    lo = (raw[0] / MM_PER_M, raw[1] / MM_PER_M, raw[2] / MM_PER_M)
    hi = (raw[3] / MM_PER_M, raw[4] / MM_PER_M, raw[5] / MM_PER_M)

    root_radius = min(abs(lo[a]), abs(hi[a]))
    tip_radius = max(abs(lo[a]), abs(hi[a]))
    span = max(tip_radius - root_radius, 1e-9)
    thickness = max(hi[b] - lo[b], 1e-6)

    def cut_at(radius: float):
        """Axial extent of the blade in the plane at one radius.

        A plane rather than a slab of finite width. A slab reports the union of
        every section it spans, so on a tapered blade it reads the chord at one
        of its edges rather than at its centre -- a clean half-band bias, 2.5 mm
        on a 3 mm band, which is 1% of a root chord and 2.5% of a tip chord.
        A plane has no width and no bias.
        """
        normal = [0.0, 0.0, 0.0]
        normal[a] = 1.0
        origin = [0.0, 0.0, 0.0]
        origin[a] = radius * MM_PER_M
        plane = gp_Pln(
            gp_Pnt(origin[0], origin[1], origin[2]),
            gp_Dir(normal[0], normal[1], normal[2]),
        )
        section = BRepAlgoAPI_Section(aligned, plane, False)
        section.ComputePCurveOn1(False)
        section.Approximation(False)
        section.Build()
        if not section.IsDone():
            return None
        result = Bnd_Box()
        BRepBndLib.Add_s(section.Shape(), result)
        if result.IsVoid():
            return None
        values = result.Get()
        return (values[axis] / MM_PER_M, values[axis + 3] / MM_PER_M)

    inner_fraction, outer_fraction = 0.15, 0.85
    inner = cut_at(root_radius + inner_fraction * span)
    outer = cut_at(root_radius + outer_fraction * span)
    if inner is None or outer is None:
        # Nothing to extrapolate from; fall back to the whole blade.
        inner = outer = (lo[axis], hi[axis])
        inner_fraction, outer_fraction = 0.0, 1.0

    def extrapolate(inner_value: float, outer_value: float, fraction: float) -> float:
        spread = outer_fraction - inner_fraction
        if spread <= 1e-9:
            return inner_value
        slope = (outer_value - inner_value) / spread
        return inner_value + slope * (fraction - inner_fraction)

    root_low = extrapolate(inner[0], outer[0], 0.0)
    root_high = extrapolate(inner[1], outer[1], 0.0)
    tip_low = extrapolate(inner[0], outer[0], 1.0)
    tip_high = extrapolate(inner[1], outer[1], 1.0)

    return {
        "count": len(components),
        "root_chord_m": max(root_high - root_low, 1e-6),
        "tip_chord_m": max(tip_high - tip_low, 0.0),
        "span_m": span,
        "thickness_m": thickness,
        "root_radius_m": root_radius,
        # Both edges in raw axial coordinates; the caller turns them into
        # stations once the vehicle orientation is known.
        "root_axial_m": (root_low, root_high),
        "tip_axial_m": (tip_low, tip_high),
    }


def measure_nose(component: StepComponent, axis: int) -> dict:
    """Length, base diameter and which analytic family the profile matches."""
    stations, radii = _axial_profile(component, axis)
    length = component.length_m(axis)
    base_radius = float(radii.max()) if len(radii) else component.max_radius_m(axis)

    profile = NoseProfile.OGIVE
    residual = float("inf")
    if len(stations) >= 3:
        # Distance from the tip, which is the end with the smaller radius.
        tip_at_low = radii[0] < radii[-1]
        from_tip = (stations - stations.min()) if tip_at_low else (stations.max() - stations)
        for candidate in NoseProfile:
            predicted = nose_radius(candidate, length, base_radius, from_tip)
            error = float(np.sqrt(np.mean((predicted - radii) ** 2)))
            if error < residual:
                profile, residual = candidate, error

    return {
        "length_m": length,
        "base_diameter_m": 2.0 * base_radius,
        "profile": profile,
        "profile_residual_m": residual,
    }


def measure_body(component: StepComponent, axis: int) -> dict:
    """Length and end diameters. Exact for a tube, ends-only for a taper."""
    stations, radii = _axial_profile(component, axis)
    length = component.length_m(axis)
    if len(radii) >= 2:
        forward, aft = float(radii[0]), float(radii[-1])
    else:
        forward = aft = component.max_radius_m(axis)
    return {
        "length_m": length,
        "radius_low_m": forward,
        "radius_high_m": aft,
        "max_radius_m": component.max_radius_m(axis),
    }


def measure_tank(component: StepComponent, axis: int) -> dict:
    """Barrel diameter, overall length and an estimated dome ratio.

    The dome height is where the profile first reaches full diameter, measured
    in from each end. A tank modelled in CAD as a plain tube reports no dome,
    which is reported rather than invented.
    """
    stations, radii = _axial_profile(component, axis, samples=40)
    length = component.length_m(axis)
    radius = float(radii.max()) if len(radii) else component.max_radius_m(axis)

    dome_height = 0.0
    if len(radii) >= 4 and radius > 0:
        full = 0.99 * radius
        low, high = stations.min(), stations.max()
        forward = next((s for s, r in zip(stations, radii) if r >= full), low)
        aft = next(
            (s for s, r in zip(stations[::-1], radii[::-1]) if r >= full), high
        )
        dome_height = max(forward - low, high - aft, 0.0)

    return {
        "length_m": length,
        "diameter_m": 2.0 * radius,
        "dome_height_m": dome_height,
        "dome_measured": dome_height > 1e-6,
    }


# ----------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------


def _density(material: str) -> float:
    from parametric.materials import get_material

    return get_material(material).density_kg_m3


def build_model(
    read: AssemblyRead,
    assignments: list[Assignment],
    name: str = "",
    material: str = "aluminium_6061_t6",
) -> tuple[VehicleModel, BuildReport]:
    """Assemble the assigned solids into a vehicle model.

    Every part built here is marked ``imported`` and carries its CAD volume as
    a mass override, so the mass is the file's and not an integration of a
    fitted profile.
    """
    axis = read.axis
    used = [a for a in assignments if a.part_type is not PartType.IGNORE and a.indices]
    if not used:
        raise ValueError("Nothing was assigned; there is no model to build.")

    low = min(c.bounds_min_m[axis] for c in read.components)
    high = max(c.bounds_max_m[axis] for c in read.components)

    # Which end is the front. The declared nose settles it; with no nose
    # declared, the end whose piece tapers most is the better of two guesses,
    # and it is reported.
    notes: list[str] = list(read.notes)
    nose_assignment = next((a for a in used if a.part_type is PartType.NOSE), None)
    if nose_assignment is not None:
        pieces = nose_assignment.components(read)
        centre = sum(
            0.5 * (c.bounds_min_m[axis] + c.bounds_max_m[axis]) for c in pieces
        ) / len(pieces)
        nose_at_high = centre > 0.5 * (low + high)
    else:
        nose_at_high = True
        notes.append(
            "No nose was assigned, so the forward end was taken to be the "
            "far end of the axis. Stations may run backwards."
        )

    def station_of(axial_m: float) -> float:
        return (high - axial_m) if nose_at_high else (axial_m - low)

    def station_range(component: StepComponent) -> tuple[float, float]:
        a_low, a_high = component.extent_m(axis)
        first, second = station_of(a_low), station_of(a_high)
        return (min(first, second), max(first, second))

    model = VehicleModel(name or read.source.stem)
    model.description = f"Imported from {read.source.name}"

    density = _density(material)
    lines: list[str] = []
    solids = 0

    for assignment in used:
        pieces = assignment.components(read)
        if not pieces:
            continue
        solids += len(pieces)
        volume = sum(c.volume_m3 for c in pieces)
        starts = [station_range(c)[0] for c in pieces]
        ends = [station_range(c)[1] for c in pieces]
        station, finish = min(starts), max(ends)

        part = _build_part(
            assignment, pieces, axis, station, finish, station_of, material, notes
        )
        if part is None:
            continue

        part.imported = True
        part.aero_role = assignment.part_type.aero_role
        # The file's volume, not the model's. See the module docstring.
        part.mass_override_kg = volume * density
        model.add(part)

        lines.append(
            "  %-14s %-11s %7.3f -> %7.3f m  %8.3f kg  %s"
            % (
                assignment.name[:14],
                assignment.part_type.value,
                station,
                finish,
                part.mass_override_kg,
                ", ".join(c.name for c in pieces)[:34],
            )
        )

    if not model.root.children:
        raise ValueError("No part could be built from the assignments.")

    report = BuildReport(
        source=read.source,
        parts=len(model.root.children),
        solids=solids,
        length_m=high - low,
        dry_mass_kg=model.mass_summary().dry_mass_kg,
        notes=notes,
        lines=lines,
    )
    return model, report


def _build_part(
    assignment: Assignment,
    pieces: list[StepComponent],
    axis: int,
    station: float,
    finish: float,
    station_of,
    material: str,
    notes: list[str],
):
    """One model component from one assignment."""
    kind = assignment.part_type
    length = max(finish - station, 1e-6)

    if kind is PartType.FINSET:
        fins = measure_finset(pieces, axis)
        # Leading edge is the forward end in station terms, which is the aft
        # end in file terms whenever the vehicle points the other way.
        root_le = min(station_of(v) for v in fins["root_axial_m"])
        tip_le = min(station_of(v) for v in fins["tip_axial_m"])
        sweep = tip_le - root_le
        part = FinSet(
            name=assignment.name,
            count=fins["count"],
            root_chord_m=fins["root_chord_m"],
            tip_chord_m=fins["tip_chord_m"],
            span_m=fins["span_m"],
            sweep_m=sweep,
            thickness_m=fins["thickness_m"],
            station_m=root_le,
        )
        part.material = "g10_fiberglass"
        return part

    if kind is PartType.PROTUBERANCE:
        part = Protuberance(name=assignment.name)
        # Membership, not a ``has_parm`` method: the container never had
        # one, and every protuberance assignment raised AttributeError.
        if "station" in part:
            part.set("station", station)
        return part

    if kind is PartType.INTERNAL:
        volume = sum(c.volume_m3 for c in pieces)
        return PointMass(
            name=assignment.name,
            mass_kg=volume * _density(material),
            station_m=0.5 * (station + finish),
        )

    if kind is PartType.TANK:
        tank = measure_tank(pieces[0], axis)
        ratio = 0.707
        if tank["dome_measured"]:
            ratio = min(max(2.0 * tank["dome_height_m"] / max(tank["diameter_m"], 1e-9), 0.1), 1.0)
        else:
            notes.append(
                f"{assignment.name}: the solid has no dome, so it was modelled "
                f"with flat ends and a nominal dome ratio."
            )
        barrel = max(length - 2.0 * (0.5 * tank["diameter_m"] * ratio), 1e-6)
        part = Tank(
            name=assignment.name,
            diameter_m=tank["diameter_m"],
            barrel_length_m=barrel,
            dome_ratio=ratio,
            station_m=station,
        )
        part.material = material
        return part

    # NOSE, BODY and INTERTANK are all stacks of sections: they differ in what
    # generates the sections and, for the tree, in what they are called.
    part = Stack(name=assignment.name)
    part.material = material
    if kind is PartType.NOSE:
        nose = measure_nose(pieces[0], axis)
        part.add_nose(nose["profile"], nose["length_m"], nose["base_diameter_m"])
        part.shift_m(station - part.station_range_m()[0])
        if nose["profile_residual_m"] > 0.002:
            notes.append(
                f"{assignment.name}: closest analytic family is "
                f"{nose['profile'].value}, but it is off by "
                f"{nose['profile_residual_m'] * 1000:.1f} mm RMS."
            )
        return part

    body = measure_body(pieces[0], axis)
    forward, aft = body["radius_low_m"], body["radius_high_m"]
    # _axial_profile runs along the file axis; stations may run the other way.
    if station_of(pieces[0].extent_m(axis)[0]) > station_of(pieces[0].extent_m(axis)[1]):
        forward, aft = aft, forward
    part.add_section(XSec(station, XSecShape.CIRCLE, 2.0 * forward, 2.0 * forward))
    part.add_section(XSec(finish, XSecShape.CIRCLE, 2.0 * aft, 2.0 * aft))
    return part


def import_assembly(
    path: str | Path,
    assignments: list[Assignment] | None = None,
    name: str = "",
    material: str = "aluminium_6061_t6",
) -> tuple[VehicleModel, BuildReport, AssemblyRead]:
    """Read, assign and build in one call.

    With no assignments the suggestion is used, which is what a script or a
    test wants; the application passes what the person chose in the dialog.
    """
    read = read_assembly(path)
    chosen = assignments if assignments is not None else suggest(read)
    model, report = build_model(read, chosen, name=name, material=material)
    return model, report, read
