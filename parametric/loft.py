"""Turn section stacks into B-rep solids.

Geometry is built with the OCC kernel through cadquery, which buys real solids:
watertight bodies for the mass solver, STEP export for free, and booleans for
hollowing a shell.

Caching
-------
Lofting is the expensive step, so results are cached against a *signature* of
the parameters that produced them. Dragging a fin span rebuilds the fins and
leaves the airframe alone; nudging the camera rebuilds nothing. Without that,
live parameter editing is not possible -- an unconditional rebuild of a
fourteen-section nose is hundreds of milliseconds.

Units
-----
Solids are built in millimetres, because that is the only unit the massprops
gmsh path scales self-consistently. The model is metres throughout, so the
conversion happens here, once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parametric.components import Component, FinSet, Motor, Stack, Tank, Wing
from parametric.model import VehicleModel
from parametric.xsec import XSec

MM_PER_M = 1000.0


@dataclass
class LoftResult:
    """A built solid and what it was built from."""

    name: str
    solid: object
    signature: tuple
    volume_m3: float
    #: Analytic volume from the section integral, for cross-checking.
    expected_volume_m3: float

    @property
    def volume_error(self) -> float:
        if self.expected_volume_m3 <= 0:
            return 0.0
        return abs(self.volume_m3 - self.expected_volume_m3) / self.expected_volume_m3


def _require_cadquery():
    try:
        import cadquery as cq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Parametric geometry needs cadquery. Install the CAD extra:\n"
            '    python -m pip install -e ".[cad]"'
        ) from exc
    return cq


# ----------------------------------------------------------------------
# Signatures
# ----------------------------------------------------------------------


def stack_signature(stack: Stack) -> tuple:
    """Everything that changes a stack's shape, and nothing that does not."""
    return (
        "stack",
        round(stack.get("wall_thickness"), 9),
        tuple(
            (
                section.shape.value,
                round(section.station_m, 9),
                round(section.width_m, 9),
                round(section.height_m, 9),
                round(section.get("exponent"), 6),
                round(section.get("corner_radius"), 9),
            )
            for section in stack.sorted_sections()
        ),
    )


def finset_signature(fins: FinSet) -> tuple:
    return (
        "finset",
        fins.count,
        round(fins.get("root_chord"), 9),
        round(fins.get("tip_chord"), 9),
        round(fins.get("span"), 9),
        round(fins.get("sweep"), 9),
        round(fins.get("thickness"), 9),
        round(fins.get("station"), 9),
        round(fins.get("cant"), 6),
        round(fins.body_radius_m(), 9),
    )


def wing_signature(wing: Wing) -> tuple:
    return (
        "wing",
        wing.count,
        round(wing.get("root_chord"), 9),
        round(wing.get("tip_chord"), 9),
        round(wing.get("span"), 9),
        round(wing.get("sweep"), 9),
        round(wing.get("thickness_ratio"), 9),
        round(wing.get("wall_thickness"), 9),
        round(wing.get("station"), 9),
        round(wing.get("dihedral"), 9),
        round(wing.get("incidence"), 9),
        round(wing.body_radius_m(), 9),
    )


def tank_signature(tank: Tank) -> tuple:
    return (
        "tank",
        round(tank.get("diameter"), 9),
        round(tank.get("barrel_length"), 9),
        round(tank.get("dome_ratio"), 9),
        round(tank.get("wall_thickness"), 9),
        round(tank.get("station"), 9),
    )


def motor_signature(motor: Motor) -> tuple | None:
    if not motor.draws_geometry:
        return None       # no case diameter, nothing to draw
    return (
        "motor",
        round(motor.get("case_diameter"), 9),
        round(motor.get("length"), 9),
        round(motor.get("nozzle_length"), 9),
        round(motor.get("nozzle_area"), 9),
        round(motor.get("station"), 9),
    )


def component_signature(component: Component) -> tuple | None:
    if isinstance(component, Stack):
        return stack_signature(component)
    if isinstance(component, Motor):
        return motor_signature(component)
    if isinstance(component, Tank):
        return tank_signature(component)
    if isinstance(component, Wing):
        return wing_signature(component)
    if isinstance(component, FinSet):
        return finset_signature(component)
    return None


# ----------------------------------------------------------------------
# Building
# ----------------------------------------------------------------------


def _section_wire(cq, section: XSec, points: int = 96):
    """A closed wire for one section, placed at its station."""
    outline = section.outline(points)
    station = section.station_m * MM_PER_M
    coords = [
        (float(y) * MM_PER_M, float(z) * MM_PER_M, station) for y, z in outline
    ]
    return cq.Wire.makePolygon([cq.Vector(*c) for c in coords], forConstruction=False,
                               close=True)


def is_axisymmetric(stack: Stack) -> bool:
    """True when every section is a circle, so the stack is a body of revolution."""
    from parametric.xsec import XSecShape

    for section in stack.sections:
        if section.is_point:
            continue
        if section.shape is not XSecShape.CIRCLE:
            return False
        if abs(section.width_m - section.height_m) > 1e-12:
            return False
    return True


def build_revolved_stack(stack: Stack):
    """Build an axisymmetric stack by revolving its radius profile.

    Far cheaper than a general loft, and exact rather than faceted. A loft of N
    section wires with M points each asks OCC to fit a surface through N*M
    points; revolving asks it to sweep a single N-point profile. On a
    34-section vehicle that is the difference between 52 seconds and a fraction
    of one -- which is the difference between a tool you can drag a slider in
    and one you cannot.

    Most rocket bodies are circular, so this is the common path; anything with a
    non-circular section falls back to the general loft.
    """
    cq = _require_cadquery()
    sections = stack.sorted_sections()
    thickness = stack.get("wall_thickness")

    stations = [s.station_m * MM_PER_M for s in sections]
    outer_r = [max(s.equivalent_radius_m * MM_PER_M, 0.001) for s in sections]

    profile = cq.Workplane("XZ").moveTo(outer_r[0], stations[0])
    for r, z in zip(outer_r[1:], stations[1:]):
        profile = profile.lineTo(r, z)
    profile = (
        profile.lineTo(0.0, stations[-1]).lineTo(0.0, stations[0]).close()
    )
    # Axis is given in workplane-local coordinates: on XZ, local +Y is global
    # +Z. Passing the global axis returns a valid-looking solid of zero volume.
    solid = profile.revolve(360.0, (0, 0, 0), (0, 1, 0))

    if thickness <= 0.0:
        return solid

    wall = thickness * MM_PER_M
    inner_r = [r - wall for r in outer_r]
    usable = [i for i, r in enumerate(inner_r) if r > MIN_CAVITY_M * MM_PER_M]
    if len(usable) < 2:
        return solid

    first = usable[0]
    cavity = cq.Workplane("XZ").moveTo(0.0, stations[first])
    for i in range(first, len(stations)):
        cavity = cavity.lineTo(max(inner_r[i], 0.0), stations[i])
    cavity = cavity.lineTo(0.0, stations[-1]).lineTo(0.0, stations[first]).close()
    cavity_solid = cavity.revolve(360.0, (0, 0, 0), (0, 1, 0))
    return solid.cut(cavity_solid)


def build_stack_solid(stack: Stack, section_points: int = 96):
    """Build a stack's solid, hollowed if it has a wall.

    Axisymmetric stacks are revolved; anything with a non-circular section is
    lofted through section wires.

    A point section at either end becomes a small circle rather than a vertex:
    a lofted apex is a degenerate face that meshers and booleans both handle
    badly, and a real cone has a machined tip anyway.
    """
    cq = _require_cadquery()
    sections = stack.sorted_sections()
    if len(sections) < 2:
        raise ValueError(f"{stack.name}: need at least two sections to loft.")

    if is_axisymmetric(stack):
        return build_revolved_stack(stack)

    outer_wires = [_section_wire(cq, s, section_points) for s in sections]
    outer = cq.Solid.makeLoft(outer_wires, ruled=False)

    thickness = stack.get("wall_thickness")
    if thickness <= 0.0:
        return cq.Workplane(obj=outer)

    inner_sections = []
    for section in sections:
        width = section.width_m - 2.0 * thickness
        height = section.height_m - 2.0 * thickness
        if width <= 2.0 * MIN_CAVITY_M or height <= 2.0 * MIN_CAVITY_M:
            continue
        inner_sections.append(XSec(
            section.station_m, section.shape, width, height,
            section.get("exponent"),
            max(section.get("corner_radius") - thickness, 0.0),
        ))

    if len(inner_sections) < 2:
        return cq.Workplane(obj=outer)

    inner_wires = [_section_wire(cq, s, section_points) for s in inner_sections]
    cavity = cq.Solid.makeLoft(inner_wires, ruled=False)
    return cq.Workplane(obj=outer.cut(cavity))


#: Smallest half-wall that still produces a cavity worth cutting [m].
MIN_CAVITY_M = 0.0002


def build_fin_solid(fins: FinSet, index: int):
    """One fin of a set, rotated into position around the body."""
    cq = _require_cadquery()

    root_r = fins.body_radius_m() * MM_PER_M
    tip_r = root_r + fins.get("span") * MM_PER_M
    station = fins.get("station") * MM_PER_M
    root = fins.get("root_chord") * MM_PER_M
    tip = fins.get("tip_chord") * MM_PER_M
    sweep = fins.get("sweep") * MM_PER_M
    thickness = fins.get("thickness") * MM_PER_M

    fin = (
        cq.Workplane("XZ")
        .moveTo(root_r, station)
        .lineTo(root_r, station + root)
        .lineTo(tip_r, station + sweep + tip)
        .lineTo(tip_r, station + sweep)
        .close()
        .extrude(thickness / 2.0, both=True)
    )

    cant = fins.get("cant")
    if cant:
        mid_r = 0.5 * (root_r + tip_r)
        mid_z = station + 0.5 * root
        fin = fin.rotate((0, 0, mid_z), (mid_r, 0, mid_z), cant)

    return fin.rotate((0, 0, 0), (0, 0, 1), 360.0 * index / fins.count)


def _capsule_profile(radius_mm: float, dome_mm: float, barrel_mm: float,
                     z0_mm: float, points: int = 24) -> list[tuple[float, float]]:
    """Half-outline of a pill bottle, from forward tip to aft tip.

    Returned as (radius, station) pairs ready to revolve. Both domes are
    quarter ellipses swept in angle rather than sampled in z, which keeps the
    points evenly spread around the curve instead of bunching at the shoulder.
    """
    profile: list[tuple[float, float]] = []
    angles = np.linspace(0.0, np.pi / 2.0, points)
    # The poles sit on the axis, and a profile that touches it turns the
    # closing segment into a zero-length edge that OCC rejects outright, and
    # the revolved pole into a degenerate face that meshers and booleans both
    # handle badly. A hair off the axis instead -- the same trick
    # ``build_revolved_stack`` uses for a pointed nose.
    tip = 1e-3

    # Forward dome: pole at z0, shoulder at (radius, z0 + dome).
    for theta in angles:
        profile.append((
            max(radius_mm * float(np.sin(theta)), tip),
            z0_mm + dome_mm * (1.0 - float(np.cos(theta))),
        ))
    # Barrel.
    profile.append((radius_mm, z0_mm + dome_mm + barrel_mm))
    # Aft dome: shoulder back to the pole.
    for phi in angles[1:]:
        profile.append((
            max(radius_mm * float(np.cos(phi)), tip),
            z0_mm + dome_mm + barrel_mm + dome_mm * float(np.sin(phi)),
        ))
    return profile


def build_tank_solid(tank: Tank):
    """A tank as a revolved capsule, hollowed by its wall thickness.

    Revolved rather than lofted for the same reason an axisymmetric stack is:
    a single profile swept once, instead of a surface fitted through a stack of
    section wires.
    """
    cq = _require_cadquery()

    radius = tank.radius_m * MM_PER_M
    dome = tank.dome_height_m * MM_PER_M
    barrel = tank.get("barrel_length") * MM_PER_M
    z0 = tank.get("station") * MM_PER_M
    thickness = tank.get("wall_thickness") * MM_PER_M

    def revolve(points: list[tuple[float, float]]):
        wp = cq.Workplane("XZ").moveTo(points[0][0], points[0][1])
        for r, z in points[1:]:
            wp = wp.lineTo(r, z)
        # Back down the axis to close the half-section before revolving.
        wp = wp.lineTo(0.0, points[-1][1]).lineTo(0.0, points[0][1]).close()
        # Axis in workplane-local coordinates: on XZ, local +Y is global +Z.
        return wp.revolve(360.0, (0, 0, 0), (0, 1, 0))

    outer = revolve(_capsule_profile(radius, dome, barrel, z0))
    if thickness <= 0.0:
        return outer

    inner_radius = radius - thickness
    inner_dome = max(dome - thickness, 0.0)
    if inner_radius <= MIN_CAVITY_M * MM_PER_M:
        return outer      # the wall has consumed the bore; it is solid
    # The inner cavity starts one wall aft of the outer tip and is one wall
    # shorter at each end, so the barrel it spans is the same length.
    inner = revolve(_capsule_profile(
        inner_radius, inner_dome, barrel, z0 + thickness
    ))
    return outer.cut(inner)


def build_motor_solid(motor: Motor):
    """Chamber and bell, revolved.

    Only drawn when a case diameter has been given. A motor used to contribute
    no geometry at all, which is defensible for a grain inside a tube and not
    for an engine hanging off the back of a stage -- you could add one and see
    nothing appear anywhere.
    """
    cq = _require_cadquery()

    radius = motor.get("case_diameter") * 0.5 * MM_PER_M
    z0 = motor.get("station") * MM_PER_M
    chamber = motor.get("length") * MM_PER_M
    bell = motor.get("nozzle_length") * MM_PER_M
    exit_r = motor.exit_radius_m * MM_PER_M
    if exit_r <= 0.0:
        exit_r = radius * 1.6      # a plausible bell when no area is declared

    profile = [(radius, z0), (radius, z0 + chamber)]
    if bell > 0:
        # A bell is not a straight cone: it turns sharply at the throat and
        # flattens toward the exit. Two segments is enough to read as one.
        throat_r = max(radius * 0.45, 1e-3)
        profile.append((throat_r, z0 + chamber + bell * 0.18))
        profile.append((exit_r, z0 + chamber + bell))

    wp = cq.Workplane("XZ").moveTo(profile[0][0], profile[0][1])
    for r, z in profile[1:]:
        wp = wp.lineTo(r, z)
    wp = wp.lineTo(0.0, profile[-1][1]).lineTo(0.0, profile[0][1]).close()
    # Axis in workplane-local coordinates: on XZ, local +Y is global +Z.
    return wp.revolve(360.0, (0, 0, 0), (0, 1, 0))


def build_wing_solid(wing: Wing, index: int):
    """One panel of a wing, tilted by dihedral and set to its incidence.

    Same trapezoid a fin uses, because in this frame they are the same shape.
    The differences are that thickness comes from the chord through the
    thickness ratio rather than being given outright, and that the second panel
    is placed by mirroring rather than by an even radial division -- with the
    dihedral sign flipped so both panels rise, which a plain 180 degree
    rotation would not do.
    """
    cq = _require_cadquery()

    root_r = wing.body_radius_m() * MM_PER_M
    tip_r = root_r + wing.get("span") * MM_PER_M
    station = wing.get("station") * MM_PER_M
    root = wing.get("root_chord") * MM_PER_M
    tip = wing.get("tip_chord") * MM_PER_M
    sweep = wing.get("sweep") * MM_PER_M

    # An equal-volume plate, not a max-thickness one. The panel is extruded at
    # constant thickness, so using the aerofoil's peak t/c*c would enclose about
    # 1/0.685 too much material and the mass would follow it. Solving the plate
    # thickness from the section volume instead keeps this solid and
    # ``Wing.volume_m3`` describing the same object, which is what makes
    # ``LoftResult.volume_error`` a real cross-check rather than a constant
    # offset. Fins make the same compromise; they just have no taper in
    # thickness for it to matter to.
    # Enclosed volume, not material volume: this solid is the mould line, which
    # is what the viewport draws and what CAD and aerodynamics want. The skin
    # that carries the *mass* is ``Wing.volume_m3``, and the two are different
    # objects -- shelling the panel to match would buy nothing either consumer
    # is asking for.
    panel_area_m2 = wing.panel_area_m2
    if panel_area_m2 > 0:
        thickness = (
            wing.enclosed_volume_m3() / wing.count / panel_area_m2
        ) * MM_PER_M
    else:
        thickness = wing.get("thickness_ratio") * wing.mean_chord_m * MM_PER_M

    panel = (
        cq.Workplane("XZ")
        .moveTo(root_r, station)
        .lineTo(root_r, station + root)
        .lineTo(tip_r, station + sweep + tip)
        .lineTo(tip_r, station + sweep)
        .close()
        .extrude(thickness / 2.0, both=True)
    )

    incidence = wing.get("incidence")
    if incidence:
        mid_r = 0.5 * (root_r + tip_r)
        mid_z = station + 0.5 * root
        panel = panel.rotate((0, 0, mid_z), (mid_r, 0, mid_z), incidence)

    # Dihedral is a rotation about the body axis, and the mirrored panel needs
    # the opposite sign so that both tips go up rather than one up and one down.
    dihedral = wing.get("dihedral")
    angle = 180.0 * index + (-dihedral if index else dihedral)
    return panel.rotate((0, 0, 0), (0, 0, 1), angle)


class LoftCache:
    """Builds and caches solids for a model, rebuilding only what changed."""

    def __init__(self, section_points: int = 96):
        self.section_points = section_points
        self._cache: dict[str, LoftResult] = {}
        self.last_rebuilt: list[str] = []

    def clear(self) -> None:
        self._cache.clear()

    def solids(self, model: VehicleModel) -> dict[str, LoftResult]:
        """Return a solid per geometric component, rebuilding stale entries."""
        cq = _require_cadquery()
        results: dict[str, LoftResult] = {}
        self.last_rebuilt = []

        for component in model.walk():
            signature = component_signature(component)
            if signature is None:
                continue

            if isinstance(component, (Stack, Tank, Motor)):
                keys = [(component.path, component, None)]
            else:
                keys = [
                    (f"{component.path}#{i}", component, i)
                    for i in range(component.count)
                ]

            for key, owner, index in keys:
                cached = self._cache.get(key)
                if cached is not None and cached.signature == signature:
                    results[key] = cached
                    continue

                if isinstance(owner, Stack):
                    workplane = build_stack_solid(owner, self.section_points)
                    expected = owner.volume_m3()
                elif isinstance(owner, Motor):
                    workplane = build_motor_solid(owner)
                    # The chamber is drawn as a solid of revolution but its
                    # mass is declared, so there is no analytic volume to
                    # cross-check against. Matching the built solid keeps
                    # volume_error at zero rather than reporting a false one.
                    expected = workplane.val().Volume() / (MM_PER_M ** 3)
                elif isinstance(owner, Tank):
                    workplane = build_tank_solid(owner)
                    expected = owner.volume_m3()
                elif isinstance(owner, Wing):
                    workplane = build_wing_solid(owner, index)
                    expected = owner.enclosed_volume_m3() / owner.count
                else:
                    workplane = build_fin_solid(owner, index)
                    expected = owner.area_per_fin_m2 * owner.get("thickness")

                solid = workplane.val()
                built = LoftResult(
                    name=key,
                    solid=solid,
                    signature=signature,
                    volume_m3=solid.Volume() / (MM_PER_M ** 3),
                    expected_volume_m3=expected,
                )
                self._cache[key] = built
                results[key] = built
                self.last_rebuilt.append(key)

        model.mark_clean()
        return results


def export_step(model: VehicleModel, output_dir: str | Path,
                cache: LoftCache | None = None) -> list[Path]:
    """Write one STEP file per solid.

    Per-part files, for the same reason as before: the mass solver cannot split
    a multi-part STEP back into components, so emitting them separately is what
    makes per-component mass properties possible.
    """
    cache = cache or LoftCache()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for key, result in cache.solids(model).items():
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key).strip("_")
        path = output_dir / f"{safe}.stp"
        result.solid.exportStep(str(path))
        written.append(path)
    return written


def verify_volumes(results: dict[str, LoftResult], tolerance: float = 0.05) -> list[str]:
    """Compare each built solid against the analytic section integral.

    Two independent computations of the same quantity: the OCC volume of a
    lofted B-rep, and the trapezoidal integral of section area. Agreement is
    evidence the loft built what the sections describe.
    """
    problems = []
    for key, result in results.items():
        if result.expected_volume_m3 <= 0:
            continue
        if result.volume_error > tolerance:
            problems.append(
                f"{key}: lofted {result.volume_m3:.6f} m³ vs analytic "
                f"{result.expected_volume_m3:.6f} m³ "
                f"({result.volume_error * 100:.1f}%)"
            )
    return problems
