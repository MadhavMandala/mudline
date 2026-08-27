"""The vehicle document: a component tree plus the roll-ups every view needs.

This is the "document" the application is built around. The tree owns the
model; the viewport, the property editor, the mass solver and the analysis
chain are all views onto it. That is the structural change from the previous
GUI, which had no document at all -- only a viewer and a set of import
commands, which is why nothing could be inspected or edited.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from parametric.components import (
    Component,
    FinSet,
    Motor,
    PointMass,
    Protuberance,
    Stack,
    Tank,
    Wing,
)
from parametric.xsec import NoseProfile, XSec, XSecShape

FORMAT = "mudline.parametric.v1"
#: What the same format was called before the tool was renamed. Files are
#: read from disk long after they are written -- refusing a vehicle over a
#: string that only ever named the program would be a rename breaking data.
LEGACY_FORMATS = ("rocket_model.parametric.v1",)

#: What the vehicle is being authored as.
#:
#: This drives the *editor* -- which parts the Add menu offers and in whose
#: vocabulary -- and nothing else. It is deliberately not consulted by the mass,
#: aerodynamic or trajectory code, which ask the model what it *contains*
#: instead: a wing means the aerodynamics needs sideslip, a rail means the
#: trajectory opens on a rail. Behaviour derived from contents survives a
#: vehicle that is both, and a rocket-plane is exactly that.
ROCKET = "rocket"
AIRCRAFT = "aircraft"
VEHICLE_CLASSES = (ROCKET, AIRCRAFT)


@dataclass
class MassSummary:
    """Analytic mass roll-up over the tree."""

    dry_mass_kg: float
    dry_mass_with_growth_kg: float
    propellant_mass_kg: float
    #: Station of the dry structure alone -- the vehicle at burnout.
    cg_station_m: float
    per_component_kg: dict[str, float]
    #: Station of the propellant load. Needed because propellant sits well aft
    #: of the dry CG, so a loaded vehicle balances somewhere quite different.
    propellant_cg_station_m: float = 0.0

    @property
    def wet_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg

    @property
    def wet_cg_station_m(self) -> float:
        """CG with the motor loaded -- the vehicle on the rail.

        This is the case stability has to be judged on. On the standard rocket
        the propellant is 41% of wet mass and sits 316 mm behind the dry CG,
        which moves the balance point 128 mm aft and costs 1.3 calibres of
        margin. Judging a design on the burnout CG flatters it at exactly the
        moment it is most likely to go unstable.
        """
        total = self.wet_mass_kg
        if total <= 0:
            return self.cg_station_m
        return (
            self.cg_station_m * self.dry_mass_kg
            + self.propellant_cg_station_m * self.propellant_mass_kg
        ) / total

    @property
    def mass_ratio(self) -> float:
        return self.wet_mass_kg / self.dry_mass_kg if self.dry_mass_kg > 0 else 0.0


class VehicleModel:
    """A parametric vehicle: the root of the component tree."""

    def __init__(self, name: str = "Vehicle", revision: str = "A",
                 vehicle_class: str = ROCKET):
        self.name = name
        self.revision = revision
        self.description = ""
        self.vehicle_class = (
            vehicle_class if vehicle_class in VEHICLE_CLASSES else ROCKET
        )
        self.root = Component("vehicle")
        self.root.name = name
        #: How far a clipped joint interpenetrates, in metres.
        #:
        #: Deliberately not zero. Two coincident faces are the classic thing
        #: that makes a CAD boolean or a CFD mesher fail, and the outer mould
        #: line is meant to be unioned into one watertight solid eventually.
        #: Half a millimetre is invisible against any real part and leaves each
        #: component's own analytic volume -- which is where mass comes from --
        #: untouched.
        self.clip_overlap_m = 0.0005

    # ------------------------------------------------------------------
    # Tree access
    # ------------------------------------------------------------------

    def add(self, component: Component) -> Component:
        return self.root.add(component)

    def walk(self) -> Iterator[Component]:
        yield from self.root.walk()

    def find(self, name: str) -> Component | None:
        return self.root.find(name)

    def components_of(self, kind: str) -> list[Component]:
        return [c for c in self.walk() if c.kind == kind]

    @property
    def stacks(self) -> list[Stack]:
        return [c for c in self.walk() if isinstance(c, Stack)]

    @property
    def fin_sets(self) -> list[FinSet]:
        return [c for c in self.walk() if isinstance(c, FinSet)]

    @property
    def wings(self) -> list[Wing]:
        return [c for c in self.walk() if isinstance(c, Wing)]

    @property
    def tanks(self) -> list[Tank]:
        return [c for c in self.walk() if isinstance(c, Tank)]

    @property
    def has_lifting_surfaces(self) -> bool:
        """Whether the aerodynamics has to account for a wing.

        The capability question the analysis code should ask instead of reading
        ``vehicle_class``: a rocket-plane answers yes and is still a rocket.
        """
        return bool(self.wings)

    @property
    def point_masses(self) -> list[PointMass]:
        return [c for c in self.walk() if isinstance(c, PointMass)]

    @property
    def protuberances(self) -> list[Protuberance]:
        return [c for c in self.walk() if isinstance(c, Protuberance)]

    @property
    def motors(self) -> list[Motor]:
        return [c for c in self.walk() if isinstance(c, Motor)]

    @property
    def dirty(self) -> bool:
        return self.root.any_dirty()

    def mark_clean(self) -> None:
        self.root.mark_all_clean()

    # ------------------------------------------------------------------
    # Derived geometry
    # ------------------------------------------------------------------

    def station_range_m(self) -> tuple[float, float]:
        """Forward-most and aft-most station of anything in the vehicle.

        Not assumed to start at zero. The origin is a datum the vehicle is
        placed against, so a part may legitimately sit forward of it at a
        negative station.
        """
        # The root has no station of its own; counting it pinned the range
        # at zero, so a vehicle whose nose sat at 0.5 m reported half a
        # metre of length it did not have.
        spans = [c.station_range_m() for c in self.walk() if c is not self.root]
        if not spans:
            return (0.0, 0.0)
        return (min(s[0] for s in spans), max(s[1] for s in spans))

    @property
    def total_length_m(self) -> float:
        low, high = self.station_range_m()
        return max(high - low, 0.0)

    @property
    def max_diameter_m(self) -> float:
        return max((s.max_diameter_m for s in self.stacks), default=0.0)

    @property
    def reference_area_m2(self) -> float:
        return float(np.pi * (0.5 * self.max_diameter_m) ** 2)

    @property
    def fineness_ratio(self) -> float:
        diameter = self.max_diameter_m
        return self.total_length_m / diameter if diameter > 0 else 0.0

    @property
    def max_span_m(self) -> float:
        spans = [
            2.0 * (fins.body_radius_m() + fins.get("span")) for fins in self.fin_sets
        ]
        return max(spans + [self.max_diameter_m])

    def nozzle_exit_diameter_m(self) -> float:
        """Exit diameter of the first motor's nozzle, for power-on base drag.

        RASAero cuts base drag while the motor burns in proportion to the
        nozzle exit area over the base area, and reads the exit diameter
        from the project file. Nothing here ever wrote it, so every vehicle
        built in the tool was solved with no nozzle at all: its power-on
        drag column was identical to power-off, and the checkbox choosing
        between them did nothing.

        Clamped to the body diameter. The area is derived from the Isp pair
        when the motor declares none, and an implausible pair can imply an
        exit wider than the vehicle, which a nozzle cannot be.
        """
        motors = self.motors
        if not motors:
            return 0.0
        area = motors[0].effective_nozzle_area_m2()
        if area <= 0.0:
            return 0.0
        return float(min(2.0 * np.sqrt(area / np.pi), self.max_diameter_m))

    def silhouette(self, samples: int = 400) -> tuple[np.ndarray, np.ndarray]:
        """Outer radius against station, across every stack."""
        # Sampled across the stations that exist, not 0..length: a vehicle whose
        # forward-most part sits at a negative station would otherwise have that
        # part fall outside the sample range entirely and vanish from the
        # silhouette, taking its normal force with it.
        low, high = self.station_range_m()
        if high <= low:
            low, high = 0.0, max(self.total_length_m, 1e-9)
        z = np.linspace(low, high, samples)
        radius = np.zeros_like(z)
        for stack in self.stacks:
            low, high = stack.station_range_m()
            inside = (z >= low) & (z <= high)
            if np.any(inside):
                radius[inside] = np.maximum(
                    radius[inside], [stack.radius_at(float(v)) for v in z[inside]]
                )
        return z, radius

    # ------------------------------------------------------------------
    # Mass
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Clipping
    # ------------------------------------------------------------------

    def clippable_components(self) -> list[Component]:
        """Parts that form the outer mould line, fore to aft."""
        parts = [c for c in self.walk() if c is not self.root and c.clippable]
        return sorted(parts, key=lambda c: c.station_range_m()[0])

    def clip_targets_for(self, component: Component) -> list[Component]:
        """What this part could legally be clipped to.

        Everything else on the mould line, minus anything downstream of it in
        the clip graph -- offering those would let you build a loop by picking
        from a menu, and a loop has no first part to position from.
        """
        by_path = {c.path: c for c in self.walk()}
        blocked = {component.path}
        frontier = [c for c in self.walk() if c.clip_to == component.path]
        while frontier:
            node = frontier.pop()
            if node.path in blocked:
                continue
            blocked.add(node.path)
            frontier.extend(
                c for c in self.walk() if c.clip_to == node.path
            )
        return [
            c for c in self.clippable_components()
            if c.path not in blocked and c.path in by_path
        ]

    def clip_station_for(self, component: Component) -> float | None:
        """Where a clipped part's forward end belongs, or None if it is free.

        The joint interpenetrates rather than meeting exactly. Coplanar faces
        are what make a later boolean union of the mould line fail, and the
        overlap is small enough that each part's own analytic volume -- which
        is what mass is taken from -- is unaffected at any scale that matters.
        """
        if not component.clip_to:
            return None
        target = next(
            (c for c in self.walk() if c.path == component.clip_to), None
        )
        if target is None:
            return None
        flush = bool(getattr(component, "clip_flush", True))
        low, high = (
            target.mating_range_m() if flush else target.station_range_m()
        )
        own_low, own_high = (
            component.mating_range_m() if flush else component.station_range_m()
        )
        # Everything is solved for the forward *extreme*, since that is what
        # set_forward_station_m moves, so carry the offset from that extreme to
        # whichever of this part's own faces is doing the joining.
        forward_extreme = component.station_range_m()[0]
        lead = own_low - forward_extreme
        trail = own_high - forward_extreme

        if getattr(component, "clip_side", "aft") == "forward":
            # In front of the target: this part's aft joint lands just inside
            # the target's forward joint.
            return low + self.clip_overlap_m - trail
        return high - self.clip_overlap_m - lead

    def apply_clips(self) -> list[str]:
        """Move every locked part onto what it is clipped to.

        Resolved target-first, so a chain of three settles in one pass rather
        than needing a second to catch up. Returns whatever went wrong.
        """
        problems: list[str] = []
        by_path = {c.path: c for c in self.walk()}
        done: set[str] = set()

        def resolve(component: Component, seen: frozenset) -> None:
            if component.path in done:
                return
            target_path = component.clip_to
            if not target_path:
                done.add(component.path)
                return
            if component.path in seen:
                problems.append(f"{component.name}: clip chain loops back on itself.")
                done.add(component.path)
                return
            target = by_path.get(target_path)
            if target is None:
                problems.append(
                    f"{component.name}: clipped to a part that no longer exists."
                )
                component.clip_to = None
                component.clip_locked = False
                done.add(component.path)
                return
            resolve(target, seen | {component.path})
            if component.clip_locked:
                station = self.clip_station_for(component)
                if station is not None:
                    component.set_forward_station_m(station)
            done.add(component.path)

        for part in self.clippable_components():
            resolve(part, frozenset())
        return problems

    def clip(self, component: Component, target: Component,
             lock: bool = False, side: str = "aft") -> bool:
        """Clip a part onto another and snap it there now.

        ``side`` is which end of the target to hang off: "aft" puts this part
        behind it, "forward" puts it in front.
        """
        if component is target or not component.clippable:
            return False
        component.clip_to = target.path
        component.clip_locked = bool(lock)
        component.clip_side = side if side in ("aft", "forward") else "aft"
        station = self.clip_station_for(component)
        if station is None:
            return False
        component.set_forward_station_m(station)
        return True

    def unclip(self, component: Component) -> None:
        component.clip_to = None
        component.clip_locked = False

    def mass_summary(self) -> MassSummary:
        per_component: dict[str, float] = {}
        moment = 0.0
        total = 0.0
        with_growth = 0.0

        for component in self.walk():
            # Motors are no longer skipped outright. They carry a declared dry
            # mass now, which defaults to zero -- and the zero-mass test just
            # below already drops anything that has not been given one, so a
            # solid motor whose case is the tube around it behaves exactly as
            # it did while a liquid engine can finally weigh something.
            mass = component.mass_kg()
            if mass <= 0:
                continue
            per_component[component.name] = mass
            low, high = component.station_range_m()
            centre = 0.5 * (low + high)
            if isinstance(component, Stack):
                centre = self._stack_centroid(component)
            elif isinstance(component, FinSet):
                centre = component.planform_centroid_station_m
            moment += mass * centre
            total += mass
            with_growth += (
                component.mass_with_growth_kg
                if isinstance(component, PointMass) else mass
            )

        # Propellant is rolled up separately: it is not structure, it drains
        # in flight, and the loop above skips anything of zero mass -- which a
        # Motor is, since its case is modelled by the tube around it.
        propellant = 0.0
        propellant_moment = 0.0
        for motor in self.motors:
            # A tank-fed engine burns propellant it does not carry. Counting
            # its load here as well as the tank's would double the mass.
            if getattr(motor, "feed", "grain") == "tanks":
                continue
            load = motor.get("propellant_mass")
            propellant += load
            propellant_moment += load * motor.centroid_station_m
        for tank in self.tanks:
            load = tank.get("propellant_mass")
            propellant += load
            propellant_moment += load * tank.centroid_station_m

        return MassSummary(
            dry_mass_kg=total,
            dry_mass_with_growth_kg=with_growth,
            propellant_mass_kg=propellant,
            cg_station_m=(moment / total if total > 0 else 0.0),
            per_component_kg=per_component,
            propellant_cg_station_m=(
                propellant_moment / propellant if propellant > 0 else 0.0
            ),
        )

    @staticmethod
    def _stack_centroid(stack: Stack) -> float:
        """Centroid of a stack's material, which is not its mid-length.

        Weighted by the *material* area at each section -- the shell -- not
        by the enclosed area. A shell's material is at the surface, so on
        an ogive nose the enclosed-area centroid sat 26 mm (6% of the nose)
        aft of the true one, and the meshed solve disagreed with the
        analytic CG by exactly that. This reproduces the meshed centroid to
        a tenth of a millimetre.
        """
        stations, areas = stack.material_areas()
        if len(stations) < 2:
            return stack.station_range_m()[0]
        weight = np.trapezoid(areas, stations)
        if weight <= 0:
            low, high = stack.station_range_m()
            return 0.5 * (low + high)
        return float(np.trapezoid(areas * stations, stations) / weight)

    @property
    def propellant_mass_kg(self) -> float:
        """The load aboard, tanks included -- the same number the roll-up uses."""
        return self.mass_summary().propellant_mass_kg

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.stacks:
            problems.append("Vehicle has no body: add a Stack component.")

        names: set[str] = set()
        for component in self.walk():
            if component is self.root:
                continue
            if component.name in names:
                problems.append(f"Duplicate component name {component.name!r}.")
            names.add(component.name)
            # Components that can check themselves, do. A tank knows whether
            # the load it has been given actually fits inside it.
            checker = getattr(component, "validate", None)
            if callable(checker) and component is not self.root:
                problems.extend(checker())

        problems.extend(self.apply_clips())

        # Gaps and interpenetrations along the mould line that are too big to
        # be the clip overlap. Reported rather than fixed: a deliberate gap
        # between a tank and a skirt is a design decision, and a part that has
        # been dragged through its neighbour is a mistake, and only the person
        # who did it knows which.
        chain = self.clippable_components()
        for ahead, behind in zip(chain, chain[1:]):
            # A pair the clip put there is already where it was asked to be,
            # and a flush joint deliberately buries a tank's dome inside its
            # neighbour. Measuring those at the extremes reported the dome
            # height as an overlap error on every single joint.
            if behind.clip_to == ahead.path or ahead.clip_to == behind.path:
                continue
            gap = behind.station_range_m()[0] - ahead.station_range_m()[1]
            if gap < -max(self.clip_overlap_m * 4.0, 1e-4):
                problems.append(
                    f"{behind.name} overlaps {ahead.name} by {-gap * 1000:.1f} mm."
                )

        # A tank-fed engine with no tanks has nothing to burn.
        for motor in self.motors:
            if getattr(motor, "feed", "grain") != "tanks":
                continue
            if not self.tanks:
                problems.append(
                    f"{motor.name}: fed from tanks, but the vehicle has none."
                )
                continue
            ratio = motor.get("mixture_ratio")
            if ratio <= 0:
                continue
            fuel = sum(t.get("propellant_mass") for t in self.tanks
                       if t.contents != "oxidizer")
            ox = sum(t.get("propellant_mass") for t in self.tanks
                     if t.contents == "oxidizer")
            if fuel <= 0 or ox <= 0:
                problems.append(
                    f"{motor.name}: mixture ratio {ratio:g} declared, but the "
                    f"tanks are not labelled both fuel and oxidizer; draining "
                    f"falls back to proportional."
                )
                continue
            # At O/F = MR, exhausting the fuel consumes fuel*(1+MR) in total
            # and exhausting the oxidizer consumes ox*(1+MR)/MR; whichever is
            # smaller is the side that runs dry, stranding the rest.
            burn_by_fuel = fuel * (1.0 + ratio)
            burn_by_ox = ox * (1.0 + ratio) / ratio
            usable = min(burn_by_fuel, burn_by_ox)
            stranded = fuel + ox - usable
            if stranded > 0.02 * (fuel + ox):
                limiting, leftover = (
                    ("oxidizer", "fuel") if burn_by_ox < burn_by_fuel
                    else ("fuel", "oxidizer")
                )
                problems.append(
                    f"{motor.name}: at O/F {ratio:g} the {limiting} runs out "
                    f"first, stranding {stranded:.1f} kg of {leftover}."
                )

        for stack in self.stacks:
            if len(stack.sections) < 2:
                problems.append(
                    f"{stack.name}: needs at least two cross-sections to loft."
                )
            stations = [s.station_m for s in stack.sorted_sections()]
            if len(set(stations)) != len(stations):
                problems.append(
                    f"{stack.name}: two sections share a station; the loft is ambiguous."
                )
            # Checked against the *largest* section, not the smallest. A wall
            # that closes off the smallest section is normal and desirable: it
            # is what leaves a nose tip solid. A wall that closes off the
            # largest section means the whole body is solid, which is a real
            # error.
            thickness = stack.get("wall_thickness")
            largest = max(
                (max(s.width_m, s.height_m) for s in stack.sections), default=0.0
            )
            if thickness > 0 and largest > 0 and 2.0 * thickness >= largest:
                problems.append(
                    f"{stack.name}: wall thickness {thickness:.4f} m closes off even "
                    f"the largest section ({largest:.4f} m across); the body would be "
                    f"entirely solid."
                )

        for fins in self.fin_sets:
            if not isinstance(fins.parent, Stack):
                problems.append(f"{fins.name}: must be attached to a Stack.")
                continue
            # The *root* is the attachment, and it is the root that has to
            # land on the parent. A swept tip trailing aft of the airframe is
            # normal -- the panel is cantilevered into free air, and plenty of
            # real vehicles are built that way. Checking the whole station
            # range, tip included, rejected those: a 12-degree sweep on a fin
            # flush with the tail was refused for overhanging by the sweep.
            low, high = fins.parent.station_range_m()
            root_low = fins.get("station")
            root_high = root_low + fins.get("root_chord")
            if root_high > high + 1e-9 or root_low < low - 1e-9:
                problems.append(
                    f"{fins.name}: root chord spans {root_low:.3f}-{root_high:.3f} m, "
                    f"off the end of its parent {fins.parent.name} "
                    f"({low:.3f}-{high:.3f} m). The root is what attaches; a "
                    f"swept tip may trail past the tail."
                )

        for motor in self.motors:
            capacity = self._motor_capacity_m3(motor)
            needed = motor.propellant_volume_m3
            if needed > 0 and capacity > 0 and needed > capacity:
                problems.append(
                    f"{motor.name}: {motor.get('propellant_mass'):.1f} kg needs "
                    f"{needed:.4f} m³ but the body holds {capacity:.4f} m³ there."
                )
        return problems

    def _motor_capacity_m3(self, motor: Motor) -> float:
        """Internal volume of the body over the grain's station range."""
        low, high = motor.grain_range_m()
        if high <= low:
            return 0.0
        stack = motor.parent if isinstance(motor.parent, Stack) else None
        if stack is None:
            stack = self.stacks[0] if self.stacks else None
        if stack is None:
            return 0.0
        thickness = stack.get("wall_thickness")
        stations = np.linspace(low, high, 40)
        radii = np.array([max(stack.radius_at(float(s)) - thickness, 0.0)
                          for s in stations])
        return float(np.trapezoid(np.pi * radii ** 2, stations))

    def require_valid(self) -> "VehicleModel":
        problems = self.validate()
        if problems:
            raise ValueError(
                f"Vehicle {self.name!r} is not valid:\n  - " + "\n  - ".join(problems)
            )
        return self

    # ------------------------------------------------------------------

    def summary(self) -> str:
        mass = self.mass_summary()
        lines = [
            f"{self.name} (rev {self.revision})",
            f"  Length          {self.total_length_m:.3f} m",
            f"  Max diameter    {self.max_diameter_m:.3f} m",
            f"  Fineness ratio  {self.fineness_ratio:.1f}",
            f"  Span over fins  {self.max_span_m:.3f} m",
            f"  Reference area  {self.reference_area_m2:.4f} m²",
            f"  Dry mass        {mass.dry_mass_kg:.2f} kg",
            f"  Propellant      {mass.propellant_mass_kg:.2f} kg",
            f"  Wet mass        {mass.wet_mass_kg:.2f} kg",
            f"  Mass ratio      {mass.mass_ratio:.2f}",
            f"  Analytic CG     {mass.cg_station_m:.3f} m from nose tip",
        ]
        sections = sum(len(s.sections) for s in self.stacks)
        lines.append(
            f"  Components      {len(list(self.walk())) - 1} "
            f"({sections} cross-sections)"
        )
        return "\n".join(lines)

    def tree_text(self) -> str:
        """The component tree as text, for a console or a test."""
        lines: list[str] = []

        def render(node: Component, depth: int) -> None:
            if node is not self.root:
                low, high = node.station_range_m()
                mass = node.mass_kg()
                extra = f"{mass:8.3f} kg" if mass > 0 else " " * 11
                lines.append(
                    f"{'  ' * depth}{node.name:<22}{node.kind:<10}"
                    f"{low:7.3f}-{high:6.3f} m {extra}"
                )
            for child in node.children:
                render(child, depth + 1)

        render(self.root, 0)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        def encode(node: Component) -> dict:
            data = {
                "name": node.name,
                "kind": node.kind,
                "material": node.material,
                "aero_role": node.aero_role.value,
                "parms": node.parm_values(),
                "children": [encode(c) for c in node.children],
            }
            if isinstance(node, Stack):
                # Only written when the nose family was declared, so files
                # from before this existed stay byte-identical.
                profile = getattr(node, "nose_profile", None)
                if profile is not None:
                    data["nose_profile"] = getattr(profile, "value", str(profile))
                    data["nose_tip_radius_m"] = float(
                        getattr(node, "nose_tip_radius_m", 0.0)
                    )
                data["sections"] = [
                    {
                        "name": s.name,
                        "shape": s.shape.value,
                        "parms": s.parm_values(),
                    }
                    for s in node.sorted_sections()
                ]
            if node.mass_override_kg is not None:
                data["mass_override_kg"] = float(node.mass_override_kg)
            if node.imported:
                data["imported"] = True
            if node.clip_to:
                data["clip_to"] = node.clip_to
                data["clip_locked"] = bool(node.clip_locked)
                data["clip_side"] = node.clip_side
                data["clip_flush"] = bool(node.clip_flush)
            if isinstance(node, Protuberance):
                data["shape"] = node.shape.value
            if isinstance(node, Wing):
                data["symmetric"] = bool(node.symmetric)
            if isinstance(node, Tank):
                data["contents"] = node.contents
            if isinstance(node, Motor):
                data["curve"] = [[float(t), float(f)] for t, f in node.curve]
                data["burn_geometry"] = node.burn_geometry
                data["feed"] = node.feed
                if node.propulsion_file:
                    data["propulsion_file"] = node.propulsion_file
            return data

        return {
            "format": FORMAT,
            "name": self.name,
            "revision": self.revision,
            "description": self.description,
            "vehicle_class": self.vehicle_class,
            "clip_overlap_m": float(self.clip_overlap_m),
            "tree": encode(self.root),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "VehicleModel":
        if data.get("format") not in (FORMAT, *LEGACY_FORMATS):
            raise ValueError(
                f"Not a parametric vehicle: expected {FORMAT!r}, got {data.get('format')!r}"
            )
        # Files written before vehicle class existed have no key and are
        # rockets, which is what the default already says.
        model = cls(data["name"], data.get("revision", "A"),
                    data.get("vehicle_class", ROCKET))
        model.description = data.get("description", "")
        model.clip_overlap_m = float(data.get("clip_overlap_m", 0.0005))

        def decode(node_data: dict) -> Component:
            kind = node_data.get("kind", "component")
            name = node_data["name"]
            parms = node_data.get("parms", {})

            if kind == "stack":
                node: Component = Stack(name)
                node.apply_values(parms)
                declared = node_data.get("nose_profile")
                if declared is not None:
                    from parametric.xsec import NoseProfile

                    try:
                        node.nose_profile = NoseProfile(declared)
                    except ValueError:
                        # An unknown family means a newer file; fall back to
                        # fitting rather than refusing to open the project.
                        node.nose_profile = None
                    node.nose_tip_radius_m = float(
                        node_data.get("nose_tip_radius_m", 0.0)
                    )
                for section_data in node_data.get("sections", []):
                    section = XSec(0.0, XSecShape(section_data["shape"]),
                                   name=section_data.get("name", "xsec"))
                    section.apply_values(section_data.get("parms", {}))
                    node.add_section(section)
            elif kind == "finset":
                node = FinSet(name)
                node.apply_values(parms)
            elif kind == "wing":
                node = Wing(name)
                node.symmetric = bool(node_data.get("symmetric", True))
                node.apply_values(parms)
            elif kind == "tank":
                node = Tank(name)
                node.apply_values(parms)
                # Files written before contents existed read as fuel, which is
                # harmless: a mixture ratio needs both labels present to bite,
                # so old vehicles keep draining in proportion to their loads.
                node.contents = node_data.get("contents", "fuel")
            elif kind == "protuberance":
                node = Protuberance(name)
                node.apply_values(parms)
                if node_data.get("shape"):
                    from parametric.roles import HoernerShape

                    node.shape = HoernerShape(node_data["shape"])
            elif kind == "pointmass":
                node = PointMass(name, 0.0, 0.0)
                node.apply_values(parms)
            elif kind == "motor":
                node = Motor(name)
                node.apply_values(parms)
                node.propulsion_file = node_data.get("propulsion_file")
                # Files written before burn geometry existed are solids, which
                # is what the fixed-centroid model already assumed.
                node.burn_geometry = node_data.get("burn_geometry", "radial")
                # Files written before tanks existed are all solid grains.
                node.feed = node_data.get("feed", "grain")
                if node_data.get("curve"):
                    node.set_curve(node_data["curve"])
            else:
                node = Component(name)

            node.material = node_data.get("material", node.material)
            override = node_data.get("mass_override_kg")
            node.mass_override_kg = None if override is None else float(override)
            node.clip_to = node_data.get("clip_to")
            node.clip_locked = bool(node_data.get("clip_locked", False))
            node.imported = bool(node_data.get("imported", False))
            # Files written before the side existed all clipped behind.
            node.clip_side = node_data.get("clip_side", "aft")
            node.clip_flush = bool(node_data.get("clip_flush", True))
            role = node_data.get("aero_role")
            if role:
                from parametric.roles import AeroRole

                node.aero_role = AeroRole(role)
            for child_data in node_data.get("children", []):
                node.add(decode(child_data))
            return node

        model.root = decode(data["tree"])
        return model

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "VehicleModel":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
