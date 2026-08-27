"""The rocket component library.

Four component types cover every vehicle this tool is meant to describe:

    Stack       a body of revolution built from a section stack -- nose, tube,
                transition, boattail and payload bulge are all this one thing
    FinSet      N identical fins attached to a parent at a station
    PointMass   mass with a position but no geometry worth modelling
    Motor       propellant, thrust curve and where the thrust acts

Components form a tree with attachment: a fin set attaches to a stack at a
fraction of its length, so moving or resizing the body carries the fins with
it. That parent/child relationship is the other half of the OpenVSP method, and
it is what makes a parametric edit propagate instead of leaving the model
inconsistent.

Coordinates match the rest of the toolchain: the vehicle axis is +Z, the origin
is the nose tip, and +Z points aft, so a component's station is its Z.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from parametric.parm import ParmContainer
from parametric.xsec import (
    NoseProfile,
    XSec,
    XSecShape,
    generate_nose_sections,
    generate_transition_sections,
    generate_tube_sections,
)


class Component(ParmContainer):
    """Base class: a named node in the vehicle tree."""

    kind = "component"

    #: Roles this kind of component may be given. Narrowing the choice per kind
    #: keeps the editor from offering "fin" for a body tube.
    allowed_roles: tuple = ()

    def __init__(self, name: str):
        super().__init__(name)
        self.children: list[Component] = []
        self.parent: Component | None = None
        self.material: str = "cfrp_quasi_isotropic"

        #: A measured mass, which wins over whatever the geometry implies.
        #: None means "compute it". This is the one fact a shape cannot carry:
        #: once hardware exists and has been weighed, the scale is right and
        #: the model is an estimate, so the scale should win -- while the
        #: geometry stays intact for aerodynamics and CAD.
        self.mass_override_kg: float | None = None
        self.visible: bool = True

        #: Built from imported CAD rather than authored here.
        #:
        #: An imported part is not editable and not designable: the STEP file is
        #: the source of truth for its shape, and a slider that silently made
        #: the model disagree with the CAD it came from would be worse than no
        #: slider. Re-importing the file is how an imported part changes.
        #: ``parametric.assembly_import`` sets this; nothing else should.
        self.imported: bool = False

        # What this component is aerodynamically. AUTO keeps the previous
        # inference, so declaring a role is opt-in; anything else overrides the
        # guess, and INTERNAL removes it from aerodynamics entirely while
        # leaving its mass in place.
        from parametric.roles import AeroRole

        self.aero_role: AeroRole = AeroRole.AUTO

        #: Path of the part this one butts onto, or None for a free part.
        #:
        #: Setting it snaps this part into place once. Whether it *stays* there
        #: is ``clip_locked``: unlocked is a one-off nudge that a later length
        #: edit will undo, locked makes the station derived and the part follows
        #: whatever it is attached to for as long as the clip lasts.
        self.clip_to: str | None = None
        self.clip_locked: bool = False
        #: Join at the mating stations rather than the extremes.
        #:
        #: On for tanks, which is what makes an intertank sit flush against the
        #: tangent line with the dome nested inside it, instead of perched on
        #: the dome's tip. Off gives a literal tip-to-tip butt, which is what
        #: you want for two parts that really do meet end on. It makes no
        #: difference at all between two flat-ended parts.
        self.clip_flush: bool = True
        #: Which end of the target this part hangs off: ``"aft"`` puts this
        #: part behind it, ``"forward"`` puts it in front.
        #:
        #: A chain is built in both directions in practice -- you add a tube
        #: and hang a nose off its front as readily as you add a boattail
        #: behind it -- and assuming "behind" makes half of those a manual
        #: station edit.
        self.clip_side: str = "aft"

    @property
    def is_external(self) -> bool:
        """Whether the air sees this component."""
        return self.aero_role.is_external

    # ------------------------------------------------------------------

    def add(self, child: "Component") -> "Component":
        child.parent = self
        self.children.append(child)
        self.mark_dirty("children")
        return child

    def remove(self, child: "Component") -> None:
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            self.mark_dirty("children")

    def walk(self) -> Iterator["Component"]:
        yield self
        for child in self.children:
            yield from child.walk()

    def find(self, name: str) -> "Component | None":
        for node in self.walk():
            if node.name == name:
                return node
        return None

    @property
    def path(self) -> str:
        if self.parent is None:
            return self.name
        return f"{self.parent.path}/{self.name}"

    def any_dirty(self) -> bool:
        return any(node.dirty for node in self.walk())

    def mark_all_clean(self) -> None:
        for node in self.walk():
            node.mark_clean()

    # ------------------------------------------------------------------
    # Geometry interface, overridden by concrete components.
    # ------------------------------------------------------------------

    #: Kinds that form the outer mould line and can therefore be clipped into a
    #: chain. A fin or a motor is positioned against its host body rather than
    #: end to end, so "the part ahead of it" is not a question with an answer.
    CLIPPABLE = ("stack", "tank")

    @property
    def clippable(self) -> bool:
        return self.kind in self.CLIPPABLE

    def station_range_m(self) -> tuple[float, float]:
        return (0.0, 0.0)

    def set_forward_station_m(self, station_m: float) -> bool:
        """Move so the forward end lands here. False if the kind cannot move."""
        return False

    def shift_m(self, delta: float) -> None:
        """Move this component and everything attached to it along the spine.

        Attachment has to mean something. A fin set is a *child* of the body it
        is mounted on, so moving that body has to take the fins with it --
        otherwise clipping a chain together leaves every fin behind at the
        station its host used to occupy, and the validator reports fins hanging
        in the air outside their own parent.
        """
        if abs(delta) < 1e-12:
            return
        if "station" in self:
            self.set("station", self.get("station") + delta)
        for child in self.children:
            child.shift_m(delta)

    def mating_range_m(self) -> tuple[float, float]:
        """Where another part actually joins this one.

        The same as the extreme range for anything with flat ends, and
        different for anything that closes with a curve. A tank's extreme is
        the tip of its dome, but nothing bolts to a dome tip: the joint is the
        tangent line where the dome meets the barrel, and the dome itself
        protrudes into whatever sits beyond it. Clipping to the extreme leaves
        a tank standing on the point of its own endcap.
        """
        return self.station_range_m()

    @property
    def has_distinct_mating(self) -> bool:
        """True when this part's joints are not at its extremes."""
        mating = self.mating_range_m()
        extreme = self.station_range_m()
        return abs(mating[0] - extreme[0]) > 1e-12 or abs(
            mating[1] - extreme[1]
        ) > 1e-12

    def volume_m3(self) -> float:
        return 0.0

    def computed_mass_kg(self, density_kg_m3: float | None = None) -> float:
        """Mass the geometry and material imply, ignoring any override."""
        from parametric.materials import get_material

        density = (
            get_material(self.material).density_kg_m3
            if density_kg_m3 is None
            else density_kg_m3
        )
        return self.volume_m3() * density

    def mass_kg(self, density_kg_m3: float | None = None) -> float:
        if self.mass_override_kg is not None:
            return float(self.mass_override_kg)
        return self.computed_mass_kg(density_kg_m3)


    @property
    def override_disagreement(self) -> float:
        """Relative gap between a measured mass and the modelled one.

        Worth surfacing rather than hiding. A part that weighs 30% more than
        its geometry says means the geometry is wrong, and the geometry is what
        aerodynamics and CAD are still being taken from.
        """
        if self.mass_override_kg is None:
            return 0.0
        computed = self.computed_mass_kg()
        if computed <= 0:
            return 0.0
        return abs(float(self.mass_override_kg) - computed) / computed


class Stack(Component):
    """A body of revolution defined by an ordered stack of cross-sections.

    Sections are kept sorted by station. The outer surface is the loft through
    them; ``wall_thickness`` hollows it, and zero leaves a solid.
    """

    kind = "stack"
    allowed_roles = ("auto", "nose", "body", "transition", "internal")

    def __init__(self, name: str = "airframe", wall_thickness_m: float = 0.003):
        super().__init__(name)
        self.sections: list[XSec] = []
        self.add_parm(
            "wall_thickness", wall_thickness_m, 0.0, 1e3, "m",
            "Shell thickness; zero for a solid", group="Construction",
            # Half the diameter is the whole radius: a wall thicker than that
            # has consumed the bore and the part is solid, not a shell.
            max_fn=lambda s: (
                s.max_diameter_m * 0.5 if s.max_diameter_m > 0 else None
            ),
        )
        # There was an "x_offset" here, a lateral offset of the whole stack. It
        # was declared, saved, and read by nothing -- no loft, mass or aero code
        # ever consulted it -- so it moved no geometry. It is gone rather than
        # wired up, because the offset a body actually needs is along the spine,
        # and that one is the station: the model axis is +Z aft from the nose
        # tip, so a station *is* a Z. See ``set_station_m``.

    # ------------------------------------------------------------------

    def add_section(self, section: XSec) -> XSec:
        self.sections.append(section)
        self.sections.sort(key=lambda s: s.station_m)
        section.on_change(lambda *_: self.mark_dirty("section"))
        self.mark_dirty("sections")
        return section

    def add_sections(self, sections: list[XSec]) -> None:
        for section in sections:
            self.add_section(section)

    def remove_section(self, section: XSec) -> None:
        if section in self.sections:
            self.sections.remove(section)
            self.mark_dirty("sections")

    def sorted_sections(self) -> list[XSec]:
        return sorted(self.sections, key=lambda s: s.station_m)

    # ------------------------------------------------------------------

    def station_range_m(self) -> tuple[float, float]:
        if not self.sections:
            return (0.0, 0.0)
        stations = [s.station_m for s in self.sections]
        return (min(stations), max(stations))

    @property
    def length_m(self) -> float:
        low, high = self.station_range_m()
        return high - low

    @property
    def max_diameter_m(self) -> float:
        return max((max(s.width_m, s.height_m) for s in self.sections), default=0.0)

    def set_length_m(self, length_m: float) -> bool:
        """Stretch or shrink the stack to an overall length.

        Section stations are scaled about the forward end, so the shape is
        preserved and only its extent changes: an ogive stays an ogive, a
        boattail keeps its taper. Editing the sections one at a time is still
        available below and is what you want for anything asymmetric; this is
        for the far commoner case of "make the body tube 600 mm".
        """
        sections = self.sorted_sections()
        if len(sections) < 2:
            return False
        low, high = self.station_range_m()
        current = high - low
        if current <= 0 or length_m <= 0:
            return False
        scale = length_m / current
        for section in sections:
            section.set("station", low + (section.station_m - low) * scale)
        self.mark_dirty("length")
        return True

    def set_station_m(self, station_m: float) -> bool:
        """Move the whole stack along the spine so it starts here.

        The longitudinal offset, which is the one a body assembly actually
        needs: nose at 0, tube behind it, boattail behind that. Stations are
        Z in this model, so this is the Z offset.
        """
        if not self.sections:
            return False
        low = self.station_range_m()[0]
        shift = station_m - low
        if abs(shift) < 1e-12:
            return False
        self.shift_m(shift)
        return True

    def shift_m(self, delta: float) -> None:
        """Move the sections, and whatever is mounted on them, together."""
        if abs(delta) < 1e-12:
            return
        for section in self.sections:
            section.set("station", section.station_m + delta)
        self.mark_dirty("station")
        for child in self.children:
            child.shift_m(delta)

    def set_forward_station_m(self, station_m: float) -> bool:
        return self.set_station_m(station_m)

    def fit_to_body(self) -> "Stack | None":
        """Match diameter and position to the body immediately aft.

        Adding a nose cone and a body tube separately leaves two parts at the
        same station with unrelated diameters, and squaring that up by hand
        means editing section rows until the numbers agree. The relationship is
        the obvious one -- a nose's base is the tube's front, and its aft end is
        where the tube begins -- so it is worth a button.

        Returns the body it fitted to, or None when there is nothing to fit to.
        """
        parent = self.parent
        if parent is None or not self.sections:
            return None
        siblings = [
            c for c in parent.children
            if isinstance(c, Stack) and c is not self and c.sections
        ]
        if not siblings:
            return None

        my_low = self.station_range_m()[0]
        # The body behind this one: the nearest sibling that does not start
        # ahead of it. Failing that, the widest, which is the airframe.
        behind = [s for s in siblings if s.station_range_m()[0] >= my_low - 1e-9]
        target = (
            min(behind, key=lambda s: s.station_range_m()[0]) if behind
            else max(siblings, key=lambda s: s.max_diameter_m)
        )

        front_sections = target.sorted_sections()
        front_diameter = max(
            front_sections[0].width_m, front_sections[0].height_m
        )
        if front_diameter > 0:
            self.set_diameter_m(front_diameter)
        # Butt the aft end of this stack against the front of that one.
        target_front = target.station_range_m()[0]
        self.set_station_m(target_front - self.length_m)
        return target

    def set_diameter_m(self, diameter_m: float) -> bool:
        """Scale every section so the widest one reaches this diameter.

        Proportional rather than absolute for the same reason: a transition
        scaled this way stays the same transition, where forcing every section
        to one diameter would flatten it into a tube.
        """
        current = self.max_diameter_m
        if current <= 0 or diameter_m <= 0:
            return False
        scale = diameter_m / current
        for section in self.sections:
            section.set("width", section.width_m * scale)
            section.set("height", section.height_m * scale)
        self.mark_dirty("diameter")
        return True

    def radius_at(self, station_m: float) -> float:
        """Equivalent radius at a station, interpolated between sections."""
        sections = self.sorted_sections()
        if not sections:
            return 0.0
        stations = np.array([s.station_m for s in sections])
        radii = np.array([s.equivalent_radius_m for s in sections])
        return float(np.interp(station_m, stations, radii))

    def enclosed_volume_m3(self) -> float:
        """Volume inside the outer surface, by integrating section area."""
        sections = self.sorted_sections()
        if len(sections) < 2:
            return 0.0
        stations = np.array([s.station_m for s in sections])
        areas = np.array([s.area_m2 for s in sections])
        return float(np.trapezoid(areas, stations))

    def material_areas(self) -> tuple[np.ndarray, np.ndarray]:
        """Stations, and the material area at each section.

        The outer section area less the section offset inward by the wall
        thickness; the whole outer area where there is no wall, or where
        the wall would consume the section -- at a nose tip the body stays
        solid, which is both what gets built and what a mesher can handle.

        This is the integrand of ``volume_m3`` and the weight the centroid
        must use: a shell's material sits at the surface, and weighting by
        the enclosed area instead put an ogive nose's centroid 6% of its
        length too far aft.
        """
        sections = self.sorted_sections()
        if len(sections) < 2:
            return np.zeros(0), np.zeros(0)

        thickness = self.get("wall_thickness")
        stations = np.array([s.station_m for s in sections])
        outer = np.array([s.area_m2 for s in sections])
        if thickness <= 0.0:
            return stations, outer

        inner = []
        for section in sections:
            width = max(section.width_m - 2.0 * thickness, 0.0)
            height = max(section.height_m - 2.0 * thickness, 0.0)
            if width <= 0.0 or height <= 0.0:
                inner.append(0.0)
                continue
            shrunk = XSec(
                section.station_m, section.shape, width, height,
                section.get("exponent"),
                max(section.get("corner_radius") - thickness, 0.0),
            )
            inner.append(shrunk.area_m2)
        return stations, outer - np.array(inner)

    def volume_m3(self) -> float:
        """Material volume: the shell, or the whole body when solid."""
        stations, areas = self.material_areas()
        if len(stations) < 2:
            return 0.0
        return float(np.trapezoid(areas, stations))

    # ------------------------------------------------------------------
    # Convenience builders that emit sections
    # ------------------------------------------------------------------

    def add_nose(self, profile: NoseProfile, length_m: float, diameter_m: float,
                 sections: int = 14, tip_radius_m: float = 0.0015) -> None:
        start = self.station_range_m()[1] if self.sections else 0.0
        # Remember which family this is. The sections that come back are a
        # discretised silhouette and the shape cannot be recovered from them
        # exactly -- it can only be re-fitted, which is what
        # ``parametric.canonical._best_nose_shape`` does. Fitting is good enough
        # to pick a family but cannot recover a power-law exponent, and a
        # solver that knows the intended shape should not have to guess it.
        #
        # Purely additive: nothing reads this unless it is set, so an existing
        # project keeps being fitted exactly as before.
        self.nose_profile = profile
        self.nose_tip_radius_m = float(tip_radius_m)
        self.add_sections(generate_nose_sections(
            profile, length_m, diameter_m, sections, tip_radius_m, start
        ))

    def add_tube(self, length_m: float, diameter_m: float, name: str = "tube") -> None:
        start = self.station_range_m()[1]
        # The forward section already exists if the previous run ended here at
        # the same diameter, so only the aft section is new.
        existing = [s for s in self.sections if abs(s.station_m - start) < 1e-9]
        if not existing:
            self.add_section(XSec(start, XSecShape.CIRCLE, diameter_m, name=f"{name}_fwd"))
        self.add_section(
            XSec(start + length_m, XSecShape.CIRCLE, diameter_m, name=f"{name}_aft")
        )

    def add_transition(self, length_m: float, rear_diameter_m: float,
                       sections: int = 6, name: str = "transition") -> None:
        start = self.station_range_m()[1]
        front = self.radius_at(start) * 2.0
        new = generate_transition_sections(
            length_m, front, rear_diameter_m, start, sections, name
        )
        self.add_sections(new[1:])   # the front section already exists


class FinSet(Component):
    """N identical trapezoidal fins attached around a parent stack."""

    kind = "finset"
    allowed_roles = ("auto", "fin", "internal")

    def __init__(
        self,
        name: str = "fins",
        count: int = 4,
        root_chord_m: float = 0.2,
        tip_chord_m: float = 0.1,
        span_m: float = 0.08,
        sweep_m: float = 0.06,
        thickness_m: float = 0.004,
        station_m: float = 1.0,
        cant_deg: float = 0.0,
    ):
        super().__init__(name)
        self.material = "g10_fiberglass"
        self.add_parm("count", count, 1, 12, "", "Number of fins",
                      designable=False, group="Planform")
        self.add_parm("root_chord", root_chord_m, 0.001, 1e4, "m", "Root chord",
                      group="Planform")
        self.add_parm("tip_chord", tip_chord_m, 0.0, 1e4, "m", "Tip chord",
                      group="Planform")
        self.add_parm("span", span_m, 0.001, 1e4, "m", "Exposed semi-span",
                      group="Planform")
        self.add_parm("sweep", sweep_m, -1e4, 1e4, "m",
                      "Leading-edge sweep distance", group="Planform")
        self.add_parm("thickness", thickness_m, 0.0005, 1e3, "m", "Fin thickness",
                      group="Section",
                      # Past half the root chord it is no longer a fin section.
                      max_fn=lambda f: f.get("root_chord") * 0.5 or None)
        self.add_parm("station", station_m, -1e5, 1e5, "m",
                      "Root leading edge station", group="Placement")
        self.add_parm("cant", cant_deg, -15.0, 15.0, "deg", "Cant angle for roll",
                      group="Placement")

    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        return int(round(self.get("count")))

    @property
    def area_per_fin_m2(self) -> float:
        return 0.5 * (self.get("root_chord") + self.get("tip_chord")) * self.get("span")

    @property
    def aspect_ratio(self) -> float:
        area = self.area_per_fin_m2
        return (self.get("span") ** 2) / area if area > 0 else 0.0

    @property
    def mid_chord_sweep_rad(self) -> float:
        dx = self.get("sweep") + 0.5 * (self.get("tip_chord") - self.get("root_chord"))
        span = self.get("span")
        return float(np.arctan2(dx, span)) if span > 0 else 0.0

    def station_range_m(self) -> tuple[float, float]:
        start = self.get("station")
        trailing = max(self.get("root_chord"), self.get("sweep") + self.get("tip_chord"))
        return (start, start + trailing)

    @property
    def planform_centroid_station_m(self) -> float:
        """Station of a panel's area centroid, sweep included.

        Where the fin's mass sits and where a stalled plate's force acts.
        The midpoint of the station range put it 13 mm aft on the basic
        rocket's fins. Reduces to root/2 for a rectangle and to
        (sweep + root)/3 for a triangle.
        """
        root = self.get("root_chord")
        tip = self.get("tip_chord")
        sweep = self.get("sweep")
        if root + tip <= 0.0:
            return self.get("station")
        return self.get("station") + (
            sweep * (root + 2.0 * tip) + root * root + root * tip + tip * tip
        ) / (3.0 * (root + tip))

    def volume_m3(self) -> float:
        return self.area_per_fin_m2 * self.get("thickness") * self.count

    def body_radius_m(self) -> float:
        """Radius of the parent body where the fins attach."""
        parent = self.parent
        if isinstance(parent, Stack):
            return parent.radius_at(self.get("station") + 0.5 * self.get("root_chord"))
        return 0.0



class Tank(Component):
    """A propellant tank: a cylindrical barrel closed by two domes.

    The pill bottle, which is what a real tank is and what a stack of circular
    sections cannot describe -- a dome is a curve the section fitter would have
    to approximate, and its *volume* is the whole point of the part. Here the
    geometry is three numbers, the capacity falls out of them in closed form,
    and the propellant is checked against it.

    Domes are ellipsoidal. ``dome_ratio`` is dome height over barrel radius, so
    1.0 is hemispherical and the common 1/sqrt(2) ~ 0.707 is the standard
    elliptical bulkhead. Zero would be a flat head, which no pressure vessel
    uses, so the floor is deliberately above it.

    Propellant is tracked like a motor's rather than as structure: it is not
    part of the dry mass and it drains in flight. ``mass_kg`` is the shell.
    """

    kind = "tank"
    allowed_roles = ("auto", "body", "internal")

    def __init__(
        self,
        name: str = "tank",
        diameter_m: float = 0.30,
        barrel_length_m: float = 0.80,
        dome_ratio: float = 0.707,
        wall_thickness_m: float = 0.003,
        station_m: float = 0.0,
        propellant_mass_kg: float = 0.0,
        propellant_density_kg_m3: float = 810.0,
        ullage: float = 0.05,
    ):
        super().__init__(name)
        self.material = "aluminium_6061_t6"
        #: ``"fuel"`` or ``"oxidizer"`` -- which side of the engine's mixture
        #: this tank feeds. Only consulted when a tank-fed motor declares a
        #: mixture ratio; unlabeled tanks drain in proportion to their loads,
        #: which is why the default is harmless for every existing vehicle.
        self.contents: str = "fuel"
        self.add_parm("diameter", diameter_m, 0.001, 1e4, "m", "Barrel diameter",
                      group="Geometry")
        self.add_parm("barrel_length", barrel_length_m, 0.0, 1e4, "m",
                      "Straight section length", group="Geometry")
        self.add_parm("dome_ratio", dome_ratio, 0.1, 1.0, "",
                      "Dome height / radius; 1 is hemispherical",
                      group="Geometry")
        self.add_parm(
            "wall_thickness", wall_thickness_m, 0.0, 1e3, "m",
            "Shell thickness", group="Geometry",
            # Half the diameter is the whole radius; past that there is no bore.
            max_fn=lambda t: t.get("diameter") * 0.5 or None,
        )
        self.add_parm("station", station_m, -1e5, 1e5, "m",
                      "Forward tip of the tank", group="Placement")
        self.add_parm("propellant_mass", propellant_mass_kg, 0.0, 1e7, "kg",
                      "Propellant load", group="Propellant")
        self.add_parm("propellant_density", propellant_density_kg_m3, 1.0, 1e5,
                      "kg/m³", "Propellant density", group="Propellant")
        self.add_parm("ullage", ullage, 0.0, 0.5, "",
                      "Fraction of volume left as gas", group="Propellant")

    # ------------------------------------------------------------------

    @property
    def radius_m(self) -> float:
        return 0.5 * self.get("diameter")

    @property
    def dome_height_m(self) -> float:
        return self.radius_m * self.get("dome_ratio")

    @property
    def overall_length_m(self) -> float:
        """Tip to tip: the barrel plus both domes."""
        return self.get("barrel_length") + 2.0 * self.dome_height_m

    def station_range_m(self) -> tuple[float, float]:
        start = self.get("station")
        return (start, start + self.overall_length_m)

    def set_forward_station_m(self, station_m: float) -> bool:
        delta = station_m - self.get("station")
        if abs(delta) < 1e-12:
            return False
        self.shift_m(delta)     # takes anything mounted on the tank with it
        return True

    def mating_range_m(self) -> tuple[float, float]:
        """The two tangent lines, where the domes meet the barrel.

        This is where a skirt or an intertank bolts on, and it is why a tank
        clipped flush sits with its dome inside its neighbour rather than
        balanced on the dome's tip with a barrel-sized gap behind it.
        """
        start = self.get("station") + self.dome_height_m
        return (start, start + self.get("barrel_length"))

    @staticmethod
    def _capsule_volume(radius: float, dome_height: float, barrel: float) -> float:
        """Cylinder plus two semi-ellipsoid caps.

        A semi-ellipsoid of base radius r and height h has volume 2/3 pi r^2 h,
        so two of them contribute 4/3 pi r^2 h -- the sphere formula when
        h = r, which is the check that the expression is right.
        """
        if radius <= 0:
            return 0.0
        return float(np.pi * radius * radius * (barrel + (4.0 / 3.0) * dome_height))

    @property
    def internal_volume_m3(self) -> float:
        """Volume inside the wall -- what the propellant actually fits in."""
        thickness = self.get("wall_thickness")
        inner_r = max(self.radius_m - thickness, 0.0)
        inner_h = max(self.dome_height_m - thickness, 0.0)
        return self._capsule_volume(inner_r, inner_h, self.get("barrel_length"))

    @property
    def enclosed_volume_m3(self) -> float:
        """Volume inside the mould line, wall included."""
        return self._capsule_volume(
            self.radius_m, self.dome_height_m, self.get("barrel_length")
        )

    @property
    def capacity_kg(self) -> float:
        """Propellant the tank can hold, after ullage."""
        usable = self.internal_volume_m3 * (1.0 - self.get("ullage"))
        return float(usable * self.get("propellant_density"))

    @property
    def propellant_volume_m3(self) -> float:
        density = self.get("propellant_density")
        return self.get("propellant_mass") / density if density > 0 else 0.0

    @property
    def fill_fraction(self) -> float:
        internal = self.internal_volume_m3
        return self.propellant_volume_m3 / internal if internal > 0 else 0.0

    @property
    def centroid_station_m(self) -> float:
        """Where the *full* propellant load acts, for the static roll-up.

        The tank's own centre. The flight simulation drains the tank -- the
        column settles aft and shortens -- but the rail-side mass summary
        wants the load as loaded.
        """
        low, high = self.station_range_m()
        return 0.5 * (low + high)

    def volume_m3(self) -> float:
        """Material volume: the shell between the outer and inner capsules."""
        shell = self.enclosed_volume_m3 - self.internal_volume_m3
        return float(max(shell, 0.0))

    def validate(self) -> list[str]:
        problems = []
        if self.get("propellant_mass") > self.capacity_kg * (1.0 + 1e-9):
            problems.append(
                f"{self.name}: {self.get('propellant_mass'):.1f} kg of propellant "
                f"exceeds the {self.capacity_kg:.1f} kg the tank holds at "
                f"{self.get('ullage'):.0%} ullage."
            )
        return problems


class Wing(Component):
    """A lifting surface: two panels either side of the body.

    Geometrically this is a fin set with the radial array replaced by dihedral.
    In the model frame the body runs along +Z, a panel extends along X and
    vertical is Y, so tilting a panel up is a rotation about Z -- the same
    rotation that spaces fins around a body. A fin at 90 degrees of dihedral is
    a wing panel, and incidence is what a fin calls cant. Sharing that geometry
    is why this is a small component rather than a new subsystem.

    What differs from ``FinSet`` is the vocabulary, and the vocabulary is the
    point. A wing is described by thickness *ratio* rather than an absolute
    thickness, because an aerofoil is specified as a fraction of its chord and
    scales with it; by dihedral and incidence, which a fin set has no notion
    of; and by a semi-span measured from the body side rather than a radius.

    **Wings are invisible to the Barrowman aerodynamics, deliberately.**
    ``extract_geometry`` collects ``model.fin_sets``, which is an ``isinstance``
    test, and a Wing is not a FinSet. That is correct rather than an oversight:
    the silhouette method reduces a vehicle to radius against station and has
    no term a wing could occupy. A wing contributes mass and geometry here, and
    its aerodynamics have to come from a method that can represent one.
    """

    kind = "wing"
    allowed_roles = ("auto", "fin", "internal")

    #: Aerofoil section area as a fraction of the chord x thickness rectangle.
    #: About 0.685 for a NACA four-digit section, and close enough for any
    #: conventional aerofoil that the material choice matters more.
    SECTION_AREA_COEFF = 0.685

    def __init__(
        self,
        name: str = "wing",
        root_chord_m: float = 1.2,
        tip_chord_m: float = 0.7,
        span_m: float = 3.0,
        sweep_m: float = 0.35,
        thickness_ratio: float = 0.12,
        station_m: float = 2.0,
        dihedral_deg: float = 3.0,
        incidence_deg: float = 1.5,
        symmetric: bool = True,
        wall_thickness_m: float = 0.0016,
    ):
        super().__init__(name)
        self.material = "aluminium_6061_t6"
        self.symmetric = bool(symmetric)
        self.add_parm("root_chord", root_chord_m, 0.001, 1e4, "m", "Root chord",
                      group="Planform")
        self.add_parm("tip_chord", tip_chord_m, 0.0, 1e4, "m", "Tip chord",
                      group="Planform")
        self.add_parm("span", span_m, 0.001, 1e4, "m",
                      "Exposed semi-span, one panel", group="Planform")
        self.add_parm("sweep", sweep_m, -1e4, 1e4, "m",
                      "Leading-edge sweep distance", group="Planform")
        self.add_parm("thickness_ratio", thickness_ratio, 0.01, 0.95, "",
                      "Aerofoil thickness / chord", group="Section")
        self.add_parm(
            "wall_thickness", wall_thickness_m, 0.0, 1e3, "m",
            "Skin thickness; zero for a solid wing", group="Section",
            # Half the aerofoil's own thickness: two skins meeting in the
            # middle leave no cavity, and the panel is solid from there on.
            max_fn=lambda w: (
                w.get("thickness_ratio") * w.mean_chord_m * 0.5 or None
            ),
        )
        self.add_parm("station", station_m, -1e5, 1e5, "m",
                      "Root leading edge station", group="Placement")
        self.add_parm("dihedral", dihedral_deg, -30.0, 60.0, "deg",
                      "Panel tilt above horizontal", group="Placement")
        self.add_parm("incidence", incidence_deg, -15.0, 15.0, "deg",
                      "Angle of the chord to the body axis", group="Placement")

    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Panels. Two for a normal wing, one for a vertical tail."""
        return 2 if self.symmetric else 1

    @property
    def mean_chord_m(self) -> float:
        return 0.5 * (self.get("root_chord") + self.get("tip_chord"))

    @property
    def panel_area_m2(self) -> float:
        return self.mean_chord_m * self.get("span")

    @property
    def reference_area_m2(self) -> float:
        """Planform area of every panel together -- the wing's ``Sref``."""
        return self.panel_area_m2 * self.count

    @property
    def aspect_ratio(self) -> float:
        """Full-span aspect ratio, the number that governs induced drag.

        Uses the full span across both panels, not the semi-span, because that
        is the definition every lifting-line and vortex-lattice result assumes.
        """
        area = self.reference_area_m2
        if area <= 0:
            return 0.0
        full_span = self.get("span") * self.count
        return float(full_span ** 2 / area)

    @property
    def taper_ratio(self) -> float:
        root = self.get("root_chord")
        return float(self.get("tip_chord") / root) if root > 0 else 0.0

    @property
    def mean_aerodynamic_chord_m(self) -> float:
        """MAC for a linearly tapered panel -- the moment reference length.

        (2/3) c_r (1 + L + L^2) / (1 + L) for taper ratio L. This is ``cref``
        when the coefficients are eventually referenced properly, which is
        exactly the quantity a single ``reference_length_m`` cannot carry.
        """
        root = self.get("root_chord")
        lam = self.taper_ratio
        if root <= 0:
            return 0.0
        return float((2.0 / 3.0) * root * (1 + lam + lam * lam) / (1 + lam))

    @property
    def dihedral_rad(self) -> float:
        return float(np.radians(self.get("dihedral")))

    @property
    def mid_chord_sweep_rad(self) -> float:
        dx = self.get("sweep") + 0.5 * (self.get("tip_chord") - self.get("root_chord"))
        span = self.get("span")
        return float(np.arctan2(dx, span)) if span > 0 else 0.0

    def station_range_m(self) -> tuple[float, float]:
        start = self.get("station")
        trailing = max(self.get("root_chord"), self.get("sweep") + self.get("tip_chord"))
        return (start, start + trailing)

    def enclosed_volume_m3(self) -> float:
        """Volume inside the mould line, every panel together.

        The aerofoil thickness follows the chord, so the integrand is the
        section area k*(t/c)*c^2 rather than a constant thickness times a
        planform. For a linear taper the chord-squared integral has a closed
        form, span*(cr^2 + cr*ct + ct^2)/3, and using it rather than the mean
        chord matters: on a taper ratio of 0.3 the mean-chord approximation is
        about 8% light, which lands straight in the wing mass.
        """
        root = self.get("root_chord")
        tip = self.get("tip_chord")
        chord_squared = self.get("span") * (root * root + root * tip + tip * tip) / 3.0
        panel = self.SECTION_AREA_COEFF * self.get("thickness_ratio") * chord_squared
        return float(panel * self.count)

    def wetted_area_m2(self) -> float:
        """Skin area over every panel.

        Twice the planform, plus a little for the thickness the section wraps
        around: the standard 2*S*(1 + 0.2*t/c) estimate.
        """
        return float(
            2.0 * self.reference_area_m2 * (1.0 + 0.2 * self.get("thickness_ratio"))
        )

    def volume_m3(self) -> float:
        """Material volume: the skin, or the whole section when solid.

        A wing is a skinned structure, not a billet. Left solid, a 3 m panel in
        aluminium comes out over a tonne -- an order of magnitude wrong, and
        wrong in the direction that quietly ruins a mass budget. So this
        follows ``Stack``: a wall thickness hollows it, zero leaves it solid.

        The skin estimate is deliberately the whole structural allowance. Spars,
        ribs and joints are not modelled individually, so a thickness chosen to
        match a real wing's mass is doing the job of all of them.
        """
        enclosed = self.enclosed_volume_m3()
        thickness = self.get("wall_thickness")
        if thickness <= 0.0:
            return enclosed
        # Never let the skin claim more material than the section contains --
        # a thick wall on a thin aerofoil would otherwise exceed the solid.
        return float(min(self.wetted_area_m2() * thickness, enclosed))

    def body_radius_m(self) -> float:
        """Radius of the parent body where the wing attaches."""
        parent = self.parent
        if isinstance(parent, Stack):
            return parent.radius_at(self.get("station") + 0.5 * self.get("root_chord"))
        return 0.0


class Protuberance(Component):
    """Something sticking into the flow that the mould line does not describe.

    Rail buttons, launch lugs, camera fairings, antenna covers, cable conduit.
    Barrowman has no term for any of them, so before this they contributed
    exactly zero drag -- which is not a conservative simplification, it is a
    missing force. Drag comes from a Hoerner shape coefficient on the declared
    frontal area.
    """

    kind = "protuberance"
    allowed_roles = ("protuberance", "internal")

    def __init__(self, name: str = "protuberance", shape: str = "rail_button",
                 frontal_area_m2: float = 1e-4, station_m: float = 0.0,
                 count: int = 1, mass_kg: float = 0.0):
        super().__init__(name)
        from parametric.roles import AeroRole, HoernerShape

        self.aero_role = AeroRole.PROTUBERANCE
        self.shape = HoernerShape(shape)
        self.add_parm("frontal_area", frontal_area_m2, 0.0, 10.0, "m²",
                      "Area presented to the flow, per item")
        self.add_parm("station", station_m, -1e5, 1e5, "m",
                      "Station along the spine from the origin")
        self.add_parm("count", count, 1, 64, "", "How many", designable=False)
        self.add_parm("mass", mass_kg, 0.0, 1e4, "kg", "Mass of one item")
        self.add_parm("immersion", 1.0, 0.0, 1.0, "",
                      "Fraction of free-stream dynamic pressure seen")

    @property
    def count(self) -> int:
        return int(round(self.get("count")))

    def station_range_m(self) -> tuple[float, float]:
        station = self.get("station")
        return (station, station)

    def computed_mass_kg(self, density_kg_m3: float | None = None) -> float:
        return self.get("mass") * self.count

    def mass_kg(self, density_kg_m3: float | None = None) -> float:
        if self.mass_override_kg is not None:
            return float(self.mass_override_kg)
        return self.computed_mass_kg(density_kg_m3)

    def to_spec(self):
        """The dataclass the drag routine consumes."""
        from parametric.roles import Protuberance as Spec

        return Spec(
            name=self.name,
            shape=self.shape,
            frontal_area_m2=self.get("frontal_area"),
            station_m=self.get("station"),
            count=self.count,
            immersion=self.get("immersion"),
        )


class PointMass(Component):
    """Mass with a station but no generated geometry."""

    kind = "pointmass"
    allowed_roles = ("internal",)

    def __init__(self, name: str, mass_kg: float, station_m: float,
                 radial_offset_m: float = 0.0, growth_allowance: float = 0.0):
        super().__init__(name)
        self.add_parm("mass", mass_kg, 0.0, 1e6, "kg", "Mass")
        self.add_parm("station", station_m, -1e5, 1e5, "m",
                      "Station along the spine from the origin")
        self.add_parm("radial_offset", radial_offset_m, -100.0, 100.0, "m",
                      "Offset from the centreline")
        self.add_parm("growth", growth_allowance, 0.0, 2.0, "",
                      "Mass growth allowance as a fraction")

    def station_range_m(self) -> tuple[float, float]:
        station = self.get("station")
        return (station, station)

    def computed_mass_kg(self, density_kg_m3: float | None = None) -> float:
        return self.get("mass")

    def mass_kg(self, density_kg_m3: float | None = None) -> float:
        if self.mass_override_kg is not None:
            return float(self.mass_override_kg)
        return self.computed_mass_kg(density_kg_m3)

    @property
    def mass_with_growth_kg(self) -> float:
        return self.mass_kg() * (1.0 + self.get("growth"))


#: Total-impulse boundaries for the amateur motor classes, in N·s. Each class is
#: double the previous, starting at A = 2.5 N·s.
#: Re-exported so a caller can set ``motor.burn_geometry`` without reaching
#: into the trajectory package.
from trajectory.vehicle.mass_properties import (  # noqa: E402
    BURN_GEOMETRIES,
    END_BURNER,
    LIQUID,
    RADIAL,
)


def impulse_class(total_impulse_ns: float) -> str:
    """Motor class letter for a total impulse, e.g. 5253 N·s -> 'M'.

    Delegates to the one table, in ``trajectory.propulsion.thrustcurve``.
    The copy that used to live here was shifted by a full letter -- it put the
    A/B boundary at 5 N·s where NAR and TRA put it at 2.5 -- so every motor in
    the tool was labelled one class low. A 6,030 N·s motor came up as an L when
    it is an M, which is the difference between two certification levels.
    """
    from trajectory.propulsion.thrustcurve import impulse_class as _impulse_class

    return _impulse_class(total_impulse_ns)


class Motor(Component):
    """A motor: propellant, where it sits, and the thrust it produces over time.

    The thrust curve lives on the component rather than in a separate file. A
    motor whose performance is stored elsewhere cannot be edited, swept or
    compared in the tool, and the curve is the single most important thing
    about it -- burn time and peak thrust decide rail exit, max-Q and apogee
    between them.

    A curve loaded from a saved propulsion model or a vendor file still ends up
    here, as points; the file becomes a source rather than a dependency.
    """

    kind = "motor"
    allowed_roles = ("internal",)

    def __init__(self, name: str = "motor", propellant_mass_kg: float = 0.0,
                 propellant_density_kg_m3: float = 1800.0,
                 station_m: float = 0.0, length_m: float = 0.0,
                 propulsion_file: str | None = None):
        super().__init__(name)
        self.propulsion_file = propulsion_file
        #: Where the propellant comes from: ``"grain"`` for a solid motor
        #: carrying its own, ``"tanks"`` for a pressure- or pump-fed engine
        #: drawing from the vehicle's tanks.
        #:
        #: A biprop's propellant is in the tanks, so a motor set to ``"tanks"``
        #: contributes none of its own to the mass roll-up. Without this the
        #: same kilograms would be counted twice -- once in the tank that holds
        #: them and once in the engine that burns them.
        self.feed = "grain"
        self.add_parm("propellant_mass", propellant_mass_kg, 0.0, 1e6, "kg",
                      "Propellant load")
        self.add_parm("propellant_density", propellant_density_kg_m3, 1.0, 1e5,
                      "kg/m³", "Bulk propellant density")
        self.add_parm("station", station_m, -1e5, 1e5, "m",
                      "Forward end of the grain")
        self.add_parm("length", length_m, 0.0, 1e4, "m", "Grain length")
        self.add_parm("isp_vac", 205.0, 1.0, 500.0, "s", "Vacuum specific impulse")
        self.add_parm("isp_sl", 180.0, 1.0, 500.0, "s", "Sea-level specific impulse")
        self.add_parm("nozzle_area", 0.0, 0.0, 10.0, "m²",
                      "Nozzle exit area; 0 derives it from the Isp pair")
        self.add_parm("max_gimbal", 0.0, 0.0, 30.0, "deg", "Gimbal limit")
        # O/F by mass, for an engine fed from the vehicle's tanks. The flight
        # sim splits its integrated mass flow by this: oxidizer tanks supply
        # MR/(1+MR) of every kilogram burned, fuel tanks the rest, so the two
        # sides drain at the rates the engine actually consumes them. Zero
        # means "in proportion to what each tank holds" -- the correctly
        # sized biprop, and the only sane reading for a solid.
        self.add_parm("mixture_ratio", 0.0, 0.0, 100.0, "",
                      "O/F mass ratio; 0 drains tanks in proportion to load")
        # Hardware, as opposed to what it burns.
        #
        # A motor had no dry mass at all, on the reasoning that a solid motor's
        # case is the tube drawn around it. That holds for a reloadable solid
        # and is plainly false for a liquid engine, which is a chamber, a
        # nozzle and a turbopump that exist whether or not anything is flowing.
        # Zero by default, so no existing vehicle changes weight.
        self.add_parm("dry_mass", 0.0, 0.0, 1e6, "kg",
                      "Engine hardware mass, empty", group="Hardware")
        # Zero means "draw nothing", which is what every motor did before this.
        # Set it and the engine becomes a real object in the viewport.
        self.add_parm("case_diameter", 0.0, 0.0, 1e3, "m",
                      "Chamber outside diameter; 0 to draw nothing",
                      group="Hardware")
        self.add_parm("nozzle_length", 0.0, 0.0, 1e3, "m",
                      "Bell length aft of the chamber", group="Hardware")

        #: Thrust curve as (time_s, thrust_n) pairs, sorted by time.
        self.curve: list[tuple[float, float]] = []

        #: How the propellant is consumed, which decides where the *remaining*
        #: propellant sits. A solid core burner keeps its centroid; a liquid
        #: tank settles aft and its centroid migrates as it drains. The CG
        #: travel over a burn is the difference between the two, and on a
        #: vehicle whose propellant is 40% of wet mass that is not small.
        self.burn_geometry: str = RADIAL

    # ------------------------------------------------------------------

    def set_curve(self, points) -> None:
        cleaned = sorted(
            (float(t), max(0.0, float(f))) for t, f in points
        )
        self.curve = cleaned
        self.mark_dirty("curve")

    def add_curve_point(self, time_s: float, thrust_n: float) -> None:
        self.curve.append((float(time_s), max(0.0, float(thrust_n))))
        self.curve.sort()
        self.mark_dirty("curve")

    def remove_curve_point(self, index: int) -> None:
        if 0 <= index < len(self.curve):
            del self.curve[index]
            self.mark_dirty("curve")

    def flat_curve(self, thrust_n: float, burn_time_s: float,
                   rise_s: float = 0.15) -> None:
        """A trapezoidal curve: the usual first cut at a motor."""
        rise = min(rise_s, burn_time_s * 0.25)
        self.set_curve([
            (0.0, 0.0),
            (rise, thrust_n),
            (burn_time_s - rise, thrust_n * 0.95),
            (burn_time_s, 0.0),
        ])

    def curve_from_impulse(self, total_impulse_ns: float, burn_time_s: float) -> None:
        """Size a flat curve to hit a target total impulse, exactly."""
        if burn_time_s <= 0:
            return
        # flat_curve's area is 0.975 * F * (T - rise): the rise triangle, the
        # plateau sloping to 0.95 F, and the tail. Invert that rather than
        # the 0.93 * T it used to divide by, which was only right at 3.25 s
        # and missed by -21% on a 0.4 s burn and +4% on a 25 s one.
        rise = min(0.15, burn_time_s * 0.25)
        self.flat_curve(total_impulse_ns / (0.975 * (burn_time_s - rise)), burn_time_s)

    # ------------------------------------------------------------------

    @property
    def burn_time_s(self) -> float:
        active = [t for t, f in self.curve if f > 0.0]
        return max(active) if active else 0.0

    @property
    def peak_thrust_n(self) -> float:
        return max((f for _, f in self.curve), default=0.0)

    @property
    def total_impulse_ns(self) -> float:
        if len(self.curve) < 2:
            return 0.0
        times = np.array([t for t, _ in self.curve])
        thrusts = np.array([f for _, f in self.curve])
        return float(np.trapezoid(thrusts, times))

    @property
    def average_thrust_n(self) -> float:
        burn = self.burn_time_s
        return self.total_impulse_ns / burn if burn > 0 else 0.0

    @property
    def impulse_class(self) -> str:
        return impulse_class(self.total_impulse_ns)

    @property
    def effective_isp_s(self) -> float:
        """Isp the curve and propellant load actually imply.

        Over-specification is the usual source of a wrong trajectory: the
        thrust curve, the propellant mass and the declared Isp are three
        numbers where two would do. When this disagrees with ``isp_vac``, one
        of the three is wrong.
        """
        propellant = self.get("propellant_mass")
        if propellant <= 0:
            return 0.0
        return self.total_impulse_ns / (propellant * 9.80665)

    @property
    def isp_consistency(self) -> float:
        """Relative disagreement between the declared and implied Isp."""
        declared = self.get("isp_vac")
        implied = self.effective_isp_s
        if declared <= 0 or implied <= 0:
            return 0.0
        return abs(implied - declared) / declared

    @property
    def propellant_radius_m(self) -> float:
        """Radius of the propellant column, from its load, density and length.

        Derived rather than declared: mass and bulk density give a volume, and
        the grain length is already a parm, so a third number would only be
        another thing to contradict the other two.
        """
        length = self.get("length")
        density = self.get("propellant_density")
        if length <= 0 or density <= 0:
            return 0.0
        volume = self.get("propellant_mass") / density
        return float((volume / (np.pi * length)) ** 0.5)

    def effective_nozzle_area_m2(self) -> float:
        """Exit area, declared or derived from the Isp pair."""
        declared = self.get("nozzle_area")
        if declared > 0:
            return declared
        from trajectory.vehicle.engine import nozzle_area_for_isp_pair

        return nozzle_area_for_isp_pair(
            self.peak_thrust_n, self.get("isp_vac"), self.get("isp_sl")
        )

    def to_engine(self):
        """Build a trajectory Engine from this motor."""
        from trajectory.vehicle.engine import Engine

        if len(self.curve) < 2:
            raise ValueError(
                f"{self.name}: needs at least two thrust-curve points to fly."
            )
        times = np.array([t for t, _ in self.curve], dtype=float)
        thrusts = np.array([f for _, f in self.curve], dtype=float)
        engine = Engine(
            thrust_curve=thrusts,
            time_points=times,
            isp_vac=self.get("isp_vac"),
            isp_sl=min(self.get("isp_sl"), self.get("isp_vac")),
            nozzle_area=self.effective_nozzle_area_m2(),
            thrust_reference="vacuum",
        )
        engine.max_gimbal = np.radians(self.get("max_gimbal"))
        return engine

    def import_thrust_curve(self, path, index: int = 0) -> "object":
        """Load a published motor -- RASP ``.eng`` or RockSim ``.rse``.

        The Isp parms are set from what the curve and the propellant load imply
        rather than from anything declared in the file. Thrust, propellant mass
        and Isp are three numbers where two would do, and the imported curve is
        the measurement; deriving the third keeps them consistent, so mass flow
        integrates back to exactly the propellant loaded.

        Both Isps are set equal because a published curve is a measurement at
        one ambient condition, not an Isp pair. Equal values give a zero nozzle
        area, which means the solver uses the curve as given instead of
        applying a pressure correction the file never justified.
        """
        from trajectory.propulsion.thrustcurve import load_thrust_curve

        curve = load_thrust_curve(path, index)
        self.set_curve(curve.points())

        isp = curve.implied_isp_s
        updates = {"propellant_mass": curve.propellant_mass_kg}
        if isp > 0:
            updates.update(isp_vac=isp, isp_sl=isp, nozzle_area=0.0)
        # The file's case dimensions, where the motor has none of its own:
        # the grain length places the propellant column the flight drains,
        # and without it the column had no length and no radius.
        if self.get("length") <= 0.0 and curve.length_mm > 0.0:
            updates["length"] = curve.length_mm / 1000.0
        if self.get("case_diameter") <= 0.0 and curve.diameter_mm > 0.0:
            updates["case_diameter"] = curve.diameter_mm / 1000.0
        self.update(**updates)

        self.propulsion_file = str(path)
        return curve

    def load_propulsion_file(self, path) -> None:
        """Pull a saved propulsion model in as editable points."""
        from trajectory.propulsion.model import load_propulsion_model

        model = load_propulsion_model(path)
        self.set_curve(zip(model.time_s, model.thrust_n))
        self.update(
            isp_vac=model.isp_vac_s,
            isp_sl=model.isp_sl_s,
            nozzle_area=model.nozzle_area_m2,
            propellant_mass=model.propellant_mass_kg,
            max_gimbal=model.max_gimbal_deg,
        )
        self.propulsion_file = str(path)

    # ------------------------------------------------------------------

    @property
    def exit_radius_m(self) -> float:
        area = self.get("nozzle_area")
        return float(np.sqrt(area / np.pi)) if area > 0 else 0.0

    @property
    def draws_geometry(self) -> bool:
        return self.get("case_diameter") > 0.0

    def station_range_m(self) -> tuple[float, float]:
        """Chamber plus bell. The nozzle is hardware and occupies station."""
        start = self.get("station")
        return (start, start + self.get("length") + self.get("nozzle_length"))

    def grain_range_m(self) -> tuple[float, float]:
        """Where the propellant is: the grain, without the nozzle bell.

        The propellant column the flight drains, and the load's centroid,
        used to be built from ``station_range_m`` and so ran over the bell
        -- a motor with a drawn nozzle carried its propellant 31% longer
        and its centroid 125 mm aft of where the grain actually was.
        """
        start = self.get("station")
        return (start, start + self.get("length"))

    def computed_mass_kg(self, density_kg_m3: float | None = None) -> float:
        """The declared hardware mass.

        Not derived from the drawn geometry: a chamber is a cooled, ribbed,
        flanged assembly, and taking its wall as a solid of revolution would
        produce a number with no relationship to what an engine weighs. The
        honest move is to make it a value you state.
        """
        return self.get("dry_mass")

    def mass_kg(self, density_kg_m3: float | None = None) -> float:
        # Propellant is still tracked separately; this is the hardware.
        if self.mass_override_kg is not None:
            return float(self.mass_override_kg)
        return self.get("dry_mass")

    @property
    def propellant_volume_m3(self) -> float:
        density = self.get("propellant_density")
        return self.get("propellant_mass") / density if density > 0 else 0.0

    @property
    def centroid_station_m(self) -> float:
        """Where the full propellant load acts: the middle of the grain."""
        start, end = self.grain_range_m()
        return 0.5 * (start + end)
