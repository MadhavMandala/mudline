"""The vocabulary RASAero's solver actually operates on.

This deliberately mirrors ``i.m_a`` -- the flat, ordered part list the engine
builds internally -- rather than either of the tool's own geometry types.

Why not reuse ``parametric.aero.AeroGeometry`` or ``parametric.canonical
.CanonicalModel``? Both are lossy relative to what the solver reads, in
different ways, and both encode a translation decision. If the engine consumed
one of them, every disagreement with RASAero would first have to be
adjudicated: physics, or translation? Owning the vocabulary makes the engine
unit-testable in isolation and lets the oracle's test vehicles map across 1:1,
so a mismatch can only be physics.

Adapters from the tool's types live in ``adapters.py``.

Conventions
-----------
* **Inches, pounds, degrees Fahrenheit, feet of altitude.** RASAero is an
  English-units program throughout and the arithmetic is transcribed from it;
  converting at the boundary is safer than converting mid-formula. Angles are
  degrees at this interface and radians inside the solver.
* ``x0`` is the forward station of a part, inches from the nose tip. Note this
  differs from the CDX1 file convention for *fins*, where ``Location`` is
  measured from the aft end of the parent part, forwards. The adapter converts.
* ``d_fwd`` / ``d_aft`` are the forward and aft diameters. For a nose cone
  ``d_fwd`` is 0 and ``d_aft`` is the base diameter; RASAero itself leaves the
  nose's forward diameter unset and relies on it reading as zero
  (``i.cs:1040-1050`` takes a max over both, so an unset 0 is harmless).

There is no ``FinCan`` and no ``BoatTail`` here, by design. RASAero has neither:
a fin can is rewritten into an Expansion plus a BodyTube with the preceding
tube shortened (``i.cs:634-655``), and a boattail is simply a Reducer. That
rewrite is a *geometry* decision, not a file-format one, so it belongs in the
builder below rather than in an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Enumerations. The string values are RASAero's own, because they are what the
# solver switches on (i.cs:1024, 1053, 3648, ...) and what appears in CDX1.
# ---------------------------------------------------------------------------


class PartType(str, Enum):
    NOSE_CONE = "NoseCone"
    BODY_TUBE = "BodyTube"
    EXPANSION = "Expansion"
    REDUCER = "Reducer"
    FINS = "Fins"
    RAIL_GUIDE = "RailGuide"
    LAUNCH_LUG = "LaunchLug"
    LAUNCH_SHOE = "LaunchShoe"
    PLATE = "Plate"
    STREAMLINED_NO_BASE_DRAG = "Streamlined_No_Base_Drag"
    STREAMLINED_WITH_BASE_DRAG = "Streamlined_With_Base_Drag"


class NoseShape(str, Enum):
    CONICAL = "Conical"
    TANGENT_OGIVE = "Tangent Ogive"
    VON_KARMAN = "Von Karman Ogive"
    POWER_LAW = "Power Law"
    LV_HAACK = "LV-Haack"
    PARABOLIC = "Parabolic"
    ELLIPTICAL = "Elliptical"


class Airfoil(str, Enum):
    HEXAGONAL = "Hexagonal"
    SUBSONIC_NACA = "Subsonic NACA"
    DOUBLE_WEDGE = "Double Wedge"
    BICONVEX = "Biconvex"
    SINGLE_WEDGE = "Single Wedge"
    HEXAGONAL_BLUNT_BASE = "Hexagonal Blunt Base"
    ROUNDED = "Rounded"
    SQUARE = "Square"


#: Equivalent sand-grain roughness height, inches (ar.cs:3599-3637).
SURFACE_ROUGHNESS: dict[str, float] = {
    "Smooth (Zero Roughness)": 0.0,
    "Polished": 5.0e-5,
    "Sheet Metal": 1.6e-4,
    "Smooth Paint": 2.5e-4,
    "Camouflage Paint": 4.0e-4,
    "Rough Camouflage Paint": 1.2e-3,
    "Galvanized Metal": 6.0e-3,
    "Cast Iron (Very Rough)": 1.0e-2,
}

#: Sections whose wave-drag shape factor depends on a chordwise thickness
#: break. For these, FX1 of zero is not a benign default: it is clamped to
#: 0.01 in (i.cs:1395-1397) and the resulting shape factor
#: K = (c/0.01)/(1 - 0.01/c) is enormous. RASAero's own dialog defaults FX1
#: and FX3 to 0.0000, so a file that never touched them is degenerate here.
WEDGE_SECTIONS = frozenset({
    Airfoil.DOUBLE_WEDGE,
    Airfoil.HEXAGONAL,
    Airfoil.SINGLE_WEDGE,
    Airfoil.HEXAGONAL_BLUNT_BASE,
    Airfoil.BICONVEX,
})


# ---------------------------------------------------------------------------
# Parts
# ---------------------------------------------------------------------------


@dataclass
class Part:
    """Base geometry every part carries (``aq`` in the original)."""

    x0: float = 0.0        # forward station, in from nose tip
    length: float = 0.0    # in
    d_fwd: float = 0.0     # in
    d_aft: float = 0.0     # in

    #: Filled by the geometry pass; kept here because the solver's assembly
    #: step reads per-part working storage rather than return values.
    part_type: PartType = PartType.BODY_TUBE


@dataclass
class NoseCone(Part):
    shape: NoseShape = NoseShape.TANGENT_OGIVE
    blunt_radius: float = 0.0
    power_law_n: float = 0.5
    part_type: PartType = PartType.NOSE_CONE

    def __post_init__(self) -> None:
        self.part_type = PartType.NOSE_CONE
        self.x0 = 0.0     # the nose is always the origin


@dataclass
class BodyTube(Part):
    part_type: PartType = PartType.BODY_TUBE

    def __post_init__(self) -> None:
        self.part_type = PartType.BODY_TUBE
        if self.d_fwd == 0.0:
            self.d_fwd = self.d_aft
        if self.d_aft == 0.0:
            self.d_aft = self.d_fwd


@dataclass
class Expansion(Part):
    """A transition whose aft diameter exceeds its forward diameter."""

    part_type: PartType = PartType.EXPANSION

    def __post_init__(self) -> None:
        self.part_type = PartType.EXPANSION


@dataclass
class Reducer(Part):
    """A transition whose aft diameter is smaller. Boattails are Reducers."""

    part_type: PartType = PartType.REDUCER

    def __post_init__(self) -> None:
        self.part_type = PartType.REDUCER


@dataclass
class Fins(Part):
    count: int = 4
    root_chord: float = 0.0
    tip_chord: float = 0.0
    span: float = 0.0            # exposed semispan of one panel
    sweep: float = 0.0           # leading-edge axial sweep distance
    thickness: float = 0.0
    le_radius: float = 0.0
    airfoil: Airfoil = Airfoil.HEXAGONAL
    fx1: float = 0.0
    fx3: float = 0.0
    part_type: PartType = PartType.FINS

    def __post_init__(self) -> None:
        self.part_type = PartType.FINS
        # i.cs:1305 -- a zero sweep would make tan(sweep/span) degenerate and
        # the leading-edge sweep angle undefined.
        if self.sweep == 0.0:
            self.sweep = 0.01
        # The fin occupies no axial extent of its own in the part list; its
        # station is x0 and its chord is carried separately.
        self.length = 0.0

    @property
    def is_degenerate(self) -> bool:
        """RASAero silently zeroes a fin set with no count, chord or span.

        See the guard repeated at i.cs:2475, 2572, 2650, 2731, 2751, 2833,
        2887, 3120, 3147, 3260. Reproduced rather than raising, because a
        design can legitimately carry a disabled fin set.
        """
        return self.count == 0 or self.root_chord == 0.0 or self.span == 0.0


@dataclass
class RailGuide(Part):
    guide_diameter: float = 0.0
    guide_height: float = 0.0
    part_type: PartType = PartType.RAIL_GUIDE

    def __post_init__(self) -> None:
        self.part_type = PartType.RAIL_GUIDE


@dataclass
class LaunchLug(Part):
    lug_diameter: float = 0.0
    lug_length: float = 0.0
    part_type: PartType = PartType.LAUNCH_LUG

    def __post_init__(self) -> None:
        self.part_type = PartType.LAUNCH_LUG


@dataclass
class LaunchShoe(Part):
    area: float = 0.0            # in^2, entered directly
    part_type: PartType = PartType.LAUNCH_SHOE

    def __post_init__(self) -> None:
        self.part_type = PartType.LAUNCH_SHOE


@dataclass
class Plate(Part):
    """An inclined flat-plate protuberance."""

    area: float = 0.0            # in^2 frontal
    angle_deg: float = 0.0
    part_type: PartType = PartType.PLATE

    def __post_init__(self) -> None:
        self.part_type = PartType.PLATE


@dataclass
class StreamlinedNoBaseDrag(Part):
    area: float = 0.0
    part_type: PartType = PartType.STREAMLINED_NO_BASE_DRAG

    def __post_init__(self) -> None:
        self.part_type = PartType.STREAMLINED_NO_BASE_DRAG


@dataclass
class StreamlinedWithBaseDrag(Part):
    area: float = 0.0
    part_type: PartType = PartType.STREAMLINED_WITH_BASE_DRAG

    def __post_init__(self) -> None:
        self.part_type = PartType.STREAMLINED_WITH_BASE_DRAG


# ---------------------------------------------------------------------------
# Design
# ---------------------------------------------------------------------------


@dataclass
class Design:
    """A part list plus the engine mode flags.

    The flags are engine state, not per-query arguments: changing any of them
    invalidates the cached geometry pass, exactly as ``ca = false`` does at
    i.cs:413, 442, 455, 468.
    """

    parts: list[Part] = field(default_factory=list)
    surface: str = "Smooth (Zero Roughness)"
    modified_barrowman: bool = False
    turbulence: bool = False
    nozzle_diameter: float = 0.0
    #: (Mach, altitude_ft) pairs. Empty means every Mach is evaluated at sea
    #: level, which is RASAero's default and the largest single source of
    #: optimistic drag at altitude.
    mach_alt: list[tuple[float, float]] = field(default_factory=list)

    @property
    def roughness(self) -> float:
        """Equivalent sand-grain height, inches. One value for the vehicle."""
        try:
            return SURFACE_ROUGHNESS[self.surface]
        except KeyError:
            raise ValueError(
                f"unknown surface finish {self.surface!r}; "
                f"expected one of {sorted(SURFACE_ROUGHNESS)}"
            ) from None

    def altitude_for(self, mach: float) -> float:
        """Interpolate the Mach/Alt table (i.cs:847-870).

        Flat-extrapolated outside the tabulated range, and zero when the table
        is empty. RASAero seeds the search with (0, 0) and (25, 0), so a table
        that does not start at Mach 0 still interpolates from sea level.
        """
        if not self.mach_alt:
            return 0.0
        table = sorted(self.mach_alt)
        lo_m, lo_a = 0.0, 0.0
        for m, a in table:
            if mach >= m:
                lo_m, lo_a = m, a
                continue
            if m == lo_m:
                return a
            return lo_a + (a - lo_a) * ((mach - lo_m) / (m - lo_m))
        return lo_a


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


def add_fin_can(
    parts: list[Part],
    *,
    length: float,
    outside_diameter: float,
    inside_diameter: float,
    shoulder_length: float,
    x0: float,
) -> None:
    """Append a fin can, RASAero-style (``i.cs:634-655``).

    There is no fin-can part in the solver. RASAero rewrites the vehicle:
    it walks back to the last BodyTube already emitted and *shortens it* by
    the can plus shoulder, then appends an Expansion for the step up to the
    can's outside diameter and a BodyTube for the can itself.

    That the preceding tube is mutated after being emitted is why a design
    has to be built in one pass rather than assembled from independent parts.
    RASAero performs no validation here, so a can longer than the tube ahead
    of it yields a negative length and nonsense downstream; that is checked
    for here because reproducing a crash is not reproducing a behaviour.
    """
    intervening: list[str] = []
    for part in reversed(parts):
        if part.part_type is PartType.BODY_TUBE:
            if intervening:
                # RASAero shortens the last BodyTube no matter what lies
                # between it and the can, which for an intervening transition
                # produces a part list that cannot describe a solid: the can
                # is placed over the transition's station range and a hole is
                # left where the tube used to reach. It still returns numbers.
                #
                # Refused rather than reproduced. Bug-for-bug fidelity is worth
                # having where the input is a real vehicle; here the input is
                # not one, and silently matching RASAero's arithmetic on
                # overlapping geometry would dress nonsense up as agreement.
                raise ValueError(
                    "a fin can must sit directly on a body tube, but "
                    f"{', '.join(intervening)} lies between them. RASAero "
                    "shortens the tube anyway, which overlaps the can with "
                    "that part and leaves a gap in the body; the resulting "
                    "geometry is not a solid. Put a body tube between the "
                    "transition and the fin can."
                )
            shortened = part.length - (length + shoulder_length)
            if shortened < 0:
                raise ValueError(
                    f"fin can ({length} + {shoulder_length} in shoulder) is longer "
                    f"than the {part.length} in body tube ahead of it"
                )
            part.length = shortened
            break
        if part.part_type in (PartType.EXPANSION, PartType.REDUCER, PartType.NOSE_CONE):
            intervening.append(part.part_type.value)
    else:
        raise ValueError("a fin can must follow a body tube")

    parts.append(Expansion(
        x0=x0, length=shoulder_length,
        d_fwd=inside_diameter, d_aft=outside_diameter,
    ))
    parts.append(BodyTube(
        x0=x0 + shoulder_length, length=length,
        d_fwd=outside_diameter, d_aft=outside_diameter,
    ))
