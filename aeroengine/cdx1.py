"""Read a RASAero II ``.CDX1`` design into an engine ``Design``.

This exists so the engine and the oracle consume *the same bytes*. Any
hand-translation between them would show up as a physics disagreement and
cost hours to attribute, so there is none: RASAero reads the file, this reads
the file, and the two part lists are built by the same rules.

Those rules live in ``a6.cs:228-402`` (``LoadRocketParts``), not in the file
format. The XML is a description of what the user drew; the part list is what
the solver sees, and the mapping between them is lossy and order-dependent:

* A **Transition** becomes an Expansion or a Reducer depending on which end is
  wider. A **BoatTail** is always a Reducer.
* A **FinCan** does not exist in the solver. It shortens the preceding body
  tube and appends an Expansion plus a BodyTube -- a sleeve over the aft tube,
  not an extension of the vehicle.
* Fin ``Location`` is measured from the aft end of the parent part, forwards,
  against a running length accumulator that some part types advance and others
  do not. A fin can and a boattail both leave the accumulator alone.

Boosters are not handled. Multi-stage designs raise rather than silently
producing a sustainer-only answer, which is the failure mode that would
otherwise waste an afternoon.
"""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

from .parts import (
    Airfoil,
    BodyTube,
    Design,
    Expansion,
    Fins,
    LaunchLug,
    LaunchShoe,
    NoseCone,
    NoseShape,
    Part,
    Plate,
    RailGuide,
    Reducer,
    StreamlinedNoBaseDrag,
    StreamlinedWithBaseDrag,
    add_fin_can,
)

__all__ = ["load", "loads"]


def _f(node: ET.Element | None, tag: str, default: float = 0.0) -> float:
    if node is None:
        return default
    child = node.find(tag)
    if child is None or child.text is None or not child.text.strip():
        return default
    try:
        return float(child.text)
    except ValueError:
        return default


def _s(node: ET.Element | None, tag: str, default: str = "") -> str:
    if node is None:
        return default
    child = node.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _b(node: ET.Element | None, tag: str, default: bool = False) -> bool:
    raw = _s(node, tag, "")
    return raw.lower() == "true" if raw else default


def _fin_from(node: ET.Element, x0: float) -> Fins | None:
    """Build a fin set, or None when the element is absent or disabled."""
    fin_el = node.find("Fin")
    if fin_el is None:
        return None
    count = int(_f(fin_el, "Count"))
    if count <= 0:
        return None
    return Fins(
        x0=x0,
        count=count,
        root_chord=_f(fin_el, "Chord"),
        tip_chord=_f(fin_el, "TipChord"),
        span=_f(fin_el, "Span"),
        sweep=_f(fin_el, "SweepDistance"),
        thickness=_f(fin_el, "Thickness"),
        le_radius=_f(fin_el, "LERadius"),
        airfoil=Airfoil(_s(fin_el, "AirfoilSection", "Hexagonal")),
        fx1=_f(fin_el, "FX1"),
        fx3=_f(fin_el, "FX3"),
        # The body diameters at the fin station. RASAero fills these from the
        # parent part rather than from the file, which is why they are set by
        # the caller and not read here.
        d_fwd=0.0,
        d_aft=0.0,
    )


def _attachments_before_fins(node: ET.Element, parts: list[Part]) -> None:
    """Lug, shoe and rail guide, in a6.cs's emission order.

    Order is NOT cosmetic here. Several formulas index the part list
    positionally -- most consequentially the supersonic base drag, which reads
    ``m_a[Count-1]`` to find the boattail it is scaling against (i.cs:3442).
    A fin set emitted at the wrong point takes that slot, and because a fin
    carries zero length the 17.5-degree clamp then computes a diameter ratio
    of exactly 1 and the boattail correction silently vanishes.
    """
    lug_d = _f(node, "LaunchLugDiameter")
    if lug_d > 0.0:
        parts.append(LaunchLug(lug_diameter=lug_d, lug_length=_f(node, "LaunchLugLength")))

    shoe = _f(node, "LaunchShoeArea")
    if shoe > 0.0:
        parts.append(LaunchShoe(area=shoe))

    guide_d = _f(node, "RailGuideDiameter")
    if guide_d > 0.0:
        parts.append(RailGuide(guide_diameter=guide_d, guide_height=_f(node, "RailGuideHeight")))


def _attachments_after_fins(node: ET.Element, parts: list[Part]) -> None:
    """Streamlined protuberances and inclined plates, emitted after the fins."""
    prot = node.find("Protuberance")
    if prot is None:
        return
    # Element names from the writer at t.cs:245-262. RASAero emits the whole
    # block or none of it, gated on any one of the four areas being non-zero,
    # so a zero-area entry inside a present block is normal and must not
    # produce a part.
    area_a = _f(prot, "StreamlinedNoBaseDrag")
    if area_a > 0.0:
        parts.append(StreamlinedNoBaseDrag(area=area_a))
    area_b = _f(prot, "StreamlinedWithBaseDrag")
    if area_b > 0.0:
        parts.append(StreamlinedWithBaseDrag(area=area_b))
    for area_tag, angle_tag in (
        ("InclinedPlate1FrontalArea", "InclinedPlate1Angle"),
        ("InclinedPlate2FrontalArea", "InclinedPlate2Angle"),
    ):
        area = _f(prot, area_tag)
        if area > 0.0:
            parts.append(Plate(area=area, angle_deg=_f(prot, angle_tag)))


def _fin_body_diameters(
    rocket: ET.Element, tube: ET.Element, tube_diameter: float
) -> tuple[float, float]:
    """The body diameters a fin set sees at its root (``ar.cs:3318-3372``).

    These are NOT in the file. RASAero recomputes them whenever the design
    changes and again after a load, and the CDX1 carries no trace of the
    result -- so a reader that takes the parent part's diameter for both is
    right only by coincidence.

    They differ from each other in exactly one situation: the fin's root chord
    runs off the aft end of its tube and onto a **boattail**. The overhang is
    derived, not stored -- ``max(chord - location, 0)``, since Location is
    measured forward from the tube's aft end -- and the aft diameter is then
    read off the boattail's taper at that overhang.

    A Transition in the way stops the walk and leaves both diameters equal, so
    a fin overhanging onto a transition is treated as if it were not
    overhanging at all. That asymmetry is RASAero's, not a simplification
    here, and it is the difference between the equivalent-volume correction in
    the geometry pass running and being skipped entirely.
    """
    fin_el = tube.find("Fin")
    if fin_el is None:
        return tube_diameter, tube_diameter

    overhang = max(_f(fin_el, "Chord") - _f(fin_el, "Location"), 0.0)
    if overhang <= 0.0:
        return tube_diameter, tube_diameter

    children = list(rocket)
    try:
        start = children.index(tube) + 1
    except ValueError:                                   # pragma: no cover
        return tube_diameter, tube_diameter

    d_aft = tube_diameter
    run = 0.0
    for node in children[start:]:
        if node.tag == "BodyTube":
            run += _f(node, "Length")
            if run > overhang:
                break
        elif node.tag == "BoatTail":
            length = _f(node, "Length")
            front = _f(node, "Diameter")
            rear = _f(node, "RearDiameter")
            if length > 0.0:
                slope = (front - rear) / length
                # Literally as written: the taper is evaluated at the full
                # overhang, without subtracting the tube length already
                # walked past. On the usual layout -- fins on the last tube,
                # boattail immediately aft -- that walked length is zero.
                d_aft = front - slope * overhang
            break
        elif node.tag == "Transition":
            break
    return tube_diameter, d_aft


def loads(text: str) -> Design:
    """Parse CDX1 text."""
    root = ET.fromstring(text)
    rocket = root.find("RocketDesign")
    if rocket is None:
        raise ValueError("not a RASAero document: no <RocketDesign>")

    parts: list[Part] = []
    run = 0.0        # a6.cs's `num`: the running length accumulator

    for node in rocket:
        kind = node.tag

        if kind == "NoseCone":
            length = _f(node, "Length")
            parts.append(NoseCone(
                length=length,
                d_aft=_f(node, "Diameter"),
                shape=NoseShape(_s(node, "Shape", "Tangent Ogive")),
                blunt_radius=_f(node, "BluntRadius"),
                power_law_n=_f(node, "PowerLaw", 0.5),
            ))
            run += length

        elif kind == "BodyTube":
            length = _f(node, "Length")
            diameter = _f(node, "Diameter")
            parts.append(BodyTube(
                x0=_f(node, "Location"), length=length, d_aft=diameter,
            ))
            run += length
            _attachments_before_fins(node, parts)
            fin = _fin_from(node, 0.0)
            if fin is not None:
                fin.d_fwd, fin.d_aft = _fin_body_diameters(rocket, node, diameter)
                fin.x0 = run - _f(node.find("Fin"), "Location")
                parts.append(fin)
            _attachments_after_fins(node, parts)

        elif kind == "Transition":
            # RASAero cannot round-trip its own Transition, and this follows
            # its READER rather than its writer, because the reader is what
            # decides the geometry actually solved.
            #
            #   writer, t.cs:96-104 : Diameter <- front, RearDiameter <- rear
            #   reader, t.cs:907-935: Diameter -> REAR, FrontDiameter -> front,
            #                         and no RearDiameter case at all
            #
            # So a transition saved by RASAero reloads with its front diameter
            # dropped to zero and its front value sitting in the rear slot.
            # Loading such a file crashes the application during its geometry
            # pass -- which is exactly how this was found, four cases of an
            # oracle run reporting "RASAero II did not open a window".
            #
            # BoatTail does not share the defect (ExtractBoat, t.cs:874-901).
            length = _f(node, "Length")
            aft = _f(node, "Diameter")
            fwd = _f(node, "FrontDiameter")
            if fwd == 0.0 and node.find("RearDiameter") is not None:
                raise ValueError(
                    "this Transition was written by RASAero and cannot be read "
                    "back by it: the writer emits Diameter/RearDiameter, the "
                    "reader expects FrontDiameter/Diameter, so the front "
                    "diameter is lost. RASAero itself fails to open this file. "
                    "Write FrontDiameter (front) and Diameter (rear) instead."
                )
            cls = Expansion if fwd < aft else Reducer
            parts.append(cls(
                x0=_f(node, "Location"), length=length, d_fwd=fwd, d_aft=aft,
            ))
            run += length

        elif kind == "BoatTail":
            # Always a Reducer, and it does NOT advance the accumulator.
            length = _f(node, "Length")
            fwd = _f(node, "Diameter")
            parts.append(Reducer(
                x0=_f(node, "Location"), length=length,
                d_fwd=fwd, d_aft=_f(node, "RearDiameter"),
            ))
            fin = _fin_from(node, 0.0)
            if fin is not None:
                fin.d_fwd = fin.d_aft = fwd
                fin.x0 = run + length - _f(node.find("Fin"), "Location")
                parts.append(fin)

        elif kind == "FinCan":
            length = _f(node, "Length")
            outside = _f(node, "Diameter")
            shoulder = _f(node, "ShoulderLength")
            add_fin_can(
                parts,
                length=length,
                outside_diameter=outside,
                inside_diameter=_f(node, "InsideDiameter"),
                shoulder_length=shoulder,
                x0=run - (length + shoulder),
            )
            # No accumulator advance: a can sleeves the aft tube rather than
            # lengthening the vehicle.
            _attachments_before_fins(node, parts)
            fin = _fin_from(node, 0.0)
            if fin is not None:
                fin.d_fwd = fin.d_aft = outside
                fin.x0 = run - _f(node.find("Fin"), "Location")
                parts.append(fin)
            _attachments_after_fins(node, parts)

        elif kind == "Booster":
            raise NotImplementedError(
                "multi-stage designs are not supported: RASAero solves each "
                "stage as a separate vehicle and this engine models one"
            )

    # <MachAlt><Item>mach, altitude</Item>...</MachAlt> -- one element per row
    # holding both numbers as comma-separated text (t.cs:168-172, 544-562).
    # RASAero's reader matches on the name "Item" and ignores anything else,
    # so a differently-shaped table is not an error: it simply yields no rows,
    # and every Mach is then evaluated at sea level.
    mach_alt: list[tuple[float, float]] = []
    table = root.find("MachAlt")
    if table is not None:
        for entry in table.findall("Item"):
            text = (entry.text or "").strip()
            if "," not in text:
                continue
            mach_text, _, alt_text = text.partition(",")
            try:
                mach_alt.append((float(mach_text), float(alt_text)))
            except ValueError:
                continue

    return Design(
        parts=parts,
        surface=_s(rocket, "Surface", "Smooth (Zero Roughness)"),
        modified_barrowman=_b(rocket, "ModifiedBarrowman"),
        turbulence=_b(rocket, "Turbulence"),
        nozzle_diameter=_f(rocket, "SustainerNozzle"),
        mach_alt=mach_alt,
    )


def load(path: str | Path) -> Design:
    return loads(Path(path).read_text(encoding="utf-8", errors="replace"))
