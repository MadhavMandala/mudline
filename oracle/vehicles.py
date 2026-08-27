"""Minimal CDX1 vehicles that each isolate one branch of RASAero's solver.

Deliberately independent of ``step_to_rasaero.rasaero_writer``. That writer
exists to express *our* vehicles in RASAero's vocabulary, and it carries the
assumptions of that translation. An oracle that shares those assumptions
cannot detect them. These files are written field-by-field so a test case can
be degenerate on purpose -- a fin can shorter than its shoulder, a boattail
past the 17.5 degree clamp -- which a sane exporter would never emit.

Geometry conventions, from ``a6.cs:228-402``:

* ``Location`` on a body part is its forward station, inches from the nose tip.
* ``Location`` on a *fin* is measured from the aft end of its parent part,
  forwards. The absolute station RASAero uses is
  ``sum(lengths through parent) - Location``.
* A Transition becomes an Expansion when its rear diameter exceeds its front
  diameter, and a Reducer otherwise. A BoatTail is always a Reducer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

SURFACE_FINISHES = [
    "Smooth (Zero Roughness)", "Polished", "Sheet Metal", "Smooth Paint",
    "Camouflage Paint", "Rough Camouflage Paint", "Galvanized Metal",
    "Cast Iron (Very Rough)",
]

NOSE_SHAPES = [
    "Conical", "Tangent Ogive", "Von Karman Ogive", "Power Law",
    "LV-Haack", "Parabolic", "Elliptical",
]

AIRFOILS = [
    "Hexagonal", "Subsonic NACA", "Double Wedge", "Biconvex",
    "Single Wedge", "Hexagonal Blunt Base", "Rounded", "Square",
]


@dataclass
class Fin:
    count: int = 4
    chord: float = 8.0            # root chord
    tip_chord: float = 4.0
    span: float = 4.0             # exposed semispan
    sweep: float = 4.0            # leading-edge axial sweep
    thickness: float = 0.125
    le_radius: float = 0.0
    airfoil: str = "Hexagonal"
    fx1: float = 2.0
    fx3: float = 2.0
    location: float = 0.0         # from the AFT end of the parent, forwards


@dataclass
class Vehicle:
    name: str
    nose_shape: str = "Tangent Ogive"
    nose_length: float = 12.0
    nose_power_law: float = 0.5
    blunt_radius: float = 0.0
    diameter: float = 4.0
    body_length: float = 48.0
    fin: Fin | None = None
    #: A second fin set, always on the body tube. ``fin`` goes on the fin can
    #: when there is one, so two sets are needed to exercise a vehicle that
    #: carries fins both forward and aft -- which is where the per-fin-set
    #: ordinal, the independent turbulence flags and two different equivalent
    #: diameters all interact.
    forward_fin: Fin | None = None
    surface: str = "Smooth (Zero Roughness)"
    modified_barrowman: bool = False
    turbulence: bool = False
    nozzle_diameter: float = 0.0
    rail_guide_dia: float = 0.0
    rail_guide_height: float = 0.0
    launch_lug_dia: float = 0.0
    launch_lug_len: float = 0.0
    launch_shoe_area: float = 0.0
    streamlined_no_base: float = 0.0
    streamlined_with_base: float = 0.0
    #: (frontal area, angle degrees) for up to two inclined plates.
    plate1: tuple[float, float] | None = None
    plate2: tuple[float, float] | None = None
    # A tail Transition or BoatTail, as (length, front_dia, rear_dia).
    boattail: tuple[float, float, float] | None = None
    transition: tuple[float, float, float] | None = None
    # FinCan as (length, outside_dia, shoulder_length); fins mount on it.
    fin_can: tuple[float, float, float] | None = None
    mach_alt: list[tuple[float, float]] = field(default_factory=list)
    comments: str = ""


def _fmt(value: float) -> str:
    """Trim a trailing '.0' so the text matches RASAero's own output."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _t(parent: ET.Element, tag: str, value: object) -> None:
    ET.SubElement(parent, tag).text = (
        "True" if value is True else "False" if value is False else str(value)
    )


def _fins(parent: ET.Element, fin: Fin) -> None:
    e = ET.SubElement(parent, "Fin")
    _t(e, "Count", fin.count)
    _t(e, "Chord", fin.chord)
    _t(e, "Span", fin.span)
    _t(e, "SweepDistance", fin.sweep)
    _t(e, "TipChord", fin.tip_chord)
    _t(e, "Thickness", fin.thickness)
    _t(e, "LERadius", fin.le_radius)
    _t(e, "Location", fin.location)
    _t(e, "AirfoilSection", fin.airfoil)
    _t(e, "FX1", fin.fx1)
    _t(e, "FX3", fin.fx3)


def _protuberance(parent: ET.Element, v: Vehicle) -> None:
    """The <Protuberance> block, named as t.cs:245-262 writes it.

    RASAero gates the whole element on any one of the four areas being
    non-zero and then writes all six fields, so a zero angle beside a
    non-zero area is normal.
    """
    p1 = v.plate1 or (0.0, 0.0)
    p2 = v.plate2 or (0.0, 0.0)
    if max(v.streamlined_no_base, v.streamlined_with_base, p1[0], p2[0]) <= 0.0:
        return
    e = ET.SubElement(parent, "Protuberance")
    _t(e, "StreamlinedNoBaseDrag", v.streamlined_no_base)
    _t(e, "StreamlinedWithBaseDrag", v.streamlined_with_base)
    _t(e, "InclinedPlate1Angle", p1[1])
    _t(e, "InclinedPlate1FrontalArea", p1[0])
    _t(e, "InclinedPlate2Angle", p2[1])
    _t(e, "InclinedPlate2FrontalArea", p2[0])


def write_cdx1(v: Vehicle, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    doc = ET.Element("RASAeroDocument")
    _t(doc, "FileVersion", 2)
    d = ET.SubElement(doc, "RocketDesign")

    nose = ET.SubElement(d, "NoseCone")
    _t(nose, "PartType", "NoseCone")
    _t(nose, "Length", v.nose_length)
    _t(nose, "Diameter", v.diameter)
    _t(nose, "Shape", v.nose_shape)
    _t(nose, "BluntRadius", v.blunt_radius)
    _t(nose, "Location", 0)
    _t(nose, "Color", "Black")
    if v.nose_shape == "Power Law":
        _t(nose, "PowerLaw", v.nose_power_law)

    x = v.nose_length
    tube = ET.SubElement(d, "BodyTube")
    _t(tube, "PartType", "BodyTube")
    _t(tube, "Length", v.body_length)
    _t(tube, "Diameter", v.diameter)
    _t(tube, "LaunchLugDiameter", v.launch_lug_dia)
    _t(tube, "LaunchLugLength", v.launch_lug_len)
    _t(tube, "RailGuideDiameter", v.rail_guide_dia)
    _t(tube, "RailGuideHeight", v.rail_guide_height)
    _t(tube, "LaunchShoeArea", v.launch_shoe_area)
    _t(tube, "Location", x)
    _t(tube, "Color", "Black")
    _t(tube, "BoattailLength", 0)
    _t(tube, "BoattailRearDiameter", 0)
    _t(tube, "BoattailOffset", 0)
    _t(tube, "Overhang", 0)
    if v.forward_fin is not None:
        _fins(tube, v.forward_fin)
    elif v.fin is not None and v.fin_can is None:
        _fins(tube, v.fin)
    _protuberance(tube, v)
    x += v.body_length

    if v.transition is not None:
        length, front, rear = v.transition
        tr = ET.SubElement(d, "Transition")
        _t(tr, "PartType", "Transition")
        _t(tr, "Length", length)
        # Named for RASAero's READER, not its writer. Its ExtractTrans
        # (t.cs:907-935) maps Diameter to the REAR diameter and takes the
        # front from FrontDiameter; it has no RearDiameter case. Emitting the
        # writer's own Diameter/RearDiameter pair leaves the front diameter at
        # zero and RASAero crashes during its geometry pass rather than
        # opening the file.
        _t(tr, "FrontDiameter", front)
        _t(tr, "Diameter", rear)
        _t(tr, "Location", x)
        _t(tr, "Color", "Black")
        x += length

    if v.fin_can is not None:
        length, outside, shoulder = v.fin_can
        fc = ET.SubElement(d, "FinCan")
        _t(fc, "PartType", "FinCan")
        _t(fc, "Length", length)
        _t(fc, "Diameter", outside)
        _t(fc, "InsideDiameter", v.diameter)
        _t(fc, "LaunchLugDiameter", 0)
        _t(fc, "LaunchLugLength", 0)
        _t(fc, "RailGuideDiameter", 0)
        _t(fc, "RailGuideHeight", 0)
        _t(fc, "LaunchShoeArea", 0)
        _t(fc, "Location", x)
        _t(fc, "ShoulderLength", shoulder)
        _t(fc, "Offset", -length)
        _t(fc, "Color", "Black")
        if v.fin is not None:
            _fins(fc, v.fin)
        x += length

    if v.boattail is not None:
        length, front, rear = v.boattail
        bt = ET.SubElement(d, "BoatTail")
        _t(bt, "PartType", "BoatTail")
        _t(bt, "Length", length)
        _t(bt, "Diameter", front)
        _t(bt, "RearDiameter", rear)
        _t(bt, "Location", x)
        _t(bt, "Color", "Black")

    _t(d, "Surface", v.surface)
    _t(d, "CP", 0)
    _t(d, "ModifiedBarrowman", v.modified_barrowman)
    _t(d, "Turbulence", v.turbulence)
    _t(d, "SustainerNozzle", v.nozzle_diameter)
    _t(d, "Booster1Nozzle", 0)
    _t(d, "Booster2Nozzle", 0)
    _t(d, "UseBooster1", False)
    _t(d, "UseBooster2", False)
    _t(d, "Comments", v.comments or f"Oracle case: {v.name}")

    site = ET.SubElement(doc, "LaunchSite")
    _t(site, "Altitude", 0)
    _t(site, "Pressure", 0)
    _t(site, "RodAngle", 0)
    _t(site, "RodLength", 10)
    _t(site, "Temperature", 59)
    _t(site, "WindSpeed", 0)

    rec = ET.SubElement(doc, "Recovery")
    for i in (1, 2):
        _t(rec, f"Altitude{i}", 1000)
        _t(rec, f"DeviceType{i}", "None")
        _t(rec, f"Event{i}", False)
        _t(rec, f"Size{i}", 1)
        _t(rec, f"EventType{i}", "None")
        _t(rec, f"CD{i}", 1.33)

    # One <Item> per row, holding "mach, altitude" as text (t.cs:168-172).
    # GetMachAlt (t.cs:544-562) matches on the element name "Item" and splits
    # on the first comma; any other structure is silently skipped, leaving the
    # table empty and every Mach evaluated at sea level. That failure is
    # invisible -- the run succeeds and the coefficients look plausible.
    ma = ET.SubElement(doc, "MachAlt")
    for mach, alt in v.mach_alt:
        ET.SubElement(ma, "Item").text = f"{_fmt(mach)}, {_fmt(alt)}"

    ET.SubElement(doc, "SimulationList")

    ET.indent(doc, space="  ")
    path.write_text(ET.tostring(doc, encoding="unicode"), encoding="utf-8")
    return path


def _slug(s: str) -> str:
    """Filename-safe slug that keeps every word.

    Truncating to the first word collides: "Smooth (Zero Roughness)" and
    "Smooth Paint" would both become "smooth", and one case would silently
    overwrite the other's dump -- producing an oracle that looks complete and
    quietly contains the same rocket twice.
    """
    out = [c.lower() if c.isalnum() else "_" for c in s]
    return "_".join(filter(None, "".join(out).split("_")))


def test_matrix() -> list[Vehicle]:
    """The per-feature matrix from the rebuild spec, section 13.

    Each case changes one thing from the baseline so a disagreement points at
    one module. A single "realistic" rocket exercises many branches at once
    and localises nothing.
    """
    base = dict(nose_shape="Tangent Ogive", diameter=4.0, nose_length=12.0, body_length=48.0)
    cases: list[Vehicle] = []

    # 1 - body alone: friction, form, base, Reynolds chain.
    cases.append(Vehicle(name="body_only", **base))

    # 2 - nose families. Power Law is separated out because its exponent
    #     selects between two different wetted-area and wave-drag fits.
    for shape in NOSE_SHAPES:
        if shape == "Power Law":
            for n in (0.0, 0.25, 0.5, 0.75, 1.0):
                cases.append(Vehicle(
                    name=f"nose_powerlaw_{n}".replace(".", "p"),
                    **{**base, "nose_shape": shape, "nose_power_law": n},
                ))
        else:
            cases.append(Vehicle(
                name=f"nose_{shape.lower().replace(' ', '_')}",
                **{**base, "nose_shape": shape},
            ))

    # 3 - blunted nose: exercises the Rayleigh stagnation term (spec E.2).
    cases.append(Vehicle(name="nose_blunt", **{**base, "blunt_radius": 0.25}))

    # 4 - surface finish: the roughness cutoff, both branches.
    for s in SURFACE_FINISHES:
        cases.append(Vehicle(name=f"finish_{_slug(s)}", **{**base, "surface": s}))

    # 5 - turbulence flag against a rough surface, where the 182500 / 2580000
    #     snap in spec C.1 can actually fire.
    for turb in (False, True):
        cases.append(Vehicle(
            name=f"turb_{'on' if turb else 'off'}",
            **{**base, "surface": "Smooth Paint", "turbulence": turb}))

    # 6 - fin sections.
    for af in AIRFOILS:
        cases.append(Vehicle(
            name=f"fin_{af.lower().replace(' ', '_')}",
            **base, fin=Fin(airfoil=af)))

    # 7 - fin count: Psi(N) and the interference factors.
    for n in range(3, 9):
        cases.append(Vehicle(name=f"fincount_{n}", **base, fin=Fin(count=n)))

    # 8 - Barrowman vs Modified: a different model, not a refinement.
    for mod in (False, True):
        cases.append(Vehicle(
            name=f"barrowman_{'mod' if mod else 'classic'}",
            **base, fin=Fin(), modified_barrowman=mod))

    # 9 - unswept and heavily swept fins: M_LE = 1/cos(sweep) moves the
    #     linear/Newtonian switchover, and sweep=0 hits the tan(0.01) guard.
    for sweep in (0.0, 2.0, 12.0):
        cases.append(Vehicle(
            name=f"finsweep_{str(sweep).replace('.', 'p')}",
            **base, fin=Fin(sweep=sweep)))

    # 10 - fin can: the equivalent-volume radius of spec A.2, the highest
    #      risk item in the port.
    cases.append(Vehicle(
        name="fincan", **base, fin=Fin(location=2.0), fin_can=(10.0, 4.5, 1.0)))

    # 11 - boattail either side of the 17.5 degree clamp.
    cases.append(Vehicle(name="boattail_shallow", **base, boattail=(6.0, 4.0, 3.0)))
    cases.append(Vehicle(name="boattail_steep", **base, boattail=(1.5, 4.0, 2.0)))

    # 12 - expansion and reduction transitions.
    cases.append(Vehicle(name="transition_expand", **base, transition=(4.0, 4.0, 5.0)))
    cases.append(Vehicle(name="transition_reduce", **base, transition=(4.0, 4.0, 3.0)))

    # 13 - protuberances.
    cases.append(Vehicle(name="railguide", **base, rail_guide_dia=0.5, rail_guide_height=0.5))
    cases.append(Vehicle(name="launchlug", **base, launch_lug_dia=0.5, launch_lug_len=2.0))

    # 14 - a populated Mach/Alt table, which couples Reynolds number to
    #      altitude. Empty is the default and pins everything to sea level.
    cases.append(Vehicle(
        name="machalt", **base,
        mach_alt=[(0.0, 0.0), (1.0, 10000.0), (3.0, 40000.0), (5.0, 80000.0)]))

    return cases


def combination_matrix() -> list[Vehicle]:
    """Vehicles that exercise several features at once.

    ``test_matrix`` changes one thing at a time, which localises a failure but
    cannot see an *interaction*. Interactions are where this port's real bugs
    lived: the part-ordering defect only appeared because a fin set and a
    boattail were on the same vehicle and the base-drag scaling indexes
    ``m_a[Count-1]`` positionally. A matrix of isolated features would never
    have caught it.

    Each case below targets a specific interaction rather than a feature.
    """
    base = dict(nose_shape="Tangent Ogive", diameter=4.0,
                nose_length=12.0, body_length=48.0)
    cases: list[Vehicle] = []

    # The exact shape that hid the part-ordering bug: a fin set followed by a
    # boattail, so the last part in the list is not the one base drag wants.
    cases.append(Vehicle(
        name="combo_fincan_boattail", **base,
        fin=Fin(location=2.0), fin_can=(10.0, 4.5, 1.0),
        boattail=(3.0, 4.5, 3.0),
    ))

    # Expansion then reducer with fins between them. Exercises the
    # aft-body-length latch (which stops accumulating at the FIRST transition
    # but keeps walking), the base-diameter search, and the rule that the
    # boattail angle is zero unless the LAST body part is itself a reducer.
    #
    # No fin can here on purpose: a can placed directly after a transition
    # makes RASAero shorten the tube out from under the transition, leaving
    # overlapping parts and a hole in the body. add_fin_can refuses that
    # layout rather than reproducing arithmetic on a shape that is not solid.
    cases.append(Vehicle(
        name="combo_expand_fins_boattail", **base,
        transition=(4.0, 4.0, 5.0),
        fin=Fin(location=2.0),
        boattail=(4.0, 5.0, 3.0),
    ))

    # Two fin sets at different stations with different sections. The square
    # set forces its own turbulence flag while the hexagonal one does not, so
    # this is the only geometry where the per-part and vehicle-wide flags
    # visibly disagree within one solve.
    cases.append(Vehicle(
        name="combo_two_fin_sets", **base,
        forward_fin=Fin(location=30.0, airfoil="Hexagonal", count=4),
        fin=Fin(location=2.0, airfoil="Square", count=3),
        fin_can=(10.0, 4.5, 1.0),
    ))

    # A fin whose root chord overhangs its tube onto a BOATTAIL. This is the
    # only configuration in which RASAero gives a fin set two different body
    # diameters (ar.cs:3355-3366), and therefore the only one in which the
    # equivalent-volume radius of Pass A.2 does any work at all. Chord 14 with
    # Location 2 gives a 12 in overhang onto a 12 in boattail.
    cases.append(Vehicle(
        name="combo_fin_overhangs_boattail", **{**base, "body_length": 40.0},
        fin=Fin(location=2.0, chord=14.0, span=4.0, sweep=3.0),
        boattail=(12.0, 4.0, 2.5),
    ))

    # The same overhang, but with a Transition in the way instead of a
    # boattail. RASAero stops its walk and leaves both diameters equal, so the
    # equivalent-volume path is skipped -- the pair is only meaningful next to
    # the case above.
    cases.append(Vehicle(
        name="combo_fin_overhangs_transition", **{**base, "body_length": 40.0},
        fin=Fin(location=2.0, chord=14.0, span=4.0, sweep=3.0),
        transition=(12.0, 4.0, 5.0),
    ))

    # Every protuberance type at once, on a vehicle that also has fins and a
    # boattail -- the streamlined pair scale off the body's assembled drag,
    # and their subsonic and supersonic term sets differ.
    cases.append(Vehicle(
        name="combo_all_protuberances", **base,
        fin=Fin(),
        rail_guide_dia=0.5, rail_guide_height=0.5,
        launch_lug_dia=0.5, launch_lug_len=2.0,
        launch_shoe_area=0.75,
        streamlined_no_base=0.6, streamlined_with_base=0.4,
        plate1=(0.3, 30.0), plate2=(0.2, 60.0),
        boattail=(3.0, 4.0, 3.0),
    ))

    # The two orderings-sensitive drag paths together: a Power-Law nose
    # (whose form drag is overwritten AFTER base drag is taken from the
    # original) on a vehicle whose base drag is also boattail-scaled.
    cases.append(Vehicle(
        name="combo_powerlaw_boattail",
        **{**base, "nose_shape": "Power Law"}, nose_power_law=0.05,
        fin=Fin(), boattail=(1.5, 4.0, 2.0),
    ))

    # Rough surface + turbulence on + a Mach/Alt table: the roughness cutoff,
    # the discontinuous Reynolds snap and the altitude coupling all at once,
    # on a vehicle with fins so the fin Reynolds is rescaled too.
    cases.append(Vehicle(
        name="combo_rough_turbulent_altitude", **base,
        surface="Rough Camouflage Paint", turbulence=True,
        fin=Fin(airfoil="Rounded"),
        mach_alt=[(0.0, 0.0), (1.0, 15000.0), (3.0, 50000.0), (6.0, 100000.0)],
    ))

    # Modified Barrowman with everything that carries normal force: nose,
    # expansion, reducer and two fin sets, so every CN contribution and every
    # centre-of-pressure station is populated at once.
    cases.append(Vehicle(
        name="combo_modified_barrowman_full", **{**base, "body_length": 36.0},
        modified_barrowman=True,
        transition=(4.0, 4.0, 5.0),
        forward_fin=Fin(location=4.0, count=6, airfoil="Double Wedge"),
        boattail=(6.0, 5.0, 3.5),
        nozzle_diameter=2.5,
    ))

    # The same normal-force coverage in classic Barrowman, so the pair
    # isolates the mode rather than the geometry.
    cases.append(Vehicle(
        name="combo_classic_barrowman_full", **{**base, "body_length": 36.0},
        modified_barrowman=False,
        transition=(4.0, 4.0, 5.0),
        forward_fin=Fin(location=4.0, count=6, airfoil="Double Wedge"),
        boattail=(6.0, 5.0, 3.5),
        nozzle_diameter=2.5,
    ))

    return cases
