"""The oracle's test vehicles, expressed as Mudline models.

    python -m oracle.mudline_vehicles          # write them all
    python -m oracle.mudline_vehicles --report # convert, report, write nothing

Why this exists. The oracle ships three things that describe the same 60
vehicles: the ``.CDX1`` files RASAero reads, the frozen Run Test dumps
RASAero produced from them, and -- here -- the same vehicles as Mudline
documents you can open in the application. Without the third, anyone without a
RASAero licence can run the comparison but cannot *look* at what is being
compared, which makes the whole oracle a wall of numbers.

Direction of travel. These are generated from ``oracle.vehicles``, the
field-by-field authored definitions, rather than by parsing the CDX1 back.
Round-tripping through the file format would test the CDX1 reader and writer
against each other and prove nothing about either.

**These are approximations, and the oracle does not use them.** Term-by-term
agreement is checked against the CDX1 files, always -- see
``aeroengine/tests/test_oracle.py``. Two reasons the conversion cannot be
exact:

* The oracle vehicles are *deliberately* degenerate. A fin can shorter than
  its own shoulder, a boattail well past the separation clamp, a nose with no
  body behind it: shapes chosen to drive one branch of RASAero's solver, not
  to be manufacturable. Mudline's parms are bounded because real parts are,
  so one of these may not be representable at all; any that is not is reported
  as skipped rather than silently clamped into a different vehicle.

  This has already earned its keep in the other direction. The only vehicle
  that would not convert, ``finsweep_12p0``, was not degenerate -- Mudline's
  validator was refusing a swept fin whose tip trails past the tail, which
  real vehicles have. The conversion found a bug in the tool rather than a
  limit of it.
* RASAero's nose catalogue is not Mudline's. LV-Haack and Parabolic have no
  generator here and are approximated; the mapping is recorded on each model
  it applies to, in its description, so a reader is never left guessing which
  shape they are actually looking at.

So: open these to see the shape, sweep them, fly them. Do not cite one as
evidence of what RASAero computed -- cite the CDX1 and the dump.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

# Roughness heights come from the engine rather than being restated here. A
# second copy of that table is exactly the bug
# ``adapters.surface_for_roughness`` documents: when two places decide a
# finish independently they eventually disagree, and then two parts of the
# tool are quietly modelling different vehicles.
from aeroengine.adapters import SURFACE_ROUGHNESS_M
from oracle.vehicles import Fin, Vehicle, combination_matrix, test_matrix
from parametric.components import FinSet, Protuberance, Stack
from parametric.model import VehicleModel
from parametric.xsec import NoseProfile

#: The oracle speaks RASAero's units. Mudline stores SI.
M_PER_IN = 0.0254

OUT_DIR = Path(__file__).resolve().parent / "vehicles_mudline"

#: RASAero's nose catalogue against Mudline's generators. The two marked
#: *approximate* have no generator here; the nearest profile is used and the
#: substitution is written into the model's description so it travels with it.
NOSE_PROFILES: dict[str, tuple[NoseProfile, str]] = {
    "Conical": (NoseProfile.CONICAL, ""),
    "Tangent Ogive": (NoseProfile.OGIVE, ""),
    "Von Karman Ogive": (NoseProfile.VON_KARMAN, ""),
    "Elliptical": (NoseProfile.ELLIPTICAL, ""),
    "Power Law": (NoseProfile.POWER_HALF, ""),
    "LV-Haack": (
        NoseProfile.VON_KARMAN,
        "RASAero's LV-Haack is approximated by a von Karman ogive: both are "
        "Sears-Haack family curves and this is the closer of the generators "
        "available. The CDX1 is the authority on the shape RASAero solved.",
    ),
    "Parabolic": (
        NoseProfile.ELLIPTICAL,
        "RASAero's Parabolic is approximated by an elliptical nose. The CDX1 "
        "is the authority on the shape RASAero solved.",
    ),
}



@dataclass
class Conversion:
    """One vehicle's outcome, so the run can report rather than just fail."""

    name: str
    model: VehicleModel | None = None
    skipped: str = ""
    notes: list[str] = None      # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.notes is None:
            self.notes = []


def _fin_set(fin: Fin, parent_aft_m: float, name: str) -> FinSet:
    """A RASAero fin set as a Mudline one.

    ``Fin.location`` is measured from the aft end of the parent part, forwards
    -- the file format's convention, and the one thing in this conversion most
    likely to be got backwards. Mudline stations are absolute from the nose
    tip, so the parent's aft station is where it is measured from.
    """
    return FinSet(
        name,
        count=fin.count,
        root_chord_m=fin.chord * M_PER_IN,
        tip_chord_m=fin.tip_chord * M_PER_IN,
        span_m=fin.span * M_PER_IN,
        sweep_m=fin.sweep * M_PER_IN,
        thickness_m=fin.thickness * M_PER_IN,
        station_m=(parent_aft_m - fin.location * M_PER_IN - fin.chord * M_PER_IN),
    )


def _protuberances(v: Vehicle, model: VehicleModel, body_aft_m: float) -> list[str]:
    """Rail guides, lugs, shoes and plates, as declared frontal areas.

    RASAero takes these as areas and angles rather than geometry, and so does
    Mudline's Protuberance, so this is one of the few parts of the conversion
    that is not an approximation at all.
    """
    notes: list[str] = []
    station = body_aft_m * 0.6         # nothing declares a station; mid-body

    if v.rail_guide_dia > 0.0 and v.rail_guide_height > 0.0:
        area = (v.rail_guide_dia * v.rail_guide_height) * M_PER_IN ** 2
        model.add(Protuberance("rail_guide", shape="rail_button",
                               frontal_area_m2=area, station_m=station))
    if v.launch_lug_dia > 0.0:
        area = 3.14159 * (v.launch_lug_dia * M_PER_IN / 2.0) ** 2
        model.add(Protuberance("launch_lug", shape="launch_lug",
                               frontal_area_m2=area, station_m=station))
    if v.launch_shoe_area > 0.0:
        model.add(Protuberance("launch_shoe", shape="rail_button",
                               frontal_area_m2=v.launch_shoe_area * M_PER_IN ** 2,
                               station_m=station))
    for index, plate in enumerate((v.plate1, v.plate2), start=1):
        if plate:
            area, angle = plate
            model.add(Protuberance(f"plate_{index}", shape="rail_button",
                                   frontal_area_m2=area * M_PER_IN ** 2,
                                   station_m=station))
            notes.append(
                f"Inclined plate {index} was entered in RASAero at {angle:g} deg; "
                "Mudline's protuberance takes a frontal area and a shape "
                "coefficient, with no angle, so the drag will not match."
            )
    return notes


def to_model(v: Vehicle) -> Conversion:
    """Build one Mudline model, or say why it cannot be built."""
    result = Conversion(name=v.name)

    profile, note = NOSE_PROFILES.get(v.nose_shape, (None, ""))
    if profile is None:
        result.skipped = f"no generator for nose shape {v.nose_shape!r}"
        return result
    if note:
        result.notes.append(note)

    try:
        model = VehicleModel(v.name, "A")

        diameter_m = v.diameter * M_PER_IN
        nose_len_m = v.nose_length * M_PER_IN

        nose = Stack("nose", wall_thickness_m=0.003)
        nose.add_nose(profile, length_m=nose_len_m, diameter_m=diameter_m,
                      sections=16, tip_radius_m=max(v.blunt_radius * M_PER_IN,
                                                    0.0015))
        model.add(nose)
        station = nose_len_m

        body = Stack("body_tube", wall_thickness_m=0.003)
        body.add_tube(length_m=v.body_length * M_PER_IN, diameter_m=diameter_m,
                      name="body")
        _shift(body, station)
        model.add(body)
        station += v.body_length * M_PER_IN
        body_aft_m = station

        # A fin can is a short tube of its own outside diameter, and the fins
        # mount on it rather than on the body.
        fin_parent, fin_parent_aft = body, body_aft_m
        if v.fin_can:
            can_len, can_dia, _shoulder = v.fin_can
            can = Stack("fin_can", wall_thickness_m=0.003)
            can.add_tube(length_m=can_len * M_PER_IN,
                         diameter_m=can_dia * M_PER_IN, name="can")
            _shift(can, station)
            model.add(can)
            station += can_len * M_PER_IN
            fin_parent, fin_parent_aft = can, station

        # Transition first, then boattail: the order they occur aft-ward.
        for label, spec in (("transition", v.transition), ("boattail", v.boattail)):
            if not spec:
                continue
            length, _front, rear = spec
            tail = Stack(label, wall_thickness_m=0.003)
            tail.add_tube(length_m=1e-4, diameter_m=_front * M_PER_IN, name="lip")
            tail.add_transition(length_m=length * M_PER_IN,
                                rear_diameter_m=rear * M_PER_IN, sections=8)
            _shift(tail, station)
            model.add(tail)
            station += length * M_PER_IN

        if v.fin:
            fin_parent.add(_fin_set(v.fin, fin_parent_aft, "fins"))
        if v.forward_fin:
            body.add(_fin_set(v.forward_fin, body_aft_m, "forward_fins"))

        result.notes += _protuberances(v, model, body_aft_m)

        problems = model.validate()
        if problems:
            result.skipped = "; ".join(problems[:3])
            return result

        roughness = SURFACE_ROUGHNESS_M.get(v.surface)
        described = [
            f"Oracle test vehicle: {v.name}.",
            f"RASAero surface finish {v.surface!r}"
            + (f" (~{roughness * 1e6:.0f} um)." if roughness else "."),
            "Generated from oracle/vehicles.py. An approximation for viewing "
            "and flying -- the CDX1 beside it is what RASAero actually solved.",
        ]
        model.description = " ".join(described + result.notes)
        result.model = model
    except Exception as exc:      # noqa: BLE001 - a degenerate shape is data
        result.skipped = f"{type(exc).__name__}: {exc}"
    return result


def _shift(stack: Stack, station_m: float) -> None:
    """Move a stack's sections aft so it starts at the given station.

    Stations go through ``Parm.set`` rather than being assigned: ``station_m``
    is a read-only view of the parm, and the parm is what carries the bounds
    and the change tracking a rebuild depends on.
    """
    low = stack.station_range_m()[0]
    for section in stack.sections:
        section.set("station", section.station_m - low + station_m)
    stack.mark_dirty("shift")


def convert_all() -> list[Conversion]:
    return [to_model(v) for v in (*test_matrix(), *combination_matrix())]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true",
                        help="convert and report, but write nothing")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)

    results = convert_all()
    built = [r for r in results if r.model is not None]
    skipped = [r for r in results if r.model is None]

    if not args.report:
        args.out.mkdir(parents=True, exist_ok=True)
        for result in built:
            path = args.out / f"{_slug(result.name)}.json"
            path.write_text(
                json.dumps(result.model.to_dict(), indent=2), encoding="utf-8"
            )

    print(f"{len(built)} of {len(results)} oracle vehicles became Mudline models")
    if not args.report:
        print(f"  written to {args.out}")

    noted = [r for r in built if r.notes]
    if noted:
        print(f"\n{len(noted)} carry an approximation note:")
        for result in noted:
            print(f"  {result.name}")

    if skipped:
        print(f"\n{len(skipped)} could not be represented "
              "(deliberately degenerate shapes -- see the module docstring):")
        for result in skipped:
            print(f"  {result.name:<34} {result.skipped}")
    return 0


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()


if __name__ == "__main__":
    raise SystemExit(main())
