"""Reference vehicles built parametrically.

These mirror the definitions in ``vehicles/`` so the parametric model can be
checked against the mass properties the previous pipeline already produced.
Agreement between two independent descriptions of the same rocket is what says
the section stack is describing the shape correctly.
"""

from __future__ import annotations

from parametric.components import FinSet, Motor, PointMass, Stack, Tank
from parametric.model import AIRCRAFT, ROCKET, VehicleModel
from parametric.xsec import NoseProfile


def empty_vehicle(name: str = "Untitled Vehicle",
                  vehicle_class: str = ROCKET) -> VehicleModel:
    """Nothing but a root, to build a vehicle from scratch.

    The other two builders here are reference articles: they exist so the
    parametric model can be checked against masses the previous pipeline
    already produced. This one is the opposite -- it asserts nothing, and is
    the starting point when you are designing rather than reproducing.

    An empty model is a valid model, not a degenerate one. ``total_length_m``
    and ``max_diameter_m`` already return 0.0 rather than raising, and the
    viewport frames a default box when there is no geometry, so the
    application has somewhere to stand before the first part is added.

    The class carries no geometry of its own. All it decides is which parts the
    editor offers and what it calls them -- RASAero's vocabulary for a rocket,
    OpenVSP's for an aircraft.
    """
    model = VehicleModel(name, "A", vehicle_class)
    model.description = f"Empty {vehicle_class}"
    return model


def empty_rocket(name: str = "Untitled Rocket") -> VehicleModel:
    """A blank rocket: nose cone, body tube, boattail, fins."""
    return empty_vehicle(name, ROCKET)


def empty_aircraft(name: str = "Untitled Aircraft") -> VehicleModel:
    """A blank aircraft: fuselage, wing, tail surfaces."""
    return empty_vehicle(name, AIRCRAFT)


def basic_rocket() -> VehicleModel:
    """The 1.85 m test article, built from cross-sections.

    One Stack per physical part rather than one for the whole airframe. A Stack
    carries a single material and wall thickness, which is not a limitation so
    much as a description of how the hardware exists: a fibreglass nose cone, a
    thin carbon forward tube and a thicker aluminium motor tube are three
    manufactured parts bolted together, and they have three different masses.
    Modelling them as one stack silently averaged them and came out 6% light.
    """
    model = VehicleModel("Basic Test Rocket", "A")
    model.description = "Parametric rebuild of vehicles/basic.json"

    nose = Stack("nose", wall_thickness_m=0.003)
    nose.material = "g10_fiberglass"
    nose.add_nose(NoseProfile.OGIVE, length_m=0.45, diameter_m=0.10,
                  sections=16, tip_radius_m=0.003)
    model.add(nose)

    forward = Stack("forward_tube", wall_thickness_m=0.002)
    forward.material = "cfrp_quasi_isotropic"
    forward.add_tube(length_m=0.60, diameter_m=0.10, name="forward")
    _shift(forward, 0.45)
    model.add(forward)

    motor_tube = Stack("motor_tube", wall_thickness_m=0.003)
    motor_tube.material = "aluminium_6061_t6"
    motor_tube.add_tube(length_m=0.80, diameter_m=0.10, name="motor")
    _shift(motor_tube, 1.05)
    model.add(motor_tube)

    motor_tube.add(FinSet(
        "fins", count=3, root_chord_m=0.20, tip_chord_m=0.10, span_m=0.09,
        sweep_m=0.08, thickness_m=0.004, station_m=1.63,
    ))

    motor = Motor(
        "motor", propellant_mass_kg=3.0, propellant_density_kg_m3=1800.0,
        station_m=1.05, length_m=0.80,
    )
    # A real regressive curve, not a placeholder. The values are scaled so the
    # curve's total impulse matches what 3 kg at 205 s actually delivers --
    # 6031 N.s. An arbitrary-looking curve beside a declared Isp is the classic
    # over-specification, and Motor.isp_consistency exists to catch it.
    motor.update(isp_vac=205.0, isp_sl=180.0)
    motor.set_curve([
        (0.00, 0.0), (0.20, 1674.0), (0.60, 1808.0),
        (3.20, 1540.0), (3.80, 781.0), (4.10, 0.0),
    ])
    motor_tube.add(motor)

    model.add(PointMass("avionics", 0.40, 0.70, growth_allowance=0.15))
    model.add(PointMass("recovery", 0.35, 0.55, growth_allowance=0.10))
    model.add(PointMass("nozzle", 0.30, 1.83, growth_allowance=0.10))
    return model


def biprop_testbed() -> VehicleModel:
    """A kerolox sounding rocket, sized from round numbers.

    2,500 lbf (11.1 kN) of thrust for 25 seconds, a 9-inch (229 mm) airframe,
    20 feet (6.10 m) tip to tail, RP-1 and LOX at O/F 2.3. Deliberately rough:
    the engine is a flat curve and the masses are estimates. What it exercises
    is everything a solid cannot -- two tanks feeding one engine, the mixture
    split, the CG walking as the columns settle -- so it is the vehicle to
    reach for when testing the biprop path.

    The propellant load (101.7 kg) is what the curve's impulse delivers at the
    declared 270 s vacuum Isp, split 70.9/30.8 by the mixture ratio so both
    tanks run dry together. Tanks are clipped flush, domes buried in their
    neighbouring bays, the way the clip system intends.
    """
    model = VehicleModel("Biprop Testbed", "A")
    model.description = (
        "2500 lbf kerolox testbed: 9 in x 20 ft, 25 s burn, O/F 2.3. "
        "Rough numbers on purpose; exists to exercise the tank-fed drain path."
    )

    nose = Stack("nose", wall_thickness_m=0.003)
    nose.material = "g10_fiberglass"
    nose.add_nose(NoseProfile.OGIVE, length_m=1.00, diameter_m=0.2286,
                  sections=16, tip_radius_m=0.004)
    model.add(nose)

    bay = Stack("recovery_bay", wall_thickness_m=0.0025)
    bay.material = "aluminium_6061_t6"
    bay.add_tube(length_m=0.65, diameter_m=0.2286, name="bay")
    model.add(bay)

    lox = Tank("lox_tank", diameter_m=0.2286, barrel_length_m=1.62,
               wall_thickness_m=0.0025, propellant_mass_kg=70.9,
               propellant_density_kg_m3=1140.0)
    lox.contents = "oxidizer"
    model.add(lox)

    intertank = Stack("intertank", wall_thickness_m=0.0025)
    intertank.material = "aluminium_6061_t6"
    intertank.add_tube(length_m=0.45, diameter_m=0.2286, name="intertank")
    model.add(intertank)

    fuel = Tank("fuel_tank", diameter_m=0.2286, barrel_length_m=0.95,
                wall_thickness_m=0.0025, propellant_mass_kg=30.8,
                propellant_density_kg_m3=810.0)
    fuel.contents = "fuel"
    model.add(fuel)

    skirt = Stack("aft_skirt", wall_thickness_m=0.0025)
    skirt.material = "aluminium_6061_t6"
    skirt.add_tube(length_m=1.43, diameter_m=0.2286, name="skirt")
    model.add(skirt)

    # Chain the mould line fore to aft. Flush joints bury each tank dome
    # inside its neighbouring bay, which is what keeps the silhouette a
    # straight 9-inch tube rather than necking to a dome tip at every joint.
    for part, ahead in ((bay, nose), (lox, bay), (intertank, lox),
                        (fuel, intertank), (skirt, fuel)):
        part.clip_to = ahead.path
        part.clip_locked = True
    model.apply_clips()
    tail = skirt.station_range_m()[1]

    engine = Motor("engine", station_m=tail - 0.70, length_m=0.45)
    engine.feed = "tanks"
    engine.update(isp_vac=270.0, isp_sl=240.0, mixture_ratio=2.3,
                  dry_mass=20.0, case_diameter=0.16, nozzle_length=0.25)
    engine.flat_curve(11120.0, 25.0)      # 2,500 lbf, quoted as vacuum
    skirt.add(engine)

    skirt.add(FinSet(
        "fins", count=4, root_chord_m=0.50, tip_chord_m=0.22, span_m=0.28,
        sweep_m=0.28, thickness_m=0.006, station_m=tail - 0.55,
    ))

    model.add(PointMass("recovery", 8.0, 1.20, growth_allowance=0.10))
    model.add(PointMass("avionics", 5.0, 1.48, growth_allowance=0.15))
    pressurant_station = 0.5 * sum(intertank.station_range_m())
    model.add(PointMass("pressurant", 9.0, pressurant_station,
                        growth_allowance=0.10))
    model.add(PointMass("feed_system", 6.0, tail - 1.25, growth_allowance=0.15))
    return model


def _shift(stack: Stack, station_m: float) -> None:
    """Move a stack's sections aft so it starts at the given station."""
    low = stack.station_range_m()[0]
    for section in stack.sections:
        section.set("station", section.station_m - low + station_m)
    stack.mark_dirty("shift")


def boattailed_rocket() -> VehicleModel:
    """Shows what the section model buys: a shape the old schema could not express.

    A payload bulge and a boattail are just sections. The previous definition
    format had no way to describe either -- it only had a nose profile and
    constant-diameter tubes.
    """
    model = VehicleModel("Boattail Demonstrator", "A")

    airframe = Stack("airframe", wall_thickness_m=0.003)
    airframe.add_nose(NoseProfile.VON_KARMAN, 0.55, 0.10, sections=16)
    airframe.add_tube(0.30, 0.10, name="forward")
    airframe.add_transition(0.18, 0.13, name="bulge_out")     # payload bulge
    airframe.add_tube(0.25, 0.13, name="payload")
    airframe.add_transition(0.18, 0.10, name="bulge_in")
    airframe.add_tube(0.70, 0.10, name="motor")
    airframe.add_transition(0.15, 0.075, name="boattail")     # boattail
    model.add(airframe)

    airframe.add(FinSet(
        "fins", count=4, root_chord_m=0.22, tip_chord_m=0.09, span_m=0.10,
        sweep_m=0.11, thickness_m=0.004, station_m=1.85,
    ))
    motor = Motor("motor", propellant_mass_kg=2.5, station_m=1.46, length_m=0.70)
    motor.update(isp_vac=210.0, isp_sl=185.0)
    motor.curve_from_impulse(2.5 * 210.0 * 9.80665, burn_time_s=3.4)
    airframe.add(motor)
    model.add(PointMass("avionics", 0.5, 0.95))
    return model


if __name__ == "__main__":
    for build in (basic_rocket, boattailed_rocket, biprop_testbed):
        model = build()
        print(model.summary())
        print()
        print(model.tree_text())
        problems = model.validate()
        print("\nvalidation:", "OK" if not problems else problems)
        print("=" * 70)
