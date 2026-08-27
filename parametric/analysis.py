"""Connect a parametric model to the analysis chain.

Everything downstream already exists and none of it cares how the geometry was
authored, so this module is thin by design: it converts a ``VehicleModel`` into
what each consumer wants and hands it over.

    VehicleModel --> STEP parts  --> massprops   --> mass, CG, inertia
                 --> canonical   --> RASAero     --> aero coefficients
                 --> simulation  --> 6-DOF       --> trajectory, dispersion

The parametric model is a strictly better source for all three than the schema
it replaces: the mass solver gets per-part solids, the aero export gets a real
section profile rather than a nose type and a tube, and the simulator gets an
inertia tensor from meshed geometry instead of a slender-rod estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from parametric.components import FinSet, Motor, PointMass, Stack, Tank, Wing
from parametric.loft import LoftCache, export_step
from parametric.model import VehicleModel

IN_PER_M = 39.3700787401575
LBM_PER_KG = 2.2046226218487757
KG_PER_LBM = 1.0 / LBM_PER_KG
M_PER_IN = 1.0 / IN_PER_M
KGM2_PER_LBMIN2 = KG_PER_LBM * M_PER_IN * M_PER_IN


@dataclass
class SolvedMass:
    """Meshed mass properties, in SI, about the vehicle axes."""

    mass_kg: float
    cg_station_m: float
    inertia_kg_m2: np.ndarray
    per_component_kg: dict[str, float]

    def summary(self) -> str:
        principal = np.diag(self.inertia_kg_m2)
        return (
            f"  Mass        {self.mass_kg:.3f} kg\n"
            f"  CG station  {self.cg_station_m:.4f} m from the nose tip\n"
            f"  Inertia     roll {principal[2]:.4g}   "
            f"pitch {principal[0]:.4g} kg·m²"
        )


# ----------------------------------------------------------------------
# Mass properties
# ----------------------------------------------------------------------


def _override_scale(component, meshed_mass_kg: float, parts: int) -> float:
    """Factor taking a meshed mass to the measured one, 1.0 when not overridden.

    Applied per part so a fin set of four shares its measured mass evenly,
    which is what a set means -- they are the same component made four times.
    """
    override = getattr(component, "mass_override_kg", None)
    if override is None or meshed_mass_kg <= 0 or parts <= 0:
        return 1.0
    return (float(override) / parts) / meshed_mass_kg


def solve_mass(
    model: VehicleModel,
    work_dir: str | Path,
    cache: LoftCache | None = None,
    mesh_size_factor: float = 0.2,
) -> SolvedMass:
    """Mesh every solid and aggregate mass properties over the tree.

    Every part with geometry is meshed: stacks, fin sets, tanks and wings.
    Tanks and wings used to be skipped -- the filter was a tuple of the
    kinds that existed when it was written -- so solving a biprop dropped
    both tanks, and the trajectory then flew on the solved mass without them.

    Declared masses are added analytically, since they have no geometry to
    mesh: point masses, an engine's hardware and protuberances, each at its
    station with the parallel-axis term carried to the vehicle CG.
    """
    from massprops.mesh.mesher import generate_watertight_mesh
    from massprops.solver import mass_properties

    from parametric.materials import get_material

    work_dir = Path(work_dir)
    cache = cache or LoftCache()
    written = export_step(model, work_dir, cache)

    per_component: dict[str, float] = {}
    total_mass = 0.0
    moment = np.zeros(3)
    contributions: list[tuple[float, np.ndarray, np.ndarray]] = []

    by_stem = {path.stem: path for path in written}
    for component in model.walk():
        if not isinstance(component, (Stack, FinSet, Tank, Wing)):
            continue
        density = get_material(component.material).density_lbm_in3
        # One solid per stack or tank; fins and wing panels are one solid
        # each, keyed the way the loft cache keys them.
        keys = [component.path] if isinstance(component, (Stack, Tank)) else [
            f"{component.path}#{i}" for i in range(component.count)
        ]
        for key in keys:
            stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in key).strip("_")
            path = by_stem.get(stem)
            if path is None:
                continue
            vertices, faces = generate_watertight_mesh(
                path, mesh_size_factor=mesh_size_factor
            )
            solved = mass_properties(vertices, faces, density=density)
            mass_kg = float(solved.mass) * KG_PER_LBM
            centroid_m = np.asarray(solved.centroid, dtype=float) * M_PER_IN
            inertia = np.asarray(solved.inertia, dtype=float) * KGM2_PER_LBMIN2

            # The mesh gives the *shape* -- centroid and the distribution of
            # inertia -- and the closed form gives the amount. A shell's
            # volume is the difference of two chorded surfaces, so the
            # mesh's chording error is amplified by R/2t on the way through:
            # 9% on a 1.5 mm wall at the default mesh size, where the
            # analytic shell integral is exact. A wing's lofted solid is a
            # billet, 25x its skin model. Scaling to the analytic volume
            # fixes both at no mesh cost.
            analytic_kg = (
                component.volume_m3()
                * get_material(component.material).density_kg_m3 / len(keys)
            )
            if mass_kg > 0.0 and analytic_kg > 0.0:
                inertia = inertia * (analytic_kg / mass_kg)
                mass_kg = analytic_kg

            # A measured mass wins over both. Solving used to discard the
            # override entirely, so a weighed part reverted to volume x
            # density the moment mass properties were run, which is
            # precisely when it matters.
            scale = _override_scale(component, mass_kg, len(keys))
            if scale != 1.0:
                mass_kg *= scale
                inertia = inertia * scale

            per_component[key] = mass_kg
            total_mass += mass_kg
            moment += mass_kg * centroid_m
            contributions.append((mass_kg, centroid_m, inertia))

    declared = [
        (point.name, point.mass_kg(),
         np.array([point.get("radial_offset"), 0.0, point.get("station")]))
        for point in model.point_masses
    ] + [
        (motor.name, motor.mass_kg(),
         np.array([0.0, 0.0, motor.centroid_station_m]))
        for motor in model.motors
    ] + [
        (item.name, item.mass_kg(), np.array([0.0, 0.0, item.get("station")]))
        for item in model.protuberances
    ]
    for name, mass_kg, position in declared:
        if mass_kg <= 0:
            continue
        per_component[name] = mass_kg
        total_mass += mass_kg
        moment += mass_kg * position
        contributions.append((mass_kg, position, np.zeros((3, 3))))

    if total_mass <= 0:
        raise ValueError("Vehicle has no mass.")

    cg = moment / total_mass
    inertia_total = np.zeros((3, 3))
    for mass_kg, centroid, inertia in contributions:
        offset = centroid - cg
        inertia_total += inertia + mass_kg * (
            np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
        )

    return SolvedMass(
        mass_kg=total_mass,
        cg_station_m=float(cg[2]),
        inertia_kg_m2=inertia_total,
        per_component_kg=per_component,
    )


# ----------------------------------------------------------------------
# Aerodynamics
# ----------------------------------------------------------------------


#: Fractional diameter change below which two runs are the same tube.
_SAME_DIAMETER = 0.01


def _piece_profile(piece) -> tuple[np.ndarray, np.ndarray]:
    """Exact (station, radius) points for one outer-mould-line component.

    Exact, not sampled: a Stack's cross-sections *are* its profile, so reading
    them puts every breakpoint on a real station instead of wherever a uniform
    resampling happened to land.
    """
    from parametric.components import Stack, Tank

    if isinstance(piece, Stack):
        sections = piece.sorted_sections()
        if len(sections) < 2:
            return np.array([]), np.array([])
        stations = np.array([s.station_m for s in sections], dtype=float)
        radii = np.array([piece.radius_at(float(z)) for z in stations])
        return stations, radii

    if isinstance(piece, Tank):
        # A tank's domes are not in RASAero's vocabulary, and on a clipped
        # vehicle they are buried inside the neighbouring part anyway: what
        # the flow sees is a cylinder of the tank's diameter over its extent.
        low, high = piece.station_range_m()
        radius = 0.5 * piece.get("diameter")
        return np.array([low, high]), np.array([radius, radius])

    return np.array([]), np.array([])


def _nose_segment(stations, radii, name: str, declared: bool):
    """Build the nose segment from its own points, and return what is left."""
    from parametric.canonical import CanonicalSegment, _best_nose_shape

    peak = float(np.max(radii))
    at_full = np.flatnonzero(radii >= peak * 0.995)
    end_index = int(at_full[0]) if len(at_full) else len(radii) - 1
    if end_index < 1:
        end_index = len(radii) - 1

    z = stations[: end_index + 1]
    r = radii[: end_index + 1]
    length = float(z[-1] - z[0])
    shape = _best_nose_shape(z - z[0], r, length, float(r[-1]))

    segment = CanonicalSegment(
        kind="nose",
        start_m=float(z[0]),
        length_m=length,
        front_diameter_m=0.0,
        rear_diameter_m=float(2.0 * r[-1]),
        nose_shape=shape,
        sources=[f"{name} ({'declared' if declared else 'inferred'})"],
    )
    return segment, stations[end_index:], radii[end_index:]


def _segments_for_piece(piece, is_forwardmost: bool) -> list:
    """Canonical segments for one component, from its measured geometry."""
    from parametric.roles import AeroRole
    from parametric.canonical import CanonicalSegment, _segment_afterbody

    stations, radii = _piece_profile(piece)
    if len(stations) < 2:
        return []

    role = getattr(piece, "aero_role", AeroRole.AUTO)
    peak = float(np.max(radii))
    if peak <= 0:
        return []

    # A declared role is an instruction, not a hint. Everything else is
    # inferred from the component's own shape -- which is reliable precisely
    # because it is asked one part at a time rather than of a whole silhouette.
    is_nose = role is AeroRole.NOSE or (
        role is AeroRole.AUTO and is_forwardmost and radii[0] < 0.2 * peak
    )

    if role is AeroRole.BODY:
        mean = float(np.mean(radii)) * 2.0
        return [CanonicalSegment(
            kind="tube", start_m=float(stations[0]),
            length_m=float(stations[-1] - stations[0]),
            front_diameter_m=mean, rear_diameter_m=mean,
            sources=[f"{piece.name} (declared)"],
        )]

    if role is AeroRole.TRANSITION:
        front, rear = float(2 * radii[0]), float(2 * radii[-1])
        return [CanonicalSegment(
            kind="boattail" if rear < front else "transition",
            start_m=float(stations[0]),
            length_m=float(stations[-1] - stations[0]),
            front_diameter_m=front, rear_diameter_m=rear,
            sources=[f"{piece.name} (declared)"],
        )]

    segments = []
    if is_nose:
        nose, stations, radii = _nose_segment(
            stations, radii, piece.name, role is AeroRole.NOSE
        )
        segments.append(nose)

    if len(stations) >= 2:
        rest = _segment_afterbody(stations, radii, _SAME_DIAMETER, peak)
        for segment in rest:
            segment.sources = [f"{piece.name} (measured)"]
        segments.extend(rest)
    return segments


def _merge_tubes(segments: list) -> list:
    """Join touching or overlapping tubes of the same diameter.

    The flow sees no joint either way. Overlap is not an anomaly: on a
    clipped chain each tank's domes are buried inside the neighbouring bays,
    so the tank's tube and the bay's tube claim the same stations, and the
    mould line is their union. Requiring exact contiguity here made every
    clipped-tank vehicle collapse to its first two parts -- the writer only
    emits one body tube, and everything after the first un-merged joint was
    silently dropped.

    A real gap is still a real feature: two separate bodies with daylight
    between them are never joined, because closing a gap silently would
    invent geometry.
    """
    merged: list = []
    for segment in segments:
        if merged:
            previous = merged[-1]
            same_kind = previous.kind == "tube" and segment.kind == "tube"
            joined = segment.start_m <= previous.end_m + 1e-6
            same_size = (
                abs(previous.rear_diameter_m - segment.rear_diameter_m)
                <= _SAME_DIAMETER * max(segment.rear_diameter_m, 1e-9)
            )
            if same_kind and joined and same_size:
                previous.length_m = (
                    max(previous.end_m, segment.end_m) - previous.start_m
                )
                previous.sources.extend(segment.sources)
                continue
        merged.append(segment)
    return merged


def canonical_from_components(model: VehicleModel):
    """Read RASAero's parts straight off the model, or ``None`` if it cannot.

    The alternative -- sampling the outer mould line and least-squares fitting
    shapes back to it -- throws away what is already known exactly. On the
    reference rocket the fit put the nose at 0.442 m when the nose component is
    0.430 m long, because a resampled silhouette has no idea where the parts
    are and the fitter is free to choose a breakpoint that fits the curve
    slightly better. Reading the parts cannot make that mistake.

    Returns ``None`` when there is nothing to read -- an imported mesh, for
    instance -- and the caller falls back to fitting.
    """
    from parametric.components import Stack, Tank
    from parametric.roles import AeroRole
    from parametric.canonical import CanonicalModel

    pieces = [
        component for component in model.walk()
        if component is not model.root
        and isinstance(component, (Stack, Tank))
        and getattr(component, "aero_role", AeroRole.AUTO) is not AeroRole.INTERNAL
        and component.is_external
    ]
    if not pieces:
        return None

    pieces.sort(key=lambda c: c.station_range_m()[0])
    segments: list = []
    for piece in pieces:
        segments.extend(_segments_for_piece(piece, piece is pieces[0]))

    segments = _merge_tubes(segments)
    if not segments or not any(s.kind == "nose" for s in segments):
        # RASAero's vocabulary starts at a nose cone; without one there is
        # nothing to write, so hand back to the fitter rather than invent one.
        return None
    return CanonicalModel(name=model.name, segments=segments)


def to_canonical(model: VehicleModel, cg_station_m: float | None = None,
                 measured: bool = True):
    """Express the model in RASAero's part vocabulary.

    Args:
        measured: Read the parts directly, which is exact. Set False to force
            the silhouette fit -- the path an imported solid has to take.
    """
    from parametric.canonical import CanonicalFin, _measure_residual

    stations, radii = model.silhouette(600)
    positive = radii > 0
    stations, radii = stations[positive], radii[positive]

    from parametric.canonical import canonical_from_profile

    canonical = canonical_from_components(model) if measured else None
    if canonical is None:
        canonical = canonical_from_profile(stations, radii, name=model.name)
    elif len(stations) >= 2:
        # Measured parts are exact by construction, but the residual against
        # the real silhouette still says how much RASAero's four shapes had to
        # give up -- a dome flattened into a tube, a curve straightened.
        _measure_residual(canonical, stations, radii)

    # The fin sets the air sees, in the order extract_geometry keeps them,
    # so the table's fin parts pair with the same sets everywhere.
    canonical.fins = [
        CanonicalFin(
            count=fins.count,
            root_chord_m=fins.get("root_chord"),
            tip_chord_m=fins.get("tip_chord"),
            span_m=fins.get("span"),
            sweep_m=fins.get("sweep"),
            thickness_m=fins.get("thickness"),
            station_m=fins.get("station"),
        )
        for fins in model.fin_sets if fins.is_external
    ]
    mass = model.mass_summary()
    # RASAero's project carries the *launch* weight, so the CG beside it is
    # the loaded one. ``cg_station_m`` is a dry CG -- a meshed solve's --
    # and used to be written out as it came, a wet weight beside a burnout
    # CG: 1.3 calibres forward of the truth on the basic rocket, 2.1 aft on
    # the biprop. The in-process table never reads it; an exported project
    # and a RASAero flight run do.
    canonical.cg_from_nose_m = loaded_cg_station_m(model, cg_station_m)
    canonical.wet_mass_kg = mass.wet_mass_kg
    canonical.nozzle_exit_diameter_m = model.nozzle_exit_diameter_m()
    return canonical


#: Which RASAero primitive each parametric protuberance shape becomes.
#:
#: RASAero's vocabulary is much coarser than Hoerner's: a launch lug, a rail
#: guide, a launch shoe, two streamlined bodies and an inclined plate. Anything
#: bluff that is not one of the named fittings goes to a plate normal to the
#: flow, which is the only primitive RASAero has for a blunt obstruction.
#:
#: The mapping is lossy and it changes numbers -- a round cylinder scored 1.17
#: under Hoerner and now takes RASAero's plate value instead. That is the
#: intended consequence of asking for RASAero's model rather than a different
#: one: the alternative is what this replaces, where protuberances reached the
#: CDX1 writer and were dropped, contributing exactly zero. On the flights
#: RASAero ships as validated examples, protuberances carry 10 to 37 percent of
#: total drag, so dropping them is far the larger error.
_PROTUBERANCE_KIND = {
    "launch_lug": "lug",
    "rail_button": "guide",
    "streamlined_fairing": "streamlined",
}


def _rasaero_protuberances(model: VehicleModel) -> dict:
    """Fold the model's protuberances into RASAero's fixed set of fittings.

    Areas add: RASAero carries one entry of each kind, so ``count`` and
    multiple components of the same kind are summed into it.

    Lug and rail-guide *lengths* are not carried by the parametric component,
    which knows only a frontal area. They are inferred as equal to the derived
    diameter -- the same as-tall-as-it-is-wide assumption ``roles`` already
    makes when it estimates boundary-layer immersion, kept identical here so
    the two do not quietly disagree.
    """
    import math

    totals = {"lug": 0.0, "guide": 0.0, "streamlined": 0.0, "plate": 0.0}
    for item in model.protuberances:
        if not item.is_external:
            continue
        spec = item.to_spec()
        shape = str(getattr(spec.shape, "value", spec.shape))
        kind = _PROTUBERANCE_KIND.get(shape, "plate")
        totals[kind] += max(spec.frontal_area_m2, 0.0) * max(spec.count, 0)

    def diameter(area_m2: float) -> float:
        return math.sqrt(4.0 * area_m2 / math.pi) if area_m2 > 0.0 else 0.0

    lug_d = diameter(totals["lug"])
    guide_d = diameter(totals["guide"])
    return {
        "launch_lug_diameter_m": lug_d,
        "launch_lug_length_m": lug_d,
        "rail_guide_diameter_m": guide_d,
        "rail_guide_height_m": guide_d,
        "launch_shoe_area_m2": 0.0,
        "streamlined_with_base_area_m2": totals["streamlined"],
        "streamlined_no_base_area_m2": 0.0,
        "plate_area_m2": totals["plate"],
        "plate_angle_deg": 90.0,
    }


def write_cdx1(model: VehicleModel, output_path: str | Path,
               cg_station_m: float | None = None,
               metadata: dict | None = None):
    """Emit a RASAero II project for the parametric model.

    Args:
        metadata: Extra project fields -- surface finish, launch altitude --
            so the conditions asked for here are the conditions RASAero runs
            at, rather than its defaults.
    """
    from parametric.canonical import to_rasaero_model
    from step_to_rasaero.rasaero_writer import write_rasaero_cdx1

    canonical = to_canonical(model, cg_station_m)
    payload = to_rasaero_model(canonical, metadata)
    payload["protuberances"] = _rasaero_protuberances(model)
    path = write_rasaero_cdx1(payload, output_path)
    return path, canonical


# ----------------------------------------------------------------------
# Trajectory
# ----------------------------------------------------------------------


def _tank_drain_shares(motor, tanks) -> list[float]:
    """Fraction of the engine's mass flow each tank supplies.

    A declared mixture ratio splits the flow O/F-wise between the tanks
    labelled oxidizer and the rest (several of a kind split in proportion to
    what they hold). Without one -- or without both labels present -- tanks
    drain in proportion to their loads, which is the correctly sized biprop:
    everything runs dry together.
    """
    loads = [tank.get("propellant_mass") for tank in tanks]
    total = sum(loads)
    if total <= 0:
        return [0.0] * len(tanks)
    contents = [getattr(tank, "contents", "fuel") for tank in tanks]
    fuel = sum(l for l, c in zip(loads, contents) if c != "oxidizer")
    ox = sum(l for l, c in zip(loads, contents) if c == "oxidizer")
    ratio = motor.get("mixture_ratio")
    if ratio > 0 and fuel > 0 and ox > 0:
        return [
            (ratio / (1.0 + ratio)) * (l / ox) if c == "oxidizer"
            else (1.0 / (1.0 + ratio)) * (l / fuel)
            for l, c in zip(loads, contents)
        ]
    return [l / total for l in loads]


def build_simulation(
    model: VehicleModel,
    solved: SolvedMass | None = None,
    aero_csv: str | Path | None = None,
):
    """Configure a RocketSimulation from a parametric model.

    The frame mapping is the same one the previous bridge established: the
    model's axis is +Z aft with the nose at the origin, the simulator's is +Y
    forward, so the inertia tensor is *rotated* rather than reordered.
    """
    from trajectory import simulation as tm
    from trajectory.frames import (
        DEF_TO_SIM,
        inertia_to_body,
        station_to_body,
    )

    sim = tm.RocketSimulation()
    mass = model.mass_summary()

    if solved is not None:
        dry_mass = solved.mass_kg
        cg_station = solved.cg_station_m
        inertia = inertia_to_body(solved.inertia_kg_m2)
    else:
        dry_mass = mass.dry_mass_kg
        cg_station = mass.cg_station_m
        length = model.total_length_m
        transverse = dry_mass * length ** 2 / 12.0
        # Half the shell value: a real vehicle's engine, avionics and
        # recovery sit on the centreline, and against a meshed biprop the
        # full m R^2 was 74% high where this is within a third.
        roll = 0.5 * dry_mass * (0.5 * model.max_diameter_m) ** 2
        inertia = inertia_to_body(np.diag([transverse, transverse, roll]))

    sim.mass_props = tm.MassProperties(
        dry_mass=dry_mass,
        prop_mass=mass.propellant_mass_kg,
        cg_dry=station_to_body(cg_station),
        i_tensor_dry=inertia,
    )

    motors = model.motors
    if motors:
        motor = motors[0]
        tanks = [t for t in model.tanks if t.get("propellant_mass") > 0.0]
        if getattr(motor, "feed", "grain") == "tanks" and tanks:
            # A liquid: the propellant lives in the tanks, so each tank
            # becomes its own settling column, emptied at the engine's
            # integrated mass flow and split by mixture ratio. Lumping the
            # whole load at the engine put half the wet mass at the tail --
            # the sim flew a different vehicle than the one the status bar
            # trimmed.
            from trajectory.vehicle.mass_properties import LIQUID, PropellantLoad

            shares = _tank_drain_shares(motor, tanks)
            sim.mass_props.set_propellant_loads([
                PropellantLoad(
                    mass_kg=tank.get("propellant_mass"),
                    forward=station_to_body(tank.station_range_m()[0]),
                    aft=station_to_body(tank.station_range_m()[1]),
                    burn_geometry=LIQUID,
                    radius_m=max(
                        tank.radius_m - tank.get("wall_thickness"), 0.0
                    ),
                    drain_share=share,
                )
                for tank, share in zip(tanks, shares)
            ])
        elif tanks:
            # A grain-fed motor flying alongside loaded tanks: the grain
            # drains, and the tank loads ride along at their own stations --
            # correctly placed dead mass, rather than being teleported into
            # the motor column as they were.
            from trajectory.vehicle.mass_properties import LIQUID, PropellantLoad

            forward_station, aft_station = motor.grain_range_m()
            loads = [PropellantLoad(
                mass_kg=motor.get("propellant_mass"),
                forward=station_to_body(forward_station),
                aft=station_to_body(aft_station),
                burn_geometry=motor.burn_geometry,
                radius_m=motor.propellant_radius_m,
                drain_share=1.0,
            )]
            loads += [
                PropellantLoad(
                    mass_kg=tank.get("propellant_mass"),
                    forward=station_to_body(tank.station_range_m()[0]),
                    aft=station_to_body(tank.station_range_m()[1]),
                    burn_geometry=LIQUID,
                    radius_m=max(
                        tank.radius_m - tank.get("wall_thickness"), 0.0
                    ),
                    drain_share=0.0,
                )
                for tank in tanks
            ]
            sim.mass_props.set_propellant_loads(loads)
        else:
            forward_station, aft_station = motor.grain_range_m()
            sim.mass_props.set_propellant_geometry(
                station_to_body(forward_station),
                station_to_body(aft_station),
                burn_geometry=motor.burn_geometry,
                radius_m=motor.propellant_radius_m,
            )
        # Thrust acts at the aft extreme -- the tail's station, which is the
        # length only when the nose sits at the origin.
        sim.thrust_position_body_m = station_to_body(model.station_range_m()[1])

        # The motor's own curve is the source of truth. Falling back to a file
        # -- or to nothing -- is how a 7 kg vehicle silently kept the
        # simulator's default 20 kN engine and flew at Mach 3.5.
        if len(motor.curve) >= 2:
            sim.engine = motor.to_engine()
        elif motor.propulsion_file and Path(motor.propulsion_file).exists():
            sim.import_saved_propulsion(motor.propulsion_file)
        else:
            raise ValueError(
                f"{motor.name} has no thrust curve. Add points in the motor "
                f"editor, or load a propulsion model."
            )

    sim.reference_area = model.reference_area_m2
    sim.thrust_position_body_m = station_to_body(model.station_range_m()[1])

    if aero_csv is not None:
        from trajectory.vehicle.aero_database import AeroDatabase

        sim.set_aero_database(AeroDatabase.from_csv(
            aero_csv, reference_length_m=model.total_length_m
        ))

    sim.vehicle_definition = model
    return sim


#: Feet per metre, exactly.
_FT_PER_M = 1.0 / 0.3048


def mach_alt_profile(result, pad_altitude_m: float = 0.0,
                     mach_step: float = 0.02) -> list[tuple[float, float]]:
    """(Mach, altitude-in-feet) samples along a flown ascent.

    The drag table is indexed by Mach alone, so the only way altitude can
    enter it is through a profile like this: the altitude the vehicle
    actually held at each Mach on the way up. Feeding these samples back
    into the table build and re-flying is what couples the aerodynamics to
    the trajectory instead of to sea level -- the same fixed-point loop the
    validation scoreboard runs, where it is worth up to two thousand feet
    of apogee on a high flight.

    Only the ascent is sampled, and each Mach is claimed by the crossing
    that carries the dynamic pressure. An ascent crosses most of its Machs
    twice -- accelerating through them under thrust, low and fast through
    thick air, then decelerating back through them in the coast, high and
    thin. The table has one altitude per Mach, so it can hold only one of
    those. It must be the high-q one: that is where the drag impulse lives.
    Letting the coast claim them instead evaluates Reynolds number in
    near-vacuum, where the laminar skin-friction correlation blows up, and
    the resulting table is built for air the vehicle barely felt -- on a
    high flight the fixed-point loop then oscillates between a draggy table
    that predicts low and a clean table that predicts high, rather than
    converging.

    The Machs are a fixed ladder, ``mach_step`` apart, and each rung is
    located by interpolating the crossing between the two integrator states
    that straddle it. That matters more than it looks. Sampling the states
    themselves -- every so many metres of altitude, or every step -- lets a
    fast vehicle climb clean past a rung between samples, and a rung the
    burn never claims falls to the coast by default. On a Mach-4 kerolox
    stage that put adjacent rungs at 3,000 and 250,000 feet, and which rungs
    fell through changed with every pass, so the loop never settled.
    Interpolating the crossing guarantees the burn claims every Mach it
    flew, whatever the time step.

    Past apogee the descent revisits Mach numbers the ascent already
    claimed. Altitudes are returned in feet, above sea level, because that
    is the unit the engine's Mach/Alt grid speaks.

    ``pad_altitude_m`` is for a result whose altitudes are above the pad. A
    flight run with ``pad_position_m`` set -- every flight ``fly_model``
    makes -- is integrated above sea level already and needs no shift.
    """
    from trajectory.environment.atmosphere import Atmosphere

    states = np.asarray(result.y, dtype=float).T
    if states.ndim != 2 or states.shape[1] < 6 or len(states) < 2:
        return []

    atmosphere = Atmosphere()
    apogee = int(np.argmax(states[:, 1]))
    ascent = states[: apogee + 1]
    if len(ascent) < 2:
        return []

    msl = ascent[:, 1] + pad_altitude_m
    speed = np.linalg.norm(ascent[:, 3:6], axis=1)
    mach = np.zeros(len(ascent))
    q = np.zeros(len(ascent))
    for i, (altitude, v) in enumerate(zip(msl, speed)):
        density, _, _, sound = atmosphere.get_conditions(float(altitude))
        mach[i] = v / sound if sound > 0 else 0.0
        q[i] = 0.5 * float(density) * v * v

    #: Rung index -> (q, altitude_ft) for the highest-q crossing seen.
    best: dict[int, tuple[float, float]] = {}
    for i in range(len(ascent) - 1):
        m0, m1 = float(mach[i]), float(mach[i + 1])
        if m0 == m1:
            continue
        lo, hi = min(m0, m1), max(m0, m1)
        first = max(int(np.ceil(lo / mach_step - 1e-9)), 1)
        last = int(np.floor(hi / mach_step + 1e-9))
        for k in range(first, last + 1):
            f = (k * mach_step - m0) / (m1 - m0)
            crossing_q = float(q[i] + f * (q[i + 1] - q[i]))
            if k not in best or crossing_q > best[k][0]:
                altitude = float(msl[i] + f * (msl[i + 1] - msl[i]))
                best[k] = (crossing_q, altitude * _FT_PER_M)
    return sorted((k * mach_step, alt) for k, (_, alt) in best.items())


def centre_of_pressure(model: VehicleModel, mach: float = 0.3,
                       alpha_deg: float = 4.0,
                       method: str | None = None) -> float:
    """Centre of pressure as a station from the nose tip, in metres.

    Uses the in-process RASAero engine. This function -- not the coefficient
    table -- is what feeds the static-margin readout, the mass dialog and the
    sweep metrics, so it has to agree with the table or the status bar quietly
    contradicts the analysis sitting next to it.

    It used to call a Barrowman routine whose centre of pressure had no Mach
    dependence whatsoever: the same station at Mach 0.3 and Mach 3. A margin
    quoted from it was the only margin it could ever report, which is
    unhelpful precisely through the transonic band where the centre of
    pressure actually moves and vehicles actually go unstable.

    ``method`` is accepted and ignored except for validation; there is one
    engine now. It stays in the signature because callers pass it through.
    """
    if method not in (None, "rasaero", "rasaero-native", "builtin"):
        raise ValueError(
            f"unknown aero method {method!r}; centre of pressure is computed "
            "in process and cannot be taken from the RASAero application"
        )

    from aeroengine.adapters import IN_PER_M
    from aeroengine.cdx1 import load as load_cdx1
    from aeroengine.solver import Engine

    import tempfile
    from pathlib import Path

    # Same route as run_analysis: through the CDX1 writer, so the centre of
    # pressure on the status bar comes from the same geometry as the table
    # -- and at the same alpha, 4 degrees, RASAero's "0 to 4 deg" pair that
    # the flight consumes; at 2 degrees the status bar read 17% more margin
    # than the results panel beside it. The engine's station is from the
    # nose tip; the model's is from its origin.
    with tempfile.TemporaryDirectory() as tmp:
        path, canonical = write_cdx1(model, Path(tmp) / "cp.CDX1", None, None)
        design = load_cdx1(path)
    return Engine(design).solve(mach, alpha_deg).cp / IN_PER_M + canonical.nose_start_m


def loaded_cg_station_m(model: VehicleModel,
                        dry_cg_station_m: float | None = None,
                        dry_mass_kg: float | None = None) -> float:
    """CG with propellant aboard, optionally from a solved dry CG.

    The meshed-CAD solver reports the *structure*, so its CG is the burnout
    one. Callers that have it should pass it here rather than use it directly,
    or they silently judge stability at the wrong moment of the flight.
    """
    summary = model.mass_summary()
    dry_cg = summary.cg_station_m if dry_cg_station_m is None else dry_cg_station_m
    dry_mass = summary.dry_mass_kg if dry_mass_kg is None else dry_mass_kg

    total = dry_mass + summary.propellant_mass_kg
    if total <= 0:
        return dry_cg
    return (
        dry_cg * dry_mass
        + summary.propellant_cg_station_m * summary.propellant_mass_kg
    ) / total


def static_margin(model: VehicleModel, cg_station_m: float | None = None,
                  mach: float = 0.3, alpha_deg: float = 4.0,
                  loaded: bool = True, method: str | None = None) -> float:
    """Static margin in calibres.

    Delegates to ``parametric.aero`` rather than carrying its own copy of
    Barrowman. There were three copies at one point and two of them squared
    ``2*span/diameter`` where the fin term wants ``span/diameter`` -- a factor
    of four on the fin contribution, which reported a marginal vehicle as
    having four calibres of margin.
    """
    diameter = model.max_diameter_m
    if diameter <= 0:
        return 0.0
    if cg_station_m is None:
        summary = model.mass_summary()
        # Default to the loaded vehicle. A rocket goes unstable off the rail or
        # not at all, and on the rail the tanks are full.
        cg = summary.wet_cg_station_m if loaded else summary.cg_station_m
    else:
        cg = cg_station_m
    return (centre_of_pressure(model, mach, alpha_deg, method) - cg) / diameter
