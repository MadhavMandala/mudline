"""Aerodynamic analysis.

Two methods, and they are the same method: RASAero II's.

* ``"rasaero-app"`` drives the actual RASAero II application, writes the
  vehicle out as a ``.CDX1``, and imports the table it exports.
* ``"rasaero"`` runs ``aeroengine``, a reimplementation of RASAero's solver
  that executes in this process. No install, no GUI automation, no half
  minute of the desktop being unusable.

The second is not an approximation of the first. It is validated against it
term by term -- friction, form, wave, base, each fin contribution,
protuberance, CN, CP -- across 130 vehicles and 7.1 million individual
comparisons, with zero disagreements outside RASAero's own printed precision
of +/-0.0005. Both methods feed the same ``.CDX1`` writer, so they solve
byte-identical geometry.

A Barrowman sweep used to live here as a third, faster option. It was removed:
it disagreed with both RASAero paths in ways that were hard to attribute, its
centre of pressure had no Mach dependence at all (constant through the
transonic band, which is exactly where a fin-stabilised vehicle goes
unstable), and having a third answer on screen made every disagreement a
three-way argument. ``aeroengine`` is fast enough to have replaced its reason
for existing.

What you still get from RASAero itself
--------------------------------------
The application remains worth running as an independent check, and as the
thing the reimplementation is measured against. It is the reference; the
built-in engine is the copy that has been proven to match it.

Accuracy against reality, as opposed to against RASAero, is a separate
question: this is a component-buildup method of the Barrowman/Hoerner
tradition, good to roughly 10% on drag for a slender axisymmetric vehicle at
low angle of attack, degrading at high alpha, on blunt bodies, and wherever a
plume interacts with the base. Reproducing RASAero exactly does not make
RASAero right.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parametric.components import FinSet, Stack
from parametric.model import VehicleModel

GAMMA = 1.4


@dataclass
class AeroSettings:
    """What to sweep and under what conditions."""

    mach_min: float = 0.05
    mach_max: float = 5.0
    mach_points: int = 40
    alpha_max_deg: float = 16.0
    alpha_points: int = 9
    #: Altitude at which Reynolds number is evaluated. Skin friction varies
    #: with it, and a table built at sea level overstates friction high up.
    altitude_m: float = 3000.0
    #: Surface roughness height. 20 microns is a good painted finish; 60 is a
    #: rough one. It sets a floor on skin friction that Reynolds cannot beat.
    roughness_m: float = 20e-6
    #: Carry the plume-filled base drag as a second column, which the flight
    #: uses while the motor burns before switching to the power-off column
    #: at burnout. Off keeps a single power-off column for the whole flight
    #: -- the old behaviour, kept for an A/B. Until the nozzle exit diameter
    #: reached the engine the two columns were identical whatever this said.
    power_on_base: bool = True
    #: Supersonic base-drag law: "rasaero" is the faithful port, "corrected"
    #: the provisional boattail replacement. See
    #: aeroengine.basedrag.base_drag_supersonic_corrected before trusting the
    #: second.
    boattail_model: str = "rasaero"
    #: Which engine fills the table:
    #:
    #: * ``"rasaero"``     - the reimplementation in ``aeroengine``, running
    #:                       in this process. The default, because it needs
    #:                       nothing installed and is validated against the
    #:                       application below.
    #: * ``"rasaero-app"`` - the actual RASAero II application. The reference.
    #:                       Requires it to be installed, and takes over the
    #:                       desktop for about half a minute while it runs.
    #:
    #: ``None`` means ``"rasaero"``. Read by ``run_analysis`` itself so every
    #: caller honours it -- the choice used to be made in the GUI, which meant
    #: a design sweep silently ignored it.
    method: str | None = None

    def mach_values(self) -> np.ndarray:
        return np.linspace(self.mach_min, self.mach_max, self.mach_points)

    def alpha_values(self) -> np.ndarray:
        return np.linspace(0.0, self.alpha_max_deg, self.alpha_points)


@dataclass
class AeroGeometry:
    """The handful of geometric quantities the aerodynamics needs."""

    length_m: float
    reference_diameter_m: float
    reference_area_m2: float
    nose_length_m: float
    nose_station_m: float
    body_wetted_area_m2: float
    fin_wetted_area_m2: float
    base_diameter_m: float
    fineness: float
    transitions: list[tuple[float, float, float]] = field(default_factory=list)
    fin_sets: list[FinSet] = field(default_factory=list)
    protuberances: list = field(default_factory=list)
    #: True when the nose came from a declared role rather than a guess.
    nose_declared: bool = False
    excluded: list[str] = field(default_factory=list)

    @property
    def base_area_m2(self) -> float:
        return float(np.pi * (0.5 * self.base_diameter_m) ** 2)


def extract_geometry(model: VehicleModel) -> AeroGeometry:
    """Measure what the solver needs from the model's outer mould line."""
    stations, radii = model.silhouette(500)
    positive = radii > 1e-9
    stations, radii = stations[positive], radii[positive]
    if len(stations) < 3:
        raise ValueError("Vehicle has no measurable body.")

    diameter = model.max_diameter_m
    length = model.total_length_m

    # Where the nose ends. A declared role wins over the silhouette guess,
    # which is the point of having roles: the guess -- "first station reaching
    # maximum radius" -- picks the wrong body entirely on a vehicle whose
    # payload bulge is wider than its nose base, and takes the centre of
    # pressure with it.
    from parametric.roles import AeroRole

    nose_declared = False
    declared_noses = [
        stack for stack in model.stacks if stack.aero_role is AeroRole.NOSE
    ]
    if declared_noses:
        nose_end = max(stack.station_range_m()[1] for stack in declared_noses)
        nose_start = min(stack.station_range_m()[0] for stack in declared_noses)
        nose_declared = True
    else:
        at_full = np.flatnonzero(radii >= radii.max() * 0.995)
        nose_end = float(stations[at_full[0]]) if len(at_full) else length * 0.2
        nose_start = float(stations[0])

    # Wetted area of a surface of revolution: 2*pi*integral(r ds).
    ds = np.sqrt(np.diff(stations) ** 2 + np.diff(radii) ** 2)
    mean_r = 0.5 * (radii[:-1] + radii[1:])
    body_wetted = float(2.0 * np.pi * np.sum(mean_r * ds))

    # Internal components are mass, not aerodynamics. This is the job OpenVSP's
    # Sets do, in the form that matters here.
    fin_sets = [fins for fins in model.fin_sets if fins.is_external]
    excluded = [c.name for c in model.walk() if not c.is_external and c.kind != "motor"]
    protuberances = [
        item.to_spec() for item in model.protuberances if item.is_external
    ]
    fin_wetted = sum(
        2.0 * fins.area_per_fin_m2 * fins.count for fins in fin_sets
    )

    # Diameter changes *aft of the nose*, for the Barrowman transition terms.
    #
    # Restricting this to the afterbody is essential. The nose's normal force
    # is already accounted for by its own CN_alpha = 2 term, so treating the
    # continuous radius growth along it as a run of transitions counts it
    # twice: on the reference vehicle that invented 38 transitions worth
    # another CN_alpha = 2 acting near the tip, which dragged the centre of
    # pressure forward far enough to report a stable rocket as unstable.
    transitions: list[tuple[float, float, float]] = []
    step = diameter * 0.02
    afterbody = stations > nose_end
    if np.any(afterbody):
        aft_stations = stations[afterbody]
        aft_radii = radii[afterbody]
        anchor = 0
        for index in range(1, len(aft_stations)):
            if abs(aft_radii[index] - aft_radii[anchor]) > step:
                transitions.append((
                    float(aft_stations[index]),
                    float(2.0 * aft_radii[anchor]),
                    float(2.0 * aft_radii[index]),
                ))
                anchor = index

    return AeroGeometry(
        length_m=length,
        reference_diameter_m=diameter,
        reference_area_m2=float(np.pi * (0.5 * diameter) ** 2),
        # These were previously the nose's END STATION and a hardcoded zero,
        # which agree with the length and the start only when the vehicle
        # begins at station 0. It usually does, which is why this went
        # unnoticed -- but a model built with any leading offset got a nose
        # fineness that was too large by the offset, and nose fineness drives
        # the transonic wave-drag peak (see wave_drag_coefficient below).
        nose_length_m=nose_end - nose_start,
        nose_station_m=nose_start,
        body_wetted_area_m2=body_wetted,
        fin_wetted_area_m2=float(fin_wetted),
        base_diameter_m=float(2.0 * radii[-1]),
        fineness=length / diameter if diameter > 0 else 10.0,
        transitions=transitions,
        fin_sets=fin_sets,
        protuberances=protuberances,
        nose_declared=nose_declared,
        excluded=excluded,
    )


# ----------------------------------------------------------------------
# Normal force and centre of pressure
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Drag
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------


def run_analysis(model: VehicleModel, settings: AeroSettings | None = None,
                 mach_alt: list[tuple[float, float]] | None = None):
    """Sweep Mach and alpha and return ``(AeroDatabase, AeroGeometry)``.

    Dispatches on ``settings.method``. The dispatch lives here rather than in
    the caller so that every entry point -- the dialog, a design sweep, a
    script -- gets the solver that was asked for.

    ``mach_alt`` is a flown (Mach, altitude-ft) profile, usually from
    :func:`parametric.analysis.mach_alt_profile`. When given, the engine
    evaluates Reynolds number along it instead of at sea level, which is what
    couples the drag table to the trajectory that will fly it.

    ``AeroGeometry`` is returned whichever solver ran, because the report and
    results panes read its attributes. It always describes the same vehicle;
    only the coefficients differ.
    """
    from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase

    settings = settings or AeroSettings()
    geometry = extract_geometry(model)

    method = settings.method or "rasaero"
    # "rasaero-native" was this engine's first name and may be sitting in a
    # saved project; "builtin" named the removed Barrowman sweep, and the
    # closest surviving behaviour is the in-process engine.
    if method in ("rasaero-native", "builtin"):
        method = "rasaero"

    if method == "rasaero-app":
        raise ValueError(
            "the 'rasaero-app' method runs the application out of process; "
            "call parametric.rasaero_run.run() for it rather than run_analysis"
        )
    if method != "rasaero":
        raise ValueError(
            f"unknown aero method {settings.method!r}; "
            "expected 'rasaero' or 'rasaero-app'"
        )

    rows = _rasaero_rows(model, settings, mach_alt)
    database = AeroDatabase(rows, reference_length_m=geometry.length_m)
    database.high_alpha = high_alpha_geometry(model, geometry)
    return database, geometry


def high_alpha_geometry(model: VehicleModel, geometry: AeroGeometry):
    """What the flight model's extension beyond the table needs: the body's
    side-projected area and its centroid, the nose length, and the fins'
    planform and centroid. Read off the silhouette and the fin sets the
    aerodynamics already uses, so the extension describes the same vehicle
    the table does.
    """
    from trajectory.vehicle.aero_model import HighAlphaGeometry

    stations, radii = model.silhouette(600)
    order = np.argsort(stations)
    s = np.asarray(stations, dtype=float)[order]
    width = 2.0 * np.asarray(radii, dtype=float)[order]
    area = float(np.trapezoid(width, s)) if len(s) > 1 else 0.0
    centroid = (
        float(np.trapezoid(width * s, s) / area) if area > 0.0
        else 0.5 * geometry.length_m
    )

    fin_area = fin_moment = 0.0
    for fins in geometry.fin_sets:
        # Half the panels' worth: broadside, a cruciform set's panels meet
        # the crossflow at every angle round the roll, and the sum of their
        # cos^2 is N/2. Counting all N put 11% too much into the flat plate.
        panels = 0.5 * fins.count * fins.area_per_fin_m2
        fin_area += panels
        fin_moment += panels * fins.planform_centroid_station_m

    return HighAlphaGeometry(
        length_m=geometry.length_m,
        diameter_m=geometry.reference_diameter_m,
        planform_area_m2=area,
        planform_centroid_m=centroid,
        nose_length_m=geometry.nose_length_m,
        fin_area_m2=fin_area,
        fin_centroid_m=fin_moment / fin_area if fin_area > 0.0 else 0.0,
    )


def _rasaero_rows(model: VehicleModel, settings: AeroSettings,
                  mach_alt: list[tuple[float, float]] | None = None):
    """Fill the table from the in-process RASAero engine.

    The engine is built once and swept, rather than rebuilt per point: its
    geometry pass is cached per design and a transonic query costs two full
    solves, so constructing it inside the loop would be roughly a hundred
    times slower for identical numbers.

    Centre of pressure comes back as an inch station from the nose tip, which
    is the same convention ``barrowman`` returns and the same one
    ``aero_csv_parser`` produces for a real RASAero export -- so nothing
    downstream has to know which solver ran.
    """
    from trajectory.vehicle.aero_database import AeroCoefficients

    import tempfile
    from pathlib import Path

    from aeroengine.adapters import IN_PER_M, surface_for_roughness
    from aeroengine.cdx1 import load as load_cdx1
    from aeroengine.solver import Engine
    from parametric import analysis

    # Go through the CDX1 writer rather than translating geometry separately.
    #
    # This looks like a detour -- write a file, read it back -- and it is the
    # single most important line in this function. The "rasaero" method hands
    # that same file to RASAero itself, so routing the native engine through it
    # guarantees the two methods model *byte-identical* geometry. A second
    # translator, however careful, is a second set of defaults to drift.
    #
    # It had already drifted. A hand-written adapter disagreed with the writer
    # on the turbulent-flow flag (worth ~20% of skin friction) and on the fin
    # thickness-break stations that set the supersonic wave-drag shape factor,
    # while every coefficient still came back looking reasonable.
    #
    # The cost is one temporary file per analysis, not per Mach point.
    with tempfile.TemporaryDirectory() as tmp:
        path, canonical = analysis.write_cdx1(
            model, Path(tmp) / "native.CDX1", None,
            {"surface": surface_for_roughness(settings.roughness_m)},
        )
        design = load_cdx1(path)
    # The engine measures stations from the nose tip; the model from its
    # origin. They coincide unless the nose has been moved.
    origin = canonical.nose_start_m

    if mach_alt:
        # The engine reads Reynolds altitude from the design's Mach/Alt grid,
        # exactly as RASAero does; an empty grid means sea level. Sorted by
        # Mach because that is the axis the grid interpolates along.
        design.mach_alt = sorted(
            (float(m), float(h)) for m, h in mach_alt
        )

    engine = Engine(design, boattail_model=settings.boattail_model)

    # The design's fin parts paired with the model's fin sets, in the order
    # the writer emitted them -- which is the first set only, since RASAero's
    # project format carries one. The pairing gives the roll derivatives the
    # planform, cant and body radius the engine's part does not keep.
    from aeroengine.parts import PartType

    fin_parts = [
        (index, part) for index, part in enumerate(design.parts)
        if part.part_type is PartType.FINS
    ]
    external = [fins for fins in model.fin_sets if fins.is_external]
    fin_pairs = [
        (index, part, fins) for (index, part), fins in zip(fin_parts, external)
    ]

    rows: list[AeroCoefficients] = []
    for mach in settings.mach_values():
        derivatives = None
        for alpha in settings.alpha_values():
            r = engine.solve(float(mach), float(alpha))
            if derivatives is None:
                # Linear in alpha by construction, so once per Mach.
                derivatives = rate_derivatives(
                    r, fin_pairs, model.max_diameter_m, origin_m=origin,
                    modified_barrowman=design.modified_barrowman,
                )
            rows.append(AeroCoefficients(
                mach=float(mach),
                alpha_deg=float(alpha),
                cd=r.cd_off_wind,
                cn=r.cn,
                cm=0.0,
                x_cp_m=r.cp / IN_PER_M + origin,
                # The plume-filled base, for while the motor burns. A single
                # column used to be chosen here for the whole flight, so a
                # coasting vehicle flew with its base still "filled".
                cd_power_on=r.cd_on_wind if settings.power_on_base else None,
                **derivatives,
            ))
    return rows


def rate_derivatives(result, fin_pairs, diameter_m: float, origin_m: float = 0.0,
                     modified_barrowman: bool = True) -> dict[str, float]:
    """Pitch-damping moments about the nose and the roll derivatives, at one Mach.

    Pitch damping: every lifting part -- nose, transitions, each fin set's
    fin-on-body and body-carryover terms -- contributes its normal-force
    slope at its own centre of pressure. Stored as the zeroth, first and
    second moments about the nose so the simulator can take ``Cmq`` about
    the CG it has at the moment, which moves as propellant drains. The
    single-surface estimate this replaces put the whole slope at the total
    CP, which on a nose-and-fins vehicle cancels the nose's arm against the
    fins' and understates the damping several-fold. The viscous crossflow
    is left out: it is not linear in alpha and carries no slope.

    Roll: from the fin planform and the engine's fin-set slope divided
    back to one panel -- see :func:`roll_derivatives`. The engine's fin-set
    ``CN_alpha`` is the four-fin chart value scaled by its fin-count
    factor, so one panel is that over twice the factor.
    """
    from aeroengine.adapters import IN_PER_M
    from aeroengine.fins import _FIN_COUNT_FACTOR

    cna_sum = cna_x = cna_x2 = 0.0
    for terms in result.per_part.values():
        for cna, cp_in in (
            (terms.cna_own, terms.cp_own),
            (terms.cna_fin_body, terms.cp_fin_body),
            (terms.cna_body_fin, terms.cp_body_fin),
        ):
            if cna == 0.0:
                continue
            x = cp_in / IN_PER_M + origin_m
            cna_sum += cna
            cna_x += cna * x
            cna_x2 += cna * x * x

    clp = cl_roll = cl_cant = 0.0
    for index, part, fins in fin_pairs:
        terms = result.per_part.get(index)
        if terms is None or part.count <= 0:
            continue
        # Modified Barrowman scales the four-fin chart value by the count
        # factor; classical Barrowman builds 4N(s/d)^2 with the N/2
        # cruciform factor already inside, so one panel is 2/N of the set.
        if modified_barrowman:
            panel = terms.cna_fin_body / (2.0 * _FIN_COUNT_FACTOR.get(part.count, 1.0))
        else:
            panel = 2.0 * terms.cna_fin_body / part.count
        damping, forcing, per_radian = roll_derivatives(fins, panel, diameter_m)
        clp += damping
        cl_roll += forcing
        cl_cant += per_radian

    return {
        "cna_sum": float(cna_sum), "cna_x_m": float(cna_x), "cna_x2_m2": float(cna_x2),
        "clp": float(clp), "cl_roll": float(cl_roll), "cl_cant": float(cl_cant),
    }


def roll_derivatives(fins, cn_alpha_panel: float,
                     diameter_m: float) -> tuple[float, float, float]:
    """Roll damping, cant forcing, and the forcing per radian of cant.

    Strip theory over the exposed span: a panel rolling at rate ``p`` sees
    a local incidence ``p xi / V`` at radius ``xi``, its lift-curve slope
    spread along the span in proportion to chord. Integrating ``xi^2 c(xi)``
    over the trapezoid gives the damping. The cant angle puts the whole
    panel at that incidence, with its force at the spanwise centre of
    pressure -- the trapezoid's mean-chord station -- which gives the
    forcing. Both scale with the per-panel ``CN_alpha`` the engine already
    computes at every Mach, so they follow the fins through the transonic
    rise like everything else.

    Conventions, chosen so the steady roll rate is the closed form
    ``p = -2 V Cl / (d Clp)``: ``Clp`` is per ``p d / 2V`` and ``Cl`` per
    ``q S d``, both on the reference diameter. Positive cant rolls positive
    about the body axis.
    """
    count = fins.count
    root = float(fins.get("root_chord"))
    tip = float(fins.get("tip_chord"))
    span = float(fins.get("span"))
    radius = float(fins.body_radius_m())
    area = float(fins.area_per_fin_m2)
    if count <= 0 or span <= 0.0 or area <= 0.0 or diameter_m <= 0.0:
        return 0.0, 0.0, 0.0

    # Integral of (r + u)^2 (c_r + k u) over the span, u from 0 to s.
    k = (tip - root) / span
    integral = (
        root * radius * radius * span
        + (2.0 * radius * root + k * radius * radius) * span ** 2 / 2.0
        + (root + 2.0 * radius * k) * span ** 3 / 3.0
        + k * span ** 4 / 4.0
    )
    clp = -2.0 * count * cn_alpha_panel * integral / (area * diameter_m * diameter_m)

    y_cp = radius + span * (root + 2.0 * tip) / (3.0 * (root + tip))
    # Per radian of cant, so a cant the table was not built with -- a
    # build error, dispersed -- can be applied in flight without a rebuild.
    cl_cant = count * cn_alpha_panel * y_cp / diameter_m
    cl_roll = cl_cant * np.radians(float(fins.get("cant")))
    return float(clp), float(cl_roll), float(cl_cant)


#: Above this Mach the supersonic base-drag branch is what the flight is
#: flying on, and the boattail caveat applies.
SUPERSONIC_MACH = 1.2


def steepest_boattail_deg(model: VehicleModel) -> float:
    """The largest boattail half-angle on the vehicle [deg], zero without one."""
    import math

    from parametric.analysis import to_canonical

    try:
        canonical = to_canonical(model)
    except Exception:  # noqa: BLE001 -- a model too odd to canonicalise has no caveat
        return 0.0
    angles = [
        math.degrees(math.atan2(
            0.5 * (segment.front_diameter_m - segment.rear_diameter_m),
            max(segment.length_m, 1e-9),
        ))
        for segment in canonical.segments if segment.kind == "boattail"
    ]
    return max(angles, default=0.0)


def boattail_caveat(half_angle_deg: float, max_mach: float,
                    boattail_model: str = "rasaero") -> str | None:
    """What the tool knows to be wrong about this vehicle's supersonic drag.

    Keyed on the two things the evidence is keyed on: a boattail steeper
    than RASAero's separation clamp, and a flight past Mach 1.2. The two
    telemetry-measured flights with such boattails -- Qu8k at 37 degrees,
    Proteus6 at 27 -- are the only measured flights the model misses by
    more than its noise, and it misses them the same way. ``None`` when
    neither condition holds: the other 21 measured flights carry no such
    boattail and score at 5.4% mean apogee error.
    """
    from aeroengine.basedrag import SEPARATION_ANGLE_DEG

    if half_angle_deg <= SEPARATION_ANGLE_DEG or max_mach < SUPERSONIC_MACH:
        return None
    if boattail_model == "corrected":
        return (
            f"Supersonic drag on this {half_angle_deg:.1f} deg boattail is the provisional "
            f"corrected law, which is unvalidated. It exists because RASAero's own law was "
            f"measured 42-103% high in drag through Mach 2.1-2.9 on Qu8k; it has not itself "
            f"been measured against anything, and two flights cannot pin an equation."
        )
    return (
        f"Supersonic drag on this boattail is not trusted. Its half-angle of "
        f"{half_angle_deg:.1f} deg is past the {SEPARATION_ANGLE_DEG:.1f} deg separation clamp "
        f"in RASAero's base-drag law, and against Qu8k's coast telemetry (a 37 deg boattail) "
        f"that law was 42-103% high in drag through Mach 2.1-2.9 -- worth 8-15% of apogee on "
        f"Qu8k and Proteus6, and nothing on the 21 measured flights without such a boattail. "
        f"Expect the apogee to read low. A provisional corrected law is offered in the "
        f"aerodynamics setup; it is itself unvalidated."
    )


def analysis_report(model: VehicleModel, database, geometry: AeroGeometry,
                    cg_station_m: float | None = None, settings=None) -> str:
    """A readable summary of what the sweep produced.

    ``cg_station_m`` is a dry CG -- a meshed solve's -- or ``None`` for the
    analytic one. The margin column is the *loaded* vehicle's, the same
    number the status bar and the results panel quote; it used to print the
    burnout margin beside them, 108% higher on the basic rocket, unlabelled.
    """
    from parametric.analysis import loaded_cg_station_m

    cg = loaded_cg_station_m(model, cg_station_m)
    diameter = geometry.reference_diameter_m

    lines = [
        f"Aerodynamic analysis: {model.name}",
        f"  reference area   {geometry.reference_area_m2 * 1e4:8.1f} cm²  "
        f"(d = {diameter:.3f} m)",
        f"  wetted area      {geometry.body_wetted_area_m2 * 1e4:8.1f} cm² body, "
        f"{geometry.fin_wetted_area_m2 * 1e4:.1f} cm² fins",
        f"  fineness         {geometry.fineness:8.1f}",
        f"  transitions      {len(geometry.transitions)}",
        f"  nose             "
        f"{'declared' if geometry.nose_declared else 'inferred from silhouette'}",
        f"  protuberances    {len(geometry.protuberances)}",
        f"  excluded         "
        f"{', '.join(geometry.excluded) if geometry.excluded else 'none'}",
        "",
    ]
    burning = bool(getattr(database, "has_power_on", False))
    header = f"  {'Mach':>6}{'CD off':>9}"
    if burning:
        header += f"{'CD on':>9}"
    lines.append(header + f"{'CN@4°':>9}{'CP m':>9}{'margin cal':>12}")
    lines.append("  (margin is the loaded vehicle's, at 4 deg alpha)")
    for mach in (0.1, 0.3, 0.6, 0.9, 1.1, 1.5, 2.0, 3.0, 4.0):
        if mach > database.mach_range[1]:
            continue
        zero = database.lookup(mach, 0.0)
        four = database.lookup(mach, 4.0)
        margin = (four.x_cp_m - cg) / diameter
        row = f"  {mach:6.2f}{zero.cd:9.3f}"
        if burning:
            row += f"{zero.cd_power_on:9.3f}"
        lines.append(row + f"{four.cn:9.3f}{four.x_cp_m:9.3f}{margin:12.2f}")

    # What the tool knows to be wrong about this vehicle, if anything.
    mach_range = database.mach_range
    mach_range = mach_range() if callable(mach_range) else mach_range
    caveat = boattail_caveat(
        steepest_boattail_deg(model), float(mach_range[1]),
        getattr(settings, "boattail_model", "rasaero"),
    )
    if caveat:
        lines += ["", "CAVEAT  " + caveat]
    return "\n".join(lines)


def write_csv(database, path: str | Path) -> Path:
    """Write the table in the same CSV format a RASAero import produces."""
    return database.to_csv(path)
