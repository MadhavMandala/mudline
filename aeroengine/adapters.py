"""Bridge the tool's geometry into the engine's vocabulary.

This is the only place SI meets English units. The engine's constants are
transcribed from a program that works in inches, pounds and degrees Rankine --
``RHO_SL = 0.002378 slug/ft^3``, ``P_SL_PSIA = 14.6958`` -- so converting at
the boundary and keeping the internals imperial is the safe direction. The
alternative, converting each formula, would mean re-deriving several hundred
transcribed constants and losing the ability to diff against ``i.cs`` by eye.

The conversion is worth being paranoid about because a missed one is silent.
``CanonicalModel``'s fields are all named ``*_m``; copying rather than
converting yields a vehicle 39x too small, which produces a Reynolds number
39x low, which lands on the wrong side of the roughness cutoff, which shifts
C_D by tens of percent -- and every coefficient still comes back dimensionless
and plausible. The strongest guard available is Reynolds number itself: the
oracle prints it, so a unit error shows up there immediately and unambiguously.
"""

from __future__ import annotations

from .parts import (
    Airfoil,
    BodyTube,
    Design,
    Expansion,
    Fins,
    NoseCone,
    NoseShape,
    Part,
    Reducer,
    SURFACE_ROUGHNESS,
)

#: Exactly 1/0.0254, so the round trip through the tool's own writers is clean.
IN_PER_M = 39.37007874015748
M_PER_FT = 0.3048

__all__ = ["design_from_canonical", "surface_for_roughness", "IN_PER_M"]

#: The tool names nose shapes in its own enum; the engine needs RASAero's
#: strings. Anything unrecognised falls back to a tangent ogive, which is what
#: RASAero's own importers do rather than refusing the file.
_NOSE_SHAPES: dict[str, NoseShape] = {
    "cone": NoseShape.CONICAL,
    "conical": NoseShape.CONICAL,
    "ogive": NoseShape.TANGENT_OGIVE,
    "tangent_ogive": NoseShape.TANGENT_OGIVE,
    "tangent ogive": NoseShape.TANGENT_OGIVE,
    "von_karman": NoseShape.VON_KARMAN,
    "von karman ogive": NoseShape.VON_KARMAN,
    "power_half": NoseShape.POWER_LAW,
    "power_law": NoseShape.POWER_LAW,
    "power law": NoseShape.POWER_LAW,
    "lv_haack": NoseShape.LV_HAACK,
    "lv-haack": NoseShape.LV_HAACK,
    "haack": NoseShape.LV_HAACK,
    "parabolic": NoseShape.PARABOLIC,
    "elliptical": NoseShape.ELLIPTICAL,
    "ellipsoid": NoseShape.ELLIPTICAL,
}


def _nose_shape(value: object) -> NoseShape:
    if value is None:
        return NoseShape.TANGENT_OGIVE
    raw = getattr(value, "value", value)
    return _NOSE_SHAPES.get(str(raw).strip().lower(), NoseShape.TANGENT_OGIVE)


#: RASAero's surface-finish list against equivalent sand-grain roughness.
#:
#: These are not RASAero's own numbers -- it offers names, not heights -- but
#: the names are Hoerner's standard table (*Fluid-Dynamic Drag*, chapter 2)
#: verbatim, in its order, so the heights are the ones that table gives. This
#: is what lets a roughness typed in metres pick a finish rather than the
#: writer hardcoding "Smooth Paint" whatever the user asked for.
SURFACE_ROUGHNESS_M: dict[str, float] = {
    "Smooth (Zero Roughness)": 0.0,
    "Polished": 0.5e-6,
    "Sheet Metal": 4.0e-6,
    "Smooth Paint": 6.35e-6,
    "Camouflage Paint": 10.0e-6,
    "Rough Camouflage Paint": 30.0e-6,
    "Galvanized Metal": 150.0e-6,
    # RASAero offers an eighth finish that this table was missing, so its
    # roughest surface could never be selected however large a roughness was
    # asked for -- everything above ~150 microns silently landed on
    # Galvanized Metal. 254 microns is 0.01 in, the height RASAero uses
    # (ar.cs:3630-3633).
    "Cast Iron (Very Rough)": 254.0e-6,
}


def surface_finish(roughness_m: float) -> str:
    """The RASAero finish whose roughness is closest to the one asked for.

    Closest in log height, not linear: the table spans zero to 150 microns, and
    a linear match would put everything below 20 microns in the same bucket
    when those are the finishes a real airframe actually has.
    """
    import math

    target = max(float(roughness_m), 1e-9)
    return min(
        SURFACE_ROUGHNESS_M,
        key=lambda name: abs(
            math.log(max(SURFACE_ROUGHNESS_M[name], 1e-9)) - math.log(target)
        ),
    )


def surface_for_roughness(roughness_m: float) -> str:
    """Nearest RASAero surface finish to a roughness height in metres.

    Delegates to ``surface_finish`` above rather than choosing
    for itself. That matters more than it looks: the tool's own RASAero path
    picks the finish with that function, and if the native engine picked
    differently the two would be modelling different vehicles while appearing
    to model one.

    This had exactly that bug. A local nearest-in-linear-height rule disagreed
    with the shared nearest-in-LOG-height rule at 20 microns -- which is the
    default value of ``AeroSettings.roughness_m``, so it was the common case,
    not an edge case. Linear picked Camouflage Paint (4.0e-4 in), log picked
    Rough Camouflage Paint (1.2e-3 in): a factor of three in roughness height
    on every default run, and roughness sets the cutoff Reynolds number that
    caps skin friction.

    Log is also the better rule on its own merits, for the reason given at
    that function: the table spans zero to 254 microns and a linear match
    buckets every realistic airframe finish together at the smooth end.
    """
    return surface_finish(roughness_m)


def design_from_canonical(
    model,
    *,
    surface: str = "Smooth Paint",
    modified_barrowman: bool = True,
    turbulence: bool = False,
    nozzle_diameter_m: float = 0.0,
    mach_alt: list[tuple[float, float]] | None = None,
) -> Design:
    """Convert a ``parametric.canonical.CanonicalModel`` into a ``Design``.

    Segments arrive already in RASAero's own vocabulary -- nose, tube,
    transition, boattail -- which is why this adapter is short. A transition is
    re-classified by comparing its ends rather than trusting ``kind``, because
    the engine's Expansion/Reducer split is on geometry and a mislabelled
    segment would otherwise take the wrong drag law.

    Fins are emitted last. That differs from ``cdx1.py``, which reproduces
    RASAero's inline emission order because the file format's fin location is
    relative to a running accumulator. Here every station is already absolute,
    so the only ordering that matters is that the boattail stays the final body
    part -- the supersonic base drag indexes it positionally (i.cs:3442).
    """
    parts: list[Part] = []
    segments = sorted(model.segments, key=lambda s: s.start_m)

    for seg in segments:
        x0 = seg.start_m * IN_PER_M
        length = seg.length_m * IN_PER_M
        d_fwd = seg.front_diameter_m * IN_PER_M
        d_aft = seg.rear_diameter_m * IN_PER_M

        if seg.kind == "nose":
            parts.append(NoseCone(
                length=length,
                d_aft=d_aft,
                shape=_nose_shape(seg.nose_shape),
                blunt_radius=seg.blunt_radius_m * IN_PER_M,
            ))
        elif abs(d_aft - d_fwd) < 1e-9:
            parts.append(BodyTube(x0=x0, length=length, d_aft=d_aft))
        elif d_aft > d_fwd:
            parts.append(Expansion(x0=x0, length=length, d_fwd=d_fwd, d_aft=d_aft))
        else:
            parts.append(Reducer(x0=x0, length=length, d_fwd=d_fwd, d_aft=d_aft))

    for fin in model.fins:
        station_in = fin.station_m * IN_PER_M
        local = _diameter_at(parts, station_in)
        parts.append(Fins(
            x0=station_in,
            count=fin.count,
            root_chord=fin.root_chord_m * IN_PER_M,
            tip_chord=fin.tip_chord_m * IN_PER_M,
            span=fin.span_m * IN_PER_M,
            sweep=fin.sweep_m * IN_PER_M,
            thickness=fin.thickness_m * IN_PER_M,
            le_radius=0.0,
            airfoil=_airfoil(fin.airfoil),
            # RASAero's own dialog defaults these to zero, which for a wedge
            # section is degenerate: FX1 = 0 is clamped to 0.01 in and the
            # wave-drag shape factor explodes. Half-chord and quarter-chord are
            # the sane hexagonal defaults and keep the section well posed.
            fx1=0.25 * fin.root_chord_m * IN_PER_M,
            fx3=0.25 * fin.root_chord_m * IN_PER_M,
            d_fwd=local,
            d_aft=local,
        ))

    return Design(
        parts=parts,
        surface=surface,
        modified_barrowman=modified_barrowman,
        turbulence=turbulence,
        nozzle_diameter=nozzle_diameter_m * IN_PER_M,
        mach_alt=list(mach_alt or []),
    )


def _airfoil(name: object) -> Airfoil:
    try:
        return Airfoil(str(name))
    except ValueError:
        return Airfoil.HEXAGONAL


def _diameter_at(parts: list[Part], station_in: float) -> float:
    """Body diameter where a fin mounts.

    RASAero reads this off the fin's parent part rather than the geometry, so
    a fin whose root chord straddles a diameter change still reports one
    number. The equivalent-volume correction in the geometry pass is what
    handles the change; this only has to name the part it sits on.
    """
    best = 0.0
    for part in parts:
        if part.length <= 0.0:
            continue
        if part.x0 <= station_in <= part.x0 + part.length:
            return part.d_aft
        best = part.d_aft
    return best
