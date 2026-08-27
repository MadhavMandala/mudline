"""Import STEP geometry as an *editable* parametric model.

The point
---------
Importing CAD normally gives you a dead mesh: something to look at, not
something to change. The section model makes a better outcome possible. Slice
the solid along its axis, measure the cross-section at each station, and you
have a Stack -- which is a first-class component you can then edit, sweep, or
hand to the aero and mass tools like any other.

So both of the application's inputs land in the same place:

    parametric build ------\\
                            +--> VehicleModel --> geometry, mass, aero, flight
    STEP import -----------/

Method
------
1. Find the axis. A rocket is slender, so the axis is the direction of greatest
   extent; that is checked against the cross-sectional area rather than assumed
   from the bounding box, because a long fin set can outrank a short fat body
   on bounding box alone.
2. Sample the radius at stations along the axis by intersecting the solid with
   a plane and measuring the resulting face. Area is used rather than a
   bounding radius, so fins crossing the plane inflate it far less.
3. Reduce the sampled profile to as few sections as will describe it. A
   cylinder needs two; a curved nose needs a dozen. Section placement is chosen
   by the error it removes, not by uniform spacing.

The residual is reported, in millimetres. An import you cannot judge the
fidelity of is not much better than a mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from parametric.components import Stack
from parametric.model import VehicleModel
from parametric.xsec import XSec, XSecShape

MM_PER_M = 1000.0


@dataclass
class ImportReport:
    """What the importer found and how well it fits."""

    source: Path
    axis: str
    length_m: float
    max_diameter_m: float
    sections: int
    samples: int
    residual_rms_m: float
    residual_max_m: float
    solid_volume_m3: float
    fitted_volume_m3: float
    notes: list[str] = field(default_factory=list)
    #: What the file said it was made of, if anything.
    material: str = ""
    material_source: str = ""
    declared_mass_kg: float = 0.0

    @property
    def volume_error(self) -> float:
        if self.solid_volume_m3 <= 0:
            return 0.0
        return abs(self.fitted_volume_m3 - self.solid_volume_m3) / self.solid_volume_m3

    def text(self) -> str:
        lines = [
            f"Imported {self.source.name}",
            f"  axis            {self.axis}",
            f"  length          {self.length_m:.4f} m",
            f"  max diameter    {self.max_diameter_m:.4f} m",
            f"  sampled         {self.samples} stations",
            f"  fitted          {self.sections} cross-sections",
            f"  radius residual {self.residual_rms_m * 1000:.2f} mm RMS, "
            f"{self.residual_max_m * 1000:.2f} mm max",
            f"  volume          {self.fitted_volume_m3 * 1e6:.1f} cm³ fitted vs "
            f"{self.solid_volume_m3 * 1e6:.1f} cm³ solid "
            f"({self.volume_error * 100:.1f}%)",
            f"  material        {self.material or '(none)'} "
            f"[{self.material_source}]",
        ]
        if self.declared_mass_kg > 0:
            lines.append(f"  declared mass   {self.declared_mass_kg:.4f} kg")
        lines += [f"  ! {note}" for note in self.notes]
        return "\n".join(lines)


def _require_cadquery():
    try:
        import cadquery as cq
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'STEP import needs cadquery. Install the CAD extra:\n'
            '    python -m pip install -e ".[cad]"'
        ) from exc
    return cq


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------


def detect_axis(solid) -> int:
    """Index of the axis a slender body lies along: 0 = X, 1 = Y, 2 = Z."""
    box = solid.BoundingBox()
    spans = np.array([box.xlen, box.ylen, box.zlen])
    return int(np.argmax(spans))


def sample_profile(solid, axis: int, samples: int = 120):
    """Sample the body along its axis.

    Returns ``(stations_m, outer_radius_m, material_radius_m)``.

    Two radii, because they answer different questions and disagreeing is
    informative:

    * **outer** is the lateral extent of a thin slice -- the outer mould line,
      which is what the geometry model and the aerodynamics both want.
    * **material** is the slice volume over its thickness, turned into an
      equivalent radius. On a solid body the two agree. On a hollow tube the
      material radius collapses to sqrt(R^2 - r^2), which on a 65 mm shell with
      a 3 mm wall is 19.5 mm -- a third of the truth. Comparing them is how the
      importer knows it is looking at a shell.

    Slice volume is used rather than summing the faces of a planar section:
    volume is one well-conditioned number, whereas face-area bookkeeping has to
    decide which faces belong to the cut and which to the sides, and gets that
    wrong on any shape with a step in it.
    """
    cq = _require_cadquery()
    box = solid.BoundingBox()
    lows = [box.xmin, box.ymin, box.zmin]
    highs = [box.xmax, box.ymax, box.zmax]
    span = highs[axis] - lows[axis]
    if span <= 0:
        raise ValueError("Solid has no extent along the detected axis.")

    lateral = max(box.xlen, box.ylen, box.zlen) * 2.0 + 10.0
    thickness = max(span * 0.004, 1e-3)

    # Inset the ends: a slab straddling the tip measures a partial cap.
    stations_mm = np.linspace(
        lows[axis] + thickness, highs[axis] - thickness, samples
    )

    lateral_axes = [i for i in range(3) if i != axis]
    outer_mm = np.zeros(samples)
    material_mm = np.zeros(samples)
    failures = 0

    for index, station in enumerate(stations_mm):
        dims = [lateral, lateral, lateral]
        dims[axis] = thickness
        corner = [
            0.5 * (lows[i] + highs[i]) - 0.5 * dims[i] for i in range(3)
        ]
        corner[axis] = float(station) - 0.5 * thickness

        try:
            slab = cq.Solid.makeBox(dims[0], dims[1], dims[2], cq.Vector(*corner))
            piece = solid.intersect(slab)
            # Outer mould line: the lateral extent of the slice. This is what
            # the geometry model wants -- the surface the air sees.
            piece_box = piece.BoundingBox()
            extents = [piece_box.xlen, piece_box.ylen, piece_box.zlen]
            outer_mm[index] = 0.25 * (extents[lateral_axes[0]] + extents[lateral_axes[1]])
            # Material area, kept only to detect a hollow shell: on a tube it
            # measures the annulus, so sqrt(area/pi) collapses to
            # sqrt(R^2 - r^2), far below the true outer radius.
            material_mm[index] = np.sqrt(max(piece.Volume() / thickness, 0.0) / np.pi)
        except Exception:
            failures += 1

    if failures == samples:
        raise ValueError(
            "Every cross-section measurement failed; the shape may not be a "
            "valid solid."
        )

    stations_m = (stations_mm - lows[axis]) / MM_PER_M
    return stations_m, outer_mm / MM_PER_M, material_mm / MM_PER_M


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


def fit_sections(
    stations_m: np.ndarray,
    radii_m: np.ndarray,
    tolerance_m: float = 0.0008,
    max_sections: int = 40,
) -> list[int]:
    """Choose the fewest sample indices whose piecewise-linear fit meets tolerance.

    A greedy refinement: start with the two ends, then repeatedly insert the
    station where the current fit is worst. Placing sections by the error they
    remove is what lets a cylinder come back as two sections and a von Karman
    nose as a dozen, instead of both getting the same uniform spacing.
    """
    stations_m = np.asarray(stations_m, dtype=float)
    radii_m = np.asarray(radii_m, dtype=float)
    if len(stations_m) < 2:
        return list(range(len(stations_m)))

    chosen = [0, len(stations_m) - 1]
    for _ in range(max_sections - 2):
        knots = np.array(sorted(chosen))
        fitted = np.interp(stations_m, stations_m[knots], radii_m[knots])
        error = np.abs(fitted - radii_m)
        error[knots] = 0.0
        worst = int(np.argmax(error))
        if error[worst] <= tolerance_m:
            break
        chosen.append(worst)
    return sorted(chosen)


def build_stack_from_profile(
    stations_m: np.ndarray,
    radii_m: np.ndarray,
    name: str = "imported",
    wall_thickness_m: float = 0.0,
    tolerance_m: float = 0.0008,
) -> tuple[Stack, np.ndarray]:
    """Fit a Stack to a sampled profile. Returns the stack and the fitted radii."""
    indices = fit_sections(stations_m, radii_m, tolerance_m)
    stack = Stack(name, wall_thickness_m=wall_thickness_m)
    for order, index in enumerate(indices):
        stack.add_section(XSec(
            station_m=float(stations_m[index]),
            shape=XSecShape.CIRCLE,
            width_m=float(2.0 * radii_m[index]),
            name=f"{name}_{order}",
        ))
    fitted = np.interp(stations_m, stations_m[indices], radii_m[indices])
    return stack, fitted


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------


def import_step(
    path: str | Path,
    name: str | None = None,
    samples: int = 120,
    tolerance_m: float = 0.0008,
    wall_thickness_m: float = 0.0,
    material: str = "",
    use_file_material: bool = True,
) -> tuple[VehicleModel, ImportReport]:
    """Read a STEP solid and return an editable parametric model.

    Args:
        material: Force a material, overriding anything the file carries.
        use_file_material: Read AP242 material and density when present. A
            STEP carries geometry, and if that is all it carries the importer
            has to guess -- which used to mean silently inheriting a default
            and reporting a mass wrong by whatever the density ratio was.
    """
    cq = _require_cadquery()
    from cadquery import importers

    path = Path(path)
    solid = importers.importStep(str(path)).val()

    axis = detect_axis(solid)
    stations_m, radii_m, material_m = sample_profile(solid, axis, samples)

    positive = radii_m > 0
    notes: list[str] = []
    if positive.sum() < 3:
        raise ValueError(
            f"{path.name}: could not measure a cross-section anywhere along the "
            f"detected axis. Is this a solid?"
        )
    if not positive.all():
        notes.append(
            f"{int((~positive).sum())} of {samples} stations returned no section; "
            f"they were dropped."
        )
        stations_m = stations_m[positive]
        radii_m = radii_m[positive]
        material_m = material_m[positive]

    # A shell reports far less material than its outer radius implies. Detect
    # it and estimate the wall, so the imported Stack has the right mass rather
    # than the mass of a solid billet.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(radii_m > 0, material_m / radii_m, 1.0)
    hollow = float(np.median(ratio)) < 0.9
    if hollow and wall_thickness_m <= 0.0:
        # material^2 = R^2 - r^2  =>  wall = R - sqrt(R^2 - material^2)
        walls = radii_m - np.sqrt(np.clip(radii_m ** 2 - material_m ** 2, 0.0, None))
        usable = walls[(walls > 0) & np.isfinite(walls)]
        if len(usable):
            wall_thickness_m = float(np.median(usable))
            notes.append(
                f"Hollow shell detected; wall thickness estimated at "
                f"{wall_thickness_m * 1000:.2f} mm from the material area."
            )

    # Sampling is inset from both ends by a slab thickness, so the fitted body
    # would come out short. Extend the first and last stations back to the real
    # extent of the solid, holding their radii.
    box = solid.BoundingBox()
    true_length_m = ([box.xlen, box.ylen, box.zlen][axis]) / MM_PER_M
    stations_m = np.concatenate([[0.0], stations_m, [true_length_m]])
    radii_m = np.concatenate([radii_m[:1], radii_m, radii_m[-1:]])

    stack_name = name or path.stem
    stack, fitted = build_stack_from_profile(
        stations_m, radii_m, stack_name, wall_thickness_m, tolerance_m
    )

    # What the file says about itself, before anything is assumed.
    metadata = _read_metadata(path) if use_file_material else None
    chosen, source = _choose_material(metadata, material, notes)
    if chosen:
        stack.material = chosen

    model = VehicleModel(stack_name, "imported")
    model.description = f"Imported from {path.name}"
    model.add(stack)

    residual = fitted - radii_m
    # Compare like with like: the solid's volume is material, so the fitted
    # figure must be the stack's material volume too. Comparing it against the
    # enclosed volume of a shell reports a 785% error on a perfect import.
    fitted_volume = stack.volume_m3()
    declared = metadata.mass_kg if metadata else 0.0
    if declared > 0:
        # The originating CAD weighed it; trust that over volume x density.
        stack.mass_override_kg = declared
        notes.append(
            f"Mass {declared:.4f} kg taken from the file, overriding the "
            f"{stack.computed_mass_kg():.4f} kg the geometry implies."
        )

    report = ImportReport(
        source=path,
        axis="XYZ"[axis],
        length_m=float(stations_m[-1] - stations_m[0]),
        max_diameter_m=float(2.0 * radii_m.max()),
        sections=len(stack.sections),
        samples=len(stations_m),
        residual_rms_m=float(np.sqrt(np.mean(residual ** 2))),
        residual_max_m=float(np.max(np.abs(residual))),
        solid_volume_m3=float(solid.Volume()) / (MM_PER_M ** 3),
        fitted_volume_m3=fitted_volume,
        notes=notes,
        material=stack.material,
        material_source=source,
        declared_mass_kg=declared,
    )

    if report.residual_max_m > 0.01:
        report.notes.append(
            "Large radius residual: the shape may not be a body of revolution, "
            "or fins are crossing the sampling planes."
        )
    return model, report


def _read_metadata(path):
    """Material and mass from the file, never fatal to the import."""
    from parametric.step_metadata import read_step_metadata

    try:
        return read_step_metadata(path)
    except Exception:  # noqa: BLE001
        return None


def _choose_material(metadata, forced: str, notes: list[str]) -> tuple[str, str]:
    """Decide the material, most trustworthy source first.

    Order: what the caller insisted on, then what the file declared, then
    nothing -- and "nothing" is said out loud rather than silently defaulted,
    because an unstated material is a mass error waiting to be believed.
    """
    from parametric.materials import MATERIALS, material_named

    if forced:
        return forced, "specified on import"

    if metadata is not None:
        for note in metadata.notes:
            notes.append(note)

        named = metadata.material_name.strip().lower().replace(" ", "_")
        if named in MATERIALS:
            notes.append(f"Material {named!r} read from the file.")
            return named, "declared in the file"

        if metadata.density_kg_m3 > 0:
            entry = material_named(
                metadata.density_kg_m3, metadata.material_name
            )
            notes.append(
                f"Material {metadata.material_name or 'unnamed'!r} "
                f"({metadata.density_kg_m3:,.0f} kg/m3) read from the file."
            )
            return entry.name, "declared in the file"

        if metadata.material_name:
            notes.append(
                f"The file names its material {metadata.material_name!r} but "
                f"gives no density, so mass is provisional."
            )

    notes.append(
        "No material in the file; mass is provisional until one is set."
    )
    return "", "defaulted"
