"""RASAero's canonical part vocabulary, and the writer that speaks it.

The engine -- native or the application itself -- describes a vehicle as
nose, tubes, transitions, boattail and trapezoidal fins. This module holds
those canonical parts, the fitting that reads them off an arbitrary radius
profile, and the CDX1 payload builder. ``parametric.analysis`` produces the
canonical model from components; everything downstream of it models
byte-identical geometry because every path funnels through here.

    parametric build --\
    STEP import -------+--> canonical parts --> CDX1 --> aero engine --> AeroDatabase --> 6-DOF
    CFD / wind tunnel --------------------------------------------------------^
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from enum import Enum


class NoseShape(str, Enum):
    """Nose cone profiles, in RASAero's vocabulary.

    ``VON_KARMAN`` is the usual choice for a supersonic vehicle because it
    minimises wave drag for a given length and base diameter; ``OGIVE`` is
    the usual choice when it has to be made on a manual lathe.
    """

    CONICAL = "conical"
    OGIVE = "ogive"
    VON_KARMAN = "von_karman"
    ELLIPTICAL = "elliptical"
    POWER_HALF = "power_half"


def nose_radius_profile(
    shape: NoseShape, length: float, radius: float, stations: Iterable[float]
) -> np.ndarray:
    """Radius at each axial station along a nose cone.

    Shared by the shape fitter and the exporter so the two cannot describe
    different noses.

    Args:
        stations: Axial positions from the tip, 0 to ``length``.
    """
    z = np.clip(np.asarray(list(stations), dtype=float), 0.0, length)
    if length <= 0 or radius <= 0:
        return np.zeros_like(z)

    fraction = z / length

    if shape is NoseShape.CONICAL:
        return radius * fraction

    if shape is NoseShape.ELLIPTICAL:
        return radius * np.sqrt(np.clip(1.0 - (1.0 - fraction) ** 2, 0.0, None))

    if shape is NoseShape.POWER_HALF:
        return radius * np.sqrt(fraction)

    if shape is NoseShape.VON_KARMAN:
        # Haack series with C = 0, the minimum-drag body for a given length
        # and base diameter.
        theta = np.arccos(np.clip(1.0 - 2.0 * fraction, -1.0, 1.0))
        return (radius / np.sqrt(np.pi)) * np.sqrt(
            np.clip(theta - 0.5 * np.sin(2.0 * theta), 0.0, None)
        )

    # Tangent ogive: an arc of radius rho tangent to the body at the base.
    rho = (radius ** 2 + length ** 2) / (2.0 * radius)
    inner = np.clip(rho ** 2 - (length - z) ** 2, 0.0, None)
    return np.sqrt(inner) + radius - rho


#: Fractional diameter change below which two adjacent runs are treated as the
#: same tube. Manufacturing steps and paint are not aerodynamic features.
DIAMETER_TOLERANCE = 0.01

from aeroengine.adapters import SURFACE_ROUGHNESS_M, surface_finish  # noqa: F401  (canonical vocabulary, re-exported)


@dataclass
class CanonicalSegment:
    """One part from RASAero's vocabulary.

    ``kind`` is one of ``nose``, ``tube``, ``transition`` (diameter increasing
    aft) or ``boattail`` (decreasing aft).
    """

    kind: str
    start_m: float
    length_m: float
    front_diameter_m: float
    rear_diameter_m: float
    nose_shape: NoseShape | None = None
    blunt_radius_m: float = 0.0
    #: Which definition components were lumped into this segment.
    sources: list[str] = field(default_factory=list)

    @property
    def end_m(self) -> float:
        return self.start_m + self.length_m

    def radius_at(self, station_m: np.ndarray) -> np.ndarray:
        """Radius of this segment's outer mold line at the given stations."""
        z = np.asarray(station_m, dtype=float)
        local = np.clip((z - self.start_m) / max(self.length_m, 1e-12), 0.0, 1.0)

        if self.kind == "nose" and self.nose_shape is not None:
            return nose_radius_profile(
                self.nose_shape, self.length_m, 0.5 * self.rear_diameter_m,
                z - self.start_m,
            )
        front_r, rear_r = 0.5 * self.front_diameter_m, 0.5 * self.rear_diameter_m
        return front_r + (rear_r - front_r) * local


@dataclass
class CanonicalFin:
    """A trapezoidal fin set, the only fin RASAero models."""

    count: int
    root_chord_m: float
    tip_chord_m: float
    span_m: float
    sweep_m: float
    thickness_m: float
    station_m: float
    airfoil: str = "Hexagonal"


@dataclass
class CanonicalModel:
    """A vehicle expressed entirely in RASAero's vocabulary."""

    name: str
    segments: list[CanonicalSegment]
    fins: list[CanonicalFin] = field(default_factory=list)
    #: RMS radius error against the source silhouette [m]. Zero when the source
    #: was already canonical.
    residual_rms_m: float = 0.0
    #: Worst-case radius error [m].
    residual_max_m: float = 0.0
    cg_from_nose_m: float = 0.0
    wet_mass_kg: float = 0.0
    #: Nozzle exit diameter, for the plume's share of the base. Zero means
    #: no nozzle, and then the power-on drag column equals power-off.
    nozzle_exit_diameter_m: float = 0.0

    @property
    def nose_start_m(self) -> float:
        """Station of the nose tip: RASAero's origin, in the model's frame.

        RASAero measures everything from the nose tip. The project is
        written with the tip at zero, and whatever comes back from it --
        a centre of pressure, a part station -- is offset by this to land
        in the model's stations again.
        """
        nose = [s for s in self.segments if s.kind == "nose"]
        return float(nose[0].start_m) if nose else 0.0

    @property
    def total_length_m(self) -> float:
        return max((s.end_m for s in self.segments), default=0.0)

    @property
    def max_diameter_m(self) -> float:
        return max(
            (max(s.front_diameter_m, s.rear_diameter_m) for s in self.segments),
            default=0.0,
        )

    def radius_at(self, station_m: np.ndarray) -> np.ndarray:
        """Outer mold line of the whole canonical model."""
        z = np.asarray(station_m, dtype=float)
        radius = np.zeros_like(z)
        for segment in self.segments:
            inside = (z >= segment.start_m) & (z <= segment.end_m + 1e-12)
            if np.any(inside):
                radius[inside] = segment.radius_at(z[inside])
        return radius

    def report(self) -> str:
        """What got lumped into what, and how well it fits."""
        lines = [
            f"Canonical model: {self.name}",
            f"  {len(self.segments)} body segments, {len(self.fins)} fin set(s)",
            "",
            f"  {'kind':<11}{'station m':>18}{'diameter m':>16}  lumped from",
        ]
        for segment in self.segments:
            span = f"{segment.start_m:6.3f} - {segment.end_m:6.3f}"
            if abs(segment.front_diameter_m - segment.rear_diameter_m) < 1e-9:
                diameter = f"{segment.rear_diameter_m:.3f}"
            else:
                diameter = f"{segment.front_diameter_m:.3f}->{segment.rear_diameter_m:.3f}"
            lines.append(
                f"  {segment.kind:<11}{span:>18}{diameter:>16}  "
                f"{', '.join(segment.sources) or '-'}"
            )
        for fin in self.fins:
            lines.append(
                f"  fins x{fin.count:<6}{fin.station_m:>12.3f}      "
                f"root {fin.root_chord_m:.3f} tip {fin.tip_chord_m:.3f} "
                f"span {fin.span_m:.3f} sweep {fin.sweep_m:.3f}"
            )
        lines += [
            "",
            f"  Silhouette residual: {self.residual_rms_m * 1000:.2f} mm RMS, "
            f"{self.residual_max_m * 1000:.2f} mm max",
        ]
        if self.max_diameter_m > 0:
            relative = self.residual_max_m / (0.5 * self.max_diameter_m)
            lines.append(f"  Worst error is {relative * 100:.2f}% of body radius")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# From a vehicle definition: an exact mapping, no fitting required
# ----------------------------------------------------------------------


def canonical_from_profile(
    stations_m: Iterable[float],
    radii_m: Iterable[float],
    name: str = "fitted",
    diameter_tolerance: float = DIAMETER_TOLERANCE,
) -> CanonicalModel:
    """Fit canonical parts to a measured radius profile.

    This is the path for geometry that did not come from a definition -- a STEP
    file from anywhere. It works from the silhouette alone, so it does not care
    what the parts are called, which is the failure mode of classifying by
    filename.

    The segmentation is deliberately simple and inspectable: walk aft, and
    start a new segment whenever the local slope changes character (flat versus
    tapering) by more than the tolerance. The leading segment is a nose, and
    its shape is chosen by testing every profile in the vocabulary and keeping
    whichever fits best -- the same principle as the rest of the module, since
    a nose family is a canonical shape and picking one is a fit, not a guess.
    """
    z = np.asarray(list(stations_m), dtype=float)
    r = np.asarray(list(radii_m), dtype=float)
    if z.size < 3:
        raise ValueError("Need at least three profile samples to fit.")

    order = np.argsort(z)
    z, r = z[order], r[order]
    z = z - z[0]
    max_radius = float(r.max())
    if max_radius <= 0:
        raise ValueError("Profile has no positive radius.")

    # Where the nose ends is fitted, not thresholded.
    #
    # The obvious rule -- "the nose ends where the radius first reaches its
    # maximum" -- is biased short, because the useful profiles approach the
    # base radius asymptotically. On a von Karman cone a 1% threshold cuts the
    # nose 6% short, and nose length drives wave drag, so that error lands
    # straight in the drag coefficient. Instead every plausible nose end is
    # tried against every shape in the vocabulary, and the combination with the
    # lowest whole-model residual wins.
    reached_full = np.flatnonzero(r >= max_radius * (1.0 - diameter_tolerance))
    earliest = int(reached_full[0]) if reached_full.size else max(z.size // 4, 1)
    latest = int(reached_full[-1]) if reached_full.size else z.size - 1
    candidates = np.unique(
        np.clip(np.linspace(earliest, latest, 40).astype(int), 1, z.size - 2)
    )

    best_model: CanonicalModel | None = None
    for nose_end_index in candidates:
        nose_length = float(z[nose_end_index])
        if nose_length <= 0:
            continue
        for shape in NoseShape:
            segments = [
                CanonicalSegment(
                    kind="nose", start_m=0.0, length_m=nose_length,
                    front_diameter_m=0.0, rear_diameter_m=2 * max_radius,
                    nose_shape=shape, sources=["fitted nose"],
                )
            ]
            segments.extend(
                _lump_segments(
                    _segment_afterbody(
                        z[nose_end_index:], r[nose_end_index:],
                        diameter_tolerance, max_radius,
                    ),
                    _segmentation_step(r[nose_end_index:], diameter_tolerance, max_radius),
                )
            )
            candidate = CanonicalModel(name=name, segments=segments)
            _measure_residual(candidate, z, r)
            if best_model is None or candidate.residual_rms_m < best_model.residual_rms_m:
                best_model = candidate

    if best_model is None:
        raise ValueError("Could not fit a canonical model to this profile.")
    return best_model


def estimate_profile_noise(r: np.ndarray) -> float:
    """Estimate the measurement noise on a radius profile [m].

    Second differences of a straight or slowly-curving profile are pure noise,
    and for independent samples of standard deviation sigma they have standard
    deviation sigma*sqrt(6). A median-absolute-deviation estimate is used
    rather than a plain standard deviation so that genuine geometric features
    -- a real step or taper -- do not inflate the estimate of the noise.
    """
    r = np.asarray(r, dtype=float)
    if r.size < 3:
        return 0.0
    second = r[:-2] - 2.0 * r[1:-1] + r[2:]
    mad = float(np.median(np.abs(second - np.median(second))))
    return 1.4826 * mad / np.sqrt(6.0)


def _segmentation_step(r: np.ndarray, tolerance: float, max_radius: float) -> float:
    """Deviation a run may show before it is split into two segments.

    Taking this as a fixed fraction of the body radius ignores how noisy the
    profile actually is: a scan with 1.5 mm of scatter then trips the split
    test constantly and shatters a plain cylinder into hundreds of segments.
    The threshold therefore never sits below a few standard deviations of the
    measured noise.
    """
    geometric = float(tolerance * max_radius)
    return max(geometric, 4.0 * estimate_profile_noise(r))


def _segment_afterbody(
    z: np.ndarray, r: np.ndarray, tolerance: float, max_radius: float
) -> list[CanonicalSegment]:
    """Split the post-nose profile into constant and tapering runs."""
    if z.size < 2:
        return []

    step = _segmentation_step(r, tolerance, max_radius)
    segments: list[CanonicalSegment] = []
    start = 0

    def flush(end: int) -> None:
        if end <= start:
            return
        z0, z1 = float(z[start]), float(z[end])
        r0, r1 = float(r[start]), float(r[end])
        if z1 - z0 <= 0:
            return
        if abs(r1 - r0) <= step:
            mean_d = 2.0 * float(np.mean(r[start:end + 1]))
            segments.append(CanonicalSegment(
                kind="tube", start_m=z0, length_m=z1 - z0,
                front_diameter_m=mean_d, rear_diameter_m=mean_d,
                sources=["fitted tube"],
            ))
        else:
            segments.append(CanonicalSegment(
                kind="boattail" if r1 < r0 else "transition",
                start_m=z0, length_m=z1 - z0,
                front_diameter_m=2 * r0, rear_diameter_m=2 * r1,
                sources=["fitted taper"],
            ))

    # A run ends when the profile stops being well described by a straight line
    # through its endpoints.
    for index in range(2, z.size):
        span = slice(start, index + 1)
        fitted = np.interp(z[span], [z[start], z[index]], [r[start], r[index]])
        if np.max(np.abs(r[span] - fitted)) > step:
            flush(index - 1)
            start = index - 1
    flush(z.size - 1)
    return segments


def _lump_segments(
    segments: list[CanonicalSegment], step: float
) -> list[CanonicalSegment]:
    """Merge segments that are not aerodynamically distinct.

    The segmenter works from local slope, so measurement noise on a plain
    cylinder splits it into a run of near-identical tubes and hairline tapers.
    Handing RASAero twelve parts where the vehicle has one is both wrong and
    unusable, so this is the same lumping applied to the fit itself: a taper
    whose diameter barely changes *is* a tube, and consecutive tubes of the
    same diameter *are* one tube.

    The threshold is deliberately the same tolerance used elsewhere: a feature
    smaller than it is below the level at which a component-buildup code
    resolves anything anyway.
    """
    normalised: list[CanonicalSegment] = []

    for segment in segments:
        working = segment
        if working.kind in {"boattail", "transition"}:
            change = abs(working.rear_diameter_m - working.front_diameter_m)
            if change <= step:
                mean_d = 0.5 * (working.front_diameter_m + working.rear_diameter_m)
                working = CanonicalSegment(
                    kind="tube", start_m=working.start_m, length_m=working.length_m,
                    front_diameter_m=mean_d, rear_diameter_m=mean_d,
                    sources=working.sources,
                )

        if not normalised:
            normalised.append(working)
            continue

        previous = normalised[-1]
        mergeable = (
            previous.kind == "tube"
            and working.kind == "tube"
            and abs(previous.rear_diameter_m - working.rear_diameter_m) <= step
        )
        if mergeable:
            total = previous.length_m + working.length_m
            if total > 0:
                # Length-weighted, so a long tube is not dragged by a sliver.
                previous.front_diameter_m = previous.rear_diameter_m = (
                    previous.rear_diameter_m * previous.length_m
                    + working.rear_diameter_m * working.length_m
                ) / total
            previous.length_m = total
            continue

        normalised.append(working)

    return normalised


def _best_nose_shape(
    z: np.ndarray, r: np.ndarray, length: float, radius: float
) -> NoseShape:
    """Pick the nose family whose profile best matches the measured one."""
    best, best_error = NoseShape.OGIVE, np.inf
    for shape in NoseShape:
        predicted = nose_radius_profile(shape, length, radius, z)
        error = float(np.sqrt(np.mean((predicted - r) ** 2)))
        if error < best_error:
            best, best_error = shape, error
    return best


def _measure_residual(model: CanonicalModel, z: np.ndarray, r: np.ndarray) -> None:
    """Record how far the canonical model departs from the true silhouette."""
    predicted = model.radius_at(z)
    error = predicted - np.asarray(r, dtype=float)
    model.residual_rms_m = float(np.sqrt(np.mean(error ** 2)))
    model.residual_max_m = float(np.max(np.abs(error)))


# ----------------------------------------------------------------------
# Emit RASAero input
# ----------------------------------------------------------------------


_NOSE_NAMES = {
    NoseShape.CONICAL: "Conical",
    NoseShape.OGIVE: "Tangent Ogive",
    NoseShape.VON_KARMAN: "Von Karman Ogive",
    NoseShape.ELLIPTICAL: "Elliptical",
    NoseShape.POWER_HALF: "Power Law",
}


def to_rasaero_model(model: CanonicalModel, metadata: dict[str, Any] | None = None) -> dict:
    """Convert to the intermediate schema ``rasaero_writer`` consumes."""
    nose = next(s for s in model.segments if s.kind == "nose")
    # Every station leaves relative to the nose tip: RASAero's origin. The
    # nose used to be written at zero while everything else kept its model
    # station, so a vehicle whose nose sat at 0.5 m went out as two
    # disjoint rockets with the CG half a metre from the CP it was judged
    # against.
    origin = nose.start_m
    body_sections = []
    for segment in model.segments:
        if segment.kind == "tube":
            body_sections.append({
                "type": "tube",
                "start": segment.start_m - origin,
                "length": segment.length_m,
                "diameter": segment.rear_diameter_m,
            })
        elif segment.kind in {"boattail", "transition"}:
            # A transition is RASAero's Transition, an expansion or a
            # reduction mid-body that advances the station count; a
            # boattail is its BoatTail, the taper at the tail. Both used to
            # be written as BoatTail, so a payload bulge was solved as a
            # taper -- 15% off in drag and normal force supersonically.
            body_sections.append({
                "type": segment.kind,
                "start": segment.start_m - origin,
                "length": segment.length_m,
                "front_diameter": segment.front_diameter_m,
                "rear_diameter": segment.rear_diameter_m,
            })

    merged_metadata = {
        "cg_from_nose_m": model.cg_from_nose_m - origin,
        "launch_weight_lbf": model.wet_mass_kg * 2.2046226218487757,
        "sustainer_nozzle_diameter_m": model.nozzle_exit_diameter_m,
        "modified_barrowman": True,
        "all_turbulent": True,
    }
    merged_metadata.update(metadata or {})

    return {
        "rocket": {
            "name": model.name,
            "length": model.total_length_m,
            "max_diameter": model.max_diameter_m,
        },
        "nose": {
            "type": _NOSE_NAMES.get(nose.nose_shape or NoseShape.OGIVE, "Tangent Ogive"),
            "start": 0.0,
            "length": nose.length_m,
            "base_diameter": nose.rear_diameter_m,
            "blunt_radius": nose.blunt_radius_m,
        },
        "body_sections": body_sections,
        "fins": [
            {
                "count": fin.count,
                "root_chord": fin.root_chord_m,
                "tip_chord": fin.tip_chord_m,
                "span": fin.span_m,
                "sweep": fin.sweep_m,
                "thickness": fin.thickness_m,
                "axial_location": fin.station_m - origin,
                "airfoil": fin.airfoil,
            }
            for fin in model.fins
        ],
        "metadata": {"source_metadata": merged_metadata},
    }


def write_cdx1(
    model: CanonicalModel,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> tuple[Path, CanonicalModel]:
    """Write a canonical model as a RASAero II project file.

    Returns the path and the model, so a caller can show what was lumped and
    how well it fits before trusting anything downstream produces.
    """
    from step_to_rasaero.rasaero_writer import write_rasaero_cdx1

    path = write_rasaero_cdx1(to_rasaero_model(model, metadata), output_path)
    return path, model
