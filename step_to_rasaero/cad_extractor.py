"""Best-effort STEP geometry extraction for RASAero-compatible parameters."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .schema import ExtractionResult, PartGuess


M_PER_IN = 0.0254
STEP_SUFFIXES = {".stp", ".step"}
AXIS_NAMES = ("x", "y", "z")


def extract_geometry(project_dir: str | Path) -> ExtractionResult:
    """Extract a simplified rocket model from a CAD project directory."""
    project_dir = Path(project_dir)
    cad_dir = project_dir / "cad" if (project_dir / "cad").exists() else project_dir
    metadata = _load_metadata(project_dir)
    step_files = sorted(p for p in cad_dir.rglob("*") if p.suffix.lower() in STEP_SUFFIXES)
    if not step_files:
        raise ValueError(f"No STEP files found under {cad_dir}")

    loaded = [_load_part(path, project_dir, metadata) for path in step_files]
    all_points_in = np.vstack([part["points_in"] for part in loaded if len(part["points_in"])])
    if len(all_points_in) == 0:
        raise ValueError("STEP files loaded, but no tessellated points were available.")

    axis_index, axis_sign, axis_reason = _infer_axis(all_points_in, loaded, metadata)
    station_min = float(np.min(all_points_in[:, axis_index]))
    station_max = float(np.max(all_points_in[:, axis_index]))
    station_origin = station_min if axis_sign > 0 else station_max
    transverse_indices = [idx for idx in range(3) if idx != axis_index]
    centerline_in = np.median(all_points_in[:, transverse_indices], axis=0)

    for part in loaded:
        points = part["points_in"]
        stations = _station(points[:, axis_index], station_origin, axis_sign)
        radial = _radial_distance(points[:, transverse_indices], centerline_in)
        part["stations_in"] = stations
        part["radial_in"] = radial
        part["station_range_in"] = [float(np.min(stations)), float(np.max(stations))]

    guesses = [_classify_part(part, metadata) for part in loaded]
    for part, guess in zip(loaded, guesses):
        part["category"] = guess.category

    body_like = [
        part
        for part in loaded
        if part["category"] in {"nose", "body", "transition", "unknown"}
    ]
    if not body_like:
        body_like = loaded

    radius_profile = _radius_profile(body_like)
    if len(radius_profile) == 0:
        raise ValueError("Could not build a body radius profile from the STEP geometry.")

    model = _build_model(project_dir, metadata, radius_profile, loaded, guesses)
    review = _build_review(project_dir, axis_index, axis_sign, axis_reason, loaded, guesses)
    return ExtractionResult(model=model, review=review)


def write_extraction(result: ExtractionResult, output_dir: str | Path) -> tuple[Path, Path]:
    """Write extracted geometry and classification review JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry_path = output_dir / "extracted_geometry.json"
    review_path = output_dir / "classification_review.json"
    geometry_path.write_text(json.dumps(result.model, indent=2), encoding="utf-8")
    review_path.write_text(json.dumps(result.review, indent=2), encoding="utf-8")
    return geometry_path, review_path


def _load_metadata(project_dir: Path) -> dict[str, Any]:
    candidates = [
        project_dir / "config" / "rocket_metadata.yaml",
        project_dir / "config" / "rocket_metadata.yml",
        project_dir / "config" / "rocket_metadata.json",
        project_dir / "rocket_metadata.yaml",
        project_dir / "rocket_metadata.yml",
        project_dir / "rocket_metadata.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        try:
            import yaml  # type: ignore

            data = yaml.safe_load(text) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return _parse_simple_yaml(text)
    return {}


def _load_part(path: Path, project_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        from trajectory.vehicle.stp_reader import load_stp

        reader = load_stp(str(path), tessellation_tolerance=float(metadata.get("tessellation_tolerance", 0.75)))
        points = np.asarray(reader.vertices, dtype=float).reshape(-1, 3)
        bounds = reader.get_bounds()
        backend = "cadquery"
        warning = None
    except Exception as exc:
        points = _load_points_with_gmsh(path, metadata)
        bounds = _bounds_from_points(points)
        backend = "gmsh"
        warning = f"CadQuery load failed, used Gmsh fallback: {exc}"
    return {
        "path": path,
        "relative_path": str(path.relative_to(project_dir)),
        "backend": backend,
        "warning": warning,
        "points_in": points,
        "bounds_in": bounds,
        "centroid_in": np.mean(points, axis=0),
    }


def _load_points_with_gmsh(path: Path, metadata: dict[str, Any]) -> np.ndarray:
    repo_root = Path(__file__).resolve().parents[1]
    massprops_src = repo_root / "massprops" / "src"
    if str(massprops_src) not in sys.path:
        sys.path.insert(0, str(massprops_src))
    from massprops.mesh.mesher import generate_watertight_mesh

    vertices, faces = generate_watertight_mesh(
        path,
        mesh_size=metadata.get("mesh_size"),
        mesh_size_factor=metadata.get("mesh_size_factor"),
    )
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"Gmsh generated no surface triangles for {path}")
    return np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)].reshape(-1, 3)


def _bounds_from_points(points: np.ndarray) -> dict[str, tuple[float, float]]:
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    return {
        "x": (float(mins[0]), float(maxs[0])),
        "y": (float(mins[1]), float(maxs[1])),
        "z": (float(mins[2]), float(maxs[2])),
    }


def _infer_axis(
    points_in: np.ndarray,
    parts: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> tuple[int, int, str]:
    hint = metadata.get("rocket_axis_hint")
    if isinstance(hint, str):
        hint = hint.strip().lower()
        sign = -1 if hint.startswith("-") else 1
        axis = hint[-1:]
        if axis in AXIS_NAMES:
            return AXIS_NAMES.index(axis), sign, "metadata rocket_axis_hint"
    if isinstance(hint, (list, tuple)) and len(hint) >= 3:
        arr = np.asarray(hint[:3], dtype=float)
        idx = int(np.argmax(np.abs(arr)))
        sign = 1 if arr[idx] >= 0 else -1
        return idx, sign, "metadata rocket_axis_hint"

    spans = np.ptp(points_in, axis=0)
    axis_index = int(np.argmax(spans))
    sign = _infer_axis_sign_from_nose(parts, axis_index, metadata)
    return axis_index, sign, "longest CAD bounding-box span"


def _infer_axis_sign_from_nose(
    parts: list[dict[str, Any]],
    axis_index: int,
    metadata: dict[str, Any],
) -> int:
    tip_hint = str(metadata.get("nose_tip_hint", "")).lower()
    if tip_hint.startswith("min"):
        return 1
    if tip_hint.startswith("max"):
        return -1

    nose_parts = [p for p in parts if "nose" in str(p["relative_path"]).lower()]
    if not nose_parts:
        return 1
    nose_center = float(np.mean([p["centroid_in"][axis_index] for p in nose_parts]))
    all_centers = [float(p["centroid_in"][axis_index]) for p in parts]
    return 1 if nose_center <= float(np.median(all_centers)) else -1


def _station(axis_values: np.ndarray, origin: float, sign: int) -> np.ndarray:
    return (axis_values - origin) * sign


def _radial_distance(points_transverse: np.ndarray, centerline_transverse: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points_transverse - centerline_transverse[None, :], axis=1)


def _classify_part(part: dict[str, Any], metadata: dict[str, Any]) -> PartGuess:
    rel = str(part["relative_path"]).replace("\\", "/")
    lower = rel.lower()
    rules = [
        ("nose", ("nose", "cone", "ogive", "haack")),
        ("fin", ("fin", "canard", "strake")),
        ("transition", ("boattail", "boat_tail", "transition", "tailcone", "tail_cone")),
        ("body", ("body", "tube", "airframe", "fuselage", "cylinder")),
        ("protuberance", ("rail", "lug", "button", "shoe", "raceway", "camera", "antenna")),
    ]
    for category, tokens in rules:
        hits = [token for token in tokens if token in lower]
        if hits:
            return PartGuess(
                path=rel,
                category=category,
                confidence=0.9,
                reasons=[f"path contains {', '.join(hits)}"],
                bounds_m=_bounds_to_m(part["bounds_in"]),
                station_range_m=[v * M_PER_IN for v in part.get("station_range_in", [0.0, 0.0])],
            )

    fin_count = int(metadata.get("fin_count", 0) or 0)
    spans = _part_spans(part)
    longest = float(np.max(spans))
    shortest = float(np.min(spans))
    if fin_count and shortest > 0 and longest / shortest > 6:
        return PartGuess(
            path=rel,
            category="fin",
            confidence=0.45,
            reasons=["thin elongated part and metadata has fin_count"],
            bounds_m=_bounds_to_m(part["bounds_in"]),
            station_range_m=[v * M_PER_IN for v in part.get("station_range_in", [0.0, 0.0])],
        )

    return PartGuess(
        path=rel,
        category="unknown",
        confidence=0.2,
        reasons=["no folder/name rule matched"],
        bounds_m=_bounds_to_m(part["bounds_in"]),
        station_range_m=[v * M_PER_IN for v in part.get("station_range_in", [0.0, 0.0])],
    )


def _part_spans(part: dict[str, Any]) -> np.ndarray:
    b = part["bounds_in"]
    return np.array([b[axis][1] - b[axis][0] for axis in AXIS_NAMES], dtype=float)


def _radius_profile(parts: list[dict[str, Any]]) -> list[dict[str, float]]:
    stations = np.concatenate([part["stations_in"] for part in parts])
    radial = np.concatenate([part["radial_in"] for part in parts])
    valid = np.isfinite(stations) & np.isfinite(radial)
    stations = stations[valid]
    radial = radial[valid]
    if len(stations) < 8:
        return []

    length = float(np.max(stations) - np.min(stations))
    bin_count = int(np.clip(length / max(length / 160.0, 0.25), 32, 200))
    edges = np.linspace(float(np.min(stations)), float(np.max(stations)), bin_count + 1)
    samples: list[dict[str, float]] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (stations >= lo) & (stations <= hi)
        if np.count_nonzero(mask) < 3:
            continue
        samples.append(
            {
                "s_m": float(((lo + hi) * 0.5) * M_PER_IN),
                "radius_m": float(np.percentile(radial[mask], 95) * M_PER_IN),
            }
        )
    return _thin_profile(samples)


def _thin_profile(samples: list[dict[str, float]]) -> list[dict[str, float]]:
    if len(samples) <= 80:
        return samples
    stride = int(np.ceil(len(samples) / 80))
    return samples[::stride] + ([] if samples[-1] is samples[::stride][-1] else [samples[-1]])


def _build_model(
    project_dir: Path,
    metadata: dict[str, Any],
    radius_profile: list[dict[str, float]],
    parts: list[dict[str, Any]],
    guesses: list[PartGuess],
) -> dict[str, Any]:
    length_m = max(point["s_m"] for point in radius_profile)
    max_radius_m = max(point["radius_m"] for point in radius_profile)
    max_diameter_m = max_radius_m * 2.0

    nose_length_m = _nose_length(parts, guesses, radius_profile, max_radius_m)
    boattail = _boattail_section(parts, guesses, radius_profile, max_radius_m, length_m)
    body_end_m = boattail["start"] if boattail else length_m
    body_length_m = max(body_end_m - nose_length_m, 0.001)
    fins = _extract_fins(parts, guesses, metadata, max_radius_m)

    model: dict[str, Any] = {
        "format": "mudline.extracted_geometry.v1",
        "source_project": str(project_dir),
        "units": "m",
        "rocket": {
            "name": str(metadata.get("name") or project_dir.name),
            "length": length_m,
            "max_diameter": max_diameter_m,
            "reference_area": float(np.pi * max_radius_m**2),
        },
        "nose": {
            "type": str(metadata.get("nose_type") or _guess_nose_shape(radius_profile, nose_length_m)),
            "length": nose_length_m,
            "base_diameter": max_diameter_m,
        },
        "body_sections": [
            {
                "type": "tube",
                "start": nose_length_m,
                "length": body_length_m,
                "diameter": max_diameter_m,
            }
        ],
        "fins": fins,
        "body_profile": radius_profile,
        "metadata": {
            "source_metadata": metadata,
        },
    }
    if boattail:
        model["body_sections"].append(boattail)
    return model


def _nose_length(
    parts: list[dict[str, Any]],
    guesses: list[PartGuess],
    profile: list[dict[str, float]],
    max_radius_m: float,
) -> float:
    nose_lengths = [
        (part["station_range_in"][1] - part["station_range_in"][0]) * M_PER_IN
        for part, guess in zip(parts, guesses)
        if guess.category == "nose"
    ]
    if nose_lengths:
        return max(float(max(nose_lengths)), 0.001)

    threshold = max_radius_m * 0.94
    for point in profile:
        if point["radius_m"] >= threshold:
            return max(point["s_m"], 0.001)
    return max(profile[-1]["s_m"] * 0.2, 0.001)


def _boattail_section(
    parts: list[dict[str, Any]],
    guesses: list[PartGuess],
    profile: list[dict[str, float]],
    max_radius_m: float,
    length_m: float,
) -> dict[str, float] | None:
    transition_ranges = [
        [v * M_PER_IN for v in part["station_range_in"]]
        for part, guess in zip(parts, guesses)
        if guess.category == "transition"
    ]
    rear_radius_m = profile[-1]["radius_m"]
    if transition_ranges:
        start = min(r[0] for r in transition_ranges)
        end = max(r[1] for r in transition_ranges)
    elif rear_radius_m < max_radius_m * 0.88:
        start = max(point["s_m"] for point in profile if point["radius_m"] >= max_radius_m * 0.92)
        end = length_m
    else:
        return None

    return {
        "type": "boattail",
        "start": float(start),
        "length": max(float(end - start), 0.001),
        "front_diameter": float(max_radius_m * 2.0),
        "rear_diameter": float(max(rear_radius_m * 2.0, 0.001)),
    }


def _guess_nose_shape(profile: list[dict[str, float]], nose_length_m: float) -> str:
    nose_points = [p for p in profile if p["s_m"] <= nose_length_m and nose_length_m > 0]
    if len(nose_points) < 5:
        return "Tangent Ogive"
    s = np.array([p["s_m"] / nose_length_m for p in nose_points])
    r = np.array([p["radius_m"] for p in nose_points])
    r = r / max(float(np.max(r)), 1e-9)
    linear_error = float(np.mean(np.abs(r - s)))
    return "Conical" if linear_error < 0.08 else "Tangent Ogive"


def _extract_fins(
    parts: list[dict[str, Any]],
    guesses: list[PartGuess],
    metadata: dict[str, Any],
    body_radius_m: float,
) -> list[dict[str, float]]:
    fin_parts = [part for part, guess in zip(parts, guesses) if guess.category == "fin"]
    if not fin_parts:
        return []
    fin_count = int(metadata.get("fin_count") or len(fin_parts))

    measurements = [_measure_fin(part, body_radius_m) for part in fin_parts]
    merged = {
        "axial_location": _median_measurement(measurements, "axial_location", 0.0, allow_zero=True),
        "root_chord": _median_measurement(measurements, "root_chord", 0.001),
        "tip_chord": _median_measurement(measurements, "tip_chord", 0.001),
        "span": _median_measurement(measurements, "span", 0.001),
        "sweep": _median_measurement(measurements, "sweep", 0.0, allow_negative=True),
        "thickness": _median_measurement(measurements, "thickness", 0.001),
    }
    merged["count"] = fin_count
    merged["airfoil"] = str(metadata.get("fin_airfoil") or "Hexagonal")
    return [merged]


def _median_measurement(
    measurements: list[dict[str, float]],
    key: str,
    fallback: float,
    allow_zero: bool = False,
    allow_negative: bool = False,
) -> float:
    values = []
    for measurement in measurements:
        value = float(measurement.get(key, np.nan))
        if not np.isfinite(value):
            continue
        if allow_negative or value > 0 or (allow_zero and value == 0):
            values.append(value)
    return float(np.median(values)) if values else float(fallback)


def _measure_fin(part: dict[str, Any], body_radius_m: float) -> dict[str, float]:
    s = np.asarray(part["stations_in"], dtype=float) * M_PER_IN
    r = np.asarray(part["radial_in"], dtype=float) * M_PER_IN
    points_m = np.asarray(part["points_in"], dtype=float) * M_PER_IN
    radial_span = max(float(np.max(r) - body_radius_m), 0.001)

    q25 = np.percentile(r, 35)
    q75 = np.percentile(r, 75)
    root_s = s[r <= q25]
    tip_s = s[r >= q75]
    if len(root_s) < 3:
        root_s = s
    if len(tip_s) < 3:
        tip_s = s

    spans = np.ptp(points_m, axis=0)
    thickness = float(max(np.min(spans), 0.001))
    return {
        "axial_location": float(np.min(root_s)),
        "root_chord": float(max(np.max(root_s) - np.min(root_s), 0.001)),
        "tip_chord": float(max(np.max(tip_s) - np.min(tip_s), 0.001)),
        "span": radial_span,
        "sweep": float(np.min(tip_s) - np.min(root_s)),
        "thickness": thickness,
    }


def _build_review(
    project_dir: Path,
    axis_index: int,
    axis_sign: int,
    axis_reason: str,
    parts: list[dict[str, Any]],
    guesses: list[PartGuess],
) -> dict[str, Any]:
    questions = []
    for guess in guesses:
        if guess.confidence < 0.65:
            questions.append(
                {
                    "path": guess.path,
                    "question": f"I guessed '{guess.category}'. What rocket role is this part?",
                    "suggested_categories": ["nose", "body", "transition", "fin", "protuberance", "ignore"],
                }
            )
    if axis_reason.startswith("longest"):
        questions.append(
            {
                "path": str(project_dir),
                "question": f"I guessed rocket axis {('+' if axis_sign > 0 else '-')}{AXIS_NAMES[axis_index]}. Please verify.",
                "suggested_categories": ["+x", "-x", "+y", "-y", "+z", "-z"],
            }
        )
    return {
        "axis": {
            "name": f"{('+' if axis_sign > 0 else '-')}{AXIS_NAMES[axis_index]}",
            "index": axis_index,
            "sign": axis_sign,
            "reason": axis_reason,
        },
        "parts": [
            {
                **guess.__dict__,
                "backend": part.get("backend"),
                "warning": part.get("warning"),
            }
            for part, guess in zip(parts, guesses)
        ],
        "questions": questions,
    }


def _bounds_to_m(bounds_in: dict[str, tuple[float, float]]) -> dict[str, list[float]]:
    return {axis: [float(lo * M_PER_IN), float(hi * M_PER_IN)] for axis, (lo, hi) in bounds_in.items()}


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("\"'")
        if value.startswith("[") and value.endswith("]"):
            result[key.strip()] = [float(item.strip()) for item in value[1:-1].split(",") if item.strip()]
        elif value.lower() in {"true", "false"}:
            result[key.strip()] = value.lower() == "true"
        else:
            try:
                result[key.strip()] = float(value)
            except ValueError:
                result[key.strip()] = value
    return result
