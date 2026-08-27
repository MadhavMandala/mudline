"""Parse RASAero II CSV exports into the simulator aero database format."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from trajectory.vehicle.aero_database import AeroCoefficients, AeroDatabase


IN_TO_M = 0.0254


def parse_rasaero_csv(
    path: str | Path,
    reference_length_m: float = 1.0,
    cg_from_nose_m: float = 0.0,
    power_on: bool = False,
) -> AeroDatabase:
    """Parse one RASAero CSV export into an ``AeroDatabase``.

    The folder walker below predates the driver: a hand-run export produced a
    directory of per-alpha files, whereas a single ``File > Export`` writes one
    file covering the whole sweep. Both end up in the same place.

    Args:
        power_on: Carry the powered drag column as well, in which the plume
            fills the base and base drag largely disappears. The flight
            simulator uses it while the motor burns and the power-off column
            after burnout. The two differ by a great deal on a blunt-based
            vehicle, which is why the same rocket decelerates far harder
            after burnout than during the burn. Off keeps one column,
            power-off, for the whole flight.
    """
    path = Path(path)
    rows, reason = _read_csv(path)
    if not rows:
        raise ValueError(f"No recognizable RASAero coefficients in {path}: {reason}")
    merged = _merge_rows(rows, reference_length_m, cg_from_nose_m, power_on=power_on)
    return AeroDatabase(merged, reference_length_m=reference_length_m)


def parse_rasaero_csv_folder(
    csv_dir: str | Path,
    output_dir: str | Path | None = None,
    reference_length_m: float = 1.0,
    cg_from_nose_m: float = 0.0,
) -> dict[str, Any]:
    """Parse every CSV in a folder and write a normalized aero database."""
    csv_dir = Path(csv_dir)
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {csv_dir}")

    parsed_rows: list[dict[str, float]] = []
    skipped: list[dict[str, Any]] = []
    for path in files:
        rows, reason = _read_csv(path)
        if rows:
            parsed_rows.extend(rows)
        else:
            skipped.append({"path": str(path), "reason": reason})

    if not parsed_rows:
        raise ValueError("No recognizable RASAero coefficient columns were found.")

    rows = _merge_rows(parsed_rows, reference_length_m, cg_from_nose_m)
    db = AeroDatabase(rows, reference_length_m=reference_length_m)

    output_dir = Path(output_dir) if output_dir is not None else csv_dir.parent / "aero_database"
    output_dir.mkdir(parents=True, exist_ok=True)
    coeff_path = db.to_csv(output_dir / "baseline_coeffs.csv")
    meta_path = db.to_metadata(output_dir / "metadata.json")
    parse_report = {
        "source_dir": str(csv_dir),
        "csv_files": [str(path) for path in files],
        "skipped": skipped,
        "output": {
            "baseline_coeffs": str(coeff_path),
            "metadata": str(meta_path),
        },
    }
    report_path = output_dir / "parse_report.json"
    report_path.write_text(json.dumps(parse_report, indent=2), encoding="utf-8")
    return {
        "database": db,
        "coeff_path": coeff_path,
        "metadata_path": meta_path,
        "report_path": report_path,
        "skipped": skipped,
    }


def _read_csv(path: Path) -> tuple[list[dict[str, float]], str]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return [], "empty file"

    header_index = _find_header_index(lines)
    if header_index is None:
        return [], "no recognizable header"

    dialect = csv.Sniffer().sniff("\n".join(lines[header_index : header_index + 4]))
    reader = csv.DictReader(lines[header_index:], dialect=dialect)
    rows = []
    for row in reader:
        normalized = {_normalize_key(k): v for k, v in row.items() if k is not None}
        item = _extract_row(normalized, path.name)
        if item is not None:
            rows.append(item)
    return rows, "parsed"


def _find_header_index(lines: list[str]) -> int | None:
    for idx, line in enumerate(lines[:25]):
        lower = line.lower()
        if any(token in lower for token in ("mach", "alpha", "angle", "cd", "cn", "cp")):
            return idx
    return None


def _extract_row(row: dict[str, str], filename: str) -> dict[str, float] | None:
    mach = _first_number(row, ("mach", "mach number"))
    alpha = _first_number(row, ("alpha", "angle of attack", "aoa"))
    # RASAero exports a bare "CD" column *and* "CD Power-Off"/"CD Power-On".
    # The bare one is the alpha-zero value repeated down every alpha for a
    # given Mach; only the power-qualified columns carry the drag rise with
    # angle of attack. Reading "CD" first -- which is what an exact-name-first
    # search does -- therefore imported a table with no induced drag at all,
    # understating drag exactly when the vehicle is flying at alpha, which is
    # the windy part of the boost where it matters. Both qualified columns
    # are read; whether the powered one is carried is the caller's choice.
    cd = _first_number(row, ("cd power off", "power off cd", "cd power-off",
                             "cd", "drag coefficient"))
    cd_on = _first_number(row, ("cd power on", "power on cd", "cd power-on"))
    cn = _first_number(row, ("cn", "normal force coefficient"))
    cp = _first_number(row, ("cp", "center of pressure", "center pressure"))

    file_alpha = _alpha_from_filename(filename)
    file_mach = _mach_from_filename(filename)
    if mach is None:
        mach = file_mach
    if alpha is None:
        alpha = file_alpha if file_alpha is not None else 0.0

    if mach is None and alpha is None:
        return None
    if cd is None and cn is None and cp is None:
        return None
    return {
        "mach": float(mach if mach is not None else 0.0),
        "alpha_deg": float(alpha if alpha is not None else 0.0),
        "cd": float(cd) if cd is not None else np.nan,
        "cd_on": float(cd_on) if cd_on is not None else np.nan,
        "cn": float(cn) if cn is not None else np.nan,
        "x_cp_m": float(cp) * IN_TO_M if cp is not None else np.nan,
    }


def _merge_rows(
    parsed: list[dict[str, float]],
    reference_length_m: float,
    cg_from_nose_m: float,
    power_on: bool = False,
) -> list[AeroCoefficients]:
    by_key: dict[tuple[float, float], dict[str, float]] = {}
    for row in parsed:
        key = (round(row["mach"], 5), round(abs(row["alpha_deg"]), 5))
        current = by_key.setdefault(key, {"mach": key[0], "alpha_deg": key[1]})
        for field in ("cd", "cd_on", "cn", "x_cp_m"):
            value = row.get(field, np.nan)
            if np.isfinite(value):
                current[field] = value

    defaults = _global_defaults(parsed)
    result = []
    for row in by_key.values():
        cd = row.get("cd", defaults["cd"])
        cd_on = row.get("cd_on", np.nan)
        cn = row.get("cn", defaults["cn"])
        x_cp_m = row.get("x_cp_m", defaults["x_cp_m"])
        cm = cn * (cg_from_nose_m - x_cp_m) / max(reference_length_m, 1e-9)
        result.append(
            AeroCoefficients(
                mach=float(row["mach"]),
                alpha_deg=float(row["alpha_deg"]),
                cd=float(cd),
                cn=float(cn),
                cm=float(cm),
                x_cp_m=float(x_cp_m),
                cd_power_on=(
                    float(cd_on) if power_on and np.isfinite(cd_on) else None
                ),
            )
        )
    result.sort(key=lambda r: (r.mach, r.alpha_deg))
    return result


def _global_defaults(rows: list[dict[str, float]]) -> dict[str, float]:
    defaults = {}
    for field, fallback in (("cd", 0.5), ("cn", 0.0), ("x_cp_m", 0.0)):
        values = [row[field] for row in rows if np.isfinite(row.get(field, np.nan))]
        defaults[field] = float(np.median(values)) if values else fallback
    return defaults


def _first_number(row: dict[str, str], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        key_norm = _normalize_key(key)
        for actual_key, value in row.items():
            if actual_key == key_norm or key_norm in actual_key:
                number = _parse_number(value)
                if number is not None:
                    return number
    return None


def _parse_number(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _alpha_from_filename(filename: str) -> float | None:
    match = re.search(r"(?:alpha|aoa)[_\-\s]*(\d+(?:\.\d+)?)", filename, re.IGNORECASE)
    return float(match.group(1)) if match else None


def _mach_from_filename(filename: str) -> float | None:
    match = re.search(r"mach[_\-\s]*(\d+(?:\.\d+)?)", filename, re.IGNORECASE)
    return float(match.group(1)) if match else None
