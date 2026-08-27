"""Run RASAero II on a parametric model and bring its table back.

Why this is a subprocess and a pile of clicks
---------------------------------------------
RASAero II has no command line, no scripting interface, and no file-based
batch mode. Its coefficient table exists only behind ``File > Export > To CSV
File`` in the Aero Plots window. Until now that meant the higher-fidelity path
was: export a ``.CDX1``, leave the tool, open RASAero, click through it, save a
CSV somewhere, and import it -- per design iteration. Almost nobody does that
more than once, which is the same as not having the path at all.

``tools/rasaero_driver.ps1`` performs those clicks. It is the ugliest part of
this codebase and it is honest about why: the alternative is a solver nobody
runs.

What comes back
---------------
An ``AeroDatabase``, the same type the built-in Barrowman sweep produces and
the same one the 6-DOF consumes, so a flight can be run on either without the
trajectory code knowing which. RASAero's own sweep is Mach 0.01-25 at alpha 0,
2 and 4 degrees -- note the alpha ceiling, which is well below the 16 degrees
the built-in sweep covers and below what a rocket sees on a windy rail.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

DRIVER = Path(__file__).resolve().parents[1] / "tools" / "rasaero_driver.ps1"
DEFAULT_EXE = Path(r"C:\Program Files (x86)\RASAero II\RASAero II.exe")


class RasaeroUnavailable(RuntimeError):
    """RASAero II is not installed, or this is not Windows."""


@dataclass
class RasaeroRun:
    """What a run produced, including the fit it had to make to get there."""

    database: object              # trajectory.vehicle.aero_database.AeroDatabase
    canonical: object             # parametric.canonical.CanonicalModel
    cdx1_path: Path
    csv_path: Path

    def report(self) -> str:
        rows = len(self.database.rows)
        machs = sorted({row.mach for row in self.database.rows})
        alphas = sorted({row.alpha_deg for row in self.database.rows})
        return (
            f"{self.canonical.report()}\n\n"
            f"RASAero returned {rows} rows\n"
            f"  Mach   {machs[0]:.2f} to {machs[-1]:.2f}\n"
            f"  alpha  {', '.join(f'{a:g}' for a in alphas)} deg\n"
            f"  table  {self.csv_path}"
        )


def executable(path: str | Path | None = None) -> Path:
    """Locate RASAero II, or say why it cannot be used."""
    candidate = Path(path) if path else DEFAULT_EXE
    if not candidate.exists():
        raise RasaeroUnavailable(
            f"RASAero II was not found at {candidate}. Install it, or pass the "
            f"path to its executable."
        )
    return candidate


def available(path: str | Path | None = None) -> bool:
    try:
        executable(path)
    except RasaeroUnavailable:
        return False
    return shutil.which("powershell") is not None


def run(model, work_dir: str | Path, cg_station_m: float | None = None,
        exe: str | Path | None = None, timeout_s: float = 300.0,
        keep_open: bool = False, settings=None) -> RasaeroRun:
    """Export ``model``, run RASAero II on it, and parse the table it exports.

    Args:
        cg_station_m: CG to write into the project, so RASAero's own static
            margin readout matches the vehicle rather than its default guess.
        settings: An ``AeroSettings``. Its Mach and alpha sweep is ignored --
            RASAero runs its own, Mach 0.01 to 25 at alpha 0, 2 and 4. Of the
            conditions, roughness picks the nearest surface finish RASAero
            offers, which moves its drag substantially, and the power-on flag
            chooses which drag column is imported.

            The Reynolds altitude does **not** carry across. RASAero's launch
            site altitude drives its own flight simulation; its aero table is
            built at sea level whatever that field says, which was checked by
            running the same vehicle at 0 m and 1500 m and finding identical
            Reynolds numbers in both exports. It is still written into the
            project so the file is complete, but nothing in the imported table
            depends on it.
        keep_open: Leave RASAero on screen afterwards, to inspect what it made
            of the geometry.
    """
    from parametric import analysis
    from step_to_rasaero.aero_csv_parser import parse_rasaero_csv
    from parametric.canonical import surface_finish

    exe_path = executable(exe)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in model.name)
    cdx1 = work_dir / f"{stem}.CDX1"
    csv = work_dir / f"{stem}_rasaero.csv"

    metadata = None
    power_on = False
    if settings is not None:
        power_on = bool(getattr(settings, "power_on_base", False))
        metadata = {
            "surface": surface_finish(getattr(settings, "roughness_m", 20e-6)),
            "launch_altitude_ft": (
                float(getattr(settings, "altitude_m", 0.0)) / 0.3048
            ),
        }

    cdx1_path, canonical = analysis.write_cdx1(
        model, cdx1, cg_station_m, metadata
    )

    command = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(DRIVER),
        "-Cdx1", str(cdx1_path),
        "-Csv", str(csv),
        "-Exe", str(exe_path),
    ]
    if keep_open:
        command.append("-KeepOpen")

    result = subprocess.run(
        command, capture_output=True, text=True, timeout=timeout_s,
    )
    if result.returncode != 0 or not csv.exists():
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"RASAero run failed:\n{detail}")

    database = parse_rasaero_csv(
        csv,
        reference_length_m=model.total_length_m,
        cg_from_nose_m=(
            cg_station_m if cg_station_m is not None
            else model.mass_summary().cg_station_m
        ),
        power_on=power_on,
    )
    return RasaeroRun(database, canonical, Path(cdx1_path), csv)
