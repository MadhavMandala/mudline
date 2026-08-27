"""Generate the oracle: run every test-matrix vehicle through RASAero.

    python -m oracle.build --out build/oracle

Writes, per case and per alpha, a CDX1, RASAero's Run Test dump, and the
parsed CSV. Cases are independent, but RASAero is a single-instance GUI app
driven by synthetic input, so they run strictly one at a time.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .runtest import parse_dump
from .vehicles import Vehicle, combination_matrix, test_matrix, write_cdx1

DRIVER = Path(__file__).resolve().parent.parent / "tools" / "rasaero_oracle.ps1"


def run_case(v: Vehicle, out_dir: Path, alpha: float, *, timeout: int = 600) -> Path | None:
    """Produce one dump. Returns the parsed CSV path, or None if it failed.

    A failure is reported and skipped rather than raised: one unloadable
    geometry should not throw away an hour of completed cases.
    """
    tag = f"{v.name}_a{str(alpha).replace('.', 'p')}"
    cdx1 = out_dir / "cdx1" / f"{v.name}.CDX1"
    dump = out_dir / "dumps" / f"{tag}.txt"
    csv = out_dir / "csv" / f"{tag}.csv"

    if not cdx1.exists():
        write_cdx1(v, cdx1)

    cmd = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(DRIVER),
        "-Cdx1", str(cdx1), "-Out", str(dump),
        "-Alpha", str(alpha), "-NozzleDiameter", str(v.nozzle_diameter),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip().splitlines()
        print(f"  FAILED {tag}: {err[0] if err else 'unknown error'}")
        return None

    try:
        d = parse_dump(dump)
    except ValueError as exc:
        print(f"  UNPARSEABLE {tag}: {exc}")
        return None

    csv.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(csv)
    events = "".join(f"\n      M {e.mach:6.2f}  {e.text}" for e in d.events)
    print(f"  {tag}: {len(d)} points -> {csv.name}{events}")
    return csv


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="build/oracle", type=Path)
    ap.add_argument("--alpha", nargs="*", type=float, default=[0.0, 4.0],
                    help="angles of attack, degrees (default: 0 4)")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these case names")
    ap.add_argument("--matrix", choices=("features", "combinations", "all"),
                    default="features",
                    help="features isolates one thing per vehicle; combinations "
                         "exercises interactions between them")
    args = ap.parse_args(argv)

    cases = {
        "features": test_matrix,
        "combinations": combination_matrix,
        "all": lambda: test_matrix() + combination_matrix(),
    }[args.matrix]()
    if args.only:
        wanted = set(args.only)
        cases = [c for c in cases if c.name in wanted]
        missing = wanted - {c.name for c in cases}
        if missing:
            print(f"unknown case(s): {', '.join(sorted(missing))}", file=sys.stderr)
            return 2

    total = len(cases) * len(args.alpha)
    print(f"{len(cases)} cases x {len(args.alpha)} alpha = {total} RASAero runs")
    started = time.time()
    ok = 0
    for i, v in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {v.name}")
        for alpha in args.alpha:
            if run_case(v, args.out, alpha) is not None:
                ok += 1

    mins = (time.time() - started) / 60
    print(f"\n{ok}/{total} runs succeeded in {mins:.1f} min -> {args.out}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
