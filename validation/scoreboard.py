"""Score the aerodynamics against real flights.

Everything else in this repository compares the engine to RASAero. This
compares it to rockets. The distinction is the point: the engine reproduces
RASAero to the last printed digit, so any remaining disagreement here is a
property of the *model*, not of the port, and it is the only number that says
whether a change to the physics helped.

Three runs per flight
---------------------
``frozen`` builds the drag table the way RASAero does -- at whatever altitudes
the Mach/Alt grid names, which for all but one of these files is an empty grid
meaning sea level -- then flies that table by Mach alone. Reproducing
RASAero's own stored apogee from this run is the control: it is what licenses
reading anything into the other two, because it shows the trajectory
integrator is not the error term.

``coupled`` rebuilds the table along the altitudes the vehicle actually flies
and iterates to a fixed point. The gap between it and ``frozen`` is what the
sea-level-table defect costs, in feet, on real flights.

Interpreting the result
-----------------------
The truth column is not exact. Some of these apogees are barometric, some GPS,
some integrated accelerometer, and where a flight carried more than one
instrument they disagree -- Kinsel's four readings span 11%. Model error below
the instrument spread on a given flight is not a measurement of anything. The
report prints the spread alongside so that is visible rather than assumed away.

Usage: ``python -m validation.scoreboard [--examples DIR] [--csv OUT]``
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aeroengine import cdx1
from aeroengine.atmosphere import Atmosphere
from aeroengine.solver import Engine

from .flights import FlightCard, load_examples
from .fly import FlightResult, fly
from .motors import Motor, load_rasp

DEFAULT_EXAMPLES = Path.home() / "Documents" / "RASAero II" / "Examples"
DEFAULT_RASP = Path.home() / "Documents" / "RASAero II" / "rasp.eng"

#: Dense through the transonic rise, where the table is least linear, and
#: coarse elsewhere. Linear interpolation between these points is well inside
#: the model's own error.
MACH_GRID = np.unique(np.concatenate([
    np.arange(0.01, 0.80, 0.05),
    np.arange(0.80, 1.35, 0.01),
    np.arange(1.35, 6.01, 0.05),
]))


@dataclass
class CaseResult:
    card: FlightCard
    motor: Motor
    frozen: FlightResult
    coupled: FlightResult
    iterations: int
    boattail_deg: float = 0.0

    @property
    def contested(self) -> bool:
        """True where the stored prediction predates the current boattail model.

        A boattailed vehicle that goes supersonic is the one configuration
        where reproducing RASAero's Run Test -- which this engine does, to the
        last printed digit, including for boattails past the separation clamp
        -- does not reproduce the prediction stored in the example file. The
        current binary cannot disagree with itself: its flight simulation and
        its Run Test read the same coefficients from the same buildup. What
        happened is versioning. The boattail drag model was rewritten for
        RASAero II (1.0.0.0, 2015), and the two flights that trip this
        condition were analysed years earlier -- their stored predictions
        embody the old equations. A boattailed vehicle that stays subsonic
        (IonDrive) shows no gap, which is why the Mach test is part of the
        condition.

        Nothing here says which generation of the model is right; an apogee
        scalar provably cannot (the two flights demand contradictory
        base-drag laws). ``validation.telemetry`` -- CD reconstructed from
        the coast-phase accelerometer -- is the instrument that can.
        """
        return self.boattail_deg > 0.0 and self.frozen.max_mach >= 1.05

    @property
    def frozen_vs_rasaero_pct(self) -> float:
        """Control: my integrator + my aero against RASAero's own answer."""
        ref = self.card.rasaero_ft
        return 100.0 * (self.frozen.apogee_ft - ref) / ref

    @property
    def frozen_error_pct(self) -> float:
        m = self.card.measured_ft
        return 100.0 * (self.frozen.apogee_ft - m) / m

    @property
    def coupled_error_pct(self) -> float:
        m = self.card.measured_ft
        return 100.0 * (self.coupled.apogee_ft - m) / m


def _tables(engine: Engine):
    """Power-off and power-on drag, as Mach-interpolating callables."""
    off = np.empty_like(MACH_GRID)
    on = np.empty_like(MACH_GRID)
    for i, mach in enumerate(MACH_GRID):
        r = engine.solve(float(mach), 0.0)
        off[i] = r.cd_off
        on[i] = r.cd_on
    lo, hi = float(MACH_GRID[0]), float(MACH_GRID[-1])

    def make(values):
        def lookup(mach: float) -> float:
            return float(np.interp(min(max(mach, lo), hi), MACH_GRID, values))
        return lookup

    return make(off), make(on)


def run_case(
    card: FlightCard,
    motor: Motor,
    *,
    dt: float = 0.005,
    max_iterations: int = 6,
    tolerance_pct: float = 0.05,
    boattail_model: str = "rasaero",
) -> CaseResult:
    """Fly one card both ways."""
    design = cdx1.load(card.path)
    atmos = Atmosphere.from_site(
        elevation_ft=card.site_altitude_ft,
        pressure_inhg=card.pressure_inhg,
        temperature_f=card.temperature_f if card.temperature_f else None,
    )

    # -- frozen: RASAero's own drag table, flown by Mach alone -------------
    engine = Engine(design, atmos, boattail_model=boattail_model)
    a_ref = engine.cache.a_ref
    off, on = _tables(engine)
    frozen = fly(card, motor, a_ref, off, on, atmos, dt=dt)

    # -- coupled: re-tabulate along the trajectory until it stops moving ---
    coupled = frozen
    iterations = 0
    previous = frozen.apogee_ft
    for _ in range(max_iterations):
        samples = sorted((m, h) for m, h in coupled.mach_alt if m > 0.0)
        if not samples:
            break
        design.mach_alt = samples
        engine = Engine(design, atmos, boattail_model=boattail_model)
        off_c, on_c = _tables(engine)
        coupled = fly(card, motor, a_ref, off_c, on_c, atmos, dt=dt)
        iterations += 1
        moved = abs(coupled.apogee_ft - previous)
        if previous > 0.0 and 100.0 * moved / previous < tolerance_pct:
            break
        previous = coupled.apogee_ft

    return CaseResult(
        card, motor, frozen, coupled, iterations,
        boattail_deg=engine.cache.boattail_angle_deg,
    )


def _stats(values):
    arr = np.asarray(values, dtype=float)
    return (
        float(arr.mean()),
        float(np.abs(arr).mean()),
        float(arr.min()),
        float(arr.max()),
    )


def _inputs_present(examples: Path, rasp: Path) -> bool:
    """Say what is missing, in words, rather than raising from deep inside.

    This scoreboard is the one tool here that cannot run on what the
    repository ships: the measured-apogee flight cards and the motor database
    are RASAero II's files, not ours, so they are not redistributed. Anyone
    without a RASAero licence reaches this point, and what they used to get
    was a FileNotFoundError traceback out of a path join two calls down --
    which reads as a broken program rather than as a missing input.
    """
    missing = [
        (label, path, why)
        for label, path, why in (
            ("--examples", examples,
             "the example flights RASAero II ships, whose <Comments> carry "
             "the measured apogees this scores against"),
            ("--rasp", rasp,
             "rasp.eng, the motor database RASAero II ships"),
        )
        if not path.exists()
    ]
    if not missing:
        return True

    print("The reality scoreboard needs RASAero II's own data files, which "
          "are\nnot redistributed with this repository.\n", file=sys.stderr)
    for label, path, why in missing:
        print(f"  missing  {label}  {path}", file=sys.stderr)
        print(f"           {why}\n", file=sys.stderr)
    print("They arrive with a RASAero II installation, normally under\n"
          "  Documents/RASAero II/\n"
          "Point at them explicitly if they live elsewhere:\n"
          '  python -m validation.scoreboard --examples "<dir>" --rasp "<file>"\n',
          file=sys.stderr)
    print("Everything else in validation/ runs on committed data. In "
          "particular\n  python -m validation.telemetry\n"
          "reconstructs drag from a real flight's accelerometer log and needs "
          "nothing\nbeyond this repository.", file=sys.stderr)
    return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reality scoreboard")
    ap.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    ap.add_argument("--rasp", type=Path, default=DEFAULT_RASP)
    ap.add_argument("--dt", type=float, default=0.005)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--include-staged", action="store_true")
    ap.add_argument(
        "--boattail", choices=("rasaero", "corrected"), default="rasaero",
        help="supersonic base-drag law: the faithful port, or the "
        "provisional boattail replacement (see aeroengine.basedrag)",
    )
    args = ap.parse_args(argv)

    if not _inputs_present(args.examples, args.rasp):
        return 2

    motors = load_rasp(args.rasp)
    cards = load_examples(args.examples)

    results = []
    skipped = []

    for card in cards:
        if card.staged and not args.include_staged:
            skipped.append((card.name, "two-stage; staging is out of scope"))
            continue
        motor = motors.lookup(card.engine)
        if motor is None:
            skipped.append((card.name, "motor not in rasp.eng: " + card.engine))
            continue
        try:
            results.append(run_case(
                card, motor, dt=args.dt, boattail_model=args.boattail,
            ))
        except Exception as exc:                      # noqa: BLE001
            skipped.append((card.name, type(exc).__name__ + ": " + str(exc)))

    _report(results, skipped)
    if args.csv:
        _write_csv(results, args.csv)
        print("\nwrote " + str(args.csv))
    return 0


def _report(results, skipped) -> None:
    line = "=" * 104
    print(line)
    print("REALITY SCOREBOARD -- apogee against measured flight data")
    print(line)
    head = (
        "FLIGHT".ljust(25)
        + "MEASURED".rjust(9) + "SPRD%".rjust(6) + "  | "
        + "RASAero".rjust(8) + "err%".rjust(7) + "  | "
        + "FROZEN".rjust(8) + "err%".rjust(7) + "vsRA%".rjust(7) + "  | "
        + "COUPLED".rjust(8) + "err%".rjust(7) + "dFT".rjust(7)
    )
    print(head)
    print("-" * 104)

    for r in sorted(results, key=lambda x: x.card.measured_ft):
        c = r.card
        print(
            ("*" if r.contested else " ") + c.name[:24].ljust(24)
            + format(c.measured_ft, ",.0f").rjust(9)
            + format(c.spread_pct, ".1f").rjust(6) + "  | "
            + format(c.rasaero_ft, ",.0f").rjust(8)
            + format(c.rasaero_error_pct, "+.2f").rjust(7) + "  | "
            + format(r.frozen.apogee_ft, ",.0f").rjust(8)
            + format(r.frozen_error_pct, "+.2f").rjust(7)
            + format(r.frozen_vs_rasaero_pct, "+.2f").rjust(7) + "  | "
            + format(r.coupled.apogee_ft, ",.0f").rjust(8)
            + format(r.coupled_error_pct, "+.2f").rjust(7)
            + format(r.coupled.apogee_ft - r.frozen.apogee_ft, "+,.0f").rjust(7)
        )

    print("-" * 104)
    if not results:
        print("no cases ran")
        return

    clean = [r for r in results if not r.contested]
    contested = [r for r in results if r.contested]

    def block(title: str, rows) -> None:
        if not rows:
            return
        print("\n" + title + "  (" + str(len(rows)) + " flights)\n")
        print("".ljust(24) + "bias".rjust(9) + "mean|err|".rjust(11)
              + "worst-".rjust(9) + "worst+".rjust(9) + "within5%".rjust(10))
        for label, vals in (
            ("RASAero (published)", [r.card.rasaero_error_pct for r in rows]),
            ("ours, frozen table", [r.frozen_error_pct for r in rows]),
            ("ours, coupled table", [r.coupled_error_pct for r in rows]),
        ):
            bias, mae, lo, hi = _stats(vals)
            hits = sum(1 for v in vals if abs(v) <= 5.0)
            print(label.ljust(24)
                  + format(bias, "+.2f").rjust(9)
                  + format(mae, ".2f").rjust(11)
                  + format(lo, "+.2f").rjust(9)
                  + format(hi, "+.2f").rjust(9)
                  + (str(hits) + "/" + str(len(rows))).rjust(10))

    block("SCORED", clean)

    bias, mae, lo, hi = _stats([r.frozen_vs_rasaero_pct for r in clean])
    print("\nCONTROL -- ours(frozen) vs RASAero's own prediction, same inputs:")
    print("  mean |difference| " + format(mae, ".2f") + "%"
          + "     range " + format(lo, "+.2f") + "% .. " + format(hi, "+.2f") + "%")
    print("  Same aero, same motor, same site. What is left is the trajectory")
    print("  integrator, and it is small next to the errors above -- so those")
    print("  errors belong to the aerodynamic model, which is the point.")

    if contested:
        block("SET ASIDE -- supersonic boattail (marked *)", contested)
        ctl2 = _stats([r.frozen_vs_rasaero_pct for r in contested])
        print("\n  Control on these: mean |difference| " + format(ctl2[1], ".2f")
              + "%, range " + format(ctl2[2], "+.2f") + "% .. "
              + format(ctl2[3], "+.2f") + "%.")
        print("  These stored predictions predate RASAero II's boattail model")
        print("  (rewritten 2015); this engine reproduces the current one, so")
        print("  the control compares across versions and cannot score these "
              + str(len(contested)) + ".")
        print("  The arbiter is measured drag: python -m validation.telemetry")

    if skipped:
        print("\nSKIPPED")
        for name, why in skipped:
            print("  " + name.ljust(28) + why)


def _write_csv(results, path: Path) -> None:
    import csv
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "flight", "motor", "launch_weight_lb", "measured_ft",
            "instrument_spread_pct", "rasaero_ft", "rasaero_error_pct",
            "frozen_ft", "frozen_error_pct", "frozen_vs_rasaero_pct",
            "coupled_ft", "coupled_error_pct", "coupled_minus_frozen_ft",
            "max_mach", "burnout_mach", "iterations",
        ])
        for r in sorted(results, key=lambda x: x.card.measured_ft):
            c = r.card
            w.writerow([
                c.name, r.motor.key, round(c.launch_weight_lb, 2),
                round(c.measured_ft), round(c.spread_pct, 2),
                round(c.rasaero_ft), round(c.rasaero_error_pct, 2),
                round(r.frozen.apogee_ft), round(r.frozen_error_pct, 2),
                round(r.frozen_vs_rasaero_pct, 2),
                round(r.coupled.apogee_ft), round(r.coupled_error_pct, 2),
                round(r.coupled.apogee_ft - r.frozen.apogee_ft),
                round(r.coupled.max_mach, 3), round(r.coupled.burnout_mach, 3),
                r.iterations,
            ])


if __name__ == "__main__":
    raise SystemExit(main())
