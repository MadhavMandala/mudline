"""Parse RASAero II's Tools > Run Test dump into a tidy table.

Why this file exists
--------------------
RASAero's CSV export gives totals. Run Test gives the *component breakdown* --
friction, form or wave, base, each fin contribution, protuberances -- at 0.01
Mach steps across all three solver regimes. When a reimplementation disagrees
on total CD, the totals tell you nothing; the breakdown tells you which term.

Format
------
Six lines per Mach point, blank-line separated, occasionally interrupted by
banner lines marking a state change (boundary-layer transition, fin leading
edge going supersonic). Field widths come from ``i.cs:787-845``: values are
right-aligned into the width of their format string, which means columns can
run together when a value overflows its field. Splitting on whitespace is
therefore wrong in general -- but the *count* of fields distinguishes the
regimes, and RASAero pads generously enough that overflow only happens for
Reynolds number, which is last. We split on whitespace and validate the count.

The subsonic and transonic blocks emit 12 fields on line 1; the supersonic
block emits 14, because body form drag is replaced by nose wave drag and the
fin terms split further. That count is how the regime is identified -- the
dump does not label it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Line 1, subsonic and transonic blocks (i.cs:800-802).
# Column 2 is field m_h, which the dump zeroes immediately before printing and
# which is never assigned anywhere else in the assembly. It is not alpha.
_LINE1_SUB = [
    "mach", "_dead", "cd_off", "cd_on",
    "cd_friction", "cd_form", "cd_base",
    "fin_profile", "fin_interference", "fin_edge",
    "cd_protuberance", "reynolds",
]

# Line 1, supersonic block (i.cs:833-835).
_LINE1_SUP = [
    "mach", "_dead", "cd_off", "cd_on",
    "cd_friction", "cd_wave_nose", "cd_base",
    "fin_friction", "fin_wave", "fin_interference", "fin_edge",
    "transition_wave",
    "cd_protuberance", "reynolds",
]

# Line 1, transonic block (i.cs:817). Five fields, and no component
# breakdown at all -- there is nothing to break down, because 0.90 < M < 1.05
# is interpolated between the two neighbouring solves rather than computed.
# Its presence in the dump is itself confirmation that the fairing evaluates
# no terms of its own.
_LINE1_TRANS = ["mach", "_dead", "cd_off", "cd_on", "reynolds"]

_LINE2 = ["mach", "cn_alpha_0to4", "cp_0to4"]
_LINE3 = ["mach", "alpha_deg", "cn_potential", "cn_viscous"]
_LINE4 = ["mach", "alpha_deg", "cn", "cp"]
_LINE5 = ["mach", "alpha_deg", "cl_off", "cd_off_wind", "cn", "ca_off"]
_LINE6 = ["mach", "alpha_deg", "cl_on", "cd_on_wind", "cn", "ca_on"]

_BANNER = "-------->"


@dataclass
class Event:
    """A state change RASAero announced mid-sweep.

    These are free assertions on a reimplementation: they pin the exact Mach
    at which the roughness cutoff bites, or a fin leading edge goes
    supersonic. Getting the coefficient right but the transition Mach wrong
    means the right answer for the wrong reason.
    """

    mach: float
    text: str


@dataclass
class Dump:
    rows: list[dict[str, float]]
    events: list[Event] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def machs(self) -> list[float]:
        return [r["mach"] for r in self.rows]

    def at(self, mach: float, tol: float = 1e-6) -> dict[str, float]:
        for r in self.rows:
            if abs(r["mach"] - mach) < tol:
                return r
        raise KeyError(f"no record at Mach {mach}")

    def to_csv(self, path: str | Path) -> Path:
        import csv

        path = Path(path)
        cols: list[str] = []
        for r in self.rows:
            for k in r:
                if k not in cols:
                    cols.append(k)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, restval="")
            w.writeheader()
            w.writerows(self.rows)
        return path


def _records(lines: list[str]) -> Iterator[tuple[list[str], list[str]]]:
    """Yield (data_lines, pending_banners) for each Mach point.

    Banners are emitted from inside the solver, before the record they belong
    to is written, so they attach forward to the next record.
    """
    pending: list[str] = []
    block: list[str] = []
    for raw in lines:
        line = raw.rstrip("\r\n")
        if _BANNER in line:
            pending.append(line.replace("-", " ").replace("<", "").replace(">", "").strip())
            continue
        if not line.strip():
            if block:
                yield block, pending
                pending = []
                block = []
            continue
        block.append(line)
    if block:
        yield block, pending


def _to_floats(line: str) -> list[float]:
    return [float(tok) for tok in line.split()]


def parse_dump(path: str | Path, *, strict: bool = True) -> Dump:
    """Read a Run Test dump.

    ``strict`` additionally checks that the component terms sum to the
    reported total, within the dump's own 3-decimal print precision. That
    catches a misread column layout immediately rather than three modules
    later.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows: list[dict[str, float]] = []
    events: list[Event] = []

    for block, banners in _records(text.splitlines()):
        if len(block) < 6:
            continue
        vals1 = _to_floats(block[0])
        if len(vals1) == len(_LINE1_TRANS):
            names1, regime = _LINE1_TRANS, "trans"
        elif len(vals1) == len(_LINE1_SUB):
            names1, regime = _LINE1_SUB, "sub"
        elif len(vals1) == len(_LINE1_SUP):
            names1, regime = _LINE1_SUP, "sup"
        else:
            raise ValueError(
                f"{path}: line 1 has {len(vals1)} fields, expected "
                f"{len(_LINE1_TRANS)} (transonic), {len(_LINE1_SUB)} (subsonic) "
                f"or {len(_LINE1_SUP)} (supersonic):\n  {block[0]!r}"
            )

        row: dict[str, float] = dict(zip(names1, vals1))
        for names, line in (
            (_LINE2, block[1]),
            (_LINE3, block[2]),
            (_LINE4, block[3]),
            (_LINE5, block[4]),
            (_LINE6, block[5]),
        ):
            vals = _to_floats(line)
            if len(vals) != len(names):
                raise ValueError(
                    f"{path}: expected {len(names)} fields, got {len(vals)}:\n  {line!r}"
                )
            for k, v in zip(names, vals):
                # mach/alpha/cn repeat across lines; they must agree.
                if k in row and abs(row[k] - v) > 5e-3:
                    raise ValueError(f"{path}: '{k}' disagrees across lines at Mach {row['mach']}")
                row[k] = v

        row.pop("_dead", None)
        row["regime"] = regime
        rows.append(row)

        for b in banners:
            events.append(Event(mach=row["mach"], text=b))

    if strict:
        _check_sums(path, rows)

    return Dump(rows=rows, events=events)


_TERMS = {
    "sub": ["cd_friction", "cd_form", "cd_base",
            "fin_profile", "fin_interference", "fin_edge", "cd_protuberance"],
    "sup": ["cd_friction", "cd_wave_nose", "cd_base",
            "fin_friction", "fin_wave", "fin_interference", "fin_edge",
            "transition_wave", "cd_protuberance"],
}


def _check_sums(path: str | Path, rows: list[dict[str, float]]) -> None:
    for r in rows:
        if r["regime"] == "trans":
            continue          # interpolated; terms are not re-summed
        terms = _TERMS[r["regime"]]
        total = sum(r[t] for t in terms)
        # Every term is printed to 3dp, so the sum carries up to n/2 * 1e-3.
        tol = 5e-4 * len(terms) + 1e-9
        if abs(total - r["cd_off"]) > tol:
            raise ValueError(
                f"{path}: at Mach {r['mach']:.2f} the {r['regime']} terms sum to "
                f"{total:.4f} but cd_off is {r['cd_off']:.4f}. The column layout "
                f"does not match the dump."
            )


if __name__ == "__main__":  # pragma: no cover
    import sys

    for arg in sys.argv[1:]:
        d = parse_dump(arg)
        print(f"{arg}: {len(d)} Mach points, {len(d.events)} events")
        for e in d.events:
            print(f"    M {e.mach:6.2f}  {e.text}")
