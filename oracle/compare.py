"""Compare the native engine against a RASAero Run Test dump, term by term.

Tolerance
---------
RASAero prints each column through a fixed format string, so the reference
value is *already rounded* before we ever see it. The achievable agreement is
therefore bounded by that rounding and nothing else -- and the bound differs
by column, which matters: CN-alpha is printed to two decimals (``" ##0.00"``,
i.cs:803) while every drag term gets three. Using one tolerance for all of
them either lets a real CN-alpha error through or fails a correct one.

The bound is half an ulp of the printed value, plus a small relative term for
float-versus-decimal accumulation. RASAero computes in .NET ``decimal`` --
base-10, 28 significant digits -- and converts to ``double`` for every
transcendental, so a long buildup can differ from float64 in the last handful
of digits. On a centre-of-pressure station of 127 inches that is worth a few
parts in 10^7, which is invisible in absolute terms but sits just outside a
pure half-ulp bound.

Anything outside this envelope is a real disagreement and should be treated
as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Decimal places each column is printed with, from the format strings in
#: ``i.cs:800-807`` (subsonic), ``817-822`` (transonic) and ``833-840``
#: (supersonic).
PRINTED_DECIMALS: dict[str, int] = {
    "mach": 2,
    "alpha_deg": 2,
    "reynolds": 0,
    "cn_alpha_0to4": 2,      # the odd one out
    # everything else is three
}
DEFAULT_DECIMALS = 3

#: Relative allowance for decimal-vs-float accumulation. Deliberately tiny:
#: it exists to cover last-digit drift on large stations, not to paper over
#: a wrong formula.
RELATIVE_SLACK = 2.0e-6

TERMS_BY_REGIME: dict[str, list[str]] = {
    "sub": [
        "cd_friction", "cd_form", "cd_base",
        "fin_profile", "fin_interference", "fin_edge", "cd_protuberance",
    ],
    "sup": [
        "cd_friction", "cd_wave_nose", "cd_base",
        "fin_friction", "fin_wave", "fin_interference", "fin_edge",
        "transition_wave", "cd_protuberance",
    ],
    "trans": [],
}

ASSEMBLED = [
    "cd_off", "cd_on", "ca_off", "ca_on", "cl_off", "cl_on",
    "cn", "cn_potential", "cn_viscous", "cp",
    "cn_alpha_0to4", "cp_0to4", "reynolds",
]


def tolerance(field_name: str, reference: float) -> float:
    decimals = PRINTED_DECIMALS.get(field_name, DEFAULT_DECIMALS)
    half_ulp = 0.5 * 10.0 ** (-decimals)
    return half_ulp + RELATIVE_SLACK * abs(reference)


@dataclass
class Mismatch:
    mach: float
    regime: str
    field: str
    mine: float
    reference: float
    diff: float
    allowed: float

    def __str__(self) -> str:
        return (
            f"M {self.mach:6.2f} {self.regime:<5} {self.field:<18} "
            f"mine={self.mine:12.6f} ras={self.reference:12.6f} "
            f"diff={self.diff:.6f} allowed={self.allowed:.6f}"
        )


@dataclass
class Report:
    case: str
    comparisons: int = 0
    mismatches: list[Mismatch] = field(default_factory=list)
    #: field -> (largest diff seen, Mach it occurred at, allowance there)
    worst: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        head = (
            f"{self.case}: {self.comparisons} comparisons, "
            f"{len(self.mismatches)} outside tolerance"
        )
        if self.passed:
            ratio, name, mach = 0.0, "-", 0.0
            for f, (diff, m, allowed) in self.worst.items():
                r = diff / allowed if allowed else 0.0
                if r > ratio:
                    ratio, name, mach = r, f, m
            head += f"  (tightest: {name} at M {mach:.2f}, {ratio:.0%} of allowance)"
        return head


def compare(engine, dump, *, alpha_deg: float = 0.0, case: str = "") -> Report:
    """Solve every Mach point in ``dump`` and diff against it.

    ``engine`` is an ``aeroengine.solver.Engine``; ``dump`` is a parsed
    ``oracle.runtest.Dump``. The dump's own Mach grid drives the sweep, so
    the two are compared at exactly the conditions RASAero evaluated.
    """
    report = Report(case=case or "case")
    for row in dump.rows:
        result = engine.solve(row["mach"], alpha_deg)
        names = TERMS_BY_REGIME[row["regime"]] + ASSEMBLED
        for name in names:
            if name not in row:
                continue
            mine = getattr(result, name, None)
            if mine is None:
                continue
            reference = row[name]
            diff = abs(mine - reference)
            allowed = tolerance(name, reference)
            report.comparisons += 1

            previous = report.worst.get(name)
            if previous is None or diff > previous[0]:
                report.worst[name] = (diff, row["mach"], allowed)

            if diff > allowed:
                report.mismatches.append(Mismatch(
                    mach=row["mach"], regime=row["regime"], field=name,
                    mine=mine, reference=reference, diff=diff, allowed=allowed,
                ))
    return report
