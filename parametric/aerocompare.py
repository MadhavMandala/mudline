"""Put two coefficient tables side by side and say where they disagree.

Both the built-in Barrowman sweep and a RASAero run produce an
``AeroDatabase``, so neither is privileged here: this compares tables, and the
same code works for a CFD table or wind-tunnel data once those exist.

What to read
------------
Drag and normal force are reported as percentage differences, which is the
form the accuracy claims are made in. Centre of pressure is reported in
**calibres**, not percent, because that is the unit the decision is made in: a
tenth of a calibre of CP disagreement is noise, and a full calibre is the
difference between a design that flies and one that does not. The static
margin line is the same comparison against the vehicle's own CG, which is what
a design review actually argues about.

Comparing at matching conditions matters. A table interpolated outside the
Mach or alpha range it was built for will happily return a number, so the
sweep here is clipped to the overlap of the two tables and the clipping is
reported rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Where a rocket's aerodynamics actually changes character. A uniform sweep
#: spends most of its points in the flat supersonic tail and skips the
#: transonic rise, which is where two methods disagree most.
DEFAULT_MACHS = (0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.05, 1.2, 1.5, 2.0, 3.0, 4.0, 5.0)


@dataclass
class Row:
    mach: float
    alpha_deg: float
    cd_a: float
    cd_b: float
    cn_a: float
    cn_b: float
    cp_a: float
    cp_b: float

    @property
    def cd_error_pct(self) -> float:
        return _pct(self.cd_a, self.cd_b)

    @property
    def cn_error_pct(self) -> float:
        return _pct(self.cn_a, self.cn_b)

    def cp_error_cal(self, diameter_m: float) -> float:
        return (self.cp_a - self.cp_b) / max(diameter_m, 1e-9)


@dataclass
class Comparison:
    """``a`` against ``b``; ``b`` is the reference the errors are measured from."""

    name_a: str
    name_b: str
    diameter_m: float
    cg_station_m: float
    rows: list[Row] = field(default_factory=list)
    #: Conditions dropped because one table did not cover them.
    skipped: list[str] = field(default_factory=list)

    def cd_errors(self) -> np.ndarray:
        return np.array([row.cd_error_pct for row in self.rows])

    def cn_errors(self) -> np.ndarray:
        return np.array([row.cn_error_pct for row in self.rows])

    def cp_errors_cal(self) -> np.ndarray:
        return np.array([row.cp_error_cal(self.diameter_m) for row in self.rows])

    def worst(self, values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        return float(max(finite, key=abs)) if finite.size else float("nan")

    def report(self) -> str:
        if not self.rows:
            return "Nothing to compare: the two tables share no conditions."

        alphas = sorted({row.alpha_deg for row in self.rows})
        lines = [
            f"{self.name_a}  vs  {self.name_b}   "
            f"(errors are {self.name_a} relative to {self.name_b})",
            f"  reference diameter {self.diameter_m:.4f} m,  CG {self.cg_station_m:.3f} m",
            "",
        ]

        for alpha in alphas:
            rows = [row for row in self.rows if row.alpha_deg == alpha]
            lines.append(f"  alpha {alpha:g} deg")
            lines.append(
                f"    {'Mach':>6}{'CD':>9}{'CD ref':>9}{'err %':>8}"
                f"{'CN':>9}{'CN ref':>9}{'err %':>8}"
                f"{'CP m':>9}{'CP ref':>9}{'err cal':>9}"
            )
            for row in rows:
                lines.append(
                    f"    {row.mach:6.2f}"
                    f"{row.cd_a:9.4f}{row.cd_b:9.4f}{row.cd_error_pct:+8.1f}"
                    f"{row.cn_a:9.4f}{row.cn_b:9.4f}{row.cn_error_pct:+8.1f}"
                    f"{row.cp_a:9.4f}{row.cp_b:9.4f}"
                    f"{row.cp_error_cal(self.diameter_m):+9.2f}"
                )
            lines.append("")

        cd, cn, cp = self.cd_errors(), self.cn_errors(), self.cp_errors_cal()
        lines.extend([
            "  summary",
            f"    CD    mean |err| {np.nanmean(np.abs(cd)):5.1f}%"
            f"    worst {self.worst(cd):+6.1f}%",
            f"    CN    mean |err| {np.nanmean(np.abs(cn)):5.1f}%"
            f"    worst {self.worst(cn):+6.1f}%",
            f"    CP    mean |err| {np.nanmean(np.abs(cp)):5.2f} cal"
            f"  worst {self.worst(cp):+6.2f} cal",
        ])

        margins = self.static_margins()
        if margins:
            lines.append("")
            lines.append("  static margin [calibres], measured about the CG above")
            lines.append(
                f"    {'Mach':>6}{self.name_a[:10]:>12}{self.name_b[:10]:>12}{'delta':>9}"
            )
            for mach, margin_a, margin_b in margins:
                lines.append(
                    f"    {mach:6.2f}{margin_a:12.2f}{margin_b:12.2f}"
                    f"{margin_a - margin_b:+9.2f}"
                )

        if self.skipped:
            lines.append("")
            lines.append(f"  skipped: {', '.join(self.skipped)}")
        return "\n".join(lines)

    def static_margins(self) -> list[tuple[float, float, float]]:
        """(Mach, margin_a, margin_b) at the highest alpha compared.

        The highest alpha, because that is where a marginal vehicle gets into
        trouble -- CP moves as alpha grows, and a margin quoted at zero alpha
        is the most flattering number available.
        """
        if not self.rows:
            return []
        alpha = max(row.alpha_deg for row in self.rows)
        return [
            (
                row.mach,
                (row.cp_a - self.cg_station_m) / self.diameter_m,
                (row.cp_b - self.cg_station_m) / self.diameter_m,
            )
            for row in self.rows if row.alpha_deg == alpha
        ]


def compare(
    database_a,
    database_b,
    diameter_m: float,
    cg_station_m: float,
    name_a: str = "built-in",
    name_b: str = "RASAero",
    machs=DEFAULT_MACHS,
    alphas=(0.0, 4.0),
) -> Comparison:
    """Sample both tables at the same conditions and difference them."""
    mach_lo = max(database_a.mach_range[0], database_b.mach_range[0])
    mach_hi = min(database_a.mach_range[1], database_b.mach_range[1])
    alpha_hi = min(_alpha_ceiling(database_a), _alpha_ceiling(database_b))

    result = Comparison(name_a, name_b, diameter_m, cg_station_m)

    dropped_mach = [m for m in machs if not mach_lo <= m <= mach_hi]
    if dropped_mach:
        result.skipped.append(
            f"Mach {', '.join(f'{m:g}' for m in dropped_mach)} "
            f"(outside the shared range {mach_lo:.2f}-{mach_hi:.2f})"
        )
    dropped_alpha = [a for a in alphas if a > alpha_hi + 1e-9]
    if dropped_alpha:
        result.skipped.append(
            f"alpha {', '.join(f'{a:g}' for a in dropped_alpha)} deg "
            f"(one table stops at {alpha_hi:g})"
        )

    for alpha in alphas:
        if alpha > alpha_hi + 1e-9:
            continue
        for mach in machs:
            if not mach_lo <= mach <= mach_hi:
                continue
            a = database_a.lookup(mach, alpha)
            b = database_b.lookup(mach, alpha)
            result.rows.append(Row(
                mach=mach, alpha_deg=alpha,
                cd_a=a.cd, cd_b=b.cd,
                cn_a=a.cn, cn_b=b.cn,
                cp_a=a.x_cp_m, cp_b=b.x_cp_m,
            ))
    return result


def _alpha_ceiling(database) -> float:
    return max(row.alpha_deg for row in database.rows)


def _pct(value: float, reference: float) -> float:
    if abs(reference) < 1e-12:
        return float("nan")
    return 100.0 * (value - reference) / reference
