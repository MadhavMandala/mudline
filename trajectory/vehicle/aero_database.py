"""Aerodynamic coefficient database and interpolation helpers.

Interpolation
-------------
Coefficients are interpolated **bilinearly** in (Mach, |alpha|). The previous
implementation returned the nearest tabulated row, which made every coefficient
a piecewise-constant step function of Mach. That has two costs:

* Accuracy. Cd typically doubles across the transonic rise. A nearest lookup
  holds the subsonic value until the midpoint of the Mach interval and then
  jumps, so drag is wrong by most of the step over half of every cell.
* Integrator behaviour. ``solve_ivp`` is an adaptive method that assumes a
  smooth right-hand side. A jump discontinuity in force fails its local error
  estimate, so it repeatedly rejects and shrinks steps at every cell boundary
  it crosses -- paying for accuracy it cannot get.

The table is treated as a rectangular Mach x alpha grid when the rows form one,
which is the normal case for a RASAero export. Ragged tables fall back to
inverse-distance weighting over the nearest neighbours, which is still
continuous. Queries outside the tabulated range are clamped to the edge rather
than extrapolated -- a Cd extrapolated off the end of a transonic table is
worse than a held one.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class AeroCoefficients:
    mach: float
    alpha_deg: float
    cd: float
    cn: float
    cm: float
    x_cp_m: float
    #: Drag with the base filled by the plume, for while the motor burns.
    #: ``None`` when the table carries one drag column, in which case ``cd``
    #: serves the whole flight. Base drag is a large share of total drag on
    #: a blunt-based vehicle, and it largely vanishes behind a plume -- the
    #: same rocket decelerates far harder after burnout than during the burn.
    cd_power_on: float | None = None
    #: Pitch-damping moments of the normal-force slope about the nose tip,
    #: per radian: the sum of CN_alpha over the lifting parts, its first
    #: moment (times station) and its second (times station squared), so
    #: that Cmq about any CG is exact::
    #:
    #:     Cmq = -2 (cna_x2 - 2 x_cg cna_x + x_cg^2 cna_sum) / L_ref^2
    #:
    #: ``None`` when the table cannot say, and the model falls back to the
    #: single-surface estimate from the static margin.
    cna_sum: float | None = None
    cna_x_m: float | None = None
    cna_x2_m2: float | None = None
    #: Roll damping derivative per ``p d / 2V`` and the roll forcing
    #: coefficient from fin cant per ``q S d``, both on the reference
    #: diameter. ``None`` without fins that say; roll is then undamped.
    clp: float | None = None
    cl_roll: float | None = None
    #: Roll forcing per radian of fin cant, per ``q S d``: what one more
    #: degree of cant would add to ``cl_roll``. The simulator applies it to
    #: a cant *offset* -- a build error, dispersed -- without rebuilding
    #: the table. ``None`` when the table cannot say.
    cl_cant: float | None = None


def _optional_float(text) -> float | None:
    """A CSV cell that may be absent or blank."""
    if text is None:
        return None
    text = str(text).strip()
    return float(text) if text else None


class AeroDatabase:
    """Tabular aerodynamic coefficients for the trajectory simulator."""

    def __init__(self, rows: Iterable[AeroCoefficients], reference_length_m: float = 1.0):
        self.rows = list(rows)
        if not self.rows:
            raise ValueError("AeroDatabase requires at least one coefficient row.")
        self.reference_length_m = max(float(reference_length_m), 1e-9)
        # Tables are stored against |alpha|: RASAero exports one side of a
        # symmetric vehicle, and the sign of the normal force is applied
        # downstream from the flow direction, not carried in the coefficient.
        self._points = np.array(
            [[r.mach, abs(r.alpha_deg)] for r in self.rows], dtype=float
        )
        self._values = {
            "cd": np.array([r.cd for r in self.rows], dtype=float),
            "cn": np.array([r.cn for r in self.rows], dtype=float),
            "cm": np.array([r.cm for r in self.rows], dtype=float),
            "x_cp_m": np.array([r.x_cp_m for r in self.rows], dtype=float),
            # Power-on drag, falling back to the power-off value row by row
            # so the grid stays full when only some rows carry it.
            "cd_on": np.array(
                [r.cd if r.cd_power_on is None else r.cd_power_on for r in self.rows],
                dtype=float,
            ),
        }
        for name in ("cna_sum", "cna_x_m", "cna_x2_m2", "clp", "cl_roll", "cl_cant"):
            self._values[name] = np.array(
                [0.0 if getattr(r, name) is None else getattr(r, name) for r in self.rows],
                dtype=float,
            )
        #: Whether the table distinguishes the burning base from the coasting
        #: one. Without it every lookup's ``cd_power_on`` is ``None``.
        self.has_power_on = any(r.cd_power_on is not None for r in self.rows)
        #: Whether the table carries the per-part damping moments and the
        #: roll derivatives; without them the model estimates the one and
        #: leaves the other at zero.
        self.has_damping = any(r.cna_sum is not None for r in self.rows)
        self.has_roll = any(r.clp is not None for r in self.rows)
        self.has_cant = any(r.cl_cant is not None for r in self.rows)
        #: Planform, fins and nose for the model's extension beyond the alpha
        #: range -- a ``HighAlphaGeometry`` set by whoever built the table
        #: from a vehicle. ``None`` means the model assumes a cylinder.
        self.high_alpha = None
        self._build_grid()

        # Whether the table carries usable centre-of-pressure data. Some
        # exports fill x_cp with zeros and describe the pitching moment through
        # cm instead; the aero model needs to know which it has.
        self.has_x_cp = bool(np.any(np.abs(self._values["x_cp_m"]) > 1e-9))
        self.has_cm = bool(np.any(np.abs(self._values["cm"]) > 1e-12))

    # ------------------------------------------------------------------
    # Interpolation structure
    # ------------------------------------------------------------------

    def _build_grid(self) -> None:
        """Detect a rectangular Mach x alpha grid and cache it if present.

        Duplicate (Mach, alpha) rows are averaged rather than rejected. Real
        exports occasionally repeat a point at a layer boundary, and averaging
        keeps such a table usable while still producing a well-defined grid.
        """
        self._machs = np.unique(self._points[:, 0])
        self._alphas = np.unique(self._points[:, 1])
        n_m, n_a = len(self._machs), len(self._alphas)

        # A scattered table with nearly-unique coordinates would allocate an
        # n_m x n_a grid that is almost all holes. Bail out before spending the
        # memory; such a table is not gridded by any useful definition.
        if n_m * n_a > 16 * max(len(self.rows), 1):
            self.is_gridded = False
            self._tables = None
            return

        accum = {k: np.zeros((n_m, n_a)) for k in self._values}
        counts = np.zeros((n_m, n_a))
        mi = np.searchsorted(self._machs, self._points[:, 0])
        ai = np.searchsorted(self._alphas, self._points[:, 1])
        for row, (i, j) in enumerate(zip(mi, ai)):
            counts[i, j] += 1
            for key, values in self._values.items():
                accum[key][i, j] += values[row]

        # A full grid has every cell covered (after averaging duplicates,
        # at least once). A grid with holes is filled from its neighbours
        # rather than abandoned: one missing row of a forty-row export used
        # to demote the whole table to the scattered fallback, whose
        # inverse-distance weights jump when a neighbour drops out of the
        # nearest set -- a 23% step in drag across one Mach step, which is
        # exactly the discontinuity that stalls the adaptive integrator.
        filled = counts > 0
        tables = {key: np.where(filled, accum[key] / np.where(filled, counts, 1.0), np.nan)
                  for key in accum}
        holes = ~filled
        while np.any(holes):
            progress = False
            for i, j in zip(*np.nonzero(holes)):
                neighbours = [
                    (i + di, j + dj) for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    if 0 <= i + di < n_m and 0 <= j + dj < n_a and not holes[i + di, j + dj]
                ]
                if not neighbours:
                    continue
                for key in tables:
                    tables[key][i, j] = float(np.mean([tables[key][a, b] for a, b in neighbours]))
                holes[i, j] = False
                progress = True
            if not progress:
                break
        self.is_gridded = not bool(np.any(holes))
        if not self.is_gridded:
            self._tables = None
            return
        self._tables = tables

    @property
    def mach_range(self) -> tuple[float, float]:
        return float(self._machs[0]), float(self._machs[-1])

    @property
    def alpha_range_deg(self) -> tuple[float, float]:
        return float(self._alphas[0]), float(self._alphas[-1])

    @staticmethod
    def _bracket(axis: np.ndarray, value: float) -> tuple[int, int, float]:
        """Return (lo, hi, weight) bracketing ``value``, clamped to the axis.

        ``weight`` is the fraction of the way from ``axis[lo]`` to ``axis[hi]``.
        A single-point axis degenerates to (0, 0, 0.0), which makes the
        bilinear blend collapse to a 1-D interpolation along the other axis --
        the common case of a table swept in Mach at a single alpha.
        """
        n = len(axis)
        if n == 1:
            return 0, 0, 0.0
        if value <= axis[0]:
            return 0, 0, 0.0
        if value >= axis[-1]:
            return n - 1, n - 1, 0.0
        hi = int(np.searchsorted(axis, value, side="left"))
        hi = max(1, min(hi, n - 1))
        lo = hi - 1
        span = axis[hi] - axis[lo]
        weight = 0.0 if span <= 0 else float((value - axis[lo]) / span)
        return lo, hi, weight

    @classmethod
    def from_csv(cls, path: str | Path, reference_length_m: float = 1.0) -> "AeroDatabase":
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = [
                AeroCoefficients(
                    mach=float(row.get("mach", 0.0)),
                    alpha_deg=float(row.get("alpha_deg", 0.0)),
                    cd=float(row.get("cd", 0.0)),
                    cn=float(row.get("cn", 0.0)),
                    cm=float(row.get("cm", 0.0)),
                    x_cp_m=float(row.get("x_cp_m", 0.0)),
                    cd_power_on=_optional_float(row.get("cd_power_on")),
                    cna_sum=_optional_float(row.get("cna_sum")),
                    cna_x_m=_optional_float(row.get("cna_x_m")),
                    cna_x2_m2=_optional_float(row.get("cna_x2_m2")),
                    clp=_optional_float(row.get("clp")),
                    cl_roll=_optional_float(row.get("cl_roll")),
                    cl_cant=_optional_float(row.get("cl_cant")),
                )
                for row in reader
                if row
            ]
        database = cls(rows, reference_length_m=reference_length_m)
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            first = next(csv.DictReader(f), None)
        if first is not None and _optional_float(first.get("hag_length_m")) is not None:
            # The planform rides along as constant columns; without it a
            # reloaded table flew its high-alpha extension on a finless
            # cylinder and turned a stable rocket divergent at 25 degrees.
            from .aero_model import HighAlphaGeometry

            database.high_alpha = HighAlphaGeometry(
                length_m=float(first["hag_length_m"]),
                diameter_m=float(first["hag_diameter_m"]),
                planform_area_m2=float(first["hag_planform_area_m2"]),
                planform_centroid_m=float(first["hag_planform_centroid_m"]),
                nose_length_m=float(first["hag_nose_length_m"]),
                fin_area_m2=float(first.get("hag_fin_area_m2") or 0.0),
                fin_centroid_m=float(first.get("hag_fin_centroid_m") or 0.0),
            )
        return database

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Optional columns are written only when the table carries them, so
        # a plain table produces the same file it always did.
        fields = ["mach", "alpha_deg", "cd", "cn", "cm", "x_cp_m"]
        if self.has_power_on:
            fields.append("cd_power_on")
        if self.has_damping:
            fields += ["cna_sum", "cna_x_m", "cna_x2_m2"]
        if self.has_roll:
            fields += ["clp", "cl_roll"]
            if self.has_cant:
                fields.append("cl_cant")
        geometry = {}
        if self.high_alpha is not None:
            shape = self.high_alpha
            geometry = {
                "hag_length_m": shape.length_m,
                "hag_diameter_m": shape.diameter_m,
                "hag_planform_area_m2": shape.planform_area_m2,
                "hag_planform_centroid_m": shape.planform_centroid_m,
                "hag_nose_length_m": shape.nose_length_m,
                "hag_fin_area_m2": shape.fin_area_m2,
                "hag_fin_centroid_m": shape.fin_centroid_m,
            }
            fields += list(geometry)
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in self.rows:
                record = {
                    key: ("" if value is None else value)
                    for key, value in row.__dict__.items()
                }
                record.update(geometry)
                writer.writerow(record)
        return path

    def to_metadata(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "mudline.aero_database.v1",
            "reference_length_m": self.reference_length_m,
            "row_count": len(self.rows),
            "mach_range": [float(np.min(self._points[:, 0])), float(np.max(self._points[:, 0]))],
            "alpha_deg_range": [float(np.min(self._points[:, 1])), float(np.max(self._points[:, 1]))],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def lookup(self, mach: float, alpha_deg: float) -> AeroCoefficients:
        """Return coefficients interpolated at (mach, |alpha_deg|).

        Continuous everywhere, including across the edges of the table, where
        the value is held rather than extrapolated.
        """
        mach = float(mach)
        alpha = abs(float(alpha_deg))
        values = (
            self._interpolate_grid(mach, alpha)
            if self.is_gridded
            else self._interpolate_scattered(mach, alpha)
        )
        return AeroCoefficients(
            mach=mach,
            alpha_deg=float(alpha_deg),
            cd=values["cd"],
            cn=values["cn"],
            cm=values["cm"],
            x_cp_m=values["x_cp_m"],
            cd_power_on=values["cd_on"] if self.has_power_on else None,
            cna_sum=values["cna_sum"] if self.has_damping else None,
            cna_x_m=values["cna_x_m"] if self.has_damping else None,
            cna_x2_m2=values["cna_x2_m2"] if self.has_damping else None,
            clp=values["clp"] if self.has_roll else None,
            cl_roll=values["cl_roll"] if self.has_roll else None,
            cl_cant=values["cl_cant"] if self.has_cant else None,
        )

    def _interpolate_grid(self, mach: float, alpha: float) -> dict[str, float]:
        """Bilinear blend of the four cells bracketing the query."""
        i0, i1, wm = self._bracket(self._machs, mach)
        j0, j1, wa = self._bracket(self._alphas, alpha)
        out = {}
        for key, table in self._tables.items():
            low = table[i0, j0] * (1.0 - wa) + table[i0, j1] * wa
            high = table[i1, j0] * (1.0 - wa) + table[i1, j1] * wa
            out[key] = float(low * (1.0 - wm) + high * wm)
        return out

    def cn_alpha_per_rad(self, mach: float, alpha_deg: float, step_deg: float = 1.0) -> float:
        """Local normal-force curve slope dCN/d(alpha) [1/rad].

        Taken as a one-sided difference of the interpolated table rather than a
        central one. The table is stored against |alpha|, so a central
        difference straddling alpha = 0 would sample the same point twice and
        report a slope of exactly zero -- which is where a rocket spends most
        of its flight, and precisely where the damping estimate needs a slope.

        Returns 0.0 for a table with a single alpha station, since no slope is
        recoverable from it. Callers must treat that as "no damping data".
        """
        alpha = abs(float(alpha_deg))
        lo_deg, hi_deg = self.alpha_range_deg
        if hi_deg - lo_deg < 1e-9:
            return 0.0

        step = min(float(step_deg), hi_deg - lo_deg)
        a0 = min(max(alpha, lo_deg), hi_deg - step)
        a1 = a0 + step
        cn0 = self.lookup(mach, a0).cn
        cn1 = self.lookup(mach, a1).cn
        return float((cn1 - cn0) / np.radians(step))

    def _interpolate_scattered(self, mach: float, alpha: float) -> dict[str, float]:
        """Inverse-distance blend over the nearest rows of a ragged table.

        Used only when the export is not a full rectangular grid. Unlike the
        nearest-neighbour lookup this replaces, it is continuous: the weight of
        each neighbour goes to zero smoothly as it leaves the neighbour set.
        """
        scales = np.ptp(self._points, axis=0)
        scales[scales < 1e-9] = 1.0
        query = np.array([mach, alpha])
        distances = np.linalg.norm((self._points - query[None, :]) / scales[None, :], axis=1)

        # Land exactly on a tabulated point and that point is the answer.
        nearest = int(np.argmin(distances))
        if distances[nearest] < 1e-12:
            return {key: float(values[nearest]) for key, values in self._values.items()}

        k = min(4, len(distances))
        idx = np.argpartition(distances, k - 1)[:k]
        weights = 1.0 / distances[idx] ** 2
        weights /= weights.sum()
        return {
            key: float(np.dot(weights, values[idx]))
            for key, values in self._values.items()
        }
