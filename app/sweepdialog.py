"""Sweep setup, and the result it produces.

The form mirrors the other analyses: choose what to vary, over what range, and
how much of the chain to run at each point. Flight is optional because it costs
seconds per point while geometry and aerodynamics cost a fraction of one, and
mass, drag and static margin already answer most shape questions.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QLabel,
    QSpinBox,
)

from app.analysisdialogs import SetupDialog, _spin


class SweepSetupDialog(SetupDialog):
    """Pick a parameter and a range."""

    def __init__(self, model, has_aero: bool = False,
                 couples_aero: bool = False, parent=None):
        super().__init__(
            "Design Sweep",
            "Varies one parameter and re-runs the analysis at each value, so a "
            "trade can be read off a curve instead of assembled by hand. The "
            "model is restored when the sweep finishes.",
            parent,
        )
        from parametric.sweep import default_range, sweepable_parms

        self._model = model
        self._couples_aero = couples_aero
        self._options = sweepable_parms(model)

        variable = self.add_group("Variable")
        self.parameter = QComboBox()
        for path, parm, label in self._options:
            self.parameter.addItem(label, (path, parm))
        self.parameter.currentIndexChanged.connect(self._on_parameter)
        variable.addRow("Parameter", self.parameter)

        self.start = _spin(-1e6, 1e6, 0.0, 0.005, "", 5)
        self.stop = _spin(-1e6, 1e6, 1.0, 0.005, "", 5)
        self.steps = QSpinBox()
        self.steps.setRange(2, 60)
        self.steps.setValue(9)
        variable.addRow("From", self.start)
        variable.addRow("To", self.stop)
        variable.addRow("Points", self.steps)

        self._current = QLabel()
        self._current.setStyleSheet("color:#5b6675; font-size:11px;")
        variable.addRow("", self._current)

        depth = self.add_group("Run at each point")
        self.include_aero = QCheckBox("Aerodynamics (drag and static margin)")
        self.include_aero.setChecked(True)
        self.include_flight = QCheckBox("Flight (apogee, max-Q, rail exit)")
        self.include_flight.setChecked(True)
        self.include_flight.toggled.connect(self._update_estimate)
        self.steps.valueChanged.connect(self._update_estimate)
        depth.addRow("", self.include_aero)
        depth.addRow("", self.include_flight)

        self._estimate = QLabel()
        self._estimate.setStyleSheet("color:#5b6675; font-size:11px;")
        depth.addRow("", self._estimate)

        self._on_parameter(0)
        self._update_estimate()
        self.finish()

    # ------------------------------------------------------------------

    def _on_parameter(self, _index: int) -> None:
        from parametric.sweep import default_range

        data = self.parameter.currentData()
        if not data:
            return
        path, parm_name = data
        low, high = default_range(self._model, path, parm_name)
        self.start.setValue(low)
        self.stop.setValue(high)

        component = next(c for c in self._model.walk() if c.path == path)
        parm = component.parm(parm_name)
        self._current.setText(
            f"currently {parm.format()}   allowed "
            f"{parm.minimum:g} to {parm.maximum:g}"
        )

    def _update_estimate(self) -> None:
        per_point = 5.0 if self.include_flight.isChecked() else 0.6
        if self.include_flight.isChecked() and self._couples_aero:
            # Each flight re-flies on a rebuilt table up to this many more
            # times, exactly as Run Flight does with the same settings.
            from parametric.flight import COUPLED_PASSES

            per_point *= 1 + COUPLED_PASSES
        seconds = per_point * self.steps.value()
        self._estimate.setText(
            f"roughly {seconds:.0f} s for {self.steps.value()} points"
            + ("  — cancellable" if seconds > 20 else "")
        )

    def settings(self):
        from parametric.sweep import SweepSettings, SweepVariable

        path, parm_name = self.parameter.currentData()
        return SweepSettings(
            variable=SweepVariable(
                component_path=path, parm=parm_name,
                start=self.start.value(), stop=self.stop.value(),
                steps=self.steps.value(),
            ),
            include_aero=self.include_aero.isChecked(),
            include_flight=self.include_flight.isChecked(),
        )


def sweep_result(result, fingerprint: str) -> dict:
    """Turn a SweepResult into a stored Result."""
    from app.results import Metric

    names = result.metric_names()
    values = result.values
    label = result.variable.label

    metrics: list[Metric] = []
    apogee = result.best("apogee m", maximise=True)
    if apogee is not None:
        metrics.append(Metric(f"Best apogee at {label}", apogee.value,
                              result.unit, 4))
        metrics.append(Metric("Best apogee", apogee.metrics["apogee m"], "m", 0,
                              higher_is_better=True))

    margins = result.series("static margin cal")
    if np.any(np.isfinite(margins)):
        stable = values[np.isfinite(margins) & (margins >= 1.0)]
        metrics.append(Metric("Points swept", float(len(values)), "", 0))
        if len(stable):
            metrics.append(Metric(f"Min {label} for 1 cal", float(stable.min()),
                                  result.unit, 4))
        metrics.append(Metric("Lowest margin", float(np.nanmin(margins)), "cal", 2,
                              higher_is_better=True))

    failures = sum(1 for p in result.points if p.failed)
    if failures:
        metrics.append(Metric("Failed points", float(failures), "", 0))

    series = {
        f"{name} vs {label}": (
            values, result.series(name), f"{label} {result.unit}".strip(), name
        )
        for name in names
    }

    return dict(
        kind="sweep",
        label=f"Sweep {label}",
        fingerprint=fingerprint,
        settings={
            "parameter": label,
            "range": f"{result.variable.start:g} – {result.variable.stop:g} "
                     f"{result.unit}".strip(),
            "points": str(len(values)),
        },
        metrics=metrics,
        series=series,
        payload={"sweep": result},
    )
