"""Thrust curve editor.

The curve is the most consequential thing about a motor -- burn time and peak
thrust decide rail exit velocity, max-Q and apogee between them -- so it gets a
plot and a table rather than a file path.

The plot is drawn with QPainter rather than pulled from a charting library:
it needs to be small, live under a slider drag, and match the panel's styling,
and none of that is worth a dependency.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from parametric.components import Motor


class ThrustCurvePlot(QWidget):
    """A small thrust-versus-time plot."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._motor: Motor | None = None
        self.setMinimumHeight(130)

    def set_motor(self, motor: Motor | None) -> None:
        self._motor = motor
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#20262e"))

        motor = self._motor
        if motor is None or len(motor.curve) < 2:
            painter.setPen(QColor("#6f7a88"))
            painter.drawText(self.rect(), Qt.AlignCenter, "no thrust curve")
            painter.end()
            return

        times = np.array([t for t, _ in motor.curve], dtype=float)
        thrusts = np.array([f for _, f in motor.curve], dtype=float)
        t_max = max(times.max(), 1e-6)
        f_max = max(thrusts.max(), 1e-6)

        margin_left, margin_bottom, margin_top = 44, 20, 10
        width = max(self.width() - margin_left - 10, 10)
        height = max(self.height() - margin_bottom - margin_top, 10)

        def to_px(t: float, f: float) -> QPointF:
            return QPointF(
                margin_left + width * (t / t_max),
                margin_top + height * (1.0 - f / f_max),
            )

        # Axes and a couple of gridlines, so the numbers can be read off.
        painter.setPen(QPen(QColor("#39424e"), 1))
        for fraction in (0.0, 0.5, 1.0):
            y = margin_top + height * (1.0 - fraction)
            painter.drawLine(margin_left, int(y), margin_left + width, int(y))

        painter.setFont(QFont("Segoe UI", 7))
        painter.setPen(QColor("#8a94a2"))
        for fraction in (0.5, 1.0):
            y = margin_top + height * (1.0 - fraction)
            painter.drawText(2, int(y) + 4, f"{f_max * fraction:6.0f}")
        painter.drawText(margin_left, self.height() - 5, "0")
        painter.drawText(
            margin_left + width - 30, self.height() - 5, f"{t_max:.2f}s"
        )

        # The curve, filled underneath so total impulse reads as an area.
        path = QPainterPath(to_px(times[0], thrusts[0]))
        for t, f in zip(times[1:], thrusts[1:]):
            path.lineTo(to_px(t, f))

        filled = QPainterPath(path)
        filled.lineTo(to_px(times[-1], 0.0))
        filled.lineTo(to_px(times[0], 0.0))
        filled.closeSubpath()
        painter.fillPath(filled, QColor(255, 141, 46, 55))

        painter.setPen(QPen(QColor("#ff8d2e"), 2))
        painter.drawPath(path)

        painter.setPen(QPen(QColor("#ffd0a0"), 1))
        for t, f in zip(times, thrusts):
            point = to_px(t, f)
            painter.drawEllipse(point, 2.5, 2.5)
        painter.end()


class MotorEditor(QWidget):
    """Thrust curve table, plot and derived performance for a Motor."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._motor: Motor | None = None
        self._guard = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(6)

        heading = QLabel("Thrust curve")
        heading.setStyleSheet("font-weight:600; color:#2b3440;")
        layout.addWidget(heading)

        self.plot = ThrustCurvePlot()
        layout.addWidget(self.plot)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Time s", "Thrust N"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMaximumHeight(150)
        self.table.setStyleSheet("font-family:Consolas,monospace; font-size:11px;")
        for column in (0, 1):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.Stretch
            )
        self.table.currentCellChanged.connect(lambda *_: self._sync_detail())
        layout.addWidget(self.table)

        detail = QHBoxLayout()
        detail.setSpacing(4)
        self._time = self._spin(0.0, 1e4, " s", 0.01)
        self._time.valueChanged.connect(lambda v: self._edit(0, v))
        self._thrust = self._spin(0.0, 1e7, " N", 10.0)
        self._thrust.valueChanged.connect(lambda v: self._edit(1, v))
        detail.addWidget(QLabel("t"))
        detail.addWidget(self._time)
        detail.addWidget(QLabel("F"))
        detail.addWidget(self._thrust)
        layout.addLayout(detail)

        # Three to a row. Five buttons side by side wanted about 390 px, and a
        # minimum that large propagates up and sets the width of the whole
        # property panel.
        from PySide6.QtWidgets import QGridLayout

        buttons = QGridLayout()
        buttons.setSpacing(4)
        for index, (text, slot) in enumerate([
            ("Add", self._add_point),
            ("Remove", self._remove_point),
            ("Flat...", self._make_flat),
            ("Import...", self._import_curve),
            ("Load...", self._load_file),
        ]):
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button, *divmod(index, 3))
        layout.addLayout(buttons)

        from PySide6.QtWidgets import QComboBox

        from trajectory.vehicle.mass_properties import END_BURNER, LIQUID, RADIAL

        burn_row = QHBoxLayout()
        burn_row.setSpacing(4)
        burn_row.addWidget(QLabel("Burns"))
        self._burn = QComboBox()
        # The label says what it does to the centroid, since that is the only
        # thing the choice changes.
        for value, label in (
            (RADIAL, "radially (solid core burner)"),
            (LIQUID, "draining aft (liquid tank)"),
            (END_BURNER, "from the aft face (end burner)"),
        ):
            self._burn.addItem(label, value)
        self._burn.currentIndexChanged.connect(self._on_burn_changed)
        burn_row.addWidget(self._burn, 1)
        layout.addLayout(burn_row)

        self._derived = QLabel()
        self._derived.setStyleSheet(
            "font-family:Consolas,monospace; font-size:11px; color:#3d4854;"
        )
        layout.addWidget(self._derived)

        self._warning = QLabel()
        self._warning.setWordWrap(True)
        self._warning.setStyleSheet("color:#b4520f; font-size:11px;")
        layout.addWidget(self._warning)

    @staticmethod
    def _spin(minimum, maximum, suffix, step) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        return spin

    # ------------------------------------------------------------------

    def set_motor(self, motor: Motor | None) -> None:
        self._motor = motor
        self.setVisible(motor is not None)
        self.refresh()

    def refresh(self) -> None:
        motor = self._motor
        self.plot.set_motor(motor)
        if motor is None:
            self.table.setRowCount(0)
            return

        self._guard = True
        self.table.setRowCount(len(motor.curve))
        for row, (time_s, thrust) in enumerate(motor.curve):
            for column, value in enumerate((f"{time_s:.3f}", f"{thrust:.1f}")):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, column, item)
        self._guard = False

        index = self._burn.findData(motor.burn_geometry)
        if index >= 0:
            self._guard = True
            self._burn.setCurrentIndex(index)
            self._guard = False

        # A tank-fed engine carries no propellant of its own, so the curve
        # implies nothing about Isp until the tanks are counted -- and the
        # editor does not see the tanks. Say so rather than print a zero.
        implied = (
            "from tanks" if getattr(motor, "feed", "grain") == "tanks"
            else f"{motor.effective_isp_s:9.1f} s"
        )
        self._derived.setText(
            f"class            {motor.impulse_class}\n"
            f"total impulse  {motor.total_impulse_ns:9.0f} N·s\n"
            f"burn time      {motor.burn_time_s:9.2f} s\n"
            f"peak thrust    {motor.peak_thrust_n:9.0f} N\n"
            f"average        {motor.average_thrust_n:9.0f} N\n"
            f"implied Isp    {implied}\n"
            f"exit area      {motor.effective_nozzle_area_m2() * 1e4:9.1f} cm²"
        )

        # The over-specification check: curve, propellant and Isp are three
        # numbers where two would do, so they can disagree.
        error = motor.isp_consistency
        if error > 0.05:
            self._warning.setText(
                f"⚠  The curve and propellant load imply "
                f"{motor.effective_isp_s:.0f} s, but Isp is declared as "
                f"{motor.get('isp_vac'):.0f} s — {error * 100:.0f}% apart. "
                f"One of the curve, the propellant mass or the Isp is wrong."
            )
        else:
            self._warning.setText("")
        self._sync_detail()

    # ------------------------------------------------------------------

    def _row(self) -> int:
        return self.table.currentRow()

    def _sync_detail(self) -> None:
        motor = self._motor
        row = self._row()
        enabled = motor is not None and 0 <= row < len(motor.curve)
        self._time.setEnabled(enabled)
        self._thrust.setEnabled(enabled)
        if not enabled:
            return
        self._guard = True
        self._time.setValue(motor.curve[row][0])
        self._thrust.setValue(motor.curve[row][1])
        self._guard = False

    def _edit(self, column: int, value: float) -> None:
        if self._guard or self._motor is None:
            return
        row = self._row()
        if not (0 <= row < len(self._motor.curve)):
            return
        time_s, thrust = self._motor.curve[row]
        point = (value, thrust) if column == 0 else (time_s, value)
        curve = list(self._motor.curve)
        curve[row] = point
        self._motor.set_curve(curve)
        self._after_change(row)

    def _on_burn_changed(self, index: int) -> None:
        """Change how the propellant is consumed.

        Geometry only -- the thrust curve and the propellant load are what they
        were. What moves is where the *remaining* propellant sits, and with it
        the CG and inertia through the burn.
        """
        if self._guard or self._motor is None or index < 0:
            return
        self._motor.burn_geometry = self._burn.itemData(index)
        self._motor.mark_dirty("burn_geometry")
        self._after_change()

    def _after_change(self, row: int = 0) -> None:
        self.refresh()
        if 0 <= row < self.table.rowCount():
            self.table.setCurrentCell(row, 0)
        self.changed.emit()

    def _add_point(self) -> None:
        if self._motor is None:
            return
        curve = self._motor.curve
        if curve:
            last_t, last_f = curve[-1]
            self._motor.add_curve_point(last_t + 0.25, last_f)
        else:
            self._motor.add_curve_point(0.0, 0.0)
        self._after_change(len(self._motor.curve) - 1)

    def _remove_point(self) -> None:
        if self._motor is None or len(self._motor.curve) <= 2:
            return          # an engine needs two points
        self._motor.remove_curve_point(self._row())
        self._after_change(max(self._row() - 1, 0))

    def _make_flat(self) -> None:
        if self._motor is None:
            return
        impulse, ok = QInputDialog.getDouble(
            self, "Flat curve", "Total impulse (N·s):",
            max(self._motor.total_impulse_ns, 1000.0), 1.0, 1e7, 0
        )
        if not ok:
            return
        burn, ok = QInputDialog.getDouble(
            self, "Flat curve", "Burn time (s):",
            max(self._motor.burn_time_s, 3.0), 0.05, 600.0, 2
        )
        if not ok:
            return
        self._motor.curve_from_impulse(impulse, burn)
        self._after_change()

    def _import_curve(self) -> None:
        """Import a published motor file, letting the user pick from a bundle."""
        from PySide6.QtWidgets import QFileDialog, QInputDialog, QMessageBox

        from trajectory.propulsion.thrustcurve import load_thrust_curves

        if self._motor is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import motor", "vehicles",
            "Motor files (*.eng *.rse);;RASP (*.eng);;RockSim (*.rse);;"
            "All files (*)",
        )
        if not path:
            return

        try:
            motors = load_thrust_curves(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Import", f"Could not read:\n{exc}")
            return

        index = 0
        if len(motors) > 1:
            # ThrustCurve bundles are whole catalogues, so importing the first
            # motor silently would almost always be the wrong one.
            labels = [m.summary() for m in motors]
            choice, ok = QInputDialog.getItem(
                self, "Import motor",
                f"{len(motors)} motors in this file:", labels, 0, False,
            )
            if not ok:
                return
            index = labels.index(choice)

        try:
            curve = self._motor.import_thrust_curve(path, index)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Import", f"Could not import:\n{exc}")
            return

        self._after_change()
        window = self.window()
        if hasattr(window, "statusBar"):
            window.statusBar().showMessage(f"Imported {curve.summary()}", 8000)

    def _load_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        if self._motor is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load propulsion model", "vehicles",
            "Propulsion model (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self._motor.load_propulsion_file(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Load", f"Could not load:\n{exc}")
            return
        self._after_change()
