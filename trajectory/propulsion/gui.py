"""PySide propulsion model editor."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from trajectory.propulsion.model import (
    PropulsionModel,
    default_propulsion_model,
    load_propulsion_model,
    save_propulsion_model,
)


SYSTEM_TYPES = [
    "Solid Rocket Motor",
    "Liquid Bipropellant",
    "Liquid Monopropellant",
    "Hybrid Rocket",
    "Cold Gas",
    "Electric",
    "Other",
]


class PropulsionMainWindow(QMainWindow):
    def __init__(
        self,
        model_path: Path | None = None,
        propulsion_handoff_path: Path | None = None,
    ):
        super().__init__()
        self.setWindowTitle("Propulsion Model")
        self.resize(980, 760)
        self._propulsion_handoff_path = (
            Path(propulsion_handoff_path) if propulsion_handoff_path else None
        )
        self._current_path: Path | None = None
        self._build_ui()

        if model_path and model_path.exists():
            self._load_model(model_path)
        else:
            self._set_model(default_propulsion_model())

    def _build_ui(self) -> None:
        toolbar = QToolBar(self)
        self.addToolBar(toolbar)

        self._save_action = toolbar.addAction("Save Propulsion")
        self._save_action.triggered.connect(self._on_save)
        self._load_action = toolbar.addAction("Open Propulsion")
        self._load_action.triggered.connect(self._on_open)
        self._flat_curve_action = toolbar.addAction("Generate Flat Curve")
        self._flat_curve_action.triggered.connect(self._on_generate_flat_curve)

        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)
        self.setCentralWidget(root)

        top_layout = QHBoxLayout()
        root_layout.addLayout(top_layout)

        identity = QGroupBox("System")
        identity_form = QFormLayout(identity)
        self.name_edit = QLineEdit()
        self.system_combo = QComboBox()
        self.system_combo.addItems(SYSTEM_TYPES)
        self.fuel_edit = QLineEdit()
        self.oxidizer_edit = QLineEdit()
        self.pressurant_edit = QLineEdit()
        identity_form.addRow("Name", self.name_edit)
        identity_form.addRow("Type", self.system_combo)
        identity_form.addRow("Fuel / Propellant", self.fuel_edit)
        identity_form.addRow("Oxidizer", self.oxidizer_edit)
        identity_form.addRow("Pressurant", self.pressurant_edit)
        top_layout.addWidget(identity, 1)

        performance = QGroupBox("Performance")
        performance_form = QFormLayout(performance)
        self.prop_mass_spin = self._spin(0.0, 1_000_000.0, 3, " kg")
        self.dry_mass_spin = self._spin(0.0, 1_000_000.0, 3, " kg")
        self.isp_sl_spin = self._spin(1.0, 10_000.0, 2, " s")
        self.isp_vac_spin = self._spin(1.0, 10_000.0, 2, " s")
        self.nozzle_area_spin = self._spin(0.0, 1000.0, 6, " m2")
        self.mixture_ratio_spin = self._spin(0.0, 1000.0, 4, "")
        self.chamber_pressure_spin = self._spin(0.0, 10_000.0, 4, " MPa")
        self.exit_pressure_spin = self._spin(0.0, 10_000.0, 4, " kPa")
        self.max_gimbal_spin = self._spin(0.0, 90.0, 2, " deg")
        performance_form.addRow("Propellant Mass", self.prop_mass_spin)
        performance_form.addRow("Engine Dry Mass", self.dry_mass_spin)
        performance_form.addRow("Isp SL", self.isp_sl_spin)
        performance_form.addRow("Isp Vacuum", self.isp_vac_spin)
        performance_form.addRow("Nozzle Area", self.nozzle_area_spin)
        performance_form.addRow("Mixture Ratio O/F", self.mixture_ratio_spin)
        performance_form.addRow("Chamber Pressure", self.chamber_pressure_spin)
        performance_form.addRow("Exit Pressure", self.exit_pressure_spin)
        performance_form.addRow("Max Gimbal", self.max_gimbal_spin)
        top_layout.addWidget(performance, 1)

        geometry = QGroupBox("Geometry / Alignment")
        geometry_grid = QGridLayout(geometry)
        self.tank_cg_spins = self._vector_spins(" m")
        self.thrust_axis_spins = self._vector_spins("")
        self.thrust_pos_spins = self._vector_spins(" m")
        self._add_vector_row(geometry_grid, 0, "Tank CG", self.tank_cg_spins)
        self._add_vector_row(geometry_grid, 1, "Thrust Axis Body", self.thrust_axis_spins)
        self._add_vector_row(geometry_grid, 2, "Thrust Position", self.thrust_pos_spins)
        root_layout.addWidget(geometry)

        curve_box = QGroupBox("Thrust Curve")
        curve_layout = QVBoxLayout(curve_box)
        self.curve_table = QTableWidget(0, 2)
        self.curve_table.setHorizontalHeaderLabels(["Time (s)", "Thrust (N)"])
        self.curve_table.horizontalHeader().setStretchLastSection(True)
        self.curve_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.curve_table.setAlternatingRowColors(True)
        curve_layout.addWidget(self.curve_table)

        curve_buttons = QHBoxLayout()
        add_btn = QPushButton("Add Point")
        remove_btn = QPushButton("Remove Selected")
        sort_btn = QPushButton("Sort")
        add_btn.clicked.connect(lambda: self._add_curve_row())
        remove_btn.clicked.connect(self._remove_selected_curve_rows)
        sort_btn.clicked.connect(self._sort_curve_table)
        curve_buttons.addWidget(add_btn)
        curve_buttons.addWidget(remove_btn)
        curve_buttons.addWidget(sort_btn)
        curve_buttons.addStretch(1)
        curve_layout.addLayout(curve_buttons)
        root_layout.addWidget(curve_box, 1)

        notes_box = QGroupBox("Notes")
        notes_layout = QVBoxLayout(notes_box)
        self.notes_edit = QTextEdit()
        self.notes_edit.setMaximumHeight(90)
        notes_layout.addWidget(self.notes_edit)
        root_layout.addWidget(notes_box)

        self.summary_label = QLabel()
        self.summary_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root_layout.addWidget(self.summary_label)

    def _spin(self, minimum: float, maximum: float, decimals: int, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setSuffix(suffix)
        spin.setSingleStep(1.0)
        return spin

    def _vector_spins(self, suffix: str) -> list[QDoubleSpinBox]:
        spins = []
        for _ in range(3):
            spin = self._spin(-1_000_000.0, 1_000_000.0, 5, suffix)
            spin.setSingleStep(0.1)
            spins.append(spin)
        return spins

    def _add_vector_row(
        self,
        grid: QGridLayout,
        row: int,
        label: str,
        spins: list[QDoubleSpinBox],
    ) -> None:
        grid.addWidget(QLabel(label), row, 0)
        for col, (axis, spin) in enumerate(zip(["X", "Y", "Z"], spins), start=1):
            holder = QVBoxLayout()
            axis_label = QLabel(axis)
            axis_label.setAlignment(Qt.AlignCenter)
            holder.addWidget(axis_label)
            holder.addWidget(spin)
            widget = QWidget()
            widget.setLayout(holder)
            grid.addWidget(widget, row, col)

    def _set_model(self, model: PropulsionModel) -> None:
        self.name_edit.setText(model.name)
        self.system_combo.setCurrentText(model.system_type)
        self.fuel_edit.setText(model.fuel)
        self.oxidizer_edit.setText(model.oxidizer)
        self.pressurant_edit.setText(model.pressurant)
        self.prop_mass_spin.setValue(model.propellant_mass_kg)
        self.dry_mass_spin.setValue(model.dry_mass_kg)
        self.isp_sl_spin.setValue(model.isp_sl_s)
        self.isp_vac_spin.setValue(model.isp_vac_s)
        self.nozzle_area_spin.setValue(model.nozzle_area_m2)
        self.mixture_ratio_spin.setValue(model.mixture_ratio)
        self.chamber_pressure_spin.setValue(model.chamber_pressure_pa / 1_000_000.0)
        self.exit_pressure_spin.setValue(model.exit_pressure_pa / 1000.0)
        self.max_gimbal_spin.setValue(model.max_gimbal_deg)
        self._set_vector(self.tank_cg_spins, model.tank_cg_m)
        self._set_vector(self.thrust_axis_spins, model.thrust_axis_body)
        self._set_vector(self.thrust_pos_spins, model.thrust_position_body_m)
        self.notes_edit.setPlainText(model.notes)
        self._set_curve(model.time_s, model.thrust_n)
        self._update_summary(model)

    def _model_from_ui(self) -> PropulsionModel:
        time_s, thrust_n = self._curve_arrays()
        order = np.argsort(time_s)
        time_s = time_s[order]
        thrust_n = thrust_n[order]
        return PropulsionModel(
            name=self.name_edit.text().strip() or "Untitled Propulsion",
            system_type=self.system_combo.currentText(),
            time_s=time_s,
            thrust_n=thrust_n,
            isp_vac_s=self.isp_vac_spin.value(),
            isp_sl_s=self.isp_sl_spin.value(),
            nozzle_area_m2=self.nozzle_area_spin.value(),
            propellant_mass_kg=self.prop_mass_spin.value(),
            dry_mass_kg=self.dry_mass_spin.value(),
            fuel=self.fuel_edit.text().strip(),
            oxidizer=self.oxidizer_edit.text().strip(),
            pressurant=self.pressurant_edit.text().strip(),
            mixture_ratio=self.mixture_ratio_spin.value(),
            chamber_pressure_pa=self.chamber_pressure_spin.value() * 1_000_000.0,
            exit_pressure_pa=self.exit_pressure_spin.value() * 1000.0,
            tank_cg_m=self._vector(self.tank_cg_spins),
            thrust_axis_body=self._normalized_vector(self.thrust_axis_spins),
            thrust_position_body_m=self._vector(self.thrust_pos_spins),
            max_gimbal_deg=self.max_gimbal_spin.value(),
            notes=self.notes_edit.toPlainText(),
            path=self._current_path,
        )

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Propulsion",
            str(self._current_path or ""),
            "Propulsion (*.propulsion.json);;JSON (*.json)",
        )
        if not path:
            return

        try:
            model = self._model_from_ui()
            saved_path = save_propulsion_model(model, Path(path))
            self._current_path = saved_path
            self._write_propulsion_handoff(saved_path)
            self._update_summary(model)
            QMessageBox.information(
                self,
                "Save Propulsion",
                "Propulsion model saved and sent to trajectory analysis.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save Propulsion", f"Failed to save propulsion model:\n{exc}")

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Propulsion",
            "",
            "Propulsion (*.propulsion.json);;JSON (*.json)",
        )
        if path:
            self._load_model(Path(path))

    def _load_model(self, path: Path) -> None:
        try:
            model = load_propulsion_model(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open Propulsion", f"Failed to load propulsion model:\n{exc}")
            return
        self._current_path = path
        self._set_model(model)

    def _on_generate_flat_curve(self) -> None:
        peak = self._max_table_thrust() or 20_000.0
        burn = max(self._max_table_time(), 30.0)
        self._set_curve(
            np.array([0.0, burn, burn + 0.1]),
            np.array([peak, peak, 0.0]),
        )

    def _set_curve(self, time_s: np.ndarray, thrust_n: np.ndarray) -> None:
        self.curve_table.setRowCount(0)
        for t, thrust in zip(time_s, thrust_n):
            self._add_curve_row(float(t), float(thrust))

    def _add_curve_row(self, time_s: float = 0.0, thrust_n: float = 0.0) -> None:
        row = self.curve_table.rowCount()
        self.curve_table.insertRow(row)
        self.curve_table.setItem(row, 0, QTableWidgetItem(f"{time_s:g}"))
        self.curve_table.setItem(row, 1, QTableWidgetItem(f"{thrust_n:g}"))

    def _remove_selected_curve_rows(self) -> None:
        rows = sorted({index.row() for index in self.curve_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.curve_table.removeRow(row)

    def _sort_curve_table(self) -> None:
        time_s, thrust_n = self._curve_arrays()
        order = np.argsort(time_s)
        self._set_curve(time_s[order], thrust_n[order])

    def _curve_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        times = []
        thrusts = []
        for row in range(self.curve_table.rowCount()):
            time_item = self.curve_table.item(row, 0)
            thrust_item = self.curve_table.item(row, 1)
            if time_item is None or thrust_item is None:
                continue
            times.append(float(time_item.text()))
            thrusts.append(float(thrust_item.text()))
        if len(times) < 2:
            raise ValueError("Add at least two thrust-curve points.")
        return np.asarray(times, dtype=float), np.asarray(thrusts, dtype=float)

    def _write_propulsion_handoff(self, propulsion_path: Path) -> None:
        if self._propulsion_handoff_path is None:
            return
        self._propulsion_handoff_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "propulsion_path": str(propulsion_path),
            "saved_at": time.time(),
        }
        tmp_path = self._propulsion_handoff_path.with_suffix(
            self._propulsion_handoff_path.suffix + ".tmp"
        )
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(self._propulsion_handoff_path)

    def _update_summary(self, model: PropulsionModel) -> None:
        self.summary_label.setText(
            f"Total impulse: {model.total_impulse_ns:,.1f} N*s    "
            f"Peak thrust: {model.peak_thrust_n:,.1f} N    "
            f"Average thrust: {model.average_thrust_n:,.1f} N    "
            f"Burn time: {model.burn_time_s:.2f} s"
        )

    def _set_vector(self, spins: list[QDoubleSpinBox], values: np.ndarray) -> None:
        for spin, value in zip(spins, values):
            spin.setValue(float(value))

    def _vector(self, spins: list[QDoubleSpinBox]) -> np.ndarray:
        return np.array([spin.value() for spin in spins], dtype=float)

    def _normalized_vector(self, spins: list[QDoubleSpinBox]) -> np.ndarray:
        vector = self._vector(spins)
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            return np.array([0.0, 1.0, 0.0])
        return vector / norm

    def _max_table_time(self) -> float:
        try:
            time_s, _ = self._curve_arrays()
        except Exception:
            return 0.0
        return float(np.max(time_s))

    def _max_table_thrust(self) -> float:
        try:
            _, thrust_n = self._curve_arrays()
        except Exception:
            return 0.0
        return float(np.max(thrust_n))
