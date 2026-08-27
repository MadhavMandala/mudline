from __future__ import annotations

import numpy as np
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QGroupBox, QFormLayout, QMessageBox,
)
from PySide6.QtCore import Qt, Signal

from massprops.model.models import Component, MassProperties
from massprops.utils.units import (
    convert_mass_from_internal,
    convert_mass_to_internal,
    convert_density_from_internal,
    convert_density_to_internal,
)


class PropertyPanel(QWidget):
    """Right-side panel showing mass properties and override controls."""

    override_applied = Signal(Component)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._current: Optional[Component] = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        # Identity
        self.name_label = QLabel("<no selection>")
        self.name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(self.name_label)

        # Computed properties
        computed_box = QGroupBox("Computed Properties")
        computed_form = QFormLayout(computed_box)
        self.mass_label = QLabel("-")
        self.volume_label = QLabel("-")
        self.cg_label = QLabel("-")
        self.inertia_label = QLabel("-")
        computed_form.addRow("Mass (lbm):", self.mass_label)
        computed_form.addRow("Volume (in³):", self.volume_label)
        computed_form.addRow("CG (in):", self.cg_label)
        computed_form.addRow("Inertia (lbm·in²):", self.inertia_label)
        layout.addWidget(computed_box)

        # Override section
        override_box = QGroupBox("Override")
        override_layout = QVBoxLayout(override_box)

        override_form = QFormLayout()
        self.override_mass_edit = QLineEdit()
        self.override_cg_x = QLineEdit()
        self.override_cg_y = QLineEdit()
        self.override_cg_z = QLineEdit()
        override_form.addRow("Mass (lbm):", self.override_mass_edit)
        cg_layout = QHBoxLayout()
        cg_layout.addWidget(self.override_cg_x)
        cg_layout.addWidget(self.override_cg_y)
        cg_layout.addWidget(self.override_cg_z)
        override_form.addRow("CG X,Y,Z (in):", cg_layout)
        override_layout.addLayout(override_form)

        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Override")
        self.clear_override_btn = QPushButton("Clear Override")
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.clear_override_btn)
        override_layout.addLayout(btn_layout)

        self.apply_btn.clicked.connect(self._on_apply)
        self.clear_override_btn.clicked.connect(self._on_clear_override)

        layout.addWidget(override_box)
        layout.addStretch()

    def set_component(self, comp: Optional[Component]) -> None:
        self._current = comp
        if comp is None:
            self.name_label.setText("<no selection>")
            self.mass_label.setText("-")
            self.volume_label.setText("-")
            self.cg_label.setText("-")
            self.inertia_label.setText("-")
            self.override_mass_edit.clear()
            self.override_cg_x.clear()
            self.override_cg_y.clear()
            self.override_cg_z.clear()
            return

        self.name_label.setText(comp.name)
        props = comp.effective_props()
        self.mass_label.setText(f"{props.mass:.4f}")
        self.volume_label.setText(f"{props.volume:.4f}")
        self.cg_label.setText(f"[{props.cg[0]:.4f}, {props.cg[1]:.4f}, {props.cg[2]:.4f}]")
        inertia_str = "\n".join(
            f"  [{props.inertia[i,0]:.2e}, {props.inertia[i,1]:.2e}, {props.inertia[i,2]:.2e}]"
            for i in range(3)
        )
        self.inertia_label.setText(inertia_str)

        # Populate override fields only if they were explicitly overridden
        if 'mass' in comp.override_fields and comp.overridden_props is not None:
            self.override_mass_edit.setText(f"{comp.overridden_props.mass:.4f}")
        else:
            self.override_mass_edit.clear()

        if 'cg' in comp.override_fields and comp.overridden_props is not None:
            self.override_cg_x.setText(f"{comp.overridden_props.cg[0]:.4f}")
            self.override_cg_y.setText(f"{comp.overridden_props.cg[1]:.4f}")
            self.override_cg_z.setText(f"{comp.overridden_props.cg[2]:.4f}")
        else:
            self.override_cg_x.clear()
            self.override_cg_y.clear()
            self.override_cg_z.clear()

    def _on_apply(self) -> None:
        if self._current is None:
            return
        try:
            mass_text = self.override_mass_edit.text().strip()
            cg_x_text = self.override_cg_x.text().strip()
            cg_y_text = self.override_cg_y.text().strip()
            cg_z_text = self.override_cg_z.text().strip()

            # Determine which fields the user actually wants to override
            fields: set[str] = set()
            mass = 0.0
            cg = np.zeros(3)

            if mass_text:
                mass = float(mass_text)
                fields.add('mass')

            if cg_x_text or cg_y_text or cg_z_text:
                cg = np.array([
                    float(cg_x_text or 0),
                    float(cg_y_text or 0),
                    float(cg_z_text or 0),
                ])
                fields.add('cg')

            if not fields:
                QMessageBox.information(self, "No Override", "Enter a mass or CG value to override.")
                return

            # For tensor override, we'd need a 3x3 dialog; for MVP, use computed inertia
            base = self._current.computed_props or MassProperties()
            inertia = base.inertia
            volume = base.volume

            self._current.overridden_props = MassProperties(mass=mass, cg=cg, inertia=inertia, volume=volume)
            self._current.override_fields = fields
            self.override_applied.emit(self._current)
            self.set_component(self._current)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Input", str(exc))

    def _on_clear_override(self) -> None:
        if self._current is None:
            return

        self._current.overridden_props = None
        self._current.override_fields = set()
        self.override_applied.emit(self._current)
        self.set_component(self._current)
