"""Cross-section editor for a Stack.

This is the part of the OpenVSP method that makes a body authorable rather than
merely tweakable. A Stack's shape *is* its section list, so being able to add,
remove, retype and move sections is the difference between adjusting a vehicle
someone else built and building one.

The table is deliberately direct: one row per section, editable in place. A
nose is not a special object with a "type" field to choose from -- it is a run
of sections whose radii happen to follow a curve, and once generated they are
ordinary rows here.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
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

from parametric.components import Stack
from parametric.xsec import NoseProfile, XSec, XSecShape

SHAPES = [shape.value for shape in XSecShape]


class SectionEditor(QWidget):
    """Table of a stack's cross-sections, with add/remove and profile builders."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack: Stack | None = None
        self._guard = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(6)

        from app.parmeditor import section_header

        layout.addWidget(section_header("Cross-sections"))

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Station m", "Shape", "Width m", "Height m"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("font-family:Consolas,monospace; font-size:11px;")
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            head.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        # The table has to be able to get narrower than its four columns would
        # like. Otherwise its minimum width becomes the minimum width of the
        # whole property panel, and every row above it -- sliders included --
        # is stretched past the right edge to match a table nobody asked to be
        # that wide. It scrolls internally instead, which is ordinary for a
        # table and not for a panel.
        self.table.setMinimumWidth(170)
        self.table.setMaximumHeight(220)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.table.currentCellChanged.connect(self._on_row_selected)
        layout.addWidget(self.table)

        # Inline editors for the selected row. Editing in a detail strip rather
        # than in the cells keeps the numbers formatted and the ranges enforced.
        # Two columns, not one row. Four labelled fields side by side needed
        # around 520 px, and because that is a minimum rather than a preference
        # it became the minimum width of the entire property panel -- which is
        # why the sliders above it ran off the edge.
        self._detail = QWidget()
        detail_layout = QGridLayout(self._detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setHorizontalSpacing(6)
        detail_layout.setVerticalSpacing(4)

        self._station = self._spin(0.0, 1e4, " m")
        self._station.valueChanged.connect(lambda v: self._edit("station", v))
        self._width = self._spin(0.0, 1e3, " m")
        self._width.valueChanged.connect(lambda v: self._edit("width", v))
        self._height = self._spin(0.0, 1e3, " m")
        self._height.valueChanged.connect(lambda v: self._edit("height", v))
        self._shape = QComboBox()
        self._shape.addItems(SHAPES)
        self._shape.currentTextChanged.connect(self._on_shape)

        for index, (label, widget) in enumerate([
            ("Station", self._station), ("Shape", self._shape),
            ("Width", self._width), ("Height", self._height),
        ]):
            row, column = divmod(index, 2)
            detail_layout.addWidget(QLabel(label), row, column * 2,
                                    alignment=Qt.AlignRight)
            detail_layout.addWidget(widget, row, column * 2 + 1)
        detail_layout.setColumnStretch(1, 1)
        detail_layout.setColumnStretch(3, 1)
        layout.addWidget(self._detail)

        buttons = QHBoxLayout()
        buttons.setSpacing(4)
        for text, slot in [
            ("Add", self._add_section),
            ("Duplicate", self._duplicate_section),
            ("Remove", self._remove_section),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            buttons.addWidget(button)
        layout.addLayout(buttons)

        builders = QHBoxLayout()
        builders.setSpacing(4)
        for text, slot in [
            ("+ Nose", self._add_nose),
            ("+ Tube", self._add_tube),
            ("+ Transition", self._add_transition),
        ]:
            button = QPushButton(text)
            button.clicked.connect(slot)
            builders.addWidget(button)
        layout.addLayout(builders)

    # ------------------------------------------------------------------

    @staticmethod
    def _spin(minimum: float, maximum: float, suffix: str) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(0.005)
        spin.setSuffix(suffix)
        spin.setMinimumWidth(72)
        return spin

    def set_stack(self, stack: Stack | None) -> None:
        self._stack = stack
        self.setVisible(stack is not None)
        self.refresh()

    def refresh(self) -> None:
        if self._stack is None:
            self.table.setRowCount(0)
            return
        self._guard = True
        sections = self._stack.sorted_sections()
        self.table.setRowCount(len(sections))
        for row, section in enumerate(sections):
            values = [
                f"{section.station_m:.4f}",
                section.shape.value,
                f"{section.width_m:.4f}",
                f"{section.height_m:.4f}",
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.table.setItem(row, column, item)
        self._guard = False
        self._sync_detail()

    # ------------------------------------------------------------------

    def _selected(self) -> XSec | None:
        if self._stack is None:
            return None
        row = self.table.currentRow()
        sections = self._stack.sorted_sections()
        if 0 <= row < len(sections):
            return sections[row]
        return None

    def _on_row_selected(self, *_args) -> None:
        self._sync_detail()

    def _sync_detail(self) -> None:
        section = self._selected()
        enabled = section is not None
        self._detail.setEnabled(enabled)
        if section is None:
            return
        self._guard = True
        self._station.setValue(section.station_m)
        self._width.setValue(section.width_m)
        self._height.setValue(section.height_m)
        self._shape.setCurrentText(section.shape.value)
        self._guard = False

    def _edit(self, parm: str, value: float) -> None:
        if self._guard:
            return
        section = self._selected()
        if section is None:
            return
        if section.set(parm, value):
            # A circle has one radius; editing width has to carry height with
            # it or the section silently stops being circular.
            if parm == "width" and section.shape is XSecShape.CIRCLE:
                section.set("height", value)
            self._stack.mark_dirty("section")
            self._after_change()

    def _on_shape(self, name: str) -> None:
        if self._guard:
            return
        section = self._selected()
        if section is None:
            return
        section.shape = XSecShape(name)
        self._stack.mark_dirty("section")
        self._after_change()

    def _after_change(self) -> None:
        row = self.table.currentRow()
        self.refresh()
        if 0 <= row < self.table.rowCount():
            self.table.setCurrentCell(row, 0)
        self.changed.emit()

    # ------------------------------------------------------------------

    def _add_section(self) -> None:
        if self._stack is None:
            return
        sections = self._stack.sorted_sections()
        if sections:
            last = sections[-1]
            station = last.station_m + max(0.05, self._stack.length_m * 0.1)
            width, height, shape = last.width_m, last.height_m, last.shape
        else:
            station, width, height, shape = 0.0, 0.1, 0.1, XSecShape.CIRCLE
        self._stack.add_section(XSec(station, shape, width, height,
                                     name=f"xsec_{len(sections)}"))
        self._after_change()

    def _duplicate_section(self) -> None:
        section = self._selected()
        if section is None or self._stack is None:
            return
        offset = max(0.02, self._stack.length_m * 0.05)
        self._stack.add_section(XSec(
            section.station_m + offset, section.shape,
            section.width_m, section.height_m,
            section.get("exponent"), section.get("corner_radius"),
            name=f"{section.name}_copy",
        ))
        self._after_change()

    def _remove_section(self) -> None:
        section = self._selected()
        if section is None or self._stack is None:
            return
        if len(self._stack.sections) <= 2:
            return          # a loft needs two; refuse rather than break geometry
        self._stack.remove_section(section)
        self._after_change()

    # ------------------------------------------------------------------

    def _add_nose(self) -> None:
        if self._stack is None:
            return
        profile, ok = QInputDialog.getItem(
            self, "Add nose", "Profile:", [p.value for p in NoseProfile], 2, False
        )
        if not ok:
            return
        length, ok = QInputDialog.getDouble(
            self, "Add nose", "Length (m):", 0.45, 0.001, 100.0, 4
        )
        if not ok:
            return
        diameter, ok = QInputDialog.getDouble(
            self, "Add nose", "Base diameter (m):",
            self._stack.max_diameter_m or 0.1, 0.001, 100.0, 4
        )
        if not ok:
            return
        self._stack.add_nose(NoseProfile(profile), length, diameter)
        self._after_change()

    def _add_tube(self) -> None:
        if self._stack is None:
            return
        length, ok = QInputDialog.getDouble(
            self, "Add tube", "Length (m):", 0.5, 0.001, 100.0, 4
        )
        if not ok:
            return
        diameter = self._stack.max_diameter_m or 0.1
        self._stack.add_tube(length, diameter, name=f"tube{len(self._stack.sections)}")
        self._after_change()

    def _add_transition(self) -> None:
        if self._stack is None:
            return
        length, ok = QInputDialog.getDouble(
            self, "Add transition", "Length (m):", 0.15, 0.001, 100.0, 4
        )
        if not ok:
            return
        rear, ok = QInputDialog.getDouble(
            self, "Add transition", "Rear diameter (m):",
            max(self._stack.max_diameter_m * 0.7, 0.01), 0.001, 100.0, 4
        )
        if not ok:
            return
        self._stack.add_transition(length, rear,
                                   name=f"trans{len(self._stack.sections)}")
        self._after_change()
