"""Mass budget: the statement a design review actually asks for.

Every number here already existed in the model and none of it was visible. A
mass roll-up that lives only in a status bar total cannot answer the questions
people ask of it -- what is the heaviest item, what fraction is structure
versus equipment, how much growth is allowed and what happens to the margin
when it is taken, and how far each item moves the centre of gravity.

Imported CAD appears here like anything else: an imported Stack carries a
material and a wall thickness, so it contributes a real line rather than an
opaque lump.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import theme
from parametric.components import FinSet, Motor, PointMass, Stack, Tank, Wing
from parametric.model import VehicleModel

COLUMNS = ["Item", "Kind", "Mass kg", "% dry", "Growth kg", "Station m", "Moment kg·m"]


class MassBudget(QWidget):
    """Line-by-line mass statement with growth allowance and CG contribution.

    Editable in the mass column. A budget is the sheet you sit in front of when
    you are reconciling a model against a set of scales, and being able to read
    it but not correct it means going back to the component tree, finding the
    part, ticking a box, and returning -- for every line.
    """

    #: A mass was overridden or cleared; the window re-solves and redraws.
    mass_overridden = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._model: VehicleModel | None = None
        self._solved = None
        self._guard = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        self._title = QLabel("Mass budget")
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)
        controls.addWidget(self._title)
        controls.addStretch(1)

        self._show_growth = QCheckBox("Apply growth allowance")
        self._show_growth.toggled.connect(lambda _: self.refresh())
        controls.addWidget(self._show_growth)
        layout.addLayout(controls)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(len(COLUMNS))
        self.tree.setHeaderLabels(COLUMNS)
        self.tree.setAlternatingRowColors(True)
        self.tree.setRootIsDecorated(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setEditTriggers(
            QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed
        )
        self.tree.itemChanged.connect(self._on_item_changed)
        self.tree.setStyleSheet(
            f"font-family:{theme.MONO_FONT},monospace; font-size:8pt;"
        )
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(COLUMNS)):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        layout.addWidget(self.tree, 1)

        self._summary = QLabel()
        self._summary.setStyleSheet(
            f"font-family:{theme.MONO_FONT},monospace; font-size:8pt;"
            f"color:{theme.TEXT_DIM};"
        )
        layout.addWidget(self._summary)

    # ------------------------------------------------------------------

    def set_model(self, model: VehicleModel | None, solved=None) -> None:
        self._model = model
        self._solved = solved
        self.refresh()

    def _on_item_changed(self, item, column: int) -> None:
        """A mass typed into the sheet becomes that component's measured mass.

        Blank clears the override and hands the part back to its geometry,
        which is the only way out that does not require remembering what the
        computed value used to be.
        """
        if self._guard or column != 2 or self._model is None:
            return
        path = item.data(0, Qt.UserRole)
        if not path:
            return
        component = next(
            (c for c in self._model.walk() if c.path == path), None
        )
        if component is None:
            return

        text = item.text(2).strip()
        if not text:
            component.mass_override_kg = None
        else:
            try:
                value = float(text)
            except ValueError:
                self.refresh()      # put the old number back
                return
            if value < 0:
                self.refresh()
                return
            component.mass_override_kg = value
        component.mark_dirty("mass_override")
        self.mass_overridden.emit(component)
        self.refresh()

    def refresh(self) -> None:
        # Rebuilding the rows fires itemChanged for every cell written, which
        # would come straight back in here and recurse.
        self._guard = True
        try:
            self._refresh()
        finally:
            self._guard = False

    def _refresh(self) -> None:
        self.tree.clear()
        model = self._model
        if model is None:
            self._summary.setText("")
            return

        with_growth = self._show_growth.isChecked()
        summary = model.mass_summary()
        measured = self._solved.per_component_kg if self._solved is not None else {}

        rows: list[tuple[str, str, float, float, float]] = []
        for component in model.walk():
            if component is model.root:
                continue

            if isinstance(component, Motor):
                # Motors were skipped outright, from when they had no dry mass
                # at all. An engine with a declared hardware mass now weighs
                # something and belongs on the sheet; one that does not is
                # dropped by the zero-mass test below, exactly as before.
                base = component.mass_kg()
                growth = 0.0
                kind = "structure"
                low, high = component.station_range_m()
                station = 0.5 * (low + high)
            elif isinstance(component, PointMass):
                base = component.get("mass")
                growth = component.mass_with_growth_kg - base
                kind = "equipment"
                station = component.get("station")
            elif isinstance(component, (Stack, FinSet, Tank, Wing)):
                # Tanks and wings were missing here: the test was a tuple of
                # the kinds that existed when this was written, so every
                # component type added since fell through the else and
                # vanished from the budget while still counting towards the
                # totals in the status bar. A mass statement that silently
                # omits the largest structural item is worse than none.
                base = measured.get(component.path, component.mass_kg())
                if isinstance(component, (FinSet, Wing)) and base == 0:
                    base = component.mass_kg()
                growth = 0.0
                kind = "structure"
                low, high = component.station_range_m()
                station = 0.5 * (low + high)
            else:
                continue

            if base <= 0:
                continue
            rows.append((component.name, kind, base, growth, station,
                         component.path,
                         component.mass_override_kg is not None))

        rows.sort(key=lambda row: row[2], reverse=True)
        dry = sum(row[2] + (row[3] if with_growth else 0.0) for row in rows)

        groups: dict[str, QTreeWidgetItem] = {}
        for kind in ("structure", "equipment"):
            total = sum(
                row[2] + (row[3] if with_growth else 0.0)
                for row in rows if row[1] == kind
            )
            item = QTreeWidgetItem([
                kind, "", f"{total:.3f}",
                f"{100 * total / dry:.1f}" if dry > 0 else "-",
                "", "", "",
            ])
            font = item.font(0)
            font.setBold(True)
            for column in range(len(COLUMNS)):
                item.setFont(column, font)
            self.tree.addTopLevelItem(item)
            groups[kind] = item

        for name, kind, base, growth, station, path, overridden in rows:
            mass = base + (growth if with_growth else 0.0)
            item = QTreeWidgetItem([
                name, kind, f"{mass:.3f}",
                f"{100 * mass / dry:.1f}" if dry > 0 else "-",
                f"{growth:.3f}" if growth > 0 else "",
                f"{station:.3f}",
                f"{mass * station:.3f}",
            ])
            for column in (2, 3, 4, 5, 6):
                item.setTextAlignment(column, Qt.AlignRight | Qt.AlignVCenter)
            if growth > 0:
                item.setForeground(4, QColor(theme.WARN))
            # The mass cell is editable in place: this is the sheet you read a
            # budget off, so it is where you want to say "this one weighs what
            # the scale says" without hunting for the part in the tree first.
            item.setData(0, Qt.UserRole, path)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            if overridden:
                item.setForeground(2, QColor(theme.ACCENT))
                item.setToolTip(2, "Measured. Clear the cell to go back to geometry.")
            groups[kind].addChild(item)

        # Propellant is not dry mass, so it sits outside the structure and
        # equipment groups rather than inflating either.
        propellant = summary.propellant_mass_kg
        if propellant > 0:
            # The roll-up already works out where the propellant balances,
            # across every motor and tank. Reading the first motor's centroid
            # instead reported the engine's station for a vehicle whose
            # propellant is actually in tanks somewhere else entirely.
            station = summary.propellant_cg_station_m
            item = QTreeWidgetItem([
                "propellant", "consumable", f"{propellant:.3f}", "",
                "", f"{station:.3f}", f"{propellant * station:.3f}",
            ])
            for column in (2, 5, 6):
                item.setTextAlignment(column, Qt.AlignRight | Qt.AlignVCenter)
            self.tree.addTopLevelItem(item)

        self.tree.expandAll()

        cg = (
            self._solved.cg_station_m if self._solved is not None
            else summary.cg_station_m
        )
        growth_total = summary.dry_mass_with_growth_kg - summary.dry_mass_kg
        wet = dry + propellant
        source = "meshed" if self._solved is not None else "analytic"

        self._summary.setText(
            f"dry {dry:8.3f} kg ({source})     "
            f"growth allowance {growth_total:6.3f} kg     "
            f"wet {wet:8.3f} kg     "
            f"mass ratio {wet / dry if dry > 0 else 0:5.2f}     "
            f"dry CG {cg:6.3f} m"
        )
