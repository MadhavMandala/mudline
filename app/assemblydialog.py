"""Assign the solids in a STEP assembly to the parts they are.

The dialog exists because the alternative is guessing, and the guess is wrong
in a way that does not announce itself: the profile fitter reads four fins as a
body collar of twice the true diameter and reports a 0.04 mm residual while
doing it. A person looking at a list of twelve named solids knows which three
are tanks in about four seconds. This asks them.

Layout follows the analysis dialogs -- everything the build depends on is on
screen and editable before it runs:

    solids                              assignment
    what the file contains, with        one row per part, with the type
    size, extent and where it sits      it has been given

Rows are pre-filled from ``assembly_import.suggest``, which separates blades
from bodies on geometry alone. That is a convenience, not an authority: every
row is editable, and the suggestion never sets tank or intertank because
nothing in the geometry distinguishes them.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from parametric.assembly_import import Assignment, PartType, suggest
from parametric.step_assembly import AssemblyRead


class AssemblyAssignDialog(QDialog):
    """Pick what each solid is, then build the model from it."""

    def __init__(self, read: AssemblyRead, parent=None):
        super().__init__(parent)
        self.read = read
        self.setWindowTitle(f"Import {read.source.name}")
        self.resize(1040, 620)

        self._assignments: list[Assignment] = suggest(read)
        self._build_ui()
        self._refresh_solids()
        self._refresh_parts()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)

        header = QLabel(self.read.text())
        header.setTextFormat(Qt.PlainText)
        header.setStyleSheet("font-family: monospace;")
        outer.addWidget(header)

        columns = QHBoxLayout()
        outer.addLayout(columns, 1)

        # -- left: what the file contains ------------------------------
        left = QVBoxLayout()
        left.addWidget(QLabel("Solids in the file"))
        self.solids = QTreeWidget()
        self.solids.setHeaderLabels(
            ["Solid", "Volume cm³", "From m", "To m", "Radius mm", "Assigned to"]
        )
        self.solids.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.solids.setAlternatingRowColors(True)
        self.solids.setRootIsDecorated(False)
        left.addWidget(self.solids, 1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Assign selection as"))
        self.new_type = QComboBox()
        for part_type in PartType:
            self.new_type.addItem(part_type.label, part_type)
        self.new_type.setCurrentIndex(list(PartType).index(PartType.BODY))
        controls.addWidget(self.new_type, 1)
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText("part name (optional)")
        controls.addWidget(self.new_name, 1)
        add = QPushButton("Add part")
        add.clicked.connect(self._add_part)
        controls.addWidget(add)
        left.addLayout(controls)
        columns.addLayout(left, 3)

        # -- right: what it will become --------------------------------
        right = QVBoxLayout()
        right.addWidget(QLabel("Parts to build"))
        self.parts = QTreeWidget()
        self.parts.setHeaderLabels(["Part", "Type", "Solids"])
        self.parts.setAlternatingRowColors(True)
        self.parts.setRootIsDecorated(False)
        self.parts.currentItemChanged.connect(self._on_part_selected)
        right.addWidget(self.parts, 1)

        part_controls = QHBoxLayout()
        self.part_type = QComboBox()
        for part_type in PartType:
            self.part_type.addItem(part_type.label, part_type)
        self.part_type.currentIndexChanged.connect(self._retype_part)
        part_controls.addWidget(self.part_type, 1)
        remove = QPushButton("Remove")
        remove.clicked.connect(self._remove_part)
        part_controls.addWidget(remove)
        right.addLayout(part_controls)
        columns.addLayout(right, 2)

        self.summary = QLabel("")
        self.summary.setTextFormat(Qt.PlainText)
        outer.addWidget(self.summary)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    @staticmethod
    def _type_of(combo) -> PartType:
        """The part type a combo box is showing.

        ``PartType`` subclasses ``str``, so Qt stores the item data as a plain
        string and hands that back rather than the enum member. Coercing here
        keeps every caller working with the enum.
        """
        return PartType(combo.currentData())

    def _owner_of(self) -> dict[int, str]:
        owner: dict[int, str] = {}
        for assignment in self._assignments:
            for index in assignment.indices:
                owner[index] = assignment.name
        return owner

    def _refresh_solids(self) -> None:
        axis = self.read.axis
        owner = self._owner_of()
        self.solids.clear()
        for component in sorted(
            self.read.components, key=lambda c: c.bounds_min_m[axis]
        ):
            low, high = component.extent_m(axis)
            item = QTreeWidgetItem(
                [
                    component.name,
                    f"{component.volume_m3 * 1e6:.1f}",
                    f"{low:.3f}",
                    f"{high:.3f}",
                    f"{component.max_radius_m(axis) * 1000:.1f}",
                    owner.get(component.index, "—"),
                ]
            )
            item.setData(0, Qt.UserRole, component.index)
            if component.index not in owner:
                item.setForeground(5, Qt.red)
            self.solids.addTopLevelItem(item)
        for column in range(self.solids.columnCount()):
            self.solids.resizeColumnToContents(column)
        self.solids.header().setSectionResizeMode(0, QHeaderView.Stretch)

    def _refresh_parts(self) -> None:
        by_index = {c.index: c for c in self.read.components}
        self.parts.clear()
        for assignment in self._assignments:
            names = ", ".join(
                by_index[i].name for i in assignment.indices if i in by_index
            )
            item = QTreeWidgetItem(
                [assignment.name, assignment.part_type.label, names]
            )
            item.setData(0, Qt.UserRole, id(assignment))
            self.parts.addTopLevelItem(item)
        for column in range(self.parts.columnCount()):
            self.parts.resizeColumnToContents(column)
        self.parts.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._refresh_summary()

    def _refresh_summary(self) -> None:
        owner = self._owner_of()
        unassigned = [c for c in self.read.components if c.index not in owner]
        parts = [a for a in self._assignments if a.part_type is not PartType.IGNORE]
        text = (
            f"{len(parts)} part(s) from "
            f"{sum(len(a.indices) for a in parts)} of "
            f"{len(self.read.components)} solids"
        )
        if unassigned:
            text += f"   ·   {len(unassigned)} unassigned: " + ", ".join(
                c.name for c in unassigned[:6]
            )
            if len(unassigned) > 6:
                text += ", …"
        if not any(a.part_type is PartType.NOSE for a in parts):
            text += "   ·   no nose assigned; stations may run backwards"
        self.summary.setText(text)

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------

    def _selected_indices(self) -> list[int]:
        return [
            item.data(0, Qt.UserRole) for item in self.solids.selectedItems()
        ]

    def _add_part(self) -> None:
        indices = self._selected_indices()
        if not indices:
            QMessageBox.information(
                self, "Assign", "Select one or more solids on the left first."
            )
            return
        part_type = self._type_of(self.new_type)
        by_index = {c.index: c for c in self.read.components}
        name = self.new_name.text().strip() or by_index[indices[0]].name

        # A solid belongs to exactly one part; adopting it moves it.
        for assignment in self._assignments:
            assignment.indices = [i for i in assignment.indices if i not in indices]
        self._assignments = [a for a in self._assignments if a.indices]
        self._assignments.append(Assignment(part_type, name, list(indices)))
        self._assignments.sort(
            key=lambda a: min(
                by_index[i].bounds_min_m[self.read.axis] for i in a.indices
            )
        )
        self.new_name.clear()
        self._refresh_solids()
        self._refresh_parts()

    def _current_assignment(self) -> Assignment | None:
        item = self.parts.currentItem()
        if item is None:
            return None
        key = item.data(0, Qt.UserRole)
        return next((a for a in self._assignments if id(a) == key), None)

    def _on_part_selected(self, *_args) -> None:
        assignment = self._current_assignment()
        if assignment is None:
            return
        self.part_type.blockSignals(True)
        self.part_type.setCurrentIndex(list(PartType).index(assignment.part_type))
        self.part_type.blockSignals(False)

    def _retype_part(self) -> None:
        assignment = self._current_assignment()
        if assignment is None:
            return
        assignment.part_type = self._type_of(self.part_type)
        self._refresh_parts()
        self._select_part(assignment)

    def _select_part(self, assignment: Assignment) -> None:
        """Reselect a part after the list has been rebuilt under it."""
        for row in range(self.parts.topLevelItemCount()):
            item = self.parts.topLevelItem(row)
            if item.data(0, Qt.UserRole) == id(assignment):
                self.parts.setCurrentItem(item)
                return

    def _remove_part(self) -> None:
        assignment = self._current_assignment()
        if assignment is None:
            return
        self._assignments.remove(assignment)
        self._refresh_solids()
        self._refresh_parts()

    # ------------------------------------------------------------------

    def _accept(self) -> None:
        if not [a for a in self._assignments if a.part_type is not PartType.IGNORE]:
            QMessageBox.warning(
                self, "Import", "Assign at least one solid before importing."
            )
            return
        self.accept()

    def assignments(self) -> list[Assignment]:
        return list(self._assignments)
