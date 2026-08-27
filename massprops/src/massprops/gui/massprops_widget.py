from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QTreeView, QSplitter, QFileDialog, QToolBar, QComboBox,
    QLabel, QMessageBox,
)
from PySide6.QtCore import Qt, QModelIndex, QAbstractItemModel, QObject
from pyvistaqt import QtInteractor

from massprops.model.models import Component
from massprops.gui.pyvista_viewer import populate_plotter
from massprops.gui.property_panel import PropertyPanel
from massprops.io.step_parser import StepParser


class ComponentTreeModel(QAbstractItemModel):
    """Tree model for Component / Assembly hierarchy."""

    def __init__(self, root: Optional[Component] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.root = root

    def set_root(self, root: Optional[Component]) -> None:
        self.beginResetModel()
        self.root = root
        self.endResetModel()

    def _node(self, index: QModelIndex) -> Optional[Component]:
        if not index.isValid():
            return self.root
        return index.internalPointer()

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        parent_node = self._node(parent)
        if not parent_node or row < 0 or row >= len(parent_node.children):
            return QModelIndex()
        return self.createIndex(row, column, parent_node.children[row])

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        child = index.internalPointer()
        if child is None or self.root is None:
            return QModelIndex()
        parent = self._find_parent(self.root, child)
        if parent is None or parent is self.root:
            return QModelIndex()
        grandparent = self._find_parent(self.root, parent)
        if grandparent is None:
            row = 0
        else:
            row = grandparent.children.index(parent)
        return self.createIndex(row, 0, parent)

    def _find_parent(self, node: Component, target: Component) -> Optional[Component]:
        for child in node.children:
            if child is target:
                return node
            result = self._find_parent(child, target)
            if result is not None:
                return result
        return None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = self._node(parent)
        return len(node.children) if node else 0

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 3  # Name, Mass, Volume

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid():
            return None
        node = index.internalPointer()
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return node.name
            elif index.column() == 1:
                props = node.effective_props()
                return f"{props.mass:.3f}"
            elif index.column() == 2:
                props = node.effective_props()
                return f"{props.volume:.3f}"
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return ["Name", "Mass (lbm)", "Volume (in³)"][section]
        return None


class MassPropsWidget(QWidget):
    """Reusable mass-properties editor widget (central content of the former MainWindow)."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        vehicle_handoff_path: Path | None = None,
    ):
        super().__init__(parent)
        self._root: Optional[Component] = None
        self._selected: Optional[Component] = None
        self._vehicle_handoff_path = Path(vehicle_handoff_path) if vehicle_handoff_path else None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QToolBar()
        self.open_btn = toolbar.addAction("Open STEP")
        self.open_btn.triggered.connect(self._on_open_step)

        self.open_folder_btn = toolbar.addAction("Open Folder")
        self.open_folder_btn.triggered.connect(self._on_open_folder)

        self.save_vehicle_btn = toolbar.addAction("Save Vehicle")
        self.save_vehicle_btn.triggered.connect(self._on_save_vehicle)

        self.save_btn = toolbar.addAction("Save Project")
        self.save_btn.triggered.connect(self._on_save_project)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Units:"))
        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["Imperial (lbm, in)", "Metric (kg, m)"])
        toolbar.addWidget(self.unit_combo)

        layout.addWidget(toolbar)

        # Content splitter
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: Tree
        self.tree = QTreeView()
        self.tree_model = ComponentTreeModel()
        self.tree.setModel(self.tree_model)
        self.tree.setColumnWidth(0, 250)
        self.tree.selectionModel().currentChanged.connect(self._on_tree_selection)
        splitter.addWidget(self.tree)

        # Middle: Selected item 3D view
        self.selection_view = QtInteractor(self)
        splitter.addWidget(self.selection_view)

        # Right: Full assembly 3D view + property panel
        right_splitter = QSplitter(Qt.Vertical)

        self.full_assembly_view = QtInteractor(self)
        right_splitter.addWidget(self.full_assembly_view)

        self.property_panel = PropertyPanel()
        self.property_panel.override_applied.connect(self._on_override_applied)
        right_splitter.addWidget(self.property_panel)

        right_splitter.setSizes([500, 300])
        splitter.addWidget(right_splitter)
        splitter.setSizes([300, 500, 600])

    def set_root(self, root: Component) -> None:
        from massprops.model.assembly import rebalance_all_assembly_overrides
        self._root = root
        rebalance_all_assembly_overrides(root)
        self._check_and_show_mass_errors()
        self.tree_model.set_root(root)
        self.tree.expandAll()
        self._refresh_3d()

    def _refresh_3d(self) -> None:
        self.full_assembly_view.clear()
        if self._root is None:
            self.full_assembly_view.add_text("No model loaded", position="upper_left")
        else:
            populate_plotter(
                self.full_assembly_view,
                self._root,
                selected=self._selected,
                show_edges=True,
            )
            self.full_assembly_view.reset_camera()

        self.selection_view.clear()
        if self._selected is None:
            self.selection_view.add_text("Select an item from the tree", position="upper_left")
        else:
            populate_plotter(
                self.selection_view,
                self._selected,
                show_edges=True,
            )
            self.selection_view.reset_camera()

    def _on_tree_selection(self, current: QModelIndex, previous: QModelIndex) -> None:
        node = current.internalPointer()
        self._selected = node
        self.property_panel.set_component(node)
        self._refresh_3d()

    def _on_override_applied(self, comp: Component) -> None:
        from massprops.model.assembly import rebalance_all_assembly_overrides
        if self._root is not None:
            rebalance_all_assembly_overrides(self._root)
        self._check_and_show_mass_errors()
        self.tree_model.layoutChanged.emit()
        self._refresh_3d()

    def _check_and_show_mass_errors(self) -> None:
        if self._root is None:
            return
        errors = []
        def _scan(node):
            err = node.step_metadata.get('mass_error')
            if err:
                errors.append(f"{node.name}: {err}")
            for child in node.children:
                _scan(child)
        _scan(self._root)
        if errors:
            QMessageBox.warning(self, "Mass Constraint Error", "\n".join(errors))

    def _on_open_step(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open STEP File", "", "STEP Files (*.stp *.step)")
        if path:
            self._load_step(Path(path))

    def _on_open_folder(self) -> None:
        from massprops.io.assembly_loader import load_from_folder, expand_external_references

        folder = QFileDialog.getExistingDirectory(self, "Open Folder with STEP Files")
        if not folder:
            return
        try:
            from massprops.io.assembly_loader import resolve_source_paths
            folder_path = Path(folder)
            root, master_path = load_from_folder(folder_path)
            expand_external_references(root, folder_path)
            resolve_source_paths(root, folder_path)
            self._load_step(master_path, preloaded_root=root)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load folder: {exc}")

    def _load_step(self, path: Path, preloaded_root: Component | None = None) -> None:
        from massprops.io.assembly_loader import load_assembly
        from massprops.io.material_extractor import apply_materials_to_tree
        from massprops.mesh.mesh_cache import MeshCache

        try:
            if preloaded_root is not None:
                root = preloaded_root
            else:
                root = load_assembly(path)
            if not path.is_dir():
                apply_materials_to_tree(root, StepParser(path))

            cache_dir = Path(__file__).parent.parent.parent / "data"
            cache = MeshCache(cache_dir)
            meshed_count = 0

            def _mesh_tree(node: Component) -> None:
                nonlocal meshed_count
                if node.source_step and Path(node.source_step).exists():
                    if not cache.load_cached(node):
                        try:
                            MeshCache.mesh_component(node, mesh_size=None)
                            cache.save_cached(node)
                            meshed_count += 1
                        except Exception:
                            pass
                for child in node.children:
                    _mesh_tree(child)

            _mesh_tree(root)

            if meshed_count == 0:
                QMessageBox.warning(self, "Meshing Warning", "Could not mesh any parts.")

            self.set_root(root)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to load STEP file: {exc}")

    def _on_save_project(self) -> None:
        from massprops.io.project_io import save_project
        if self._root is None:
            QMessageBox.information(self, "Save", "No model to save.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "JSON (*.json)")
        if path:
            try:
                save_project(self._root, Path(path))
                QMessageBox.information(self, "Save", "Project saved successfully.")
            except Exception as exc:
                QMessageBox.critical(self, "Error", f"Failed to save: {exc}")

    def _on_save_vehicle(self) -> None:
        from massprops.io.project_io import save_vehicle
        if self._root is None:
            QMessageBox.information(self, "Save Vehicle", "No model to save.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Vehicle",
            "",
            "Vehicle (*.vehicle.json);;JSON (*.json)",
        )
        if not path:
            return

        try:
            vehicle_path = save_vehicle(self._root, Path(path))
            self._write_vehicle_handoff(vehicle_path)
            QMessageBox.information(
                self,
                "Save Vehicle",
                "Vehicle saved and sent to the trajectory analysis.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Failed to save vehicle: {exc}")

    def _write_vehicle_handoff(self, vehicle_path: Path) -> None:
        if self._vehicle_handoff_path is None:
            return

        self._vehicle_handoff_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "vehicle_path": str(vehicle_path),
            "saved_at": time.time(),
        }
        tmp_path = self._vehicle_handoff_path.with_suffix(
            self._vehicle_handoff_path.suffix + ".tmp"
        )
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        tmp_path.replace(self._vehicle_handoff_path)
