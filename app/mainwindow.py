"""The application window: a document, three views onto it, and nothing else.

Layout follows what every CAD-adjacent tool converges on, because it works:

    tree        what the vehicle is made of, and how it nests
    viewport    what it looks like, at a readable scale
    properties  the selected thing, editable, with derived values
    status bar  the roll-up you watch while editing

The important difference from the previous window is that there *is* a
document. Before, the application was a viewer plus a menu of import commands;
nothing could be inspected because nothing was owned. Here the VehicleModel is
the document, and the tree, the viewport and the editor are all views onto it.
Editing a parm rebuilds only the geometry that changed and updates the rest.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)

from app.parmeditor import ParmEditor
from app.viewport import Viewport
from parametric.components import (
    Component,
    FinSet,
    Motor,
    PointMass,
    Protuberance,
    Stack,
    Tank,
    Wing,
)
from parametric.loft import LoftCache
from parametric.model import VehicleModel


def _shift_stack(stack: Stack, station_m: float) -> None:
    """Move a stack's sections so it begins at the given station.

    New parts are otherwise all created at zero and land inside each other,
    which reads as a single malformed body rather than as several parts.
    """
    low = stack.station_range_m()[0]
    for section in stack.sections:
        section.set("station", section.station_m - low + station_m)
    stack.mark_dirty("shift")
from parametric.standard import (
    basic_rocket,
    boattailed_rocket,
    empty_aircraft,
    empty_rocket,
)

KIND_COLORS = {
    "stack": (0.72, 0.76, 0.82),
    "finset": (0.62, 0.70, 0.80),
}


class MainWindow(QMainWindow):
    """Document-centric main window."""

    def __init__(self, model: VehicleModel | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mudline")
        self.resize(1600, 950)

        self.model = model or basic_rocket()
        self.cache = LoftCache()

        from app.pipeline import Pipeline
        from app.results import ResultStore
        from app.undo import UndoStack

        self.pipeline = Pipeline()
        self.results = ResultStore()
        self.undo_stack = UndoStack()

        from app.project import DocumentState

        self.document = DocumentState()

        self._build_ui()
        self._build_menu()

        # Rebuilds are coalesced: dragging a slider fires many changes a second
        # and each one would otherwise start an OCC rebuild we immediately throw
        # away.
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(60)
        self._rebuild_timer.timeout.connect(self._rebuild_geometry)

        # Autosave. Untitled work is exactly what gets lost, so it is saved
        # to a temp file until the document has a home of its own.
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setInterval(120_000)
        self._autosave_timer.timeout.connect(self._autosave)
        self._autosave_timer.start()

        self.set_model(self.model)
        self._restore_settings()

    # ------------------------------------------------------------------
    # Preferences that outlive the session
    # ------------------------------------------------------------------

    @staticmethod
    def _settings() -> QSettings:
        """Where preferences live -- and, headless, somewhere disposable.

        The test suite builds this window dozens of times. Pointed at the real
        store it would read whatever geometry the developer last left behind,
        which makes a test's behaviour depend on the machine it runs on, and
        it would *write* there too -- a suite that quietly rearranges your
        application is not one you want to run. No display, no persistence.
        """
        from PySide6.QtGui import QGuiApplication

        if QGuiApplication.platformName() in ("offscreen", "minimal"):
            import tempfile

            transient = Path(tempfile.gettempdir()) / "mudline" / "transient.ini"
            transient.parent.mkdir(parents=True, exist_ok=True)
            return QSettings(str(transient), QSettings.IniFormat)

        return QSettings("Mudline", "Mudline")

    def _restore_settings(self) -> None:
        """Window shape, unit system and recent files, from the last session.

        None of this was kept before, so Recent Files was permanently empty and
        anyone working in inches re-chose them on every launch -- small things,
        but each one is a daily reminder that the tool forgets you.
        """
        settings = self._settings()

        geometry = settings.value("window/geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        state = settings.value("window/state")
        if state is not None:
            self.restoreState(state)

        stored = settings.value("units/system")
        if stored:
            from app.units import UnitSystem, set_system

            try:
                system = UnitSystem(stored)
            except ValueError:
                system = None
            if system is not None:
                set_system(system)
                action = self._unit_actions.get(system)
                if action is not None:
                    action.setChecked(True)
                self.editor.set_component(self._current_component())
                self._update_readout()

        recent = settings.value("files/recent") or []
        if isinstance(recent, str):        # a one-entry list comes back bare
            recent = [recent]
        # Files that have since been deleted or moved would otherwise sit in
        # the menu forever, failing when clicked.
        self._recent = [p for p in recent if Path(p).exists()][:8]
        self._rebuild_recent_menu()

    def _store_settings(self) -> None:
        settings = self._settings()
        settings.setValue("window/geometry", self.saveGeometry())
        settings.setValue("window/state", self.saveState())

        from app.units import UNITS

        settings.setValue("units/system", UNITS.system.value)
        settings.setValue("files/recent", list(getattr(self, "_recent", [])))

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Component", "Mass kg"])
        self.tree.setMinimumWidth(260)
        self.tree.setMaximumWidth(420)
        self.tree.setAlternatingRowColors(True)
        self.tree.currentItemChanged.connect(self._on_tree_selection)
        splitter.addWidget(self.tree)

        self.viewport = Viewport()
        splitter.addWidget(self.viewport)

        self.editor = ParmEditor()
        # Wide enough that a label, a value and a slider fit on one line. At the
        # old 320 they did not, and the slider ended up beyond the right edge.
        self.editor.setMinimumWidth(360)
        self.editor.setMaximumWidth(560)
        self.editor.parm_changed.connect(self._on_parm_changed)
        self.editor.section_changed.connect(self._on_section_changed)
        self.editor.fit_failed.connect(
            lambda message: self.statusBar().showMessage(message, 4000)
        )
        splitter.addWidget(self.editor)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([300, 900, 420])
        self.setCentralWidget(splitter)

        from app.massbudget import MassBudget

        self.budget = MassBudget()
        self.budget.mass_overridden.connect(self._on_budget_override)
        dock = QDockWidget("Mass Budget", self)
        dock.setWidget(self.budget)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        dock.hide()          # opt-in; the status bar carries the totals
        self._budget_dock = dock

        from app.resultspanel import ResultsPanel

        self.results_panel = ResultsPanel(self.results)
        self.results_panel.show_trajectory.connect(self._show_trajectory)
        self.results_panel.message.connect(
            lambda text: self.statusBar().showMessage(text, 6000)
        )
        results_dock = QDockWidget("Results", self)
        results_dock.setWidget(self.results_panel)
        results_dock.setAllowedAreas(
            Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea
        )
        self.addDockWidget(Qt.BottomDockWidgetArea, results_dock)
        self.tabifyDockWidget(dock, results_dock)
        results_dock.hide()
        self._results_dock = results_dock

        self.setStatusBar(QStatusBar())
        from app import theme

        self._pipeline_label = QLabel()
        self._pipeline_label.setStyleSheet(
            f"font-family:{theme.MONO_FONT},monospace; color:{theme.TEXT_FAINT};"
            " padding:0 10px;"
        )
        self.statusBar().addPermanentWidget(self._pipeline_label)

        self._mass_label = QLabel()
        self._mass_label.setStyleSheet(
            f"font-family:{theme.MONO_FONT},monospace; color:{theme.TEXT};"
            " padding:0 6px;"
        )
        self.statusBar().addPermanentWidget(self._mass_label)

    def _build_menu(self) -> None:
        bar = self.menuBar()

        file_menu = bar.addMenu("&File")
        self._add_action(file_menu, "Build New &Rocket", self._new_rocket,
                         QKeySequence.New)
        self._add_action(file_menu, "Build New &Aircraft", self._new_aircraft)
        file_menu.addSeparator()
        self._add_action(file_menu, "New &Basic Rocket", self._new_basic)
        self._add_action(file_menu, "New Boat&tail Demo", self._new_boattail)
        file_menu.addSeparator()
        self._add_action(file_menu, "&Open...", self._open, QKeySequence.Open)
        self._add_action(file_menu, "&Save", self._save, QKeySequence.Save)
        self._add_action(file_menu, "Save &As...", self._save_as, QKeySequence.SaveAs)
        self._recent_menu = file_menu.addMenu("Recent &Files")
        file_menu.addSeparator()
        self._add_action(file_menu, "&Import STEP...", self._import_step)
        self._add_action(file_menu, "Export &STEP...", self._export_step)
        file_menu.addSeparator()
        self._add_action(file_menu, "E&xit", self.close)

        edit_menu = bar.addMenu("&Edit")
        self._action_undo = self._add_action(
            edit_menu, "&Undo", self.undo, QKeySequence.Undo
        )
        self._action_redo = self._add_action(
            edit_menu, "&Redo", self.redo, QKeySequence.Redo
        )

        view_menu = bar.addMenu("&View")
        self._add_action(view_menu, "&Frame All", self.viewport.frame_all, "F")
        self._add_action(view_menu, "Toggle &Grid", self._toggle_grid, "G")
        self._add_action(view_menu, "&Cut Away", self._toggle_cutaway, "X")
        self._add_action(view_menu, "Mass &Budget", self._toggle_budget, "B")
        self._add_action(view_menu, "&Results", self._toggle_results, "R")
        view_menu.addSeparator()
        units_menu = view_menu.addMenu("&Units")
        from PySide6.QtGui import QActionGroup

        from app.units import UNITS, UnitSystem

        # Checkable and exclusive: the menu is also where you find out which
        # system you are in, which matters once the choice survives a restart.
        unit_group = QActionGroup(self)
        unit_group.setExclusive(True)
        self._unit_actions: dict[UnitSystem, QAction] = {}
        for system in UnitSystem:
            action = self._add_action(
                units_menu, system.label,
                lambda checked=False, chosen=system: self._set_units(chosen),
            )
            action.setCheckable(True)
            action.setChecked(system is UNITS.system)
            unit_group.addAction(action)
            self._unit_actions[system] = action
        view_menu.addSeparator()
        self._add_action(view_menu, "&Front", lambda: self._set_view(0.0, 0.0), "1")
        self._add_action(view_menu, "&Side", lambda: self._set_view(np.pi / 2, 0.0), "3")
        self._add_action(view_menu, "&Top", lambda: self._set_view(0.0, np.radians(89)), "7")

        model_menu = bar.addMenu("&Model")
        self._add_menu = model_menu.addMenu("&Add")
        self._populate_add_menu()
        self._add_action(model_menu, "&Delete Selected", self._delete_component, "Del")
        model_menu.addSeparator()
        self._add_action(model_menu, "&Validate", self._validate)
        self._add_action(model_menu, "&Rebuild Geometry", self._rebuild_geometry)

        analysis_menu = bar.addMenu("&Analysis")
        self._add_action(analysis_menu, "Solve &Mass Properties", self._solve_mass)
        self._add_action(analysis_menu, "&Aerodynamics...", self._run_aero)
        analysis_menu.addSeparator()
        self._add_action(analysis_menu, "Run &Flight...", self._run_flight)
        self._add_action(analysis_menu, "&Dispersion Study...", self._run_dispersion)
        self._add_action(analysis_menu, "Design &Sweep...", self._run_sweep)
        analysis_menu.addSeparator()
        self._add_action(analysis_menu, "&Compare Aero with RASAero", self._compare_aero)
        analysis_menu.addSeparator()
        self._add_action(analysis_menu, "Export RASAero &Model...", self._export_rasaero)
        self._add_action(analysis_menu, "Export Aero &Table...", self._export_aero_csv)

        help_menu = bar.addMenu("&Help")
        self._add_action(help_menu, "&About Mudline", self._about)
        self._add_action(help_menu, "&Limitations", self._show_limitations)
        help_menu.addSeparator()
        self._add_action(help_menu, "Open &Log Folder", self._open_log_folder)
        self._add_action(help_menu, "Check &Environment", self._check_environment)

    #: What each vehicle class offers, and what it calls it.
    #:
    #: A rocket is built in RASAero's part vocabulary -- nose cone, body tube,
    #: boattail, fin set -- because those are the four shapes its aerodynamics
    #: can describe, so naming the parts anything else would invite a model it
    #: cannot analyse. An aircraft is built in OpenVSP's, for the same reason
    #: pointed at a different tool.
    #:
    #: Both produce ordinary components. The vocabulary is a label on the menu
    #: and a set of sensible starting dimensions, not a separate type system:
    #: a nose cone is a Stack whose sections happen to form a nose, and adding
    #: a wing to a rocket is allowed, because a rocket-plane is a real vehicle.
    ADD_MENUS = {
        "rocket": [
            ("&Nose Cone", "nose"),
            ("&Body Tube", "bodytube"),
            ("&Transition / Boattail", "boattail"),
            (None, None),
            ("&Tank", "tank"),
            ("&Intertank", "intertank"),
            (None, None),
            ("&Fin Set", "finset"),
            ("&Motor", "motor"),
            (None, None),
            ("&Point Mass", "pointmass"),
            ("P&rotuberance", "protuberance"),
        ],
        "aircraft": [
            ("&Fuselage", "fuselage"),
            (None, None),
            ("&Wing", "wing"),
            ("&Horizontal Tail", "htail"),
            ("&Vertical Tail", "vtail"),
            (None, None),
            ("&Tank", "tank"),
            ("&Engine", "motor"),
            ("&Point Mass", "pointmass"),
            ("P&rotuberance", "protuberance"),
        ],
    }

    def _populate_add_menu(self) -> None:
        """Rebuild the Add menu for the current vehicle class."""
        self._add_menu.clear()
        entries = self.ADD_MENUS.get(
            getattr(self.model, "vehicle_class", "rocket"),
            self.ADD_MENUS["rocket"],
        )
        for label, kind in entries:
            if label is None:
                self._add_menu.addSeparator()
            else:
                self._add_action(
                    self._add_menu, label,
                    lambda checked=False, k=kind: self._add_component(k),
                )

    def _add_action(self, menu, text, slot, shortcut=None) -> QAction:
        action = QAction(text, self)
        action.triggered.connect(slot)
        if shortcut is not None:
            action.setShortcut(shortcut)
        menu.addAction(action)
        return action

    # ------------------------------------------------------------------
    # Document
    # ------------------------------------------------------------------

    def set_model(self, model: VehicleModel) -> None:
        self.model = model
        # The panel needs the vehicle only to list what a part may clip to.
        if getattr(self, "editor", None) is not None:
            self.editor.model = model
        self.cache.clear()
        # The Add menu is vocabulary, and the vocabulary belongs to the vehicle
        # in front of you -- so it is rebuilt per document, not once at startup.
        if getattr(self, "_add_menu", None) is not None:
            self._populate_add_menu()
        self._refresh_title()
        self._populate_tree()
        self._solved_mass = None
        self._aero_database = None
        self.pipeline.clear()
        self.results.clear()
        self.undo_stack.reset(model.to_dict())
        self.document.dirty = False
        if getattr(self, "results_panel", None) is not None:
            self.results_panel.set_fingerprint(self._fingerprint())
        self._rebuild_geometry()
        self.budget.set_model(model)
        self.viewport.frame_all()

    def _populate_tree(self) -> None:
        self.tree.clear()
        self._items: dict[int, Component] = {}

        def add(node: Component, parent) -> None:
            mass = node.mass_kg()
            item = QTreeWidgetItem([node.name, f"{mass:.3f}" if mass > 0 else ""])
            item.setData(0, Qt.UserRole, node.path)
            self._items[id(item)] = node
            if parent is None:
                self.tree.addTopLevelItem(item)
            else:
                parent.addChild(item)
            for child in node.children:
                add(child, item)

        add(self.model.root, None)
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)

    def _refresh_tree_masses(self) -> None:
        """Update the mass column in place, without losing selection."""
        iterator = self.tree.findItems("", Qt.MatchContains | Qt.MatchRecursive, 0)
        for item in iterator:
            node = self._items.get(id(item))
            if node is None:
                continue
            mass = node.mass_kg()
            item.setText(1, f"{mass:.3f}" if mass > 0 else "")

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _rebuild_geometry(self) -> None:
        # Locked clips first: a part whose station is derived has to be moved
        # before its solid is built, or the geometry is one edit behind.
        self.model.apply_clips()
        try:
            solids = self.cache.solids(self.model)
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Geometry failed: {exc}", 8000)
            return

        # Colour by material, not by kind. A solid key is a component path,
        # optionally with a "#n" suffix for one of several identical parts, so
        # the path before the suffix is what identifies the owner.
        from parametric.materials import MATERIALS

        by_path = {c.path: c for c in self.model.walk()}
        colors: dict[str, tuple] = {}
        sheens: dict[str, float] = {}
        for key in solids:
            component = by_path.get(key.split("#", 1)[0])
            material = MATERIALS.get(
                component.material if component is not None else ""
            )
            if material is None:
                # An imported material registered without appearance, or a name
                # that no longer resolves. Neutral rather than wrong.
                colors[key] = KIND_COLORS["finset" if "fins" in key else "stack"]
                sheens[key] = 0.4
            else:
                colors[key] = material.color
                sheens[key] = material.sheen

        self.viewport.set_solids(
            solids, self.model.total_length_m, colors, sheens
        )
        self._update_readout(len(self.cache.last_rebuilt))

    def _update_readout(self, rebuilt: int = 0) -> None:
        mass = self.model.mass_summary()
        if getattr(self, "budget", None) is not None:
            self.budget.set_model(self.model, getattr(self, "_solved_mass", None))
        if getattr(self, "_pipeline_label", None) is not None:
            self._pipeline_label.setText(self.pipeline.summary(self.model))
        if getattr(self, "results_panel", None) is not None:
            self.results_panel.set_fingerprint(self._fingerprint())
        margin = self._static_margin()
        from app.units import UNITS

        self._mass_label.setText(
            f"dry {UNITS.format(mass.dry_mass_kg, 'kg', 3)}   "
            f"wet {UNITS.format(mass.wet_mass_kg, 'kg', 3)}   "
            f"CG {UNITS.format(mass.cg_station_m, 'm', 3, prefer_feet=False)}   "
            f"{margin}   "
            f"L {UNITS.format(self.model.total_length_m, 'm', 3, prefer_feet=False)}   "
            f"fineness {self.model.fineness_ratio:4.1f}"
        )
        if rebuilt:
            self.statusBar().showMessage(f"rebuilt {rebuilt} solid(s)", 2500)

    def _static_margin(self) -> str:
        """Static margin in calibres.

        Delegates to parametric.aero, which is the single Barrowman
        implementation. This used to carry its own copy that squared
        2*span/diameter instead of span/diameter -- four times too much fin --
        and reported this vehicle at 4.3 calibres when it has 1.4.
        """
        try:
            from parametric.analysis import static_margin

            cg = getattr(self, "_solved_mass", None)
            from parametric.analysis import loaded_cg_station_m

            calibres = static_margin(
                self.model,
                loaded_cg_station_m(self.model, cg.cg_station_m if cg else None),
            )
        except Exception:  # noqa: BLE001
            return "margin    -"
        return f"margin {calibres:5.2f} cal loaded"

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------

    def _on_tree_selection(self, current, _previous) -> None:
        node = self._items.get(id(current)) if current is not None else None
        self.editor.set_component(node)
        # The root stands for the whole vehicle, so highlighting it would light
        # up every part at once. Treat it as no selection.
        highlight = node.path if (node is not None and node is not self.model.root) else None
        self.viewport.set_selection(highlight)
        if node is not None:
            low, high = node.station_range_m()
            self.viewport.set_status(
                f"{node.name}   {node.kind}   station {low:.3f}-{high:.3f} m"
            )

    def _on_parm_changed(self, component, name: str, value: float) -> None:
        # Labelled by component and parm so a slider drag coalesces into one
        # entry, while editing a different parm starts a new one.
        self._push_undo(f"{component.name}.{name}")
        self._refresh_tree_masses()
        self.editor.refresh_derived()
        # A linked bound belongs to a different row than the one just edited --
        # widen a tube and the wall thickness limit moves with it -- so the
        # other rows re-read their ranges. The edited row is left alone;
        # rewriting it here would fight the pointer mid-drag.
        self.editor.refresh_bounds(exclude=name)
        self._update_readout()
        self._rebuild_timer.start()

    def _on_budget_override(self, component) -> None:
        """A mass edited in the budget sheet, same as ticking Measured mass."""
        self._push_undo(f"{component.name}.measured_mass")
        self._refresh_tree_masses()
        self._update_readout()
        # If that component is open in the property panel, its checkbox and
        # value are now stale.
        if self.editor._component is component:
            self.editor.set_component(component)

    def _toggle_cutaway(self) -> None:
        """Half-section the vehicle so the inside is visible."""
        self.viewport.cutaway = not self.viewport.cutaway
        self.viewport.update()
        self.statusBar().showMessage(
            "Cut-away view on - the far half is hidden."
            if self.viewport.cutaway else "Cut-away view off.",
            3000,
        )

    def _toggle_budget(self) -> None:
        self._budget_dock.setVisible(not self._budget_dock.isVisible())

    def _toggle_results(self) -> None:
        visible = not self._results_dock.isVisible()
        self._results_dock.setVisible(visible)
        if visible:
            self._results_dock.raise_()

    def _show_trajectory(self, result) -> None:
        """Draw a stored run in the viewport, so an old run can be looked at."""
        if result is None:
            return
        self.viewport.set_trajectory(self._trajectory_points(result))
        self.viewport.frame_all()

    @staticmethod
    def _trajectory_points(result) -> np.ndarray:
        """Positions relative to the rail foot, so the path leaves the model.

        A flight from a raised pad is integrated above sea level; drawn as
        it is, it would float that far above the vehicle in the viewport.
        """
        points = np.asarray(result.y, dtype=float).T[:, 0:3]
        pad = getattr(result, "pad_position_m", None)
        return points if pad is None else points - np.asarray(pad, dtype=float)

    def _record_result(self, fields: dict) -> None:
        self.results.add(**fields)
        self.results_panel.set_fingerprint(self._fingerprint())
        self._results_dock.setVisible(True)
        self._results_dock.raise_()

    def _fingerprint(self) -> str:
        from app.pipeline import model_fingerprint

        return model_fingerprint(self.model)

    def _set_units(self, system) -> None:
        """Switch display units. Stored values are untouched and stay SI."""
        from app.units import set_system

        set_system(system)
        self.editor.set_component(self._current_component())
        self._update_readout()
        if getattr(self, "results_panel", None) is not None:
            self.results_panel.refresh()
        self.statusBar().showMessage(f"Units: {system.label}", 4000)

    def _toggle_grid(self) -> None:
        self.viewport.show_grid = not self.viewport.show_grid
        self.viewport.update()

    def _set_view(self, azimuth: float, elevation: float) -> None:
        self.viewport.camera.azimuth = azimuth
        self.viewport.camera.elevation = elevation
        self.viewport.update()

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------

    def _new_rocket(self) -> None:
        """Start an empty rocket, built in RASAero's vocabulary."""
        if not self._confirm_discard():
            return
        self.set_model(empty_rocket())

    def _new_aircraft(self) -> None:
        """Start an empty aircraft, built in OpenVSP's vocabulary."""
        if not self._confirm_discard():
            return
        self.set_model(empty_aircraft())

    def _new_basic(self) -> None:
        if not self._confirm_discard():
            return
        self.set_model(basic_rocket())

    def _new_boattail(self) -> None:
        if not self._confirm_discard():
            return
        self.set_model(boattailed_rocket())

    def _open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Open project", "vehicles",
            "Rocket project or vehicle (*.json)",
        )
        if path:
            self._open_path(path)

    def _open_path(self, path) -> None:
        from app.project import load_project

        try:
            model, results = load_project(path)
        except Exception as exc:  # noqa: BLE001
            self._complain("Open", f"Could not open:\n{exc}")
            return

        self.set_model(model)
        self.results.restore(results)
        if results:
            self.results_panel.set_fingerprint(self._fingerprint())
        self.document.mark_saved(Path(path))
        self._remember_recent(Path(path))
        self._refresh_title()
        self.statusBar().showMessage(
            f"Opened {Path(path).name}"
            + (f" with {len(results)} stored run(s)" if results else ""),
            5000,
        )

    def _save_as(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", f"{self.model.name}.json",
            "Rocket project (*.json)",
        )
        if path:
            self._write_project(path)

    def _import_step(self) -> None:
        """Import CAD as an editable model, not a dead mesh."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import STEP", "", "STEP files (*.stp *.step);;All files (*)"
        )
        if not path:
            return

        # An assembly is read as the assembly it is and assigned by hand; a
        # single solid has nothing to assign and falls back to the profile
        # fitter. The split is on what the file contains, not on a preference:
        # fitting a vehicle that arrived as separate solids is what turned four
        # fins into a body collar of twice the true diameter, quietly.
        from parametric.step_assembly import read_assembly

        self.statusBar().showMessage("Reading assembly...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            read = read_assembly(path)
        except Exception as exc:  # noqa: BLE001
            read = None
            self.statusBar().showMessage(f"Assembly read failed: {exc}", 6000)
        finally:
            QApplication.restoreOverrideCursor()

        if read is not None and len(read.components) > 1:
            if self._import_assembly(read):
                return
            # Cancelled in the dialog. No import, and no silent fall back to a
            # fit that was not asked for.
            self.statusBar().showMessage("Import cancelled.", 3000)
            return

        from parametric.cad_import import import_step

        self.statusBar().showMessage("Slicing solid...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            model, report = import_step(path)
        except Exception as exc:  # noqa: BLE001
            QApplication.restoreOverrideCursor()
            self._complain("Import STEP", f"Could not import:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        # A STEP carries geometry; if it carried no material the mass on
        # screen is a guess, and a guess that looks like a measurement is
        # worse than a question.
        if report.material_source == "defaulted" and self._can_prompt():
            from PySide6.QtWidgets import QInputDialog

            from parametric.materials import MATERIALS

            names = sorted(MATERIALS)
            current = model.stacks[0].material if model.stacks else names[0]
            choice, ok = QInputDialog.getItem(
                self, "Import STEP",
                f"{Path(path).name} declares no material.\n"
                f"Mass is provisional until one is chosen.",
                names, names.index(current) if current in names else 0, False,
            )
            if ok:
                for stack in model.stacks:
                    stack.material = choice
                report.material = choice
                report.material_source = "chosen on import"

        self.set_model(model)
        self._tell("Import STEP", report.text())

    def _import_assembly(self, read) -> bool:
        """Assign the solids of an assembly, then build from the assignment.

        Returns whether the import was carried through; False means the person
        cancelled, which is not an error and must not fall back to a fit.
        """
        from app.assemblydialog import AssemblyAssignDialog
        from parametric.assembly_import import build_model

        dialog = AssemblyAssignDialog(read, self)
        if self._exec_dialog(dialog) != QDialog.Accepted:
            return False

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            model, report = build_model(read, dialog.assignments())
        except Exception as exc:  # noqa: BLE001
            self._complain("Import STEP", f"Could not build the model:\n{exc}")
            return True
        finally:
            QApplication.restoreOverrideCursor()

        self.set_model(model)
        self._tell("Import STEP", report.text())
        return True

    def _refresh_title(self) -> None:
        # The build is in the title because the alternative is asking a
        # colleague which commit they are on, and nobody knows offhand.
        from app.version import build_string

        self.setWindowTitle(
            f"Mudline {build_string()}"
            f"  -  {self.model.name}  [{self.document.title}]"
        )

    def _save(self) -> None:
        if self.document.path is None:
            self._save_as()
            return
        self._write_project(self.document.path)

    def _write_project(self, path) -> None:
        from app.project import save_project

        try:
            save_project(path, self.model, self.results)
        except Exception as exc:  # noqa: BLE001
            self._complain("Save", f"Could not save:\n{exc}")
            return
        self.document.mark_saved(Path(path))
        self._remember_recent(Path(path))
        self._refresh_title()
        self.statusBar().showMessage(f"Saved {path}", 4000)

    def _autosave(self) -> None:
        """Write a recovery copy, quietly."""
        if not self.document.dirty:
            return
        from app.project import autosave_path, save_project

        try:
            save_project(
                autosave_path(self.document.path, self.model.name),
                self.model, self.results,
            )
        except Exception:  # noqa: BLE001
            pass          # autosave must never interrupt what the user is doing

    def _remember_recent(self, path) -> None:
        recent = getattr(self, "_recent", [])
        path = str(path)
        if path in recent:
            recent.remove(path)
        recent.insert(0, path)
        self._recent = recent[:8]
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self) -> None:
        self._recent_menu.clear()
        for path in getattr(self, "_recent", []):
            self._add_action(
                self._recent_menu, Path(path).name,
                lambda checked=False, p=path: self._open_path(p),
            )

    @staticmethod
    def _can_prompt() -> bool:
        """Whether a modal dialog can actually be answered.

        Under the offscreen platform there is no window and no user, so a modal
        blocks forever. Every blocking prompt in this window checks this first.
        """
        from PySide6.QtGui import QGuiApplication

        return QGuiApplication.platformName() not in ("offscreen", "minimal")

    def _report_box(self, icon, title: str, text: str) -> None:
        """Show a report in a dialog, in a font its columns survive.

        Every analysis result that lands here is a fixed-width table -- mass
        statements, coefficient sweeps, the canonical part list. Rendered in
        the proportional UI font those columns shear apart and the report is
        unreadable, so the message body is monospaced while the buttons keep
        the ordinary chrome.

        The headless check is here rather than only in the callers because it
        is the kind of guard that gets forgotten by the next one: ``exec`` on a
        platform with no user does not fail, it waits forever, and the symptom
        is a test run that hangs instead of a test that fails.
        """
        if not self._can_prompt():
            self.statusBar().showMessage(
                f"{title}: {text.splitlines()[0] if text else ''}", 6000
            )
            return

        from app import theme

        box = QMessageBox(self)
        box.setIcon(icon)
        box.setWindowTitle(title)
        box.setText(text)
        box.setTextFormat(Qt.PlainText)
        box.setStyleSheet(
            f"QLabel{{font-family:{theme.MONO_FONT},monospace; font-size:9pt;}}"
        )
        box.exec()

    def _tell(self, title: str, text: str) -> None:
        """An informational dialog, or the status bar when nothing can answer.

        Both of these called *themselves* on the dialog branch, so any result
        shown to a real user recursed until the stack gave out. It survived
        because the tests run under the offscreen platform, which takes the
        status-bar branch -- the branch a person never sees.
        """
        if self._can_prompt():
            self._report_box(QMessageBox.Information, title, text)
        else:
            self.statusBar().showMessage(f"{title}: {text.splitlines()[0]}", 6000)

    def _complain(self, title: str, text: str) -> None:
        """Same, for a failure. Never silent -- it always lands somewhere."""
        if self._can_prompt():
            self._report_box(QMessageBox.Critical, title, text)
        else:
            self.statusBar().showMessage(f"{title} failed: {text.splitlines()[0]}", 8000)

    def _warn(self, title: str, text: str) -> None:
        """A non-blocking warning, safe to raise with no user present."""
        if self._can_prompt():
            QMessageBox.warning(self, title, text)
        else:
            self.statusBar().showMessage(f"{title}: {text.splitlines()[0]}", 8000)

    def _exec_dialog(self, dialog) -> int:
        """Run a setup dialog, or accept its defaults when headless.

        A modal with no user waits forever, so the rule is that headless takes
        the values the dialog was constructed with.
        """
        from PySide6.QtWidgets import QDialog

        if not self._can_prompt():
            return QDialog.Accepted
        return dialog.exec()

    def _confirm_discard(self) -> bool:
        """Ask before throwing away unsaved work. True means carry on."""
        if not self.document.dirty:
            return True
        if not self._can_prompt():
            # No one is there to answer. A modal here waits for a button press
            # that will never come, which wedges the whole process -- so on a
            # headless platform the close is allowed to proceed instead.
            return True
        choice = QMessageBox.question(
            self, "Unsaved changes",
            f"{self.model.name} has unsaved changes.\n\nSave before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if choice == QMessageBox.Cancel:
            return False
        if choice == QMessageBox.Save:
            self._save()
            return not self.document.dirty
        return True

    def closeEvent(self, event) -> None:
        if not self._confirm_discard():
            event.ignore()
            return
        # After the discard prompt, so a cancelled close does not persist a
        # layout the user is about to keep changing.
        try:
            self._store_settings()
        except Exception:      # noqa: BLE001 - never block a close on a preference
            from app.diagnostics import LOGGER

            LOGGER.exception("Could not store settings")
        event.accept()

    def _export_step(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Export STEP parts to")
        if not directory:
            return
        from parametric.loft import export_step

        written = export_step(self.model, directory, self.cache)
        self._tell("Export STEP", f"Wrote {len(written)} parts to\n{directory}"
        )

    # ------------------------------------------------------------------
    # Undo
    # ------------------------------------------------------------------

    def _selected_path(self) -> str | None:
        component = self._current_component()
        return component.path if component is not None else None

    def _push_undo(self, label: str) -> None:
        """Record the model after an edit, and mark the document dirty."""
        self.document.mark_dirty()
        self._refresh_title()
        self.undo_stack.push(self.model.to_dict(), label, self._selected_path())
        self._refresh_undo_actions()

    def _refresh_undo_actions(self) -> None:
        stack = self.undo_stack
        self._action_undo.setEnabled(stack.can_undo)
        self._action_redo.setEnabled(stack.can_redo)
        self._action_undo.setText(
            f"&Undo {stack.undo_label}" if stack.can_undo else "&Undo"
        )
        self._action_redo.setText(
            f"&Redo {stack.redo_label}" if stack.can_redo else "&Redo"
        )

    def undo(self) -> None:
        self._apply_snapshot(self.undo_stack.undo(), "Undo")

    def redo(self) -> None:
        self._apply_snapshot(self.undo_stack.redo(), "Redo")

    def _apply_snapshot(self, snapshot, verb: str) -> None:
        """Replace the model with a recorded state.

        Restoring builds fresh Component objects, so every view holding a
        reference to the old ones has to be repointed. Selection is restored by
        *path* rather than by object identity for the same reason -- the
        component the user was editing still exists conceptually, but it is not
        the same Python object any more.
        """
        if snapshot is None:
            return

        # Keep whatever the user is looking at, falling back to what was
        # selected when the snapshot was taken. Undo should change the model,
        # not move the cursor: jumping the selection because the *opening*
        # snapshot happened to have none is disorienting.
        keep = self._selected_path() or snapshot.selection

        self.undo_stack.applying = True
        try:
            self.model = VehicleModel.from_dict(snapshot.state)
            self.cache.clear()
            self._populate_tree()
            self._rebuild_geometry()
            self._restore_selection(keep)
            self._update_readout()
        finally:
            self.undo_stack.applying = False

        self._refresh_undo_actions()
        self.statusBar().showMessage(f"{verb} {snapshot.label}", 3000)

    def _restore_selection(self, path: str | None) -> None:
        if not path:
            self.editor.set_component(None)
            self.viewport.set_selection(None)
            return
        for node in self.model.walk():
            if node.path == path:
                self._select_component(node)
                return
        self.editor.set_component(None)
        self.viewport.set_selection(None)

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def _unique_name(self, base: str) -> str:
        existing = {node.name for node in self.model.walk()}
        if base not in existing:
            return base
        index = 2
        while f"{base}_{index}" in existing:
            index += 1
        return f"{base}_{index}"

    def _add_component(self, kind: str) -> None:
        """Add a component, attached to the selection where that makes sense."""
        from parametric.xsec import NoseProfile

        selected = self._current_component()
        parent = self.model.root
        if kind in ("finset", "motor", "wing", "htail", "vtail") and isinstance(
            selected, Stack
        ):
            parent = selected   # lifting surfaces and motors attach to a body

        if kind in ("stack", "fuselage"):
            name = "fuselage" if kind == "fuselage" else "body"
            component = Stack(self._unique_name(name), wall_thickness_m=0.003)
            # A new stack starts as a nose plus a tube: an empty one cannot loft
            # and would show as a validation error the moment it is created.
            component.add_nose(NoseProfile.OGIVE, 0.3, 0.1, sections=12)
            component.add_tube(0.5, 0.1, name="tube")
            # ...and it lands behind whatever is already there. Built at station
            # zero it materialised inside the nose of an existing vehicle, which
            # the mould-line overlap check now reports -- correctly, since a new
            # body buried in the old one is not what anyone asked for.
            _shift_stack(component, self.model.station_range_m()[1])
        elif kind == "nose":
            component = Stack(self._unique_name("nose"), wall_thickness_m=0.003)
            component.add_nose(NoseProfile.OGIVE, 0.3, 0.1, sections=12)
        elif kind == "bodytube":
            component = Stack(self._unique_name("body_tube"), wall_thickness_m=0.002)
            # Butt it against whatever is already there, so parts added in
            # sequence stack up instead of piling on top of each other at zero.
            component.add_tube(0.5, 0.1, name="tube")
            _shift_stack(component, self.model.total_length_m)
        elif kind == "boattail":
            # add_transition takes its front diameter from whatever section
            # already exists, and drops its own first section on the assumption
            # that one does. On a brand new Stack there is none, so the front
            # diameter came out zero and a "boattail" was built as a cone
            # expanding from a point -- the opposite shape, with a wall the
            # analytic volume and the revolved solid disagreed about by 361%.
            # Seeding the front section fixes the shape and the volume together.
            from parametric.xsec import XSec, XSecShape

            diameter = self.model.max_diameter_m or 0.10
            component = Stack(self._unique_name("boattail"), wall_thickness_m=0.003)
            component.add_section(
                XSec(0.0, XSecShape.CIRCLE, diameter, name="boattail_fwd")
            )
            component.add_transition(diameter * 1.5, diameter * 0.6,
                                     name="boattail")
            _shift_stack(component, self.model.station_range_m()[1])
        elif kind == "tank":
            # Sized to whatever body it is joining, so it lands as part of the
            # vehicle rather than at some default scale beside it.
            diameter = self.model.max_diameter_m or 0.30
            component = Tank(
                self._unique_name("tank"),
                diameter_m=diameter, barrel_length_m=diameter * 3.0,
                dome_ratio=0.707, wall_thickness_m=max(diameter * 0.01, 0.001),
                station_m=self.model.station_range_m()[1],
            )
        elif kind == "intertank":
            # Structurally a plain barrel, so it is a Stack rather than a type
            # of its own: an intertank carries load between two tanks and holds
            # nothing, which is exactly a body tube with a name that says so.
            diameter = self.model.max_diameter_m or 0.30
            component = Stack(self._unique_name("intertank"),
                              wall_thickness_m=max(diameter * 0.01, 0.001))
            component.add_tube(diameter * 0.6, diameter, name="barrel")
            _shift_stack(component, self.model.station_range_m()[1])
        elif kind in ("wing", "htail", "vtail"):
            host = parent if isinstance(parent, Stack) else (
                self.model.stacks[0] if self.model.stacks else None
            )
            if host is None:
                self.statusBar().showMessage(
                    "Add a fuselage first - a wing attaches to one.", 4000
                )
                return
            parent = host
            low, high = host.station_range_m()
            length = max(high - low, 0.1)
            # A main wing sits near the quarter point of the body and spans
            # several body lengths; a tail sits aft and is a fraction of it.
            if kind == "wing":
                root = length * 0.22
                component = Wing(
                    self._unique_name("wing"),
                    root_chord_m=root, tip_chord_m=root * 0.55,
                    span_m=length * 0.55, sweep_m=root * 0.30,
                    station_m=low + length * 0.30,
                    dihedral_deg=3.0, incidence_deg=1.5,
                )
            elif kind == "htail":
                root = length * 0.12
                component = Wing(
                    self._unique_name("htail"),
                    root_chord_m=root, tip_chord_m=root * 0.6,
                    span_m=length * 0.20, sweep_m=root * 0.35,
                    station_m=max(high - root * 1.4, low),
                    dihedral_deg=0.0, incidence_deg=0.0,
                )
            else:
                root = length * 0.14
                component = Wing(
                    self._unique_name("vtail"),
                    root_chord_m=root, tip_chord_m=root * 0.5,
                    span_m=length * 0.13, sweep_m=root * 0.55,
                    station_m=max(high - root * 1.3, low),
                    dihedral_deg=90.0, incidence_deg=0.0,
                    symmetric=False,          # one panel, straight up
                )
        elif kind == "finset":
            host = parent if isinstance(parent, Stack) else (
                self.model.stacks[0] if self.model.stacks else None
            )
            if host is None:
                # A soft refusal belongs in the status bar. A modal here also
                # deadlocks any headless driver, which is how this was found.
                self.statusBar().showMessage(
                    "Add a body first - fins attach to one.", 4000
                )
                return
            parent = host
            low, high = host.station_range_m()
            root = min(0.2, max(high - low, 0.05) * 0.3)
            component = FinSet(
                self._unique_name("fins"), count=4, root_chord_m=root,
                tip_chord_m=root * 0.5, span_m=host.max_diameter_m * 0.8,
                sweep_m=root * 0.5, thickness_m=0.004,
                station_m=max(high - root, low),
            )
        elif kind == "pointmass":
            component = PointMass(
                self._unique_name("mass"), 0.5, self.model.total_length_m * 0.4
            )
        elif kind == "protuberance":
            component = Protuberance(
                self._unique_name("protuberance"), "rail_button",
                frontal_area_m2=1.5e-4,
                station_m=self.model.total_length_m * 0.7,
                count=2, mass_kg=0.005,
            )
        elif kind == "motor":
            host = parent if isinstance(parent, Stack) else None
            low, high = (host.station_range_m() if host
                         else (0.0, self.model.total_length_m))
            length = max(high - low, 0.1)
            component = Motor(
                self._unique_name("motor"), propellant_mass_kg=1.0,
                station_m=low, length_m=length,
            )
            # Give it a body straight away. A motor added with no case diameter
            # draws nothing, and adding a part that produces no visible result
            # anywhere reads as the command having failed.
            diameter = (
                host.max_diameter_m if host is not None else self.model.max_diameter_m
            ) or 0.10
            component.update(
                case_diameter=diameter * 0.85,
                nozzle_length=length * 0.35,
                nozzle_area=np.pi * (diameter * 0.45) ** 2,
            )
            if host is not None:
                parent = host
        else:
            return

        parent.add(component)
        self._push_undo(f"add {component.name}")
        self._populate_tree()
        self._select_component(component)
        self._rebuild_geometry()
        self.statusBar().showMessage(f"Added {component.name}", 3000)

    def _delete_component(self) -> None:
        component = self._current_component()
        if component is None or component is self.model.root:
            return
        if isinstance(component, Stack) and len(self.model.stacks) <= 1:
            self.statusBar().showMessage(
                "A vehicle needs at least one body.", 4000
            )
            return
        parent = component.parent
        if parent is None:
            return
        parent.remove(component)
        self._push_undo(f"delete {component.name}")
        self._populate_tree()
        self.editor.set_component(None)
        self.viewport.set_selection(None)
        self.cache.clear()
        self._rebuild_geometry()
        self.statusBar().showMessage(f"Deleted {component.name}", 3000)

    def _current_component(self):
        item = self.tree.currentItem()
        return self._items.get(id(item)) if item is not None else None

    def _select_component(self, component) -> None:
        for item, node in self._items.items():
            if node is component:
                for index in range(self.tree.topLevelItemCount()):
                    def walk(entry):
                        yield entry
                        for child_index in range(entry.childCount()):
                            yield from walk(entry.child(child_index))
                    for entry in walk(self.tree.topLevelItem(index)):
                        if id(entry) == item:
                            self.tree.setCurrentItem(entry)
                            return

    def _on_section_changed(self, component) -> None:
        """A cross-section was added, removed or moved."""
        self._push_undo(f"edit {component.name}")
        self._refresh_tree_masses()
        self.editor.refresh_derived()
        # The other direction: Length, Diameter and Station are *read from* the
        # sections, so moving a section station changes the length shown above
        # the table. Without this the two halves of the same panel disagree.
        self.editor.refresh_bounds()
        self._update_readout()
        self._rebuild_timer.start()

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def _analysis_work_dir(self) -> Path:
        import tempfile

        directory = Path(tempfile.gettempdir()) / "mudline" / self.model.name
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _solve_mass(self) -> None:
        """Mesh the geometry and solve mass properties, replacing the estimate."""
        from parametric import analysis

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Meshing and solving...")
        try:
            solved = analysis.solve_mass(self.model, self._analysis_work_dir(), self.cache)
        except Exception as exc:  # noqa: BLE001
            self._complain("Mass Properties", f"Failed:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._solved_mass = solved
        self.pipeline.record("mass", self.model, f"{solved.mass_kg:.3f} kg")
        from app.results import mass_result

        self._record_result(
            mass_result(solved, self.model, self._fingerprint())
        )
        analytic = self.model.mass_summary()
        error = (
            100.0 * (solved.mass_kg - analytic.dry_mass_kg) / analytic.dry_mass_kg
            if analytic.dry_mass_kg > 0 else 0.0
        )
        self._tell("Mass Properties",
            f"{solved.summary()}\n\n"
            f"  Analytic estimate was {analytic.dry_mass_kg:.3f} kg "
            f"({error:+.2f}% difference)\n"
            f"  Static margin  "
            f"{analysis.static_margin(self.model, analysis.loaded_cg_station_m(self.model, solved.cg_station_m)):.2f}"
            f" calibres loaded, "
            f"{analysis.static_margin(self.model, solved.cg_station_m):.2f} at burnout",
        )
        self._update_readout()

    def _export_rasaero(self) -> None:
        from parametric import analysis

        path, _ = QFileDialog.getSaveFileName(
            self, "Export RASAero model",
            str(self._analysis_work_dir() / f"{self.model.name}.cdx1"),
            "RASAero project (*.cdx1)",
        )
        if not path:
            return
        try:
            cg = getattr(self, "_solved_mass", None)
            written, canonical = analysis.write_cdx1(
                self.model, path, cg.cg_station_m if cg else None
            )
        except Exception as exc:  # noqa: BLE001
            self._complain("RASAero", f"Failed:\n{exc}")
            return

        self._tell("RASAero Model",
            f"{canonical.report()}\n\nWrote {written}",
        )

    def _compare_aero(self) -> None:
        """Difference the built-in table against RASAero's, at matched conditions."""
        from parametric import aero, aerocompare

        run = getattr(self, "_rasaero_run", None)
        if run is None:
            self.statusBar().showMessage("Run RASAero II first.", 4000)
            return

        # Built fresh rather than reusing whatever the Aerodynamics dialog last
        # produced, for two reasons: RASAero works at the launch-site altitude
        # written into the project, so a table swept at 3 km is not the same
        # experiment; and the transonic feature is a few hundredths of a Mach
        # wide, so a 40-point sweep smears the very thing being compared.
        # Analytic evaluations are cheap enough that neither is worth saving.
        try:
            settings = aero.AeroSettings(altitude_m=0.0, mach_points=200)
            built_in, geometry = aero.run_analysis(self.model, settings)
            self._builtin_database = built_in
        except Exception as exc:  # noqa: BLE001
            self._complain("Compare", f"Failed:\n{exc}")
            return

        solved = getattr(self, "_solved_mass", None)
        cg = (
            solved.cg_station_m if solved
            else self.model.mass_summary().cg_station_m
        )
        comparison = aerocompare.compare(
            built_in, run.database,
            diameter_m=geometry.reference_diameter_m,
            cg_station_m=cg,
        )

        # A table of thirteen Mach numbers cannot show *where* two methods
        # part company, and for a transonic feature a few points wide that is
        # the only interesting question. The curves are cheap; draw them.
        plots: list = []
        try:
            from parametric import aeroplots

            plots = aeroplots.write_comparison_plots(
                built_in, run.database, self._analysis_work_dir(),
                stem=self.model.name.replace(" ", "_"),
                settings=aeroplots.PlotSettings(mach_max=5.0),
            )
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Plots unavailable: {exc}", 5000)

        text = comparison.report()
        if plots:
            text += "\n\n  plots\n" + "\n".join(f"    {path}" for path in plots)
        self._tell("Built-in vs RASAero", text)

    def _run_aero(self) -> None:
        """Build a coefficient table from the geometry."""
        from app.analysisdialogs import AeroSetupDialog
        from parametric import aero

        dialog = AeroSetupDialog(
            getattr(self, "_aero_settings", None), self,
            nozzle_exit_diameter_m=self.model.nozzle_exit_diameter_m(),
            boattail_half_angle_deg=aero.steepest_boattail_deg(self.model),
        )
        if self._exec_dialog(dialog) != QDialog.Accepted:
            return
        settings = dialog.settings()

        solved = getattr(self, "_solved_mass", None)
        cg = solved.cg_station_m if solved else None
        # Only the out-of-process application takes this branch; the
        # built-in engine is just another run_analysis method.
        use_rasaero = getattr(settings, "method", None) == "rasaero-app"

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage(
            "Driving RASAero II..." if use_rasaero else "Sweeping Mach and alpha..."
        )
        run = None
        try:
            # The geometry measurements are wanted either way -- the report
            # quotes them, and they are what a stale check compares -- so they
            # are taken even when RASAero produces the coefficients.
            geometry = aero.extract_geometry(self.model)
            if use_rasaero:
                from parametric import rasaero_run

                run = rasaero_run.run(
                    self.model, self._analysis_work_dir(), cg, settings=settings
                )
                database = run.database
                # The application's table stops at 4 degrees; the flight's
                # extension beyond it needs the same vehicle the engine
                # path would have attached.
                database.high_alpha = aero.high_alpha_geometry(self.model, geometry)
            else:
                database, geometry = aero.run_analysis(self.model, settings)
        except Exception as exc:  # noqa: BLE001
            self._complain("Aerodynamics", f"Failed:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        self._aero_settings = settings
        self._aero_database = database
        self._aero_geometry = geometry
        self._rasaero_run = run
        source = "RASAero II (application)" if use_rasaero else "RASAero (built in)"
        self.pipeline.record(
            "aero", self.model, f"{source}, {len(database.rows)} rows"
        )
        from app.results import aero_result

        self._record_result(aero_result(
            database, geometry, settings, self.model, self._fingerprint(), cg,
        ))

        report = aero.analysis_report(self.model, database, geometry, cg, settings=settings)
        if run is not None:
            report = f"{run.report()}\n\n{report}"
        # Replace the progress message rather than leaving it: "Driving
        # RASAero II..." sitting in the status bar after the run has finished
        # says the tool is still working when it is not.
        self.statusBar().showMessage(
            f"Aerodynamics: {len(database.rows)} rows from {source}.", 6000
        )
        self._tell(f"Aerodynamic Analysis ({source})", report)
        self._update_readout()

    def _export_aero_csv(self) -> None:
        database = getattr(self, "_aero_database", None)
        if database is None:
            self.statusBar().showMessage("Run Aerodynamics first.", 4000)
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export aero table",
            str(self._analysis_work_dir() / f"{self.model.name}_aero.csv"),
            "Coefficient table (*.csv)",
        )
        if path:
            database.to_csv(path)
            self.statusBar().showMessage(f"Wrote {path}", 4000)

    def _run_flight(self) -> None:
        """Fly the vehicle with settings the user chose."""
        from app.analysisdialogs import FlightSetupDialog
        from parametric.flight import fly_model

        database = getattr(self, "_aero_database", None)
        dialog = FlightSetupDialog(
            getattr(self, "_flight_settings", None),
            has_aero=database is not None, parent=self,
        )
        if self._exec_dialog(dialog) != QDialog.Accepted:
            return
        settings = dialog.settings()
        self._flight_settings = settings

        # Refuse to quietly fly on coefficients built for a different vehicle.
        if settings.use_aero_table and database is not None:
            if not self._accept_stale_aero():
                return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Flying...")
        try:
            def progress(message: str) -> None:
                self.statusBar().showMessage(message)
                QApplication.processEvents()

            # The same launch sequence the sweep and the dispersion study
            # fly -- pad altitude, wind and the coupled-aero rebuild included.
            outcome = fly_model(
                self.model, settings, database,
                solved=getattr(self, "_solved_mass", None),
                aero_settings=getattr(self, "_aero_settings", None),
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            self._complain("Flight", f"Failed:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        result = outcome.result
        stats = outcome.stats
        using_table = outcome.used_table
        coupled_passes = outcome.coupled_passes
        # The table the flight settled on is the one a dispersion of this
        # flight should reuse, so it is kept with the model it describes.
        self._flown_database = outcome.database
        self._flown_fingerprint = self._fingerprint()

        self.viewport.set_trajectory(outcome.positions_from_pad())
        self.viewport.frame_all()
        self._last_flight = result
        landed = outcome.landed
        apogee = outcome.apogee_agl_m
        note = f"apogee {apogee:,.0f} m"
        if coupled_passes:
            note += f", table coupled x{coupled_passes}"
        if not landed:
            note += ", did not reach the ground"
        self.pipeline.record("flight", self.model, note)
        peak = outcome.peak
        # The log replays the force model along the flown states: what the
        # integrator computed and threw away, now kept with the run.
        self.statusBar().showMessage("Logging the flight...")
        QApplication.processEvents()
        log = outcome.log

        from app.results import flight_result

        caveats = outcome.caveats()
        self._record_result(flight_result(
            result, stats, peak, settings, self._fingerprint(), using_table, log=log,
            caveats=caveats,
        ))
        self._update_readout()

        exit_state = outcome.rail_exit
        rail_text = (
            f"{exit_state['velocity_mps']:.1f} m/s at t = {exit_state['time_s']:.2f} s, "
            f"alpha {exit_state.get('alpha_deg', 0.0):.1f}°"
            if exit_state else "-"
        )
        notes = caveats
        notes_text = ("\n\n" + "\n".join(notes)) if notes else ""
        pad_text = (
            f" above the pad ({settings.pad_altitude_m:,.0f} m ASL)"
            if settings.pad_altitude_m > 0 else ""
        )
        burnout = log.burnout_index
        margin_text = ""
        if burnout is not None and np.isfinite(log.static_margin_cal[burnout]):
            margin_text = (
                f"Margin        {log.min_static_margin_cal():.2f} cal lowest in boost, "
                f"{log.static_margin_cal[burnout]:.2f} at burnout\n"
            )
        self._tell("Flight",
            f"Apogee        {apogee:,.0f} m{pad_text} at "
            f"t = {stats['apogee_time']:.1f} s\n"
            f"Max speed     {stats['max_velocity']:,.0f} m/s\n"
            f"Max-Q         {peak['pressure_pa'] / 1000:,.0f} kPa at "
            f"Mach {peak['mach']:.2f}\n"
            f"Max accel     {log.max_acceleration_g:.1f} g at "
            f"t = {log.max_acceleration_time_s:.1f} s\n"
            f"{margin_text}"
            f"Rail exit     {rail_text}\n"
            f"Downrange     {stats['range']:,.0f} m\n"
            f"Flight time   {stats['flight_time']:.0f} s"
            f"{'' if landed else '  (stopped in the air: never reached the ground)'}\n"
            f"Phases        {', '.join(p['name'] for p in result.phases)}\n\n"
            f"Aero          "
            f"{'coefficient table' if using_table else 'fallback drag law'}"
            f"{notes_text}",
        )
        if notes:
            self.statusBar().showMessage(notes[0], 10000)

    def _accept_stale_aero(self) -> bool:
        """Refuse, or ask, before flying on a table built for another vehicle."""
        warning = self.pipeline.stale_warning("aero", self.model)
        if warning is None:
            return True
        if not self._can_prompt():
            # Nobody to accept the risk, so take the safe answer: do not
            # fly on coefficients built for a different vehicle.
            self.statusBar().showMessage(f"Stale aerodynamics: {warning}", 8000)
            return False
        choice = QMessageBox.warning(
            self, "Stale aerodynamics",
            f"{warning}\n\nFly anyway with the old table?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return choice == QMessageBox.Yes

    def _table_for_this_model(self):
        """The best coefficient table for the model as it stands.

        The coupled table a flight settled on, when that flight was of this
        very model; otherwise the aero stage's table; otherwise nothing.
        """
        flown = getattr(self, "_flown_database", None)
        if (flown is not None
                and getattr(self, "_flown_fingerprint", None) == self._fingerprint()):
            return flown
        return getattr(self, "_aero_database", None)

    def _run_dispersion(self) -> None:
        """Fly the open vehicle many times, perturbed, and report where it lands."""
        from app.analysisdialogs import DispersionSetupDialog
        from parametric.dispersion import ModelCaseRunner
        from parametric.flight import FlightSettings
        from trajectory.analysis.dispersion import run_dispersion

        settings = getattr(self, "_flight_settings", None) or FlightSettings()
        solved = getattr(self, "_solved_mass", None)
        database = self._table_for_this_model()
        if not settings.use_aero_table:
            database = None
        elif database is not None and database is getattr(self, "_aero_database", None):
            # The aero stage's own table can be stale; a table coupled on
            # this very model cannot.
            if not self._accept_stale_aero():
                return
        dry_mass = solved.mass_kg if solved else self.model.mass_summary().dry_mass_kg

        dialog = DispersionSetupDialog(
            settings, dry_mass, has_aero=database is not None, parent=self,
        )
        if self._exec_dialog(dialog) != QDialog.Accepted:
            return

        # Every case rebuilds this vehicle and flies it through the same
        # sequence Run Flight uses. Without a case function the library
        # flies its own placeholder vehicle, which is what this used to do
        # -- and then filed the result against the open model.
        runner = ModelCaseRunner(self.model, settings, database, solved)

        # A case is a second or two of flying and a study is hundreds of them,
        # so this is the one analysis long enough that a frozen window reads as
        # a hung application -- which is how it was reported. A progress dialog
        # with a working Cancel, pumped from the per-case callback.
        from PySide6.QtWidgets import QProgressDialog

        requested = dialog.cases.value()
        progress_dialog = QProgressDialog(
            "Flying case 1...", "Cancel", 0, requested, self
        )
        progress_dialog.setWindowTitle("Dispersion Study")
        progress_dialog.setWindowModality(Qt.WindowModal)
        # Headless has no one to watch it and no one to press Cancel.
        progress_dialog.setMinimumDuration(0 if self._can_prompt() else 10_000_000)

        def tick(done: int, total: int) -> bool:
            progress_dialog.setMaximum(total)
            progress_dialog.setValue(done)
            progress_dialog.setLabelText(f"Flown {done} of {total} cases...")
            QApplication.processEvents()
            return not progress_dialog.wasCanceled()

        self.statusBar().showMessage("Running dispersion...")
        try:
            result = run_dispersion(
                n_cases=requested,
                dispersions=dialog.dispersions(),
                seed=dialog.seed.value(),
                n_processes=dialog.processes.value(),
                case_fn=runner,
                progress=tick,
            )
        except Exception as exc:  # noqa: BLE001
            # Cancelling before the first case lands is a choice, not a fault.
            if progress_dialog.wasCanceled():
                self.statusBar().showMessage("Dispersion cancelled.", 4000)
            else:
                self._complain("Dispersion", f"Failed:\n{exc}")
            return
        finally:
            progress_dialog.close()

        flown = len(result.cases)
        if flown < requested:
            self._warn(
                "Dispersion",
                f"Stopped after {flown} of {requested} cases.\n\n"
                "The statistics below are over the cases that flew, so they "
                "are wider than the full study would have been.",
            )

        self._dispersion = result
        from app.results import dispersion_result

        self._record_result(dispersion_result(
            # The cases that actually flew, not the number asked for: a
            # cancelled study filed under its requested count would read
            # afterwards as a full study that happened to be noisy.
            result, self._fingerprint(),
            flown, dialog.seed.value(),
            used_table=database is not None,
        ))
        self.pipeline.record("dispersion", self.model)
        self._tell("Dispersion Study", result.report())

    def _run_sweep(self) -> None:
        """Vary one parameter and record what it costs."""
        from PySide6.QtWidgets import QProgressDialog

        from app.sweepdialog import SweepSetupDialog, sweep_result
        from parametric.sweep import run_sweep

        has_aero = getattr(self, "_aero_database", None) is not None
        flight_settings = getattr(self, "_flight_settings", None)
        couples = has_aero and (
            flight_settings is None or flight_settings.couple_aero_altitude
        )
        dialog = SweepSetupDialog(
            self.model, has_aero=has_aero, couples_aero=couples, parent=self,
        )
        if self._exec_dialog(dialog) != QDialog.Accepted:
            return
        settings = dialog.settings()
        settings.aero_settings = getattr(self, "_aero_settings", None)
        settings.flight_settings = getattr(self, "_flight_settings", None)

        progress = QProgressDialog(
            "Sweeping...", "Cancel", 0, settings.variable.steps, self
        )
        progress.setWindowModality(Qt.WindowModal)
        # Headless has no one to watch it and no one to press Cancel.
        progress.setMinimumDuration(0 if self._can_prompt() else 10_000_000)

        def tick(index: int, total: int, value: float) -> bool:
            progress.setMaximum(total)
            progress.setValue(index)
            progress.setLabelText(
                f"{settings.variable.label} = {value:.4g}   "
                f"({index + 1} of {total})"
            )
            QApplication.processEvents()
            return not progress.wasCanceled()

        # The sweep mutates the model to evaluate each point. Undo must not
        # record those intermediate states, and the stack would otherwise fill
        # with a point per step.
        self.undo_stack.applying = True
        try:
            result = run_sweep(self.model, settings, progress=tick)
        except Exception as exc:  # noqa: BLE001
            self._complain("Design Sweep", f"Failed:\n{exc}")
            return
        finally:
            self.undo_stack.applying = False
            progress.close()

        if not result.points:
            self.statusBar().showMessage("Sweep cancelled.", 4000)
            return

        self._record_result(sweep_result(result, self._fingerprint()))
        self._rebuild_geometry()
        self._update_readout()
        self._tell("Design Sweep", result.report())

    def _validate(self) -> None:
        problems = self.model.validate()
        if not problems:
            self._tell("Validate", "No problems found.")
        else:
            self._warn("Validate", "\n\n".join(f"• {p}" for p in problems))

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _about(self) -> None:
        """What build this is, and what it was built against.

        Written to be read out loud into a bug report: the version, the commit,
        and the versions of the libraries whose disagreements have historically
        been the difference between a working install and a broken one.
        """
        from importlib.metadata import PackageNotFoundError, version

        from app.diagnostics import log_path
        from app.version import __version__, git_revision

        lines = [
            f"Mudline {__version__}",
            f"revision  {git_revision() or 'unknown (not a checkout)'}",
            "",
            f"python    {sys.version.split()[0]}",
        ]
        for package in ("numpy", "scipy", "matplotlib", "PySide6",
                        "moderngl", "cadquery"):
            try:
                lines.append(f"{package:<10}{version(package)}")
            except PackageNotFoundError:
                lines.append(f"{package:<10}not installed")
        lines += ["", f"log       {log_path() or 'not started'}"]
        self._report_box(QMessageBox.Information, "About", "\n".join(lines))

    def _show_limitations(self) -> None:
        """What the tool does not model -- plus what is wrong with this vehicle.

        Kept one keystroke away rather than in a document nobody opens, because
        the failure this guards against is someone trusting a number the model
        was never able to produce.
        """
        from app.limitations import limitations_report

        self._report_box(
            QMessageBox.Information, "Limitations", limitations_report(self.model)
        )

    def _open_log_folder(self) -> None:
        from app.diagnostics import _reveal, log_directory, log_path

        _reveal(log_directory())
        self.statusBar().showMessage(f"Log: {log_path()}", 8000)

    def _check_environment(self) -> None:
        """Run the install check and show its output.

        The same command CI runs and the same one a teammate runs after
        installing -- here too, because the person who needs it most is the one
        who has just watched something not work and has no terminal open.
        """
        import subprocess

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Checking the environment...")
        try:
            finished = subprocess.run(
                [sys.executable, "-m", "tools.check_environment"],
                capture_output=True, text=True, timeout=300,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            text = finished.stdout or finished.stderr or "(no output)"
        except Exception as exc:      # noqa: BLE001
            self._complain("Check Environment", f"Could not run the check:\n{exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.statusBar().clearMessage()

        icon = QMessageBox.Warning if finished.returncode else QMessageBox.Information
        self._report_box(icon, "Environment", text)
