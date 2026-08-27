"""Editable property panel for the selected component.

Every row is generated from a Parm, so the editor knows nothing about rockets.
Adding a parameter to a component makes it appear here, with the right range,
units and label, without touching this file. That is the payoff for making
values Parms instead of floats.

Editing is live: dragging a slider emits on every step so geometry and mass
update as you move, which is the loop the previous read-only panel could not
support at all.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from parametric.components import (
    Component,
    FinSet,
    Motor,
    PointMass,
    Stack,
    Tank,
)
from parametric.xsec import XSecShape

SLIDER_STEPS = 1000

#: A hard range wider than this is a sentinel meaning "uncapped" rather than a
#: real engineering limit, so the slider covers a window of it instead of all
#: of it. Every authored bound in the model is far below this; every uncapped
#: one is far above.
UNCAPPED_SPAN = 1000.0


#: Longest description that still reads as a field label rather than a sentence.
MAX_LABEL_CHARS = 20


def parm_label(parm, unit_label: str = "") -> str:
    """A human label for a parm row, in RASAero's style.

    RASAero writes "Nose Tip Radius (in)" -- a short phrase with the unit
    attached -- where this panel used to write "nose tip radius" from the parm
    name and hide the unit inside the spin box.

    The description is the better phrase when it is short enough to be one
    ("Root chord"), and the wrong thing entirely when it is really a sentence
    ("Angle of the chord to the body axis"), which would push the value column
    halfway across the panel. Past a length the parm name wins, and the full
    description stays as the tooltip either way.
    """
    text = (parm.description or "").strip()
    for cut in (";", ".", " -- ", ","):
        if cut in text:
            text = text.split(cut, 1)[0].strip()
    if not text or len(text) > MAX_LABEL_CHARS:
        text = parm.name.replace("_", " ").strip()
    if text:
        text = text[0].upper() + text[1:]
    return f"{text} ({unit_label})" if unit_label else text


def section_header(title: str) -> QLabel:
    """A group heading inside the property panel."""
    from app import theme

    label = QLabel(title.upper())
    font = label.font()
    font.setBold(True)
    font.setPointSize(max(font.pointSize() - 1, 7))
    label.setFont(font)
    label.setStyleSheet(
        f"color:{theme.TEXT_FAINT}; letter-spacing:1px;"
        f"border-bottom:1px solid {theme.BORDER_SOFT};"
        "padding-bottom:3px; margin-top:6px;"
    )
    return label


class ParmRow(QWidget):
    """A spin box and slider bound to one Parm."""

    changed = Signal(str, float)

    def __init__(self, parm, parent=None):
        super().__init__(parent)
        self._parm = parm
        self._guard = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Component dimensions are always small, so a length here reads in
        # inches rather than feet.
        from app.units import UNITS

        self._units = UNITS
        self._scale = UNITS.factor(parm.unit, prefer_feet=False)

        # Before anything reads a range: the slider window is derived from the
        # value, and the step size is derived from the window.
        self._soft_low, self._soft_high = self._soft_bounds()

        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(self._decimals())
        self._spin.setRange(
            self._display_min() * self._scale, self._display_max() * self._scale
        )
        self._spin.setSingleStep(self._step() * abs(self._scale))
        self._spin.setValue(parm.value * self._scale)
        # The unit goes in the row's label, not in the box. RASAero labels its
        # fields "Diameter (in)" and leaves the number alone, which is easier to
        # scan down a column and gives the value more room -- a chord in
        # millimetres needs the digits more than it needs a repeated "mm".
        self.unit_label = UNITS.unit_label(parm.unit, prefer_feet=False)
        # Both controls must be able to shrink. The panel is a few hundred
        # pixels wide, and a row that refuses to compress pushes the spin box
        # and slider off the right-hand edge -- which does not look like a
        # layout bug from the outside, it looks like an editor that does
        # nothing, because the only parts you can actually operate are the ones
        # that scrolled out of sight.
        self._spin.setMinimumWidth(78)
        self._spin.setMaximumWidth(120)
        self._spin.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._spin.valueChanged.connect(self._on_spin)
        layout.addWidget(self._spin)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, SLIDER_STEPS)
        self._slider.setEnabled(self._bounded())
        self._slider.setMinimumWidth(40)
        self._slider.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        if self._bounded():
            span = self._soft_high - self._soft_low
            fraction = 0.0 if span <= 0 else (parm.value - self._soft_low) / span
            self._slider.setValue(int(np.clip(fraction, 0.0, 1.0) * SLIDER_STEPS))
        self._slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self._slider, 1)

        if parm.description:
            self.setToolTip(f"{parm.description}\n[{parm.minimum:g}, {parm.maximum:g}]")

    # ------------------------------------------------------------------

    @property
    def parm_name(self) -> str:
        return self._parm.name

    def _bounded(self) -> bool:
        # bool(), not the numpy scalar: Qt's setEnabled rejects numpy.bool_.
        return bool(np.isfinite(self._parm.lower) and np.isfinite(self._parm.upper))

    def _display_min(self) -> float:
        return self._parm.lower if np.isfinite(self._parm.lower) else -1e9

    def _display_max(self) -> float:
        return self._parm.upper if np.isfinite(self._parm.upper) else 1e9

    def _soft_bounds(self, value: float | None = None) -> tuple[float, float]:
        """The span the *slider* covers, which is not the span the value may take.

        The hard bound on a chord is now 10 km, because a length should not be
        capped at whatever seemed generous when it was declared. Mapping a
        slider across that makes it useless -- every real rocket lives in the
        first pixel. So the slider works over a window around the current value
        and the spin box keeps the full range, which is what "uncapped" has to
        mean in practice: type any number, drag within a useful one.

        A range that is already sensible for its value is left alone, so a
        dihedral bounded -30..60 still gets its whole travel.
        """
        low, high = self._parm.lower, self._parm.upper
        if not (np.isfinite(low) and np.isfinite(high)):
            return low, high
        value = self._parm.value if value is None else value
        span = high - low
        # An authored bound is a small number that means something -- a cant of
        # +/-15 degrees, a taper of 0.01..0.95, four to twelve fins. An uncapped
        # one is a sentinel in the thousands. Telling them apart by the size of
        # the span is what keeps a real range at full travel while a nominal one
        # gets a window; judging by the value instead gave a cant sitting at
        # zero a slider of +/-0.009 degrees.
        if span < UNCAPPED_SPAN:
            return low, high
        # Scale comes from the value. A parm sitting at zero has no scale of its
        # own, so fall back to a small fraction of its range.
        reference = abs(value)
        if reference < 1e-9:
            reference = max(span * 1e-4, 1e-6)
        if low < 0 < high:
            # Symmetric about zero: centre the window on the value rather than
            # measuring from a floor that is 10 km away.
            half = max(reference * 3.0, 1e-6)
            return max(low, value - half), min(high, value + half)
        return low, min(high, low + max(reference * 4.0, 1e-6))

    def _sync_soft_bounds(self, value: float) -> None:
        """Grow the slider window when the value is driven outside it."""
        if not self._bounded():
            return
        if self._soft_low <= value <= self._soft_high:
            return
        self._soft_low, self._soft_high = self._soft_bounds(value)
        if value > self._soft_high:
            self._soft_high = min(self._parm.upper, value * 1.25 or 1e-6)

    def _decimals(self) -> int:
        span = abs(self._parm.value) or 1.0
        return 4 if span < 10 else 2

    def _step(self) -> float:
        if self._bounded():
            return max((self._soft_high - self._soft_low) / 200.0, 1e-4)
        return 0.01

    # ------------------------------------------------------------------

    def _on_spin(self, value: float) -> None:
        if self._guard:
            return
        # The box shows display units; the model only ever stores SI.
        si_value = value / self._scale if self._scale else value
        self._guard = True
        if self._bounded():
            self._sync_soft_bounds(si_value)
            span = self._soft_high - self._soft_low
            fraction = 0.0 if span <= 0 else (si_value - self._soft_low) / span
            self._slider.setValue(int(np.clip(fraction, 0.0, 1.0) * SLIDER_STEPS))
        self._guard = False
        self.changed.emit(self._parm.name, si_value)

    def _on_slider(self, position: int) -> None:
        if self._guard or not self._bounded():
            return
        self._guard = True
        span = self._soft_high - self._soft_low
        value = self._soft_low + span * position / SLIDER_STEPS
        self._spin.setValue(value * self._scale)
        self._guard = False
        self.changed.emit(self._parm.name, value)

    def refresh_bounds(self) -> None:
        """Re-read the bounds, because a linked one moves when its partner does.

        A wall thickness limited to half the diameter is correct in the model
        the instant the diameter changes, and wrong in the widget until someone
        says so: the spin box keeps the range it was built with, so widening a
        tube from 100 to 600 mm left the wall still refusing to pass 50 mm.
        """
        self._guard = True
        try:
            value = self._parm.value
            # Soft bounds first -- the step size is derived from them.
            self._soft_low, self._soft_high = self._soft_bounds()
            self._sync_soft_bounds(value)
            self._spin.setRange(
                self._display_min() * self._scale,
                self._display_max() * self._scale,
            )
            self._spin.setSingleStep(self._step() * abs(self._scale))
            self._spin.setValue(value * self._scale)
            if self._bounded():
                span = self._soft_high - self._soft_low
                fraction = 0.0 if span <= 0 else (value - self._soft_low) / span
                self._slider.setValue(int(np.clip(fraction, 0.0, 1.0) * SLIDER_STEPS))
        finally:
            self._guard = False

    def refresh(self) -> None:
        self._guard = True
        value = self._parm.value
        # Linked bounds move when their partner does -- a wall thickness limit
        # follows the diameter -- so the range is re-read, not just the value.
        self._spin.setRange(
            self._display_min() * self._scale, self._display_max() * self._scale
        )
        self._spin.setValue(value * self._scale)
        self._soft_low, self._soft_high = self._soft_bounds()
        self._sync_soft_bounds(value)
        if self._bounded():
            span = self._soft_high - self._soft_low
            fraction = 0.0 if span <= 0 else (value - self._soft_low) / span
            self._slider.setValue(int(np.clip(fraction, 0.0, 1.0) * SLIDER_STEPS))
        self._guard = False


class ParmEditor(QScrollArea):
    """Shows the parms of whichever component is selected."""

    parm_changed = Signal(object, str, float)
    section_changed = Signal(object)
    #: A soft refusal for the status bar. A modal here would deadlock a
    #: headless driver, which is how the equivalent in the tree was found.
    fit_failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        # Never scroll sideways. A property panel is a column of rows; if a row
        # does not fit it has to wrap or compress, because anything parked off
        # the right edge is functionally missing.
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # A left rule against the viewport. Without it the panel and the 3D
        # view are the same colour with no edge between them, and an empty
        # panel reads as more viewport rather than as somewhere to look.
        from app import theme

        self.setStyleSheet(
            f"ParmEditor {{ border-left: 1px solid {theme.BORDER}; }}"
        )
        self._component: Component | None = None
        #: The vehicle, needed only to list what a part may be clipped to.
        self.model = None
        self._rows: list[ParmRow] = []
        self._rebuild(None)

    # ------------------------------------------------------------------

    def _rebuild_later(self, component: Component | None) -> None:
        """Rebuild the panel once the current signal has finished delivering.

        Rebuilding replaces the scroll area's widget, which deletes every
        control inside it -- including whichever combo box or button is at that
        moment still emitting the signal that got us here. Qt does not survive
        having a sender destroyed mid-emission: the process dies natively, with
        no Python traceback to show for it. Deferring by one event-loop turn
        lets the signal finish first.
        """
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, lambda: self._rebuild(component))

    def set_component(self, component: Component | None) -> None:
        self._component = component
        self._rebuild(component)

    def refresh(self) -> None:
        for row in self._rows:
            row.refresh()
        self._refresh_derived_values()
        for row, _parm, _getter in getattr(self, "_derived", []):
            row.refresh()

    def _refresh_derived_values(self) -> None:
        """Pull derived values back off the component; nothing else owns them."""
        for _row, parm, getter in getattr(self, "_derived", []):
            if self._component is None:
                continue
            try:
                parm.value = float(getter(self._component))
            except Exception:
                continue

    def refresh_bounds(self, exclude: str = "") -> None:
        """Re-read every row's bounds after one of them changed.

        Linked bounds are relationships, so moving one value moves another
        value's limits. ``exclude`` skips the row that caused the change --
        rewriting the control mid-drag would fight the pointer.
        """
        for row in self._rows:
            if row.parm_name != exclude:
                row.refresh_bounds()
        self._refresh_derived_values()
        for row, parm, _getter in getattr(self, "_derived", []):
            if parm.name != exclude:
                row.refresh_bounds()

    # ------------------------------------------------------------------

    def _imported_banner(self, component) -> QWidget:
        """What an imported part shows instead of editable rows."""
        from app import theme

        panel = QWidget()
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(22, 26, 22, 22)
        panel_layout.setSpacing(7)

        heading = QLabel(component.name)
        heading_font = heading.font()
        heading_font.setBold(True)
        heading_font.setPointSize(heading_font.pointSize() + 2)
        heading.setFont(heading_font)
        panel_layout.addWidget(heading)

        kind = QLabel(f"{component.kind}  ·  imported from CAD")
        kind.setStyleSheet(f"color:{theme.TEXT_DIM};")
        panel_layout.addWidget(kind)

        facts = []
        try:
            low, high = component.station_range_m()
            facts.append(f"station    {low:.3f} -> {high:.3f} m")
        except Exception:
            pass
        try:
            facts.append(f"mass       {component.mass_kg():.3f} kg")
        except Exception:
            pass
        facts.append(f"material   {component.material}")
        facts.append(f"aero role  {component.aero_role.value}")

        values = QLabel("\n".join(facts))
        values.setStyleSheet(f"color:{theme.TEXT_DIM}; font-family: monospace;")
        panel_layout.addWidget(values)

        hint = QLabel(
            "This part came from a STEP assembly, so its geometry is not "
            "editable here and a design sweep will not drive it. Re-import "
            "the file to change it."
        )
        hint.setStyleSheet(f"color:{theme.TEXT_FAINT};")
        hint.setWordWrap(True)
        panel_layout.addWidget(hint)
        panel_layout.addStretch(1)
        return panel

    def _rebuild(self, component: Component | None) -> None:
        from app import theme

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self._rows = []
        self._derived = []

        if component is None:
            empty = QWidget()
            empty_layout = QVBoxLayout(empty)
            empty_layout.setContentsMargins(22, 26, 22, 22)
            empty_layout.setSpacing(7)

            heading = QLabel("Nothing selected")
            heading_font = heading.font()
            heading_font.setBold(True)
            heading.setFont(heading_font)
            heading.setStyleSheet(f"color:{theme.TEXT_DIM};")
            empty_layout.addWidget(heading)

            hint = QLabel(
                "Pick a component in the tree to edit its parameters.\n\n"
                "Nothing there yet? Use Model › Add."
            )
            hint.setStyleSheet(f"color:{theme.TEXT_FAINT};")
            hint.setWordWrap(True)
            empty_layout.addWidget(hint)
            empty_layout.addStretch(1)

            layout.addWidget(empty)
            layout.addStretch(1)
            self.setWidget(container)
            return

        if getattr(component, "imported", False):
            # An imported part is shown, not edited. Its shape belongs to the
            # STEP file, and a slider that quietly made the model disagree with
            # the CAD it came from would be worse than no slider at all.
            layout.addWidget(self._imported_banner(component))
            layout.addStretch(1)
            self.setWidget(container)
            return

        # ---- header band -------------------------------------------------
        header = QWidget()
        header.setStyleSheet(
            f"background:{theme.BG_RAISED};"
            f"border-bottom:1px solid {theme.BORDER};"
        )
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(14, 10, 14, 10)
        header_layout.setSpacing(2)

        title = QLabel(component.name)
        font = title.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 2)
        title.setFont(font)
        title.setStyleSheet("background:transparent;")
        header_layout.addWidget(title)

        subtitle = QLabel(f"{component.kind}   ·   {component.path}")
        subtitle.setStyleSheet(
            f"color:{theme.TEXT_FAINT}; font-size:8pt; background:transparent;"
        )
        subtitle.setWordWrap(True)
        header_layout.addWidget(subtitle)
        layout.addWidget(header)

        # ---- body --------------------------------------------------------
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 12, 14, 12)
        body_layout.setSpacing(8)
        layout.addWidget(body)
        # The rest of this method builds into the body, below the header band.
        layout = body_layout

        # Dimensions first. RASAero's dialogs open on the numbers that define
        # the part, and classification comes after; the panel used to lead with
        # aero role and material and bury the geometry below them.
        # One form for every group, with the headings as spanning rows. Giving
        # each group its own layout left the value columns starting at a
        # different x per section, which reads as sloppy in a panel whose whole
        # job is a column of numbers.
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        # When the panel is too narrow for label and field side by side, put the
        # label on its own line rather than letting the field overflow.
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(5)

        # A body's defining numbers are its length and diameter, and neither is
        # a Parm -- they emerge from the section stations and widths. That is
        # right for the model and wrong for the panel: RASAero's Body Tube
        # dialog is Length and Diameter and nothing else, while this showed
        # "Shell thickness" and "Lateral offset" and hid the two numbers that
        # actually describe the part. They go first, and they are editable.
        for parm, getter in self._derived_parms(component):
            row = ParmRow(parm)
            row.changed.connect(self._on_derived)
            self._derived.append((row, parm, getter))
            if not form.rowCount():
                form.addRow(section_header("Dimensions"))
            form.addRow(parm_label(parm, row.unit_label), row)

        # Squaring a nose cone up against the tube behind it by hand means
        # editing section rows until two diameters agree and two stations meet.
        # The relationship is fixed, so it is a button.
        if isinstance(component, Stack) and component.sections:
            fit = QPushButton("Fit to body behind")
            fit.setToolTip(
                "Match this part's diameter to the front of the body behind it,\n"
                "and butt its aft end against where that body starts."
            )
            fit.clicked.connect(self._on_fit_to_body)
            form.addRow("", fit)

        for title, parms in component.parm_groups():
            if title:
                form.addRow(section_header(title))
            for parm in parms:
                row = ParmRow(parm)
                row.changed.connect(self._on_parm)
                self._rows.append(row)
                form.addRow(parm_label(parm, row.unit_label), row)
        layout.addLayout(form)

        # Clipping. Only the mould-line chain: a fin is placed along its host
        # body, so "the part ahead of it" is not a question with an answer.
        if component.clippable and self.model is not None:
            targets = self.model.clip_targets_for(component)
            layout.addWidget(section_header("Clipping"))

            clip_row = QHBoxLayout()
            clip_row.addWidget(QLabel("Clips to"))
            clip_combo = QComboBox()
            clip_combo.addItem("Nothing - free", "")
            for target in targets:
                clip_combo.addItem(target.name, target.path)
            clip_combo.setCurrentIndex(
                max(clip_combo.findData(component.clip_to or ""), 0)
            )
            clip_combo.currentIndexChanged.connect(
                lambda i, c=clip_combo: self._on_clip_to(c.itemData(i))
            )
            clip_row.addWidget(clip_combo, 1)
            layout.addLayout(clip_row)

            side_row = QHBoxLayout()
            side_row.addWidget(QLabel("Sitting"))
            side_combo = QComboBox()
            side_combo.addItem("Behind it (aft)", "aft")
            side_combo.addItem("In front of it (forward)", "forward")
            side_combo.setCurrentIndex(
                max(side_combo.findData(getattr(component, "clip_side", "aft")), 0)
            )
            side_combo.setEnabled(bool(component.clip_to))
            side_combo.currentIndexChanged.connect(
                lambda i, c=side_combo: self._on_clip_side(c.itemData(i))
            )
            side_row.addWidget(side_combo, 1)
            layout.addLayout(side_row)

            # Only worth showing when one of the two parts has curved ends --
            # between two flat-ended parts the setting changes nothing, and an
            # inert control is worse than no control.
            target = next(
                (c for c in self.model.walk() if c.path == component.clip_to), None
            )
            if component.has_distinct_mating or (
                target is not None and target.has_distinct_mating
            ):
                self._clip_flush = QCheckBox("Sit flush (nest the domes)")
                self._clip_flush.setToolTip(
                    "On: join at the tangent lines, so a dome sits inside its\n"
                    "neighbour the way a real intertank bolts to a tank.\n"
                    "Off: butt the extreme ends together, dome tip and all."
                )
                self._clip_flush.setChecked(bool(component.clip_flush))
                self._clip_flush.toggled.connect(self._on_clip_flush)
                layout.addWidget(self._clip_flush)

            self._clip_lock = QCheckBox("Keep clipped when things move")
            self._clip_lock.setToolTip(
                "Off: the clip snapped it into place once and let go.\n"
                "On: the station is derived - lengthen what it is clipped to\n"
                "and this part follows."
            )
            self._clip_lock.setChecked(bool(component.clip_locked))
            self._clip_lock.setEnabled(bool(component.clip_to))
            self._clip_lock.toggled.connect(self._on_clip_lock)
            layout.addWidget(self._clip_lock)

            if not targets and not component.clip_to:
                hint = QLabel("Nothing on the mould line to clip to yet.")
                hint.setStyleSheet(f"color:{theme.TEXT_FAINT};")
                hint.setWordWrap(True)
                layout.addWidget(hint)

        if component.allowed_roles or component.kind == "protuberance":
            layout.addWidget(section_header("Aerodynamics"))

        # Aerodynamic role. This is the Set mechanism: a component declared
        # internal keeps its mass and disappears from the aerodynamics.
        if component.allowed_roles:
            from parametric.roles import AeroRole

            role_row = QHBoxLayout()
            role_row.addWidget(QLabel("Aero role"))
            role_combo = QComboBox()
            for value in component.allowed_roles:
                role_combo.addItem(AeroRole(value).label, value)
            index = role_combo.findData(component.aero_role.value)
            role_combo.setCurrentIndex(max(index, 0))
            role_combo.currentIndexChanged.connect(
                lambda i, c=role_combo: self._on_role(c.itemData(i))
            )
            role_row.addWidget(role_combo, 1)
            layout.addLayout(role_row)

        # Where a motor's propellant comes from. A biprop is fed from tanks, so
        # its own propellant load stops counting -- otherwise the same
        # kilograms appear twice, once in the tank and once in the engine.
        if isinstance(component, Motor):
            feed_row = QHBoxLayout()
            feed_row.addWidget(QLabel("Propellant from"))
            feed_combo = QComboBox()
            feed_combo.addItem("Its own grain (solid)", "grain")
            feed_combo.addItem("The vehicle's tanks (bi-prop)", "tanks")
            feed_combo.setCurrentIndex(
                max(feed_combo.findData(getattr(component, "feed", "grain")), 0)
            )
            feed_combo.currentIndexChanged.connect(
                lambda i, c=feed_combo: self._on_feed(c.itemData(i))
            )
            feed_row.addWidget(feed_combo, 1)
            layout.addLayout(feed_row)

        # Which side of the engine's mixture a tank feeds. Only consulted when
        # a tank-fed motor declares a mixture ratio; unlabeled tanks drain in
        # proportion to their loads.
        if isinstance(component, Tank):
            contents_row = QHBoxLayout()
            contents_row.addWidget(QLabel("Contains"))
            contents_combo = QComboBox()
            contents_combo.addItem("Fuel", "fuel")
            contents_combo.addItem("Oxidizer", "oxidizer")
            contents_combo.setCurrentIndex(
                max(contents_combo.findData(
                    getattr(component, "contents", "fuel")
                ), 0)
            )
            contents_combo.currentIndexChanged.connect(
                lambda i, c=contents_combo: self._on_contents(c.itemData(i))
            )
            contents_row.addWidget(contents_combo, 1)
            layout.addLayout(contents_row)

        # Hoerner shape, for a protuberance.
        if component.kind == "protuberance":
            from parametric.roles import HOERNER_CD, HoernerShape

            shape_row = QHBoxLayout()
            shape_row.addWidget(QLabel("Shape"))
            shape_combo = QComboBox()
            for shape in HoernerShape:
                shape_combo.addItem(
                    f"{shape.label}  (Cd {HOERNER_CD[shape]:.2f})", shape.value
                )
            shape_combo.setCurrentIndex(
                max(shape_combo.findData(component.shape.value), 0)
            )
            shape_combo.currentIndexChanged.connect(
                lambda i, c=shape_combo: self._on_shape(c.itemData(i))
            )
            shape_row.addWidget(shape_combo, 1)
            layout.addLayout(shape_row)

        layout.addWidget(section_header("Material and mass"))

        # Material, where it means something. A motor's is declared rather than
        # derived from a material and a volume, so it has no material row --
        # but it does now get the measured-mass row below.
        if not isinstance(component, (Motor,)):
            from parametric.materials import MATERIALS

            material_row = QHBoxLayout()
            material_row.addWidget(QLabel("Material"))
            combo = QComboBox()
            combo.addItems(sorted(MATERIALS))
            combo.setCurrentText(component.material)
            combo.currentTextChanged.connect(self._on_material)
            material_row.addWidget(combo, 1)
            layout.addLayout(material_row)

        # Measured mass. The one fact geometry cannot carry: once hardware
        # exists and has been weighed, the scale is right and the model is an
        # estimate. Ticking this lets the scale win while the geometry stays
        # intact for aerodynamics and CAD.
        # QCheckBox and QDoubleSpinBox are imported at module level. Naming them
        # again here made them locals for the whole function, so the clip
        # checkbox above -- earlier in the same body -- could not see the
        # module-level name and raised UnboundLocalError.
        from app.units import UNITS

        override_row = QHBoxLayout()
        self._override_check = QCheckBox("Measured mass")
        self._override_check.setChecked(component.mass_override_kg is not None)
        override_row.addWidget(self._override_check)

        scale = UNITS.factor("kg")
        self._override_spin = QDoubleSpinBox()
        self._override_spin.setDecimals(4)
        self._override_spin.setRange(0.0, 1e6)
        self._override_spin.setSuffix(f" {UNITS.unit_label('kg')}")
        shown = (
            component.mass_override_kg
            if component.mass_override_kg is not None
            else component.computed_mass_kg()
        )
        self._override_spin.setValue(shown * scale)
        self._override_spin.setEnabled(component.mass_override_kg is not None)
        override_row.addWidget(self._override_spin, 1)
        layout.addLayout(override_row)

        self._override_note = QLabel()
        self._override_note.setWordWrap(True)
        self._override_note.setStyleSheet(
            f"color:{theme.WARN}; font-size:8pt;"
        )
        layout.addWidget(self._override_note)
        self._refresh_override_note()

        self._override_check.toggled.connect(self._on_override_toggled)
        self._override_spin.valueChanged.connect(self._on_override_value)

        self._sections = None
        self._motor = None
        if isinstance(component, Stack):
            from app.sectioneditor import SectionEditor

            self._sections = SectionEditor()
            self._sections.set_stack(component)
            self._sections.changed.connect(lambda: self.section_changed.emit(component))
            layout.addWidget(self._sections)
        elif isinstance(component, Motor):
            from app.motoreditor import MotorEditor

            self._motor = MotorEditor()
            self._motor.set_motor(component)
            self._motor.changed.connect(lambda: self.section_changed.emit(component))
            layout.addWidget(self._motor)

        self._derived_label = None
        derived = self._derived_readout(component)
        if derived is not None:
            layout.addWidget(derived)

        layout.addStretch(1)
        self.setWidget(container)

    def refresh_derived(self) -> None:
        """Recompute the derived block in place.

        Called on every parm change while dragging. Rebuilding the whole panel
        instead would throw away keyboard focus and scroll position mid-drag.
        """
        if self._component is not None and self._derived_label is not None:
            self._derived_label.setText("\n".join(self._derived_lines(self._component)))
        # Length, diameter and station are edits *to the sections*: they rewrite
        # every station or every width in the stack. The table below was still
        # showing the numbers from before the edit, which reads as the panel
        # disagreeing with itself. It repopulates in place rather than
        # rebuilding, so the selected row and the detail strip survive.
        if getattr(self, "_sections", None) is not None:
            self._sections.refresh()

    def _derived_lines(self, component: Component) -> list[str]:
        """Quantities computed from the parms, which is what you watch while editing."""
        lines: list[str] = []
        if isinstance(component, FinSet):
            lines = [
                f"area per fin     {component.area_per_fin_m2 * 1e4:8.1f} cm²",
                f"aspect ratio     {component.aspect_ratio:8.2f}",
                f"mid-chord sweep  {np.degrees(component.mid_chord_sweep_rad):8.1f}°",
                f"body radius      {component.body_radius_m() * 1000:8.1f} mm",
                f"total mass       {component.mass_kg():8.3f} kg",
            ]
        elif isinstance(component, Stack):
            low, high = component.station_range_m()
            lines = [
                f"length           {high - low:8.3f} m",
                f"max diameter     {component.max_diameter_m:8.3f} m",
                f"enclosed volume  {component.enclosed_volume_m3() * 1e3:8.2f} L",
                f"material volume  {component.volume_m3() * 1e6:8.1f} cm³",
                f"mass             {component.mass_kg():8.3f} kg",
            ]
        elif isinstance(component, Tank):
            fill = component.fill_fraction
            lines = [
                f"overall length   {component.overall_length_m:8.3f} m",
                f"dome height      {component.dome_height_m * 1000:8.1f} mm",
                f"internal volume  {component.internal_volume_m3 * 1e3:8.2f} L",
                f"capacity         {component.capacity_kg:8.1f} kg",
                f"loaded           {component.get('propellant_mass'):8.1f} kg"
                f"  ({fill:.0%} full)",
                f"shell mass       {component.mass_kg():8.3f} kg",
            ]
            if component.get("propellant_mass") > component.capacity_kg:
                lines.append("OVER CAPACITY - see Model > Validate")
        elif isinstance(component, Motor):
            lines = [
                f"propellant vol   {component.propellant_volume_m3 * 1e3:8.2f} L",
                f"centroid         {component.centroid_station_m:8.3f} m",
            ]
            if getattr(component, "feed", "grain") == "tanks":
                lines.append("fed from tanks - own load not counted")
        elif isinstance(component, PointMass):
            lines = [f"with growth      {component.mass_with_growth_kg:8.3f} kg"]
        elif component.kind == "protuberance":
            from parametric.roles import HOERNER_CD

            spec = component.to_spec()
            lines = [
                f"Hoerner Cd       {HOERNER_CD[component.shape]:8.2f}",
                f"drag area M0     {spec.drag_area_m2(0.0) * 1e4:8.2f} cm²",
                f"drag area M1.5   {spec.drag_area_m2(1.5) * 1e4:8.2f} cm²",
                f"total mass       {component.mass_kg():8.3f} kg",
            ]
        return lines

    def _derived_readout(self, component: Component) -> QWidget | None:
        lines = self._derived_lines(component)
        if not lines:
            return None

        from app import theme

        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(section_header("Derived"))
        body = QLabel("\n".join(lines))
        body.setStyleSheet(
            f"font-family:{theme.MONO_FONT},monospace; font-size:8pt;"
            f"color:{theme.TEXT_DIM};"
        )
        layout.addWidget(body)
        self._derived_label = body
        return box

    # ------------------------------------------------------------------

    #: Derived dimensions, by component kind: (parm name, label, getter, setter).
    DERIVED = {
        "stack": [
            ("length", "Length", lambda c: c.length_m or None,
             lambda c, v: c.set_length_m(v)),
            ("diameter", "Diameter", lambda c: c.max_diameter_m or None,
             lambda c, v: c.set_diameter_m(v)),
            # Along the spine. Stations are Z in this model -- +Z aft from the
            # nose tip -- so this is the Z offset, and zero is a real value.
            ("station", "Station",
             lambda c: (c.station_range_m()[0] if c.sections else None),
             lambda c, v: c.set_station_m(v)),
        ],
    }

    def _derived_parms(self, component):
        """Synthetic parms for values the component computes rather than stores.

        They are real ``Parm`` objects so ``ParmRow`` needs no special case, but
        nothing owns them: the value is read back from the component on every
        refresh, and writing one calls the component's setter rather than
        ``set``, which would fail because there is no parm of that name.
        """
        from parametric.parm import Parm

        out = []
        for name, label, getter, _setter in self.DERIVED.get(component.kind, []):
            try:
                raw = getter(component)
            except Exception:
                continue
            # None means the component cannot answer yet -- a stack with no
            # sections has no length to scale. Zero is a real value: a nose cone
            # legitimately sits at station zero.
            if raw is None:
                continue
            value = float(raw)
            # Uncapped. A body's length and diameter have no natural ceiling,
            # and one derived from the current value would have made the part
            # unable to grow past a few times whatever it happened to be. The
            # slider picks a workable window from these; the box takes anything.
            out.append((Parm(name, value, 0.0, 1e5, "m", label), getter))
        return out

    def _on_clip_to(self, path: str) -> None:
        """Attach to a part and snap there now, or let go."""
        component = self._component
        if component is None or self.model is None:
            return
        if not path:
            self.model.unclip(component)
        else:
            target = next(
                (c for c in self.model.walk() if c.path == path), None
            )
            if target is None or not self.model.clip(
                component, target,
                lock=bool(getattr(component, "clip_locked", False)),
                side=getattr(component, "clip_side", "aft"),
            ):
                self.fit_failed.emit("That part cannot be clipped to.")
                return
        self.parm_changed.emit(component, "clip", 0.0)
        self._rebuild_later(component)

    def _on_clip_side(self, side: str) -> None:
        """Swap which end of the target this part hangs off, and re-snap."""
        component = self._component
        if component is None or self.model is None or not side:
            return
        component.clip_side = side
        station = self.model.clip_station_for(component)
        if station is not None:
            component.set_forward_station_m(station)
        component.mark_dirty("clip_side")
        self.parm_changed.emit(component, "clip_side", 0.0)
        self._rebuild_later(component)

    def _on_clip_flush(self, flush: bool) -> None:
        """Switch between joining at the tangent lines and at the extremes."""
        component = self._component
        if component is None or self.model is None:
            return
        component.clip_flush = bool(flush)
        station = self.model.clip_station_for(component)
        if station is not None:
            component.set_forward_station_m(station)
        component.mark_dirty("clip_flush")
        self.parm_changed.emit(component, "clip_flush", 0.0)

    def _on_clip_lock(self, locked: bool) -> None:
        """Turn a one-off snap into a constraint, or back."""
        component = self._component
        if component is None or self.model is None:
            return
        component.clip_locked = bool(locked)
        if locked:
            station = self.model.clip_station_for(component)
            if station is not None:
                component.set_forward_station_m(station)
        component.mark_dirty("clip")
        self.parm_changed.emit(component, "clip_locked", 0.0)

    def _on_feed(self, value: str) -> None:
        if self._component is None or not value:
            return
        self._component.feed = value
        self._component.mark_dirty("feed")
        self.parm_changed.emit(self._component, "feed", 0.0)

    def _on_contents(self, value: str) -> None:
        if self._component is None or not value:
            return
        self._component.contents = value
        self._component.mark_dirty("contents")
        self.parm_changed.emit(self._component, "contents", 0.0)

    def _on_fit_to_body(self) -> None:
        component = self._component
        if not isinstance(component, Stack):
            return
        target = component.fit_to_body()
        if target is None:
            self.fit_failed.emit("Nothing to fit to - add a body first.")
            return
        # Diameter, station and the section table all moved, so rebuild rather
        # than refresh: this is not one value changing.
        self.parm_changed.emit(component, "fit", 0.0)
        self._rebuild_later(component)

    def _on_derived(self, name: str, value: float) -> None:
        """Write a derived dimension back through the component's setter."""
        component = self._component
        if component is None:
            return
        for parm_name, _label, _getter, setter in self.DERIVED.get(
            component.kind, []
        ):
            if parm_name == name and setter(component, value):
                self.parm_changed.emit(component, name, value)
                return

    def _on_parm(self, name: str, value: float) -> None:
        if self._component is None:
            return
        if self._component.set(name, value):
            self.parm_changed.emit(self._component, name, value)

    def _refresh_override_note(self) -> None:
        """Say how far the scale and the geometry disagree, if they do."""
        component = self._component
        if component is None or not hasattr(self, "_override_note"):
            return
        if component.mass_override_kg is None:
            computed = component.computed_mass_kg()
            from app.units import UNITS

            self._override_note.setText(
                f"computed from geometry: {UNITS.format(computed, 'kg', 4)}"
            )
            self._override_note.setStyleSheet("color:#667; font-size:11px;")
            return

        disagreement = component.override_disagreement
        from app.units import UNITS

        computed = component.computed_mass_kg()
        text = (
            f"geometry says {UNITS.format(computed, 'kg', 4)} "
            f"({disagreement * 100:.1f}% out)"
        )
        # A large gap means the geometry is wrong, and the geometry is still
        # what aerodynamics and CAD are taken from.
        colour = "#a05a00" if disagreement > 0.15 else "#667"
        self._override_note.setText(text)
        self._override_note.setStyleSheet(f"color:{colour}; font-size:11px;")

    def _on_override_toggled(self, checked: bool) -> None:
        if self._component is None:
            return
        self._override_spin.setEnabled(checked)
        if checked:
            from app.units import UNITS

            self._component.mass_override_kg = UNITS.to_si(
                self._override_spin.value(), "kg"
            )
        else:
            self._component.mass_override_kg = None
        self._refresh_override_note()
        self.parm_changed.emit(self._component.path, "mass_override", 0.0)

    def _on_override_value(self, value: float) -> None:
        if self._component is None or self._component.mass_override_kg is None:
            return
        from app.units import UNITS

        self._component.mass_override_kg = UNITS.to_si(value, "kg")
        self._refresh_override_note()
        self.parm_changed.emit(self._component.path, "mass_override", 0.0)

    def _on_material(self, material: str) -> None:
        if self._component is None:
            return
        self._component.material = material
        self._component.mark_dirty("material")
        self.parm_changed.emit(self._component, "material", 0.0)

    def _on_role(self, value: str) -> None:
        if self._component is None or not value:
            return
        from parametric.roles import AeroRole

        self._component.aero_role = AeroRole(value)
        self._component.mark_dirty("aero_role")
        self.parm_changed.emit(self._component, "aero_role", 0.0)

    def _on_shape(self, value: str) -> None:
        if self._component is None or not value:
            return
        from parametric.roles import HoernerShape

        self._component.shape = HoernerShape(value)
        self._component.mark_dirty("shape")
        self.parm_changed.emit(self._component, "shape", 0.0)
