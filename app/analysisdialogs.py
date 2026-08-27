"""Setup dialogs for the analyses.

OpenVSP's arrangement, and the reason it is worth copying: an analysis is not a
button that runs on assumptions you cannot see. It is a form of inputs you set,
a run, and a result. Every number the run depends on is on screen and editable
before it starts.

Before this, Run Flight launched at 85 degrees off a 3 m rail with a recovery
system nobody chose, and there was no aerodynamic analysis at all.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

# The settings live with the code that flies them, so the sweep and the
# dispersion study fly exactly what the dialog edits. Re-exported here
# because callers import them from the dialog module.
from parametric.flight import FlightSettings  # noqa: F401


def _spin(minimum, maximum, value, step, suffix="", decimals=3) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setDecimals(decimals)
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(f" {suffix}")
    box.setMinimumWidth(130)
    return box


class SetupDialog(QDialog):
    """Base: a form, an OK/Cancel bar, and a note about what the run assumes."""

    def __init__(self, title: str, note: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(430)

        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(10)

        if note:
            label = QLabel(note)
            label.setWordWrap(True)
            label.setStyleSheet("color:#5b6675; font-size:11px;")
            self._layout.addWidget(label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)

    def add_group(self, title: str) -> QFormLayout:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setSpacing(6)
        self._layout.addWidget(group)
        return form

    def finish(self) -> None:
        self._layout.addStretch(1)
        self._layout.addWidget(self.buttons)


# ----------------------------------------------------------------------
# Aerodynamics
# ----------------------------------------------------------------------


class AeroSetupDialog(SetupDialog):
    """Mach and alpha sweep, and the conditions the coefficients are built at."""

    #: Value, label, and what the method is actually good for.
    #:
    #: Both entries are RASAero's method. The first runs it here; the second
    #: runs the application. They are not an approximation and a reference --
    #: the first is validated against the second across 130 vehicles and 7.1
    #: million coefficient comparisons, with nothing outside RASAero's own
    #: printed precision.
    METHODS = [
        ("rasaero", "RASAero (built in)",
         "RASAero II's method, running here. Instant, needs nothing "
         "installed, and sweeps whatever Mach and alpha grid you ask for. "
         "Validated against the application below term by term."),
        ("rasaero-app", "RASAero II (application)",
         "Runs the real RASAero II and imports its export. The reference the "
         "built-in engine is measured against — worth running as an "
         "independent check. Needs it installed, only tabulates to 4° alpha, "
         "and takes over the desktop for about half a minute."),
    ]

    def __init__(self, settings=None, parent=None,
                 nozzle_exit_diameter_m: float | None = None,
                 boattail_half_angle_deg: float | None = None):
        super().__init__(
            "Aerodynamic Analysis",
            "Builds a coefficient table for this vehicle. Both methods fill "
            "the same table, so a flight run does not care which produced it.",
            parent,
        )
        from parametric import rasaero_run
        from parametric.aero import AeroSettings

        settings = settings or AeroSettings()
        self._rasaero_available = rasaero_run.available()

        method_group = self.add_group("Method")
        self.method = QComboBox()
        for value, label, _ in self.METHODS:
            usable = value != "rasaero-app" or self._rasaero_available
            self.method.addItem(
                label if usable else f"{label}  (not installed)", value
            )
            if not usable:
                self.method.model().item(self.method.count() - 1).setEnabled(False)

        # Default to the built-in engine whether or not RASAero is installed:
        # it gives the same answer without taking the machine away for half a
        # minute, and it is the only one that can sweep past 4 degrees alpha.
        preferred = getattr(settings, "method", None) or "rasaero"
        if preferred in ("rasaero-native", "builtin"):
            preferred = "rasaero"          # names used before the engine landed
        if preferred == "rasaero-app" and not self._rasaero_available:
            preferred = "rasaero"
        self.method.setCurrentIndex(max(self.method.findData(preferred), 0))

        self._method_note = QLabel()
        self._method_note.setWordWrap(True)
        self._method_note.setStyleSheet("color:#5b6675; font-size:11px;")
        method_group.addRow("Generated by", self.method)
        method_group.addRow("", self._method_note)

        sweep = self.add_group("Sweep")
        self.mach_min = _spin(0.0, 10.0, settings.mach_min, 0.05, "", 2)
        self.mach_max = _spin(0.1, 20.0, settings.mach_max, 0.5, "", 2)
        self.mach_points = QSpinBox()
        self.mach_points.setRange(4, 400)
        self.mach_points.setValue(settings.mach_points)
        self.alpha_max = _spin(1.0, 45.0, settings.alpha_max_deg, 1.0, "deg", 1)
        self.alpha_points = QSpinBox()
        self.alpha_points.setRange(2, 60)
        self.alpha_points.setValue(settings.alpha_points)

        sweep.addRow("Mach from", self.mach_min)
        sweep.addRow("Mach to", self.mach_max)
        sweep.addRow("Mach points", self.mach_points)
        sweep.addRow("Alpha max", self.alpha_max)
        sweep.addRow("Alpha points", self.alpha_points)

        conditions = self.add_group("Conditions")
        self.altitude = _spin(0.0, 80000.0, settings.altitude_m, 500.0, "m", 0)
        self.roughness = _spin(1.0, 500.0, settings.roughness_m * 1e6, 5.0, "µm", 0)
        self.power_on = QCheckBox("Plume fills the base while the motor burns")
        self.power_on.setToolTip(
            "Tabulates a second drag column with the base filled by the plume. "
            "The flight uses it during the burn and switches to the power-off "
            "column at burnout. Off flies power-off drag throughout."
        )
        self.power_on.setChecked(settings.power_on_base)
        if nozzle_exit_diameter_m is not None and nozzle_exit_diameter_m <= 0.0:
            # The plume's share of the base is the nozzle exit area over the
            # base area. With no exit area there is nothing to fill it with
            # and the two drag columns come out identical -- so say so,
            # rather than offer a checkbox that changes nothing.
            self.power_on.setChecked(False)
            self.power_on.setEnabled(False)
            self.power_on.setText(
                "Plume fills the base (the motor declares no nozzle exit area)"
            )

        # The supersonic base-drag law, and what is known to be wrong with
        # it on this vehicle. The switch used to exist only in code.
        self.boattail_law = QComboBox()
        self.boattail_law.addItem("RASAero's law (faithful port)", "rasaero")
        self.boattail_law.addItem("Corrected law (provisional, unvalidated)", "corrected")
        self.boattail_law.setCurrentIndex(
            max(self.boattail_law.findData(getattr(settings, "boattail_model", "rasaero")), 0)
        )
        self._boattail_half_angle = boattail_half_angle_deg
        self._boattail_note = QLabel()
        self._boattail_note.setWordWrap(True)
        self._boattail_note.setStyleSheet("color:#5b6675; font-size:11px;")

        conditions.addRow("Reynolds altitude", self.altitude)
        conditions.addRow("Surface roughness", self.roughness)
        conditions.addRow("", self.power_on)
        conditions.addRow("Supersonic boattail", self.boattail_law)
        conditions.addRow("", self._boattail_note)
        self.boattail_law.currentIndexChanged.connect(lambda _: self._on_boattail())
        self.mach_max.valueChanged.connect(lambda _: self._on_boattail())
        self._on_boattail()

        self._sweep_group = sweep.parentWidget()
        self.method.currentIndexChanged.connect(self._on_method)
        # The finish RASAero will be given depends on the roughness typed in,
        # so the note has to follow it rather than being written once.
        self.roughness.valueChanged.connect(lambda _: self._on_method())
        self._on_method()
        self.finish()

    def _on_boattail(self) -> None:
        """Say what is known to be wrong about this vehicle's supersonic drag."""
        from aeroengine.basedrag import SEPARATION_ANGLE_DEG
        from parametric.aero import boattail_caveat

        angle = self._boattail_half_angle
        if angle is None or angle <= 0.0:
            self._boattail_note.setText(
                "No boattail on this vehicle; the law only matters past one."
            )
            self._boattail_note.setStyleSheet("color:#5b6675; font-size:11px;")
            return
        caveat = boattail_caveat(angle, self.mach_max.value(), self.boattail_law.currentData())
        if caveat:
            self._boattail_note.setText(caveat)
            self._boattail_note.setStyleSheet("color:#9a4a12; font-size:11px;")
        else:
            self._boattail_note.setText(
                f"Boattail half-angle {angle:.1f}° is within RASAero's "
                f"{SEPARATION_ANGLE_DEG:.1f}° separation clamp, or the sweep stays subsonic; "
                f"the base-drag law applies as validated."
            )
            self._boattail_note.setStyleSheet("color:#5b6675; font-size:11px;")

    def _on_method(self) -> None:
        """Show what this method will do, and hide what it will ignore.

        RASAero sweeps a fixed Mach and alpha grid of its own, so leaving the
        sweep fields live would let someone set a range that quietly has no
        effect. The conditions below them do carry across -- altitude sets the
        atmosphere, roughness picks the nearest finish RASAero offers, and the
        power-on flag chooses which of its two drag columns is imported.
        """
        method = self.method.currentData()
        note = next(text for value, _, text in self.METHODS if value == method)
        rasaero = method == "rasaero-app"

        if rasaero:
            from parametric.canonical import surface_finish

            finish = surface_finish(self.roughness.value() * 1e-6)
            note += (
                f"\n\nRoughness maps to RASAero's “{finish}”, which moves its "
                f"drag a long way. Its table is always built at sea level, so "
                f"the Reynolds altitude does not apply."
            )
        self._method_note.setText(note)
        self._sweep_group.setEnabled(not rasaero)
        # Greyed rather than hidden: the field still applies to the other
        # method, and a control that vanishes is harder to find again than one
        # that is visibly inactive.
        self.altitude.setEnabled(not rasaero)

    def settings(self):
        from parametric.aero import AeroSettings

        settings = AeroSettings(
            mach_min=self.mach_min.value(),
            mach_max=max(self.mach_max.value(), self.mach_min.value() + 0.05),
            mach_points=self.mach_points.value(),
            alpha_max_deg=self.alpha_max.value(),
            alpha_points=self.alpha_points.value(),
            altitude_m=self.altitude.value(),
            roughness_m=self.roughness.value() * 1e-6,
            power_on_base=self.power_on.isChecked(),
            boattail_model=self.boattail_law.currentData(),
        )
        settings.method = self.method.currentData()
        return settings


# ----------------------------------------------------------------------
# Flight
# ----------------------------------------------------------------------


# ``FlightSettings`` itself is defined in ``parametric.flight`` and imported
# at the top of this module.


class FlightSetupDialog(SetupDialog):
    """Launch conditions, wind, recovery and integration."""

    #: Label and (rtol, atol). ``None`` is the integrator's own default.
    ACCURACIES = [
        ("Fast preview (1e-3)", 1e-3, 1e-6),
        ("Standard (1e-6)", None, None),
        ("Tight (1e-8)", 1e-8, 1e-10),
    ]

    def __init__(self, settings: FlightSettings | None = None,
                 has_aero: bool = False, parent=None):
        super().__init__(
            "Run Flight",
            "Every value the trajectory depends on. Without an aerodynamic "
            "table the simulator falls back to a crude drag law with no normal "
            "force, so the vehicle cannot weathercock.",
            parent,
        )
        settings = settings or FlightSettings()

        launch = self.add_group("Launch")
        self.elevation = _spin(1.0, 90.0, settings.elevation_deg, 1.0, "deg", 1)
        self.azimuth = _spin(-360.0, 360.0, settings.azimuth_deg, 5.0, "deg", 1)
        self.rail = _spin(0.0, 100.0, settings.rail_length_m, 0.5, "m", 2)
        self.pad_altitude = _spin(0.0, 5000.0, settings.pad_altitude_m, 50.0, "m", 0)
        self.use_latitude = QCheckBox("Earth's rotation (Coriolis) at latitude")
        self.use_latitude.setChecked(settings.latitude_deg is not None)
        self.latitude = _spin(
            -90.0, 90.0,
            settings.latitude_deg if settings.latitude_deg is not None else 0.0,
            1.0, "deg", 2,
        )
        self.latitude.setEnabled(self.use_latitude.isChecked())
        self.use_latitude.toggled.connect(self.latitude.setEnabled)
        self.use_latitude.setToolTip(
            "Adds the Coriolis term of a frame fixed to the rotating Earth. "
            "A rising vehicle is deflected west and a falling one east; a long "
            "arc turns right in the northern hemisphere. Tens of metres on a "
            "high flight -- a bias, not a spread."
        )
        self.use_buttons = QCheckBox("Two rail buttons, with tip-off")
        self.use_buttons.setChecked(settings.rail_buttons)
        self.use_buttons.setToolTip(
            "The vehicle slides on both buttons, then pivots on the aft one "
            "from the moment the forward one leaves until the aft one does. "
            "Gravity, the wind and its own acceleration turn it in that "
            "interval; the rate it leaves with is the tip-off rate. Off, the "
            "CG is constrained for the rail's length."
        )
        named = settings.rail_buttons_m
        self.button_forward = _spin(0.0, 100.0, named[0] if named else 0.0, 0.05, "m from nose", 2)
        self.button_aft = _spin(0.0, 100.0, named[1] if named else 0.0, 0.05, "m from nose", 2)
        self.button_forward.setSpecialValueText("loaded CG (auto)")
        self.button_aft.setSpecialValueText("near the tail (auto)")
        for widget in (self.button_forward, self.button_aft):
            widget.setEnabled(self.use_buttons.isChecked())
            self.use_buttons.toggled.connect(widget.setEnabled)
        launch.addRow("Elevation", self.elevation)
        launch.addRow("Azimuth from North", self.azimuth)
        launch.addRow("Rail length", self.rail)
        launch.addRow("", self.use_buttons)
        launch.addRow("Forward button", self.button_forward)
        launch.addRow("Aft button", self.button_aft)
        launch.addRow("Pad altitude", self.pad_altitude)
        launch.addRow("", self.use_latitude)
        launch.addRow("Latitude", self.latitude)

        wind = self.add_group("Wind")
        self.wind_speed = _spin(0.0, 60.0, settings.wind_speed_mps, 1.0, "m/s", 1)
        self.wind_direction = _spin(
            -360.0, 360.0, settings.wind_direction_deg, 10.0, "deg", 0
        )
        self.wind_aloft = QPlainTextEdit()
        self.wind_aloft.setPlaceholderText("altitude m   speed m/s   from deg\none level per line")
        self.wind_aloft.setPlainText("\n".join(
            f"{h:.0f} {v:.1f} {b:.0f}" for h, v, b in settings.wind_aloft
        ))
        self.wind_aloft.setFixedHeight(72)
        self.wind_aloft.setToolTip(
            "A sounding: winds above the surface layer, interpolated as "
            "vectors between levels and held above the top one. Below 10 m "
            "the surface profile rules."
        )
        self.turbulence = QComboBox()
        for level in ("none", "light", "moderate", "severe"):
            self.turbulence.addItem(level, level)
        self.turbulence.setCurrentIndex(max(self.turbulence.findData(settings.turbulence), 0))
        self.turbulence.setToolTip(
            "Dryden continuous turbulence (MIL-F-8785C), generated once "
            "along altitude from the seed. Light is 1.5 m/s RMS aloft, "
            "moderate 3, severe 6; near the ground it follows the surface wind."
        )
        self.turbulence_seed = QSpinBox()
        self.turbulence_seed.setRange(0, 10_000_000)
        self.turbulence_seed.setValue(settings.turbulence_seed)
        wind.addRow("Surface speed", self.wind_speed)
        wind.addRow("From direction", self.wind_direction)
        wind.addRow("Winds aloft", self.wind_aloft)
        wind.addRow("Turbulence", self.turbulence)
        wind.addRow("Turbulence seed", self.turbulence_seed)

        recovery = self.add_group("Recovery")
        self.use_recovery = QCheckBox("Deploy recovery")
        self.use_recovery.setChecked(settings.use_recovery)
        self.drogue = _spin(1.0, 100.0, settings.drogue_descent_mps, 1.0, "m/s", 1)
        self.main = _spin(1.0, 60.0, settings.main_descent_mps, 0.5, "m/s", 1)
        self.deploy_altitude = _spin(
            10.0, 5000.0, settings.main_deploy_altitude_m, 25.0, "m", 0
        )
        self.at_attachment = QCheckBox("Canopy pulls at the harness attachment")
        self.at_attachment.setChecked(settings.chute_at_attachment)
        self.at_attachment.setToolTip(
            "The vehicle hangs from the harness point and swings about it. "
            "Off, the canopy pulls through the CG with no moment, and the "
            "descent attitude means nothing."
        )
        self.attachment = _spin(
            0.0, 100.0,
            settings.chute_attachment_station_m or 0.0, 0.05, "m from nose", 2,
        )
        self.attachment.setSpecialValueText("nose shoulder (auto)")
        self.attachment.setEnabled(self.at_attachment.isChecked())
        self.at_attachment.toggled.connect(self.attachment.setEnabled)
        recovery.addRow("", self.use_recovery)
        recovery.addRow("Drogue descent", self.drogue)
        recovery.addRow("Main descent", self.main)
        recovery.addRow("Main deploys at", self.deploy_altitude)
        recovery.addRow("", self.at_attachment)
        recovery.addRow("Harness attaches at", self.attachment)

        build = self.add_group("Build imperfections")
        self.thrust_tilt = _spin(0.0, 10.0, settings.thrust_misalignment_deg, 0.05, "deg", 2)
        self.thrust_clock = _spin(0.0, 360.0, settings.thrust_misalignment_clock_deg, 15.0, "deg", 0)
        self.cg_offset = _spin(0.0, 200.0, settings.cg_offset_m * 1000.0, 0.5, "mm", 1)
        self.cg_clock = _spin(0.0, 360.0, settings.cg_offset_clock_deg, 15.0, "deg", 0)
        self.cant_offset = _spin(-10.0, 10.0, settings.fin_cant_offset_deg, 0.05, "deg", 2)
        for widget in (self.thrust_tilt, self.thrust_clock, self.cg_offset, self.cg_clock):
            widget.setToolTip(
                "A magnitude and a clock angle about the body axis, from body X "
                "toward body Z. These are moments: without a coefficient table "
                "there is no restoring moment to trim against and they tumble "
                "the vehicle."
            )
        self.cant_offset.setToolTip(
            "Cant the fins were built with that the table was not. Applied "
            "through the table's roll forcing per degree of cant."
        )
        build.addRow("Thrust misalignment", self.thrust_tilt)
        build.addRow("  toward clock angle", self.thrust_clock)
        build.addRow("CG off centreline", self.cg_offset)
        build.addRow("  toward clock angle", self.cg_clock)
        build.addRow("Fin cant offset", self.cant_offset)

        integration = self.add_group("Integration")
        self.dt = _spin(0.005, 1.0, settings.dt_s, 0.01, "s", 3)
        self.accuracy = QComboBox()
        for label, rtol, atol in self.ACCURACIES:
            self.accuracy.addItem(label, (rtol, atol))
        current = (settings.rtol, settings.atol)
        index = next(
            (i for i, (_, r, a) in enumerate(self.ACCURACIES) if (r, a) == current),
            1,
        )
        self.accuracy.setCurrentIndex(index)
        self.accuracy.setToolTip(
            "The solver's error tolerance per step. The default resolves the "
            "attitude to a hundredth of a degree; the fast setting is SciPy's "
            "own default, good to a tenth of a percent of apogee and about "
            "twice as quick."
        )
        self.use_aero = QCheckBox("Use the aerodynamic table")
        self.use_aero.setChecked(settings.use_aero_table and has_aero)
        self.use_aero.setEnabled(has_aero)
        if not has_aero:
            self.use_aero.setText("Use the aerodynamic table (run Aerodynamics first)")
        self.couple_aero = QCheckBox("Rebuild the table along the trajectory")
        self.couple_aero.setChecked(settings.couple_aero_altitude and has_aero)
        self.couple_aero.setEnabled(has_aero)
        self.couple_aero.setToolTip(
            "Fly, rebuild the drag table at the altitudes actually flown, and "
            "fly again until the apogee settles. Without this the table is "
            "evaluated at sea level, which overstates skin friction high up."
        )
        integration.addRow("Output interval", self.dt)
        integration.addRow("Solver accuracy", self.accuracy)
        integration.addRow("", self.use_aero)
        integration.addRow("", self.couple_aero)

        self.finish()

    def accept(self) -> None:
        from parametric.flight import parse_wind_aloft

        try:
            parse_wind_aloft(self.wind_aloft.toPlainText())
        except ValueError as exc:
            QMessageBox.warning(self, "Winds aloft", f"Could not read the winds aloft:\n{exc}")
            return
        super().accept()

    def settings(self) -> FlightSettings:
        from parametric.flight import parse_wind_aloft

        return FlightSettings(
            elevation_deg=self.elevation.value(),
            azimuth_deg=self.azimuth.value(),
            rail_length_m=self.rail.value(),
            rail_buttons=self.use_buttons.isChecked(),
            rail_buttons_m=(
                (self.button_forward.value(), self.button_aft.value())
                if self.button_forward.value() > 0.0 and self.button_aft.value() > 0.0
                and self.button_aft.value() > self.button_forward.value()
                else None
            ),
            pad_altitude_m=self.pad_altitude.value(),
            latitude_deg=self.latitude.value() if self.use_latitude.isChecked() else None,
            wind_speed_mps=self.wind_speed.value(),
            wind_direction_deg=self.wind_direction.value(),
            wind_aloft=parse_wind_aloft(self.wind_aloft.toPlainText()),
            turbulence=self.turbulence.currentData(),
            turbulence_seed=self.turbulence_seed.value(),
            use_recovery=self.use_recovery.isChecked(),
            drogue_descent_mps=self.drogue.value(),
            main_descent_mps=self.main.value(),
            main_deploy_altitude_m=self.deploy_altitude.value(),
            chute_at_attachment=self.at_attachment.isChecked(),
            chute_attachment_station_m=(
                self.attachment.value() if self.attachment.value() > 0.0 else None
            ),
            dt_s=self.dt.value(),
            rtol=self.accuracy.currentData()[0],
            atol=self.accuracy.currentData()[1],
            use_aero_table=self.use_aero.isChecked(),
            couple_aero_altitude=self.couple_aero.isChecked(),
            thrust_misalignment_deg=self.thrust_tilt.value(),
            thrust_misalignment_clock_deg=self.thrust_clock.value(),
            cg_offset_m=self.cg_offset.value() / 1000.0,
            cg_offset_clock_deg=self.cg_clock.value(),
            fin_cant_offset_deg=self.cant_offset.value(),
        )


# ----------------------------------------------------------------------
# Dispersion
# ----------------------------------------------------------------------


class DispersionSetupDialog(SetupDialog):
    """How many cases, and how much each input is allowed to vary.

    The spreads are centred on the flight that was set up -- its elevation,
    azimuth and wind -- and on the vehicle's own dry mass. They used to be
    centred on 85 degrees, no wind and 4.4 kg regardless, and the study then
    flew the simulator's placeholder vehicle rather than the one on screen.
    """

    def __init__(self, settings: FlightSettings, dry_mass_kg: float,
                 has_aero: bool = False, parent=None):
        super().__init__(
            "Dispersion Study",
            "Flies this vehicle many times with perturbed inputs and reports "
            "where it lands. Sampling is seeded, so a result can be reproduced.",
            parent,
        )
        self._settings = settings
        self._dry_mass_kg = float(dry_mass_kg)

        centre = self.add_group("Centred on")
        aero = (
            "coefficient table" if has_aero and settings.use_aero_table
            else "fallback drag law"
        )
        for label, text in (
            ("Vehicle", f"{dry_mass_kg:.2f} kg dry, {aero}"),
            ("Launch", f"{settings.elevation_deg:.1f}° elevation, "
                       f"{settings.azimuth_deg:.0f}° azimuth, "
                       f"pad at {settings.pad_altitude_m:.0f} m"),
            ("Wind", settings.describe_wind()),
        ):
            note = QLabel(text)
            note.setStyleSheet("color:#5b6675;")
            centre.addRow(label, note)

        batch = self.add_group("Batch")
        self.cases = QSpinBox()
        self.cases.setRange(2, 5000)
        self.cases.setValue(50)
        self.seed = QSpinBox()
        self.seed.setRange(0, 10_000_000)
        self.seed.setValue(12345)
        self.processes = QSpinBox()
        self.processes.setRange(1, 32)
        self.processes.setValue(4)
        batch.addRow("Cases", self.cases)
        batch.addRow("Seed", self.seed)
        batch.addRow("Worker processes", self.processes)

        spread = self.add_group("1-sigma spread")
        self.impulse = _spin(0.0, 0.5, 0.02, 0.01, "", 3)
        # Three percent of the vehicle rather than a fixed 1.5 kg: a build
        # tolerance scales with what is being built.
        self.mass = _spin(
            0.0, 10000.0, round(max(0.03 * self._dry_mass_kg, 0.001), 3),
            0.1, "kg", 3,
        )
        self.aero = _spin(0.0, 0.5, 0.08, 0.01, "", 3)
        self.elevation = _spin(0.0, 10.0, 0.5, 0.1, "deg", 2)
        self.azimuth = _spin(0.0, 45.0, 2.0, 0.5, "deg", 1)
        self.wind = _spin(0.0, 30.0, 2.5, 0.5, "m/s", 2)
        self.wind_direction = _spin(0.0, 180.0, 60.0, 5.0, "deg", 0)
        spread.addRow("Total impulse", self.impulse)
        spread.addRow("Dry mass", self.mass)
        spread.addRow("Aero coefficients", self.aero)
        spread.addRow("Launch elevation", self.elevation)
        spread.addRow("Launch azimuth", self.azimuth)
        spread.addRow("Wind speed", self.wind)
        spread.addRow("Wind direction", self.wind_direction)

        # Build imperfections are moments, and they need the table's
        # restoring moment to trim against; without a table they only
        # tumble the vehicle, so they stay at zero and say why.
        trims = has_aero and settings.use_aero_table
        self.thrust_tilt = _spin(0.0, 5.0, 0.10 if trims else 0.0, 0.05, "deg", 2)
        self.cg_offset = _spin(0.0, 50.0, 1.0 if trims else 0.0, 0.5, "mm", 1)
        self.fin_cant = _spin(0.0, 5.0, 0.10 if trims else 0.0, 0.05, "deg", 2)
        for widget in (self.thrust_tilt, self.cg_offset, self.fin_cant):
            widget.setEnabled(trims)
            if not trims:
                widget.setToolTip(
                    "Needs the coefficient table: these are moments, and the "
                    "fallback drag law has no restoring moment to trim them."
                )
        spread.addRow("Thrust misalignment", self.thrust_tilt)
        spread.addRow("CG off centreline", self.cg_offset)
        spread.addRow("Fin cant", self.fin_cant)

        self.finish()

    def dispersions(self) -> dict:
        from parametric.dispersion import dispersions_about

        return dispersions_about(
            self._settings, self._dry_mass_kg,
            impulse_sd=self.impulse.value(),
            dry_mass_sd_kg=self.mass.value(),
            aero_sd=self.aero.value(),
            elevation_sd_deg=self.elevation.value(),
            azimuth_sd_deg=self.azimuth.value(),
            wind_speed_sd_mps=self.wind.value(),
            wind_direction_sd_deg=self.wind_direction.value(),
            thrust_misalignment_sd_deg=self.thrust_tilt.value(),
            cg_offset_sd_m=self.cg_offset.value() / 1000.0,
            fin_cant_sd_deg=self.fin_cant.value(),
        )
