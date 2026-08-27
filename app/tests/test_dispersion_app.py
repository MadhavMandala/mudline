"""The dispersion study flies the vehicle on screen.

It used to call the library's dispersion with no case function, which flies
the simulator's built-in placeholder -- 50 kg dry, a 20 kN motor, 150 km of
apogee -- and then filed the landing ellipse against the open model.

Runs under pytest, and standalone via
``python -m pytest app/tests/test_dispersion_app.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("cadquery", reason="the app needs the cad extra")
pytest.importorskip("PySide6", reason="the app needs PySide6")

from parametric.standard import basic_rocket  # noqa: E402


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    from app.viewport import configure_surface_format

    configure_surface_format()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


@pytest.fixture
def window(qt_app):
    from app.mainwindow import MainWindow

    win = MainWindow(basic_rocket())
    win.show()
    qt_app.processEvents()
    yield win
    win.close()


@pytest.mark.slow
def test_run_flight_reports_the_rail_and_the_log(window, monkeypatch):
    """The flight action end to end, headless: log, rail exit, report."""
    from PySide6.QtWidgets import QDialog

    from app.analysisdialogs import FlightSettings

    window._flight_settings = FlightSettings(use_aero_table=False, dt_s=0.2)
    monkeypatch.setattr(window, "_exec_dialog", lambda dialog: QDialog.Accepted)
    window._run_flight()

    recorded = window.results.latest("flight")
    assert recorded is not None
    labels = {m.label for m in recorded.metrics}
    assert {"Max acceleration", "Rail exit", "Rail exit alpha"} <= labels
    assert "Angle of attack vs time" in recorded.series
    assert recorded.payload["log"] is not None
    assert window._last_flight.rail_exit["exact"]


@pytest.mark.slow
def test_the_dispersion_flies_the_open_vehicle(window, monkeypatch):
    from PySide6.QtWidgets import QDialog

    from app.analysisdialogs import FlightSettings

    window._flight_settings = FlightSettings(use_aero_table=False, dt_s=0.2)

    def accept(dialog):
        dialog.cases.setValue(3)
        dialog.processes.setValue(1)
        return QDialog.Accepted

    monkeypatch.setattr(window, "_exec_dialog", accept)
    window._run_dispersion()

    result = window._dispersion
    assert result.n_cases == 3
    apogee = result.summary["max_altitude"]["mean"]
    assert 1000.0 < apogee < 10000.0, "the placeholder vehicle flies to ~150 km"
    recorded = window.results.latest("dispersion")
    assert recorded is not None and recorded.is_current(window._fingerprint())
