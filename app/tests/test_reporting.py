"""How results reach the user.

Every other GUI test runs under the offscreen platform, where the window
deliberately routes reports to the status bar because a modal with nobody to
answer it blocks forever. That is the right behaviour and it is also a blind
spot: the branch a real user takes was never executed by anything, and it
contained a call that recursed until the stack gave out. These tests force the
other branch.
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
    yield QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def window(qt_app):
    from app.mainwindow import MainWindow

    win = MainWindow(basic_rocket())
    yield win
    win.close()


@pytest.fixture
def shown(window, monkeypatch):
    """Make the window believe a user is present, and capture the dialog.

    ``exec`` is stubbed rather than the whole QMessageBox, so the box is
    really constructed -- a report that cannot be built is as broken as one
    that recurses.
    """
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(type(window), "_can_prompt", staticmethod(lambda: True))
    captured: list[QMessageBox] = []
    monkeypatch.setattr(
        QMessageBox, "exec", lambda self: captured.append(self) or 0
    )
    return captured


@pytest.mark.slow
def test_an_informational_report_opens_a_dialog(window, shown):
    window._tell("Mass Properties", "dry 5.791 kg\nCG 1.134 m")
    assert len(shown) == 1
    assert shown[0].text().startswith("dry 5.791 kg")
    assert shown[0].windowTitle() == "Mass Properties"


@pytest.mark.slow
def test_a_failure_report_opens_a_dialog(window, shown):
    from PySide6.QtWidgets import QMessageBox

    window._complain("Aerodynamics", "Failed:\nVehicle has no measurable body.")
    assert len(shown) == 1
    assert shown[0].icon() == QMessageBox.Critical


@pytest.mark.slow
def test_reports_are_monospaced(window, shown):
    """A coefficient table in a proportional font is not a table."""
    window._tell("Aerodynamic Analysis", "  Mach     CD\n  0.10  0.402")
    assert "mono" in shown[0].styleSheet().lower()


@pytest.mark.slow
def test_a_long_report_is_not_truncated(window, shown):
    report = "\n".join(f"  {i:6.2f}{i * 0.01:9.4f}" for i in range(200))
    window._tell("Sweep", report)
    assert shown[0].text().count("\n") == report.count("\n")


@pytest.mark.slow
def test_headless_still_falls_back_to_the_status_bar(window):
    """The offscreen path must keep working -- it is what the suite relies on."""
    window._tell("Mass Properties", "dry 5.791 kg\nCG 1.134 m")
    assert "dry 5.791 kg" in window.statusBar().currentMessage()
