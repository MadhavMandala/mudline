"""Tests for the things that make a build identifiable and a failure reportable.

None of this is physics. It is the machinery that answers "which version made
this number" and "what actually went wrong", both of which turned out to be
unanswerable the first time the tool was handed to somebody else.

Runs under pytest, and standalone via
``python -m pytest app/tests/test_provenance.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.limitations import GENERAL, limitations_report  # noqa: E402
from app.project import load_project, save_project  # noqa: E402
from app.results import Result, ResultStore  # noqa: E402
from app.version import __version__, build_string, provenance  # noqa: E402
from parametric.standard import basic_rocket  # noqa: E402


# ----------------------------------------------------------------------
# Version
# ----------------------------------------------------------------------

def test_build_string_always_names_the_version():
    assert build_string().startswith(__version__)


def test_version_matches_pyproject():
    """The declared version and the one the app reports are one number.

    They live in two files because one is read by pip and the other by a
    person in a bug report, and a disagreement between them is worse than
    having no version at all.
    """
    text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    declared = next(
        line.split("=", 1)[1].strip().strip('"')
        for line in text.splitlines()
        if line.startswith("version")
    )
    assert declared == __version__


def test_provenance_has_both_keys():
    keys = provenance()
    assert set(keys) == {"app_version", "app_revision"}


# ----------------------------------------------------------------------
# Results and projects carry the build that made them
# ----------------------------------------------------------------------

def test_a_result_stamps_itself_with_the_build():
    result = Result(index=1, kind="flight", label="Flight", fingerprint="abc")
    assert result.build["app_version"] == __version__


def test_a_saved_run_still_knows_which_build_flew_it(tmp_path):
    """The case this exists for: a project file outliving its session."""
    store = ResultStore()
    store.add("flight", "Flight", "fingerprint-1")

    path = tmp_path / "project.json"
    save_project(path, basic_rocket(), store)
    _, restored = load_project(path)

    assert restored[0].build["app_version"] == __version__


def test_a_project_records_the_build_that_wrote_it(tmp_path):
    import json

    path = tmp_path / "project.json"
    save_project(path, basic_rocket(), ResultStore())
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["app"]["app_version"] == __version__


def test_a_vehicle_file_written_before_provenance_still_opens(tmp_path):
    """Backwards compatibility is the whole reason the loader is forgiving."""
    import json

    path = tmp_path / "old.json"
    path.write_text(json.dumps(basic_rocket().to_dict()), encoding="utf-8")

    model, results = load_project(path)
    assert results == []
    assert model.name


def test_a_result_from_before_provenance_reads_back_empty():
    from app.project import result_from_dict

    restored = result_from_dict({"index": 3, "kind": "flight", "label": "Flight"})
    assert restored.build == {}


# ----------------------------------------------------------------------
# Limitations
# ----------------------------------------------------------------------

def test_the_limitations_report_names_every_limitation():
    report = limitations_report(basic_rocket())
    for title, _ in GENERAL:
        assert title in report


def test_a_gentle_boattail_raises_no_caveat():
    """The demonstrator's taper is well inside the clamp; it must stay quiet."""
    from parametric.standard import boattailed_rocket

    assert "This vehicle in particular" not in limitations_report(boattailed_rocket())


def test_the_report_survives_a_model_it_cannot_read():
    class Awkward:
        def walk(self):
            raise RuntimeError("not a vehicle")

    assert limitations_report(Awkward())        # no exception, still a report


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

def test_installing_diagnostics_returns_a_writable_log():
    from app import diagnostics

    path = diagnostics.install()
    assert path.parent.is_dir()
    assert diagnostics.install() == path        # idempotent


def test_an_unhandled_exception_reaches_the_log():
    from app import diagnostics

    diagnostics.install()
    try:
        raise ValueError("a distinctive message for the log")
    except ValueError:
        diagnostics.report(*sys.exc_info())

    text = diagnostics.log_path().read_text(encoding="utf-8", errors="replace")
    assert "a distinctive message for the log" in text


def test_the_same_failure_is_only_shown_once():
    """A paint or timer callback fails every frame; one dialog, not a hundred."""
    from app import diagnostics

    diagnostics.install()
    diagnostics._reported.clear()

    shown: list[str] = []
    original = diagnostics._show
    diagnostics._show = lambda exc_type, exc_value: shown.append(str(exc_value))
    try:
        for _ in range(5):
            try:
                raise RuntimeError("repeating failure")
            except RuntimeError:
                diagnostics.report(*sys.exc_info())
    finally:
        diagnostics._show = original

    assert len(shown) == 1


# ----------------------------------------------------------------------
# The headless guard
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method", ["_about", "_show_limitations"])
def test_help_dialogs_do_not_block_a_headless_run(qt_window, method):
    """Regression: these called ``exec`` directly and hung the suite.

    ``QMessageBox.exec`` under the offscreen platform does not fail, it waits
    for a button press that cannot come -- so the failure mode is a test run
    that never finishes, which is far harder to diagnose than a red test.
    """
    getattr(qt_window, method)()        # must simply return


@pytest.fixture
def qt_window():
    from PySide6.QtWidgets import QApplication

    from app.mainwindow import MainWindow

    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(basic_rocket())
    yield window
    window.close()
    app.processEvents()
