"""Tests for the results store and comparison panel.

The point of keeping results is answering "did that change help". These check
that a run survives being read, that two runs of the same kind difference
correctly, and that a run whose model has moved on is marked rather than
presented as current.

Runs under pytest, and standalone via ``python -m pytest app/tests/test_results.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.results import Metric, Result, ResultStore  # noqa: E402


def _make(store: ResultStore, kind="flight", fingerprint="aaa", apogee=4000.0):
    return store.add(
        kind, kind.title(), fingerprint,
        settings={"elevation": "85°"},
        metrics=[
            Metric("Apogee", apogee, "m", 0, higher_is_better=True),
            Metric("Max-Q", 180.0, "kPa", 0),
        ],
        series={"Altitude vs time": (np.arange(5.0), np.arange(5.0) * 100)},
    )


# ------------------------------------------------------------------ store


def test_results_are_numbered_and_kept():
    store = ResultStore()
    first = _make(store)
    second = _make(store)
    assert (first.index, second.index) == (1, 2)
    assert len(store) == 2


def test_newest_first_is_reversed():
    store = ResultStore()
    _make(store)
    latest = _make(store)
    assert store.newest_first()[0] is latest


def test_of_kind_filters():
    store = ResultStore()
    _make(store, kind="flight")
    _make(store, kind="aero")
    assert len(store.of_kind("flight")) == 1
    assert store.latest("aero").kind == "aero"


def test_clear_resets_the_numbering():
    store = ResultStore()
    _make(store)
    store.clear()
    assert len(store) == 0
    assert _make(store).index == 1


def test_latest_of_a_missing_kind_is_none():
    assert ResultStore().latest("flight") is None


# ----------------------------------------------------------------- metrics


def test_metric_formats_with_units():
    assert Metric("Apogee", 4467.8, "m", 0).format() == "4,468 m"
    assert Metric("Mach", 1.723, "", 2).format() == "1.72"


def test_delta_reports_absolute_and_percent():
    newer = Metric("Apogee", 4377.0, "m", 0)
    older = Metric("Apogee", 4468.0, "m", 0)
    text = newer.delta_text(older)
    assert "-91" in text and "-2.0%" in text


def test_delta_handles_a_zero_baseline():
    assert "+" in Metric("x", 5.0).delta_text(Metric("x", 0.0))


def test_metric_lookup_by_label():
    store = ResultStore()
    result = _make(store)
    assert result.metric("Apogee").value == 4000.0
    assert result.metric("nope") is None


# ------------------------------------------------------------- staleness


def test_a_result_knows_which_model_it_describes():
    store = ResultStore()
    result = _make(store, fingerprint="abc123")
    assert result.is_current("abc123")
    assert not result.is_current("def456")


# ------------------------------------------------------------ the panel

pytest.importorskip("cadquery", reason="the app needs the cad extra")
pytest.importorskip("PySide6", reason="the app needs PySide6")


@pytest.fixture(scope="module")
def qt_app():
    from PySide6.QtWidgets import QApplication

    from app.viewport import configure_surface_format

    configure_surface_format()
    yield QApplication.instance() or QApplication(sys.argv[:1])


@pytest.fixture
def panel(qt_app):
    from app.resultspanel import ResultsPanel

    store = ResultStore()
    widget = ResultsPanel(store)
    yield widget, store
    widget.close()


def _select(widget, results):
    widget.runs.clearSelection()
    for index in range(widget.runs.topLevelItemCount()):
        item = widget.runs.topLevelItem(index)
        if widget._result_of(item) in results:
            item.setSelected(True)


@pytest.mark.slow
def test_panel_lists_runs_newest_first(panel):
    widget, store = panel
    _make(store, apogee=1000.0)
    latest = _make(store, apogee=2000.0)
    widget.set_fingerprint("aaa")
    assert widget.runs.topLevelItem(0).text(0) == latest.title


@pytest.mark.slow
def test_panel_marks_stale_runs(panel):
    widget, store = panel
    _make(store, fingerprint="old")
    widget.set_fingerprint("new")
    assert widget.runs.topLevelItem(0).text(2) == "stale"
    widget.set_fingerprint("old")
    assert widget.runs.topLevelItem(0).text(2) == "current"


@pytest.mark.slow
def test_selecting_two_runs_produces_a_diff(panel):
    widget, store = panel
    older = _make(store, apogee=4468.0)
    newer = _make(store, apogee=4377.0)
    widget.set_fingerprint("aaa")
    _select(widget, [older, newer])

    rows = {}
    for index in range(widget.detail.topLevelItemCount()):
        top = widget.detail.topLevelItem(index)
        if top.text(0) != "results":
            continue
        for child_index in range(top.childCount()):
            child = top.child(child_index)
            rows[child.text(0)] = (child.text(1), child.text(2), child.text(3))

    assert "Apogee" in rows
    value, compared, delta = rows["Apogee"]
    assert "4,377" in value and "4,468" in compared
    assert "-91" in delta


@pytest.mark.slow
def test_a_single_selection_shows_no_comparison(panel):
    widget, store = panel
    only = _make(store)
    widget.set_fingerprint("aaa")
    _select(widget, [only])
    for index in range(widget.detail.topLevelItemCount()):
        top = widget.detail.topLevelItem(index)
        if top.text(0) == "results":
            assert top.child(0).text(2) == ""


@pytest.mark.slow
def test_runs_of_different_kinds_are_not_differenced(panel):
    """Comparing a flight to an aero sweep is meaningless, so it is not offered."""
    widget, store = panel
    flight = _make(store, kind="flight")
    aero = _make(store, kind="aero")
    widget.set_fingerprint("aaa")
    _select(widget, [flight, aero])
    for index in range(widget.detail.topLevelItemCount()):
        top = widget.detail.topLevelItem(index)
        if top.text(0) == "results":
            assert top.child(0).text(2) == ""


@pytest.mark.slow
def test_series_list_follows_the_selection(panel):
    widget, store = panel
    result = _make(store)
    widget.set_fingerprint("aaa")
    _select(widget, [result])
    names = [widget.series_combo.itemText(i)
             for i in range(widget.series_combo.count())]
    assert "Altitude vs time" in names


@pytest.mark.slow
def test_show_trajectory_only_enabled_for_one_flight(panel):
    widget, store = panel
    flight = _make(store, kind="flight")
    other = _make(store, kind="flight")
    flight.payload["result"] = object()          # a trajectory in hand
    other.payload["result"] = object()
    widget.set_fingerprint("aaa")

    _select(widget, [flight])
    assert widget._show_button.isEnabled()
    _select(widget, [flight, other])
    assert not widget._show_button.isEnabled()


@pytest.mark.slow
def test_show_trajectory_is_greyed_for_a_restored_run(panel):
    """A run read back from a project has no trajectory; the button says so."""
    widget, store = panel
    restored = _make(store, kind="flight")
    restored.payload["restored"] = True
    widget.set_fingerprint("aaa")
    _select(widget, [restored])
    assert not widget._show_button.isEnabled()
    assert widget._export_button.isEnabled()


@pytest.mark.slow
def test_plot_survives_an_empty_and_a_flat_series(panel):
    """A constant series has zero range; the axis mapping must not divide by it."""
    widget, store = panel
    store.add("flight", "Flight", "aaa",
              series={"flat": (np.zeros(5), np.full(5, 3.0))},
              metrics=[Metric("x", 1.0)])
    widget.set_fingerprint("aaa")
    _select(widget, list(store))
    widget.plot.set_series([], [])
    widget.plot.repaint()
    widget.plot.set_series([(np.zeros(5), np.full(5, 3.0))], ["flat"])
    widget.plot.repaint()


if __name__ == "__main__":
    print("Run under pytest: python -m pytest app/tests/test_results.py")


# ----------------------------------------------------- flight series


def test_vertical_velocity_crosses_zero_at_apogee_and_speed_does_not():
    """A tilted launch never brings the speed magnitude to zero: at apogee
    the vehicle is still moving sideways. The vertical component is the
    signal that does cross zero, and the horizontal one is the drift."""
    from types import SimpleNamespace

    from app.results import flight_result
    from trajectory import simulation as tm
    from trajectory.analysis.export import max_q
    from trajectory.analysis.statistics import flight_statistics

    result = tm.RocketSimulation().run(launch_elevation=np.radians(80.0), dt=0.1)
    states = result.y.T
    stats = flight_statistics(states, result.t)
    settings = SimpleNamespace(
        elevation_deg=80.0, azimuth_deg=0.0, rail_length_m=5.0,
        wind_speed_mps=0.0, wind_direction_deg=0.0, use_recovery=False,
    )
    entry = flight_result(result, stats, max_q(result), settings, "fp", False)

    t, vy, *_ = entry["series"]["Vertical velocity vs time"]
    _, vh, *_ = entry["series"]["Horizontal speed vs time"]
    _, speed, *_ = entry["series"]["Speed vs time"]

    crossings = np.flatnonzero(np.diff(np.sign(vy)) < 0)
    assert len(crossings) == 1, "one apogee, one sign change"
    assert t[crossings[0]] == pytest.approx(stats["apogee_time"], abs=0.2)
    i_ap = int(np.argmax(states[:, 1]))
    assert vh[i_ap] > 10.0, "an 80 deg launch drifts"
    assert speed[i_ap] == pytest.approx(vh[i_ap], rel=1e-3)
    assert speed.min() > 0.0 or speed.argmin() == 0


def _flown():
    """The default vehicle, flown and logged, for the log-carrying result."""
    from types import SimpleNamespace

    from app.results import flight_result
    from trajectory import simulation as tm
    from trajectory.analysis.export import max_q
    from trajectory.analysis.flightlog import FlightLog
    from trajectory.analysis.statistics import flight_statistics

    sim = tm.RocketSimulation()
    result = sim.run(launch_elevation=np.radians(85.0), t_max=90.0, dt=0.5)
    stats = flight_statistics(result.y.T, result.t)
    settings = SimpleNamespace(
        elevation_deg=85.0, azimuth_deg=0.0, rail_length_m=5.0,
        wind_speed_mps=0.0, wind_direction_deg=0.0, use_recovery=False,
    )
    log = FlightLog.from_flight(sim, result)
    return flight_result(result, stats, max_q(result), settings, "fp", False, log=log)


def test_a_logged_flight_reports_what_the_airframe_felt():
    entry = _flown()
    labels = [m.label for m in entry["metrics"]]
    assert "Max acceleration" in labels
    assert "Angle of attack vs time" in entry["series"]
    assert "Acceleration vs time" in entry["series"]
    assert "Thrust vs time" in entry["series"]
    # No table on the default vehicle, so no margin is claimed.
    assert "Static margin vs time" not in entry["series"]
    assert entry["payload"]["log"] is not None


@pytest.mark.slow
def test_a_flight_exports_its_full_history(panel, tmp_path):
    widget, store = panel
    entry = _flown()
    result = store.add(entry["kind"], entry["label"], entry["fingerprint"],
                       settings=entry["settings"], metrics=entry["metrics"],
                       series=entry["series"], payload=entry["payload"])
    widget.set_fingerprint("fp")
    written = widget.export_result(result, tmp_path / "run.csv")
    names = {p.name for p in written}
    assert names == {"run.csv", "run.png"}
    header = (tmp_path / "run.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "alpha_deg" in header and "acceleration_g" in header


@pytest.mark.slow
def test_any_run_exports_its_series(panel, tmp_path):
    widget, store = panel
    result = _make(store, kind="aero")
    widget.set_fingerprint("aaa")
    _select(widget, [result])
    written = widget.export_result(result, tmp_path / "aero.csv")
    assert (tmp_path / "aero.csv").exists()
    header = (tmp_path / "aero.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "Altitude vs time" in header
    assert any(p.suffix == ".png" for p in written)
