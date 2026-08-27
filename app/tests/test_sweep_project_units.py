"""Tests for design sweeps, project persistence and unit display.

Runs under pytest, and standalone via
``python -m pytest app/tests/test_sweep_project_units.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.units import UNITS, Units, UnitSystem  # noqa: E402


# ------------------------------------------------------------------ units


@pytest.fixture(autouse=True)
def si_units():
    """Every test starts in SI, since the unit system is shared state."""
    UNITS.system = UnitSystem.SI
    yield
    UNITS.system = UnitSystem.SI


def test_si_is_a_passthrough():
    units = Units(UnitSystem.SI)
    assert units.display(1.234, "m") == (1.234, "m")
    assert units.to_si(1.234, "m") == 1.234


def test_small_lengths_read_in_inches():
    units = Units(UnitSystem.IMPERIAL)
    value, unit = units.display(0.09, "m")
    assert unit == "in"
    assert np.isclose(value, 3.5433, atol=1e-3)


def test_large_lengths_read_in_feet():
    """A fin span in feet and an apogee in inches are both useless."""
    units = Units(UnitSystem.IMPERIAL)
    value, unit = units.display(4467.0, "m")
    assert unit == "ft"
    assert np.isclose(value, 14656.0, rtol=1e-3)


def test_prefer_feet_can_be_forced_either_way():
    units = Units(UnitSystem.IMPERIAL)
    assert units.display(4467.0, "m", prefer_feet=False)[1] == "in"
    assert units.display(0.09, "m", prefer_feet=True)[1] == "ft"


def test_mass_force_and_pressure_convert():
    units = Units(UnitSystem.IMPERIAL)
    assert np.isclose(units.display(1.0, "kg")[0], 2.20462, rtol=1e-4)
    assert np.isclose(units.display(1000.0, "N")[0], 224.809, rtol=1e-4)
    assert np.isclose(units.display(101325.0, "Pa")[0], 14.6959, rtol=1e-4)


def test_round_trip_is_exact_enough_to_type_into():
    units = Units(UnitSystem.IMPERIAL)
    for value, unit in [(0.09, "m"), (4.4, "kg"), (1500.0, "N")]:
        shown, _ = units.display(value, unit, prefer_feet=False)
        assert np.isclose(units.to_si(shown, unit, prefer_feet=False), value)


def test_typed_inches_land_on_the_right_metres():
    units = Units(UnitSystem.IMPERIAL)
    assert np.isclose(units.to_si(4.0, "m", prefer_feet=False), 0.1016, atol=1e-6)


def test_an_unknown_unit_is_left_alone():
    units = Units(UnitSystem.IMPERIAL)
    assert units.display(2.5, "cal") == (2.5, "cal")
    assert units.display(1.7, "") == (1.7, "")


def test_metric_formatting_follows_the_system():
    from app.results import Metric

    metric = Metric("Dry mass", 4.3946, "kg", 3)
    assert "kg" in metric.format()
    UNITS.system = UnitSystem.IMPERIAL
    assert "lb" in metric.format()
    assert "9.688" in metric.format()


def test_a_comparison_never_mixes_inches_with_feet():
    """Both sides scale by the same factor, taken from the primary value."""
    from app.results import Metric

    UNITS.system = UnitSystem.IMPERIAL
    newer = Metric("Apogee", 4377.0, "m", 0)
    older = Metric("Apogee", 4468.0, "m", 0)
    text = newer.delta_text(older)
    assert "-2.0%" in text
    # -91 m in feet is about -299; in inches it would be -3583.
    assert "-299" in text


# ----------------------------------------------------------------- sweeps

pytest.importorskip("cadquery", reason="sweeps need the cad extra")

from parametric.standard import basic_rocket  # noqa: E402
from parametric.sweep import (  # noqa: E402
    SweepSettings,
    SweepVariable,
    default_range,
    run_sweep,
    sweepable_parms,
)


def test_only_bounded_designable_parms_are_offered():
    model = basic_rocket()
    offered = sweepable_parms(model)
    assert offered
    names = {parm for _, parm, _ in offered}
    assert "span" in names
    assert "count" not in names, "fin count is a choice, not a continuum"


def test_default_range_brackets_the_current_value():
    model = basic_rocket()
    fins = model.find("fins")
    low, high = default_range(model, fins.path, "span")
    assert low < fins.get("span") < high


def test_default_range_respects_the_parm_bounds():
    model = basic_rocket()
    tube = model.find("motor_tube")
    low, _ = default_range(model, tube.path, "wall_thickness")
    assert low >= tube.parm("wall_thickness").minimum


def test_sweep_values_span_the_range():
    variable = SweepVariable("a/b", "span", 0.06, 0.15, steps=4)
    values = variable.values()
    assert len(values) == 4
    assert np.isclose(values[0], 0.06) and np.isclose(values[-1], 0.15)


@pytest.mark.slow
def test_a_sweep_restores_the_model():
    """It mutates the vehicle to evaluate each point; it must put it back."""
    model = basic_rocket()
    fins = model.find("fins")
    before = fins.get("span")
    settings = SweepSettings(
        SweepVariable(fins.path, "span", 0.06, 0.14, steps=3),
        include_flight=False,
    )
    run_sweep(model, settings)
    assert np.isclose(model.find("fins").get("span"), before)


@pytest.mark.slow
def test_a_sweep_records_a_point_per_value():
    model = basic_rocket()
    fins = model.find("fins")
    settings = SweepSettings(
        SweepVariable(fins.path, "span", 0.06, 0.14, steps=4),
        include_flight=False,
    )
    result = run_sweep(model, settings)
    assert len(result.points) == 4
    assert "static margin cal" in result.metric_names()


@pytest.mark.slow
def test_bigger_fins_raise_the_margin_and_the_mass():
    """The trade the sweep exists to show."""
    model = basic_rocket()
    fins = model.find("fins")
    settings = SweepSettings(
        SweepVariable(fins.path, "span", 0.06, 0.15, steps=5),
        include_flight=False,
    )
    result = run_sweep(model, settings)
    margins = result.series("static margin cal")
    masses = result.series("dry mass kg")
    assert margins[-1] > margins[0]
    assert masses[-1] > masses[0]


@pytest.mark.slow
def test_a_sweep_can_be_cancelled():
    model = basic_rocket()
    fins = model.find("fins")
    settings = SweepSettings(
        SweepVariable(fins.path, "span", 0.06, 0.15, steps=8),
        include_flight=False,
    )
    result = run_sweep(model, settings, progress=lambda i, n, v: i < 3)
    assert len(result.points) == 3


@pytest.mark.slow
def test_best_finds_the_extreme():
    model = basic_rocket()
    fins = model.find("fins")
    settings = SweepSettings(
        SweepVariable(fins.path, "span", 0.06, 0.15, steps=4),
        include_flight=False,
    )
    result = run_sweep(model, settings)
    best = result.best("static margin cal", maximise=True)
    assert np.isclose(best.value, 0.15)


# --------------------------------------------------------------- project


def test_a_project_round_trips_with_its_results(tmp_path):
    from app.project import load_project, save_project
    from app.results import Metric, ResultStore

    model = basic_rocket()
    store = ResultStore()
    store.add(
        "flight", "Flight", "fp123",
        settings={"elevation": "85°"},
        metrics=[Metric("Apogee", 4467.0, "m", 0, higher_is_better=True)],
        series={"Altitude vs time": (np.arange(4.0), np.arange(4.0) * 100)},
    )

    path = save_project(tmp_path / "p.json", model, store)
    reloaded, results = load_project(path)

    assert reloaded.name == model.name
    assert len(results) == 1
    assert results[0].metrics[0].label == "Apogee"
    assert np.isclose(results[0].metrics[0].value, 4467.0)
    assert "Altitude vs time" in results[0].series
    assert len(results[0].series["Altitude vs time"][0]) == 4


def test_a_bare_vehicle_file_still_opens(tmp_path):
    """Files written before projects existed must not become unreadable."""
    from app.project import load_project

    model = basic_rocket()
    path = model.save(tmp_path / "vehicle.json")
    reloaded, results = load_project(path)
    assert reloaded.name == model.name
    assert results == []


def test_restored_results_are_marked_as_restored(tmp_path):
    from app.project import load_project, save_project
    from app.results import Metric, ResultStore

    store = ResultStore()
    store.add("flight", "Flight", "fp", metrics=[Metric("Apogee", 1.0, "m", 0)])
    path = save_project(tmp_path / "p.json", basic_rocket(), store)
    _, results = load_project(path)
    assert results[0].payload.get("restored") is True


def test_document_state_tracks_dirtiness(tmp_path):
    from app.project import DocumentState

    state = DocumentState()
    assert state.title == "untitled"
    state.mark_dirty()
    assert state.title.endswith("*")
    state.mark_saved(tmp_path / "x.json")
    assert state.title == "x.json"


def test_autosave_goes_beside_the_document(tmp_path):
    from app.project import autosave_path

    beside = autosave_path(tmp_path / "rocket.json", "Rocket")
    assert beside.parent == tmp_path
    assert "autosave" in beside.name

    orphan = autosave_path(None, "Untitled Rocket")
    assert orphan.parent.exists()
    assert "autosave" in orphan.name


if __name__ == "__main__":
    print("Run under pytest: python -m pytest app/tests/test_sweep_project_units.py")
