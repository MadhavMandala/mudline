"""Tests for building a vehicle through the application.

These drive the window's own actions rather than the model API, because the
question being asked is whether the *tool* can build a rocket -- adding
components, editing cross-sections, deleting things -- not whether the
underlying data structures allow it.

Runs under pytest, and standalone via ``python app/tests/test_build.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

pytest.importorskip("cadquery", reason="the app needs the cad extra")
PySide6 = pytest.importorskip("PySide6", reason="the app needs PySide6")

from parametric.components import Stack  # noqa: E402
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


# ----------------------------------------------------------- components


@pytest.mark.slow
def test_adding_a_body_gives_something_that_lofts(window):
    """An empty stack cannot loft, so a new one must arrive already valid."""
    window._add_component("stack")
    body = window.model.find("body")
    assert isinstance(body, Stack)
    assert len(body.sections) >= 2
    assert window.model.validate() == []


@pytest.mark.slow
def test_fins_attach_to_the_selected_body(window):
    window._add_component("stack")
    window._select_component(window.model.find("body"))
    window._add_component("finset")
    added = [f for f in window.model.fin_sets if f.name != "fins"][0]
    assert added.parent.name == "body"


@pytest.mark.slow
def test_fins_without_a_body_are_refused_gracefully(qt_app):
    from app.mainwindow import MainWindow
    from parametric.model import VehicleModel

    window = MainWindow(VehicleModel("empty"))
    qt_app.processEvents()
    window._add_component("finset")          # must not raise
    assert window.model.fin_sets == []
    window.close()


@pytest.mark.slow
def test_added_names_do_not_collide(window):
    window._add_component("pointmass")
    window._add_component("pointmass")
    names = [c.name for c in window.model.point_masses]
    assert len(names) == len(set(names))


@pytest.mark.slow
def test_deleting_removes_it_from_the_model(window):
    window._add_component("pointmass")
    added = window.model.point_masses[-1]
    window._select_component(added)
    window._delete_component()
    assert added not in window.model.point_masses


@pytest.mark.slow
def test_the_last_body_cannot_be_deleted(window):
    """A vehicle with no body has nothing to loft, aero or fly."""
    for stack in list(window.model.stacks)[1:]:
        window._select_component(stack)
        window._delete_component()
    remaining = window.model.stacks[0]
    window._select_component(remaining)
    window._delete_component()
    assert remaining in window.model.stacks


# ------------------------------------------------------------- sections


@pytest.mark.slow
def test_sections_can_be_added_and_removed(window):
    from app.sectioneditor import SectionEditor

    stack = window.model.find("nose")
    editor = SectionEditor()
    editor.set_stack(stack)
    editor.table.setCurrentCell(0, 0)

    before = len(stack.sections)
    editor._add_section()
    assert len(stack.sections) == before + 1

    editor._duplicate_section()
    assert len(stack.sections) == before + 2

    editor._remove_section()
    assert len(stack.sections) == before + 1


@pytest.mark.slow
def test_a_stack_cannot_be_reduced_below_two_sections(window):
    from app.sectioneditor import SectionEditor

    stack = Stack("tiny", wall_thickness_m=0.0)
    stack.add_tube(0.4, 0.08)
    editor = SectionEditor()
    editor.set_stack(stack)
    for _ in range(5):
        editor.table.setCurrentCell(0, 0)
        editor._remove_section()
    assert len(stack.sections) == 2


@pytest.mark.slow
def test_editing_a_circular_section_keeps_it_circular(window):
    """Width and height must move together, or a circle silently becomes an ellipse."""
    from app.sectioneditor import SectionEditor

    stack = window.model.find("motor_tube")
    editor = SectionEditor()
    editor.set_stack(stack)
    editor.table.setCurrentCell(0, 0)
    editor._edit("width", 0.15)

    section = stack.sorted_sections()[0]
    assert np.isclose(section.width_m, 0.15)
    assert np.isclose(section.height_m, 0.15)


@pytest.mark.slow
def test_section_edits_change_the_mass(window):
    from app.sectioneditor import SectionEditor

    stack = window.model.find("motor_tube")
    before = stack.mass_kg()
    editor = SectionEditor()
    editor.set_stack(stack)
    editor.table.setCurrentCell(0, 0)
    editor._edit("width", stack.max_diameter_m * 1.6)
    assert stack.mass_kg() > before


# ------------------------------------------------------------ readouts


@pytest.mark.slow
def test_status_readout_reports_the_roll_up(window):
    text = window._mass_label.text()
    for token in ("dry", "wet", "CG", "margin", "fineness"):
        assert token in text


@pytest.mark.slow
def test_selecting_a_component_shows_its_parms(window):
    window._select_component(window.model.find("fins"))
    assert window.editor._component is window.model.find("fins")
    assert window.editor._rows, "the fin set should expose editable parms"


if __name__ == "__main__":
    print("Run under pytest: python -m pytest app/tests/test_build.py")
