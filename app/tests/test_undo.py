"""Tests for undo and redo.

Undo is only worth having if it is trustworthy, so these check the awkward
cases rather than the happy path: that a slider drag is one step and not two
hundred, that redo is discarded when you branch, that the stack is bounded,
and that restoring a snapshot puts the selection back on a component that is no
longer the same Python object.

Runs under pytest, and standalone via ``python -m pytest app/tests/test_undo.py``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.undo import COALESCE_SECONDS, UndoStack  # noqa: E402


# ------------------------------------------------------------------ stack


def _stack():
    stack = UndoStack()
    stack.reset({"n": 0})
    return stack


def test_a_fresh_stack_cannot_undo():
    stack = _stack()
    assert not stack.can_undo
    assert not stack.can_redo
    assert stack.undo() is None


def test_push_then_undo_returns_the_previous_state():
    stack = _stack()
    stack.push({"n": 1}, "first")
    assert stack.can_undo
    assert stack.undo().state == {"n": 0}


def test_redo_returns_forward_again():
    stack = _stack()
    stack.push({"n": 1}, "first")
    stack.undo()
    assert stack.can_redo
    assert stack.redo().state == {"n": 1}
    assert not stack.can_redo


def test_a_new_edit_discards_the_redo_future():
    """Branching after undo throws away what was ahead, as every editor does."""
    stack = _stack()
    stack.push({"n": 1}, "first")
    stack.push({"n": 2}, "second")
    stack.undo()
    assert stack.can_redo
    stack.push({"n": 99}, "third")
    assert not stack.can_redo
    assert stack.undo().state == {"n": 1}


def test_same_label_in_quick_succession_coalesces():
    """A slider drag must undo as one action, not two hundred."""
    stack = _stack()
    stack.push({"n": 1}, "fins.span")
    depth = stack.depth
    for value in range(2, 12):
        stack.push({"n": value}, "fins.span")
    assert stack.depth == depth
    assert stack.undo().state == {"n": 0}


def test_a_different_label_starts_a_new_entry():
    stack = _stack()
    stack.push({"n": 1}, "fins.span")
    stack.push({"n": 2}, "fins.sweep")
    assert stack.depth == 3


def test_coalescing_expires():
    stack = UndoStack()
    stack.reset({"n": 0})
    stack.push({"n": 1}, "fins.span")
    stack.push({"n": 2}, "fins.span")          # coalesces
    depth = stack.depth
    stack._entries[stack.position].stamp -= COALESCE_SECONDS * 2
    stack.push({"n": 3}, "fins.span")          # too late to coalesce
    assert stack.depth == depth + 1


def test_the_first_edit_is_never_coalesced_away():
    """Coalescing must not eat the entry that returns you to the opening state."""
    stack = _stack()
    stack.push({"n": 1}, "open")
    stack.push({"n": 2}, "open")
    assert stack.undo().state == {"n": 0}


def test_the_stack_is_bounded():
    stack = UndoStack(limit=10)
    stack.reset({"n": 0})
    for value in range(1, 50):
        stack.push({"n": value}, f"edit {value}")
    assert stack.depth <= 10
    assert stack.can_undo


def test_applying_suppresses_recording():
    """Restoring a snapshot must not push the restore back onto the stack."""
    stack = _stack()
    stack.push({"n": 1}, "first")
    depth = stack.depth
    stack.applying = True
    stack.push({"n": 2}, "should be ignored")
    stack.applying = False
    assert stack.depth == depth


def test_labels_describe_the_action():
    stack = _stack()
    stack.push({"n": 1}, "fins.span")
    assert stack.undo_label == "fins.span"
    stack.undo()
    assert stack.redo_label == "fins.span"


def test_reset_starts_a_new_history():
    stack = _stack()
    stack.push({"n": 1}, "first")
    stack.reset({"n": 100})
    assert not stack.can_undo and not stack.can_redo
    assert stack.depth == 1


# ------------------------------------------------------------ in the window

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
    win.show()
    qt_app.processEvents()
    yield win
    win.close()


@pytest.mark.slow
def test_a_parm_edit_is_undoable(window):
    fins = window.model.find("fins")
    before = fins.get("span")
    fins.set("span", before * 1.5)
    window._on_parm_changed(fins, "span", fins.get("span"))

    window.undo()
    assert np.isclose(window.model.find("fins").get("span"), before)


@pytest.mark.slow
def test_adding_and_deleting_are_undoable(window):
    before = len(list(window.model.walk()))
    window._add_component("pointmass")
    assert len(list(window.model.walk())) == before + 1

    window.undo()
    assert len(list(window.model.walk())) == before

    window.redo()
    assert len(list(window.model.walk())) == before + 1


@pytest.mark.slow
def test_undo_restores_mass_and_geometry(window):
    """The whole model comes back, not just the parm that changed."""
    before_mass = window.model.mass_summary().dry_mass_kg
    tube = window.model.find("motor_tube")
    window._select_component(tube)
    tube.set("wall_thickness", tube.get("wall_thickness") * 2)
    window._on_parm_changed(tube, "wall_thickness", tube.get("wall_thickness"))
    assert window.model.mass_summary().dry_mass_kg > before_mass

    window.undo()
    assert np.isclose(window.model.mass_summary().dry_mass_kg, before_mass)


@pytest.mark.slow
def test_undo_restores_the_selection(window):
    """Restoring builds new objects, so selection has to survive by path."""
    fins = window.model.find("fins")
    window._select_component(fins)
    fins.set("span", 0.13)
    window._on_parm_changed(fins, "span", 0.13)

    window.undo()
    selected = window._current_component()
    assert selected is not None
    assert selected.name == "fins"
    assert selected is not fins, "the restored model should be fresh objects"


@pytest.mark.slow
def test_a_role_change_is_undoable(window):
    from parametric.roles import AeroRole

    nose = window.model.find("nose")
    window._select_component(nose)
    window.editor._on_role("internal")
    assert window.model.find("nose").aero_role is AeroRole.INTERNAL

    window.undo()
    assert window.model.find("nose").aero_role is AeroRole.AUTO


@pytest.mark.slow
def test_a_motor_curve_edit_is_undoable(window):
    motor = window.model.motors[0]
    before = len(motor.curve)
    window._select_component(motor)
    motor.add_curve_point(9.0, 50.0)
    window._on_section_changed(motor)
    assert len(window.model.motors[0].curve) == before + 1

    window.undo()
    assert len(window.model.motors[0].curve) == before


@pytest.mark.slow
def test_a_section_edit_is_undoable(window):
    stack = window.model.find("nose")
    before = len(stack.sections)
    window._select_component(stack)
    from parametric.xsec import XSec, XSecShape

    stack.add_section(XSec(0.2, XSecShape.CIRCLE, 0.08, name="extra"))
    window._on_section_changed(stack)
    assert len(window.model.find("nose").sections) == before + 1

    window.undo()
    assert len(window.model.find("nose").sections) == before


@pytest.mark.slow
def test_loading_a_model_clears_the_history(window):
    from parametric.standard import boattailed_rocket

    fins = window.model.find("fins")
    fins.set("span", 0.13)
    window._on_parm_changed(fins, "span", 0.13)
    assert window.undo_stack.can_undo

    window.set_model(boattailed_rocket())
    assert not window.undo_stack.can_undo


@pytest.mark.slow
def test_menu_labels_name_the_action(window):
    fins = window.model.find("fins")
    fins.set("span", 0.12)
    window._on_parm_changed(fins, "span", 0.12)
    assert "fins.span" in window._action_undo.text()
    assert window._action_undo.isEnabled()

    window.undo()
    assert "fins.span" in window._action_redo.text()


if __name__ == "__main__":
    print("Run under pytest: python -m pytest app/tests/test_undo.py")
