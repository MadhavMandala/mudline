"""Where a crash goes.

A GUI launched from a shortcut has no console. Until this module existed, an
exception raised inside a Qt slot printed a traceback to a stderr nobody was
attached to and the application carried on with whatever half-finished state
the exception left behind -- so the report that came back was "it broke", and
there was nothing to work from.

Three things are installed here, and all three end in the same log file:

``sys.excepthook``      anything that escapes a slot or a menu action
``threading.excepthook``  the same, on a worker thread
Qt's message handler    Qt's own warnings, which precede a good share of
                        crashes and are otherwise discarded

The user-facing half matters as much as the file. An unexpected error raises a
dialog that says what happened, names the log, and offers to open it -- once
per distinct error, because a failure inside a paint or timer callback repeats
at the frame rate and a dialog per occurrence makes the application impossible
to even close.

The log is deliberately outside the repository, under the user's local app
data, so it survives a reinstall and a fresh clone, and so a teammate can find
and send it without being walked through a checkout.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER = logging.getLogger("mudline")

#: Signatures already shown to the user this session. A repeating paint error
#: should reach the log every time and the screen once.
_reported: set[str] = set()

_log_path: Path | None = None


def log_directory() -> Path:
    """Per-user, outside the checkout, stable across reinstalls."""
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        directory = root / "Mudline" / "logs"
    elif sys.platform == "darwin":
        directory = Path.home() / "Library" / "Logs" / "Mudline"
    else:
        root = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state"))
        directory = root / "mudline" / "logs"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_path() -> Path | None:
    """The file being written this session, once :func:`install` has run."""
    return _log_path


def install(app_name: str = "Mudline") -> Path:
    """Set up logging and take over the exception hooks. Returns the log path.

    Safe to call twice; the second call is a no-op beyond returning the path.
    """
    global _log_path
    if _log_path is not None:
        return _log_path

    _log_path = log_directory() / "mudline.log"

    handler = RotatingFileHandler(
        _log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")
    )
    LOGGER.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    LOGGER.propagate = False

    _log_session_header(app_name)
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    _install_qt_handler()
    return _log_path


def _log_session_header(app_name: str) -> None:
    """Everything needed to reproduce, written before anything can go wrong."""
    from app.version import build_string

    LOGGER.info("=" * 72)
    LOGGER.info("%s %s", app_name, build_string())
    LOGGER.info("python %s", sys.version.replace("\n", " "))
    LOGGER.info("platform %s", platform.platform())
    for package in ("numpy", "scipy", "matplotlib", "PySide6", "moderngl", "cadquery"):
        try:
            from importlib.metadata import version

            LOGGER.info("  %s %s", package, version(package))
        except Exception:      # noqa: BLE001 - absent is itself worth knowing
            LOGGER.info("  %s not installed", package)


def _install_qt_handler() -> None:
    """Route Qt's own diagnostics into the same file.

    Qt warns before it fails -- a missing OpenGL context, a layout that cannot
    be satisfied, a signal connected to a dead object. Those lines are usually
    the first evidence of what went wrong and they were going nowhere.
    """
    try:
        from PySide6.QtCore import QtMsgType, qInstallMessageHandler
    except Exception:          # noqa: BLE001
        return

    levels = {
        QtMsgType.QtDebugMsg: logging.DEBUG,
        QtMsgType.QtInfoMsg: logging.INFO,
        QtMsgType.QtWarningMsg: logging.WARNING,
        QtMsgType.QtCriticalMsg: logging.ERROR,
        QtMsgType.QtFatalMsg: logging.CRITICAL,
    }

    def handler(mode, context, message: str) -> None:
        logging.getLogger("mudline.qt").log(
            levels.get(mode, logging.INFO), "%s", message
        )

    qInstallMessageHandler(handler)


def _excepthook(exc_type, exc_value, exc_tb) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_tb)
        return
    report(exc_type, exc_value, exc_tb)


def _thread_excepthook(args) -> None:
    if issubclass(args.exc_type, SystemExit):
        return
    report(args.exc_type, args.exc_value, args.exc_traceback,
           where=f"thread {args.thread.name if args.thread else '?'}")


def report(exc_type, exc_value, exc_tb, where: str = "") -> None:
    """Log an unhandled exception, and tell the user once per signature."""
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    LOGGER.error("Unhandled exception%s\n%s", f" on {where}" if where else "", text)

    # Also to stderr, for whoever is running from a terminal.
    sys.stderr.write(text)

    signature = f"{exc_type.__name__}:{_origin(exc_tb)}"
    if signature in _reported:
        return
    _reported.add(signature)
    _show(exc_type, exc_value)


def _origin(exc_tb) -> str:
    """The deepest frame in our own code -- what distinguishes one bug."""
    frames = traceback.extract_tb(exc_tb)
    if not frames:
        return "?"
    last = frames[-1]
    return f"{Path(last.filename).name}:{last.lineno}"


def _show(exc_type, exc_value) -> None:
    """A dialog, if there is a GUI up and someone to read it."""
    try:
        from PySide6.QtGui import QGuiApplication
        from PySide6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is None:
            return
        if QGuiApplication.platformName() in ("offscreen", "minimal"):
            return

        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Unexpected error")
        box.setText(
            "Something went wrong that the application did not expect.\n\n"
            f"{exc_type.__name__}: {exc_value}"
        )
        box.setInformativeText(
            "Your work has not been closed, but this part of the tool may now "
            "be in an unreliable state -- save under a new name if you carry "
            "on.\n\nThe full details are in the log:\n"
            f"{_log_path}\n\nSend that file with a bug report."
        )
        open_button = box.addButton("Open log folder", QMessageBox.ActionRole)
        box.addButton("Continue", QMessageBox.AcceptRole)
        box.exec()
        if box.clickedButton() is open_button:
            _reveal(log_directory())
    except Exception:          # noqa: BLE001 - reporting must never re-raise
        pass


def _reveal(directory: Path) -> None:
    try:
        if sys.platform == "win32":
            os.startfile(directory)          # noqa: S606
        else:
            import subprocess

            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.Popen([opener, str(directory)])
    except Exception:          # noqa: BLE001
        pass
