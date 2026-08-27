"""Entry point: ``python -m app``.

    python -m app                        # the basic test rocket
    python -m app --boattail             # a vehicle with a bulge and boattail
    python -m app vehicles/mine.json     # a saved parametric vehicle
    python -m app --version              # what build this is
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app", description=__doc__)
    parser.add_argument("vehicle", nargs="?", help="Parametric vehicle JSON to open")
    parser.add_argument("--boattail", action="store_true",
                        help="Start with the boattail demonstrator")
    parser.add_argument("--empty", action="store_true",
                        help="Start with an empty vehicle")
    parser.add_argument("--version", action="store_true",
                        help="Print the version and revision, and exit")
    args = parser.parse_args(argv)

    from app.version import build_string

    if args.version:
        print(f"Mudline {build_string()}")
        return 0

    # Before anything else that can fail. A traceback from the lines below --
    # a vehicle file that will not parse, a Qt platform that will not start --
    # is exactly the kind that used to vanish into an unattached stderr.
    from app import diagnostics

    log = diagnostics.install()

    # The surface format has to be set before the QApplication is constructed,
    # or the widget gets whatever context the platform defaults to.
    from app.viewport import configure_surface_format

    configure_surface_format()

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Mudline")
    # Named so QSettings has somewhere of its own to write: without both of
    # these Qt files preferences under a generic key and they are shared with
    # whatever else forgot to set them.
    app.setOrganizationName("Mudline")
    app.setApplicationVersion(build_string())

    from app import theme

    theme.apply(app)

    from app.mainwindow import MainWindow
    from parametric.model import VehicleModel
    from parametric.standard import basic_rocket, boattailed_rocket, empty_vehicle

    try:
        if args.vehicle:
            model = VehicleModel.load(Path(args.vehicle))
        elif args.boattail:
            model = boattailed_rocket()
        elif args.empty:
            model = empty_vehicle()
        else:
            model = basic_rocket()
    except Exception as exc:      # noqa: BLE001
        # A bad file on the command line should say so plainly rather than
        # open an error dialog over a window that never appeared.
        diagnostics.LOGGER.exception("Could not open %s", args.vehicle)
        print(f"Could not open {args.vehicle}: {exc}\nSee {log}", file=sys.stderr)
        return 2

    window = MainWindow(model)
    window.show()

    diagnostics.LOGGER.info("Window shown; logging to %s", log)

    # Windows will not let a background process take the foreground; briefly
    # topmost is the reliable way to land in front.
    try:
        import ctypes

        hwnd = int(window.winId())
        user32 = ctypes.windll.user32
        flags = 0x0002 | 0x0001 | 0x0040
        user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, flags)
        user32.SetWindowPos(hwnd, -2, 0, 0, 0, 0, flags)
    except Exception:
        pass

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
