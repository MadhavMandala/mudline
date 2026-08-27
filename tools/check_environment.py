"""Check that this machine can actually run Mudline.

    python -m tools.check_environment

Run it straight after installing. Every failure it reports is one that would
otherwise turn up later as something that looks unrelated -- a black viewport,
a plot button that does nothing, geometry that will not rebuild -- with the
real cause on a stderr nobody is reading.

Why this exists as a separate thing from the test suite: the suite cannot see
these. It runs Qt under the offscreen platform, where ``initializeGL`` is
never called and the viewport's ``moderngl`` import never executes; and it is
run, on a developer machine, in an environment that was assembled months ago
and works. Both of the breakages this project has shipped to a clean install
were invisible for exactly those reasons. So the check imports what the app
imports, in the order the app imports it, and does the one operation -- render
a figure, build a GL context -- that fails when the versions are wrong.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Import name -> what stops working when it is missing.
REQUIRED = [
    ("numpy", "everything"),
    ("scipy", "the trajectory integrator"),
    ("matplotlib", "every plot and plot export"),
    ("PySide6", "the application itself"),
    ("moderngl", "the 3D viewport"),
]

#: The CAD extra. Optional in principle -- the tool refuses politely without
#: it -- but a design session without geometry is not what anyone installed.
CAD = [
    ("cadquery", "parametric geometry and STEP import/export"),
    ("gmsh", "meshed mass properties"),
    ("trimesh", "meshed mass properties"),
]


class Report:
    """Collects results so the run can show everything, not just the first."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def ok(self, what: str, detail: str = "") -> None:
        print(f"  ok       {what}{'  ' + detail if detail else ''}")

    def fail(self, what: str, detail: str) -> None:
        print(f"  FAILED   {what}\n             {detail}")
        self.failures.append(what)

    def warn(self, what: str, detail: str) -> None:
        print(f"  warning  {what}\n             {detail}")
        self.warnings.append(what)


def _version(module) -> str:
    for attr in ("__version__", "VERSION", "version"):
        value = getattr(module, attr, None)
        if isinstance(value, str):
            return value
    return ""


def check_python(report: Report) -> None:
    print("Python")
    if sys.version_info < (3, 12):
        report.fail(
            "python >= 3.12",
            f"this is {sys.version.split()[0]}; the project needs 3.12 or newer",
        )
    else:
        report.ok("python", sys.version.split()[0])


def check_imports(report: Report, group: str, modules: list[tuple[str, str]],
                  fatal: bool) -> None:
    print(group)
    for name, matters_for in modules:
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - any import failure counts
            complain = report.fail if fatal else report.warn
            complain(
                name,
                f"{type(exc).__name__}: {exc}\n"
                f"             without it: {matters_for}",
            )
        else:
            report.ok(name, _version(module))


def check_cadquery_vtk(report: Report) -> None:
    """cadquery's OCP can arrive without its VTK half, depending on order.

    ``import cadquery`` on its own succeeds; it is only once something else has
    touched the import machinery first that the proxy resolves to the build
    with no ``IVtkOCC``, and then geometry dies at the point of use rather than
    at import. Ask for the symbol directly.
    """
    print("cadquery's OCP build")
    try:
        from OCP.IVtkOCC import IVtkOCC_Shape  # noqa: F401
    except ImportError as exc:
        report.fail(
            "OCP.IVtkOCC",
            f"{exc}\n"
            "             cadquery installed without its VTK half. Reinstall\n"
            "             under the pins: pip install -e \".[cad,dev]\" -c constraints.txt",
        )
    except Exception as exc:  # noqa: BLE001
        report.warn("OCP.IVtkOCC", f"{type(exc).__name__}: {exc}")
    else:
        report.ok("OCP.IVtkOCC")


def check_plotting(report: Report) -> None:
    """Render and save a figure -- the operation that actually fails.

    matplotlib 3.11 against the ``six`` shim vtk installs raises
    ``'_SixMetaPathImporter' object has no attribute '_path'`` from deep in the
    import machinery, and only once something has put six on ``sys.meta_path``.
    Importing matplotlib proves nothing; drawing does.
    """
    print("plotting")
    try:
        try:
            import six  # noqa: F401  -- present via vtk; the trigger
        except ImportError:
            pass
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots()
        axes.plot([0.0, 1.0], [0.0, 1.0])
        axes.set_xlabel("check")
        axes.legend(["check"])
        import io

        buffer = io.BytesIO()
        figure.savefig(buffer, format="png")
        plt.close(figure)
    except Exception as exc:  # noqa: BLE001
        report.fail(
            "render a figure",
            f"{type(exc).__name__}: {exc}\n"
            "             plot export and the aero plots will not work.\n"
            "             Reinstall under the pins: "
            "pip install -e \".[cad,dev]\" -c constraints.txt",
        )
    else:
        report.ok("render a figure")


#: Asks the graphics driver for the context the viewport needs, and prints one
#: line. Run in its own process -- see ``check_opengl``.
_GL_PROBE = """
import sys
from PySide6.QtGui import QOffscreenSurface, QOpenGLContext, QSurfaceFormat
from PySide6.QtWidgets import QApplication

app = QApplication(sys.argv[:1])
fmt = QSurfaceFormat()
fmt.setVersion(3, 3)
fmt.setProfile(QSurfaceFormat.CoreProfile)

surface = QOffscreenSurface()
surface.setFormat(fmt)
surface.create()

context = QOpenGLContext()
context.setFormat(fmt)
if context.create() and context.makeCurrent(surface):
    actual = context.format()
    print(f"VERSION {actual.majorVersion()}.{actual.minorVersion()}")
    context.doneCurrent()
else:
    print("NOCONTEXT")
"""


def check_opengl(report: Report) -> None:
    """The viewport wants a 3.3 core context. Plenty of machines cannot.

    A warning rather than a failure: over RDP, in a VM, or on a thin GPU the
    rest of the tool -- mass, aero, trajectory, export -- is perfectly usable
    without a 3D view, and refusing to start would be worse than saying so.

    In its own process, because asking a graphics driver for a context is the
    one thing here that can take the interpreter down rather than raise: a
    mismatched or broken driver aborts, and on Windows that arrives as an
    access violation at exit with no traceback. The machines where that
    happens are precisely the ones this check is for, so the probe must not be
    able to kill the checker before it prints its verdict.
    """
    print("OpenGL")
    if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        report.warn("GL context", "skipped: running under the offscreen platform")
        return

    import subprocess

    try:
        finished = subprocess.run(
            [sys.executable, "-c", _GL_PROBE],
            capture_output=True, text=True, timeout=90, cwd=REPO,
        )
    except subprocess.TimeoutExpired:
        report.warn("GL context", "the driver did not answer within 90 s")
        return
    except Exception as exc:  # noqa: BLE001
        report.warn("GL context", f"could not run the probe: {exc}")
        return

    answer = next(
        (line for line in finished.stdout.splitlines()
         if line.startswith(("VERSION ", "NOCONTEXT"))),
        "",
    )
    unusable = (
        "could not create OpenGL 3.3 core. The 3D viewport will not\n"
        "             render; everything else works. Common over RDP,\n"
        "             in a VM, or with a stale display driver."
    )
    if answer.startswith("VERSION "):
        report.ok("GL context", answer.removeprefix("VERSION "))
    elif answer == "NOCONTEXT":
        report.warn("GL context", unusable)
    else:
        detail = (finished.stderr or "").strip().splitlines()
        report.warn(
            "GL context",
            f"the probe died (exit {finished.returncode}) instead of answering.\n"
            f"             {detail[-1] if detail else 'no output'}\n"
            f"             Treat as: {unusable}",
        )


def check_pins(report: Report) -> None:
    """Say where this environment has drifted from the validated set."""
    print("versions against constraints.txt")
    constraints = REPO / "constraints.txt"
    if not constraints.exists():
        report.warn("constraints.txt", "not found; skipping the comparison")
        return

    from importlib.metadata import PackageNotFoundError, version

    pinned: dict[str, str] = {}
    for line in constraints.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, _, want = line.partition("==")
        pinned[name.strip().lower().replace("_", "-")] = want.strip()

    drifted: list[str] = []
    for name, want in sorted(pinned.items()):
        try:
            have = version(name)
        except PackageNotFoundError:
            continue
        if have != want:
            drifted.append(f"{name}: {have} installed, {want} validated")

    if not drifted:
        report.ok("versions", f"all {len(pinned)} pins match")
    else:
        report.warn(
            "versions",
            "\n             ".join(
                [f"{len(drifted)} package(s) differ from the validated set:"]
                + drifted
                + ["the tool may still work; its published numbers were not "
                   "measured here"]
            ),
        )


def main() -> int:
    print(f"Mudline environment check\n{REPO}\n")
    report = Report()

    check_python(report)
    check_imports(report, "core", REQUIRED, fatal=True)
    check_imports(report, "cad extra", CAD, fatal=False)
    check_cadquery_vtk(report)
    check_plotting(report)
    check_opengl(report)
    check_pins(report)

    print()
    if report.failures:
        print(f"{len(report.failures)} problem(s) that will stop the tool working:")
        for name in report.failures:
            print(f"  - {name}")
        print(
            "\nThe install to reach for, from the repository root:\n"
            '    python -m pip install -e ".[cad,dev]" -c constraints.txt'
        )
        return 1

    if report.warnings:
        print(f"Usable. {len(report.warnings)} warning(s) above are worth reading.")
    else:
        print("Everything checks out.")
    return 0


if __name__ == "__main__":
    code = main()
    # Leave without running interpreter shutdown. This checker deliberately
    # imports Qt, vtk and OCP together, and a mismatched set of those -- the
    # thing it is here to find -- can abort in native teardown *after* the
    # verdict has been printed. On Windows that surfaces as an access
    # violation exit code, which would report a broken environment as a broken
    # checker. The report is already on stdout; flush it and go.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
