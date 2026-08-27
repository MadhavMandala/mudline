"""The 3-D viewport, on a native QOpenGLWidget.

This replaces the previous arrangement, which created a borderless SDL window
through pygame and reparented it into Qt by raw HWND. That approach was the
root of a surprising amount of trouble: Qt could not lay out around a native
child window reliably, the surface could not be captured or composited, input
had to be tracked twice (once through SDL, once through Qt), and the
mass-properties and propulsion editors had to run in *separate processes*
because moderngl and VTK fought over the GL context.

A QOpenGLWidget is an ordinary widget. It lays out, composites, takes a
QPainter overlay for text, and shares the application's context, so all of the
above goes away.

moderngl still does the drawing. Qt owns the context and makes it current; the
one subtlety is that Qt renders into its own framebuffer object rather than the
default one, so the context has to be pointed at it every frame.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from app.scene import (
    GridMesh,
    OrbitCamera,
    RenderMesh,
    build_axis_triad,
    build_grid,
    display_transform,
    tessellate_solid,
)
from app.shaders import LINE_FRAG, LINE_VERT, SOLID_FRAG, SOLID_VERT
from app.theme import VIEWPORT_BG

#: Same colour the panels are painted, so the viewport reads as part of the
#: window rather than a hole cut in it. Defined in ``app.theme`` now; the value
#: is unchanged.
BACKGROUND = VIEWPORT_BG


def configure_surface_format() -> None:
    """Ask for a 3.3 core context. Must run before the QApplication exists."""
    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)                     # MSAA; edges on a slender body alias badly
    QSurfaceFormat.setDefaultFormat(fmt)


class Viewport(QOpenGLWidget):
    """Renders the vehicle, a scale grid, and optionally a trajectory."""

    selection_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        self.camera = OrbitCamera()
        #: Half-section view: throw away everything on one side of the axis.
        self.cutaway = False
        self.light_dir = np.array([0.42, 0.78, 0.46])
        self.light_dir /= np.linalg.norm(self.light_dir)

        self._ctx = None
        #: (headline, advice) once GL has failed; None while it is fine.
        self._gl_error: tuple[str, str] | None = None
        self._solid_prog = None
        self._line_prog = None
        self._meshes: dict[str, RenderMesh] = {}
        self._mesh_buffers: dict[str, tuple] = {}
        self._grid: GridMesh | None = None
        self._grid_buffer = None
        self._axis_buffer = None
        self._trajectory_buffer = None
        self._trajectory_points = 0

        self._model_length = 1.0
        self._display = np.eye(4, dtype=np.float32)
        self.selected: str | None = None
        self.show_grid = True
        self.show_axes = True
        self.show_hud = True

        self._last_mouse: QPoint | None = None
        self._button = None
        self._status = ""

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        """Bring up the GL context, or record why not and draw an apology.

        Everything here can fail on a machine that is otherwise fine: moderngl
        can be missing, and a 3.3 core context is not available over RDP, in
        many VMs, or behind a stale driver. It used to fail into a stderr
        nobody was reading, leaving a black rectangle and no explanation --
        and the rest of the tool works perfectly without a 3D view, so
        stopping the application would be the wrong answer too.
        """
        try:
            import moderngl as mgl

            self._ctx = mgl.create_context()
            self._solid_prog = self._ctx.program(
                vertex_shader=SOLID_VERT, fragment_shader=SOLID_FRAG
            )
            self._line_prog = self._ctx.program(
                vertex_shader=LINE_VERT, fragment_shader=LINE_FRAG
            )
            self._upload_pending()
        except ImportError as exc:
            self._fail_gl(
                "The 3D viewport needs the moderngl package, which is not "
                "installed.",
                'Reinstall with:  pip install -e ".[cad,dev]" -c constraints.txt',
                exc,
            )
        except Exception as exc:      # noqa: BLE001 - drivers fail in every way
            self._fail_gl(
                "This machine could not provide the OpenGL 3.3 context the "
                "viewport needs.",
                "Common over remote desktop, in a virtual machine, or with an "
                "out-of-date display driver. Everything else -- mass, "
                "aerodynamics, trajectory, export -- still works.",
                exc,
            )

    def _fail_gl(self, headline: str, advice: str, exc: Exception) -> None:
        """Remember the failure so :meth:`paintGL` can say it on screen."""
        self._ctx = None
        self._gl_error = (headline, advice)
        try:
            from app.diagnostics import LOGGER

            LOGGER.error("Viewport unavailable: %s (%s: %s)",
                         headline, type(exc).__name__, exc)
        except Exception:          # noqa: BLE001
            pass

    def resizeGL(self, width: int, height: int) -> None:
        self.camera.aspect = max(width / max(height, 1), 1e-6)

    def _paint_gl_failure(self) -> None:
        """Say what went wrong, in the space where the vehicle should be."""
        message = getattr(self, "_gl_error", None)
        if message is None:
            return
        headline, advice = message

        painter = QPainter(self)
        try:
            painter.fillRect(self.rect(), QColor.fromRgbF(*BACKGROUND))
            box = self.rect().adjusted(28, 28, -28, -28)

            font = painter.font()
            font.setPointSize(max(font.pointSize() + 1, 10))
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(QColor("#c8cdd6"))
            painter.drawText(
                box, Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextWordWrap,
                f"3D view unavailable\n\n{headline}\n\n{advice}",
            )
        finally:
            painter.end()

    def paintGL(self) -> None:
        if self._ctx is None:
            self._paint_gl_failure()
            return
        import moderngl as mgl

        # Qt draws into its own FBO, not the default framebuffer. Without this
        # the scene renders somewhere invisible.
        screen = self._ctx.detect_framebuffer(self.defaultFramebufferObject())
        screen.use()

        ratio = self.devicePixelRatioF()
        self._ctx.viewport = (0, 0, int(self.width() * ratio), int(self.height() * ratio))
        self._ctx.enable(mgl.DEPTH_TEST)
        self._ctx.clear(*BACKGROUND, 1.0)

        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix()
        eye = self.camera.eye().astype(np.float32)

        self._draw_lines(view, proj, eye)
        self._draw_solids(view, proj, eye)

        if self.show_hud:
            self._draw_hud()

    # ------------------------------------------------------------------
    # Content
    # ------------------------------------------------------------------

    def set_solids(self, solids: dict, model_length_m: float,
                   colors: dict[str, tuple] | None = None,
                   sheens: dict[str, float] | None = None) -> None:
        """Replace the displayed geometry with tessellated OCC solids."""
        colors = colors or {}
        # Kept beside the meshes rather than inside them: a sheen is a shading
        # parameter, and RenderMesh is geometry that the tessellator produces.
        self._sheens = dict(sheens or {})
        meshes: dict[str, RenderMesh] = {}
        tolerance = max(0.4, model_length_m * 1000.0 * 0.0012)

        for name, result in solids.items():
            solid = getattr(result, "solid", result)
            mesh = tessellate_solid(
                name, solid, tolerance_mm=tolerance,
                color=colors.get(name, (0.72, 0.76, 0.82)),
            )
            if mesh is not None:
                meshes[name] = mesh

        self._meshes = meshes
        self._model_length = max(model_length_m, 1e-6)
        self._display = display_transform(self._model_length)
        self._with_context(self._upload_meshes)
        self.update()

    def set_trajectory(self, positions_m: np.ndarray | None) -> None:
        self._trajectory_points = 0 if positions_m is None else len(positions_m)
        self._pending_trajectory = (
            None if positions_m is None else np.asarray(positions_m, dtype=np.float32)
        )
        self._with_context(self._upload_trajectory)
        self.update()

    def set_selection(self, name: str | None) -> None:
        self.selected = name
        self.update()

    def set_status(self, text: str) -> None:
        self._status = text
        self.update()

    def frame_all(self) -> None:
        """Fit everything in view -- geometry and trajectory both."""
        corners: list[np.ndarray] = []

        if self._meshes:
            lows = np.array([m.bounds_low for m in self._meshes.values()])
            highs = np.array([m.bounds_high for m in self._meshes.values()])
            corners.append(self._to_display(lows.min(axis=0)))
            corners.append(self._to_display(highs.max(axis=0)))

        # A trajectory is already in world coordinates and dwarfs the vehicle;
        # framing on the vehicle alone would leave it entirely off screen.
        points = getattr(self, "_pending_trajectory", None)
        if points is not None and len(points):
            corners.append(points.min(axis=0))
            corners.append(points.max(axis=0))

        if not corners:
            self.camera.frame(np.array([-0.5, 0.0, -0.5]), np.array([0.5, 1.0, 0.5]))
        else:
            stacked = np.array(corners)
            self.camera.frame(stacked.min(axis=0), stacked.max(axis=0))
        self.update()

    def _to_display(self, point: np.ndarray) -> np.ndarray:
        homogeneous = np.append(np.asarray(point, dtype=np.float32), 1.0)
        return (self._display @ homogeneous)[:3]

    # ------------------------------------------------------------------
    # Buffers
    # ------------------------------------------------------------------

    def _with_context(self, action) -> None:
        """Run something that touches GL objects, with the context made current.

        Qt makes the context current only inside initializeGL, paintGL and
        resizeGL. Anything called from application code -- loading a model,
        pushing a trajectory -- runs with *no* current context, and creating a
        buffer there silently produces an object that belongs to nothing and
        renders as nothing. No GL error is raised; the geometry simply never
        appears.
        """
        if self._ctx is None:
            return          # not initialised yet; initializeGL will do it
        self.makeCurrent()
        try:
            action()
        finally:
            self.doneCurrent()

    def _upload_pending(self) -> None:
        self._upload_meshes()
        self._upload_trajectory()

    def _upload_meshes(self) -> None:
        if self._ctx is None:
            return
        self._release_mesh_buffers()
        for name, mesh in self._meshes.items():
            vbo = self._ctx.buffer(mesh.vertices.tobytes())
            vao = self._ctx.vertex_array(
                self._solid_prog, [(vbo, "3f 3f", "in_position", "in_normal")]
            )
            self._mesh_buffers[name] = (vao, vbo)

    def _release_mesh_buffers(self) -> None:
        for vao, vbo in self._mesh_buffers.values():
            for obj in (vao, vbo):
                try:
                    obj.release()
                except Exception:
                    pass
        self._mesh_buffers.clear()

    def _upload_trajectory(self) -> None:
        points = getattr(self, "_pending_trajectory", None)
        if self._trajectory_buffer is not None:
            try:
                self._trajectory_buffer[0].release()
                self._trajectory_buffer[1].release()
            except Exception:
                pass
            self._trajectory_buffer = None
        if points is None or len(points) < 2 or self._ctx is None:
            return
        colors = np.tile(np.array([[1.0, 0.55, 0.18]], dtype=np.float32), (len(points), 1))
        data = np.hstack([points, colors]).astype(np.float32)
        vbo = self._ctx.buffer(data.tobytes())
        vao = self._ctx.vertex_array(
            self._line_prog, [(vbo, "3f 3f", "in_position", "in_color")]
        )
        self._trajectory_buffer = (vao, vbo)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_lines(self, view, proj, eye) -> None:
        import moderngl as mgl

        self._line_prog["view"].write(view.T.tobytes())
        self._line_prog["projection"].write(proj.T.tobytes())
        self._line_prog["view_pos"].value = tuple(float(v) for v in eye)
        self._line_prog["background"].value = BACKGROUND

        if self.show_grid:
            self._grid = build_grid(self.camera.distance, self.camera.target)
            self._line_prog["fade_start"].value = float(self.camera.distance * 0.6)
            self._line_prog["fade_end"].value = float(self._grid.extent * 1.1)
            data = np.hstack([self._grid.positions, self._grid.colors]).astype(np.float32)
            vbo = self._ctx.buffer(data.tobytes())
            vao = self._ctx.vertex_array(
                self._line_prog, [(vbo, "3f 3f", "in_position", "in_color")]
            )
            vao.render(mgl.LINES)
            vao.release()
            vbo.release()

        if self.show_axes:
            length = self.camera.distance * 0.12
            positions, colors = build_axis_triad(length)
            self._line_prog["fade_start"].value = 1e9
            self._line_prog["fade_end"].value = 1e9 + 1.0
            data = np.hstack([positions, colors]).astype(np.float32)
            vbo = self._ctx.buffer(data.tobytes())
            vao = self._ctx.vertex_array(
                self._line_prog, [(vbo, "3f 3f", "in_position", "in_color")]
            )
            self._ctx.line_width = 2.0
            vao.render(mgl.LINES)
            vao.release()
            vbo.release()

        if self._trajectory_buffer is not None:
            self._line_prog["fade_start"].value = 1e9
            self._line_prog["fade_end"].value = 1e9 + 1.0
            self._trajectory_buffer[0].render(mgl.LINE_STRIP)

    def _draw_solids(self, view, proj, eye) -> None:
        if not self._mesh_buffers:
            return
        self._solid_prog["view"].write(view.T.tobytes())
        self._solid_prog["projection"].write(proj.T.tobytes())
        self._solid_prog["view_pos"].value = tuple(float(v) for v in eye)
        self._solid_prog["light_dir"].value = tuple(float(v) for v in self.light_dir)
        self._solid_prog["model"].write(self._display.T.tobytes())
        self._solid_prog["alpha"].value = 1.0

        cut = getattr(self, "cutaway", False)
        self._solid_prog["cut_enabled"].value = 1.0 if cut else 0.0
        # Sliced along the vehicle's own axis so the cut face runs the length of
        # it. In display coordinates the axis is world Y and the radius lies in
        # XZ, so a plane through world X = 0 halves every part lengthwise.
        self._solid_prog["cut_plane"].value = (1.0, 0.0, 0.0, 0.0)

        sheens = getattr(self, "_sheens", {})
        for name, (vao, _) in self._mesh_buffers.items():
            mesh = self._meshes[name]
            self._solid_prog["base_color"].value = mesh.color
            self._solid_prog["sheen"].value = float(sheens.get(name, 0.4))
            self._solid_prog["selected"].value = 1.0 if self._is_selected(name) else 0.0
            vao.render()

    def _is_selected(self, name: str) -> bool:
        """Whether a solid belongs to the selected component.

        Matched on path boundaries, not a bare prefix. Solid keys are paths like
        ``vehicle/airframe`` and ``vehicle/airframe/fins#2``; a plain startswith
        means selecting the root highlights the entire vehicle, which is what
        the window does on load and reads as the app being broken.
        """
        if not self.selected:
            return False
        return (
            name == self.selected
            or name.startswith(f"{self.selected}/")
            or name.startswith(f"{self.selected}#")
        )

    def _draw_hud(self) -> None:
        """Text overlay: scale bar, axis labels, status.

        A QPainter over a QOpenGLWidget is the whole reason this is easy now.
        The SDL surface had no way to draw text at all, which is why the old
        viewer had no scale reference of any kind.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        painter.setPen(QColor(190, 200, 214))

        if self._grid is not None:
            spacing = self._grid.spacing
            label = (
                f"grid {spacing * 100:.0f} cm" if spacing < 1.0
                else (f"grid {spacing:.0f} m" if spacing < 1000.0
                      else f"grid {spacing / 1000.0:.0f} km")
            )
            painter.drawText(14, self.height() - 16, label)

        painter.setPen(QColor(150, 160, 175))
        painter.drawText(14, 22, f"{len(self._meshes)} parts")
        if self._status:
            painter.setPen(QColor(210, 220, 234))
            painter.drawText(14, 40, self._status)

        # Axis key, matching the triad colours.
        x = self.width() - 96
        for offset, (name, colour) in enumerate([
            ("X", QColor(217, 77, 77)),
            ("Y up", QColor(89, 199, 102)),
            ("Z", QColor(87, 140, 230)),
        ]):
            painter.setPen(colour)
            painter.drawText(x, 22 + offset * 16, name)

        painter.end()

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        self._last_mouse = event.position().toPoint()
        self._button = event.button()
        self.setFocus(Qt.MouseFocusReason)

    def mouseMoveEvent(self, event) -> None:
        if self._last_mouse is None or self._button is None:
            return
        point = event.position().toPoint()
        dx = point.x() - self._last_mouse.x()
        dy = point.y() - self._last_mouse.y()
        self._last_mouse = point

        if self._button == Qt.LeftButton:
            self.camera.orbit(dx, dy)
        elif self._button in (Qt.MiddleButton, Qt.RightButton):
            self.camera.pan(dx, dy)
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        self._last_mouse = None
        self._button = None

    def wheelEvent(self, event) -> None:
        self.camera.zoom(event.angleDelta().y() / 120.0)
        self.update()

    def keyPressEvent(self, event) -> None:
        key = event.key()
        if key == Qt.Key_F:
            self.frame_all()
        elif key == Qt.Key_G:
            self.show_grid = not self.show_grid
            self.update()
        elif key in (Qt.Key_1, Qt.Key_3, Qt.Key_7):
            # Standard orthographic-ish views, as in every CAD package.
            self.camera.azimuth, self.camera.elevation = {
                Qt.Key_1: (0.0, 0.0),
                Qt.Key_3: (np.pi / 2, 0.0),
                Qt.Key_7: (0.0, np.radians(89.0)),
            }[key]
            self.update()
        else:
            super().keyPressEvent(event)
