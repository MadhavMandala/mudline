from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import QMainWindow, QWidget

from massprops.gui.massprops_widget import MassPropsWidget
from massprops.model.models import Component


class MainWindow(QMainWindow):
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        vehicle_handoff_path: Path | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("MassProp")
        self.resize(1400, 900)

        self._widget = MassPropsWidget(self, vehicle_handoff_path=vehicle_handoff_path)
        self.setCentralWidget(self._widget)

    def set_root(self, root: Component) -> None:
        self._widget.set_root(root)

    def _load_step(self, path: Path, preloaded_root: Component | None = None) -> None:
        self._widget._load_step(path, preloaded_root)
