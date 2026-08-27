"""The results panel: run history, plots, and a two-run comparison.

Selecting one run shows what it produced and what it was run with. Selecting
two of the same kind shows the difference between them, which is the question a
design tool exists to answer -- did that change help, and by how much.

Runs whose fingerprint no longer matches the model are marked. A stale result
is still worth keeping (it is the "before" half of every comparison) but it must
never be presented as though it describes the vehicle on screen.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.results import Result, ResultStore

PLOT_COLORS = [QColor("#ff8d2e"), QColor("#4da3ff")]


class SeriesPlot(QWidget):
    """Line plot of one or two series, drawn directly.

    Two series so a comparison can be overlaid rather than described. A charting
    dependency would buy interactivity this panel does not need and a build step
    it definitely does not.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._series: list[tuple] = []
        self._labels: list[str] = []
        self.setMinimumHeight(180)

    def set_series(self, series: list[tuple], labels: list[str]) -> None:
        self._series = [s for s in series if s is not None]
        self._labels = labels
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor("#20262e"))

        if not self._series:
            painter.setPen(QColor("#6f7a88"))
            painter.drawText(self.rect(), Qt.AlignCenter, "no series")
            painter.end()
            return

        xs = np.concatenate([np.asarray(s[0], dtype=float) for s in self._series])
        ys = np.concatenate([np.asarray(s[1], dtype=float) for s in self._series])
        finite = np.isfinite(xs) & np.isfinite(ys)
        if not np.any(finite):
            painter.end()
            return
        x_min, x_max = float(xs[finite].min()), float(xs[finite].max())
        y_min, y_max = float(ys[finite].min()), float(ys[finite].max())
        if x_max - x_min < 1e-12:
            x_max = x_min + 1.0
        if y_max - y_min < 1e-12:
            y_max = y_min + 1.0
        # A little headroom so the curve does not touch the frame.
        pad = 0.06 * (y_max - y_min)
        y_min, y_max = y_min - pad, y_max + pad

        left, bottom, top, right = 62, 26, 12, 12
        width = max(self.width() - left - right, 10)
        height = max(self.height() - bottom - top, 10)

        def to_px(x: float, y: float) -> QPointF:
            return QPointF(
                left + width * (x - x_min) / (x_max - x_min),
                top + height * (1.0 - (y - y_min) / (y_max - y_min)),
            )

        painter.setPen(QPen(QColor("#39424e"), 1))
        painter.setFont(QFont("Segoe UI", 7))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + height * (1.0 - fraction)
            painter.setPen(QPen(QColor("#39424e"), 1))
            painter.drawLine(left, int(y), left + width, int(y))
            painter.setPen(QColor("#8a94a2"))
            painter.drawText(4, int(y) + 4, f"{y_min + fraction * (y_max - y_min):9.4g}")

        # A series that changes sign -- vertical velocity through apogee --
        # is read against zero, and the gridlines land wherever the range
        # puts them. Draw the axis so the crossing is a line, not a guess.
        if y_min < 0.0 < y_max:
            zero = to_px(x_min, 0.0).y()
            painter.setPen(QPen(QColor("#6f7a88"), 1, Qt.DashLine))
            painter.drawLine(left, int(zero), left + width, int(zero))

        painter.setPen(QColor("#8a94a2"))
        painter.drawText(left, self.height() - 6, f"{x_min:.4g}")
        painter.drawText(
            left + width - 52, self.height() - 6, f"{x_max:.4g}"
        )
        first = self._series[0]
        if len(first) >= 4:
            painter.drawText(
                left + width // 2 - 30, self.height() - 6, str(first[2])
            )

        for index, series in enumerate(self._series):
            x = np.asarray(series[0], dtype=float)
            y = np.asarray(series[1], dtype=float)
            good = np.isfinite(x) & np.isfinite(y)
            if good.sum() < 2:
                continue
            colour = PLOT_COLORS[index % len(PLOT_COLORS)]
            path = QPainterPath(to_px(float(x[good][0]), float(y[good][0])))
            for px, py in zip(x[good][1:], y[good][1:]):
                path.lineTo(to_px(float(px), float(py)))
            painter.setPen(QPen(colour, 2))
            painter.drawPath(path)

            if index < len(self._labels):
                painter.setPen(colour)
                painter.drawText(left + 8, top + 14 + index * 14, self._labels[index])
        painter.end()


class ResultsPanel(QWidget):
    """Run history with details, plots and comparison."""

    show_trajectory = Signal(object)
    #: Something worth a line in the status bar -- what an export wrote.
    message = Signal(str)

    def __init__(self, store: ResultStore, parent=None):
        super().__init__(parent)
        self._store = store
        self._fingerprint = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Results")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        header.addWidget(title)
        header.addStretch(1)

        self._hint = QLabel("Ctrl-click a second run of the same kind to compare")
        self._hint.setStyleSheet("color:#7c8794; font-size:11px;")
        header.addWidget(self._hint)

        self._show_button = QPushButton("Show trajectory")
        self._show_button.setEnabled(False)
        self._show_button.clicked.connect(self._on_show)
        header.addWidget(self._show_button)

        self._export_button = QPushButton("Export…")
        self._export_button.setEnabled(False)
        self._export_button.setToolTip(
            "Write the selected run to CSV, with a PNG of its plot beside it. "
            "A flight run writes its full time history."
        )
        self._export_button.clicked.connect(self._on_export)
        header.addWidget(self._export_button)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Horizontal)

        self.runs = QTreeWidget()
        self.runs.setHeaderLabels(["Run", "Time", "State"])
        self.runs.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.runs.setRootIsDecorated(False)
        self.runs.setAlternatingRowColors(True)
        self.runs.setMaximumWidth(300)
        self.runs.setStyleSheet("font-size:11px;")
        self.runs.itemSelectionChanged.connect(self._refresh_detail)
        head = self.runs.header()
        head.setSectionResizeMode(0, QHeaderView.Stretch)
        head.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        splitter.addWidget(self.runs)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self.detail = QTreeWidget()
        self.detail.setHeaderLabels(["Quantity", "Value", "Compared", "Change"])
        self.detail.setRootIsDecorated(True)
        self.detail.setAlternatingRowColors(True)
        self.detail.setStyleSheet("font-family:Consolas,monospace; font-size:11px;")
        detail_head = self.detail.header()
        detail_head.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in (1, 2, 3):
            detail_head.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        right_layout.addWidget(self.detail, 1)

        plot_row = QHBoxLayout()
        plot_row.addWidget(QLabel("Plot"))
        self.series_combo = QComboBox()
        self.series_combo.currentIndexChanged.connect(self._refresh_plot)
        plot_row.addWidget(self.series_combo, 1)
        right_layout.addLayout(plot_row)

        self.plot = SeriesPlot()
        right_layout.addWidget(self.plot)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 720])
        layout.addWidget(splitter, 1)

    # ------------------------------------------------------------------

    def set_fingerprint(self, fingerprint: str) -> None:
        """Tell the panel which model version is current, so runs can be marked."""
        self._fingerprint = fingerprint
        self.refresh()

    def refresh(self) -> None:
        selected = {self._result_of(item).index
                    for item in self.runs.selectedItems()
                    if self._result_of(item) is not None}
        self.runs.clear()
        self._items: dict[int, Result] = {}

        for result in self._store.newest_first():
            state = "current" if result.is_current(self._fingerprint) else "stale"
            item = QTreeWidgetItem([result.title, result.clock, state])
            if state == "stale":
                for column in range(3):
                    item.setForeground(column, QColor("#8a94a2"))
            self._items[id(item)] = result
            self.runs.addTopLevelItem(item)
            if result.index in selected:
                item.setSelected(True)

        if not selected and self.runs.topLevelItemCount():
            self.runs.topLevelItem(0).setSelected(True)
        self._refresh_detail()

    def _result_of(self, item) -> Result | None:
        return self._items.get(id(item)) if item is not None else None

    def _selection(self) -> list[Result]:
        results = [
            self._result_of(item) for item in self.runs.selectedItems()
        ]
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------

    def _refresh_detail(self) -> None:
        self.detail.clear()
        selection = self._selection()
        # A restored run has no trajectory to show; the button says so by
        # staying grey rather than doing nothing when pressed.
        self._show_button.setEnabled(
            len(selection) == 1 and selection[0].kind == "flight"
            and selection[0].payload.get("result") is not None
        )
        self._export_button.setEnabled(len(selection) == 1)

        if not selection:
            self._refresh_series_list([])
            return

        primary = selection[0]
        other = None
        if len(selection) >= 2:
            candidates = [r for r in selection[1:] if r.kind == primary.kind]
            other = candidates[0] if candidates else None
            if other is not None and other.index > primary.index:
                primary, other = other, primary

        header = QTreeWidgetItem([
            primary.title, "",
            other.title if other is not None else "", "",
        ])
        font = header.font(0)
        font.setBold(True)
        for column in range(4):
            header.setFont(column, font)
        self.detail.addTopLevelItem(header)

        results_node = QTreeWidgetItem(["results", "", "", ""])
        self.detail.addTopLevelItem(results_node)
        for metric in primary.metrics:
            compared = other.metric(metric.label) if other is not None else None
            row = QTreeWidgetItem([
                metric.label,
                metric.format(),
                # In the primary's unit, so a pair straddling the feet
                # threshold does not read as feet beside inches.
                compared.format_like(metric) if compared is not None else "",
                metric.delta_text(compared) if compared is not None else "",
            ])
            for column in (1, 2, 3):
                row.setTextAlignment(column, Qt.AlignRight | Qt.AlignVCenter)
            if compared is not None:
                row.setForeground(3, self._delta_colour(metric, compared))
            results_node.addChild(row)

        settings_node = QTreeWidgetItem(["settings", "", "", ""])
        self.detail.addTopLevelItem(settings_node)
        for key, value in primary.settings.items():
            other_value = other.settings.get(key, "") if other is not None else ""
            row = QTreeWidgetItem([key, value, other_value, ""])
            if other is not None and other_value and other_value != value:
                row.setForeground(2, QColor("#b4520f"))
            settings_node.addChild(row)

        self._add_build_row(settings_node, primary, other)

        self.detail.expandAll()
        self._refresh_series_list(selection)

    @staticmethod
    def _add_build_row(parent, primary, other) -> None:
        """Name the build, and say so loudly when a comparison straddles two.

        Comparing a run made by one build against a run made by another is the
        one comparison where a difference in the numbers may be nothing to do
        with the design. It is also invisible unless something says it, and
        restored runs from a colleague's file are exactly where it happens.
        """
        def describe(result) -> str:
            build = getattr(result, "build", None) or {}
            version = build.get("app_version", "")
            revision = build.get("app_revision", "")
            if not version and not revision:
                return "unknown"
            return f"{version} ({revision})" if revision else version

        mine = describe(primary)
        theirs = describe(other) if other is not None else ""
        row = QTreeWidgetItem(["built by", mine, theirs, ""])
        if other is not None and theirs and theirs != mine:
            for column in (1, 2):
                row.setForeground(column, QColor("#b4520f"))
            row.setText(3, "different build")
            row.setForeground(3, QColor("#b4520f"))
        parent.addChild(row)

    @staticmethod
    def _delta_colour(metric, compared) -> QColor:
        """Green when a change moved a quantity the way you wanted, else grey.

        Only quantities with a declared direction are coloured. Apogee going up
        is good; max-Q going up is not obviously anything, so it stays neutral
        rather than implying a judgement the tool has not earned.
        """
        if metric.higher_is_better is None:
            return QColor("#5b6675")
        difference = metric.value - compared.value
        if abs(difference) < 1e-12:
            return QColor("#5b6675")
        improved = (difference > 0) == metric.higher_is_better
        return QColor("#2f7d32") if improved else QColor("#b4520f")

    # ------------------------------------------------------------------

    def _refresh_series_list(self, selection: list[Result]) -> None:
        previous = self.series_combo.currentText()
        self.series_combo.blockSignals(True)
        self.series_combo.clear()
        if selection:
            for name in selection[0].series:
                self.series_combo.addItem(name)
            index = self.series_combo.findText(previous)
            if index >= 0:
                self.series_combo.setCurrentIndex(index)
        self.series_combo.blockSignals(False)
        self._refresh_plot()

    def _refresh_plot(self) -> None:
        name = self.series_combo.currentText()
        selection = self._selection()
        if not name or not selection:
            self.plot.set_series([], [])
            return
        series, labels = [], []
        for result in selection[:2]:
            entry = result.series.get(name)
            if entry is not None:
                series.append(entry)
                labels.append(result.title)
        self.plot.set_series(series, labels)

    def _on_show(self) -> None:
        selection = self._selection()
        if len(selection) == 1 and selection[0].kind == "flight":
            self.show_trajectory.emit(selection[0].payload.get("result"))

    # ------------------------------------------------------------------

    def _on_export(self) -> None:
        selection = self._selection()
        if len(selection) != 1:
            return
        from PySide6.QtWidgets import QFileDialog

        result = selection[0]
        stem = "".join(c if c.isalnum() else "_" for c in result.title).strip("_")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export run", f"{stem}.csv", "CSV (*.csv)"
        )
        if not path:
            return
        written = self.export_result(result, path)
        self.message.emit("Wrote " + ", ".join(p.name for p in written))

    def export_result(self, result: Result, path) -> list[Path]:
        """Write a run to ``path`` (a .csv) with a PNG beside it.

        A flight run with its trajectory still in hand writes the full time
        history through ``trajectory.analysis.export`` -- every state, the
        derived quantities and the flight log's columns -- and the four-panel
        summary figure. Anything else, including a run restored from a
        project file, writes its plotted series and a capture of the panel's
        own plot. Nothing here needs a window: the figure is drawn headless.
        """
        path = Path(path)
        stem = path.with_suffix("")
        payload = result.payload or {}
        flight = payload.get("result")
        written: list[Path] = []

        if result.kind == "flight" and flight is not None:
            from trajectory.analysis import export as exporter

            written.append(exporter.write_trajectory_csv(
                flight, stem.with_suffix(".csv"), log=payload.get("log"),
            ))
            written.append(exporter.plot_trajectory(
                flight, stem.with_suffix(".png"), title=result.title,
            ))
            return written

        written.append(_write_series_csv(result, stem.with_suffix(".csv")))
        png = stem.with_suffix(".png")
        if self.plot.grab().save(str(png)):
            written.append(png)
        return written


def _write_series_csv(result: Result, path: Path) -> Path:
    """Every plotted series side by side, padded to the longest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[tuple[str, np.ndarray]] = []
    for name, entry in result.series.items():
        x, y = np.asarray(entry[0], dtype=float), np.asarray(entry[1], dtype=float)
        labels = list(entry[2:4]) + ["x", "y"]
        columns.append((f"{name} [{labels[0]}]", x))
        columns.append((f"{name} [{labels[1]}]", y))
    length = max((len(c[1]) for c in columns), default=0)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name for name, _ in columns])
        for i in range(length):
            writer.writerow([
                f"{values[i]:.6g}" if i < len(values) and np.isfinite(values[i]) else ""
                for _, values in columns
            ])
    return path
