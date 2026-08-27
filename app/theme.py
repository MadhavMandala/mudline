"""One palette, shared by the Qt chrome and the OpenGL viewport.

Before this the application had no theme: two inline stylesheets on the status
bar and whatever the platform handed us, while the viewport carried its own
hardcoded background and accent. The two were never going to agree, and the
seam between the panels and the 3D view showed it.

The colours live here as hex strings and are converted for whichever consumer
needs them -- Qt wants ``#rrggbb``, OpenGL wants floats in 0..1. ``VIEWPORT_BG``
is the value the viewport already used, kept exactly, so this is a
consolidation rather than a re-colour.

The accent is the orange the viewport already draws its highlight in. Deriving
the interface accent from it rather than picking a new one is what makes the
selected row in the tree and the highlighted solid in the viewport read as the
same act.
"""

from __future__ import annotations

# ----------------------------------------------------------------------
# Tokens
# ----------------------------------------------------------------------

BG_DEEP = "#14171C"      # window behind everything
BG_VIEW = "#1B1E24"      # the 3D viewport -- the viewport's original value
BG_PANEL = "#21252D"     # tree, property editor, docks
BG_RAISED = "#282D36"    # menu bar, headers, input fields
BG_HOVER = "#2F3540"

BORDER = "#333945"
BORDER_SOFT = "#2A2F39"

TEXT = "#D6DBE3"
TEXT_DIM = "#8A94A3"
TEXT_FAINT = "#5B6675"   # the status bar's original pipeline colour

ACCENT = "#FF8C2E"       # the viewport highlight orange
ACCENT_DIM = "#C46A1F"
ACCENT_WASH = "#3A2C1E"  # accent at low weight, for selected rows

OK = "#4FB477"
WARN = "#E0A33E"
BAD = "#E0574A"

UI_FONT = "Segoe UI"
MONO_FONT = "Consolas"


def rgb01(value: str) -> tuple[float, float, float]:
    """``#rrggbb`` to a 0..1 triple, for OpenGL."""
    value = value.lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


#: What ``viewport.py`` clears to. Same numbers as before, one source now.
VIEWPORT_BG = rgb01(BG_VIEW)
VIEWPORT_ACCENT = rgb01(ACCENT)


# ----------------------------------------------------------------------
# Stylesheet
# ----------------------------------------------------------------------

def stylesheet() -> str:
    """The application stylesheet.

    Deliberately conservative about geometry: padding and borders only, no
    fixed sizes. A stylesheet that hardcodes widget dimensions looks correct
    at one font scale and breaks at every other, and this runs on displays
    from a laptop panel to a 4K monitor.
    """
    return f"""
    QWidget {{
        background: {BG_PANEL};
        color: {TEXT};
        font-family: "{UI_FONT}";
        font-size: 9pt;
    }}
    QMainWindow, QDialog {{ background: {BG_DEEP}; }}

    /* ---- menu bar ---- */
    QMenuBar {{
        background: {BG_RAISED};
        border-bottom: 1px solid {BORDER};
        padding: 2px 4px;
    }}
    QMenuBar::item {{
        background: transparent;
        padding: 5px 11px;
        border-radius: 3px;
    }}
    QMenuBar::item:selected {{ background: {BG_HOVER}; }}
    QMenuBar::item:pressed {{ background: {ACCENT_WASH}; color: {ACCENT}; }}

    QMenu {{
        background: {BG_RAISED};
        border: 1px solid {BORDER};
        padding: 4px;
    }}
    QMenu::item {{ padding: 6px 26px 6px 22px; border-radius: 3px; }}
    QMenu::item:selected {{ background: {ACCENT_WASH}; color: {ACCENT}; }}
    QMenu::item:disabled {{ color: {TEXT_FAINT}; }}
    QMenu::separator {{
        height: 1px;
        background: {BORDER_SOFT};
        margin: 5px 8px;
    }}

    /* ---- tree ---- */
    QTreeWidget, QTreeView, QListView, QTableView {{
        background: {BG_PANEL};
        alternate-background-color: {BG_DEEP};
        border: none;
        outline: none;
        selection-background-color: {ACCENT_WASH};
    }}
    QTreeWidget::item, QTreeView::item {{
        padding: 4px 2px;
        border: none;
    }}
    QTreeWidget::item:hover, QTreeView::item:hover {{ background: {BG_HOVER}; }}
    QTreeWidget::item:selected, QTreeView::item:selected {{
        background: {ACCENT_WASH};
        color: {ACCENT};
    }}
    QHeaderView::section {{
        background: {BG_RAISED};
        color: {TEXT_DIM};
        padding: 6px 8px;
        border: none;
        border-bottom: 1px solid {BORDER};
        border-right: 1px solid {BORDER_SOFT};
        font-weight: 600;
    }}

    /* ---- splitter: the seam between panels and viewport ---- */
    QSplitter::handle {{ background: {BORDER_SOFT}; }}
    QSplitter::handle:horizontal {{ width: 1px; }}
    QSplitter::handle:vertical {{ height: 1px; }}
    QSplitter::handle:hover {{ background: {ACCENT_DIM}; }}

    /* ---- inputs ---- */
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background: {BG_RAISED};
        border: 1px solid {BORDER};
        border-radius: 3px;
        padding: 4px 6px;
        selection-background-color: {ACCENT_DIM};
        selection-color: #FFFFFF;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
        border: 1px solid {ACCENT_DIM};
    }}
    QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
        background: {BG_PANEL};
        color: {TEXT_FAINT};
    }}
    QComboBox::drop-down {{ border: none; width: 18px; }}
    QComboBox QAbstractItemView {{
        background: {BG_RAISED};
        border: 1px solid {BORDER};
        selection-background-color: {ACCENT_WASH};
        selection-color: {ACCENT};
    }}

    /* ---- buttons ---- */
    QPushButton {{
        background: {BG_RAISED};
        border: 1px solid {BORDER};
        border-radius: 3px;
        padding: 5px 14px;
    }}
    QPushButton:hover {{ background: {BG_HOVER}; border-color: {ACCENT_DIM}; }}
    QPushButton:pressed {{ background: {ACCENT_WASH}; color: {ACCENT}; }}
    QPushButton:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER_SOFT}; }}
    QPushButton:default {{ border-color: {ACCENT_DIM}; }}

    /* ---- sliders: the parm editor's main control ---- */
    QSlider::groove:horizontal {{
        height: 3px;
        background: {BORDER};
        border-radius: 1px;
    }}
    QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; border-radius: 1px; }}
    QSlider::handle:horizontal {{
        background: {TEXT};
        width: 11px;
        margin: -5px 0;
        border-radius: 5px;
    }}
    QSlider::handle:horizontal:hover {{ background: {ACCENT}; }}

    /* ---- docks, tabs, group boxes ---- */
    QDockWidget {{ titlebar-close-icon: none; }}
    QDockWidget::title {{
        background: {BG_RAISED};
        padding: 6px 10px;
        border-bottom: 1px solid {BORDER};
    }}
    QTabBar::tab {{
        background: {BG_PANEL};
        color: {TEXT_DIM};
        padding: 6px 14px;
        border: none;
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:selected {{
        color: {ACCENT};
        border-bottom: 2px solid {ACCENT};
    }}
    QTabBar::tab:hover:!selected {{ color: {TEXT}; }}
    QGroupBox {{
        border: 1px solid {BORDER_SOFT};
        border-radius: 4px;
        margin-top: 9px;
        padding-top: 8px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 9px;
        padding: 0 5px;
        color: {TEXT_DIM};
        font-weight: 600;
    }}

    /* ---- status bar ---- */
    QStatusBar {{
        background: {BG_RAISED};
        border-top: 1px solid {BORDER};
        color: {TEXT_DIM};
    }}
    QStatusBar::item {{ border: none; }}

    /* ---- scrollbars ---- */
    QScrollBar:vertical {{
        background: transparent; width: 11px; margin: 0;
    }}
    QScrollBar::handle:vertical {{
        background: {BORDER}; border-radius: 5px; min-height: 26px;
    }}
    QScrollBar::handle:vertical:hover {{ background: {TEXT_FAINT}; }}
    QScrollBar:horizontal {{
        background: transparent; height: 11px; margin: 0;
    }}
    QScrollBar::handle:horizontal {{
        background: {BORDER}; border-radius: 5px; min-width: 26px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: {TEXT_FAINT}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}

    QToolTip {{
        background: {BG_RAISED};
        color: {TEXT};
        border: 1px solid {ACCENT_DIM};
        padding: 4px 7px;
    }}
    QProgressBar {{
        background: {BG_RAISED};
        border: 1px solid {BORDER};
        border-radius: 3px;
        text-align: center;
    }}
    QProgressBar::chunk {{ background: {ACCENT_DIM}; border-radius: 2px; }}
    """


def apply(app) -> None:
    """Put the theme on a ``QApplication``.

    Fusion first: the native Windows style ignores much of a stylesheet and
    paints its own colours for headers and scrollbars, so the palette would be
    honoured in some widgets and not others. Fusion is consistent across
    platforms, which is the whole point of setting a theme.
    """
    from PySide6.QtGui import QColor, QFont, QPalette

    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG_PANEL))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(BG_PANEL))
    palette.setColor(QPalette.AlternateBase, QColor(BG_DEEP))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(BG_RAISED))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT_DIM))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ToolTipBase, QColor(BG_RAISED))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.PlaceholderText, QColor(TEXT_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(TEXT_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(TEXT_FAINT))
    palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(TEXT_FAINT))
    app.setPalette(palette)

    app.setFont(QFont(UI_FONT, 9))
    app.setStyleSheet(stylesheet())
