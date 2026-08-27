"""Test-session setup that must happen before anything else imports Qt.

The GUI tests build real widgets, and by default Qt puts them on the actual
display: running the suite spawned windows that took focus, and any test that
opened a modal waited forever for a human to dismiss it. A suite that can wedge
the machine it runs on is not a suite anyone will run.

Forcing the offscreen platform makes widget construction, layout and painting
all happen in memory. The tests exercise the same code; nothing reaches the
screen and nothing can block on a person.

This lives at the repo root, and sets the variable at import time, because Qt
reads it once when QGuiApplication is first created -- setting it from a
fixture would be too late.
"""

from __future__ import annotations

import os

# Respect an explicit choice, so a developer can still watch the widgets by
# running with QT_QPA_PLATFORM=windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Belt and braces: if a test does reach a modal despite the above, fail rather
# than wait. Qt honours neither of these automatically, so the app code uses
# status-bar messages instead of dialogs for anything on an analysis path.
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")
