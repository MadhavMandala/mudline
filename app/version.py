"""What build this is.

A design tool whose output people compare needs to be able to say which build
produced a number. Two engineers on two commits get two apogees for the same
vehicle, and until the run says which code flew it the difference is
unattributable -- the model fingerprint in ``app.pipeline`` answers "was this
the same vehicle", never "was this the same tool".

So the version goes three places: the title bar, so it is visible without
looking for it; Help > About, so it can be read out in a bug report; and into
every saved project and every recorded run, so a result file carries its own
provenance long after the session is gone.

The revision is read from git when the working copy is a checkout -- which,
for an internally distributed tool, is how everyone runs it. It is cached: the
title bar refreshes on every edit and none of them can change the commit.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

#: Kept in step with ``pyproject.toml``. The declared version is the one that
#: means something to a person; the revision below is the one that identifies
#: the code exactly.
__version__ = "0.1.0"

_REPO = Path(__file__).resolve().parent.parent


@lru_cache(maxsize=1)
def git_revision() -> str:
    """The short commit, with ``+`` appended when the tree has edits.

    Empty when this is not a checkout or git is not installed -- an installed
    copy is still perfectly usable, it just cannot name its own commit.
    """
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=_REPO, capture_output=True, text=True,
            timeout=5, check=True,
        ).stdout.strip()

    try:
        revision = _git("rev-parse", "--short", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
    except Exception:      # noqa: BLE001 - no git, no checkout, no answer
        return ""
    return f"{revision}{'+' if dirty else ''}"


@lru_cache(maxsize=1)
def build_string() -> str:
    """``0.1.0 (a1b2c3d)``, or just the version when the commit is unknown."""
    revision = git_revision()
    return f"{__version__} ({revision})" if revision else __version__


def provenance() -> dict[str, str]:
    """What to record with a result so it can be traced back to this build."""
    return {"app_version": __version__, "app_revision": git_revision()}
