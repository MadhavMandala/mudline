"""Discovery and launch helpers for RASAero II."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import load_local_tools


@dataclass(frozen=True)
class RasaeroInstall:
    exe_path: Path | None
    documents_dir: Path
    examples_dir: Path

    @property
    def available(self) -> bool:
        return self.exe_path is not None and self.exe_path.exists()


def discover_rasaero() -> RasaeroInstall:
    """Find RASAero II using local config first, then common install paths."""
    cfg = load_local_tools().get("rasaero", {})
    exe_candidates: list[Path] = []
    docs_candidates: list[Path] = []

    if isinstance(cfg, dict):
        if cfg.get("exe_path"):
            exe_candidates.append(Path(str(cfg["exe_path"])))
        if cfg.get("documents_dir"):
            docs_candidates.append(Path(str(cfg["documents_dir"])))

    exe_candidates.extend(
        [
            Path("C:/Program Files (x86)/RASAero II/RASAero II.exe"),
            Path("C:/Program Files/RASAero II/RASAero II.exe"),
        ]
    )
    docs_candidates.append(Path.home() / "Documents" / "RASAero II")

    exe_path = next((p for p in exe_candidates if p.exists()), None)
    documents_dir = next((p for p in docs_candidates if p.exists()), docs_candidates[-1])
    return RasaeroInstall(
        exe_path=exe_path,
        documents_dir=documents_dir,
        examples_dir=documents_dir / "Examples",
    )


def open_in_rasaero(cdx1_path: str | Path, install: RasaeroInstall | None = None) -> subprocess.Popen:
    """Launch RASAero II with a generated CDX1 file."""
    install = install or discover_rasaero()
    if not install.available or install.exe_path is None:
        raise FileNotFoundError("RASAero II executable was not found. Set config/local_tools.yaml.")
    cdx1_path = Path(cdx1_path)
    if not cdx1_path.exists():
        raise FileNotFoundError(f"CDX1 file does not exist: {cdx1_path}")
    return subprocess.Popen([str(install.exe_path), str(cdx1_path)])
