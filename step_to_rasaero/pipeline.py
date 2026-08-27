"""End-to-end project generation for STEP-to-RASAero workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cad_extractor import extract_geometry, write_extraction
from .rasaero_tools import open_in_rasaero
from .rasaero_writer import write_rasaero_cdx1


def generate_rasaero_project(
    project_dir: str | Path,
    output_dir: str | Path | None = None,
    open_after_generate: bool = False,
) -> dict[str, Any]:
    """Extract geometry, write debug JSON, and generate a RASAero CDX1 file."""
    project_dir = Path(project_dir)
    output_dir = Path(output_dir) if output_dir is not None else project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = extract_geometry(project_dir)
    geometry_path, review_path = write_extraction(result, output_dir)
    cdx1_path = write_rasaero_cdx1(result.model, output_dir / "rasaero_model.cdx1")
    opened = False
    if open_after_generate:
        open_in_rasaero(cdx1_path)
        opened = True
    return {
        "model": result.model,
        "review": result.review,
        "geometry_path": geometry_path,
        "review_path": review_path,
        "cdx1_path": cdx1_path,
        "opened": opened,
    }
