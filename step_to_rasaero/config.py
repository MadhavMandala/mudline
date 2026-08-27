"""Configuration helpers for the RASAero integration."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_TOOLS_PATH = PROJECT_ROOT / "config" / "local_tools.yaml"


def load_local_tools(path: Path | None = None) -> dict[str, Any]:
    """Load machine-local tool paths.

    The parser prefers PyYAML when available, but the integration does not
    require PyYAML just to discover a normal Windows RASAero install.
    """
    path = path or LOCAL_TOOLS_PATH
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        pass

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return _parse_simple_yaml(text)


def write_local_tools(data: dict[str, Any], path: Path | None = None) -> Path:
    """Write a small local YAML file without requiring PyYAML."""
    path = path or LOCAL_TOOLS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for child_key, child_value in value.items():
                lines.append(f"  {child_key}: {_format_scalar(child_value)}")
        else:
            lines.append(f"{key}: {_format_scalar(value)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current: dict[str, Any] | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if value:
                root[key] = _parse_scalar(value)
                current = None
            else:
                current = {}
                root[key] = current
        elif current is not None:
            key, _, value = line.strip().partition(":")
            current[key.strip()] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    value = value.strip().strip("\"'")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("%USERPROFILE%"):
        value = str(Path(os.environ.get("USERPROFILE", "")) / value[len("%USERPROFILE%") :].lstrip("/\\"))
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _format_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value).replace("\\", "/"))
