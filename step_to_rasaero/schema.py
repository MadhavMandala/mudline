"""Shared intermediate schema for generated RASAero projects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PartGuess:
    path: str
    category: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    bounds_m: dict[str, list[float]] = field(default_factory=dict)
    station_range_m: list[float] = field(default_factory=list)


@dataclass
class ExtractionResult:
    model: dict[str, Any]
    review: dict[str, Any]
