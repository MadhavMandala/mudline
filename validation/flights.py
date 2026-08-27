"""Flight cards: the measured-altitude examples RASAero II ships with.

``Documents/RASAero II/Examples`` holds two dozen CDX1 files whose ``<Comments>``
record what the rocket actually did -- barometric, GPS or integrated-accelerometer
apogee -- next to what RASAero predicted. They are the only tie to reality
available without flying something, so they are the whole basis of the scoreboard.

Reading the measured number is the delicate part. The comments are free prose
written by a dozen different people over fifteen years, and the phrasings do not
converge: ``Altitude = 121478 ft``, ``Barometric Altitude = 3577 ft``,
``Average of Two Barometric Altimeters = 6188 ft``, ``Apogee Altitude 85067 ft
based on integrated accelerometer data``. Worse, several files quote three or
four instruments that disagree, and a regex has no way to know which one the
author treated as the answer -- Kinsel lists 40113, 42231, 42771 and 44924 ft
for a single flight.

So the measured value is not scraped. It is *derived* from the two numbers
RASAero itself printed and stands behind::

    measured = prediction / (1 + error/100)

which inverts the definition ``error = (prediction - measured) / measured``.
That picks out exactly the instrument the author scored against, with no
guessing. The scraped value is still read where the prose allows it, and the
two are cross-checked: a disagreement past rounding means the file is
inconsistent and gets flagged rather than silently averaged.

The spread between instruments is not noise to be tidied away, though -- it is
the floor on how sharply any model can be judged here. ``instrument_spread``
keeps it, and the scoreboard reports it.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["FlightCard", "load_card", "load_examples"]

_ERROR_RE = re.compile(r"Error\s*=\s*(-?[\d.]+)\s*%")
_PRED_RE = re.compile(r"RASAero(?:\s*II)?\s*Prediction\s*=\s*([\d,]+)\s*ft(?!/)")
#: Any "<something> = 12345 ft" that is not a RASAero prediction, plus the
#: bare "Apogee Altitude 85067 ft" form Proteus 6 uses.
_MEAS_RE = re.compile(r"(?:=|Altitude)\s*([\d,]+)\s*ft(?!/)", re.IGNORECASE)


def _num(text: str) -> float:
    return float(text.replace(",", ""))


@dataclass(frozen=True)
class FlightCard:
    """One flown vehicle, its inputs, and what it actually reached."""

    name: str
    path: Path

    # Vehicle and motor
    engine: str
    launch_weight_lb: float
    cg_in: float
    nozzle_diameter_in: float
    ignition_delay_s: float

    # Launch site
    site_altitude_ft: float
    temperature_f: float
    pressure_inhg: float
    rod_angle_deg: float
    rod_length_ft: float
    wind_mph: float

    # Truth and RASAero's own answer
    measured_ft: float
    rasaero_ft: float
    rasaero_error_pct: float
    rasaero_stored_ft: float = 0.0
    staged: bool = False
    instrument_spread: tuple[float, ...] = field(default_factory=tuple)
    scrape_agrees: bool = True

    @property
    def spread_pct(self) -> float:
        """Disagreement between instruments on this flight, as a percentage.

        Zero when only one altitude was reported. This is the measurement
        floor: a model cannot be scored more finely than the truth is known.
        """
        vals = [v for v in self.instrument_spread if v > 0.0]
        if len(vals) < 2:
            return 0.0
        return 100.0 * (max(vals) - min(vals)) / self.measured_ft


def _text(root: ET.Element, path: str, default: str = "") -> str:
    node = root.find(path)
    return default if node is None or node.text is None else node.text.strip()


def _f(root: ET.Element, path: str, default: float = 0.0) -> float:
    try:
        return float(_text(root, path) or default)
    except ValueError:
        return default


def load_card(path: str | Path) -> FlightCard | None:
    """Read one CDX1. Returns ``None`` when it carries no measured altitude."""
    path = Path(path)
    root = ET.fromstring(path.read_text(errors="replace"))
    comments = _text(root, "./RocketDesign/Comments")

    err_m = _ERROR_RE.search(comments)
    pred_m = _PRED_RE.search(comments)
    if not err_m or not pred_m:
        return None

    error_pct = float(err_m.group(1))
    prediction = _num(pred_m.group(1))
    measured = prediction / (1.0 + error_pct / 100.0)

    # Everything the prose offers as an altitude, for the spread; and the
    # closest one to the derived value, to confirm the derivation landed on a
    # number the author actually wrote down.
    scraped = [_num(m) for m in _MEAS_RE.findall(comments)]
    scraped = [v for v in scraped if abs(v - prediction) > 0.5 and v > 100.0]
    agrees = any(abs(v - measured) <= max(1.0, 0.005 * measured) for v in scraped)

    sim = root.find("./SimulationList/Simulation")
    site = root.find("./LaunchSite")
    if sim is None or site is None:
        return None

    return FlightCard(
        name=path.stem,
        path=path,
        engine=_text(sim, "SustainerEngine"),
        launch_weight_lb=_f(sim, "SustainerLaunchWt"),
        cg_in=_f(sim, "SustainerCG"),
        nozzle_diameter_in=_f(sim, "SustainerNozzleDiameter"),
        ignition_delay_s=_f(sim, "SustainerIgnitionDelay"),
        site_altitude_ft=_f(site, "Altitude"),
        temperature_f=_f(site, "Temperature"),
        pressure_inhg=_f(site, "Pressure"),
        rod_angle_deg=_f(site, "RodAngle"),
        rod_length_ft=_f(site, "RodLength"),
        wind_mph=_f(site, "WindSpeed"),
        measured_ft=measured,
        rasaero_ft=prediction,
        rasaero_error_pct=error_pct,
        rasaero_stored_ft=_f(sim, "MaxAltitude"),
        staged=(_text(sim, "IncludeBooster1").lower() == "true"
                or _text(root, "./RocketDesign/UseBooster1").lower() == "true"),
        instrument_spread=tuple(sorted(scraped)),
        scrape_agrees=agrees,
    )


def load_examples(directory: str | Path) -> list[FlightCard]:
    """Every example carrying a measured altitude, in name order."""
    cards = []
    for path in sorted(Path(directory).glob("*.CDX1")):
        card = load_card(path)
        if card is not None:
            cards.append(card)
    return cards
