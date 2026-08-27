"""Readers for the two formats motors are actually distributed in.

Every commercial motor you can buy is published as a RASP ``.eng`` or a RockSim
``.rse`` -- that is what ThrustCurve.org serves, what the manufacturers ship,
and what every other rocketry tool reads. Until now this tool read neither, so
using a real motor meant transcribing a thrust curve by hand, which is both
tedious and a good way to introduce a quiet error into the one input that sets
rail exit, max-Q and apogee.

Both formats carry the same physics with different spellings:

    .eng    a header line of seven fields, then ``time thrust`` pairs
    .rse    XML, with the curve as ``<eng-data t=".." f=".."/>`` elements

Units are the trap. RASP writes propellant and total mass in *kilograms* but
diameter and length in millimetres; RockSim writes masses in *grams*. Both are
normalised to SI here, once, so nothing downstream has to remember.

A file may hold many motors -- ThrustCurve bundles are usually a whole
manufacturer's catalogue -- so parsing returns a list and the caller chooses.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

G0 = 9.80665

#: Total impulse ceiling for each class letter, per NAR/TRA.
IMPULSE_CLASSES = [
    (0.3125, "1/8A"), (0.625, "1/4A"), (1.25, "1/2A"), (2.5, "A"),
    (5.0, "B"), (10.0, "C"), (20.0, "D"), (40.0, "E"), (80.0, "F"),
    (160.0, "G"), (320.0, "H"), (640.0, "I"), (1280.0, "J"),
    (2560.0, "K"), (5120.0, "L"), (10240.0, "M"), (20480.0, "N"),
    (40960.0, "O"),
]


def impulse_class(total_impulse_ns: float) -> str:
    if total_impulse_ns <= 0:
        return "-"
    for limit, letter in IMPULSE_CLASSES:
        if total_impulse_ns <= limit:
            return letter
    return "O+"


@dataclass
class ThrustCurve:
    """One motor as published: its curve, its masses, and what they imply."""

    name: str
    manufacturer: str = ""
    diameter_mm: float = 0.0
    length_mm: float = 0.0
    delays: str = ""
    propellant_mass_kg: float = 0.0
    total_mass_kg: float = 0.0
    times_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    thrust_n: np.ndarray = field(default_factory=lambda: np.zeros(0))
    #: Only RockSim files carry a declared Isp; RASP files do not.
    declared_isp_s: float = 0.0
    source: str = ""

    # ------------------------------------------------------------------

    @property
    def total_impulse_ns(self) -> float:
        if len(self.times_s) < 2:
            return 0.0
        return float(np.trapezoid(self.thrust_n, self.times_s))

    @property
    def burn_time_s(self) -> float:
        return float(self.times_s[-1]) if len(self.times_s) else 0.0

    @property
    def peak_thrust_n(self) -> float:
        return float(np.max(self.thrust_n)) if len(self.thrust_n) else 0.0

    @property
    def average_thrust_n(self) -> float:
        burn = self.burn_time_s
        return self.total_impulse_ns / burn if burn > 0 else 0.0

    @property
    def impulse_class(self) -> str:
        return impulse_class(self.total_impulse_ns)

    @property
    def implied_isp_s(self) -> float:
        """Isp the curve and the propellant load imply between them.

        Preferred over any declared figure, because it is the one that makes
        mass flow integrate back to the propellant actually loaded. A declared
        Isp that disagrees means the file is internally inconsistent.
        """
        if self.propellant_mass_kg <= 0:
            return 0.0
        return self.total_impulse_ns / (self.propellant_mass_kg * G0)

    @property
    def dry_mass_kg(self) -> float:
        return max(0.0, self.total_mass_kg - self.propellant_mass_kg)

    def points(self) -> list[tuple[float, float]]:
        return [(float(t), float(f)) for t, f in zip(self.times_s, self.thrust_n)]

    def summary(self) -> str:
        return (
            f"{self.name} ({self.manufacturer})  {self.impulse_class} class, "
            f"{self.total_impulse_ns:,.0f} N-s, {self.burn_time_s:.2f} s burn, "
            f"{self.peak_thrust_n:,.0f} N peak, "
            f"{self.propellant_mass_kg * 1000:,.0f} g propellant"
        )


# ----------------------------------------------------------------------
# RASP .eng
# ----------------------------------------------------------------------


def parse_eng(text: str) -> list[ThrustCurve]:
    """Parse a RASP ``.eng`` file, which may hold several motors.

    The header is seven whitespace-separated fields:

        name  diameter_mm  length_mm  delays  propellant_kg  total_kg  mfg

    A header is told from a data line by shape: data lines are two numbers, so
    anything whose first field does not parse as a number starts a new motor.
    """
    motors: list[ThrustCurve] = []
    current: ThrustCurve | None = None
    times: list[float] = []
    thrusts: list[float] = []

    def flush() -> None:
        nonlocal current, times, thrusts
        if current is None:
            return
        if len(times) >= 2:
            current.times_s = np.array(times, dtype=float)
            current.thrust_n = np.maximum(np.array(thrusts, dtype=float), 0.0)
            motors.append(current)
        current, times, thrusts = None, [], []

    for raw in text.splitlines():
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue

        fields = line.split()
        try:
            first = float(fields[0])
        except ValueError:
            first = None

        if first is None:
            # A name that does not parse as a number: a new motor header.
            flush()
            if len(fields) < 7:
                raise ValueError(
                    f"Malformed .eng header, expected 7 fields: {line!r}"
                )
            current = ThrustCurve(
                name=fields[0],
                diameter_mm=float(fields[1]),
                length_mm=float(fields[2]),
                delays=fields[3],
                propellant_mass_kg=float(fields[4]),
                total_mass_kg=float(fields[5]),
                manufacturer=" ".join(fields[6:]),
                source="eng",
            )
            times, thrusts = [], []
        elif len(fields) >= 2 and current is not None:
            times.append(first)
            thrusts.append(float(fields[1]))

    flush()

    for motor in motors:
        _prepend_ignition(motor)
    if not motors:
        raise ValueError("No motors found in this .eng file.")
    return motors


# ----------------------------------------------------------------------
# RockSim .rse
# ----------------------------------------------------------------------


def parse_rse(text: str) -> list[ThrustCurve]:
    """Parse a RockSim ``.rse`` file. Masses in these are grams."""
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValueError(f"Not valid RockSim XML: {exc}") from exc

    motors: list[ThrustCurve] = []
    for element in root.iter("engine"):
        times: list[float] = []
        thrusts: list[float] = []
        for point in element.iter("eng-data"):
            time = point.get("t")
            thrust = point.get("f")
            if time is None or thrust is None:
                continue
            times.append(float(time))
            thrusts.append(max(0.0, float(thrust)))

        if len(times) < 2:
            continue

        motor = ThrustCurve(
            name=element.get("code", "").strip() or "motor",
            manufacturer=element.get("mfg", "").strip(),
            diameter_mm=_number(element.get("dia")),
            length_mm=_number(element.get("len")),
            delays=element.get("delays", "").strip(),
            # Grams here, unlike .eng.
            propellant_mass_kg=_number(element.get("propWt")) / 1000.0,
            total_mass_kg=_number(element.get("initWt")) / 1000.0,
            declared_isp_s=_number(element.get("Isp")),
            times_s=np.array(times, dtype=float),
            thrust_n=np.array(thrusts, dtype=float),
            source="rse",
        )
        _prepend_ignition(motor)
        motors.append(motor)

    if not motors:
        raise ValueError("No motors with a usable thrust curve in this .rse file.")
    return motors


def _number(value) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).strip())
    except ValueError:
        return 0.0


def _prepend_ignition(motor: ThrustCurve) -> None:
    """Both formats usually omit the (0, 0) point; the integrator needs it.

    Without it the curve starts mid-rise and the first sample is treated as
    thrust already present at t=0, which overstates total impulse.
    """
    if len(motor.times_s) and motor.times_s[0] > 0.0:
        motor.times_s = np.concatenate([[0.0], motor.times_s])
        motor.thrust_n = np.concatenate([[0.0], motor.thrust_n])


# ----------------------------------------------------------------------


def load_thrust_curves(path: str | Path) -> list[ThrustCurve]:
    """Read a motor file, dispatching on its extension then its content."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    suffix = path.suffix.lower()

    # Content wins over the extension. Files renamed by hand are common, and a
    # .eng that is plainly XML should still be read rather than refused.
    looks_like_xml = re.search(r"<\s*engine[\s>]", text) is not None
    if looks_like_xml or suffix == ".rse":
        motors = parse_rse(text)
    else:
        motors = parse_eng(text)

    for motor in motors:
        if not motor.name:
            motor.name = path.stem
    return motors


def load_thrust_curve(path: str | Path, index: int = 0) -> ThrustCurve:
    """The one motor a caller wants, from a file that may hold many."""
    motors = load_thrust_curves(path)
    if not 0 <= index < len(motors):
        raise IndexError(
            f"{path} holds {len(motors)} motor(s); asked for index {index}."
        )
    return motors[index]
