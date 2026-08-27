"""The RASP ``.eng`` motor database that ships with RASAero II.

RASAero reads ``Documents/RASAero II/rasp.eng``, a concatenation of the
public NAR/TRA certification curves in the format wRASP established. Each
motor is a header line followed by ``time thrust`` pairs::

    Q18000 203 2438 0 72.515 115.696 DEAP-EX
    0.01 5000
    ...

The header fields are name, diameter (mm), length (mm), delays, propellant
mass (kg), total mass (kg), manufacturer. The CDX1 files name a motor as
``"Q18000  (DEAP-EX)"``, so the manufacturer is part of the key -- several
names collide across manufacturers.

Two conventions matter downstream. Thrust is in newtons and the masses are in
kilograms, while every other number in this validation lives in the pound and
foot units RASAero works in; the conversion happens here, once. And the curve
does not start at t=0 -- the first sample is typically 0.01 s at some large
thrust -- so thrust is taken as zero before the first sample rather than
extrapolated back, which is what wRASP and RASAero both do.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from pathlib import Path

LBF_PER_N = 0.2248089430997105
LB_PER_KG = 2.2046226218487757

__all__ = ["Motor", "load_rasp", "MotorDatabase"]


@dataclass(frozen=True)
class Motor:
    """One thrust curve, converted to pounds and seconds."""

    name: str
    manufacturer: str
    diameter_mm: float
    length_mm: float
    propellant_lb: float
    total_lb: float
    times: tuple[float, ...]
    thrust_lbf: tuple[float, ...]

    @property
    def key(self) -> str:
        return f"{self.name}  ({self.manufacturer})"

    @property
    def burn_time(self) -> float:
        return self.times[-1] if self.times else 0.0

    @property
    def total_impulse(self) -> float:
        """lbf-seconds, by the trapezoid rule over the sampled curve."""
        total = 0.0
        for i in range(1, len(self.times)):
            dt = self.times[i] - self.times[i - 1]
            total += 0.5 * dt * (self.thrust_lbf[i] + self.thrust_lbf[i - 1])
        return total

    def thrust_at(self, t: float) -> float:
        """Linearly interpolated thrust, zero outside the sampled window."""
        if t <= 0.0 or not self.times or t >= self.times[-1]:
            return 0.0
        i = bisect.bisect_left(self.times, t)
        if i == 0:
            return 0.0
        t0, t1 = self.times[i - 1], self.times[i]
        f0, f1 = self.thrust_lbf[i - 1], self.thrust_lbf[i]
        if t1 == t0:
            return f1
        return f0 + (f1 - f0) * (t - t0) / (t1 - t0)

    def impulse_to(self, t: float) -> float:
        """Impulse delivered up to ``t``, for the propellant-burned fraction."""
        if t <= 0.0 or not self.times:
            return 0.0
        total = 0.0
        for i in range(1, len(self.times)):
            t0, t1 = self.times[i - 1], self.times[i]
            f0, f1 = self.thrust_lbf[i - 1], self.thrust_lbf[i]
            if t <= t0:
                break
            if t >= t1:
                total += 0.5 * (t1 - t0) * (f1 + f0)
            else:
                frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                f = f0 + (f1 - f0) * frac
                total += 0.5 * (t - t0) * (f + f0)
                break
        return total

    def propellant_burned(self, t: float) -> float:
        """Pounds of propellant consumed by ``t``.

        Apportioned by delivered impulse rather than by time. That assumes a
        constant exhaust velocity, which is the same assumption RASAero and
        every other RASP-driven simulator makes, and it matters for motors
        with a long low-thrust tail: apportioning by time would have the mass
        gone long before the thrust is.
        """
        total = self.total_impulse
        if total <= 0.0:
            return 0.0
        return self.propellant_lb * min(1.0, self.impulse_to(t) / total)


class MotorDatabase(dict):
    """Motors keyed by ``"NAME  (MFG)"`` as the CDX1 files spell it."""

    def lookup(self, engine_field: str) -> Motor | None:
        """Resolve a ``<SustainerEngine>`` value, tolerating spacing drift."""
        exact = self.get(engine_field)
        if exact is not None:
            return exact
        want = " ".join(engine_field.split())
        for key, motor in self.items():
            if " ".join(key.split()) == want:
                return motor
        # Last resort: name alone, but only when it is unambiguous.
        bare = want.split("(")[0].strip()
        hits = [m for m in self.values() if m.name == bare]
        return hits[0] if len(hits) == 1 else None


def _is_header(fields: list[str]) -> bool:
    if len(fields) < 7:
        return False
    try:
        float(fields[0])
    except ValueError:
        return True
    return False


def load_rasp(path: str | Path) -> MotorDatabase:
    """Parse a RASP ``.eng`` file into a database keyed by name and maker."""
    db = MotorDatabase()
    name = mfg = ""
    diameter = length = prop_kg = total_kg = 0.0
    times: list[float] = []
    thrusts: list[float] = []

    def flush() -> None:
        if name and times:
            motor = Motor(
                name=name,
                manufacturer=mfg,
                diameter_mm=diameter,
                length_mm=length,
                propellant_lb=prop_kg * LB_PER_KG,
                total_lb=total_kg * LB_PER_KG,
                times=tuple(times),
                thrust_lbf=tuple(t * LBF_PER_N for t in thrusts),
            )
            db.setdefault(motor.key, motor)

    for raw in Path(path).read_text(errors="replace").splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        fields = line.split()
        if _is_header(fields):
            flush()
            name, mfg = fields[0], fields[6]
            diameter, length = float(fields[1]), float(fields[2])
            prop_kg, total_kg = float(fields[4]), float(fields[5])
            times, thrusts = [], []
        elif len(fields) >= 2 and times is not None:
            try:
                t, f = float(fields[0]), float(fields[1])
            except ValueError:
                continue
            if times and t <= times[-1]:
                continue
            times.append(t)
            thrusts.append(f)
    flush()
    return db
