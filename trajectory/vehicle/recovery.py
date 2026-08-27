"""Recovery system: drogue and main parachute deployment.

Why phases rather than a flag
-----------------------------
Parachute deployment is a step change in drag area -- typically two to three
orders of magnitude within a fraction of a second. Handling it with a boolean
checked inside the derivative function would put a jump discontinuity in the
middle of an integration step, and an adaptive solver cannot integrate across
that: it would reject the step, halve it, and repeat, converging slowly onto the
event while reporting nonsense in between.

The correct structure is to stop the integration at the deployment condition and
restart it with the new configuration. ``solve_ivp`` already supports exactly
this through terminal events, so the flight is integrated as a sequence of
phases:

    ascent  --(apogee)-->  drogue descent  --(altitude)-->  main descent  --> ground

Each phase is a separate call with a continuous right-hand side, and the state
at the end of one is the initial condition of the next.

Inflation
---------
A real canopy takes finite time to inflate, and the peak load during inflation
sizes the recovery harness. That transient is modelled here as a linear ramp of
effective drag area over ``inflation_time_s``, which keeps the derivative
continuous across deployment. It is not a substitute for a real inflation model
(the classic reference curves are strongly non-linear in canopy loading), but it
avoids pretending a canopy reaches full area instantaneously.

Where the canopy pulls
----------------------
A canopy pulls on the airframe where the harness is anchored -- the nose
shoulder, a coupler bulkhead -- not at the centre of gravity. With
``attachment_station_m`` set, the drag acts at that point and the vehicle
hangs from it: the moment swings the body until the attachment sits
above the CG, nose up, and the swinging attachment point's own motion
through the air is what damps the pendulum. The relative wind the canopy
sees is that point's, ``v + w x r``, since in a rigid model the canopy
has no motion of its own. Left ``None`` the drag goes through the CG with
no moment, which is what it always did; the descent attitude then means
nothing. A separated airframe -- two halves on a shock cord -- is beyond
a rigid body either way.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Parachute:
    """A single canopy.

    Args:
        cda_m2: Drag area, Cd * S [m^2]. This is the quantity recovery
            hardware is actually specified by; splitting it into a Cd and an
            area invites the two to disagree.
        deploy_altitude_m: Deploy when descending through this altitude [m].
            ``None`` means deploy at apogee (the usual drogue trigger).
        inflation_time_s: Ramp from zero to full drag area over this time [s].
        deploy_delay_s: Delay between the trigger and the start of inflation [s].
        attachment_station_m: Where the harness is anchored, as a station
            from the nose tip [m]; ``None`` pulls through the CG.
    """

    cda_m2: float
    deploy_altitude_m: float | None = None
    inflation_time_s: float = 0.5
    deploy_delay_s: float = 0.0
    attachment_station_m: float | None = None

    def drag_area_at(self, time_since_trigger_s: float) -> float:
        """Effective drag area this many seconds after the deploy trigger."""
        t = time_since_trigger_s - self.deploy_delay_s
        if t <= 0.0:
            return 0.0
        if self.inflation_time_s <= 0.0 or t >= self.inflation_time_s:
            return float(self.cda_m2)
        return float(self.cda_m2 * t / self.inflation_time_s)

    def terminal_velocity(self, mass_kg: float, rho_kg_m3: float = 1.225) -> float:
        """Steady descent rate under this canopy [m/s].

        The number recovery systems are sized by: an amateur main is usually
        targeted at 5-7 m/s, a drogue at 20-30 m/s.
        """
        if self.cda_m2 <= 0.0 or rho_kg_m3 <= 0.0:
            return float("inf")
        return float(np.sqrt(2.0 * mass_kg * 9.80665 / (rho_kg_m3 * self.cda_m2)))


@dataclass
class RecoverySystem:
    """Drogue plus main, either of which may be absent."""

    drogue: Parachute | None = None
    main: Parachute | None = None

    @property
    def enabled(self) -> bool:
        return self.drogue is not None or self.main is not None

    @property
    def main_deploy_altitude_m(self) -> float | None:
        if self.main is None:
            return None
        return self.main.deploy_altitude_m

    def describe(self, mass_kg: float) -> str:
        parts = []
        if self.drogue is not None:
            parts.append(
                f"drogue CdA={self.drogue.cda_m2:.2f} m^2 "
                f"({self.drogue.terminal_velocity(mass_kg):.1f} m/s at sea level)"
            )
        if self.main is not None:
            parts.append(
                f"main CdA={self.main.cda_m2:.2f} m^2 at "
                f"{self.main.deploy_altitude_m:.0f} m "
                f"({self.main.terminal_velocity(mass_kg):.1f} m/s at sea level)"
            )
        return "; ".join(parts) if parts else "no recovery"


def standard_recovery(
    dry_mass_kg: float,
    main_descent_mps: float = 6.0,
    drogue_descent_mps: float = 25.0,
    main_deploy_altitude_m: float = 300.0,
    attachment_station_m: float | None = None,
) -> RecoverySystem:
    """Size a dual-deploy system from target descent rates.

    Inverts ``terminal_velocity`` at sea level, which is how recovery hardware
    is chosen in practice: pick a landing speed, then buy the canopy.

    ``main_deploy_altitude_m`` is above the launch site, as an altimeter
    reads it; the simulation adds the pad's own altitude when it arms the
    trigger.
    """
    rho = 1.225
    g = 9.80665

    def cda_for(speed: float) -> float:
        return 2.0 * dry_mass_kg * g / (rho * speed * speed)

    return RecoverySystem(
        drogue=Parachute(cda_for(drogue_descent_mps), deploy_altitude_m=None,
                         inflation_time_s=0.4,
                         attachment_station_m=attachment_station_m),
        main=Parachute(cda_for(main_descent_mps),
                       deploy_altitude_m=main_deploy_altitude_m,
                       inflation_time_s=1.2,
                       attachment_station_m=attachment_station_m),
    )
