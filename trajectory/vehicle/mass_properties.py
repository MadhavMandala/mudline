"""Vehicle mass properties as a function of remaining propellant.

Propellant mass is a state the integrator carries, not something reconstructed
from elapsed time. The previous ``at_time(t, mass_flow)`` computed

    mass = max(dry_mass, mass_0 - mass_flow * t)

which treats the *instantaneous* mass flow as if it had applied for the whole
flight. That is only correct while mdot is constant, and it failed hard at
burnout: once thrust reached zero mdot went to zero, the subtraction vanished,
and the vehicle recovered its full launch mass for the entire coast. Taking the
remaining propellant directly removes the possibility.

Where the propellant sits
-------------------------
The remaining propellant has a centroid, and where that centroid is depends on
how the propellant is consumed:

    radial      A solid core burner. The grain burns outward from the bore, so
                the remaining annulus keeps the same axial centroid throughout.
                Fixed centroid, and this is the case that needs no geometry.

    liquid      A tank draining under axial acceleration. Propellant settles
                against the aft bulkhead, so the liquid column shortens from the
                forward end and its centroid migrates aft as the tank empties.

    end_burner  A solid burning from one face. The flame front regresses, so the
                remaining grain shortens from the burning end and its centroid
                migrates away from it.

For a constant cross-section the centroid is *exactly* linear in fill fraction
-- a column of length ``f*L`` settled against one end has its centroid at
``end -/+ f*L/2`` -- so the interpolation below is not an approximation for the
usual cylindrical tank or grain. Both endpoints are supplied by the caller,
which is what makes a liquid expressible at all: the old class held a single
fixed point and had no way to say where the *remaining* propellant was.

More than one column
--------------------
A biprop does not have *a* propellant column; it has a fuel tank and an
oxidizer tank at different stations, draining together at the engine's
mixture ratio. ``set_propellant_loads`` describes each as its own
``PropellantLoad`` with a ``drain_share`` -- the fraction of the engine's
mass flow it supplies. The integrator still carries one propellant state;
the split into tanks is bookkeeping done here, with a waterfall so a tank
that runs dry hands its share to the ones still wet and the total always
matches what the integrator says is left. A single column is the one-load
special case, kept as ``set_propellant_geometry`` for the callers and tests
that already speak it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: How the propellant is consumed, and therefore how its centroid moves.
RADIAL = "radial"
LIQUID = "liquid"
END_BURNER = "end_burner"
BURN_GEOMETRIES = (RADIAL, LIQUID, END_BURNER)


@dataclass
class PropellantLoad:
    """One draining propellant column -- a tank, or a grain.

    ``drain_share`` is the fraction of the engine's total mass flow this
    column supplies while it still has anything in it: for a biprop at
    mixture ratio MR, the oxidizer side carries MR/(1+MR) and the fuel side
    the rest. Shares need not sum to one across loads -- they are normalized
    over whatever is still wet -- and a share of zero means the load only
    drains once every share-bearing load is empty.
    """

    mass_kg: float
    forward: np.ndarray
    aft: np.ndarray
    burn_geometry: str = RADIAL
    radius_m: float = 0.0
    drain_share: float = 1.0

    def __post_init__(self):
        if self.burn_geometry not in BURN_GEOMETRIES:
            raise ValueError(
                f"burn_geometry must be one of {BURN_GEOMETRIES}; "
                f"got {self.burn_geometry!r}"
            )
        self.forward = np.asarray(self.forward, dtype=float)
        self.aft = np.asarray(self.aft, dtype=float)
        self.mass_kg = max(float(self.mass_kg), 0.0)
        self.drain_share = max(float(self.drain_share), 0.0)

    @property
    def length_m(self) -> float:
        return float(np.linalg.norm(self.aft - self.forward))

    def centroid(self, fraction: float) -> np.ndarray:
        """Centroid of the remaining contents, in body coordinates."""
        centre = 0.5 * (self.forward + self.aft)
        if self.burn_geometry == LIQUID:
            empty = self.aft
        elif self.burn_geometry == END_BURNER:
            empty = self.forward
        else:
            empty = centre
        f = float(np.clip(fraction, 0.0, 1.0))
        return empty + (centre - empty) * f

    def inertia(self, mass: float, fraction: float, roll_axis: int) -> np.ndarray:
        """Inertia of the remaining contents about their own centroid.

        The same cylinder model as the single-column path: a draining or
        regressing column shortens; a core burner keeps its outer radius
        and its length and opens a bore.
        """
        return _column_inertia(
            mass, fraction, self.length_m, self.radius_m, self.burn_geometry, roll_axis,
        )


class MassProperties:
    """Mass properties as a function of remaining propellant mass."""

    def __init__(self, dry_mass: float, prop_mass: float,
                 cg_dry: np.ndarray, i_tensor_dry: np.ndarray,
                 roll_axis: int = 1):
        """
        Args:
            dry_mass: Vehicle dry mass [kg]
            prop_mass: Initial propellant mass [kg]
            cg_dry: CG position at dry mass [m]
            i_tensor_dry: Inertia tensor at dry mass [kg-m^2]
            roll_axis: Which body axis the vehicle rolls about. The simulator's
                body frame is +Y forward, so 1 -- but this used to be assumed
                to be 2, and the roll term was quietly applied to a transverse
                axis instead. On an axisymmetric vehicle that split pitch and
                yaw inertia by 40% at burnout, which is not a thing a body of
                revolution can do.
        """
        self.dry_mass = dry_mass
        self.prop_mass = prop_mass
        self.mass_0 = dry_mass + prop_mass
        self.cg_dry = np.array(cg_dry, dtype=float)
        self.i_tensor_dry = np.array(i_tensor_dry, dtype=float)
        self.roll_axis = int(roll_axis)

        # Centroid of a full propellant load, in body coordinates.
        self.cg_prop_full = np.array([0.0, -2.0, 0.0])
        # Centroid the load tends to as it empties. Equal to cg_prop_full means
        # a fixed centroid, which is the radial-burning solid.
        self.cg_prop_empty = None

        self.burn_geometry = RADIAL
        #: Propellant column length and radius, for its own inertia. Zero
        #: length falls back to treating the load as a point mass at its
        #: centroid, which is what the class did before.
        self.prop_length_m = 0.0
        self.prop_radius_m = 0.0

        #: Individually draining columns -- the biprop case. Empty means the
        #: single-column attributes above describe the whole load.
        self.loads: list[PropellantLoad] = []

    # ------------------------------------------------------------------

    def set_propellant_loads(self, loads) -> None:
        """Describe the propellant as several draining columns.

        Replaces whatever ``set_propellant_geometry`` said: with loads
        present, the single-column attributes are ignored. The loads' masses
        should sum to ``prop_mass``; if the two were configured
        inconsistently, the integrator's propellant state stays authoritative
        for the total and the per-load masses are rescaled to it.
        """
        self.loads = [
            load if isinstance(load, PropellantLoad) else PropellantLoad(**load)
            for load in loads
        ]

    def set_propellant_geometry(self, forward_station_point: np.ndarray,
                                aft_station_point: np.ndarray,
                                burn_geometry: str = RADIAL,
                                radius_m: float = 0.0) -> None:
        """Describe the tank or grain, in body coordinates.

        Args:
            forward_station_point: Body position of the forward end.
            aft_station_point: Body position of the aft end.
            burn_geometry: One of ``RADIAL``, ``LIQUID``, ``END_BURNER``.
            radius_m: Propellant column radius, for its own inertia.
        """
        if burn_geometry not in BURN_GEOMETRIES:
            raise ValueError(
                f"burn_geometry must be one of {BURN_GEOMETRIES}; "
                f"got {burn_geometry!r}"
            )

        forward = np.asarray(forward_station_point, dtype=float)
        aft = np.asarray(aft_station_point, dtype=float)
        centre = 0.5 * (forward + aft)

        self.loads = []          # last caller wins; this is the one-column story
        self.burn_geometry = burn_geometry
        self.cg_prop_full = centre
        self.prop_length_m = float(np.linalg.norm(aft - forward))
        self.prop_radius_m = float(radius_m)

        if burn_geometry == LIQUID:
            # Settles against the aft bulkhead under thrust, so the last drop
            # sits at the aft end.
            self.cg_prop_empty = aft
        elif burn_geometry == END_BURNER:
            # Burns from the aft face forward, so what is left is at the top.
            self.cg_prop_empty = forward
        else:
            self.cg_prop_empty = centre

    # ------------------------------------------------------------------

    def propellant_centroid(self, prop_fraction: float) -> np.ndarray:
        """Centroid of the remaining propellant, in body coordinates."""
        full = self.cg_prop_full
        empty = self.cg_prop_empty
        if empty is None:
            return full
        fraction = float(np.clip(prop_fraction, 0.0, 1.0))
        return empty + (full - empty) * fraction

    def _propellant_inertia(self, mass: float, fraction: float) -> np.ndarray:
        """Inertia of the remaining propellant about its own centroid.

        A cylinder whose length shrinks with fill for a draining or
        regressing column, or whose bore opens for a core burner -- the two
        ways a load actually gives up volume.
        """
        return _column_inertia(
            mass, fraction, self.prop_length_m, self.prop_radius_m,
            self.burn_geometry, self.roll_axis,
        )

    # ------------------------------------------------------------------

    def _remaining_per_load(self, consumed: float) -> list[float]:
        """What each load still holds after ``consumed`` kg have been burned.

        Consumption is split by ``drain_share`` over the loads that still
        have anything in them, waterfalling when one runs dry: at O/F 2.3
        the oxidizer side supplies 70% of every kilogram burned until the
        oxidizer is gone, after which whatever still bears a share supplies
        the rest. Shares of zero mark a load that only drains once every
        share-bearing load is empty -- then proportional to what remains, so
        the total always reconciles. Each pass either finishes the job or
        empties at least one load, which bounds the loop.
        """
        remaining = [load.mass_kg for load in self.loads]
        left = max(float(consumed), 0.0)
        tiny = 1e-12
        for _ in range(len(remaining) + 1):
            if left <= tiny:
                break
            wet = [i for i, r in enumerate(remaining) if r > tiny]
            if not wet:
                break
            shares = [self.loads[i].drain_share for i in wet]
            total_share = sum(shares)
            if total_share <= 0.0:
                shares = [remaining[i] for i in wet]
                total_share = sum(shares)
            taken = 0.0
            for i, share in zip(wet, shares):
                take = min(left * share / total_share, remaining[i])
                remaining[i] -= take
                taken += take
            left -= taken
            if taken <= tiny:
                break
        return remaining

    def _at_propellant_loads(self, prop_remaining: float, mass: float) -> tuple:
        """The multi-column path: several loads, one propellant state."""
        loads_total = sum(load.mass_kg for load in self.loads)
        consumed = float(np.clip(loads_total - prop_remaining, 0.0, loads_total))
        remaining = self._remaining_per_load(consumed)

        # The integrator's state is authoritative for the total. If the loads
        # were configured against a different number -- a propulsion file's
        # mass disagreeing with the tanks' -- scale rather than losing the
        # difference into an unpositioned remainder, which would drag the CG
        # toward the origin.
        located = sum(remaining)
        scale = prop_remaining / located if located > 0 else 0.0

        moment = self.cg_dry * self.dry_mass
        placed = []          # (mass, centroid, load, fraction)
        for load, left in zip(self.loads, remaining):
            fraction = left / load.mass_kg if load.mass_kg > 0 else 0.0
            m_i = left * scale
            c_i = load.centroid(fraction)
            placed.append((m_i, c_i, load, fraction))
            moment = moment + c_i * m_i
        cg = moment / mass

        i_tensor = self.i_tensor_dry + _parallel_axis(
            self.dry_mass, self.cg_dry - cg
        )
        for m_i, c_i, load, fraction in placed:
            i_tensor = i_tensor + load.inertia(m_i, fraction, self.roll_axis)
            i_tensor = i_tensor + _parallel_axis(m_i, c_i - cg)
        return mass, cg, i_tensor

    def at_propellant(self, prop_remaining: float) -> tuple:
        """Return (mass, cg, inertia) for the given remaining propellant [kg]."""
        prop_remaining = max(0.0, float(prop_remaining))
        mass = self.dry_mass + prop_remaining

        if self.loads and mass > 0 and any(l.mass_kg > 0 for l in self.loads):
            return self._at_propellant_loads(prop_remaining, mass)

        fraction = (
            prop_remaining / self.prop_mass if self.prop_mass > 0 else 0.0
        )

        if mass <= 0:
            return mass, self.cg_dry.copy(), self.i_tensor_dry.copy()

        centroid = self.propellant_centroid(fraction)
        cg = (
            self.cg_dry * self.dry_mass + centroid * prop_remaining
        ) / mass

        # Compose about the *current* CG rather than scaling the dry tensor.
        # Scaling was a fudge that ignored the parallel-axis contribution of
        # the propellant, which for a load 41% of wet mass sitting 316 mm off
        # the dry CG is not a small term.
        i_tensor = self.i_tensor_dry + _parallel_axis(
            self.dry_mass, self.cg_dry - cg
        )
        i_tensor = i_tensor + self._propellant_inertia(prop_remaining, fraction)
        i_tensor = i_tensor + _parallel_axis(prop_remaining, centroid - cg)

        return mass, cg, i_tensor


def _column_inertia(mass: float, fraction: float, length_m: float, radius_m: float,
                    burn_geometry: str, roll_axis: int) -> np.ndarray:
    """Inertia of what is left of a propellant column, about its own centroid.

    A core burner keeps its outer radius and opens a bore: what remains is
    an annulus, ``bore^2 = R^2 (1 - f)``, with roll inertia
    ``m (R^2 + bore^2) / 2`` -- which *grows* per kilogram as the grain
    burns out toward the case. It used to be modelled as a solid cylinder
    losing radius, ``R sqrt(f)``, whose roll inertia falls the other way: 3x
    low at half burn, 19x at a tenth. A draining or regressing column
    shortens at its full radius.
    """
    tensor = np.zeros((3, 3))
    if mass <= 0:
        return tensor
    f = float(np.clip(fraction, 0.0, 1.0))
    if burn_geometry == RADIAL:
        length = length_m
        outer2 = radius_m ** 2
        radii2 = outer2 + outer2 * (1.0 - f)          # R^2 + bore^2
    else:
        length = length_m * f
        radii2 = radius_m ** 2
    transverse = mass * (3.0 * radii2 + length ** 2) / 12.0
    roll = 0.5 * mass * radii2
    for axis in range(3):
        tensor[axis, axis] = roll if axis == roll_axis else transverse
    return tensor


def _parallel_axis(mass: float, offset: np.ndarray) -> np.ndarray:
    """Steiner term for shifting an inertia tensor by ``offset``."""
    if mass <= 0:
        return np.zeros((3, 3))
    offset = np.asarray(offset, dtype=float)
    return mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
