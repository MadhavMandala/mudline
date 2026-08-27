"""Compressible-flow primitives shared across the engine.

``cp_stagnation`` in particular appears in five unrelated places in the original
-- nose blunting, launch-lug drag, inclined-plate drag, fin Newtonian normal
force and the base-drag vacuum limit -- always written out longhand. It is
factored out here because five copies of a Rayleigh pitot expression is five
chances to typo one of them.
"""

from __future__ import annotations

import math

GAMMA = 1.4


def beta(mach: float) -> float:
    """Prandtl-Glauert parameter, sqrt(M^2 - 1). Supersonic only."""
    return math.sqrt(mach * mach - 1.0)


def total_pressure_ratio(mach: float) -> float:
    """p02/p_inf across a normal shock -- the Rayleigh pitot formula.

    Written in the original as

        (2.4 / (2.8 M^2 - 1.4 + 1.0))^2.5 * (1.2 M^2)^3.5

    which is the gamma = 1.4 case of

        [(g+1)M^2 / 2]^(g/(g-1)) * [(g+1) / (2gM^2 - (g-1))]^(1/(g-1))

    verified algebraically. The ``- 1.4 + 1.0`` is left as RASAero wrote it
    rather than folded to ``- 0.4``; the two differ in the last bits and this
    value is multiplied by a 3.5 power.
    """
    return (2.4 / (2.8 * mach ** 2 - 1.4 + 1.0)) ** 2.5000000000000004 * (
        1.2 * mach ** 2
    ) ** 3.5000000000000004


def cp_stagnation(mach: float) -> float:
    """Stagnation pressure coefficient behind a normal shock.

    Cp = 2/(gamma M^2) * (p02/p_inf - 1). This is the ``Cp_max`` of modified
    Newtonian theory, and it is what RASAero scales sin^2(deflection) by.
    """
    return 2.0 / (1.4 * mach ** 2) * (total_pressure_ratio(mach) - 1.0)


def normal_shock_pressure_ratio(mach: float) -> float:
    """p2/p1 across a normal shock for gamma = 1.4: (2.8 M^2 - 0.4) / 2.4.

    Appears inside the hypersonic base-pressure correlation, where the base
    pressure is referenced to post-shock rather than freestream conditions.
    """
    return (2.8 * mach ** 2 - 0.3999999999999999) / 2.4
