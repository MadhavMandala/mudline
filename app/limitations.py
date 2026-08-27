"""What the tool does not model, said out loud, inside the tool.

The README carries this list too, but a README is read once and a limitation
matters at the moment someone is about to trust a number. So it is on the Help
menu, and the general list is followed by whatever is specifically wrong with
the vehicle currently open -- the steep-boattail caveat being the one the
scoreboard can actually measure.

Deliberate limitations, stated so they are not mistaken for oversights. Keep
this in step with the README's Limitations section; they are the same claim
made to two audiences.
"""

from __future__ import annotations

GENERAL: list[tuple[str, str]] = [
    ("No control",
     "The gimbal is modelled but never commanded. Every flight is unguided."),
    ("No staging",
     "Single stage, no separation events."),
    ("No slosh",
     "A tank's propellant drains and its CG moves, but the liquid is rigid; a "
     "motor burns by its declared geometry, not a grain-regression model."),
    ("Flat-Earth gravity",
     "Coriolis is available and off by default. Fine for sounding rockets, "
     "not for orbit."),
    ("Roll is rigid-body only",
     "Roll damping and cant forcing come from the fin planform and the "
     "engine's fin normal-force slope. No roll-pitch resonance or lock-in "
     "analysis, and no fin-alpha reduction at high roll rates."),
    ("No structures or thermal",
     "Max-Q is reported but nothing consumes it -- no fin flutter, buckling, "
     "or aeroheating."),
    ("Aero is component-buildup",
     "RASAero-class, measured at 5.4% mean apogee error over 21 flights with "
     "measured apogees. It degrades at high alpha, on blunt bodies, and "
     "wherever a plume fills the base. Matching RASAero exactly does not make "
     "RASAero right."),
    ("Beyond the table's alpha range",
     "Past 16 degrees by default the flight model blends into an empirical "
     "extension -- Jorgensen crossflow on the body, stalled flat plates for "
     "the fins. It keeps the forces bounded and the right size out to 90 "
     "degrees. It is a correlation, not a computation."),
    ("Fins are flat plates",
     "In geometry; no airfoil section yet."),
]

HEADER = (
    "What this tool does not model. These are choices, not bugs -- but a "
    "number that depends on one of them is a number to be careful with."
)


def _wrap(text: str, width: int = 66, indent: str = "    ") -> str:
    import textwrap

    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def limitations_report(model=None) -> str:
    """The general list, then anything specific to *model*."""
    lines = [_wrap(HEADER, indent=""), ""]
    for title, detail in GENERAL:
        lines.append(f"  {title}")
        lines.append(_wrap(detail))
        lines.append("")

    specific = _model_caveats(model)
    if specific:
        lines.append("-" * 68)
        lines.append("")
        lines.append("  This vehicle in particular")
        lines.append("")
        for caveat in specific:
            lines.append(_wrap(f"- {caveat}"))
            lines.append("")

    return "\n".join(lines).rstrip()


def _model_caveats(model) -> list[str]:
    """Caveats the open vehicle triggers, from the aero code that owns them.

    The boattail caveat proper needs a flight, because it is keyed on Mach as
    well as geometry. Here there is only a vehicle, so the geometry half is
    checked and the Mach half is stated as the condition it is: this shows up
    before anyone has flown anything, which is when it is worth knowing.
    """
    if model is None:
        return []
    try:
        from aeroengine.basedrag import SEPARATION_ANGLE_DEG
        from parametric.aero import SUPERSONIC_MACH, boattail_caveat, \
            steepest_boattail_deg

        angle = steepest_boattail_deg(model)
        if angle <= SEPARATION_ANGLE_DEG:
            return []
        # Ask the owning code for the wording, at a Mach that satisfies its
        # second condition, so this never drifts from the aero report's text.
        caveat = boattail_caveat(angle, SUPERSONIC_MACH)
    except Exception:      # noqa: BLE001 - Help must not be able to fail
        return []
    if not caveat:
        return []
    return [
        f"Applies only if this vehicle goes supersonic (past Mach "
        f"{SUPERSONIC_MACH:g}); below that its drag carries no such caveat.",
        caveat,
    ]
