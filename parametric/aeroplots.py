"""Draw two coefficient tables against each other.

The numbers in ``aerocompare`` say *how far apart* two methods are at a
handful of conditions. These say *where* they part company, which is the
question you actually have when one of them is a cheap approximation: a 17%
mean drag error is meaningless if it is 2% everywhere except a spike you never
fly through, and alarming if it is 30% across the whole boost.

Two figures, because they answer different questions:

* **Lift-curve slope against Mach** -- does the vehicle respond to angle of
  attack the same way in both? This is the term that sets weathercocking, gust
  response and the whole stability story, and it is where a Prandtl-Glauert
  correction either behaves or does not.
* **Drag against Mach** -- what does it cost to get there? Drawn at zero and at
  angle of attack, because the induced term is a large part of the answer for
  a rocket flying through wind.

Both curves are sampled from ``AeroDatabase`` objects rather than from any
particular solver, so a CFD or wind-tunnel table plots the same way.

Lift versus normal force
------------------------
The tables carry normal force, which is perpendicular to the *body*. Lift is
perpendicular to the *flow*, and the two differ by the axial force resolved
through alpha -- about 3% at 4 degrees, and it grows. Lift is what the label
says, so lift is what is drawn, recovered as

    CL = CN cos(a) - CA sin(a),    CA = CD - CN sin(a)

which is the same decomposition RASAero exports as separate columns, and
reproduces its CL column to six figures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: Matched to the application's palette, restated rather than imported so that
#: the analysis packages do not acquire a dependency on the GUI.
BG = "#1B1E24"
PANEL = "#21252D"
TEXT = "#D6DBE3"
TEXT_DIM = "#8A94A3"
GRID = "#333945"
#: Orange and blue: the pairing that survives every common colour deficiency.
COLOUR_A = "#FF8C2E"
COLOUR_B = "#5AA9E6"
BAD = "#E0574A"


@dataclass
class PlotSettings:
    mach_min: float | None = None
    mach_max: float | None = None
    samples: int = 400
    #: Angle at which the slope is measured, and the second drag curve drawn.
    alpha_deg: float = 4.0
    dpi: int = 140
    figsize: tuple[float, float] = (9.0, 5.0)


def lift_curve_slope(database, mach: np.ndarray, alpha_deg: float) -> np.ndarray:
    """CL per radian at each Mach, measured through ``alpha_deg``."""
    alpha = np.radians(alpha_deg)
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    values = []
    for m in mach:
        row = database.lookup(float(m), alpha_deg)
        # Drag is CA cos(a) + CN sin(a), so recovering the axial force means
        # dividing by cos(a) and not merely subtracting the normal-force term.
        axial = (row.cd - row.cn * sin_a) / max(cos_a, 1e-9)
        lift = row.cn * cos_a - axial * sin_a
        values.append(lift / alpha if alpha else 0.0)
    return np.array(values)


def drag(database, mach: np.ndarray, alpha_deg: float) -> np.ndarray:
    return np.array([database.lookup(float(m), alpha_deg).cd for m in mach])


def _mach_grid(database_a, database_b, settings: PlotSettings) -> np.ndarray:
    low = max(database_a.mach_range[0], database_b.mach_range[0])
    high = min(database_a.mach_range[1], database_b.mach_range[1])
    if settings.mach_min is not None:
        low = max(low, settings.mach_min)
    if settings.mach_max is not None:
        high = min(high, settings.mach_max)
    if high <= low:
        raise ValueError("The two tables share no Mach range.")
    return np.linspace(low, high, settings.samples)


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(PANEL)
    ax.set_title(title, color=TEXT, fontsize=11, pad=12)
    ax.set_xlabel(xlabel, color=TEXT_DIM, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_DIM, fontsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _mark_transonic(ax) -> None:
    """Shade Mach 0.8-1.2, where component-buildup methods are least reliable."""
    ax.axvspan(0.8, 1.2, color=BAD, alpha=0.07, linewidth=0)
    ax.axvline(1.0, color=TEXT_DIM, linewidth=0.7, linestyle=":", alpha=0.6)


def lift_slope_figure(database_a, database_b, name_a: str, name_b: str,
                      settings: PlotSettings | None = None):
    """CL/alpha against Mach, both tables on one pair of axes."""
    import matplotlib.pyplot as plt

    settings = settings or PlotSettings()
    mach = _mach_grid(database_a, database_b, settings)
    slope_a = lift_curve_slope(database_a, mach, settings.alpha_deg)
    slope_b = lift_curve_slope(database_b, mach, settings.alpha_deg)

    figure, (ax, ax_error) = plt.subplots(
        2, 1, figsize=settings.figsize, dpi=settings.dpi,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35},
    )
    figure.patch.set_facecolor(BG)

    _mark_transonic(ax)
    ax.plot(mach, slope_a, color=COLOUR_A, linewidth=1.8, label=name_a)
    ax.plot(mach, slope_b, color=COLOUR_B, linewidth=1.8, label=name_b)
    _style(
        ax,
        f"Lift-curve slope through {settings.alpha_deg:g}° angle of attack",
        "", "CL$_\\alpha$  [per radian]",
    )
    legend = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=9)
    for text in legend.get_texts():
        text.set_color(TEXT)

    _mark_transonic(ax_error)
    with np.errstate(divide="ignore", invalid="ignore"):
        error = 100.0 * (slope_a - slope_b) / slope_b
    ax_error.axhline(0.0, color=TEXT_DIM, linewidth=0.8)
    ax_error.plot(mach, error, color=COLOUR_A, linewidth=1.4)
    ax_error.fill_between(mach, 0, error, color=COLOUR_A, alpha=0.15)
    _style(ax_error, "", "Mach", f"{name_a} error [%]")
    return figure


def drag_figure(database_a, database_b, name_a: str, name_b: str,
                settings: PlotSettings | None = None):
    """CD against Mach at zero alpha and at ``settings.alpha_deg``."""
    import matplotlib.pyplot as plt

    settings = settings or PlotSettings()
    mach = _mach_grid(database_a, database_b, settings)

    figure, (ax, ax_error) = plt.subplots(
        2, 1, figsize=settings.figsize, dpi=settings.dpi,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35},
    )
    figure.patch.set_facecolor(BG)

    _mark_transonic(ax)
    zero_a, zero_b = drag(database_a, mach, 0.0), drag(database_b, mach, 0.0)
    alpha_a = drag(database_a, mach, settings.alpha_deg)
    alpha_b = drag(database_b, mach, settings.alpha_deg)

    ax.plot(mach, zero_a, color=COLOUR_A, linewidth=1.8, label=f"{name_a}, α=0°")
    ax.plot(mach, zero_b, color=COLOUR_B, linewidth=1.8, label=f"{name_b}, α=0°")
    ax.plot(mach, alpha_a, color=COLOUR_A, linewidth=1.3, linestyle="--",
            label=f"{name_a}, α={settings.alpha_deg:g}°")
    ax.plot(mach, alpha_b, color=COLOUR_B, linewidth=1.3, linestyle="--",
            label=f"{name_b}, α={settings.alpha_deg:g}°")
    _style(ax, "Drag coefficient against Mach", "", "CD  [on frontal area]")
    legend = ax.legend(facecolor=PANEL, edgecolor=GRID, fontsize=8, ncol=2)
    for text in legend.get_texts():
        text.set_color(TEXT)

    _mark_transonic(ax_error)
    with np.errstate(divide="ignore", invalid="ignore"):
        error_zero = 100.0 * (zero_a - zero_b) / zero_b
        error_alpha = 100.0 * (alpha_a - alpha_b) / alpha_b
    ax_error.axhline(0.0, color=TEXT_DIM, linewidth=0.8)
    ax_error.plot(mach, error_zero, color=COLOUR_A, linewidth=1.4, label="α=0°")
    ax_error.plot(mach, error_alpha, color=COLOUR_A, linewidth=1.2,
                  linestyle="--", label=f"α={settings.alpha_deg:g}°")
    ax_error.fill_between(mach, 0, error_zero, color=COLOUR_A, alpha=0.15)
    _style(ax_error, "", "Mach", f"{name_a} error [%]")
    return figure


def write_comparison_plots(
    database_a, database_b, out_dir: str | Path,
    name_a: str = "built-in", name_b: str = "RASAero",
    stem: str = "aero", settings: PlotSettings | None = None,
) -> list[Path]:
    """Render both figures to PNG and return where they landed."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for suffix, builder in (
        ("lift_slope", lift_slope_figure),
        ("drag", drag_figure),
    ):
        figure = builder(database_a, database_b, name_a, name_b, settings)
        path = out_dir / f"{stem}_{suffix}.png"
        figure.savefig(path, facecolor=figure.get_facecolor(), bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written
