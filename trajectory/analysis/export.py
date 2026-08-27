"""Trajectory export: CSV time histories and summary plots.

A simulation that can only report its results through a modal dialog is not a
tool anyone can do analysis with. Previously ``run()`` handed back a raw
``solve_ivp`` object and the GUI printed six numbers into a message box; there
was no way to get a time history out, plot it, compare two runs, or hand the
data to anyone else.

This module writes the two things an engineer actually wants:

* a CSV time history, including the derived quantities that are not in the
  state vector but are what get looked at first -- altitude, speed, Mach,
  dynamic pressure, and mass;
* a summary figure, because max-Q and the trajectory shape are things you see
  in one glance at a plot and cannot see at all in a table.

Matplotlib is imported lazily and forced onto the Agg backend at import time
inside each function, so that exporting from a headless run or a worker process
does not try to open a window.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from trajectory.environment.atmosphere import Atmosphere


CSV_COLUMNS = [
    "time_s",
    "east_m", "altitude_m", "north_m",
    "vel_east_mps", "vel_up_mps", "vel_north_mps",
    "speed_mps",
    "qw", "qx", "qy", "qz",
    "p_radps", "q_radps", "r_radps",
    "propellant_kg", "mass_kg",
    "mach", "dynamic_pressure_pa", "density_kgm3",
]


def derived_quantities(states: np.ndarray, dry_mass_kg: float = 0.0) -> dict:
    """Compute the flight quantities that are not carried in the state vector.

    Args:
        states: (n, >=14) state history.
        dry_mass_kg: Added to the propellant column to give total mass. Zero
            leaves the mass column equal to propellant alone.
    """
    atm = Atmosphere()
    altitudes = states[:, 1]
    speeds = np.linalg.norm(states[:, 3:6], axis=1)

    conditions = np.array([atm.get_conditions(float(h))[:4] for h in altitudes])
    density = conditions[:, 0]
    sound = conditions[:, 3]

    mach = np.divide(speeds, sound, out=np.zeros_like(speeds), where=sound > 1e-9)
    q_dyn = 0.5 * density * speeds ** 2

    # The integrator can overshoot an empty tank by a gram; the force model
    # treats that as empty, and so does the record.
    propellant = (
        np.maximum(states[:, 13], 0.0) if states.shape[1] > 13 else np.zeros(len(states))
    )
    return {
        "speed_mps": speeds,
        "mach": mach,
        "dynamic_pressure_pa": q_dyn,
        "density_kgm3": density,
        "propellant_kg": propellant,
        "mass_kg": propellant + dry_mass_kg,
    }


def write_trajectory_csv(result, path: str | Path, dry_mass_kg: float = 0.0,
                         log=None) -> Path:
    """Write the full time history to CSV. Returns the path written.

    With a ``FlightLog`` the columns it carries -- thrust, drag, angle of
    attack, CG, CP, static margin, felt acceleration, phase -- follow the
    state columns, one row per sample.
    """
    from trajectory.analysis.flightlog import csv_cell

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray(result.t, dtype=float)
    states = np.asarray(result.y, dtype=float).T
    derived = derived_quantities(states, dry_mass_kg)
    extra = log.columns() if log is not None else {}
    if extra and any(len(column) != len(times) for column in extra.values()):
        raise ValueError("the flight log does not match the trajectory's samples")
    if log is not None:
        # The log knows the vehicle's mass; without it the column was
        # propellant alone whenever the caller did not pass a dry mass.
        derived["mass_kg"] = np.asarray(log.mass_kg, dtype=float)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*CSV_COLUMNS, *extra])
        for i, t in enumerate(times):
            s = states[i]
            writer.writerow([
                f"{t:.4f}",
                *(f"{v:.6g}" for v in s[0:3]),
                *(f"{v:.6g}" for v in s[3:6]),
                f"{derived['speed_mps'][i]:.6g}",
                *(f"{v:.8g}" for v in s[6:10]),
                *(f"{v:.6g}" for v in s[10:13]),
                f"{derived['propellant_kg'][i]:.6g}",
                f"{derived['mass_kg'][i]:.6g}",
                f"{derived['mach'][i]:.6g}",
                f"{derived['dynamic_pressure_pa'][i]:.6g}",
                f"{derived['density_kgm3'][i]:.6g}",
                *(csv_cell(column[i]) for column in extra.values()),
            ])
    return path


def max_q(result) -> dict:
    """Peak dynamic pressure and where it occurs.

    Max-Q sizes the airframe structure, so it is worth reporting explicitly
    rather than leaving the user to find it in a column of numbers.
    """
    times = np.asarray(result.t, dtype=float)
    states = np.asarray(result.y, dtype=float).T
    q_dyn = derived_quantities(states)["dynamic_pressure_pa"]
    idx = int(np.argmax(q_dyn))
    return {
        "pressure_pa": float(q_dyn[idx]),
        "time_s": float(times[idx]),
        "altitude_m": float(states[idx, 1]),
        "mach": float(derived_quantities(states)["mach"][idx]),
    }


def plot_trajectory(result, path: str | Path, title: str = "Trajectory") -> Path:
    """Write a four-panel summary figure. Returns the path written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    times = np.asarray(result.t, dtype=float)
    states = np.asarray(result.y, dtype=float).T
    derived = derived_quantities(states)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title)

    ax = axes[0, 0]
    ax.plot(times, states[:, 1] / 1000.0, color="tab:blue")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Altitude [km]")
    ax.set_title("Altitude")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(times, derived["speed_mps"], color="tab:orange")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [m/s]")
    ax.set_title("Speed")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(states[:, 0] / 1000.0, states[:, 2] / 1000.0, color="tab:green")
    ax.plot([states[0, 0] / 1000.0], [states[0, 2] / 1000.0], "ko", label="launch")
    ax.plot([states[-1, 0] / 1000.0], [states[-1, 2] / 1000.0], "rx", label="landing")
    ax.set_xlabel("East [km]")
    ax.set_ylabel("North [km]")
    ax.set_title("Ground track")
    ax.axis("equal")
    ax.legend(loc="best", fontsize="small")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(times, derived["dynamic_pressure_pa"] / 1000.0, color="tab:red")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Dynamic pressure [kPa]")
    ax.set_title("Dynamic pressure")
    ax.grid(alpha=0.3)
    peak = max_q(result)
    ax.axvline(peak["time_s"], color="k", linestyle="--", linewidth=0.8)
    ax.annotate(
        f"max-Q {peak['pressure_pa'] / 1000:.0f} kPa\n"
        f"M {peak['mach']:.2f} at {peak['altitude_m'] / 1000:.1f} km",
        xy=(peak["time_s"], peak["pressure_pa"] / 1000.0),
        xytext=(0.45, 0.6), textcoords="axes fraction", fontsize="small",
    )

    # Mark the recovery phases when the run had any.
    for phase in getattr(result, "phases", []) or []:
        if phase.get("name") in {"drogue", "main"}:
            for panel in (axes[0, 0], axes[0, 1]):
                panel.axvline(phase["t_start"], color="tab:purple",
                              linestyle=":", linewidth=1.0)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_dispersion_csv(dispersion, path: str | Path) -> Path:
    """Write one row per dispersion case: sampled inputs and outcomes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cases = dispersion.cases
    if not cases:
        raise ValueError("No dispersion cases to write.")

    param_names = sorted(cases[0].get("params", {}))
    outcome_names = [
        k for k in sorted(cases[0]) if k != "params" and not isinstance(cases[0][k], dict)
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case", *param_names, *outcome_names])
        for i, case in enumerate(cases):
            writer.writerow([
                i,
                *(f"{case['params'].get(n, ''):.6g}" for n in param_names),
                *(case.get(n, "") for n in outcome_names),
            ])
    return path


def plot_dispersion(dispersion, path: str | Path, title: str = "Landing dispersion") -> Path:
    """Scatter the landing points with the confidence ellipse over them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(dispersion.landing_points, dtype=float) / 1000.0
    center, semi_major, semi_minor, orientation = dispersion.ellipse

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(points[:, 0], points[:, 1], s=18, alpha=0.7,
               color="tab:blue", label=f"{len(points)} cases")
    ax.plot(center[0] / 1000.0, center[1] / 1000.0, "k+", markersize=12, label="mean")

    ax.add_patch(Ellipse(
        (center[0] / 1000.0, center[1] / 1000.0),
        width=2 * semi_major / 1000.0,
        height=2 * semi_minor / 1000.0,
        angle=np.degrees(orientation),
        facecolor="none", edgecolor="tab:red", linestyle="--",
        label="95% ellipse",
    ))

    ax.set_xlabel("East [km]")
    ax.set_ylabel("North [km]")
    ax.set_title(f"{title}\nCEP {dispersion.cep_m:,.0f} m")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize="small")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def export_all(result, output_dir: str | Path, dry_mass_kg: float = 0.0,
               stem: str = "trajectory") -> dict:
    """Write both the CSV and the figure for one flight."""
    output_dir = Path(output_dir)
    return {
        "csv": write_trajectory_csv(result, output_dir / f"{stem}.csv", dry_mass_kg),
        "plot": plot_trajectory(result, output_dir / f"{stem}.png"),
    }
