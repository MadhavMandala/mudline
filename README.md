# Mudline

A launch-vehicle design tool. Build a rocket parametrically or import it from
CAD, solve its mass properties, generate its aerodynamics, and fly it in 6-DOF,
all from one application on one model.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[cad,dev]" -c constraints.txt
.venv/Scripts/python.exe -m tools.check_environment
.venv/Scripts/python.exe -m app
```

Install with `-c constraints.txt`. Without the pins a fresh environment
resolves newer releases that break plot export and parametric geometry.

[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) goes from a blank window to
a flown trajectory.

---

## Where the aerodynamics come from

`aeroengine` is a transcription of RASAero II's aerodynamics solver, derived
from the decompiled binary. It is not a reimplementation from published theory
and not a clean-room rewrite.

RASAero II is Charles E. Rogers' work. The aerodynamic model, the correlations
and the curve fits are his. `aeroengine` and `oracle` are the only derived
parts of this repository; everything else is original.

The transcription is bug-for-bug on purpose. It reproduces known defects
because per-term agreement with RASAero is the acceptance bar, and a port that
silently fixes things cannot demonstrate it is a port. The defects are listed
below.

If you own RASAero II, the `"rasaero-app"` aero method drives the real
application and imports its export.

---

## The model

Geometry follows OpenVSP's method. A body is an ordered stack of
cross-sections along a spine, so a nose, a transition and a boattail are the
same object with different sections.

```
VehicleModel
 ├─ Stack "nose"          lofted from XSecs → B-rep solid
 ├─ Stack "forward_tube"
 └─ Stack "motor_tube"
     ├─ FinSet "fins"     attached at a station on its parent
     └─ Motor  "motor"    propellant, thrust curve, where thrust acts
 └─ PointMass "avionics"  mass with a station, no geometry
```

Every number is a Parm: named, bounded, change-tracked. One Stack is one
manufactured part, with one material and one wall thickness.

Analytic nose profiles (conical, ogive, von Kármán, elliptical, power) are
generators that emit sections. Once emitted the sections are ordinary and
individually editable.

`File → Import STEP` slices a solid along its axis, measures the cross-section
at each station, and fits a Stack, so an import is editable rather than a dead
mesh. It measures two radii per station: the lateral extent gives the outer
mould line and the slab volume gives an equivalent material radius, and their
disagreement recovers wall thickness. Round-tripping known geometry gives exact
length, diameter −0.1%, wall exact, mass +0.3%, 0.20 mm RMS radius residual.

A STEP assembly imports as an assembly, with each solid assigned to a part and
materials read from the file.

---

## The application

```
tree            viewport                     properties
what it's       scaled grid, axis triad,     selected component, editable,
made of         HUD, selection highlight     with derived values
                                             cross-section table for bodies
──────────────────────────────────────────────────────────────────────────
status bar      dry / wet mass · CG · static margin · length · fineness
```

Editing is live. Drag a fin span and the geometry rebuilds, tree masses update,
and CG and static margin move with it. Only changed solids rebuild, in about
0.1 s.

| Menu | Contents |
|---|---|
| File | new, open, save, import/export STEP |
| Edit | undo, redo |
| View | frame all (`F`), grid (`G`), cutaway (`X`), mass budget (`B`), results (`R`), units, standard views (`1`/`3`/`7`) |
| Model | add Stack / FinSet / PointMass / Motor / Tank / Wing / Protuberance, delete, validate, rebuild |
| Analysis | mass properties, aerodynamics, run flight, dispersion study, design sweep, compare with RASAero, export RASAero model / aero table |
| Help | about, limitations, open log folder, check environment |

Also: undo by snapshot, mass budget panel, thrust curve editor, results panel
with run history and two-run comparison, autosave every two minutes, and a
project file holding the analysis runs alongside the vehicle.

---

## Analysis

| Stage | Produces |
|---|---|
| Mass | Meshes each part and aggregates mass, CG, full inertia tensor |
| Aero | Solves the canonical parts, in-process or by driving RASAero II |
| Flight | 6-DOF trajectory with launch rail, recovery phases, dispersion |

`AeroDatabase` is the stable interface, so CFD or wind-tunnel data can replace
either aero method without changing trajectory code.

The simulation runs 6-DOF over a 14-element state (position, velocity,
quaternion, body rates, propellant), with US Standard Atmosphere 1976 to
300 km, aerodynamic damping, a launch rail with two-button tip-off, phased
recovery, and Monte Carlo dispersion with CEP and landing ellipses.

A flight runs until the vehicle returns to its launch altitude. There is no
time cutoff, and each phase integrates until its own event fires. Pad altitude
is real: the state is integrated above sea level, and recovery altitudes read
the way an altimeter reads them. Run Flight, the design sweep and the
dispersion study share one launch sequence.

Every flight keeps a log of the force model replayed along the stored states,
giving thrust, drag, angle of attack, CG, CP, static margin, felt acceleration
and body rates per sample.

Beyond the port, the flight model adds pitch damping as a moment sum about the
current CG, roll from fin cant, jet damping and the `İω` term, a high-alpha
extension past the table edge (Jorgensen crossflow, Allen–Perkins `η`, stalled
flat plates, blended over 15°), winds aloft with Dryden turbulence, build
imperfections as dispersion variables, and per-tank propellant draining split
by mixture ratio.

---

## Where this differs from RASAero II

### Reproduced defects

| What | Effect |
|---|---|
| Two values of π | `3.14159` in most area work, `Math.PI` in some of the same expressions, so two values of A_ref differ by 8.4e-7 relative |
| Reducer closed to a point | A tail cone's CP is never stored and keeps the loop's leftover zero, so its negative normal force applies at the nose tip |
| Inclined plate at 0° | Divides by `sin(0)`, yielding ±Infinity or NaN, which propagates into the vehicle total. Reproduced in IEEE-754 explicitly |
| Roughness-cutoff step | A 48% jump in the friction coefficient, visible as a step in C_D |
| A dimensionally inconsistent fin term | Left alone: it is what RASAero prints |

### Refused

A fin can over a transition. RASAero shortens the last body tube regardless of
what lies between it and the can, leaving a hole in the body. The geometry is
not a solid. This raises an error instead.

### Corrected, opt-in and off by default

`boattail_model="corrected"` replaces the supersonic boattail branch with the
same Hoerner cube law closed on the geometric base area, and no separation
clamp. Base drag can no longer go negative, a short steep boattail is no longer
charged as a near-cylinder, and a fin set after the boattail no longer degrades
to zero. It is provisional and closes about a quarter of the gap against
measured drag.

### How faithfulness is checked

`oracle/` holds RASAero II's own Tools → Run Test output, frozen: 130
vehicle/alpha cases, 2,500 Mach points each, 28 terms per point. Every term is
compared, with zero disagreements outside RASAero's printed precision. Sixty
minimal CDX1 vehicles each isolate one branch of the solver. See
[oracle/README.md](oracle/README.md).

---

## What the flight data says

`python -m validation.scoreboard` scores against the measured-altitude flights
RASAero II ships as examples.

| | bias | mean \|err\| | within 5% |
|---|---|---|---|
| RASAero II (published predictions) | +2.20% | 5.66% | 10/21 |
| this tool, frozen table | +0.99% | 5.71% | 7/21 |
| this tool, altitude-coupled table | +0.48% | 5.38% | 8/21 |

Mean absolute error is not meaningfully better, and RASAero gets more flights
inside 5%. What changes is the bias: coupling the drag table to the altitudes
actually flown removes most of the systematic high bias, worth up to 2,000 ft
on a high flight.

A control run separates the aerodynamics from the integrator. This engine's
frozen-table prediction differs from RASAero's own by a mean of 1.50% on the
same inputs, so the 5% errors belong to the aerodynamic model rather than the
trajectory code.

Two supersonic-boattail flights are set aside. This tool does worse on them
than RASAero's published numbers (−10.8% against −2.8%) because those
predictions predate RASAero II's 2015 boattail rewrite.

`python -m validation.telemetry` reconstructs CD(Mach) from a flight's
coast-phase accelerometer log. The committed Qu8k card integrates to 121,052 ft
against a published 121,478 ft.

```
     Mach 1.98 – 2.88, 19 bins, 66.8 → 32.2 kft

     flight CD          0.28 – 0.40
     RASAero            0.44 – 0.59     mean  +65.8%   range +42% .. +103%
     corrected boattail 0.39 – 0.53     mean  +48.5%   range +28% ..  +82%
```

The vehicle flew at CD ≈ 0.30 where the model predicts 0.44 to 0.59, for any
generation of the model. That is larger than base drag alone accounts for, so
the wave, fin and friction terms are implicated. The reduction was
cross-checked against Deville's own integration (2 fps, 46 ft), the barometric
overlap (0–2% at the highest-Mach bins), and the accelerometer bias measured at
apogee (−0.016 G, the wrong sign to rescue the model), with burnout mass
corrected to the weighed 154.5 lb.

The error moves apogee only single digits because over the coast gravity took
2,579 fps and drag only 623 fps of a 3,203 fps burnout speed.

---

## Layout

| Package | Role | Origin |
|---|---|---|
| `parametric/` | Parms, cross-sections, components, lofting, import, analysis bridge | original |
| `app/` | Viewport, tree, parm and section editors | original |
| `trajectory/` | 6-DOF simulation, environment, recovery, dispersion, export | original |
| `massprops/` | STEP meshing and exact mass properties | original |
| `validation/` | Scores the aerodynamics against measured flights and telemetry | original |
| `step_to_rasaero/` | RASAero project writing and CSV ingest | original |
| `aeroengine/` | Transcription of RASAero II's aerodynamics solver | derived |
| `oracle/` | RASAero's own Run Test output, frozen per term | RASAero's output |

---

## Conventions

* Model axis is +Z aft, origin at the nose tip, so a station is a Z.
* Simulator body frame is +Y forward. The mapping is a rotation, so the inertia
  tensor transforms rather than being reordered.
* `quat_to_dcm(q)` returns body-to-inertial; inertial-to-body is its transpose.
* Thrust curves declare their reference (`"vacuum"` or `"sea_level"`).
* The engine is internally imperial, in inches, pounds and °R. SI meets English
  in one place, `aeroengine/adapters.py`.

---

## Limitations

Also on Help → Limitations, next to whatever the open vehicle triggers.

* No control. The gimbal is modelled but never commanded.
* No staging. Single stage, no separation events.
* No slosh. Propellant drains and CG moves, but the liquid is rigid, and a
  motor burns by declared geometry rather than grain regression.
* Flat-Earth gravity. Coriolis is available and off by default.
* Roll is rigid-body only. No roll-pitch resonance, lock-in analysis, or
  fin-alpha reduction at high roll rates.
* No structures or thermal. Max-Q is reported but nothing consumes it.
* Aero is component-buildup. It degrades at high alpha, on blunt bodies, and
  wherever a plume fills the base. Past the table's alpha range the model
  blends into an empirical extension, which is a correlation rather than a
  computation.
* Fins are flat plates in geometry. No airfoil section.
* The supersonic drag disagreement above is unresolved. A vehicle whose
  boattail is steeper than the 17.5° separation clamp and flies past Mach 1.2
  carries the caveat in its aero report, aerodynamics setup, flight summary and
  saved run.

---

## Development

```bash
python -m pytest                      # everything
python -m pytest -m "not slow"        # skip CAD and meshing
python -m tools.check_environment     # is this machine set up correctly
```

Errors are logged outside the repository, at `%LOCALAPPDATA%\Mudline\logs` on
Windows, reachable from Help → Open Log Folder. Send that file with a bug
report. Saved projects and recorded runs store the version and commit that
produced them.

---

## Licence

No licence is granted. `aeroengine` is derived from a commercial product and is
not mine to relicense, so a permissive licence over the whole repository would
claim a right I do not have. To use part of this in your own work, ask; the
original packages are the ones I can answer for.

RASAero II is © Charles E. Rogers and Rogers Aeroscience. This project is not
affiliated with, endorsed by, or supported by them.
