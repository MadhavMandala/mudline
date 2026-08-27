# Mudline

An end-to-end launch-vehicle design tool: build a rocket parametrically or
import it from CAD, solve its mass properties, generate its aerodynamics, and
fly it in 6-DOF — from one application, on one model.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[cad,dev]" -c constraints.txt
.venv/Scripts/python.exe -m tools.check_environment
.venv/Scripts/python.exe -m app
```

New here? **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)** walks from a blank
window to a flown trajectory in about twenty minutes.

---

## Where the aerodynamics come from

Read this before you trust a number out of this tool.

The `aeroengine` package is a **line-by-line transcription of RASAero II's
aerodynamics solver, derived from the decompiled binary.** Not a
reimplementation from published theory, and not a clean-room rewrite — a
transcription. The comments cite the decompiled source by line number, 622
times, because that is how it was written and pretending otherwise would make
the code impossible to review.

Being explicit about the consequences:

* **RASAero II is Charles E. Rogers' work.** The aerodynamic model, the
  correlations, the curve fits and the branch structure are his. What is
  original here is the surrounding tool, the validation instrumentation, and
  the analysis of where his model departs from flight data.
* **The transcription is deliberately bug-for-bug.** It reproduces defects on
  purpose — see the table below. That is the only way to prove a port is
  faithful, and an unfaithful port is worse than none.
* **It is 15% of this repository.** `aeroengine` and `oracle` are 7,808 lines
  of 53,000. The parametric modelling, CAD import, mass properties, 6-DOF
  simulation, and validation work are original and contain no transcribed
  code — verified: not one decompiled-source citation exists outside those two
  packages.
* **If you own RASAero II, use it.** The `"rasaero-app"` aero method drives
  the real application and imports its export. It is the reference; this is
  the copy.

If you are Charles Rogers, or represent him, and you want this taken down or
changed, open an issue or email the address in the commit log. That is not a
formality — I will do it.

---

## What this tool has

### The model

Geometry follows OpenVSP's method rather than a fixed catalogue of shapes. A
body is not "a nose type plus tubes"; it is an **ordered stack of
cross-sections** along a spine. An ogive nose, a conical transition, a
boattail and a payload bulge are the same object with different sections, so
there is no shape enumeration to extend when a design needs something new.

```
VehicleModel
 ├─ Stack "nose"          lofted from XSecs → B-rep solid
 ├─ Stack "forward_tube"
 └─ Stack "motor_tube"
     ├─ FinSet "fins"     attached at a station on its parent
     └─ Motor  "motor"    propellant, thrust curve, where thrust acts
 └─ PointMass "avionics"  mass with a station, no geometry
```

Every number is a **Parm** — named, bounded, change-tracked. That gives the
editor its slider ranges for free, lets a rebuild skip untouched geometry, and
gives a design variable or optimiser a stable address to drive.

Analytic nose profiles (conical, ogive, von Kármán, elliptical, power) survive
as *generators* that emit sections, clustered toward the tip where the
curvature is. Once emitted they are ordinary, individually editable sections.

**One Stack is one manufactured part** — one material, one wall thickness. A
fibreglass nose, a thin carbon tube and a thicker aluminium motor tube are
three parts with three masses.

### Two ways in, one model

```
parametric build ──┐
                   ├──→ VehicleModel ──→ mass · aero · trajectory · dispersion
STEP import ───────┘
```

`File → Import STEP` slices a solid along its axis, measures the cross-section
at each station, and fits a Stack — so an import is **editable**, not a dead
mesh. Sections are placed by the error they remove, so a cylinder returns as
two and a curved nose as a dozen. The importer measures two radii per station:
the lateral extent gives the outer mould line, the slab volume gives an
equivalent material radius, and their disagreement is how a hollow shell is
detected and its wall thickness recovered.

Round-tripping known geometry: length exact, diameter −0.1%, wall recovered
exactly, mass +0.3%, 0.20 mm RMS radius residual.

A STEP **assembly** can be imported as the assembly it is: each solid assigned
to the part it represents, with materials and mass properties read out of the
file where the CAD package wrote them.

### The application

```
tree            viewport                     properties
what it's       scaled grid, axis triad,     selected component, editable,
made of         HUD, selection highlight     with derived values
                                             cross-section table for bodies
──────────────────────────────────────────────────────────────────────────
status bar      dry / wet mass · CG · static margin · length · fineness
```

Editing is live. Drag a fin span and the geometry rebuilds, the tree masses
update, and CG and static margin move with it — only the changed solids are
rebuilt, in about 0.1 s.

| Menu | What it does |
|---|---|
| **File** | new, open, save, import/export STEP |
| **Edit** | undo, redo |
| **View** | frame all (`F`), grid (`G`), cutaway (`X`), mass budget (`B`), results (`R`), units, standard views (`1`/`3`/`7`) |
| **Model** | add Stack / FinSet / PointMass / Motor / Tank / Wing / Protuberance, delete, validate, rebuild |
| **Analysis** | mass properties, aerodynamics, run flight, dispersion study, design sweep, compare with RASAero, export RASAero model / aero table |
| **Help** | about, limitations, open log folder, check environment |

Also in the window: an undo stack by snapshot, a mass budget panel, a thrust
curve editor, a results panel with run history and two-run comparison, autosave
every two minutes, and a project file that keeps the analysis runs alongside
the vehicle.

### Analysis chain

| Stage | What it produces |
|---|---|
| **Mass** | Meshes each part and aggregates: mass, CG, full inertia tensor |
| **Aero** | Solves the canonical parts, in-process or by driving RASAero II |
| **Flight** | 6-DOF trajectory with launch rail, recovery phases, dispersion |

The aero stage runs in-process by default. `AeroDatabase` is the stable
interface, so CFD or wind-tunnel data can replace either method without the
trajectory code changing.

### Simulation

6-DOF over a 14-element state (position, velocity, quaternion, body rates,
propellant). US Standard Atmosphere 1976 to 300 km, aerodynamic damping,
launch rail with two-button tip-off, phased recovery with drogue and main,
Monte Carlo dispersion with CEP and landing ellipses, CSV and plot export.

A flight runs until the vehicle is back at the altitude it launched from;
there is no time cutoff. Each phase is integrated until its own event fires —
apogee, main deployment, ground impact — and the result says whether it landed.

The pad altitude is real. The state is integrated above sea level, so the air
is as thin as it is at the site, the ground is the pad, and recovery altitudes
are read the way an altimeter reads them — above the pad. Run Flight, the
design sweep and the dispersion study all fly through one launch sequence, so a
swept or dispersed apogee is the one Run Flight would report for the same
design and settings.

Every flight keeps a log: the force model replayed along the stored states, so
thrust, drag, angle of attack, CG, CP, static margin, felt acceleration and
body rates are there per sample, in the results panel and the exported CSV.

Beyond the faithful port, the flight model carries things RASAero has no
equivalent for:

* **Pitch damping as a moment sum.** Every lifting part contributes its
  normal-force slope at its own centre of pressure, stored as zeroth, first
  and second moments about the nose so `Cmq` can be taken about the CG the
  vehicle has *at that instant* as propellant drains. The single-surface
  estimate this replaces cancels the nose's arm against the fins' and
  understates damping several-fold.
* **Roll from fin cant**, with `Clp` and `Cl` referenced to diameter and
  `pd/2V`.
* **Jet damping and the `İω` term** of a body losing mass, in Euler's
  equations.
* **A high-alpha extension** past the table's edge: Jorgensen crossflow on the
  body, Allen–Perkins `η`, stalled flat plates for the fins, blended in over
  15°, bounded and the right size out to 90°. A correlation, not a computation.
* **Winds aloft and Dryden turbulence**, MIL-F-8785C scales, frozen along
  altitude.
* **Build imperfections** as dispersion variables: thrust misalignment, CG
  offset, fin cant error.
* **Biprop tank draining**: each tank drains as its own settling column, split
  by mixture ratio, with a waterfall when one side runs dry — so CG and inertia
  follow the propellant that is actually left, where it actually is.

---

## Where this differs from RASAero II

The port is faithful by mandate. Every deviation is deliberate and listed here.

### Reproduced defects — bugs kept on purpose

Kept because the acceptance bar is per-term agreement, and a port that quietly
fixes things cannot prove it is a port.

| What | Effect |
|---|---|
| Two values of π | `3.14159` in most area work, `Math.PI` in some of the same expressions, so two values of A_ref differ by 8.4e-7 relative. Which appears where is load-bearing. |
| Reducer closed to a point | A tail cone's CP is never stored and keeps the loop's leftover zero — its negative normal force is applied **at the nose tip**, dragging vehicle CP forward. |
| Inclined plate at 0° | Divides by `sin(0)`, yielding ±Infinity or NaN, which propagates into the vehicle total. Reproduced in IEEE-754 explicitly, since Python would otherwise raise. |
| Roughness-cutoff step | A 48% jump in the friction coefficient, visible as a step in C_D. |
| A dimensionally inconsistent fin term | Left alone: it is what RASAero prints. |

### Refused — where reproducing would dress nonsense as agreement

**A fin can over a transition.** RASAero shortens the last body tube
regardless of what lies between it and the can, overlapping the can with that
part and leaving a hole in the body. The geometry is not a solid. It still
returns numbers. This raises an error instead.

### Corrected — opt-in, off by default

**`boattail_model="corrected"`** replaces the supersonic boattail branch.
Same Hoerner cube law, closed with the geometric base area instead of the
boattail's forward area, and with no separation clamp. Three defects go away
by construction: base drag can no longer go *negative* (the clamp's unfloored
effective diameter allowed it), a short steep boattail is no longer charged as
a near-cylinder, and a fin set listed after the boattail no longer degrades to
zero.

Provisional means provisional. Against measured drag it closes about a quarter
of the gap — see below. The default everywhere is the faithful port, and the
oracle tests always run the port.

### How faithfulness is proved

`oracle/` holds RASAero II's own **Tools → Run Test** output, frozen: 130
vehicle/alpha cases, 2,500 Mach points each, 28 terms per point — friction,
form, wave, base, each fin contribution, Reynolds number. Every one is
compared term by term, with zero disagreements outside RASAero's printed
precision. Sixty minimal CDX1 vehicles each isolate one branch of the solver.

---

## What the flight data says

This is the part worth publishing.

### Against measured apogees

`python -m validation.scoreboard` scores against the measured-altitude flights
RASAero II ships as examples, using its own motor database and launch sites.

| | bias | mean \|err\| | within 5% |
|---|---|---|---|
| RASAero II (published predictions) | **+2.20%** | 5.66% | **10/21** |
| this tool, frozen table | +0.99% | 5.71% | 7/21 |
| this tool, altitude-coupled table | **+0.48%** | **5.38%** | 8/21 |

Read honestly: **mean absolute error is not meaningfully better** — 5.66%
against 5.38% — and RASAero gets *more* flights inside 5% than this does. What
changes is the **bias**. RASAero's published predictions run systematically
high; coupling the drag table to the altitudes actually flown removes most of
that. Fly, rebuild the table where the vehicle actually went, re-fly until
apogee settles. It is worth up to 2,000 ft on a high flight, and it is a bias
correction, not an accuracy improvement.

A control run pins where the remaining error lives: this engine's frozen-table
prediction differs from RASAero's own by a mean of **1.50%** on the same
inputs. Same aero, same motor, same site — so what is left is the trajectory
integrator, and it is small next to the 5% errors. **Those errors belong to
the aerodynamic model.**

Two supersonic-boattail flights are set aside, and this tool does *worse* on
them than RASAero's published numbers (−10.8% against −2.8%). The reason is
version, not physics: those predictions predate RASAero II's 2015 boattail
rewrite, and this engine reproduces the current one. The control on those two
is 8.2%, against 1.5% everywhere else — the software generations disagree, so
apogee cannot score them.

### Against measured drag

Apogee is the integral of drag, and integrals keep secrets. So there is a
second instrument: `python -m validation.telemetry` reconstructs CD(Mach)
directly from a flight's coast-phase accelerometer log.

The committed Qu8k card integrates to **121,052 ft** against a published
**121,478 ft** — −0.35%. Its verdict on the model is the uncomfortable part:

```
     Mach 1.98 – 2.88, 19 bins, 66.8 → 32.2 kft

     flight CD          0.28 – 0.40
     RASAero            0.44 – 0.59     mean  +65.8%   range +42% .. +103%
     corrected boattail 0.39 – 0.53     mean  +48.5%   range +28% ..  +82%
```

Through Mach 2.1–2.9 the vehicle flew at CD ≈ 0.30 while the model — **any
generation of it** — predicts 0.44–0.59. That is far larger than base drag
alone can account for; the wave, fin and friction terms are implicated too.
The provisional boattail correction improves it and does not fix it.

The reduction was cross-checked five ways: it matches Deville's own
integration to 2 fps and 46 ft; the barometric overlap agrees to 0–2% at the
highest-Mach bins; the accelerometer bias measured in the apogee quiet window
is −0.016 G, small and the *wrong sign* to rescue the model; and the burnout
mass was corrected to the weighed 154.5 lb rather than spec-sheet arithmetic.

Why an error that large moves apogee only single digits: over the coast,
gravity took 2,579 fps and drag only 623 fps of a 3,203 fps burnout speed.

**The takeaway.** Matching RASAero exactly does not make RASAero right. A tool
that is bug-compatible with a model that is 40–100% high on supersonic drag is
bug-compatible with an error. This repository contains both the copy and the
measurement that indicts it, which is the honest way to ship it.

---

## Layout

| Package | Role | Origin |
|---|---|---|
| `parametric/` | The model: parms, cross-sections, components, lofting, import, analysis bridge | original |
| `app/` | The application: viewport, tree, parm and section editors | original |
| `trajectory/` | 6-DOF simulation, environment, recovery, dispersion, export | original |
| `massprops/` | STEP meshing and exact mass properties | original |
| `validation/` | Scores the aerodynamics against measured flights and telemetry | original |
| `step_to_rasaero/` | RASAero project writing and CSV ingest | original |
| `aeroengine/` | Transcription of RASAero II's aerodynamics solver | **derived** |
| `oracle/` | RASAero's own Run Test output, frozen per term | **RASAero's output** |

---

## Conventions, which are easy to get wrong

* Model axis is **+Z aft**, origin at the nose tip — so a station is a Z.
* Simulator body frame is **+Y forward**; the mapping is a *rotation*, so the
  inertia tensor transforms rather than being reordered.
* `quat_to_dcm(q)` returns **body-to-inertial**; inertial-to-body is its
  transpose.
* Thrust curves declare their reference (`"vacuum"` or `"sea_level"`).
* The engine is internally **imperial** — inches, pounds, °R — because the
  transcribed constants are. SI meets English in exactly one place,
  `aeroengine/adapters.py`.

---

## Limitations

Deliberate, so they are not mistaken for oversights. Also on **Help →
Limitations**, next to whatever the open vehicle specifically triggers.

* **No control.** The gimbal is modelled but never commanded; flights are unguided.
* **No staging.** Single stage, no separation events.
* **No slosh.** A tank's propellant drains and its CG moves, but the liquid is
  rigid; a motor burns by its declared geometry, not a grain-regression model.
* **Flat-Earth gravity.** Coriolis is available and off by default. Fine for
  sounding rockets, not for orbit.
* **Roll is rigid-body only.** No roll-pitch resonance or lock-in analysis, and
  no fin-alpha reduction at high roll rates.
* **No structures or thermal.** Max-Q is reported but nothing consumes it — no
  fin flutter, buckling, or aeroheating.
* **Aero is component-buildup.** It degrades at high alpha, on blunt bodies,
  and wherever a plume fills the base. Past the table's alpha range the model
  blends into an empirical extension that is a correlation, not a computation.
* **Fins are flat plates** in geometry; no airfoil section yet.
* **The supersonic drag disagreement above is unresolved.** A vehicle whose
  boattail is steeper than RASAero's 17.5° separation clamp and flies past
  Mach 1.2 carries the caveat in its aero report, its aerodynamics setup, its
  flight summary and its saved run.

---

## Development

```bash
python -m pytest                      # everything
python -m pytest -m "not slow"        # skip CAD and meshing
python -m tools.check_environment     # is this machine set up correctly
```

Install **with `-c constraints.txt`**. Without the pins a fresh environment
resolves newer releases that break plot export and parametric geometry —
neither of which a test suite on an already-working machine can see.

When something goes wrong in the application it is written to a log outside
the repository (`%LOCALAPPDATA%\Mudline\logs` on Windows); **Help → Open Log
Folder** goes there. Send that file with a bug report. Every saved project and
every recorded run stores the version and commit that produced it.

---

## Licence and status

**No licence is granted.** The source is published so it can be read,
checked and argued with — not so it can be reused. That is a deliberate
position rather than an oversight: `aeroengine` is derived from a commercial
product and is not mine to relicense, and issuing a permissive licence over
the whole repository would be claiming a right I do not have.

If you want to use part of this in your own work, ask. The original packages —
everything outside `aeroengine/` and `oracle/` — are the ones I can actually
say yes about.

RASAero II is © Charles E. Rogers and Rogers Aeroscience. This project is not
affiliated with, endorsed by, or supported by them.
