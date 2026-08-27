# Getting started

From a fresh clone to a flown trajectory. About twenty minutes, most of it
waiting for one `pip install`.

If you only want to know whether this runs on your machine, do steps 1 and 2
and stop.

---

## 1. Install

You need Python 3.12 or newer. Check with `python --version`. Nothing else is
required: no CAD package, no RASAero II, no compiler.

```bash
git clone https://github.com/MadhavMandala/mudline.git
cd mudline

python -m venv .venv
```

Then, on Windows:

```bash
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -e ".[cad,dev]" -c constraints.txt
```

macOS and Linux use `.venv/bin/python` in place of `.venv\Scripts\python.exe`
throughout this document. The tool is developed and tested on Windows. The
solvers are pure Python and portable, but the 3D viewport and the RASAero
automation scripts are not exercised elsewhere.

Install with `-c constraints.txt`. It is not optional decoration. Without the
pins, pip resolves newer releases of matplotlib and cadquery that break plot
export and parametric geometry respectively. Both failures stay silent until
you hit them, and the published accuracy figures were measured on the pinned
set.

The install pulls about 1.5 GB of cadquery, OpenCASCADE, gmsh, VTK and Qt, and
takes several minutes. That is normal.

### Skipping the CAD extra

If you only want the aerodynamics, trajectory and validation tools and never
intend to build or import geometry, `pip install -e ".[dev]" -c
constraints.txt` is much smaller. The application will start, but any geometry
rebuild stops with a message telling you to install the extra.

---

## 2. Check that it worked

```bash
.venv\Scripts\python.exe -m tools.check_environment
```

You want to see `Everything checks out.` The check imports what the
application imports, renders a figure, asks your graphics driver for the
OpenGL 3.3 context the viewport needs, and compares your installed versions
against the pins.

Two results are worth understanding.

`FAILED  moderngl` or `FAILED  render a figure` means your environment did not
install as intended. Re-run the install command including `-c
constraints.txt`.

`warning  GL context` means your machine cannot give the 3D viewport an OpenGL
3.3 core context. This is common over Remote Desktop, in a virtual machine, or
with an out-of-date display driver. The tool still works. Mass properties,
aerodynamics, trajectory, dispersion and every export are unaffected. You lose
the 3D view, and the viewport says so rather than showing you a black
rectangle.

---

## 3. Open it

```bash
.venv\Scripts\python.exe -m app
```

You get a basic test rocket. Three panes:

```
  tree                  viewport                  properties
  what the vehicle      the vehicle, on a         the selected component,
  is made of            scaled grid               editable
  ─────────────────────────────────────────────────────────────────────
  status bar:  dry / wet mass · CG · static margin · length · fineness
```

Other starting points:

```bash
python -m app --boattail            # a vehicle with a bulge and a boattail
python -m app --empty               # nothing; build from scratch
python -m app vehicles/mine.json    # a saved vehicle or project
python -m app --version             # which build this is
```

Get oriented first. Press `F` to frame the vehicle. Drag with the left mouse
button to orbit, scroll to zoom. `1`, `3` and `7` give front, side and top
views. `G` toggles the grid, and `X` cuts the vehicle away so you can see
inside it.

If you work in inches and pounds, set View → Units → Imperial now. Values are
always *stored* in SI; the unit system changes only what is displayed and how
typed input is read. Be aware that the analysis dialogs are still SI-only.
Every field in them is labelled with its unit, so read the suffixes.

---

## 4. Build something

### The idea

A body is not "a nose type plus tubes". It is an ordered stack of
cross-sections along a spine, exactly as OpenVSP does it. A nose, a transition
and a boattail are the same kind of object with different sections. That is why
there is no menu of shapes to pick from, and no shape you cannot build.

One Stack is one manufactured part: one material, one wall thickness. Three
tubes of different materials are three Stacks.

### Try it

1. Model → Add → Nose Cone. It appears in the tree and in the viewport.
2. Select it. The properties pane on the right shows its parms (length, base
   diameter, wall thickness, material) each with a slider inside its own
   bounds.
3. Drag the length slider. The geometry rebuilds live, the tree masses update,
   and the CG and static margin in the status bar move with it.
4. Below the parms is the cross-section table. Those sections are what the nose
   actually is. The analytic profile you picked only *generated* them, and
   every one is individually editable, so you can pull a single station and
   make a shape no profile formula describes.
5. Model → Add → Body Tube, then Fin Set, then Motor. New parts are placed at
   the aft end of what is already there.
6. Model → Validate at any point. It lists what is wrong or missing rather than
   letting you discover it during an analysis.

Undo is a snapshot stack. `Ctrl+Z` steps back through edits, geometry rebuilds
included.

Save early with `Ctrl+S`. The project file holds the vehicle *and* the analysis
runs you have done, so a comparison survives closing the application. There is
also an autosave every two minutes, beside your document.

### Importing from CAD instead

File → Import STEP slices a solid along its axis, measures the cross-section at
each station, and fits a Stack. The import is editable: you get parms and
sections rather than a dead mesh. It detects a hollow shell and recovers its
wall thickness by comparing the lateral extent against the slab volume at each
station.

A STEP *assembly* opens a dialog that lets you assign each solid to the part it
represents. Materials and mass properties are read out of the file if the CAD
package wrote them.

---

## 5. Run the analysis chain

The three stages are in the Analysis menu and are meant to be run in order.
Each takes a few seconds on a normal vehicle.

### Mass properties

Analysis → Solve Mass Properties meshes every part and aggregates exact mass,
CG and the full inertia tensor. The report compares it against the analytic
estimate the status bar has been showing. A large disagreement means a wall
thickness or material is not what you think it is.

Open View → Mass Budget for the per-part statement a design review asks for.

### Aerodynamics

Analysis → Aerodynamics sweeps Mach and angle of attack and builds a
coefficient table.

The method matters. **RASAero (built in)** runs in-process, instantly, and
needs nothing installed; this is the transcribed engine, so read the provenance
section of the README before relying on it. **RASAero II (application)** drives
the real RASAero II and imports its export, and is only available if you have
it installed. That one is the reference.

The report ends with a CAVEAT block if your vehicle triggers a known weakness.
The one that matters most is a boattail steeper than 17.5° on a vehicle that
will go supersonic, where the model is known to disagree with flight data. Read
it. It is there because the number above it is suspect.

### Flight

Analysis → Run Flight sets up and flies a 6-DOF trajectory. The settings that
change the answer most:

| Setting | Why it matters |
|---|---|
| Launch rail length and elevation | Rail exit speed sets how much the wind can weathercock you |
| Pad altitude | Real, not sea level. Thinner air, and recovery altitudes read as an altimeter reads them |
| Wind, winds aloft, turbulence | Surface wind alone understates dispersion badly |
| Couple table to altitude | On by default. Re-solves the drag table at the altitudes actually flown, then re-flies until apogee settles. Worth up to 2,000 ft on a high flight |
| Solver accuracy | Default is tight. "Fast preview" is about 0.003% different and much quicker |

When it finishes, the trajectory is drawn in the viewport and a summary
appears. The Results panel (`R`) keeps every run.

### Beyond one flight

Design Sweep varies one parameter across a range and reports what it costs,
with a progress bar and a Cancel.

Dispersion Study flies hundreds of Monte Carlo cases with dispersed thrust,
mass, drag, wind, thrust misalignment, CG offset and fin cant, then reports CEP
and a landing ellipse. It runs across processes. It has a progress bar and a
Cancel, and stopping early gives you a study over fewer cases rather than a
failure.

---

## 6. Read the results

Open the Results panel with `R`.

Every run is listed with its metrics. Select two and it shows them side by side
with the difference, coloured by whether the change helped, and only for
quantities that have an obvious direction. Apogee going up is good. Max-Q going
up is not obviously anything, so it stays neutral.

Two things in that panel are there to stop you fooling yourself. A run whose
model has moved on is marked stale: the fingerprint covers dimensions,
materials and aerodynamic roles, so changing a material invalidates a run even
though no vertex moved. And a comparison spanning two builds of the tool says
so, in orange. If a colleague sends you a project file, a difference in the
numbers may be a difference in the code rather than the design.

Export writes the full flight log as CSV: thrust, drag, angle of attack, CG,
CP, static margin, felt acceleration, body rates and wind, per sample.

---

## 7. Check what the tool does not do

Help → Limitations lists what is not modelled: no control, no staging, no
slosh, no structures or thermal, flat-Earth gravity, flat-plate fins. It then
adds anything your specific vehicle triggers.

Read it once before you make a decision on a number out of this tool.

---

## Command-line tools

Everything below runs without the GUI.

```bash
# Score the aerodynamics against measured flights
python -m validation.scoreboard
python -m validation.scoreboard --boattail corrected

# Reconstruct CD(Mach) from a real flight's accelerometer log
python -m validation.telemetry

# Compare a flight's rotational degrees of freedom against an IMU log
python -m validation.imu

# Check this machine
python -m tools.check_environment
```

The scoreboard needs RASAero II's example files and motor database, and says
what is missing if it cannot find them. The telemetry reconstruction is
self-contained, because the Qu8k flight card is committed.

---

## When something goes wrong

The application writes a log. Help → Open Log Folder finds it, or look in
`%LOCALAPPDATA%\Mudline\logs` on Windows. Every session starts with the build,
the Python version and the library versions, and unhandled errors land there
with a full traceback. Attach it to any bug report.

Help → About gives the version and commit to quote.

Help → Check Environment runs the install check from inside the application,
for when something is wrong and you have no terminal open.

| Symptom | Likely cause |
|---|---|
| Viewport shows a message instead of the rocket | No OpenGL 3.3. See step 2. The rest of the tool is fine |
| "needs cadquery" on any geometry action | Installed without the `cad` extra |
| Plot export does nothing, or errors | Installed without `-c constraints.txt` |
| Aero method "RASAero II (application)" is greyed out | RASAero II is not installed. Use the built-in method |
| Analysis result marked stale | The vehicle changed since that run. Re-run it |
| A number looks wrong | Check Help → Limitations and the aero report's CAVEAT block first |

---

## Where to go next

The README covers what the tool is, where the aerodynamics come from, and what
the flight data says about them.

`python -m pytest -m "not slow"` runs the fast suite, if you are going to
change anything.

Read the conventions section of the README before you touch frames or inertia.
The axis and quaternion conventions are easy to get wrong and expensive to
debug.
