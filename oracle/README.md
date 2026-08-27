# The oracle

Ground truth for `aeroengine`: RASAero II's own answers, frozen, so the
transcription can be checked term by term without RASAero being installed.

This directory answers one question, for sixty vehicles at a time:

> *This exact file, run through RASAero II, gives these exact numbers. Does
> our engine?*

---

## What is committed

```
golden/
  cdx1/*.CDX1          60 minimal vehicles, each isolating one solver branch
  *_a{alpha}.csv.gz   130 Run Test dumps, full resolution, 2,500 Mach points
  manifest.json        which dump came from which vehicle, at which alpha
vehicles_mudline/
  *.json               the same 60 vehicles as Mudline documents
```

Three descriptions of the same sixty vehicles, and you need all three for
different reasons.

The CDX1 files are what RASAero actually read, and they are the authority on
geometry. They are written field-by-field by `vehicles.py` rather than by the
tool's own exporter, deliberately, because an oracle that shared the exporter's
assumptions could not detect them. That also means they can be degenerate on
purpose: a fin can shorter than its own shoulder, or a boattail well past the
17.5° separation clamp. These are shapes chosen to drive one branch of the
solver, not to be built.

The dumps are RASAero's per-term output at 0.01 Mach steps across all three
solver regimes, covering friction, form, wave, base, each fin contribution,
protuberance, CN potential and viscous, CP and Reynolds number. A disagreement
in total C_D tells you nothing about which term is wrong, which is why the
per-term dump is the reference rather than the Aero Plots CSV.

Full resolution is kept. The whole set compresses to a few megabytes, and
decimating it would leave gaps exactly where a regression is most likely to
hide: at the branch boundaries the sweep steps across.

The Mudline models are there so you can *look* at what is being compared. Open
one in the application, orbit it, fly it. They are generated approximations and
the oracle does not use them, for the reasons in the caveat below.

---

## Running the comparison

No RASAero licence needed. Everything required is committed.

```bash
python -m pytest -m oracle           # the whole term-by-term comparison
python -m pytest aeroengine/tests/test_oracle.py -k boattail_steep
```

Each case loads its CDX1 into an engine `Design`, runs it, and compares every
term against the frozen dump at RASAero's own printed precision. A mismatch
names the term, the Mach number and both values.

---

## Looking at the vehicles

```bash
python -m app oracle/vehicles_mudline/boattail_steep.json
```

Regenerate them after changing `vehicles.py`:

```bash
python -m oracle.mudline_vehicles            # write them
python -m oracle.mudline_vehicles --report   # convert and report, write nothing
```

These are approximations. Do not cite one as evidence of what RASAero
computed; cite the CDX1 and the dump. Two things can fail to survive the
conversion.

First, vehicles that are deliberately degenerate. Mudline's parms are bounded
because real parts are, so a shape chosen to break RASAero's solver may not be
representable at all. Any that are not get reported as skipped rather than
silently clamped into a different vehicle. All sixty convert today. The one
that did not, `finsweep_12p0`, turned out to be Mudline's validator wrongly
refusing a swept fin whose tip trails past the tail, which is a configuration
real vehicles have. Fixing that was worth more than the vehicle was.

Second, nose shapes with no Mudline generator. LV-Haack and Parabolic are
approximated by von Kármán and elliptical respectively, and every model that
relies on a substitution carries it in its own description, so a reader is
never left guessing which shape they are looking at.

Inclined-plate protuberances are also approximate: RASAero takes an area and
an angle, Mudline's `Protuberance` takes an area and a shape coefficient.

---

## Regenerating from RASAero II

Only needed when adding test vehicles or checking against a new RASAero
release. It requires a licensed RASAero II 1.0.2.0 on Windows, and it takes
about twenty minutes of exclusive control of the desktop, because the driver
synthesises keystrokes and steals foreground. You cannot use the machine while
it runs.

```bash
# 1. Run every test-matrix vehicle through RASAero. Writes CDX1, the raw
#    Run Test dump, and the parsed CSV, per case and per alpha.
python -m oracle.build --out build/oracle

# 2. Freeze the result into committed reference data: gzipped CSV plus the
#    exact CDX1 that produced it, and a manifest tying them together.
python -m oracle.package

# 3. Regenerate the Mudline models to match.
python -m oracle.mudline_vehicles

# 4. Confirm the engine still agrees with the new reference.
python -m pytest -m oracle
```

`build/oracle` is gitignored. Only the frozen output in `golden/` is
committed.

### How the driver works, and how it fails

RASAero II has no command line and no scripting interface. Run Test lives
behind a dialog, so `tools/rasaero_oracle.ps1` drives it by position: controls
are located by their client-area origin within the dialog, which is fixed by
RASAero 1.0.2.0's designer code and does not move with display scaling,
because the process is DPI-unaware.

Every step verifies it actually happened rather than assuming it did. A
layout change in a different RASAero version fails loudly instead of producing
a stale or truncated dump. If you are on a version other than 1.0.2.0, expect
the driver to stop rather than to quietly lie.

One case failing is reported and skipped rather than raised, because a single
unloadable geometry should not throw away an hour of completed work.

There is a second driver, `tools/rasaero_driver.ps1`, which does the same for
the Aero Plots CSV export. That one feeds the application's `"rasaero-app"`
aero method rather than the oracle.

---

## What this proves, and what it does not

It proves the transcription is faithful: that `aeroengine` computes what
RASAero II computes, term by term, including the defects it reproduces on
purpose.

It does not prove either of them is right. For that, see
`validation/scoreboard.py`, which scores both against measured flight
apogees, and `validation/telemetry.py`, which reconstructs drag directly from a
flight's accelerometer log and finds the model 40 to 100% high through the
supersonic coast.

Faithfulness and correctness are different claims. This directory only makes
the first one.
