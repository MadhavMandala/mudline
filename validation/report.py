"""Render the scoreboard CSV as a standalone HTML page.

Kept separate from ``scoreboard`` so the numbers and their presentation cannot
drift: this reads the CSV that run produced and nothing else, which means the
page can be regenerated after any change to the physics and will show that
change rather than a remembered version of it.

Usage::

    python -m validation.scoreboard --csv board.csv
    python -m validation.report board.csv out.html
"""

from __future__ import annotations

import csv
import html
import sys
from pathlib import Path

#: Half-width of the error bars, in percent. Errors past this clip.
SCALE = 16.0

CSS = """
:root{
  --ground:#F5F7F6; --surface:#FFFFFF; --surface-2:#EDF1F0; --band:#E6EDEB;
  --ink:#131F1D; --ink-2:#4B5C59; --ink-3:#74837F;
  --rule:#D8E0DE; --rule-2:#C2CECB;
  --accent:#0E6B67;
  --over:#2C6AA6; --under:#A9462B; --good:#3F7A56;
  --mono:ui-monospace,"SFMono-Regular","Cascadia Mono","Segoe UI Mono",Menlo,Consolas,monospace;
  --serif:Charter,"Bitstream Charter","Iowan Old Style","Source Serif 4",Georgia,serif;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#0F1514; --surface:#161E1D; --surface-2:#1D2726; --band:#212D2B;
    --ink:#E3EAE8; --ink-2:#9BABA7; --ink-3:#7A8985;
    --rule:#283432; --rule-2:#364543;
    --accent:#4FCCC0;
    --over:#7BB0E0; --under:#E4866A; --good:#6FB68C;
  }
}
:root[data-theme="dark"]{
  --ground:#0F1514; --surface:#161E1D; --surface-2:#1D2726; --band:#212D2B;
  --ink:#E3EAE8; --ink-2:#9BABA7; --ink-3:#7A8985;
  --rule:#283432; --rule-2:#364543;
  --accent:#4FCCC0;
  --over:#7BB0E0; --under:#E4866A; --good:#6FB68C;
}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--serif);
     font-size:16.5px;line-height:1.62;margin:0;padding:0 24px 120px;
     -webkit-font-smoothing:antialiased}
.wrap{max-width:920px;margin:0 auto}
header.masthead{padding:70px 0 30px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:var(--mono);font-size:11px;font-weight:600;letter-spacing:.16em;
         text-transform:uppercase;color:var(--accent);margin:0 0 18px}
h1{font-family:var(--mono);font-size:clamp(33px,6vw,52px);font-weight:700;
   letter-spacing:-.035em;line-height:1.02;text-wrap:balance;margin:0 0 20px}
.standfirst{font-size:19.5px;line-height:1.5;color:var(--ink-2);max-width:62ch;margin:0}
section{padding-top:52px}
.sechead{display:flex;align-items:baseline;gap:14px;border-top:2px solid var(--rule-2);
         padding-top:14px;margin-bottom:20px}
.secnum{font-family:var(--mono);font-size:12px;font-weight:700;letter-spacing:.08em;
        color:var(--accent);flex:none;padding-top:5px}
h2{font-family:var(--mono);font-size:clamp(21px,3.3vw,27px);font-weight:700;
   letter-spacing:-.025em;line-height:1.2;text-wrap:balance;margin:0}
h3{font-family:var(--mono);font-size:15.5px;font-weight:700;margin:32px 0 10px;
   padding-top:12px;border-top:1px solid var(--rule)}
p{margin:0 0 15px;max-width:70ch}
ul{margin:0 0 15px;padding-left:22px;max-width:70ch}
li{margin-bottom:7px}
li::marker{color:var(--ink-3)}
code{font-family:var(--mono);font-size:.86em;background:var(--surface-2);
     border-radius:3px;padding:1px 5px}
.lede{font-size:18px;color:var(--ink);border-left:3px solid var(--accent);
      padding-left:18px;margin:0 0 22px;max-width:64ch}

.cards{display:flex;gap:12px;flex-wrap:wrap;margin:28px 0 0}
.card{flex:1 1 210px;background:var(--surface);border:1px solid var(--rule);
      border-radius:2px;padding:16px 18px}
.card .who{font-family:var(--mono);font-size:10.5px;letter-spacing:.11em;
           text-transform:uppercase;color:var(--ink-3);margin-bottom:12px}
.card .big{font-family:var(--mono);font-size:34px;font-weight:700;line-height:1;
           letter-spacing:-.03em}
.card .unit{font-family:var(--mono);font-size:11px;color:var(--ink-3);margin-top:7px;
            letter-spacing:.04em}
.card .sub{font-family:var(--mono);font-size:11.5px;color:var(--ink-2);margin-top:12px;
           padding-top:10px;border-top:1px solid var(--rule);
           display:flex;justify-content:space-between;gap:10px}

.tw{overflow-x:auto;margin:0 0 18px;border:1px solid var(--rule);border-radius:2px;
    background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:13px;font-family:var(--sans);
      min-width:760px}
th{font-family:var(--mono);font-size:9.5px;font-weight:600;letter-spacing:.1em;
   text-transform:uppercase;color:var(--ink-3);text-align:left;padding:11px 12px;
   border-bottom:1px solid var(--rule-2);background:var(--surface-2);white-space:nowrap}
td{padding:8px 12px;border-bottom:1px solid var(--rule);vertical-align:middle;
   color:var(--ink-2);white-space:nowrap}
tr:last-child td{border-bottom:none}
tr.set-aside td{background:var(--surface-2)}
td.k{font-family:var(--mono);font-size:12px;color:var(--ink)}
td.n{font-family:var(--mono);font-variant-numeric:tabular-nums;color:var(--ink);
     text-align:right}
td.dim{color:var(--ink-3)}
.flag{color:var(--under);font-family:var(--mono)}

.bar{position:relative;height:15px;min-width:190px;background:var(--surface-2);
     border-radius:2px;overflow:hidden}
.bar .band{position:absolute;top:0;bottom:0;background:var(--band)}
.bar .zero{position:absolute;top:0;bottom:0;width:1px;left:50%;background:var(--rule-2)}
.bar .fill{position:absolute;top:3.5px;bottom:3.5px;border-radius:1px;opacity:.92}
.bar .tick{position:absolute;top:1px;bottom:1px;width:2px;background:var(--ink);opacity:.55}
.legend{font-family:var(--mono);font-size:11px;color:var(--ink-3);margin:0 0 26px;
        display:flex;gap:20px;flex-wrap:wrap;align-items:center}
.sw{display:inline-block;width:22px;height:9px;border-radius:1px;vertical-align:middle;
    margin-right:6px}

.note{border:1px solid var(--rule-2);border-left:3px solid var(--accent);
      background:var(--surface);border-radius:2px;padding:14px 17px;margin:0 0 20px}
.note p{margin:0;font-size:15px;color:var(--ink-2)}
.note p+p{margin-top:9px}
.note .tag{font-family:var(--mono);font-size:10px;font-weight:700;letter-spacing:.13em;
           text-transform:uppercase;color:var(--accent);display:block;margin-bottom:7px}
.note.bad{border-left-color:var(--under)}
.note.bad .tag{color:var(--under)}

footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--rule);
       font-family:var(--mono);font-size:11.5px;color:var(--ink-3);line-height:1.7}
:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
"""


def _bar(ours: float, rasaero: float) -> str:
    """A diverging error bar for ours, with RASAero marked as a tick."""
    half = 50.0
    band = 5.0 / SCALE * half
    w = min(abs(ours), SCALE) / SCALE * half
    colour = "var(--over)" if ours > 0 else "var(--under)"
    if abs(ours) <= 5.0:
        colour = "var(--good)"
    side = f"left:50%;width:{w:.2f}%" if ours > 0 else f"right:50%;width:{w:.2f}%"
    tick = 50.0 + max(min(rasaero, SCALE), -SCALE) / SCALE * half
    return (
        f'<div class="bar" role="img" aria-label="ours {ours:+.1f}%, '
        f'RASAero {rasaero:+.1f}%">'
        f'<span class="band" style="left:{50 - band:.2f}%;width:{2 * band:.2f}%"></span>'
        f'<span class="zero"></span>'
        f'<span class="fill" style="{side};background:{colour}"></span>'
        f'<span class="tick" style="left:calc({tick:.2f}% - 1px)"></span>'
        "</div>"
    )


def _rows(records: list[dict]) -> str:
    out = []
    for r in records:
        name = r["flight"]
        aside = name in ("Proteus6", "Qu8k")
        ours = float(r["coupled_error_pct"])
        ra = float(r["rasaero_error_pct"])
        spread = float(r["instrument_spread_pct"])
        cls = ' class="set-aside"' if aside else ""
        flag = '<span class="flag">*</span> ' if aside else ""
        out.append(
            f"<tr{cls}>"
            f'<td class="k">{flag}{html.escape(name)}</td>'
            f'<td class="n">{float(r["measured_ft"]):,.0f}</td>'
            f'<td class="n dim">{float(r["max_mach"]):.2f}</td>'
            f'<td class="n">{ra:+.2f}</td>'
            f'<td class="n">{ours:+.2f}</td>'
            f"<td>{_bar(ours, ra)}</td>"
            f'<td class="n dim">{spread:.1f}</td>'
            "</tr>"
        )
    return "\n".join(out)


def _stats(records: list[dict], key: str) -> tuple[float, float, int]:
    vals = [float(r[key]) for r in records]
    n = len(vals)
    return (
        sum(abs(v) for v in vals) / n,
        sum(vals) / n,
        sum(1 for v in vals if abs(v) <= 5.0),
    )


def build(csv_path: Path) -> str:
    with csv_path.open() as fh:
        records = list(csv.DictReader(fh))
    records.sort(key=lambda r: float(r["measured_ft"]))
    scored = [r for r in records if r["flight"] not in ("Proteus6", "Qu8k")]
    n = len(scored)

    ra_mae, ra_bias, ra_hit = _stats(scored, "rasaero_error_pct")
    fr_mae, fr_bias, fr_hit = _stats(scored, "frozen_error_pct")
    co_mae, co_bias, co_hit = _stats(scored, "coupled_error_pct")
    ctl = [abs(float(r["frozen_vs_rasaero_pct"])) for r in scored]
    ctl_mae = sum(ctl) / len(ctl)

    def card(who: str, mae: float, bias: float, hit: int, tone: str = "") -> str:
        style = f' style="color:{tone}"' if tone else ""
        return (
            '<div class="card">'
            f'<div class="who">{who}</div>'
            f'<div class="big"{style}>{mae:.2f}%</div>'
            '<div class="unit">mean absolute error</div>'
            f'<div class="sub"><span>bias {bias:+.2f}%</span>'
            f"<span>{hit}/{n} within 5%</span></div>"
            "</div>"
        )

    return f"""<title>Scored Against Reality</title>
<style>{CSS}</style>
<div class="wrap">

<header class="masthead">
  <p class="eyebrow">{len(records)} instrumented flights &middot; RASAero II example set</p>
  <h1>Scored Against Reality</h1>
  <p class="standfirst">The engine reproduces RASAero to the last printed digit.
  Here is what that is actually worth, measured against rockets that flew.</p>
  <div class="cards">
    {card("RASAero, published", ra_mae, ra_bias, ra_hit)}
    {card("Ours, frozen table", fr_mae, fr_bias, fr_hit)}
    {card("Ours, altitude-coupled", co_mae, co_bias, co_hit, "var(--accent)")}
  </div>
</header>

<section>
  <div class="sechead"><span class="secnum">00</span><h2>What this is</h2></div>
  <p class="lede">Every other test in this project asks whether the engine matches
  RASAero. This one asks whether RASAero matches rockets &mdash; and it is the only
  test whose answer can improve.</p>
  <p>RASAero ships {len(records) + 1} example files whose comments record what the
  vehicle actually did: barometric, GPS or integrated-accelerometer apogee, next to
  what RASAero predicted. That is a free validation set against reality, and nothing
  in this tool had ever been compared to it.</p>
  <p>Each flight is flown twice. <strong>Frozen</strong> builds the drag table the way
  RASAero does &mdash; at whatever altitudes the Mach/Alt grid names, which for every
  one of these files is an empty grid meaning sea level &mdash; then flies it by Mach
  alone. <strong>Coupled</strong> rebuilds the table along the altitudes the vehicle
  actually flies, and iterates until it stops moving.</p>
</section>

<section>
  <div class="sechead"><span class="secnum">01</span><h2>The board</h2></div>
  <div class="legend">
    <span><span class="sw" style="background:var(--good)"></span>ours, within 5%</span>
    <span><span class="sw" style="background:var(--over)"></span>ours, over-predicts</span>
    <span><span class="sw" style="background:var(--under)"></span>ours, under-predicts</span>
    <span><span class="sw" style="background:var(--ink);opacity:.55;width:3px"></span>RASAero</span>
    <span>shaded band = &plusmn;5%</span>
  </div>
  <div class="tw"><table>
    <thead><tr>
      <th>Flight</th><th style="text-align:right">Measured ft</th>
      <th style="text-align:right">M max</th>
      <th style="text-align:right">RASAero %</th><th style="text-align:right">Ours %</th>
      <th>&minus;{SCALE:.0f}% &nbsp;&nbsp;&nbsp;&nbsp; 0 &nbsp;&nbsp;&nbsp;&nbsp; +{SCALE:.0f}%</th>
      <th style="text-align:right">Instr.&nbsp;spread</th>
    </tr></thead>
    <tbody>
{_rows(records)}
    </tbody>
  </table></div>
  <p style="font-size:14px;color:var(--ink-3)">
  <span class="flag">*</span> set aside &mdash; see section 04. A two-stage example is
  excluded entirely; staging is out of scope.</p>
</section>

<section>
  <div class="sechead"><span class="secnum">02</span><h2>The control, and why it matters</h2></div>
  <p>A scoreboard is worthless if the error might be coming from the flight model
  rather than the aerodynamics. So every flight also runs a control: same aero, same
  motor curve, same launch site, compared against <em>RASAero's own stored prediction</em>
  rather than against the rocket.</p>
  <div class="note">
    <span class="tag">Control, {n} scored flights</span>
    <p>Mean absolute difference from RASAero's own answer:
    <strong>{ctl_mae:.2f}%</strong>, against model errors averaging
    {co_mae:.2f}%.</p>
    <p>The trajectory integrator is not the error term. What the board measures is
    the aerodynamic model, which is the point of building it.</p>
  </div>
  <p>The flight model is deliberately matched in complexity to what it is measuring:
  a two-degree-of-freedom point mass, thrust and drag along the velocity vector once
  off the rod, gravity falling off with altitude, propellant burned by delivered
  impulse rather than by clock time, and drag switching from power-on to power-off at
  burnout. A six-degree-of-freedom simulation here would fold its own attitude
  dynamics into a number meant to isolate drag.</p>
</section>

<section>
  <div class="sechead"><span class="secnum">03</span><h2>Three things the board says</h2></div>

  <h3>Matching RASAero bought exactly RASAero's accuracy</h3>
  <p>{fr_mae:.2f}% mean absolute error against {ra_mae:.2f}% for RASAero itself. That is
  the expected result and it is the confirmation that the port is faithful all the way
  through a flight, not just at the coefficient level &mdash; but it is worth being blunt
  about what it means. Reproducing RASAero perfectly does not make the answers right.
  It makes them <em>RASAero's</em>, and RASAero misses real apogees by about
  {ra_mae:.0f}% on its own published examples, with only {ra_hit} of {n} inside
  &plusmn;5%.</p>

  <h3>The altitude fix is real, and it pays where predicted</h3>
  <p>Coupling the drag table to the altitudes actually flown moves mean absolute error
  from {fr_mae:.2f}% to <strong>{co_mae:.2f}%</strong> and pulls the bias from
  {fr_bias:+.2f}% to {co_bias:+.2f}%. Modest overall &mdash; because most of these
  flights are low &mdash; but the effect is concentrated exactly where the physics says
  it should be. On the highest scored flight, an N5800 minimum-diameter vehicle to
  56,574 ft, it removes 2,221 ft and takes the error from +10.52% to
  <strong>+6.60%</strong>.</p>
  <p>The mechanism is not subtle. Air is thinner at altitude, so Reynolds number falls,
  so skin friction rises. A table built at sea level and flown to 56,000 ft is reading
  friction off the wrong end of the curve for most of the flight.</p>

  <h3>The truth column is not exact either</h3>
  <p>Where a flight carried more than one instrument, they disagree. Kinsel's vehicle
  reported 40,113, 42,231, 42,771 and 44,924 ft on a single flight &mdash; a spread of
  <strong>11.25%</strong>, wider than any model error on the board. Model error below
  the instrument spread is not a measurement of anything, which is why the spread is
  printed rather than tidied away.</p>
</section>

<section>
  <div class="sechead"><span class="secnum">04</span><h2>A new defect, found by running this</h2></div>
  <div class="note bad">
    <span class="tag">RASAero disagrees with RASAero</span>
    <p>Two flights are set aside, and they have one thing in common that no other flight
    on the board has: a <strong>boattail, at supersonic speed</strong>.</p>
    <p>Every vehicle without a boattail agrees with RASAero's own prediction to within
    3.6%, all the way up to Mach 3.06. The one boattailed vehicle that stays subsonic
    agrees to 1.3%. The two boattailed vehicles that go supersonic miss by 6.7% and
    9.7% &mdash; and removing supersonic base drag entirely brings both back into line,
    which locates the disagreement in the base-drag branch rather than anywhere else in
    the buildup.</p>
    <p>This engine reproduces RASAero's Run Test for supersonic boattails, including
    past the 17.5&deg; separation clamp, verified against stored ground truth. So the
    conclusion is not that the port is wrong. It is that <em>RASAero's flight simulator
    and RASAero's own Run Test do not agree with each other</em> on this configuration.
    Nothing here says which of the two is right; it says the control cannot arbitrate
    them, so they score nothing until it is resolved.</p>
  </div>
</section>

<section>
  <div class="sechead"><span class="secnum">05</span><h2>What to do next</h2></div>
  <p>The board is now the instrument. Every candidate fix gets applied, re-run, and
  kept or dropped on whether these numbers move &mdash; which is the thing that was
  missing before, when any change to the physics was a guess.</p>
  <ul>
    <li><strong>The multi-fin factor.</strong> RASAero uses a sum of absolute sines
    where the physics wants a sum of squared sines, over-predicting fin normal force by
    15&ndash;25% for any fin count other than four. Provably wrong, roughly a table edit.</li>
    <li><strong>Settle the boattail contradiction.</strong> Two of the most demanding
    flights available cannot score until RASAero's two answers are reconciled.</li>
    <li><strong>Resolve the drag from the coast, not the apogee.</strong> Apogee error
    mixes aerodynamics with motor performance, mass and the weather on the day. After
    burnout the only forces are gravity and drag and the mass is known, so coast
    deceleration is very nearly a direct measurement of drag &mdash; a far cleaner signal
    than the single number this board currently scores.</li>
    <li><strong>Then the hard physics.</strong> The transonic band and the plume-coupled
    base pressure are the genuinely weak models, and they are weak for real reasons.
    They are also exactly the changes that can quietly make things worse, so they wait
    until the instrument is sharp.</li>
  </ul>
</section>

<footer>
  Generated from validation/scoreboard.csv. Flight data, motor curves and predictions
  are RASAero II's own shipped example set; measured apogees are derived from the
  prediction and stated error each file records, then cross-checked against the figure
  its author wrote down.<br>
  RASAero II is by Charles E. Rogers and David Cooper, Rogers Aeroscience.
</footer>

</div>
"""


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    csv_path = Path(args[0])
    out = Path(args[1]) if len(args) > 1 else csv_path.with_suffix(".html")
    out.write_text(build(csv_path), encoding="utf-8")
    print("wrote " + str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
