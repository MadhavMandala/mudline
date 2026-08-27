<#
.SYNOPSIS
  Drive RASAero II's Tools > Run Test and capture its per-term aero dump.

.DESCRIPTION
  Run Test writes the internal component breakdown -- friction, form, base,
  per-fin, protuberance, CN potential/viscous, CP -- at 0.01 Mach steps across
  all three solver regimes. That is the ground truth a reimplementation needs:
  a disagreement in total CD tells you nothing about which term is wrong.

  Unlike the Aero Plots CSV export, Run Test takes an output path in a text
  field, so there is no Save dialog to drive. It also calls the licence
  unlock internally, so it always sweeps to Mach 25 regardless of licence.

  Controls are located by their client-area origin within the dialog, which is
  fixed by RASAero 1.0.2.0's designer code. Every step verifies it happened
  rather than assuming it did, so a layout change fails loudly instead of
  producing a stale or truncated dump.

.EXAMPLE
  powershell -File rasaero_oracle.ps1 -Cdx1 tube.CDX1 -Out tube_a0.txt -Alpha 0
#>
param(
  [Parameter(Mandatory = $true)][string]$Cdx1,
  [Parameter(Mandatory = $true)][string]$Out,
  [double]$Alpha = 0,
  [double]$NozzleDiameter = 0,
  [string]$Exe = "C:\Program Files (x86)\RASAero II\RASAero II.exe",
  [int]$TimeoutSeconds = 300,
  [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot\ro_common.ps1"


$WM_SETTEXT = 0x000C
$BM_CLICK   = 0x00F5

function Get-TopWindow {
  param([int]$ProcId, [string]$Title, [int]$WaitSeconds = 20, [switch]$Exact)
  $deadline = (Get-Date).AddSeconds($WaitSeconds)
  while ((Get-Date) -lt $deadline) {
    foreach ($w in [RO]::TopWindows([uint32]$ProcId)) {
      $match = if ($Exact) { $w[1].Trim() -eq $Title } else { $w[1] -like "*$Title*" }
      if ($match) {
        return [pscustomobject]@{ Hwnd = [IntPtr]$w[0]; Title = $w[1]; X = $w[2]; Y = $w[3]; W = $w[4]; H = $w[5] }
      }
    }
    Start-Sleep -Milliseconds 300
  }
  return $null
}

# Controls are matched on their designer client origin. A tolerance is allowed
# because Windows may nudge a control by a pixel or two under some DPI
# settings, but a real layout change moves things much further than that.
function Find-Control {
  param([IntPtr]$Parent, [int]$X, [int]$Y, [string]$Label, [string]$Class, [int]$Tolerance = 6)
  $best = $null; $bestDist = [int]::MaxValue
  foreach ($c in [RO]::Children($Parent)) {
    if ($Class -and ([string]$c[1]) -notlike "*$Class*") { continue }
    $d = [Math]::Abs([int]$c[3] - $X) + [Math]::Abs([int]$c[4] - $Y)
    if ($d -lt $bestDist) { $bestDist = $d; $best = $c }
  }
  if (-not $best -or $bestDist -gt $Tolerance) {
    Write-Host "--- controls found on the dialog ---"
    foreach ($c in [RO]::Children($Parent)) {
      Write-Host ("    {0,-40} at ({1},{2})  {3}x{4}  text='{5}'" -f $c[1], $c[3], $c[4], $c[5], $c[6], $c[2])
    }
    throw "Could not locate '$Label' at client ($X,$Y); nearest was $bestDist px away. The dialog layout has changed."
  }
  return [IntPtr]$best[0]
}

$Cdx1 = (Resolve-Path $Cdx1).Path
$Out  = [System.IO.Path]::GetFullPath($Out)
$outDir = Split-Path -Parent $Out
if ($outDir -and -not (Test-Path $outDir)) { New-Item -ItemType Directory -Force $outDir | Out-Null }
if (Test-Path $Out) { Remove-Item $Out -Force }

# A surviving instance still holds the previous design, so a run against it
# would dump the wrong rocket under the new filename.
foreach ($old in @(Get-Process "RASAero II" -ErrorAction SilentlyContinue)) {
  try { $old.Kill(); $old.WaitForExit(5000) } catch {}
}
Start-Sleep -Milliseconds 500

# argv[0] goes straight to the file-open routine (ar.cs:2661), which removes
# the File > Open dialog from the sequence entirely.
$proc = Start-Process -FilePath $Exe -ArgumentList "`"$Cdx1`"" -PassThru
$main = Get-TopWindow -ProcId $proc.Id -Title "RASAero II" -WaitSeconds 30
if (-not $main) { throw "RASAero II did not open a window." }

# RASAero restores its last window placement, and a minimised window reports
# a rect of (-32000,-32000) which would put every click off-screen.
[void][RO]::ShowWindow($main.Hwnd, 9)
Start-Sleep -Milliseconds 400
[void][RO]::SetWindowPos($main.Hwnd, [IntPtr](-1), 40, 40, 1000, 640, 0x0040)
Start-Sleep -Milliseconds 500
[RO]::ForceForeground($main.Hwnd)
Start-Sleep -Milliseconds 600
$main = Get-TopWindow -ProcId $proc.Id -Title "RASAero II" -WaitSeconds 5
if (-not $main -or $main.X -lt 0) { throw "Could not place the RASAero window." }

# The Tools menu is "&Tools" (ar.cs:586) and its dropdown holds exactly one
# item, so Alt+T then Down,Enter is deterministic. Keyboard rather than mouse
# because a menu that has captured the input queue cannot be missed, whereas
# a synthesised click can land while the window is still painting.
$shell = New-Object -ComObject WScript.Shell
$runTest = $null
foreach ($attempt in 1..3) {
  [RO]::ForceForeground($main.Hwnd)
  Start-Sleep -Milliseconds 500

  $shell.SendKeys("%t")
  Start-Sleep -Milliseconds 900
  $shell.SendKeys("{DOWN}")
  Start-Sleep -Milliseconds 350
  $shell.SendKeys("{ENTER}")
  Start-Sleep -Milliseconds 1200

  $runTest = Get-TopWindow -ProcId $proc.Id -Title "Run Test" -WaitSeconds 8 -Exact
  if ($runTest) { break }
  $shell.SendKeys("{ESC}{ESC}{ESC}")
  Start-Sleep -Milliseconds 600
}
if (-not $runTest) { throw "Tools > Run Test did not open." }
[RO]::ForceForeground($runTest.Hwnd)
Start-Sleep -Milliseconds 500

# Designer coordinates from z.cs InitializeComponent. Each FarPoint FpDouble
# is a composite: an outer container at the designer point, with the real
# EDIT inset by 2px. Targeting the container does nothing, so the EDIT is
# addressed directly and the class is asserted so a layout change cannot
# silently select the wrong control.
$cAlpha   = Find-Control -Parent $runTest.Hwnd -X 92  -Y 82  -Class "EDIT"   -Label "Alpha (deg) edit"
$cNozzle1 = Find-Control -Parent $runTest.Hwnd -X 389 -Y 52  -Class "EDIT"   -Label "Nozzle 1 edit"
$cPath    = Find-Control -Parent $runTest.Hwnd -X 16  -Y 171 -Class "EDIT"   -Label "File Name edit (UsePath)"
$cBoost1  = Find-Control -Parent $runTest.Hwnd -X 275 -Y 82  -Class "BUTTON" -Label "Booster 1 checkbox"
$cBoost2  = Find-Control -Parent $runTest.Hwnd -X 275 -Y 112 -Class "BUTTON" -Label "Booster 2 checkbox"
$cRun     = Find-Control -Parent $runTest.Hwnd -X 174 -Y 237 -Class "BUTTON" -Label "Run Test button"

# RASAero reads these fields through FpDouble.Value (z.cs:650), not through
# the window text, so WM_SETTEXT is accepted and then ignored -- the run then
# silently uses whatever the dialog remembered. They have to be typed into.
#
# Synthesised mouse clicks do not move focus in this dialog at all: the caret
# stays wherever it was no matter where you click. Tab does work, and its
# cycle is deterministic, so focus is walked rather than pointed at. The walk
# tests the focused handle each step instead of counting tabs, because the
# cycle length depends on the design -- a single-stage rocket disables the
# booster fields and they drop out of the order entirely.
function Focus-Control {
  param([IntPtr]$Dialog, [IntPtr]$Target, [string]$Label, [int]$MaxTabs = 16)
  for ($i = 0; $i -lt $MaxTabs; $i++) {
    if ([RO]::FocusOf($Dialog) -eq $Target) { return }
    [System.Windows.Forms.SendKeys]::SendWait("{TAB}")
    Start-Sleep -Milliseconds 180
  }
  if ([RO]::FocusOf($Dialog) -ne $Target) {
    throw "Could not move focus to '$Label' after $MaxTabs tabs. The dialog's tab order has changed."
  }
}

function Set-FpValue {
  param([IntPtr]$Dialog, [IntPtr]$Ctl, [string]$Text, [string]$Label)
  Focus-Control -Dialog $Dialog -Target $Ctl -Label $Label
  [System.Windows.Forms.SendKeys]::SendWait("^a")
  Start-Sleep -Milliseconds 120
  [System.Windows.Forms.SendKeys]::SendWait($Text)
  Start-Sleep -Milliseconds 150
  [System.Windows.Forms.SendKeys]::SendWait("{TAB}")   # commit: FpDouble parses on validate
  Start-Sleep -Milliseconds 250
}

# The dialog persists alpha, nozzle diameters and both booster checkboxes
# between runs (z.cs:673-689), so every one of them is set explicitly rather
# than trusted to be at its default.
foreach ($cb in @($cBoost1, $cBoost2)) {
  $state = [RO]::SendMessage($cb, 0x00F0, [IntPtr]::Zero, [IntPtr]::Zero)   # BM_GETCHECK
  if ($state -ne [IntPtr]::Zero) {
    [void][RO]::SendMessage($cb, $BM_CLICK, [IntPtr]::Zero, [IntPtr]::Zero)
    Start-Sleep -Milliseconds 250
  }
}

Set-FpValue -Dialog $runTest.Hwnd -Ctl $cNozzle1 -Text ("{0:F4}" -f $NozzleDiameter) -Label "Nozzle Exit Diameter"
Set-FpValue -Dialog $runTest.Hwnd -Ctl $cAlpha   -Text ("{0}"   -f $Alpha)          -Label "Alpha (deg)"

# UsePath is a plain TextBox, whose .Text does come from the window text.
[void][RO]::SendMessage($cPath, $WM_SETTEXT, [IntPtr]::Zero, $Out)
Start-Sleep -Milliseconds 300

# The FarPoint fields cannot be read back: they are owner-drawn, and the
# inner EDIT only carries text while the control is actively being edited.
# GetWindowText on it returns a stale default, so a readback assertion here
# would be measuring nothing. The inputs are verified from the dump instead,
# after the run -- see the alpha and nozzle checks below.

[void][RO]::SendMessage($cRun, $BM_CLICK, [IntPtr]::Zero, [IntPtr]::Zero)

# The sweep is ~2500 Mach points, each a full solve, and transonic points run
# both neighbouring solvers. Completion is detected by the file size going
# quiet rather than by a fixed wait.
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$stableFor = 0
$lastSize = -1
while ((Get-Date) -lt $deadline) {
  if (Test-Path $Out) {
    $size = (Get-Item $Out).Length
    if ($size -gt 0 -and $size -eq $lastSize) {
      $stableFor += 1
      if ($stableFor -ge 4) { break }
    } else {
      $stableFor = 0
    }
    $lastSize = $size
  }
  Start-Sleep -Milliseconds 500
}
if (-not (Test-Path $Out)) { throw "RASAero did not write $Out" }
$bytes = (Get-Item $Out).Length
if ($bytes -eq 0) { throw "RASAero wrote an empty dump to $Out" }

# End-to-end verification that the dialog inputs actually took. The dump
# itself is the only honest witness: the controls cannot be read back, and a
# dump silently produced at the wrong alpha would poison every comparison
# made against it while looking perfectly valid.
$firstRecord = @(Get-Content $Out -TotalCount 400 | Where-Object { $_ -match '^\s*\d+\.\d+\s' })
if ($firstRecord.Count -lt 6) { throw "Dump has no complete Mach record to verify against." }
$line1 = ($firstRecord[0] -split '\s+') | Where-Object { $_ -ne '' }
$line3 = ($firstRecord[2] -split '\s+') | Where-Object { $_ -ne '' }

$dumpAlpha = [double]$line3[1]
if ([Math]::Abs($dumpAlpha - [Math]::Abs($Alpha)) -gt 0.005) {
  throw "Alpha did not take: asked for $Alpha, dump reports $dumpAlpha. The FarPoint field did not accept the typed value."
}
if ($NozzleDiameter -eq 0) {
  $cdOff = [double]$line1[2]; $cdOn = [double]$line1[3]
  if ([Math]::Abs($cdOff - $cdOn) -gt 1e-9) {
    throw "Nozzle diameter did not take: asked for 0, but CD power-on ($cdOn) differs from power-off ($cdOff)."
  }
}

if (-not $KeepOpen) {
  Get-Process -Id $proc.Id -ErrorAction SilentlyContinue | Stop-Process -Force
} else {
  [void][RO]::SetWindowPos($main.Hwnd, [IntPtr](-2), 0, 0, 0, 0, 0x0003)
}

Write-Output "wrote $Out ($bytes bytes, alpha=$Alpha, nozzle=$NozzleDiameter)"
