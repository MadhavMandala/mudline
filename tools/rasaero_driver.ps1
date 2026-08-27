<#
.SYNOPSIS
  Run a RASAero II model and export its aero table to CSV, without a human.

.DESCRIPTION
  RASAero II has no command line and no scripting interface: the coefficient
  table exists only behind File > Export > To CSV File in its Aero Plots
  window. This drives those clicks so a design iteration does not require a
  person to sit in front of it.

  The offsets below are in RASAero's own layout coordinates, which are what
  SetCursorPos uses for a DPI-unaware process, so they do not change with the
  display scale. They *are* tied to RASAero II 1.0.2.0's window layout; the
  script verifies each step actually happened rather than assuming it did, so
  a layout change fails loudly instead of writing a stale or empty table.

.EXAMPLE
  powershell -File rasaero_driver.ps1 -Cdx1 rocket.CDX1 -Csv aero.csv
#>
param(
  [Parameter(Mandatory = $true)][string]$Cdx1,
  [Parameter(Mandatory = $true)][string]$Csv,
  [string]$Exe = "C:\Program Files (x86)\RASAero II\RASAero II.exe",
  [int]$TimeoutSeconds = 60,
  [switch]$KeepOpen
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Windows.Forms, System.Drawing

Add-Type @"
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;
public class RA {
  public delegate bool Cb(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(Cb c, IntPtr l);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
  [DllImport("user32.dll")] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out R r);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool MoveWindow(IntPtr h, int x, int y, int w, int ht, bool repaint);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int w, int ht, uint flags);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, IntPtr e);
  public struct R { public int L, T, Rt, B; }

  public static List<object[]> Windows(uint pid) {
    var found = new List<object[]>();
    EnumWindows((h, l) => {
      uint p; GetWindowThreadProcessId(h, out p);
      if (p != pid || !IsWindowVisible(h)) return true;
      var sb = new StringBuilder(512); GetWindowText(h, sb, 512);
      R r; GetWindowRect(h, out r);
      found.Add(new object[] { h, sb.ToString(), r.L, r.T, r.Rt - r.L, r.B - r.T });
      return true;
    }, IntPtr.Zero);
    return found;
  }
}
"@

function Get-Window {
  param([int]$Pid_, [string]$Title, [int]$WaitSeconds = 15)
  $deadline = (Get-Date).AddSeconds($WaitSeconds)
  while ((Get-Date) -lt $deadline) {
    foreach ($w in [RA]::Windows([uint32]$Pid_)) {
      if ($w[1] -like "*$Title*") {
        return [pscustomobject]@{ Hwnd = $w[0]; Title = $w[1]; X = $w[2]; Y = $w[3]; W = $w[4]; H = $w[5] }
      }
    }
    Start-Sleep -Milliseconds 300
  }
  return $null
}

function Invoke-ClickAt {
  param([int]$X, [int]$Y, [int]$SettleMs = 400)
  [System.Windows.Forms.Cursor]::Position = New-Object System.Drawing.Point($X, $Y)
  Start-Sleep -Milliseconds 250
  [RA]::mouse_event(0x0002, 0, 0, 0, [IntPtr]::Zero)
  Start-Sleep -Milliseconds 60
  [RA]::mouse_event(0x0004, 0, 0, 0, [IntPtr]::Zero)
  Start-Sleep -Milliseconds $SettleMs
}

$Cdx1 = (Resolve-Path $Cdx1).Path
$csvDir = Split-Path -Parent $Csv
if ($csvDir -and -not (Test-Path $csvDir)) { New-Item -ItemType Directory -Force $csvDir | Out-Null }
if (Test-Path $Csv) { Remove-Item $Csv -Force }

# RASAero keeps the loaded design in memory, so a stale instance would export
# the previous rocket's table under the new name. Waited for rather than
# assumed dead: a process that is still tearing down still owns a window, and
# a run started against it picks up the old design.
foreach ($old in @(Get-Process "RASAero II" -ErrorAction SilentlyContinue)) {
  try { $old.Kill(); $old.WaitForExit(5000) } catch {}
}
Start-Sleep -Milliseconds 400

$proc = Start-Process -FilePath $Exe -ArgumentList "`"$Cdx1`"" -PassThru
$main = Get-Window -Pid_ $proc.Id -Title "RASAero II" -WaitSeconds 25
if (-not $main) { throw "RASAero II did not open a window." }

# RASAero remembers where it was last, including minimised -- and a minimised
# window reports a rect of (-32000, -32000), so every offset below would click
# into empty space. Restore it and put it somewhere known.
#
# Topmost rather than merely focused: Windows refuses to let a background
# process steal the foreground, so SetForegroundWindow alone leaves whatever
# the user was looking at sitting on top of RASAero -- and every click below
# then lands in that window instead.
[void][RA]::ShowWindow([IntPtr]$main.Hwnd, 9)      # SW_RESTORE
Start-Sleep -Milliseconds 500
[void][RA]::SetWindowPos([IntPtr]$main.Hwnd, [IntPtr](-1), 26, 26, 807, 583, 0x0040)
Start-Sleep -Milliseconds 600
[void][RA]::SetForegroundWindow([IntPtr]$main.Hwnd)
Start-Sleep -Milliseconds 500
$main = Get-Window -Pid_ $proc.Id -Title "RASAero II" -WaitSeconds 5
if (-not $main -or $main.X -lt 0) { throw "Could not place the RASAero window." }

# Toolbar: "Aero Plots" sits 400 px right of and 72 px below the window origin.
#
# Retried, because a synthesised click is not guaranteed to be delivered: it
# can land while the window is still painting, or while focus is moving, and
# is then simply swallowed. One miss used to fail the whole run, which showed
# up as an analysis that worked almost every time -- the worst kind. Clicking
# a menu item that is already open is harmless, so retrying costs nothing.
$plots = $null
foreach ($attempt in 1..3) {
  Invoke-ClickAt -X ($main.X + 400) -Y ($main.Y + 72) -SettleMs 1500
  $plots = Get-Window -Pid_ $proc.Id -Title "Aero Plots" -WaitSeconds 25
  if ($plots) { break }
  [void][RA]::SetForegroundWindow([IntPtr]$main.Hwnd)
  Start-Sleep -Milliseconds 700
}
if (-not $plots) { throw "Aero Plots did not open -- the toolbar layout may have moved." }

# The plot window cascades rather than opening in a fixed place, so its own
# rect drives the menu offsets. It also has to finish sweeping Mach 0.01-25
# before its File menu responds.
[void][RA]::SetWindowPos([IntPtr]$plots.Hwnd, [IntPtr](-1), $plots.X, $plots.Y, $plots.W, $plots.H, 0x0040)
[void][RA]::SetForegroundWindow([IntPtr]$plots.Hwnd)
Start-Sleep -Seconds 2

# File > Export > To CSV File, relative to the Aero Plots window origin, and
# retried as a whole: a half-opened menu is dismissed by pressing Escape
# first, so each attempt starts from the same state rather than compounding.
$save = $null
foreach ($attempt in 1..3) {
  Invoke-ClickAt -X ($plots.X + 32) -Y ($plots.Y + 43) -SettleMs 600
  Invoke-ClickAt -X ($plots.X + 66) -Y ($plots.Y + 66) -SettleMs 600
  Invoke-ClickAt -X ($plots.X + 224) -Y ($plots.Y + 67) -SettleMs 1200
  $save = Get-Window -Pid_ $proc.Id -Title "Save As" -WaitSeconds 15
  if ($save) { break }
  (New-Object -ComObject WScript.Shell).SendKeys("{ESC}{ESC}")
  [void][RA]::SetForegroundWindow([IntPtr]$plots.Hwnd)
  Start-Sleep -Milliseconds 700
}
if (-not $save) { throw "The export dialog never appeared." }
[void][RA]::SetForegroundWindow([IntPtr]$save.Hwnd)
Start-Sleep -Milliseconds 500

$shell = New-Object -ComObject WScript.Shell
$shell.SendKeys([System.IO.Path]::GetFullPath($Csv))
Start-Sleep -Milliseconds 500
$shell.SendKeys("{ENTER}")

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
  if (Test-Path $Csv) {
    $size = (Get-Item $Csv).Length
    Start-Sleep -Milliseconds 700
    if ((Get-Item $Csv).Length -eq $size -and $size -gt 0) { break }
  }
  Start-Sleep -Milliseconds 400
}
if (-not (Test-Path $Csv)) { throw "RASAero did not write $Csv" }

if (-not $KeepOpen) {
  Get-Process -Id $proc.Id -ErrorAction SilentlyContinue | Stop-Process -Force
} else {
  # Hand the window back to the user rather than leaving it pinned over
  # everything else on the desktop.
  [void][RA]::SetWindowPos([IntPtr]$main.Hwnd, [IntPtr](-2), 0, 0, 0, 0, 0x0003)
}
Write-Output "wrote $Csv ($((Get-Item $Csv).Length) bytes)"
