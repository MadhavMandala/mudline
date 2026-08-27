# Win32 interop shared by the oracle driver and its diagnostics.
Add-Type -AssemblyName System.Windows.Forms

if (-not ("RO" -as [type])) {
Add-Type @"
using System;
using System.Text;
using System.Collections.Generic;
using System.Runtime.InteropServices;

public class RO {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumWindows(EnumProc c, IntPtr l);
  [DllImport("user32.dll")] public static extern bool EnumChildWindows(IntPtr p, EnumProc c, IntPtr l);
  [DllImport("user32.dll")] public static extern uint GetWindowThreadProcessId(IntPtr h, out uint p);
  [DllImport("user32.dll", CharSet=CharSet.Auto)] public static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet=CharSet.Auto)] public static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] public static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr h, out RECT r);
  [DllImport("user32.dll")] public static extern bool ClientToScreen(IntPtr h, ref POINT p);
  [DllImport("user32.dll")] public static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] public static extern bool SetWindowPos(IntPtr h, IntPtr after, int x, int y, int w, int ht, uint f);
  [DllImport("user32.dll")] public static extern bool AttachThreadInput(uint a, uint b, bool f);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("kernel32.dll")] public static extern uint GetCurrentThreadId();
  [DllImport("user32.dll", CharSet=CharSet.Auto)] public static extern IntPtr SendMessage(IntPtr h, uint m, IntPtr w, string l);
  [DllImport("user32.dll")] public static extern IntPtr SendMessage(IntPtr h, uint m, IntPtr w, IntPtr l);
  [DllImport("user32.dll")] public static extern void mouse_event(uint f, uint dx, uint dy, uint d, IntPtr e);
  [DllImport("user32.dll")] public static extern bool SetCursorPos(int x, int y);

  public struct RECT { public int L, T, R, B; }
  public struct POINT { public int X, Y; }

  public static List<object[]> TopWindows(uint pid) {
    var found = new List<object[]>();
    EnumWindows((h, l) => {
      uint p; GetWindowThreadProcessId(h, out p);
      if (p != pid || !IsWindowVisible(h)) return true;
      var sb = new StringBuilder(512); GetWindowText(h, sb, 512);
      RECT r; GetWindowRect(h, out r);
      found.Add(new object[] { h, sb.ToString(), r.L, r.T, r.R - r.L, r.B - r.T });
      return true;
    }, IntPtr.Zero);
    return found;
  }

  // Children with their origin in the parent's CLIENT coordinates, which is
  // the coordinate system RASAero's designer code is written in.
  public static List<object[]> Children(IntPtr parent) {
    var found = new List<object[]>();
    POINT origin = new POINT(); origin.X = 0; origin.Y = 0;
    ClientToScreen(parent, ref origin);
    EnumChildWindows(parent, (h, l) => {
      var cls = new StringBuilder(256); GetClassName(h, cls, 256);
      var txt = new StringBuilder(512); GetWindowText(h, txt, 512);
      RECT r; GetWindowRect(h, out r);
      found.Add(new object[] { h, cls.ToString(), txt.ToString(),
                               r.L - origin.X, r.T - origin.Y, r.R - r.L, r.B - r.T });
      return true;
    }, IntPtr.Zero);
    return found;
  }

  // SetForegroundWindow alone is refused for a background process, which
  // leaves keystrokes going wherever the user was last looking.
  public static void ForceForeground(IntPtr h) {
    uint fgPid;
    uint fgThread = GetWindowThreadProcessId(GetForegroundWindow(), out fgPid);
    uint me = GetCurrentThreadId();
    AttachThreadInput(fgThread, me, true);
    ShowWindow(h, 9);
    SetForegroundWindow(h);
    AttachThreadInput(fgThread, me, false);
  }

  [StructLayout(LayoutKind.Sequential)]
  public struct GUITHREADINFO {
    public uint cbSize; public uint flags;
    public IntPtr hwndActive, hwndFocus, hwndCapture, hwndMenuOwner, hwndMoveSize, hwndCaret;
    public int rcL, rcT, rcR, rcB;
  }
  [DllImport("user32.dll")] public static extern bool GetGUIThreadInfo(uint idThread, ref GUITHREADINFO g);

  // Which control in the target app currently owns the keyboard. Synthesised
  // mouse clicks do not move focus in this dialog, so this is the only way to
  // know where typed input will land.
  public static IntPtr FocusOf(IntPtr anyWindowInThread) {
    uint pid; uint tid = GetWindowThreadProcessId(anyWindowInThread, out pid);
    GUITHREADINFO g = new GUITHREADINFO();
    g.cbSize = (uint)Marshal.SizeOf(typeof(GUITHREADINFO));
    if (!GetGUIThreadInfo(tid, ref g)) return IntPtr.Zero;
    return g.hwndFocus;
  }

  public static void ClickCenter(IntPtr ctl) {
    RECT r; GetWindowRect(ctl, out r);
    SetCursorPos((r.L + r.R) / 2, (r.T + r.B) / 2);
    System.Threading.Thread.Sleep(120);
    mouse_event(0x0002, 0, 0, 0, IntPtr.Zero);
    System.Threading.Thread.Sleep(50);
    mouse_event(0x0004, 0, 0, 0, IntPtr.Zero);
    System.Threading.Thread.Sleep(180);
  }
}
"@
}
