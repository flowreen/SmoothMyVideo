# Exact refresh rates of all ACTIVE display paths via QueryDisplayConfig - the same rational
# (numerator/denominator) Windows shows in Display settings (e.g. 359.98 Hz), which Electron's
# integer-only display.displayFrequency cannot express. Prints one "numerator denominator" line
# per active path; the app matches them to a display by nearest integer (src/main.ts).
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
public static class DC {
  [StructLayout(LayoutKind.Sequential)] public struct LUID { public uint Low; public int High; }
  [StructLayout(LayoutKind.Sequential)] public struct RATIONAL { public uint Num; public uint Den; }
  [StructLayout(LayoutKind.Sequential)] public struct PATH_SOURCE { public LUID adapter; public uint id; public uint modeIdx; public uint statusFlags; }
  [StructLayout(LayoutKind.Sequential)] public struct PATH_TARGET { public LUID adapter; public uint id; public uint modeIdx; public uint outputTech; public uint rotation; public uint scaling; public RATIONAL refresh; public uint scanline; public int available; public uint statusFlags; }
  [StructLayout(LayoutKind.Sequential)] public struct PATH_INFO { public PATH_SOURCE source; public PATH_TARGET target; public uint flags; }
  [StructLayout(LayoutKind.Explicit, Size = 64)] public struct MODE_INFO { [FieldOffset(0)] public uint infoType; }
  [DllImport("user32.dll")] public static extern int GetDisplayConfigBufferSizes(uint flags, out uint nPaths, out uint nModes);
  [DllImport("user32.dll")] public static extern int QueryDisplayConfig(uint flags, ref uint nPaths, [In, Out] PATH_INFO[] paths, ref uint nModes, [In, Out] MODE_INFO[] modes, IntPtr topology);
}
"@
$QDC_ONLY_ACTIVE_PATHS = 2
$np = 0; $nm = 0
if ([DC]::GetDisplayConfigBufferSizes($QDC_ONLY_ACTIVE_PATHS, [ref]$np, [ref]$nm) -ne 0) { exit 1 }
$paths = New-Object 'DC+PATH_INFO[]' $np
$modes = New-Object 'DC+MODE_INFO[]' $nm
if ([DC]::QueryDisplayConfig($QDC_ONLY_ACTIVE_PATHS, [ref]$np, $paths, [ref]$nm, $modes, [IntPtr]::Zero) -ne 0) { exit 1 }
for ($i = 0; $i -lt $np; $i++) {
  $r = $paths[$i].target.refresh
  if ($r.Den -gt 0) { "{0} {1}" -f $r.Num, $r.Den }
}
