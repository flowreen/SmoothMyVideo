// C++ static checks of the native host (engine/live/build_src/smv-live.cpp and the DLSS 5 core
// engine/dlssnr/build_src/nr_host.cpp), compiled from build.bat's own cl line into a temporary folder, so the shipped
// engine/live/smv-live.exe is never touched:
//   analyze  MSVC /analyze (the C6xxx / C26xxx / C28xxx checks; the SDK headers are external, not analysed)
//   w4       warning level 4 (the ship build uses /W3), the SDK headers external
// Needs what build.bat needs: SL_SDK, TRT_RTX_SDK and NGX_SDK (SMV_CU defaults to the dev runtime's CUDA wheel) and
// Visual Studio 2026 (vcvars64 through vswhere). Prints the warnings grouped by code and every warning line of our own
// sources; exit 0 = none in our sources. --keep DIR writes the objects and the log there instead; --root DIR reads
// engine/live/build_src and engine/dlssnr/build_src from another tree laid out like the repo (a working copy).
// Usage: node scripts/cpp-check.ts analyze|w4 [--keep DIR] [--root DIR]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const REPO = path.join(import.meta.dirname, '..');
const args = process.argv.slice(2);
const option = (name: string) => {
  const i = args.indexOf(name);
  if (i < 0) return undefined;
  const value = args[i + 1];
  if (value === undefined || value.startsWith('--')) {
    console.log(`${name} needs a folder`);
    process.exit(2);
  }
  return value;
};
const root = option('--root');
const SRC_ROOT = root ? path.resolve(root) : REPO;
const BSRC = path.join(SRC_ROOT, 'engine', 'live', 'build_src');
const NRSRC = path.join(SRC_ROOT, 'engine', 'dlssnr', 'build_src');
const OWN = ['smv-live.cpp', 'smv-live-', 'nr_host.', 'sl_focus_shim.h'];

function main(): number {
  const mode = args[0] ?? '';
  if (mode !== 'analyze' && mode !== 'w4') {
    console.log(
      'usage: node scripts/cpp-check.ts analyze|w4 [--keep DIR] [--root DIR] (needs SL_SDK, TRT_RTX_SDK, NGX_SDK)',
    );
    return 2;
  }
  const env: Record<string, string> = {};
  for (const k of ['SL_SDK', 'TRT_RTX_SDK', 'NGX_SDK']) env[k] = (process.env[k] ?? '').trim();
  const missing = Object.keys(env).filter((k) => !env[k]);
  if (missing.length) {
    console.log(`set ${missing.join(', ')} first (the same variables build.bat needs, see Building the native host)`);
    return 2;
  }
  env.SMV_CU = process.env.SMV_CU || path.join(REPO, 'engine', 'runtime', 'Lib', 'site-packages', 'nvidia', 'cu13');
  env.NRSRC = NRSRC;
  const bat = fs.readFileSync(path.join(BSRC, 'build.bat'), 'utf8');
  const m = /^cl \/nologo [\s\S]*?\/OUT:\.\.\\smv-live\.exe/m.exec(bat);
  if (!m) {
    console.log('build.bat: the cl line was not found');
    return 2;
  }
  // replacement functions: a `$` in a path is literal text, never a replacement pattern
  let cl = m[0].replace(/\^\r?\n\s*/g, ' ').replaceAll('%~dp0', () => BSRC + '\\');
  for (const [k, v] of Object.entries(env)) cl = cl.replaceAll(`%${k}%`, () => v);
  if (cl.includes('%')) {
    console.log('build.bat: an unresolved variable in the cl line: ' + cl);
    return 2;
  }
  const link = cl.indexOf(' /link ');
  if (link < 0) {
    console.log('build.bat: no /link in the cl line');
    return 2;
  }
  const keep = option('--keep');
  const work = keep ? path.resolve(keep) : fs.mkdtempSync(path.join(os.tmpdir(), 'smv_cpp_check_'));
  fs.mkdirSync(work, { recursive: true });
  const external = [
    `${env.SL_SDK}\\include`,
    `${env.TRT_RTX_SDK}\\include`,
    `${env.SMV_CU}\\include`,
    `${env.NGX_SDK}\\include`,
    path.join(BSRC, 'nvofa'),
    path.join(BSRC, 'cuda_shim'),
  ];
  const ext = external.map((d) => `/external:I "${d}"`).join(' ');
  // C6262 at 32 KB instead of 16: every host thread runs on the default 1 MB stack
  const extra = mode === 'analyze' ? '/analyze /analyze:external- /analyze:stacksize 32768' : '/W4';
  cl = cl
    .slice(0, link)
    .replace('cl /nologo', () => `cl /nologo /c /Fo${work}\\ ${extra} /external:anglebrackets /external:W0 ${ext}`);
  if (mode === 'w4') cl = cl.replaceAll(' /W3 ', ' ');
  const cmd = path.join(work, mode + '.cmd');
  const log = path.join(work, mode + '.log');
  fs.writeFileSync(
    cmd,
    [
      '@echo off',
      'setlocal',
      'where cl >nul 2>nul',
      'if not errorlevel 1 goto compile',
      'set "VSWHERE=%ProgramFiles(x86)%\\Microsoft Visual Studio\\Installer\\vswhere.exe"',
      'for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -property installationPath`) do set "VSDIR=%%i"',
      'call "%VSDIR%\\VC\\Auxiliary\\Build\\vcvars64.bat" >nul || exit /b 1',
      ':compile',
      `cd /d "${BSRC}"`,
      cl,
      'exit /b %errorlevel%',
      '',
    ].join('\r\n'),
  );
  const fd = fs.openSync(log, 'w');
  const r = spawnSync('cmd', ['/c', cmd], { stdio: ['ignore', fd, fd] });
  fs.closeSync(fd);
  if (r.error) console.log(`cmd could not start: ${r.error.message}`);
  const rc = r.status ?? 1;
  const txt = fs.readFileSync(log, 'utf8');
  const warn = [
    ...new Set(
      txt
        .split(/\r?\n/)
        .filter((ln) => /\bwarning [A-Z]+\d+/.test(ln))
        .map((ln) => ln.trim()),
    ),
  ].sort();
  const own = warn.filter((ln) => OWN.some((o) => ln.includes(o)));
  const codes = new Map<string, number>();
  for (const ln of warn) {
    const c = /\bwarning ([A-Z]+\d+)/.exec(ln)?.[1];
    if (c) codes.set(c, (codes.get(c) ?? 0) + 1);
  }
  console.log(`${mode}: compiler rc ${rc}, ${warn.length} distinct warning line(s), ${own.length} in our sources`);
  for (const [c, n] of [...codes].sort((a, b) => b[1] - a[1])) console.log(`  ${c}: ${n}`);
  for (const ln of own) console.log('  ' + ln);
  if (rc) console.log('compile failed, log: ' + log);
  else if (!keep) {
    try {
      fs.rmSync(work, { recursive: true, force: true });
    } catch {
      // a file still held by another process: the temporary folder stays, the result does not change
    }
  }
  return rc ? rc : own.length ? 1 : 0;
}

process.exit(main());
