// scripts/stage-gpu-runtime.ts
//
// npm setup and dist: copy the CUDA 13 DLLs the native host (engine/live/smv-live.exe) loads (the
// CUDA runtime, NVRTC and its builtins) out of the dev python's wheels (engine/runtime, never
// shipped) into engine/gpu_runtime (shipped, gitignored).
//
// Behaviour: a file already staged with the same size and mtime is skipped, so a rerun is cheap; any
// other file in the folder is removed. setup is NON-FATAL (a fresh clone may not have
// engine/runtime yet); `--required` (dist) fails the build instead, so a release never ships
// without them.

import fs from 'node:fs';
import path from 'node:path';

const ENGINE = path.resolve(import.meta.dirname, '..', 'engine');
const SP = path.join(ENGINE, 'runtime', 'Lib', 'site-packages');
const OUT = path.join(ENGINE, 'gpu_runtime');
const required = process.argv.includes('--required');

function fail(msg: string): void {
  if (required) {
    console.error(`[stage-gpu-runtime] ${msg}`);
    process.exit(1);
  }
  console.warn(`[stage-gpu-runtime] ${msg} - skipping; renders need engine/gpu_runtime.`);
}

function stage(): void {
  const cuda = path.join(SP, 'nvidia', 'cu13', 'bin', 'x86_64');
  if (!fs.existsSync(cuda)) return fail(`no CUDA 13 wheels under ${SP}`);
  const builtins = fs.readdirSync(cuda).filter((n) => /^nvrtc-builtins64_\d+\.dll$/i.test(n));
  const want: [string, string][] = [
    [cuda, 'cudart64_13.dll'],
    [cuda, 'nvrtc64_130_0.dll'],
    ...builtins.map((n): [string, string] => [cuda, n]),
  ];
  const missing = want.filter(([d, n]) => !fs.existsSync(path.join(d, n))).map(([, n]) => n);
  if (missing.length || !builtins.length)
    return fail(`missing in the wheels: ${missing.join(', ') || 'nvrtc-builtins64_*.dll'}`);
  fs.mkdirSync(OUT, { recursive: true });
  const keep = new Set(want.map(([, n]) => n.toLowerCase()));
  for (const n of fs.readdirSync(OUT)) {
    if (!keep.has(n.toLowerCase())) fs.rmSync(path.join(OUT, n), { recursive: true, force: true });
  }
  let copied = 0;
  for (const [d, n] of want) {
    const src = path.join(d, n),
      dst = path.join(OUT, n);
    if (fs.existsSync(dst)) {
      // size AND mtime (copyFileSync keeps it): a CUDA patch bump can keep a DLL's size
      const a = fs.statSync(src),
        b = fs.statSync(dst);
      if (a.size === b.size && a.mtimeMs === b.mtimeMs) continue;
    }
    fs.copyFileSync(src, dst);
    copied++;
  }
  console.log(`[stage-gpu-runtime] ${want.length} DLLs in engine/gpu_runtime (${copied} copied)`);
}

if (process.platform !== 'win32') fail('not Windows (this build is win64 only)');
else stage();
