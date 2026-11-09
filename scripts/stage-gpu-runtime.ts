// scripts/stage-gpu-runtime.ts
//
// npm setup, dist and the first start of SmoothMyVideo.cmd: put the CUDA 13 DLLs the native host
// (engine/live/smv-live.exe) loads (the CUDA runtime, NVRTC and its builtins) into engine/gpu_runtime
// (shipped, gitignored). Two sources, the same DLLs:
//   * the dev python's wheels when engine/runtime exists (never shipped; the harness then runs the
//     CUDA version the app ships);
//   * else NVIDIA's own two wheels straight from PyPI (a wheel is a zip), pinned below by URL and
//     SHA-256, so a fresh clone needs no python at all.
//
// Behaviour: a file already staged with the same size and mtime is skipped, so a rerun is cheap; any
// other file in the folder is removed; the PyPI source is not asked again while the folder holds the
// pinned files. setup is NON-FATAL (a failed download leaves a warning); `--required` (dist) fails the
// build instead, so a release never ships without them.

import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const ENGINE = path.resolve(import.meta.dirname, '..', 'engine');
const WHEEL_BIN = path.join('nvidia', 'cu13', 'bin', 'x86_64');
const DEV_WHEELS = path.join(ENGINE, 'runtime', 'Lib', 'site-packages', WHEEL_BIN);
const OUT = path.join(ENGINE, 'gpu_runtime');
const required = process.argv.includes('--required');

// A CUDA bump = pip in the dev python AND these pins (the version, each wheel's URL and SHA-256 from
// https://pypi.org/pypi/<package>/<version>/json, each DLL's size).
const PYPI_VERSION = '13.5.37';
const PYPI_WHEELS: { url: string; sha256: string; mb: number }[] = [
  {
    url: 'https://files.pythonhosted.org/packages/35/99/e07dd9565d2e0317c1a62ff39ae9fc0542c048f74bcfc777c8be97710347/nvidia_cuda_runtime-13.5.37-py3-none-win_amd64.whl',
    sha256: 'f225d3fa5ea6bfbb58fc4979ef72375cbe661d9107f0d8a4ff22cc151525e4a9',
    mb: 3,
  },
  {
    url: 'https://files.pythonhosted.org/packages/9f/6c/241c5155199a6de55bad8862213ed64b487dc9526aae817e4af31afea1a4/nvidia_cuda_nvrtc-13.5.37-py3-none-win_amd64.whl',
    sha256: 'e8e6c10b251385707ae3abd7b4ee118de6736fd7678d0f0f4eab4d759f679c21',
    mb: 47,
  },
];
const PYPI_DLLS: [name: string, size: number][] = [
  ['cudart64_13.dll', 551024],
  ['nvrtc64_130_0.dll', 111767152],
  ['nvrtc-builtins64_135.dll', 7407216],
];

function fail(msg: string): void {
  if (required) {
    console.error(`[stage-gpu-runtime] ${msg}`);
    process.exit(1);
  }
  console.warn(`[stage-gpu-runtime] ${msg} - skipping; renders need engine/gpu_runtime.`);
}

function stageFrom(cuda: string, source: string): void {
  if (!fs.existsSync(cuda)) return fail(`no CUDA 13 DLL folder in ${source}`);
  const builtins = fs.readdirSync(cuda).filter((n) => /^nvrtc-builtins64_\d+\.dll$/i.test(n));
  const want: [string, string][] = [
    [cuda, 'cudart64_13.dll'],
    [cuda, 'nvrtc64_130_0.dll'],
    ...builtins.map((n): [string, string] => [cuda, n]),
  ];
  const missing = want.filter(([d, n]) => !fs.existsSync(path.join(d, n))).map(([, n]) => n);
  if (missing.length || !builtins.length)
    return fail(`missing in ${source}: ${missing.join(', ') || 'nvrtc-builtins64_*.dll'}`);
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
  console.log(`[stage-gpu-runtime] ${want.length} DLLs in engine/gpu_runtime from ${source} (${copied} copied)`);
}

function fromPyPI(): void {
  const staged = PYPI_DLLS.every(([n, size]) => {
    const f = path.join(OUT, n);
    return fs.existsSync(f) && fs.statSync(f).size === size;
  });
  if (staged) {
    console.log(`[stage-gpu-runtime] engine/gpu_runtime already holds CUDA ${PYPI_VERSION} - skipping.`);
    return;
  }
  const sys32 = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32');
  const curl = path.join(sys32, 'curl.exe'),
    tar = path.join(sys32, 'tar.exe');
  if (!fs.existsSync(curl) || !fs.existsSync(tar)) return fail('System32 curl.exe / tar.exe not found');
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-cuda-'));
  let error = '';
  try {
    for (const w of PYPI_WHEELS) {
      const name = path.basename(w.url);
      const whl = path.join(tmp, name);
      console.log(`[stage-gpu-runtime] downloading ${name} (~${w.mb} MB) from PyPI ...`);
      execFileSync(curl, ['-L', '--fail', '--retry', '3', '-o', whl, w.url], { stdio: 'inherit' });
      const got = crypto.createHash('sha256').update(fs.readFileSync(whl)).digest('hex');
      if (got !== w.sha256) throw new Error(`${name}: SHA-256 ${got} is not the pinned ${w.sha256}`);
      execFileSync(tar, ['-xf', whl, '-C', tmp], { stdio: 'inherit' });
    }
    stageFrom(path.join(tmp, WHEEL_BIN), `NVIDIA's PyPI wheels ${PYPI_VERSION}`);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  if (error) fail(`the CUDA DLLs could not be fetched from PyPI (${error})`);
}

if (process.platform !== 'win32') fail('not Windows (this build is win64 only)');
else if (fs.existsSync(DEV_WHEELS)) stageFrom(DEV_WHEELS, "the dev python's wheels");
else fromPyPI();
