// scripts/stage-gpu-runtime.ts
//
// npm setup, dist and the first start of SmoothMyVideo.cmd: put NVIDIA's CUDA 13 DLLs the native host
// (engine/live/smv-live.exe) loads (the CUDA runtime, NVRTC and its builtins) into engine/gpu_runtime (shipped,
// gitignored). With --sdk also the CUDA headers and import libs the host's build.bat compiles against, into
// engine/live/build_src/cuda (gitignored, never shipped). Both come out of NVIDIA's own two wheels on PyPI (a wheel
// is a zip), pinned below by URL and SHA-256, so neither running nor building needs a CUDA toolkit.
//
// Behaviour: nothing is downloaded while the folders hold the pinned version; any other file in engine/gpu_runtime
// is removed. setup is NON-FATAL (a failed download leaves a warning); `--required` (dist) fails the build instead,
// so a release never ships without the DLLs.

import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const ENGINE = path.resolve(import.meta.dirname, '..', 'engine');
const OUT = path.join(ENGINE, 'gpu_runtime');
const SDK = path.join(ENGINE, 'live', 'build_src', 'cuda');
const required = process.argv.includes('--required');
const wantSdk = process.argv.includes('--sdk');

// A CUDA bump changes these pins together: the version, each wheel's URL and SHA-256 (from
// https://pypi.org/pypi/<package>/<version>/json), each DLL's name and size.
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
// what the host's build must find in the SDK: the headers it includes by name and the two loader libs
const SDK_FILES = [
  path.join('include', 'cuda.h'),
  path.join('include', 'cuda_runtime_api.h'),
  path.join('include', 'nvrtc.h'),
  path.join('lib', 'x64', 'cuda.lib'),
  path.join('lib', 'x64', 'cudart.lib'),
];

function fail(msg: string): void {
  if (required) {
    console.error(`[stage-gpu-runtime] ${msg}`);
    process.exit(1);
  }
  console.warn(`[stage-gpu-runtime] ${msg} - skipping; renders need engine/gpu_runtime.`);
}

function dllsStaged(): boolean {
  return PYPI_DLLS.every(([n, size]) => {
    const f = path.join(OUT, n);
    return fs.existsSync(f) && fs.statSync(f).size === size;
  });
}

function sdkStaged(): boolean {
  const stamp = path.join(SDK, 'VERSION');
  return (
    fs.existsSync(stamp) &&
    fs.readFileSync(stamp, 'utf8').trim() === PYPI_VERSION &&
    SDK_FILES.every((f) => fs.existsSync(path.join(SDK, f)))
  );
}

function stageDlls(bin: string): void {
  const missing = PYPI_DLLS.filter(([n]) => !fs.existsSync(path.join(bin, n))).map(([n]) => n);
  if (missing.length) throw new Error(`missing in the wheels: ${missing.join(', ')}`);
  fs.mkdirSync(OUT, { recursive: true });
  const keep = new Set(PYPI_DLLS.map(([n]) => n.toLowerCase()));
  for (const n of fs.readdirSync(OUT)) {
    if (!keep.has(n.toLowerCase())) fs.rmSync(path.join(OUT, n), { recursive: true, force: true });
  }
  for (const [n] of PYPI_DLLS) fs.copyFileSync(path.join(bin, n), path.join(OUT, n));
  console.log(`[stage-gpu-runtime] ${PYPI_DLLS.length} DLLs in engine/gpu_runtime (CUDA ${PYPI_VERSION})`);
}

function stageSdk(cu: string): void {
  const missing = SDK_FILES.filter((f) => !fs.existsSync(path.join(cu, f)));
  if (missing.length) throw new Error(`missing in the wheels: ${missing.join(', ')}`);
  fs.rmSync(SDK, { recursive: true, force: true });
  fs.mkdirSync(SDK, { recursive: true });
  for (const d of ['include', 'lib']) fs.cpSync(path.join(cu, d), path.join(SDK, d), { recursive: true });
  fs.writeFileSync(path.join(SDK, 'VERSION'), PYPI_VERSION + '\n');
  console.log(`[stage-gpu-runtime] CUDA ${PYPI_VERSION} headers and libs in engine/live/build_src/cuda`);
}

function run(): void {
  const needDlls = !dllsStaged();
  const needSdk = wantSdk && !sdkStaged();
  if (!needDlls && !needSdk) {
    console.log(`[stage-gpu-runtime] CUDA ${PYPI_VERSION} already staged - skipping.`);
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
    const cu = path.join(tmp, 'nvidia', 'cu13');
    if (needDlls) stageDlls(path.join(cu, 'bin', 'x86_64'));
    if (needSdk) stageSdk(cu);
  } catch (e) {
    error = e instanceof Error ? e.message : String(e);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
  if (error) fail(`CUDA ${PYPI_VERSION} could not be fetched from PyPI (${error})`);
}

if (process.platform !== 'win32') fail('not Windows (this build is win64 only)');
else run();
