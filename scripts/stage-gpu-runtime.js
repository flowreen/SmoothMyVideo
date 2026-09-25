// scripts/stage-gpu-runtime.js
//
// npm setup and dist: copy the CUDA 13 + TensorRT-RTX DLLs the native host (engine/live/smv-live.exe)
// loads out of the dev python's wheels (engine/runtime, never shipped) into engine/gpu_runtime
// (shipped, gitignored), plus tensorrt_rtx_version.txt: the wheel version the host bakes into every
// engine name (trt1_6_1_120), so the names stay those the dev tools (trt_lookup.py) produce.
//
// Behaviour: a file already staged with the same size and mtime is skipped, so a rerun is cheap; a stale
// DLL the wheels no longer carry is removed. setup is NON-FATAL (a fresh clone may not have
// engine/runtime yet); `--required` (dist) fails the build instead, so a release never ships
// without them.

'use strict';
const fs = require('fs');
const path = require('path');

const ENGINE = path.resolve(__dirname, '..', 'engine');
const SP = path.join(ENGINE, 'runtime', 'Lib', 'site-packages');
const OUT = path.join(ENGINE, 'gpu_runtime');
const required = process.argv.includes('--required');

function fail(msg) {
  if (required) {
    console.error(`[stage-gpu-runtime] ${msg}`);
    process.exit(1);
  }
  console.warn(`[stage-gpu-runtime] ${msg} - skipping; renders need engine/gpu_runtime.`);
}

function stage() {
  const cuda = path.join(SP, 'nvidia', 'cu13', 'bin', 'x86_64');
  const trt = path.join(SP, 'tensorrt_rtx_libs');
  if (!fs.existsSync(cuda) || !fs.existsSync(trt)) return fail(`no CUDA 13 / TensorRT-RTX wheels under ${SP}`);
  const info = fs.readdirSync(SP).find((n) => /^tensorrt_rtx_cu13-.+\.dist-info$/.test(n));
  if (!info) return fail(`no tensorrt_rtx_cu13-*.dist-info under ${SP}`);
  const builtins = fs.readdirSync(cuda).filter((n) => /^nvrtc-builtins64_\d+\.dll$/i.test(n));
  const want = [
    [cuda, 'cudart64_13.dll'],
    [cuda, 'nvrtc64_130_0.dll'],
    ...builtins.map((n) => [cuda, n]),
    [trt, 'tensorrt_rtx_1_6.dll'],
    [trt, 'tensorrt_onnxparser_rtx_1_6.dll'],
  ];
  const missing = want.filter(([d, n]) => !fs.existsSync(path.join(d, n))).map(([, n]) => n);
  if (missing.length || !builtins.length) return fail(`missing in the wheels: ${missing.join(', ') || 'nvrtc-builtins64_*.dll'}`);
  fs.mkdirSync(OUT, { recursive: true });
  const keep = new Set(want.map(([, n]) => n.toLowerCase()));
  for (const n of fs.readdirSync(OUT)) {
    if (/\.dll$/i.test(n) && !keep.has(n.toLowerCase())) fs.rmSync(path.join(OUT, n));
  }
  let copied = 0;
  for (const [d, n] of want) {
    const src = path.join(d, n), dst = path.join(OUT, n);
    if (fs.existsSync(dst)) {
      // size AND mtime (copyFileSync keeps it): a CUDA patch bump can keep a DLL's size
      const a = fs.statSync(src), b = fs.statSync(dst);
      if (a.size === b.size && a.mtimeMs === b.mtimeMs) continue;
    }
    fs.copyFileSync(src, dst);
    copied++;
  }
  const version = info.slice('tensorrt_rtx_cu13-'.length, -'.dist-info'.length);
  fs.writeFileSync(path.join(OUT, 'tensorrt_rtx_version.txt'), version + '\n');
  console.log(`[stage-gpu-runtime] ${want.length} DLLs in engine/gpu_runtime (${copied} copied), TensorRT-RTX ${version}`);
}

if (process.platform !== 'win32') fail('not Windows (this build is win64 only)');
else stage();
