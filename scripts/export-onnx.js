// scripts/export-onnx.js
//
// npm setup and dist: export the size-free ONNX of every graph into engine/onnx through the dev
// python (engine/onnx_export.py on engine/runtime, never shipped). Generated from the committed
// weights, never committed (about 120 MB after the shared-weights tidy, which would duplicate the
// weights), shipped in the release instead of the weight files: the native host builds every
// engine from these files (about 1 s per new size) and names them from weights_tags.txt.
//
// Behaviour: every file already present is skipped by the python side, so a rerun is cheap. setup
// is NON-FATAL (a fresh clone may not have engine/runtime yet; renders need the files, so run it
// once the dev python is in). `--required` (dist) fails the build instead, so a release never ships
// without the files.

'use strict';
const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

const ENGINE = path.resolve(__dirname, '..', 'engine');
const PY = path.join(ENGINE, 'runtime', 'python.exe');
const required = process.argv.includes('--required');

function fail(msg) {
  if (required) {
    console.error(`[export-onnx] ${msg}`);
    process.exit(1);
  }
  console.warn(`[export-onnx] ${msg} - skipping; renders need engine/onnx.`);
}

if (process.platform !== 'win32') {
  fail('not Windows (this build is win64 only)');
} else if (!fs.existsSync(PY)) {
  fail(`no dev python at ${PY}`);
} else {
  const r = spawnSync(PY, [path.join(ENGINE, 'onnx_export.py')], { stdio: 'inherit', cwd: ENGINE });
  if (r.status !== 0) fail(`onnx_export.py exited with ${r.status}`);
}
