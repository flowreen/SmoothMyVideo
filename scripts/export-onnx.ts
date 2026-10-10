// scripts/export-onnx.ts
//
// npm setup and dist: export the size-free ONNX of every graph into engine/onnx through the dev
// python (engine/onnx_export.py on engine/runtime, never shipped). Generated from the committed
// weights, never committed (about 120 MB after the shared-weights tidy, which would duplicate the
// weights), shipped in the release instead of the weight files: the native host builds every
// engine from these files (about 1 s per new size) and names them from weights_tags.txt.
//
// Behaviour: while the export stamp (the sources, the exporter packages, the weights) is unchanged,
// every file already present is skipped by the python side, so a rerun is cheap; a changed stamp
// exports every graph again, so run it after any change under engine (no number to bump). setup
// is NON-FATAL (a fresh clone may not have engine/runtime yet; renders need the files, so run it
// once the dev python is in). `--required` (dist) fails the build instead, so a release never ships
// without the files.

import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const ENGINE = path.resolve(import.meta.dirname, '..', 'engine');
const PY = path.join(ENGINE, 'runtime', 'python.exe');
const required = process.argv.includes('--required');

function fail(msg: string): void {
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
