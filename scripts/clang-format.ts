// Formats our own C++ with the repo's .clang-format, the way the lint's Prettier step formats TypeScript:
// `node scripts/clang-format.ts` rewrites the files in place.
// Extra file arguments are formatted too (a working copy of the host sources outside the repo).
// Uses CLANG_FORMAT, else the clang-format that ships with Visual Studio (found with vswhere), else PATH.
// Scope: every .cpp / .h / .inl directly in an engine/*/build_src folder, so a new bridge is formatted without a list
// edit. Out of scope (they keep their own layout): the vendored NVIDIA headers in subfolders (nvofa, cuda_shim), the
// RTX Video SDK bridge sources (engine/rtxvideo) and sl_focus_shim.h.
import { execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.join(import.meta.dirname, '..');
const SKIP_DIRS = new Set(['rtxvideo']);
const SKIP_FILES = new Set(['sl_focus_shim.h']);
const FILES = fs
  .readdirSync(path.join(ROOT, 'engine'), { withFileTypes: true })
  .filter((d) => d.isDirectory() && !SKIP_DIRS.has(d.name))
  .map((d) => `engine/${d.name}/build_src`)
  .filter((dir) => fs.existsSync(path.join(ROOT, dir)))
  .flatMap((dir) =>
    fs
      .readdirSync(path.join(ROOT, dir))
      .filter((f) => /\.(cpp|h|inl)$/.test(f) && !SKIP_FILES.has(f))
      .map((f) => `${dir}/${f}`),
  );

function findClangFormat(): string {
  if (process.env.CLANG_FORMAT) return process.env.CLANG_FORMAT;
  const vswhere = path.join(
    process.env['ProgramFiles(x86)'] || 'C:\\Program Files (x86)',
    'Microsoft Visual Studio',
    'Installer',
    'vswhere.exe',
  );
  if (fs.existsSync(vswhere)) {
    const vs = execFileSync(vswhere, ['-latest', '-property', 'installationPath'], { encoding: 'utf8' }).trim();
    const exe = path.join(vs, 'VC', 'Tools', 'Llvm', 'x64', 'bin', 'clang-format.exe');
    if (fs.existsSync(exe)) return exe;
  }
  return 'clang-format';
}

const extra = process.argv.slice(2);
const exe = findClangFormat();
const files = FILES.map((f) => path.join(ROOT, f)).concat(extra);
const run = () =>
  spawnSync(exe, ['-i', `--style=file:${path.join(ROOT, '.clang-format')}`, ...files], { stdio: 'inherit' });
let r = run();
// a trailing comment after `};` can move again on a second pass: a second write settles it
if (!r.error && r.status === 0) r = run();
if (r.error) {
  console.error(
    `clang-format not found (${exe}): add Visual Studio's "C++ Clang tools for Windows" or set CLANG_FORMAT`,
  );
  process.exit(1);
}
process.exit(r.status ?? 1);
