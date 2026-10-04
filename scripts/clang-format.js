// Formats our own C++ with the repo's .clang-format, the way the lint's Prettier step formats TypeScript:
// `node scripts/clang-format.js` rewrites the files in place, `--check` only reports what would change.
// Extra file arguments are formatted too (a working copy of the host sources outside the repo).
// Uses CLANG_FORMAT, else the clang-format that ships with Visual Studio (found with vswhere), else PATH.
// Out of scope: the vendored NVIDIA headers (nvofa, cuda_shim), the RTX Video SDK bridge sources and
// sl_focus_shim.h, which keep their own layout.
const { execFileSync, spawnSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const LIVE = 'engine/live/build_src';
const FILES = [
  `${LIVE}/smv-live.cpp`,
  ...fs
    .readdirSync(path.join(ROOT, LIVE))
    .filter((f) => /^smv-live-.*\.inl$/.test(f))
    .map((f) => `${LIVE}/${f}`),
  'engine/dlssnr/build_src/nr_host.cpp',
  'engine/dlssnr/build_src/nr_host.h',
  'engine/dlssnr/build_src/shim.cpp',
  'engine/dlssnr/build_src/shim_abi.h',
  'engine/dlssg/build_src/main.cpp',
  'engine/nvoffruc/build_src/nvoffruc_bridge.cpp',
  'engine/fsrfg/build_src/fsrfg_bridge.cpp',
  'engine/fsrup/build_src/fsrup_bridge.cpp',
];

function findClangFormat() {
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

const check = process.argv.includes('--check');
const extra = process.argv.slice(2).filter((a) => a !== '--check');
const exe = findClangFormat();
const files = FILES.map((f) => path.join(ROOT, f)).concat(extra);
const mode = check ? ['--dry-run', '--Werror'] : ['-i'];
const run = () =>
  spawnSync(exe, [...mode, `--style=file:${path.join(ROOT, '.clang-format')}`, ...files], { stdio: 'inherit' });
let r = run();
// a trailing comment after `};` can move again on a second pass: writing twice leaves what --check accepts
if (!check && !r.error && r.status === 0) r = run();
if (r.error) {
  console.error(
    `clang-format not found (${exe}): add Visual Studio's "C++ Clang tools for Windows" or set CLANG_FORMAT`,
  );
  process.exit(1);
}
process.exit(r.status ?? 1);
