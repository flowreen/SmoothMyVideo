// Post-build packaging step (run after electron-builder, which is configured with the "dir" target
// so it only produces release/win-unpacked). Wrap the unpacked app in a single top-level
// "SmoothMyVideo" folder and zip that, so opening the archive shows one folder instead of loose files
// spilling into wherever the user extracts it.
//
// Uses Windows' bundled bsdtar (System32\tar.exe): it writes a real ZIP (`-a` picks the format from the
// .zip extension) and handles the >4 GB archive that PowerShell's Compress-Archive chokes on. The
// folder passed to tar becomes the archive root, which is exactly the layout we want.
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const version = require('../package.json').version;
const rel = path.join(__dirname, '..', 'release');
const unpacked = path.join(rel, 'win-unpacked');
const folder = path.join(rel, 'SmoothMyVideo');
const zipName = `SmoothMyVideo-${version}-win.zip`;

if (!fs.existsSync(unpacked)) {
  console.error('release/win-unpacked not found — did electron-builder (target "dir") run first?');
  process.exit(1);
}

// win-unpacked -> SmoothMyVideo so the archive root is a single named folder.
fs.rmSync(folder, { recursive: true, force: true });
fs.renameSync(unpacked, folder);

const tar = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe');
fs.rmSync(path.join(rel, zipName), { force: true });
console.log(`packing ${zipName} (root folder: SmoothMyVideo/) — ~4 GB, this takes a minute...`);
execFileSync(tar, ['-a', '-cf', zipName, 'SmoothMyVideo'], { cwd: rel, stdio: 'inherit' });
console.log(`done -> release/${zipName}  (contains SmoothMyVideo/, plus the unpacked app at release/SmoothMyVideo/)`);
