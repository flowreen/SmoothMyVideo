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
console.log(`packing ${zipName} (root folder: SmoothMyVideo/), ~4 GB, this takes a minute...`);
execFileSync(tar, ['-a', '-cf', zipName, 'SmoothMyVideo'], { cwd: rel, stdio: 'inherit' });

// The zip is the deliverable; the multi-GB staging folder would otherwise linger in release/
// forever. Delete it only after sanity-checking that tar really produced a plausible archive
// (execFileSync already threw on a nonzero exit, so this is belt and braces against a silent
// truncation): anything under 1 GB cannot be a real bundle with the 5.7 GB runtime inside.
const zipSize = fs.statSync(path.join(rel, zipName)).size;
if (zipSize < 1e9) {
  console.error(`zip is only ${(zipSize / 1e6).toFixed(0)} MB — keeping release/SmoothMyVideo/ for inspection`);
  process.exit(1);
}
fs.rmSync(folder, { recursive: true, force: true });

// SHA-256 sidecar for the GitHub release page (the zip itself is too large for GitHub, so the
// checksum file is what lets users verify the SourceForge download). Streamed: the zip is ~4 GB.
const crypto = require('crypto');
const hash = crypto.createHash('sha256');
const stream = fs.createReadStream(path.join(rel, zipName));
stream.on('data', (d) => hash.update(d));
stream.on('end', () => {
  const digest = hash.digest('hex');
  fs.writeFileSync(path.join(rel, zipName + '.sha256'), `${digest} *${zipName}\n`);
  console.log(`done -> release/${zipName} (${(zipSize / 1e9).toFixed(2)} GB) + .sha256; staging folder cleaned up`);
});
