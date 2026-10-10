// Post-build packaging step (run after electron-builder, which is configured with the "dir" target
// so it only produces release/win-unpacked). Wrap the unpacked app in a single top-level
// "SmoothMyVideo" folder and pack that, so opening the archive shows one folder instead of loose files
// spilling into wherever the user extracts it.
//
// The archive is a .7z at 7-Zip's maximum compression (LZMA2, level 9, solid): about 37 % smaller than
// the deflate zip on this bundle. Windows 11 23H2 and later open it in Explorer, older Windows needs
// 7-Zip or NanaZip. The build machine needs 7-Zip: 7z.exe in %ProgramFiles%\7-Zip, or SMV_7Z naming it.
// The folder passed to 7-Zip becomes the archive root, which is exactly the layout we want.
import { execFileSync } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.join(import.meta.dirname, '..');
const version: string = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8')).version;
const rel = path.join(ROOT, 'release');
const unpacked = path.join(rel, 'win-unpacked');
const folder = path.join(rel, 'SmoothMyVideo');
const archiveName = `SmoothMyVideo-${version}-win.7z`;

const sevenZip = process.env.SMV_7Z || path.join(process.env.ProgramFiles || 'C:\\Program Files', '7-Zip', '7z.exe');
if (!fs.existsSync(sevenZip)) {
  console.error(`7-Zip not found at ${sevenZip}: install 7-Zip or point SMV_7Z at its 7z.exe`);
  process.exit(1);
}
if (!fs.existsSync(unpacked)) {
  console.error('release/win-unpacked not found — did electron-builder (target "dir") run first?');
  process.exit(1);
}

// win-unpacked -> SmoothMyVideo so the archive root is a single named folder.
fs.rmSync(folder, { recursive: true, force: true });
fs.renameSync(unpacked, folder);

function tree(dir: string): { files: number; bytes: number } {
  let files = 0;
  let bytes = 0;
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) {
      const t = tree(p);
      files += t.files;
      bytes += t.bytes;
    } else {
      files += 1;
      bytes += fs.statSync(p).size;
    }
  }
  return { files, bytes };
}
const staged = tree(folder);

fs.rmSync(path.join(rel, archiveName), { force: true });
console.log(
  `packing ${archiveName} (root folder: SmoothMyVideo/, ${(staged.bytes / 1e9).toFixed(2)} GB), this takes a few minutes...`,
);
execFileSync(sevenZip, ['a', '-t7z', '-m0=lzma2', '-mx=9', '-ms=on', '-mmt=on', '-bd', archiveName, 'SmoothMyVideo'], {
  cwd: rel,
  stdio: 'inherit',
});

// The archive is the deliverable; the staging folder would otherwise linger in release/ forever.
// Delete it only after 7-Zip's own test reads every file back with its CRC and finds the staged
// folder's file count and bytes (execFileSync already threw on a nonzero exit, so this is belt and
// braces against a silent truncation). GitHub refuses release assets of 2 GiB and more (a community
// report reads the cutoff as 2 GB), so the archive must stay under 2e9 bytes.
const test = execFileSync(sevenZip, ['t', '-bd', archiveName], { cwd: rel, encoding: 'utf8' });
const files = Number((/^Files: (\d+)/m.exec(test) || [])[1]);
const bytes = Number((/^Size: +(\d+)/m.exec(test) || [])[1]);
if (!/Everything is Ok/.test(test) || files !== staged.files || bytes !== staged.bytes) {
  console.error(
    `the archive test read ${files} files / ${bytes} bytes of the folder's ${staged.files} / ${staged.bytes} — ` +
      'keeping release/SmoothMyVideo/ for inspection',
  );
  process.exit(1);
}
const archiveSize = fs.statSync(path.join(rel, archiveName)).size;
if (archiveSize >= 2e9) {
  console.error(`archive is ${(archiveSize / 1e9).toFixed(2)} GB, too large for a GitHub release asset (under 2 GB)`);
  process.exit(1);
}
fs.rmSync(folder, { recursive: true, force: true });

// SHA-256 sidecar, attached beside the archive on the GitHub release so users can verify the download.
const hash = crypto.createHash('sha256');
const stream = fs.createReadStream(path.join(rel, archiveName));
stream.on('data', (d) => hash.update(d));
stream.on('end', () => {
  const digest = hash.digest('hex');
  fs.writeFileSync(path.join(rel, archiveName + '.sha256'), `${digest} *${archiveName}\n`);
  console.log(
    `done -> release/${archiveName} (${(archiveSize / 1e9).toFixed(2)} GB) + .sha256; staging folder cleaned up`,
  );
});
