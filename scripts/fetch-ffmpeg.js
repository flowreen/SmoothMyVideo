// scripts/fetch-ffmpeg.js
//
// npm postinstall: fetch the bundled ffmpeg into engine/bin on a fresh clone.
//
// engine/bin (about 137 MB: ffmpeg.exe + ffprobe.exe + 7 shared DLLs) is gitignored - too heavy for
// the repo - so instead of committing it we download it here. Source is BtbN's public FFmpeg-Builds
// release, the win64 LGPL SHARED build: LGPL (no GPL x264/x265, keeps the "redistributable, only the
// NVIDIA driver assumed" promise), SHARED (matches engine/bin's exe+DLL layout, not the ~3x static
// build), win64. BtbN's "latest" rolls forward, which is the accepted trade for not rehosting a pinned
// copy; the engine only uses version-tolerant nvenc option forms, so newer builds keep working.
//
// Behaviour: idempotent (skips when engine/bin already has ffmpeg), Windows only, and NON-FATAL - any
// failure just warns and lets `npm install` finish. The engine already falls back to ffmpeg on PATH,
// and the GUI has a "Choose .zip" installer, so a missing download is a soft state, never a broken
// install. Uses the Windows-bundled curl.exe + tar.exe (the same System32 tools main.ts uses), so it
// adds no npm dependencies.

'use strict';
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

const FFMPEG_URL =
  'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-lgpl-shared.zip';
const ENGINE_BIN = path.resolve(__dirname, '..', 'engine', 'bin');
const SYS32 = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32');
const CURL = path.join(SYS32, 'curl.exe');
const TAR = path.join(SYS32, 'tar.exe');

const here = (name) => fs.existsSync(path.join(ENGINE_BIN, name));

function run() {
  if (process.platform !== 'win32') {
    console.log('[fetch-ffmpeg] not Windows - skipping (this build is win64 only).');
    return;
  }
  if (here('ffmpeg.exe') && here('ffprobe.exe')) {
    console.log('[fetch-ffmpeg] engine/bin already has ffmpeg - skipping.');
    return;
  }
  if (!fs.existsSync(CURL) || !fs.existsSync(TAR)) {
    console.warn('[fetch-ffmpeg] System32 curl.exe/tar.exe not found - skipping. See README "Setup".');
    return;
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-ffmpeg-'));
  const zip = path.join(tmp, 'ffmpeg.zip');
  try {
    fs.mkdirSync(ENGINE_BIN, { recursive: true });
    console.log('[fetch-ffmpeg] downloading ffmpeg (win64 lgpl-shared, ~67 MB) from BtbN ...');
    execFileSync(CURL, ['-L', '--fail', '--retry', '3', '-o', zip, FFMPEG_URL], { stdio: 'inherit' });
    console.log('[fetch-ffmpeg] extracting ...');
    execFileSync(TAR, ['-xf', zip, '-C', tmp], { stdio: 'inherit' });

    // BtbN packs everything under one top folder: <name>/bin/{ffmpeg.exe,ffprobe.exe,*.dll,...}
    const binDir = fs.readdirSync(tmp, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => path.join(tmp, e.name, 'bin'))
      .find((b) => fs.existsSync(b));
    if (!binDir) throw new Error('no bin/ folder inside the archive');

    // Take the exes + shared DLLs as a matched set from this one build (never mix DLLs across builds -
    // the exe links specific SONAME majors like avcodec-63). Skip ffplay.exe (unused).
    let copied = 0;
    for (const name of fs.readdirSync(binDir)) {
      const low = name.toLowerCase();
      if (low === 'ffplay.exe') continue;
      if (!low.endsWith('.exe') && !low.endsWith('.dll')) continue;
      const dst = path.join(ENGINE_BIN, name);
      try { if (fs.existsSync(dst)) fs.chmodSync(dst, 0o666); } catch { /* best effort */ }
      fs.copyFileSync(path.join(binDir, name), dst);
      copied++;
    }
    // Carry ffmpeg's LGPL license text (top-level in the archive) into engine/bin so a packaged
    // `npm run dist` redistributes it alongside the DLLs, as LGPL requires.
    const top = path.dirname(binDir);
    for (const lic of ['LICENSE.txt', 'LICENSE']) {
      const src = path.join(top, lic);
      if (fs.existsSync(src)) { fs.copyFileSync(src, path.join(ENGINE_BIN, 'FFMPEG-LICENSE.txt')); break; }
    }
    if (!here('ffmpeg.exe') || !here('ffprobe.exe')) {
      throw new Error('ffmpeg.exe / ffprobe.exe missing after copy');
    }
    console.log(`[fetch-ffmpeg] done - ${copied} files in engine/bin.`);
  } catch (err) {
    console.warn('[fetch-ffmpeg] could not fetch ffmpeg: ' + (err && err.message ? err.message : err));
    console.warn('[fetch-ffmpeg] npm install continues; the app will use ffmpeg on PATH if available,');
    console.warn('[fetch-ffmpeg] or add it later via the GUI "Choose .zip" button or README "Setup".');
    console.warn('[fetch-ffmpeg] Direct source: ' + FFMPEG_URL);
  } finally {
    try { fs.rmSync(tmp, { recursive: true, force: true }); } catch { /* best effort */ }
  }
}

run();
