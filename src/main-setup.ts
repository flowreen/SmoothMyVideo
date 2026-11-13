// Main process: readiness and one-click installs (RTX Video, NVIDIA Video Frame Generation, Smooth Motion, DLSS 4.5
// support, DLSS 5, AnimeJaNai, the Dolby Vision and HDR10+ tools, Intel XeSS).
import { ipcMain, dialog, shell } from 'electron';
import { execFile, execFileSync } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import * as os from 'os';
import * as crypto from 'crypto';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';
import { ENGINE } from './main-paths';
import { win } from './main-window';

// A release archive a setup row downloads in one click (installPinnedZip): its fields move together.
type PinnedZip = { version: string; url: string; page: string; zipSha256: string; zipBytes: number };
// A superseded runtime build, known by its file's size and SHA256 (retireOldRuntimes).
type Retired = { label: string; bytes: number; sha256: string };

// --- RTX Video runtime: readiness + one-click install --------------------------------------------
// The opt-in RTX features (VSR / TrueHDR) run through the compiled CUDA bridge (rtxvideo_cuda.dll,
// which ships with the app) plus NVIDIA's two RTX Video feature DLLs. Those feature DLLs are NVIDIA
// proprietary and NON-redistributable, so they are never bundled: one click downloads NVIDIA's own
// nvidia-vfx package (a Python wheel, which is a zip) from NVIDIA's package index, checks its SHA256
// before anything is extracted and copies the two DLLs into engine/rtxvideo. A feature is "ready" only
// when the bridge AND its feature DLL are present there. Bump version, url, hash and size together when
// moving to a newer package, and RTX_PINNED and RTX_LABEL with them.
const RTX_DIR = path.join(ENGINE, 'rtxvideo');
const NVVFX_DL: PinnedZip = {
  version: '0.2.0.0',
  url: 'https://pypi.nvidia.com/nvidia-vfx/nvidia_vfx-0.2.0.0-cp312-abi3-win_amd64.whl',
  page: 'https://pypi.nvidia.com/nvidia-vfx/',
  zipSha256: 'cebaa12411f7e22b39293205c5c4d07b8073d14ca1e0e7727ee4375207a36445',
  zipBytes: 435807359,
};
// The pair NVVFX_DL installs, each file by its size and SHA256. An installed file that is not it is removed at
// app start (retireOldRuntimes), so the RTX row offers the download: SMV runs this pair only.
const RTX_PINNED: Record<string, { bytes: number; sha256: string }> = {
  'nvngx_vsr.dll': {
    bytes: 59120752,
    sha256: '0cc98643a76c05ca6d73ec81a898d545b18e5c080667d72854a1bfaebd56e1ed',
  },
  'nvngx_truehdr.dll': {
    bytes: 3692144,
    sha256: '90e1d0884276f3e9b5f17756df46667c7b16aab0413bb56e27de7c8a0cf10d75',
  },
};
const RTX_FILES: Array<[string, string]> = Object.keys(RTX_PINNED).map((file) => [file, file]);
// the pair as the retire lines name it
const RTX_LABEL = 'RTX VSR 1.8.3 + RTX HDR 1.2.3';
// NvOFFRUC ("NVIDIA Smooth Motion"): our bridge (nvoffruc_bridge.dll) is locally built and ships,
// but NvOFFRUC.dll + cudart64_110.dll are NVIDIA proprietary and user-installed from the Optical
// Flow SDK .zip, which NVIDIA serves behind a developer login.
const NVOFFRUC_DIR = path.join(ENGINE, 'nvoffruc');
const NVOFFRUC_DLLS = ['NvOFFRUC.dll', 'cudart64_110.dll'];
const OF_SDK_URL = 'https://developer.nvidia.com/opticalflow/download';
// "DLSS 4.5" (DLSS Frame Generation): fully bundled, no installer - the offline host exe
// (engine/dlssg/dlssg2f.exe, built from build_src) plus NVIDIA's redistributable frame generation
// DLL (its license alongside). Readiness = all files present (False means a broken
// install); actual GPU/driver support surfaces as a clear engine error at render time.
const DLSSG_DIR = path.join(ENGINE, 'dlssg');
const DLSSG_FILES = ['dlssg2f.exe', 'nvngx_dlssg.dll'];
// "NVIDIA DLSS 5" (Neural Rendering): the caller shim engine/dlssnr/nvngx.dll (built from build_src;
// the native host loads the runtime through it) ships, but the NR runtime nvngx_dlssnr.dll is NOT shipped: NVIDIA
// publishes no download (no SDK, no driver copy, no NVIDIA App override as of 2026-09), the only
// NVIDIA copy sits inside NBA 2K27. So the app offers a one-click download of the community build
// every DLSS 5 tool pulls from (RankFTW/rhi-repo release assets, the RenoDX author's RTX 40 + 50
// rebuild), pinned to one asset and verified twice (zip SHA256 from the GitHub API digest, then the
// DLL inside). nvngx_dlss.dll is the DLSS SR runtime; the NGX core only warns when it is absent
// (probe-verified), so it is reported, never required.
const DLSSNR_DIR = path.join(ENGINE, 'dlssnr');
const DLSSNR_HOST = ['nvngx.dll'];
const DLSSNR_RUNTIME = 'nvngx_dlssnr.dll';
const DLSSNR_SR = 'nvngx_dlss.dll';
// The one-click source: the pinned rhi-repo asset, and the ONE hash this app holds: the
// SHA256 of the nvngx_dlssnr.dll inside it (probe-verified on the host). The download refuses to install any other
// DLL. Bump tag, asset and hash together when moving to a
// newer build and add the replaced one to DLSSNR_RETIRED (the zip's own digest is at
// https://api.github.com/repos/RankFTW/rhi-repo/releases).
const DLSSNR_DL = {
  url: 'https://github.com/RankFTW/rhi-repo/releases/download/dlssnr-310.9.16/nvngx_dlssnr_310.9.16.zip',
  page: 'https://github.com/RankFTW/rhi-repo/releases/tag/dlssnr-310.9.16',
  dllSha256: 'cdfa6c1ec784867182164c7836f324f087b5c9f374d0c9fe46f54999feabf3ed',
  label: '310.9.16 (rhi-repo, RTX 40 + 50)',
  zipBytes: 192378839,
};
// Superseded one-click builds (size + SHA256 of nvngx_dlssnr.dll). An installed copy, downloaded or dropped, is
// removed at app start so the DLSS 5 row offers DLSSNR_DL's build: SMV offers one build only.
const DLSSNR_RETIRED: Retired[] = [
  {
    label: '310.8.SF-v2',
    bytes: 165830144,
    sha256: '6eb209e764f39872625debd6abaf45e2bb6322f6f270f781f70c059ae30b3927',
  },
];
// whether the file `p` is exactly this build; the size test comes first, so only a same-size file gets hashed
function isBuild(p: string, bytes: number, sha256: string): boolean {
  try {
    return fs.statSync(p).size === bytes && sha256File(p) === sha256;
  } catch {
    return false;
  }
}
/** Remove the superseded runtime file `p`: the line says it went, or that a running host still holds it (it
 *  stays and is retried next start). */
function retire(what: string, p: string, label: string, offer: string): string {
  try {
    fs.rmSync(p);
  } catch {
    return `${what}: the retired ${label} runtime is in use, retried next start`;
  }
  return `${what}: removed the retired ${label} runtime, the download offers ${offer}`;
}
/** Remove every installed superseded runtime, one line for each: DLSS 5's when it is a build of DLSSNR_RETIRED
 *  (a runtime the user dropped in stays), an RTX Video file when it is not RTX_PINNED's. */
export function retireOldRuntimes(): string[] {
  const lines: string[] = [];
  const nr = path.join(DLSSNR_DIR, DLSSNR_RUNTIME);
  const old = DLSSNR_RETIRED.find((r) => isBuild(nr, r.bytes, r.sha256));
  if (old) lines.push(retire('DLSS 5', nr, old.label, DLSSNR_DL.label));
  for (const [file, pin] of Object.entries(RTX_PINNED)) {
    const p = path.join(RTX_DIR, file);
    if (fileExists(p) && !isBuild(p, pin.bytes, pin.sha256)) lines.push(retire('RTX Video', p, file, RTX_LABEL));
  }
  return lines;
}
export function sha256File(p: string): string {
  const h = crypto.createHash('sha256');
  const fd = fs.openSync(p, 'r');
  try {
    const buf = Buffer.alloc(1 << 20);
    for (;;) {
      const n = fs.readSync(fd, buf, 0, buf.length, null);
      if (n <= 0) break;
      h.update(buf.subarray(0, n));
    }
  } finally {
    fs.closeSync(fd);
  }
  return h.digest('hex');
}
const SYS_TAR = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'tar.exe');
export const fileExists = (p: string) => {
  try {
    return fs.existsSync(p);
  } catch {
    return false;
  }
};

type InstallResult = { ok: boolean; error?: string; copied: string[] };

// Every file under root whose name matches one of `names` (case-insensitive; a release zip can
// nest its payload, and the __MACOSX ._ copies fall out because their basename differs).
function findFilesNamed(root: string, names: string[]): string[] {
  const want = new Set(names.map((n) => n.toLowerCase()));
  const hits: string[] = [];
  const stack = [root];
  while (stack.length) {
    const d = stack.pop()!;
    let ents: fs.Dirent[];
    try {
      ents = fs.readdirSync(d, { withFileTypes: true });
    } catch {
      continue;
    }
    for (const e of ents) {
      const p = path.join(d, e.name);
      if (e.isDirectory()) stack.push(p);
      else if (want.has(e.name.toLowerCase())) hits.push(p);
    }
  }
  return hits;
}

// The zip-extract-and-copy flow behind a runtime the user picks as a file (the NvOFFRUC pair). `source`
// is a .zip (only the wanted members are extracted, with Windows' bundled bsdtar, into a temp dir that
// is removed again), an extracted folder, or a picked file (its folder is searched). `prefer` breaks
// ties when an archive ships several copies of a name (the Optical Flow SDK's win32 / win64). Every
// wanted name must be found, else nothing is copied.
function installFiles(
  source: string,
  names: string[],
  destDir: string,
  opts: { what: string; where: string; prefer?: RegExp },
): InstallResult {
  try {
    fs.mkdirSync(destDir, { recursive: true });
  } catch {
    /* exists */
  }
  let tmp: string | null = null;
  const srcFiles: string[] = [];
  try {
    let searchDir = source;
    if (/\.zip$/i.test(source)) {
      tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-install-'));
      try {
        execFileSync(SYS_TAR, ['-xf', source, '-C', tmp, ...names.map((n) => '*' + n)]);
      } catch {
        // bsdtar's member patterns are case-sensitive and fail when one matches nothing (a
        // DOVI_TOOL.EXE member, a renamed SDK layout): take the whole archive and search it
        execFileSync(SYS_TAR, ['-xf', source, '-C', tmp]);
      }
      searchDir = tmp;
    } else if (!fs.statSync(source).isDirectory()) {
      searchDir = path.dirname(source);
    }
    const found = findFilesNamed(searchDir, names);
    for (const n of names) {
      const all = found.filter((f) => path.basename(f).toLowerCase() === n.toLowerCase());
      const hit = (opts.prefer && all.find((f) => opts.prefer!.test(f))) || all[0];
      if (hit) srcFiles.push(hit);
    }
    if (srcFiles.length < names.length)
      return { ok: false, error: names.join(' / ') + ' not found in ' + opts.what, copied: [] };
    const copied: string[] = [];
    try {
      for (const f of srcFiles) {
        const dest = path.join(destDir, path.basename(f));
        // Overwrite any existing copy (a newer release replaces the old files), even one a previous
        // extraction left read-only; clearing the flag first avoids an EPERM on copy.
        try {
          if (fileExists(dest)) fs.chmodSync(dest, 0o666);
        } catch {
          /* best effort */
        }
        fs.copyFileSync(f, dest);
        copied.push(path.basename(f));
      }
    } catch (e) {
      return {
        ok: false,
        error: 'Could not write ' + opts.where + ' (' + String(e) + '). If a render is running, stop it and try again.',
        copied,
      };
    }
    return { ok: true, copied };
  } catch (e) {
    return { ok: false, error: String(e), copied: [] };
  } finally {
    if (tmp)
      try {
        fs.rmSync(tmp, { recursive: true, force: true });
      } catch {
        /* temp cleanup best effort */
      }
  }
}

// NvOFFRUC.dll + cudart64_110.dll into engine/nvoffruc, beside the locally built bridge (the SDK
// ships win32 + win64 copies; take the x64 build).
const installFruc = (source: string): InstallResult =>
  installFiles(source, NVOFFRUC_DLLS, NVOFFRUC_DIR, {
    what: 'the selected Optical Flow SDK',
    where: 'the FRUC DLLs in engine/nvoffruc',
    prefer: /win64/i,
  });

ipcMain.handle('rtx-ready', () => {
  const bridge = fileExists(path.join(RTX_DIR, 'rtxvideo_cuda.dll'));
  return {
    vsr: bridge && fileExists(path.join(RTX_DIR, 'nvngx_vsr.dll')),
    hdr: bridge && fileExists(path.join(RTX_DIR, 'nvngx_truehdr.dll')),
    bridge,
    dir: RTX_DIR,
    ver: NVVFX_DL.version,
    mb: toolMb(NVVFX_DL),
  };
});

ipcMain.handle('rtx-open-download', () => {
  shell.openExternal(NVVFX_DL.page);
  return true;
});

ipcMain.handle('rtx-download', () => installPinnedZip(NVVFX_DL, RTX_FILES, RTX_DIR, 'rtx-progress'));

// NVIDIA Video Frame Generation: the four files of NVIDIA's Video Effects runtime the host opens in engine/nvvfx
// (never bundled). They come from the same nvidia-vfx package as the RTX Video pair: one click downloads it,
// checks its SHA256 and copies the four files. "Ready" = all four present.
const NVVFX_DIR = path.join(ENGINE, 'nvvfx');
const VFG_FILES: Array<[string, string]> = [
  'NVVideoEffects.dll',
  'NVCVImage.dll',
  'nvVFXVideoFrameGeneration.dll',
  'cudart64_12.dll',
].map((file) => [file, file]);

ipcMain.handle('nvvfg-ready', () => ({
  ready: VFG_FILES.every(([file]) => fileExists(path.join(NVVFX_DIR, file))),
  dir: NVVFX_DIR,
  ver: NVVFX_DL.version,
  mb: toolMb(NVVFX_DL),
}));

ipcMain.handle('nvvfg-download', () => installPinnedZip(NVVFX_DL, VFG_FILES, NVVFX_DIR, 'nvvfg-progress'));

// "NVIDIA Smooth Motion" (NvOFFRUC) runtime: ready only when our bridge AND NvOFFRUC.dll are present.
// "DLSS 4.5" (DLSS Frame Generation): bundled, so this only guards against a broken install.
// DLSS 4.5 runs on RTX 40 / 50 GPUs with Windows' hardware-accelerated GPU scheduling on (HwSchMode 2); read once a run.
// The renderer makes it the default model only when both hold
let dlssgSupport: Promise<{ gpu: string; hags: boolean; supported: boolean }> | null = null;
function dlssgSupportOnce(): Promise<{ gpu: string; hags: boolean; supported: boolean }> {
  if (!dlssgSupport)
    dlssgSupport = new Promise((resolve) => {
      execFile(
        'nvidia-smi',
        ['--query-gpu=name', '--format=csv,noheader'],
        { timeout: 5000, windowsHide: true },
        (e1, o1) => {
          const gpu = e1 ? '' : String(o1).split(/\r?\n/)[0].trim();
          const key = 'HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers';
          execFile('reg', ['query', key, '/v', 'HwSchMode'], { timeout: 5000, windowsHide: true }, (e2, o2) => {
            const hags = !e2 && /HwSchMode\s+REG_DWORD\s+0x2\b/i.test(String(o2));
            resolve({ gpu, hags, supported: hags && /RTX\s*(40|50)\d{2}\b|\bAda\b|Blackwell/i.test(gpu) });
          });
        },
      );
    });
  return dlssgSupport;
}

ipcMain.handle('dlssg-ready', async () => {
  const missing = DLSSG_FILES.filter((f) => !fileExists(path.join(DLSSG_DIR, f)));
  const s = await dlssgSupportOnce();
  return { ready: missing.length === 0, missing, dir: DLSSG_DIR, gpu: s.gpu, hags: s.hags, supported: s.supported };
});

// "NVIDIA DLSS 5" (Neural Rendering): ready when the shipped caller shim AND the user-supplied
// runtime are present in engine/dlssnr (see the DLSSNR_* constants).
ipcMain.handle('dlssnr-ready', () => {
  const host = DLSSNR_HOST.every((f) => fileExists(path.join(DLSSNR_DIR, f)));
  const runtime = fileExists(path.join(DLSSNR_DIR, DLSSNR_RUNTIME));
  const sr = fileExists(path.join(DLSSNR_DIR, DLSSNR_SR));
  return {
    ready: host && runtime,
    host,
    runtime,
    sr,
    dir: DLSSNR_DIR,
    file: DLSSNR_RUNTIME,
    srFile: DLSSNR_SR,
    download: { page: DLSSNR_DL.page, mb: Math.round(DLSSNR_DL.zipBytes / 1048576) },
  };
});

// One-click: download the pinned rhi-repo zip into a temp dir (progress to the renderer as
// 'dlssnr-progress'), extract with bsdtar, verify the DLL hash, copy it into engine/dlssnr. A
// mismatch aborts before the copy, so a replaced release asset can never install anything.
ipcMain.handle('dlssnr-download', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-dlssnr-'));
  const zip = path.join(tmp, 'runtime.zip');
  const progress = (received: number, total: number, stage: string) => {
    try {
      if (win && !win.webContents.isDestroyed()) win.webContents.send('dlssnr-progress', { received, total, stage });
    } catch {
      /* window gone */
    }
  };
  try {
    const res = await fetch(DLSSNR_DL.url, { headers: { 'user-agent': 'SmoothMyVideo' } });
    if (!res.ok || !res.body) return { ok: false, error: `download failed: HTTP ${res.status}`, copied: [] };
    const total = parseInt(res.headers.get('content-length') || '0', 10) || DLSSNR_DL.zipBytes;
    let received = 0;
    let last = 0;
    const src = Readable.fromWeb(res.body as never);
    src.on('data', (chunk: Buffer) => {
      received += chunk.length;
      if (received - last > 2 * 1048576 || received === total) {
        last = received;
        progress(received, total, 'download');
      }
    });
    await pipeline(src, fs.createWriteStream(zip));
    progress(received, total, 'verify');
    execFileSync(SYS_TAR, ['-xf', zip, '-C', tmp]);
    const found = findDvBin(tmp, DLSSNR_RUNTIME);
    if (!found)
      return { ok: false, error: DLSSNR_RUNTIME + ' not in the downloaded zip, nothing installed', copied: [] };
    const dh = sha256File(found);
    if (dh !== DLSSNR_DL.dllSha256)
      return { ok: false, error: `runtime checksum mismatch (${dh.slice(0, 12)}...), nothing installed`, copied: [] };
    fs.mkdirSync(DLSSNR_DIR, { recursive: true });
    const dest = path.join(DLSSNR_DIR, DLSSNR_RUNTIME);
    try {
      if (fileExists(dest)) fs.chmodSync(dest, 0o666);
    } catch {
      /* best effort */
    }
    fs.copyFileSync(found, dest);
    return { ok: true, copied: [DLSSNR_RUNTIME], sha256: dh, known: DLSSNR_DL.label };
  } catch (e) {
    return { ok: false, error: String(e), copied: [] };
  } finally {
    try {
      fs.rmSync(tmp, { recursive: true, force: true });
    } catch {
      /* temp cleanup best effort */
    }
  }
});

ipcMain.handle('dlssnr-open-download', () => {
  shell.openExternal(DLSSNR_DL.page);
  return true;
});

// "AnimeJaNai HD", Restore's second model: the author's 2x ONNX (CC BY-NC-SA 4.0, so never bundled: package.json's
// filter drops engine/animejanai). One click downloads the pinned release archive, checks its SHA256, extracts the
// one model with Windows' bsdtar, checks the model's SHA256 and has the host convert it into the .pkl the app keeps
// like every other model's weights (--convert-restore-model: the same weights for the kernels, the frames identical);
// the UI says when the installed .pkl is not the verified one. The host gets it as --restore-model.
const AJ_FILE = '2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx';
const AJ_PKL = AJ_FILE.replace(/\.onnx$/, '.pkl');
// the host's conversion of the author's ONNX into the .pkl (written only when it reads back as the ONNX's weights)
function ajConvert(onnx: string, pkl: string): boolean {
  try {
    execFileSync(path.join(ENGINE, 'live', 'smv-live.exe'), ['--convert-restore-model', onnx, pkl], {
      stdio: 'ignore',
      windowsHide: true,
    });
    return fileExists(pkl);
  } catch {
    return false;
  }
}
// engine/animejanai, the model's own folder like engine/artcnn. An install that holds the author's ONNX (in
// engine/restore, the folder's earlier name, or here) gets it moved here and converted once, at start, so an update
// keeps the download; a read-only or locked install keeps the old file and the setup row offers the download.
function ajDir(): string {
  const dir = path.join(ENGINE, 'animejanai');
  const oldDir = path.join(ENGINE, 'restore');
  const old = path.join(oldDir, AJ_FILE);
  const onnx = path.join(dir, AJ_FILE);
  const pkl = path.join(dir, AJ_PKL);
  try {
    if (fs.existsSync(old) && !fs.existsSync(onnx)) {
      fs.mkdirSync(dir, { recursive: true });
      fs.renameSync(old, onnx);
    }
    if (fs.existsSync(oldDir) && fs.readdirSync(oldDir).length === 0) fs.rmdirSync(oldDir);
    if (fs.existsSync(onnx) && (fs.existsSync(pkl) || ajConvert(onnx, pkl))) fs.rmSync(onnx);
  } catch {
    /* left as it is */
  }
  return dir;
}
const AJ_DIR = ajDir();
const AJ_DL = {
  url: 'https://github.com/the-database/mpv-upscale-2x_animejanai/releases/download/3.9.0/mpv-upscale-2x_animejanai-overlay-3.9.0.7z',
  page: 'https://github.com/the-database/mpv-upscale-2x_animejanai/releases/tag/3.9.0',
  licence: 'https://creativecommons.org/licenses/by-nc-sa/4.0/',
  member: 'animejanai/onnx/' + AJ_FILE,
  archiveSha256: '8933e1003dc9bdf5bc9cd4bb8da91051c28e03129841846b17bb65f1c0446ab0',
  onnxSha256: 'b308e93c3dda3c4f9f968295b2b7b8d1c0583a711bca53a08d51134d9d1cbd5c',
  // the host's conversion of that ONNX (the same bytes every time)
  pklSha256: 'e48fd48fbafb752fad9db5c599bcdcc600a82a82de24ed496f13c96bdd843363',
  archiveBytes: 42945210,
};
export const ajPath = () => path.join(AJ_DIR, AJ_PKL);
/** The host's Restore model arguments for a model id: 'animejanai' = --restore-model with the installed file;
 * 'anime4k' = --restore-anime4k (Anime4K's Restore CNN UL, inside the host); none = Real-ESRGAN. */
export function restoreModelArgs(id?: string): string[] {
  if (id === 'anime4k') return ['--restore-anime4k'];
  return id === 'animejanai' && fileExists(ajPath()) ? ['--restore-model', ajPath()] : [];
}

ipcMain.handle('aj-ready', () => {
  const ready = fileExists(ajPath());
  let sha256 = '';
  try {
    if (ready) sha256 = sha256File(ajPath());
  } catch {
    /* the hash is informational */
  }
  const known = sha256 ? sha256 === AJ_DL.pklSha256 : undefined;
  return { ready, known, sha256, file: AJ_PKL, mb: Math.round(AJ_DL.archiveBytes / 1048576) };
});

ipcMain.handle('aj-download', async () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-animejanai-'));
  const archive = path.join(tmp, 'release.7z');
  const progress = (received: number, total: number, stage: string) => {
    try {
      if (win && !win.webContents.isDestroyed()) win.webContents.send('aj-progress', { received, total, stage });
    } catch {
      /* window gone */
    }
  };
  try {
    const res = await fetch(AJ_DL.url, { headers: { 'user-agent': 'SmoothMyVideo' } });
    if (!res.ok || !res.body) return { ok: false, error: `download failed: HTTP ${res.status}` };
    const total = parseInt(res.headers.get('content-length') || '0', 10) || AJ_DL.archiveBytes;
    let received = 0;
    let last = 0;
    const src = Readable.fromWeb(res.body as never);
    src.on('data', (chunk: Buffer) => {
      received += chunk.length;
      if (received - last > 2 * 1048576 || received === total) {
        last = received;
        progress(received, total, 'download');
      }
    });
    await pipeline(src, fs.createWriteStream(archive));
    progress(received, total, 'verify');
    const ah = sha256File(archive);
    if (ah !== AJ_DL.archiveSha256)
      return { ok: false, error: `archive checksum mismatch (${ah.slice(0, 12)}...), nothing installed` };
    execFileSync(SYS_TAR, ['-xf', archive, '-C', tmp, AJ_DL.member]);
    const found = path.join(tmp, ...AJ_DL.member.split('/'));
    if (!fileExists(found)) return { ok: false, error: AJ_FILE + ' not in the downloaded archive, nothing installed' };
    const mh = sha256File(found);
    if (mh !== AJ_DL.onnxSha256)
      return { ok: false, error: `model checksum mismatch (${mh.slice(0, 12)}...), nothing installed` };
    fs.mkdirSync(AJ_DIR, { recursive: true });
    if (!ajConvert(found, ajPath())) return { ok: false, error: 'the model did not convert, nothing installed' };
    const ph = sha256File(ajPath());
    return { ok: true, sha256: ph, known: ph === AJ_DL.pklSha256 };
  } catch (e) {
    return { ok: false, error: String(e) };
  } finally {
    try {
      fs.rmSync(tmp, { recursive: true, force: true });
    } catch {
      /* temp cleanup best effort */
    }
  }
});

ipcMain.handle('aj-open', (_e, which: string) => {
  shell.openExternal(which === 'licence' ? AJ_DL.licence : AJ_DL.page);
  return true;
});

ipcMain.handle('fruc-ready', () => {
  const bridge = fileExists(path.join(NVOFFRUC_DIR, 'nvoffruc_bridge.dll'));
  const dll = fileExists(path.join(NVOFFRUC_DIR, 'NvOFFRUC.dll'));
  return { ready: bridge && dll, bridge, dll, dir: NVOFFRUC_DIR };
});
ipcMain.handle('fruc-open-download', () => {
  shell.openExternal(OF_SDK_URL);
  return true;
});
ipcMain.handle('fruc-install', (_e, source?: string) => {
  if (!source)
    return { ok: false, error: 'No Optical Flow SDK selected. Use "Get from NVIDIA", then "Choose .zip".', copied: [] };
  return installFruc(source);
});
ipcMain.handle('fruc-choose', async () => {
  const r = await dialog.showOpenDialog(win!, {
    title: 'Select the Optical Flow SDK .zip',
    properties: ['openFile'],
    filters: [{ name: 'Zip', extensions: ['zip'] }],
  });
  return r.canceled ? null : r.filePaths[0] || null;
});

// --- Dolby Vision + HDR10+ export tools: readiness + one-click install -----------------------------
// DV Profile 8.1 export layers a Dolby Vision RPU on top of the HDR10 render, then tags the MP4 with a
// dvvC box the engine writes itself (src/render/hdr10meta.ts injectDvConfig) - so the bundled ffmpeg is enough
// to mux DV and NO GPAC/MP4Box is needed. HDR10+ export embeds ST 2094-40 dynamic metadata into the
// HDR10 render; the engine collects the per-frame stats itself and hdr10plus_tool injects the SEI
// (src/render/dvhp.ts hpExport). The two non-bundled pieces are quietvoid's MIT tools: one click downloads the pinned
// Windows x64 release zip, checks its SHA256 (the GitHub API digest) before anything is extracted,
// extracts the exe with Windows' bsdtar and copies it into engine/dvtools / engine/hptools. "Ready" =
// the exe present. Bump version, url, page, hash and size together.
const DV_DIR = path.join(ENGINE, 'dvtools');
const DOVI_BIN = 'dovi_tool.exe';
const DOVI_DL: PinnedZip = {
  version: '2.3.4',
  url: 'https://github.com/quietvoid/dovi_tool/releases/download/2.3.4/dovi_tool-2.3.4-x86_64-pc-windows-msvc.zip',
  page: 'https://github.com/quietvoid/dovi_tool/releases/tag/2.3.4',
  zipSha256: '0d9733d311dfe49f9dca0c350f7e6d978182673d333c6f07f7365daf9e6c81eb',
  zipBytes: 1346827,
};
const HP_DIR = path.join(ENGINE, 'hptools');
const HP_BIN = 'hdr10plus_tool.exe';
const HP_DL: PinnedZip = {
  version: '1.7.2',
  url: 'https://github.com/quietvoid/hdr10plus_tool/releases/download/1.7.2/hdr10plus_tool-1.7.2-x86_64-pc-windows-msvc.zip',
  page: 'https://github.com/quietvoid/hdr10plus_tool/releases/tag/1.7.2',
  zipSha256: '82b2d560073941b14c6511a431f429e33e134e5caefb60d7e8f6f6e6da8e16ba',
  zipBytes: 948827,
};

// Recursive case-insensitive search for a file by name under root (a release zip can nest its payload).
const findDvBin = (root: string, name: string): string | null => findFilesNamed(root, [name])[0] || null;

// One-click install of a pinned release zip: download into a temp dir (progress to the renderer as
// `channel`, at most 200 lines a download), check the zip's SHA256 before anything is extracted (a
// replaced asset installs nothing), extract the wanted members with bsdtar and copy each into destDir:
// `files` = [name in the zip, name in destDir] pairs, all found before any is copied. The temp dir is
// removed again.
async function installPinnedZip(
  dl: PinnedZip,
  files: Array<[string, string]>,
  destDir: string,
  channel: string,
): Promise<InstallResult> {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'smv-tool-'));
  const zip = path.join(tmp, 'tool.zip');
  const progress = (received: number, total: number, stage: string) => {
    try {
      if (win && !win.webContents.isDestroyed()) win.webContents.send(channel, { received, total, stage });
    } catch {
      /* window gone */
    }
  };
  try {
    const res = await fetch(dl.url, { headers: { 'user-agent': 'SmoothMyVideo' } });
    if (!res.ok || !res.body) return { ok: false, error: `download failed: HTTP ${res.status}`, copied: [] };
    const total = parseInt(res.headers.get('content-length') || '0', 10) || dl.zipBytes;
    let received = 0;
    let shown = 0;
    const src = Readable.fromWeb(res.body as never);
    src.on('data', (chunk: Buffer) => {
      received += chunk.length;
      if (received - shown >= total / 200 || received >= total) {
        shown = received;
        progress(received, total, 'download');
      }
    });
    await pipeline(src, fs.createWriteStream(zip));
    progress(received, total, 'verify');
    const zh = sha256File(zip);
    if (zh !== dl.zipSha256)
      return { ok: false, error: `checksum mismatch (${zh.slice(0, 12)}...), nothing installed`, copied: [] };
    try {
      execFileSync(SYS_TAR, ['-xf', zip, '-C', tmp, ...files.map(([member]) => '*' + member)]);
    } catch {
      // bsdtar fails when a member pattern matches nothing (its patterns are case-sensitive): take the
      // whole archive and let the search below decide
      execFileSync(SYS_TAR, ['-xf', zip, '-C', tmp]);
    }
    const found = files.map(([member]) => findDvBin(tmp, member));
    const missing = files.filter((_, k) => !found[k]).map(([member]) => member);
    if (missing.length)
      return { ok: false, error: missing.join(', ') + ' not in the downloaded zip, nothing installed', copied: [] };
    fs.mkdirSync(destDir, { recursive: true });
    files.forEach(([, name], k) => {
      const dest = path.join(destDir, name);
      try {
        if (fileExists(dest)) fs.chmodSync(dest, 0o666);
      } catch {
        /* best effort */
      }
      fs.copyFileSync(found[k] as string, dest);
    });
    return { ok: true, copied: files.map(([, name]) => name) };
  } catch (e) {
    return { ok: false, error: String(e), copied: [] };
  } finally {
    try {
      fs.rmSync(tmp, { recursive: true, force: true });
    } catch {
      /* temp cleanup best effort */
    }
  }
}
const toolMb = (dl: PinnedZip) => (dl.zipBytes / 1048576).toFixed(1);

ipcMain.handle('dv-ready', () => {
  const dovi = fileExists(path.join(DV_DIR, DOVI_BIN));
  return { dovi, ready: dovi, dir: DV_DIR, ver: DOVI_DL.version, mb: toolMb(DOVI_DL) };
});

ipcMain.handle('dv-open-download', () => {
  shell.openExternal(DOVI_DL.page);
  return true;
});

ipcMain.handle('dv-download', () => installPinnedZip(DOVI_DL, [[DOVI_BIN, DOVI_BIN]], DV_DIR, 'dv-progress'));

ipcMain.handle('hp-ready', () => {
  const ready = fileExists(path.join(HP_DIR, HP_BIN));
  return { ready, dir: HP_DIR, ver: HP_DL.version, mb: toolMb(HP_DL) };
});

ipcMain.handle('hp-open-download', () => {
  shell.openExternal(HP_DL.page);
  return true;
});

ipcMain.handle('hp-download', () => installPinnedZip(HP_DL, [[HP_BIN, HP_BIN]], HP_DIR, 'hp-progress'));

// --- Intel XeSS: readiness + one-click install ----------------------------------------------------
// XeSS-SR runs through the bundled engine/xess/smv_xess_bridge.dll and Intel's libxess.dll, which is not
// bundled: one click downloads Intel's pinned XeSS SDK release zip (github.com/intel/xess), checks its
// SHA256 (the GitHub API digest) before anything is extracted, and copies libxess.dll (XeSS-SR),
// libxess_fg.dll + libxell.dll (XeSS frame generation and its required XeLL) and Intel's licence into
// engine/xess, unmodified (the licence allows redistribution without modification). "Ready" = libxess.dll
// present (XeSS-SR); "fg" = libxess_fg.dll + libxell.dll present (XeSS-FG through the bundled
// smv_xefg_bridge.dll). Bump version, url, page, hash and size together.
const XESS_DIR = path.join(ENGINE, 'xess');
const XESS_DL: PinnedZip = {
  version: '3.0.2',
  url: 'https://github.com/intel/xess/releases/download/v3.0.2/XeSS_SDK_3.0.2.zip',
  page: 'https://github.com/intel/xess/releases/tag/v3.0.2',
  zipSha256: '88b8a373f30e33f3558a77a93e634f11b8132fc3047ea1a8edeead32b8471990',
  zipBytes: 76839749,
};
const XESS_FILES: Array<[string, string]> = [
  ['libxess.dll', 'libxess.dll'],
  ['libxess_fg.dll', 'libxess_fg.dll'],
  ['libxell.dll', 'libxell.dll'],
  ['LICENSE.txt', 'XeSS_LICENSE.txt'],
];

ipcMain.handle('xess-ready', () => {
  const ready = fileExists(path.join(XESS_DIR, 'libxess.dll'));
  const fg = ['libxess_fg.dll', 'libxell.dll'].every((f) => fileExists(path.join(XESS_DIR, f)));
  return { ready, fg, dir: XESS_DIR, ver: XESS_DL.version, mb: toolMb(XESS_DL) };
});

ipcMain.handle('xess-open-download', () => {
  shell.openExternal(XESS_DL.page);
  return true;
});

ipcMain.handle('xess-download', () => installPinnedZip(XESS_DL, XESS_FILES, XESS_DIR, 'xess-progress'));
