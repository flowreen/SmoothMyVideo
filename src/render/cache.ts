// The engine cache stamp. Engines are found by NAME (net, shapes, TensorRT-RTX version,
// weight tags) and the name does not say which ONNX graph an engine was built from, so after a new
// export the old engines would keep being reused. The stamp = the TensorRT-RTX version the host
// loads + onnx/weights_tags.txt (the weight tags and the `export` line, trt_lookup.export_stamp:
// it changes with every export input); when it differs from the one stored in the cache
// folder, the folder is emptied once (every engine rebuilds at its size on first use) and the new
// stamp is written. Runs at app start and CLI start, before anything spawns the host.
import * as fs from 'fs';
import * as path from 'path';

const STAMP = 'engine_stamp.txt';
const CACHE_NAME = 'model_cache_safe_to_delete';
// where earlier versions kept the cache, inside the engine folder
const OLD_NAMES = ['trt_cache_safe_to_delete', 'trt_cache'];

/** The app's top folder: the repo root in a dev tree, the folder holding SmoothMyVideo.exe in an
 *  install (its engine folder is resources\engine). The host (lkModelCacheDir) derives the same. */
function appTopDir(engine: string): string {
  const parent = path.dirname(path.resolve(engine));
  return path.basename(parent).toLowerCase() === 'resources' ? path.dirname(parent) : parent;
}

export function engineCacheDir(engine: string, env: NodeJS.ProcessEnv = process.env): string {
  return env.SMV_TRT_CACHE || path.join(appTopDir(engine), CACHE_NAME);
}

/** Move `from` to `to`; across drives (AppData on C:) a rename fails with EXDEV, so copy instead. */
function moveEntry(from: string, to: string): void {
  try {
    fs.renameSync(from, to);
  } catch (e) {
    if ((e as NodeJS.ErrnoException).code !== 'EXDEV') throw e;
    fs.cpSync(from, to, { recursive: true });
    fs.rmSync(from, { recursive: true, force: true });
  }
}

/** Move every cache an earlier version left elsewhere into the top folder's cache once: the engine
 *  folder's (OLD_NAMES) and `legacy` (1.0.x's AppData one). Into an existing cache the old entries
 *  merge, the cache's own entries win; the stamp check then empties whatever is stale. */
function adoptOldCache(engine: string, dir: string, legacy: string[]): string {
  if (process.env.SMV_TRT_CACHE) return '';
  const notes: string[] = [];
  for (const old of [...OLD_NAMES.map((name) => path.join(engine, name)), ...legacy]) {
    if (path.resolve(old) === path.resolve(dir) || !fs.existsSync(old)) continue;
    try {
      if (!fs.existsSync(dir)) moveEntry(old, dir);
      else {
        for (const f of fs.readdirSync(old)) {
          const to = path.join(dir, f);
          if (!fs.existsSync(to)) moveEntry(path.join(old, f), to);
        }
        fs.rmSync(old, { recursive: true, force: true });
      }
      notes.push(`moved ${old} to ${dir}`);
    } catch {
      notes.push(`${old} in use, left (retried next start)`);
    }
  }
  return notes.length ? `engine cache: ${notes.join(', ')}\n` : '';
}

/** The stamp this tree's engines must carry, or null when a part is missing (then nothing is
 *  wiped: a tree without its ONNX or its runtime cannot build engines anyway). */
function currentStamp(engine: string): string | null {
  try {
    const trt = fs.readFileSync(path.join(engine, 'gpu_runtime', 'tensorrt_rtx_version.txt'), 'utf8').trim();
    const onnxDir = process.env.SMV_ONNX_DIR || path.join(engine, 'onnx');
    const tags = fs.readFileSync(path.join(onnxDir, 'weights_tags.txt'), 'utf8').trim();
    return trt && tags ? `trt ${trt}\n${tags}\n` : null;
  } catch {
    return null;
  }
}

/** Empty the engine cache when its stamp is stale; returns what it did (for the caller's log). */
export function checkEngineCache(engine: string, legacy: string[] = []): string {
  const dir = engineCacheDir(engine);
  const moved = adoptOldCache(engine, dir, legacy);
  return moved + stampCheck(engine, dir);
}

function stampCheck(engine: string, dir: string): string {
  const want = currentStamp(engine);
  if (want === null) return 'engine cache: no stamp inputs, left as is';
  let have = '';
  try {
    have = fs.readFileSync(path.join(dir, STAMP), 'utf8');
  } catch {
    /* no stamp yet: a fresh folder, or one from before the stamp existed */
  }
  if (have === want) return 'engine cache: current';
  let removed = 0;
  let left = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (f === STAMP) continue;
      try {
        fs.rmSync(path.join(dir, f), { recursive: true, force: true });
        removed++;
      } catch {
        left++; // held open by another process: its name would match a new build, so no stamp yet
      }
    }
  } catch {
    /* no folder yet */
  }
  if (left) return `engine cache: stamp changed, emptied ${removed} entries, ${left} in use (retried next start)`;
  try {
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, STAMP), want);
  } catch {
    return `engine cache: emptied ${removed} entries, stamp not written`;
  }
  return `engine cache: stamp changed, emptied ${removed} entries`;
}
