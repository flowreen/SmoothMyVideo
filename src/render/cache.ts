// The model cache: the folder the host keeps its compiled kernel cubins in (named by their source
// hash, so a new build never reuses a stale one). Earlier versions also kept TensorRT engines there
// (*.engine, their *.jit / *.warm files and the engine_stamp.txt that guarded them): those are
// removed once, at app start and CLI start, before anything spawns the host.
import * as fs from 'fs';
import * as path from 'path';

const CACHE_NAME = 'model_cache_safe_to_delete';
// where earlier versions kept the cache, inside the engine folder
const OLD_NAMES = ['trt_cache_safe_to_delete', 'trt_cache'];
// what earlier versions' TensorRT engines left in the cache
const ENGINE_LEFTOVER = /(\.engine|\.jit|\.warm)$|^engine_stamp\.txt$/i;

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
 *  merge, the cache's own entries win. */
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

/** Remove what earlier versions' TensorRT engines left in the cache folder (top level only). */
function dropEngineLeftovers(dir: string): string {
  let removed = 0;
  let left = 0;
  try {
    for (const f of fs.readdirSync(dir)) {
      if (!ENGINE_LEFTOVER.test(f)) continue;
      try {
        fs.rmSync(path.join(dir, f), { recursive: true, force: true });
        removed++;
      } catch {
        left++; // held open by another process: retried next start
      }
    }
  } catch {
    return 'engine cache: no folder yet';
  }
  if (!removed && !left) return 'engine cache: current';
  return `engine cache: removed ${removed} TensorRT leftovers` + (left ? `, ${left} in use (retried next start)` : '');
}

/** Adopt an older cache and remove its TensorRT leftovers; returns what it did (for the caller's log). */
export function checkEngineCache(engine: string, legacy: string[] = []): string {
  const dir = engineCacheDir(engine);
  const moved = adoptOldCache(engine, dir, legacy);
  return moved + dropEngineLeftovers(dir);
}
