// The engine cache stamp (priority 30 step 2, user 2026-09-25 "can first app run on a new version
// empty the folder automatically?"). Engines are found by NAME (net, shapes, TensorRT-RTX version,
// weight tags) and the name does not say which ONNX graph an engine was built from, so after an
// export change (trt_lookup.ONNX_REV, e.g. rev 2's PRelu rewrite) the old engines would keep being
// reused. The stamp = the TensorRT-RTX version the host loads + onnx/weights_tags.txt (the weight
// tags and, since rev 2, the `x <rev>` line); when it differs from the one stored in the cache
// folder, the folder is emptied once (every engine rebuilds at its size on first use) and the new
// stamp is written. Runs at app start and CLI start, before anything spawns the host.
import * as fs from 'fs';
import * as path from 'path';

const STAMP = 'engine_stamp.txt';

export function engineCacheDir(engine: string): string {
  return process.env.SMV_TRT_CACHE || path.join(engine, 'trt_cache_safe_to_delete');
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
export function checkEngineCache(engine: string): string {
  const want = currentStamp(engine);
  if (want === null) return 'engine cache: no stamp inputs, left as is';
  const dir = engineCacheDir(engine);
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
