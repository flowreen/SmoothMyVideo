// --- UI mode (Video / Live) ----------------------------------------------------------------
// ONE class on <html> drives the split: mode-video hides every .lonly element, mode-live every
// .vonly one (see the CSS). The head script paints it before first paint; this keeps it in sync,
// persists the choice, and re-writes the key after a settings-version wipe above. Anything with
// neither class (model group, Speed, Image scale, RTX, Sharpen) is shared by both modes.
let uiMode = document.documentElement.classList.contains('mode-live') ? 'live' : 'video';
let lvReady = false; // main's lv-ready answer; false = show the unavailable note in Live mode
function lvUnavailUi() {
  $('lvunavail').style.display = uiMode === 'live' && !lvReady ? '' : 'none';
}
function setMode(m) {
  uiMode = m === 'live' ? 'live' : 'video';
  document.documentElement.className = 'mode-' + uiMode;
  $('modevideo').classList.toggle('on', uiMode === 'video');
  $('modelive').classList.toggle('on', uiMode === 'live');
  try {
    localStorage.setItem('uiMode', uiMode);
  } catch {}
  try {
    lvSendOpts(); // the tab is part of the Live options; the first call comes before they exist
  } catch {}
  lvUnavailUi();
  // the numbered panels follow the processing order, NVIDIA's, the same in both modes
  applyOrder();
  if (dlssUiReady) syncDlssMode();
}
// Switching mid-job would hide the running thing, so both buttons grey out while a live session
// or a file render is going (called from lvUi and from the run start/end enable sweeps).
let modeBusy = false;
function modeBtnUi(busy) {
  modeBusy = !!busy;
  $('modevideo').disabled = modeBusy;
  $('modelive').disabled = modeBusy;
  $('modevideo').textContent = tr('mode.video');
  $('modelive').textContent = tr('mode.live');
  $('modevideo').title = modeBusy ? tr('mode.busy') : tr('mode.video.tip');
  $('modelive').title = modeBusy ? tr('mode.busy') : tr('mode.live.tip');
}
modeBtnUi(false);
i18nRelabel.push(() => modeBtnUi(modeBusy));
// the Size hint under Upscale reads differently per mode (syncUpscale is defined further down,
// so it runs on the click, not at boot where the boot-time syncUpscale call covers it)
$('modevideo').onclick = () => {
  setMode('video');
  try {
    syncUpscale();
  } catch {}
};
$('modelive').onclick = () => {
  setMode('live');
  try {
    syncUpscale();
  } catch {}
};
setMode(uiMode);

function log(t) {
  $('log').textContent += t;
  $('log').scrollTop = $('log').scrollHeight;
}
// Reveal the operational UI (controls, preview, progress, log). The first page shows it WITH the welcome/drop hint
// still visible (keepWelcome); loading a video (or an RTX install) hides the welcome panel too.
function showWorkspace(keepWelcome?) {
  if (!keepWelcome) $('welcome').style.display = 'none';
  $('workspace').style.display = '';
}
// Repaint the frames/speed/ETA line. Called on every PROGRESS frame and on each timer tick; the
// tick recomputes the remaining seconds from the anchor so the ETA counts down between frames.
function paint() {
  if (!procStart) return; // still warming up: keep the warmup message
  const fps = Math.round(lastFps),
    remain = lastFps > 0 ? Math.max(0, etaAnchor - (Date.now() - etaAnchorTime) / 1000) : -1;
  // Projected final size (engine SIZE lines: bytes so far, linearly extrapolated to 100%), so a
  // long render reveals its disk cost in the first minutes instead of at the end.
  const size = projBytes > 0 ? '   ·   ~' + fmtBytes(projBytes) : '';
  const done = Math.round(curDone * fpp),
    total = totalGen;
  txt(
    'frames',
    () =>
      tr('run.frames', { done, total }) +
      '   ·   ' +
      (remain < 0 ? tr('run.measuring') : tr('run.speedEta', { fps, eta: fmtEta(remain), finish: fmtFinish(remain) })) +
      size,
  );
}
i18nRelabel.push(() => paint());
function fmtBytes(b) {
  return b >= 1e9 ? (b / 1e9).toFixed(b >= 1e10 ? 0 : 1) + ' GB' : Math.round(b / 1e6) + ' MB';
}
// Interpolation is ON when ANY model is ticked. The model checkboxes ARE the master toggle (there
// is no separate Interpolate box): unticking them all means "keep the source frames", which greys
// the Speed controls. Ids are inlined rather than using MODEL_BOXES() so this stays callable before
// that const is initialised.
function interpOn() {
  return ['modelgmfss', 'modelrife', 'modeldlss', 'modelfruc', 'modelfsrfg', 'modelxefg', 'modelnvof'].some((id) => {
    const e = $(id);
    return !!e && e.checked;
  });
}
function srcFps() {
  const [n, d] = ((info && info.fps) || '24/1').split('/');
  return +n / (+d || 1);
}

// SPEED: ONE control, two mutually exclusive modes. What PERSISTS is the user's choice (the mode
// plus its value), never a number derived from whichever video happens to be loaded.
// That decoupling is the whole point: the bundled sample clip used to leak its 30 fps in,
// so the app looked like it had picked 60 fps on the user's behalf, and Live inherited that.
//   fps    - absolute: this exact rate, whatever the source is (up to 3 decimals, so an exact
//            multiple of a fractional source stays on-grid: 47.952 = 2x of 23.976)
//   screen - absolute: the monitor's refresh rate
// (No relative Multiplier mode: the fps box expresses any multiple, and fixed-timing live models now derive their whole multiplier
// from the target themselves.)
const SPEED_DEF_FPS = 60;
const FPS_TOP = 10000; // the fps box's top = Live's cap whatever file is loaded; a file render clamps its own target to sliderMax
function speedMode() {
  return localStorage.getItem('speedMode') === 'screen' ? 'screen' : 'fps';
}
function curFps() {
  const v = +localStorage.getItem('speedFps');
  return v > 0 ? v : SPEED_DEF_FPS;
}
// The effective target the ENGINE renders to. Every existing consumer (gridMulti, outRatio,
// fpsValid, outName, the run payload) reads this one function, so the mode change stays contained.
function targetFps() {
  if (!interpOn()) return srcFps(); // nothing ticked = output keeps the source rate
  return speedMode() === 'screen' ? screenHz : Math.min(curFps(), Math.floor(sliderMax() * 1000) / 1000);
}
function sliderMin() {
  return info ? Math.max(1, Math.round(srcFps())) : 1;
} // never target below the source rate
function sliderMax() {
  return info ? srcFps() * 100 : 2400;
} // hard cap at exactly 100x the EXACT source (24000/1001 -> 2397.602, kept on-grid; not a rounded 2400)
// Decimals to display: match the source's natural precision (23.976 -> 3, 24 -> 0, 29.97 -> 2), taking
// the MORE precise of source and monitor rate so screen-match values also render exactly (24 src +
// 359.99 Hz -> 2). r_frame_rate is a rational (24000/1001), so read precision off a 3-dp rounding.
function decimalsOf(x) {
  const s = (+x).toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  const i = s.indexOf('.');
  return i < 0 ? 0 : s.length - 1 - i;
}
function targetDecimals() {
  return info ? Math.max(decimalsOf(srcFps()), decimalsOf(screenHz)) : 0;
}
function fpsValid(t) {
  if (!interpOn()) return true;
  return t >= sliderMin() && t <= sliderMax();
}
function outRatio() {
  if (!interpOn()) return 1;
  return targetFps() / srcFps();
}
// On-grid eligibility: if the integer target is an exact integer multiple of the (fractional) source
// fps, run the GMFSS integer --multi path (real endpoints, exact for 23.976); else resample via --fps.
// Returns the multiplier N (>=2) for the on-grid case, or 0 for the resample case. Uses the fractional
// srcFps so 48 -> N=2 on a 23.976 source (round(23.976*2)=48).
function gridMulti() {
  if (!info) return 0;
  const s = srcFps();
  const t = targetFps();
  const N = Math.round(t / s);
  return N >= 2 && Math.abs(s * N - t) < 0.02 ? N : 0;
} // on-grid when the target is an exact multiple (tolerance covers the 3-dp display rounding)
// 3-decimal precision, so an exact multiple of a fractional source rate is expressible
// (47.952 = 2x of 23.976 stays on the integer --multi path).
// The box tops at FPS_TOP; a file render's target (targetFps) clamps to 100x the loaded source (sliderMax,
// floored to 3 dp so it lands exactly on-grid: 2397.602 on a 23.976 source), 2400 with no video.
function setFps(v) {
  if (isNaN(v)) return;
  localStorage.setItem('speedFps', String(Math.min(FPS_TOP, Math.max(1, Math.round(v * 1000) / 1000))));
  syncTargetUI();
  refresh();
  try {
    lvSendOpts();
    lvTargetChanged();
  } catch {}
  if (cmpOn()) refreshCompare();
} // Compare runs the render's multiplier
function setSpeedMode(m) {
  localStorage.setItem('speedMode', m);
  syncTargetUI();
  refresh();
  try {
    lvSendOpts();
    lvTargetChanged();
  } catch {}
  if (cmpOn()) refreshCompare();
}
// Ratio text for the current target: the exact integer when on-grid (2, 3, ...), else
// the fractional ratio (e.g. 2.08 for a 50 fps target on a 24 source, which the engine resamples).
function ratioText() {
  const gm = gridMulti();
  return gm ? String(gm) : (targetFps() / srcFps()).toFixed(2).replace(/0$/, '');
}
// Show ONLY the active mode's input, filled from that mode's own value, and grey everything while a
// run owns the controls or no model is ticked. Skip rewriting an input while it is focused, so a
// two-keystroke entry ("6" -> "60") is not reset mid-type.
function syncTargetUI() {
  const m = speedMode(),
    on = interpOn(),
    dis = !on || goMode === 'running';
  for (const [id, mode] of [
    ['smfps', 'fps'],
    ['smscreen', 'screen'],
  ]) {
    const b = $(id);
    if (!b) continue;
    b.classList.toggle('on', m === mode);
    b.disabled = !on;
  }
  const show = (id, vis) => {
    const e = $(id);
    if (e) e.style.display = vis ? '' : 'none';
  };
  show('fpswrap', m === 'fps');
  show('screenwrap', m === 'screen');
  ['fpsin', 'fpsup', 'fpsdn', 'fpsround'].forEach((id) => {
    const e = $(id);
    if (e) e.disabled = dis;
  });
  // Nothing ticked = no rate change to describe: the fps box empties.
  if ($('fpsin') && document.activeElement !== $('fpsin')) $('fpsin').value = on ? curFps() : '';
  // The round button: only for a resampled target, snaps it to the closest whole multiple of the
  // EXACT source rate (14.97x of 23.976 -> 15x = 359.64), so nobody types 3 decimals to get on-grid.
  const rn = roundMulti();
  show('fpsround', m === 'fps' && on && rn > 0);
  if (rn && $('fpsround')) $('fpsround').title = tr('speed.round.tip', { n: rn, fps: roundFps(rn) });
}
i18nRelabel.push(() => syncTargetUI());
// The closest whole multiple (>= 2) for a resampled target, else 0 (already on-grid / no video).
function roundMulti() {
  if (!info || gridMulti()) return 0;
  const N = Math.max(2, Math.round(targetFps() / srcFps()));
  return fpsValid(roundFps(N)) ? N : 0;
}
function roundFps(N) {
  return Math.round(srcFps() * N * 1000) / 1000;
} // setFps's 3-dp rounding
$('fpsround').onclick = () => {
  const N = roundMulti();
  if (N) setFps(srcFps() * N);
};
// Press-and-hold auto-repeat: one step on press, then repeat every 60ms after a 350ms hold. Each step
// re-reads the current target, so holding + walks the value up (and stops at the clamp) instead of once.
function holdRepeat(btn, fn) {
  let to, iv;
  const stop = () => {
    clearTimeout(to);
    clearInterval(iv);
  };
  const step = () => {
    if (btn.disabled) {
      stop();
      return;
    }
    fn();
  };
  btn.addEventListener('mousedown', (e) => {
    if (btn.disabled) return;
    e.preventDefault();
    step();
    to = setTimeout(() => {
      iv = setInterval(step, 60);
    }, 350);
  });
  ['mouseup', 'mouseleave'].forEach((ev) => btn.addEventListener(ev, stop));
}
// The steppers walk the persisted value; neither reads the loaded video.
holdRepeat($('fpsup'), () => setFps(curFps() + 1));
holdRepeat($('fpsdn'), () => setFps(curFps() - 1));
$('speedseg').onclick = (e) => {
  const b = e.target.closest('button');
  if (!b || b.disabled) return;
  setSpeedMode(b.id === 'smscreen' ? 'screen' : 'fps');
};
// Interp-off runs are named for the pass that actually runs (priority: restore > DLSS 5 > sharpen > HDR;
// an upscale keeps its _<h>p tag instead, as before). The '_unchanged' fallback covers the
// nothing-enabled combo, which is only ever a display placeholder: startRun refuses it with the
// "Nothing to do" alert, so no file by that name is produced - the tag just tells the user the
// current settings would change nothing, instead of claiming a sharpen that will not happen.
function passTag() {
  if (restoreOn()) return '_restored';
  if (nrOn()) return '_dlss5';
  if ($('sharpen').checked && (parseFloat($('sharpval').value) || 0) > 0) return '_sharpened';
  if (hdrOn()) return '_hdr';
  return '_unchanged';
}
function outName() {
  if (!input) return '';
  const i = input.lastIndexOf('.');
  const b = i < 0 ? input : input.slice(0, i);
  const d = upDims(); // upscale target ({w,h} or null); tag the new height
  const res = d ? '_' + d.h + 'p' : '';
  // subtitles/extra tracks ride in mkv; above 1000 fps MKV cannot time the frames (whole milliseconds): .mp4, and the
  // render saves the subtitles beside it
  const ext = info && info.needMkv && targetFps() <= 1000 ? '.mkv' : '.mp4';
  return interpOn() ? b + '_' + Math.round(targetFps()) + 'fps' + res + ext : b + (d ? res : passTag()) + ext;
}
// A custom name that STILL CARRIES an auto tag (_<n>fps or _<n>p) stays in sync with the settings:
// the user's folder, base words and extension are preserved, and the _<n>fps / _<n>p suffix is
// regenerated from the current fps + upscale - so changing the fps updates the fps tag, and enabling
// or changing the upscale resolution updates (or appends) the resolution tag, layered on top of the
// user's edit. A name the user renamed with neither tag left is treated as fully custom and never rewritten.
function retagCustom(name) {
  if (!/_\d+fps(?=[._]|$)|_\d+p(?=[._]|$)/.test(name)) return name;
  const slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
  const dir = name.slice(0, slash + 1); // '' when there is no directory part
  const rest = name.slice(slash + 1);
  const dot = rest.lastIndexOf('.');
  const ext = dot >= 1 ? rest.slice(dot) : ''; // keep the user's chosen extension
  let base = dot >= 1 ? rest.slice(0, dot) : rest;
  base = base
    .replace(/_\d+fps/g, '')
    .replace(/_\d+p(?=_|$)/g, '')
    .replace(/_sharpened|_restored|_dlss5|_unchanged|_hdr(?=_|$)/g, '');
  const d = upDims();
  const res = d ? '_' + d.h + 'p' : '';
  const tags = interpOn() ? '_' + Math.round(targetFps()) + 'fps' + res : d ? res : passTag();
  return dir + base + tags + ext;
}
function refresh() {
  if (!input) return;
  const t = targetFps();
  const N = gridMulti();
  const kind = N
    ? tr('rate.ongrid') // exact integer multiple of the source
    : Math.round(t) <= Math.round(srcFps())
      ? tr('rate.none') // degenerate: output fps = source fps
      : tr('rate.resampled'); // non-integer multiple: engine uses --fps
  // The hint carries the derived other side of the relationship: the implied ratio to the source.
  const derived = tr('rate.derived', { ratio: ratioText(), fps: +srcFps().toFixed(targetDecimals()) });
  $('ratehint').textContent = !interpOn() ? '' : derived + ' · ' + kind;
  syncTargetUI();
  if (!fpsValid(t)) {
    $('out').value = '';
    $('out').placeholder = tr('out.ph.range', { min: sliderMin(), max: sliderMax() });
    lastOut = null;
    return;
  }
  $('out').placeholder = tr('out.ph');
  if (customOut) customOut = retagCustom(customOut); // re-sync the _fps/_p tags in the user's name
  lastOut = customOut || outName();
  $('out').value = lastOut;
  if (goMode === 'idle') $('out').disabled = false; // editable while idle; startRun locks it
  updateDvHpHints(); // the DV/HDR10+ "skipped for MKV" hints track the resolved output name
  scheduleCheckResume(); // the Resume detection tracks the resolved output name too
}
i18nRelabel.push(() => refresh());
// Crash/exit resume detection: an interrupted render leaves its stage-1 video + sidecar next to
// the output (see the engine's resume block); when the resolved output name has one, #go becomes
// "Resume" and the render continues where it left off (the engine re-validates the settings
// signature itself, so a stale match here degrades to a fresh render with a log notice, never a
// wrong file). Debounced: refresh() fires on every slider scrub.
let resumeInfo = null,
  resumeTimer = null,
  resumeLoggedFor = null,
  resumeSeq = 0,
  resumeTipShown = false;
function scheduleCheckResume() {
  clearTimeout(resumeTimer);
  resumeTimer = setTimeout(checkResume, 150);
}
async function checkResume() {
  const seq = ++resumeSeq;
  let r = null;
  try {
    if (input && lastOut) r = await ipcRenderer.invoke('check-resume', lastOut);
  } catch {}
  if (seq !== resumeSeq) return; // a newer check superseded this one mid-flight
  resumeInfo = r;
  if (goMode !== 'idle') return; // never touch the button while it is the Pause toggle
  $('go').textContent = resumeInfo ? tr('run.resume') : tr('run.go');
  if (resumeInfo) {
    const pct = resumeInfo.total ? Math.round((100 * resumeInfo.pair) / resumeInfo.total) : 0;
    $('go').title = tr('run.resume.tip', { pct });
    setStatus('run.status.interrupted', { pct });
    if (resumeLoggedFor !== lastOut) {
      resumeLoggedFor = lastOut;
      log('>> Interrupted render found for this file; it will resume from ~' + pct + '% instead of starting over\n');
    }
  } else {
    if (resumeLoggedFor === lastOut) resumeLoggedFor = null; // gone (e.g. cancelled): a future interruption logs anew
    $('go').title = tr('run.go.tip');
  }
}
// the code-owned texts before a video: the Go button, the path line, the status line
function idleTexts() {
  if (goMode !== 'idle') return;
  if (!input) {
    $('go').textContent = tr('run.go');
    $('go').title = tr('run.go.tip');
    $('path').textContent = tr('pick.none');
    setStatus('run.status.select');
  } else {
    scheduleCheckResume();
    if (!resumeInfo) setStatus('run.status.ready');
  }
}
idleTexts();
// a switch relabels the idle Go button and path; the status line rebuilds its own last text (txt), so a finished run
// still reads Done / Failed / Cancelled
i18nRelabel.push(() => {
  if (goMode !== 'idle') return;
  if (!input) {
    $('go').textContent = tr('run.go');
    $('go').title = tr('run.go.tip');
    $('path').textContent = tr('pick.none');
  } else scheduleCheckResume();
});
// Editing the output path inline: a non-empty value becomes the custom output (kept across setting
// changes, like the Change... dialog); clearing it on blur reverts to the auto-generated name.
$('out').oninput = () => {
  const v = $('out').value.trim();
  customOut = v || null;
  lastOut = v || outName();
};
$('out').onchange = () => {
  if (!$('out').value.trim()) {
    customOut = null;
    refresh();
  }
};

async function loadVideo(f) {
  if (!f) return false;
  input = f;
  customOut = null;
  $('cmpto').value = '';
  cmpRes = null;
  syncCompareUi(); // a new video (a batch's next one too) starts at Off: loading one never runs two models
  $('path').textContent = f;
  try {
    localStorage.setItem('dir', path.dirname(f));
  } catch {}
  const p = await ipcRenderer.invoke('probe', f);
  // A modal alert would stall an unattended batch, so mid-batch probe failures go to the log.
  if (p.error) {
    if (batchAuto) log('>> Could not read ' + f + ': ' + p.error + '\n');
    else alert(tr('err.read', { err: p.error }));
    return false;
  }
  const streams = p.streams || [];
  const s = streams.find((x) => x.codec_type === 'video') || streams[0] || {};
  // Track passthrough: subtitles or mp4-incompatible audio push the output container to .mkv
  // (mirrors the engine's own MP4_AUDIO_OK rule, so the name shown matches what it writes).
  const MP4A = ['aac', 'ac3', 'eac3', 'mp3', 'alac', 'opus', 'flac'];
  const auds = streams.filter((x) => x.codec_type === 'audio');
  const subs = streams.filter((x) => x.codec_type === 'subtitle');
  info = {
    fps: s.r_frame_rate,
    w: s.width,
    h: s.height,
    nb: +(s.nb_frames || 0),
    dur: +((p.format || {}).duration || 0),
    codec: s.codec_name,
    transfer: s.color_transfer || '',
    nAud: auds.length,
    nSub: subs.length,
    needMkv: subs.length > 0 || auds.some((a) => !MP4A.includes((a.codec_name || '').toLowerCase())),
  };
  info.srcHdr = info.transfer === 'smpte2084' || info.transfer === 'arib-std-b67';
  // TrueHDR converts SDR only; for an HDR source the engine carries the HDR through untouched.
  $('rtxhdr').disabled = info.srcHdr;
  syncRtxHdrHint();
  showWorkspace();
  $('info').style.display = 'block';
  $('res').textContent = s.width + ' x ' + s.height;
  $('fps').textContent = srcFps().toFixed(decimalsOf(srcFps())); // natural precision: 24 -> "24", 23.976 -> "23.976"
  const d = info.dur | 0;
  $('dur').textContent =
    ((d / 3600) | 0) + ':' + String(((d % 3600) / 60) | 0).padStart(2, '0') + ':' + String(d % 60).padStart(2, '0');
  info.codecName = s.codec_name || '?';
  paintInfoTexts();
  $('go').disabled = false;
  $('changeout').disabled = false;
  $('open').disabled = true;
  $('play').disabled = true;
  $('outrow').style.display = '';
  $('live').style.display = 'none'; // stale live thumbnail belongs to the previous video
  setStatus('run.status.ready');
  applyScreenFps(); // re-asserts the interp / screen-rate state and recomputes the output name
  syncUpscale(); // recompute the upscale target + output dims for this video's resolution
  // The preview starts the native host and a CUDA context, a GPU load that can stall Chromium's
  // compositor - so if it spawns before the
  // file info has actually painted, the info (already probed in ~100ms, DOM updated synchronously
  // above) can't reach the screen until that load eases, which is the "video info loads slowly"
  // bug. So: reveal the menu + file info now (showWorkspace above, synchronous), paint a loading
  // wheel on the preview panes (showPreviewInitializing, synchronous), then wait for a REAL composited
  // frame (two rAFs = one full paint) so the info is guaranteed on screen BEFORE the heavy preview
  // starts. Skipped when a batch auto-advances (no per-item preview).
  if (!batchAuto) {
    showPreviewInitializing();
    requestAnimationFrame(() => requestAnimationFrame(openPreviewForVideo));
  }
  return true;
}
// the RTX HDR hint, the exact source rate's tooltip and the tracks line, from the loaded video
function syncRtxHdrHint() {
  $('rtxhdrhint').textContent = info && info.srcHdr ? tr('rtxhdr.srcHdr') : tr('rtxhdr.hint');
}
function paintInfoTexts() {
  if (!info) return;
  $('fps').title = tr('info.fps.tip', { exact: (info.fps ? info.fps + ' = ' : '') + +srcFps().toFixed(6) }); // e.g. 24000/1001 = 23.976024
  $('codec').textContent =
    info.codecName +
    (info.nAud > 1 || info.nSub
      ? '  ·  ' +
        tr(info.nSub === 1 ? 'info.tracks1' : 'info.tracksN', { aud: info.nAud, sub: info.nSub }) +
        ' (' +
        tr(info.needMkv ? 'info.keptMkv' : 'info.kept') +
        ')'
      : '');
}
syncRtxHdrHint();
i18nRelabel.push(() => {
  syncRtxHdrHint();
  paintInfoTexts();
});
// One or more inputs: the first loads now, the rest queue for the same settings back to back.
function queueVideos(paths) {
  const list = (paths || []).filter(Boolean);
  if (!list.length) return;
  batch = list.slice(1);
  batchTotal = list.length;
  batchFailed = [];
  if (batchTotal > 1) log('>> Queued ' + batchTotal + ' files (same settings, default output names)\n');
  loadVideo(list[0]);
}
$('pick').onclick = async () => {
  queueVideos(await ipcRenderer.invoke('pick-video', localStorage.getItem('dir') || undefined));
};
// The fps box holds the user's ABSOLUTE target. Commit on change only, so a partial entry on the way
// to a bigger number is never clamped mid-type. It is independent of any loaded video.
$('fpsin').onchange = () => {
  const v = parseFloat($('fpsin').value);
  if (isNaN(v)) {
    $('fpsin').value = curFps();
    return;
  }
  setFps(v);
  $('fpsin').value = curFps();
};
