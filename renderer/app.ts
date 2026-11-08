const { ipcRenderer, shell, webUtils, webFrame } = require('electron');
// Announce every page load (app launch OR a reload). On a reload this lets main kill any engine
// orphaned by the previous page - e.g. an accidental Ctrl+R mid-render - so the app comes up clean
// instead of the refreshed UI racing a still-running engine (the 1%->15%->1% progress ping-pong).
ipcRenderer.send('renderer-ready');

// --- UI zoom: Ctrl+= / Ctrl+- / Ctrl+0 and Ctrl+scrollwheel, browser-style ------------------------
// Chromium zoom levels: factor = 1.2^level; half-level steps ~= 10% per notch, clamped to a range
// where the tall single-column layout stays usable. Persisted so the app reopens at the same zoom.
function applyZoom(z) {
  z = Math.max(-4, Math.min(4, z));
  webFrame.setZoomLevel(z);
  localStorage.setItem('zoomLevel', String(z));
}
applyZoom(parseFloat(localStorage.getItem('zoomLevel')) || 0);
window.addEventListener('keydown', (e) => {
  if (!e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.key === '=' || e.key === '+') {
    applyZoom(webFrame.getZoomLevel() + 0.5);
    e.preventDefault();
  } else if (e.key === '-') {
    applyZoom(webFrame.getZoomLevel() - 0.5);
    e.preventDefault();
  } else if (e.key === '0') {
    applyZoom(0);
    e.preventDefault();
  }
});
window.addEventListener(
  'wheel',
  (e) => {
    if (!e.ctrlKey) return;
    e.preventDefault(); // never page-scroll while zooming (needs the non-passive listener)
    applyZoom(webFrame.getZoomLevel() + (e.deltaY < 0 ? 0.5 : -0.5));
  },
  { passive: false },
);
const path = require('path');
let info = null,
  input = null,
  lastOut = null,
  customOut = null,
  fpp = 1,
  totalGen = 1,
  procStart = 0,
  baseK = 0,
  cancelled = false;
let lastPreview = null; // path of the latest pause-generated partial preview (PREVIEW_READY), or null
let dlssPreempt = null; // pct if DLSS-FG stopped on RTX Video preemption (a clean resumable stop), else null
// Batch queue: inputs waiting after the current one. Filled by a multi-select pick or a multi-file
// drop; consumed one at a time on engine-done (same settings for every file, default output names).
let batch = [],
  batchTotal = 1,
  batchAuto = false,
  batchFailed = [];
// The unfinished inputs (current file first) are mirrored to localStorage while a run is live, so a
// crash or accidental close mid-batch can be resumed: the next launch re-queues them (no auto-start).
function saveBatch(list) {
  try {
    if (list && list.length) localStorage.setItem('batchPending', JSON.stringify(list));
    else localStorage.removeItem('batchPending');
  } catch {}
}
// #go doubles as the run's Pause/Resume toggle: 'idle' = "Smooth It!" (starts a run), 'running' =
// "Pause" (holds generation), 'paused' = "Resume" (continues). engine-done resets it to 'idle'.
let goMode = 'idle';
// the batch's "File n of N · " prefix as it reads now, rebuilt in the language of the moment
function fileTag(): () => string {
  const n = batchTotal - batch.length,
    total = batchTotal;
  return () => (total > 1 ? tr('run.file', { n, total }) + ' · ' : '');
}
// the render status line (txt below): one language string, with the batch prefix when tagged
function setStatus(key: string, vars?: Record<string, string | number>, tagged = false) {
  const tag = tagged ? fileTag() : () => '';
  txt('status', () => tag() + tr(key, vars));
}
// ETA is a live countdown. Each PROGRESS frame re-anchors it to the average pace since warmup,
// and a timer ticks the displayed seconds down between frames so it never sits frozen. curDone/
// curPct hold the last frame numbers so the timer can repaint the line with no new PROGRESS.
let etaAnchor = 0,
  etaAnchorTime = 0,
  lastFps = 0,
  etaTimer = null,
  curDone = 0,
  curPct = 0,
  projBytes = 0;
function fmtEta(s) {
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600),
    m = Math.floor((s % 3600) / 60),
    ss = s % 60;
  const p = (x) => String(x).padStart(2, '0');
  return h > 0
    ? tr('time.hms', { h, m: p(m), s: p(ss) })
    : m > 0
      ? tr('time.ms', { m, s: p(ss) })
      : tr('time.s', { s: ss });
}
// the wall-clock end time, its month in the UI language
function fmtFinish(s) {
  const d = new Date(Date.now() + Math.max(0, s) * 1000);
  return (
    String(d.getHours()).padStart(2, '0') +
    ':' +
    String(d.getMinutes()).padStart(2, '0') +
    ' ' +
    String(d.getDate()).padStart(2, '0') +
    '-' +
    new Intl.DateTimeFormat(uiLang, { month: 'long' }).format(d) +
    '-' +
    d.getFullYear()
  );
}

const $ = (id: string): any => document.getElementById(id);
// Stored-settings schema version: bump whenever a localStorage key or its scale/meaning changes,
// and the first launch of the new build wipes every stored setting once, so stale keys can never
// conflict with the new UI (e.g. an HDR slider rescale or a key rename).
const SETTINGS_VERSION = '3'; // 3 = the unified target-fps slider (key fpsTarget) replaced the multiplier dropdown
if (localStorage.getItem('settingsVersion') !== SETTINGS_VERSION) {
  localStorage.clear();
  localStorage.setItem('settingsVersion', SETTINGS_VERSION);
}
// --- UI text: every visible string lives in i18n\<code>.json -------------------------------------
// en.json is the source and the fallback for a key another language lacks. index.html marks its texts with data-i18n
// (the element's HTML) and data-i18n-title / -ph / -aria / -alt (that attribute); the code asks tr(key, {name: value})
// for every text it writes, and each code-owned text registers in i18nRelabel so a switch repaints it from the
// current state. The language = the user's pick (localStorage uiLang), else the first of Windows' languages SMV
// has, else English.
const fs = require('fs');
const { fileURLToPath } = require('url');
const I18N_DIR = path.join(path.dirname(fileURLToPath(location.href)), 'i18n');
// the selector offers the languages whose file is present (en always)
const I18N_CODES = ['en', 'ja', 'zh-CN', 'ko', 'es', 'pt-BR', 'fr', 'de', 'id', 'hi', 'ar'].filter(
  (c) => c === 'en' || fs.existsSync(path.join(I18N_DIR, c + '.json')),
);
function i18nLoad(code: string): Record<string, string> {
  try {
    return JSON.parse(fs.readFileSync(path.join(I18N_DIR, code + '.json'), 'utf8'));
  } catch {
    return {};
  }
}
const I18N_EN = i18nLoad('en');
let I18N_CUR: Record<string, string> = I18N_EN;
let uiLang = 'en';
function i18nPick(): string {
  const saved = localStorage.getItem('uiLang');
  if (saved && I18N_CODES.includes(saved)) return saved;
  for (const l of navigator.languages && navigator.languages.length
    ? navigator.languages
    : [navigator.language || 'en']) {
    const s = String(l).toLowerCase();
    if (s.startsWith('ja')) return 'ja';
    if (s === 'zh' || s.startsWith('zh-cn') || s.startsWith('zh-sg') || s.startsWith('zh-hans')) return 'zh-CN';
    if (s.startsWith('ko')) return 'ko';
    if (s.startsWith('es')) return 'es';
    if (s.startsWith('pt')) return 'pt-BR';
    if (s.startsWith('fr')) return 'fr';
    if (s.startsWith('de')) return 'de';
    if (s.startsWith('id') || s.startsWith('in')) return 'id';
    if (s.startsWith('hi')) return 'hi';
    if (s.startsWith('ar')) return 'ar';
    if (s.startsWith('en')) return 'en';
  }
  return 'en';
}
function tr(key: string, vars?: Record<string, string | number>): string {
  let s = I18N_CUR[key] ?? I18N_EN[key] ?? key;
  if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m));
  return s;
}
const I18N_ATTRS: [string, string, string][] = [
  ['i18nTitle', 'data-i18n-title', 'title'],
  ['i18nPh', 'data-i18n-ph', 'placeholder'],
  ['i18nAria', 'data-i18n-aria', 'aria-label'],
  ['i18nAlt', 'data-i18n-alt', 'alt'],
];
// Arabic reads right to left: <html dir> mirrors the layout (index.html keeps the pictures, the log and the paths
// left to right)
const I18N_RTL = ['ar'];
function applyI18n() {
  document.documentElement.lang = uiLang;
  document.documentElement.dir = I18N_RTL.includes(uiLang) ? 'rtl' : 'ltr';
  document.querySelectorAll('[data-i18n]').forEach((e: any) => {
    e.innerHTML = tr(e.dataset.i18n);
  });
  for (const [ds, sel, attr] of I18N_ATTRS)
    document.querySelectorAll('[' + sel + ']').forEach((e: any) => e.setAttribute(attr, tr(e.dataset[ds])));
}
const i18nRelabel: (() => void)[] = [];
// The lines an event writes (the render status, the frames line, the Live status, the preview note, the hotkey hint):
// txt(id, build) writes build() and keeps it, so a switch rebuilds the line in the new language even when no later
// event comes (Stopped, Done, Paused, a saved clip); raw(id, s) writes the host's own words as they are and drops it.
const i18nLine = new Map<string, () => string>();
function txt(id: string, build: () => string) {
  i18nLine.set(id, build);
  $(id).textContent = build();
}
function raw(id: string, s: string) {
  i18nLine.delete(id);
  $(id).textContent = s;
}
// the line's current text as a builder, for an event that appends to it
function lineOf(id: string): () => string {
  const f = i18nLine.get(id);
  if (f) return f;
  const s = $(id).textContent;
  return () => s;
}
function setLang(code: string) {
  uiLang = I18N_CODES.includes(code) ? code : 'en';
  I18N_CUR = uiLang === 'en' ? I18N_EN : i18nLoad(uiLang);
  try {
    localStorage.setItem('uiLang', uiLang);
  } catch {}
  applyI18n();
  for (const [id, f] of i18nLine) {
    try {
      $(id).textContent = f();
    } catch {}
  }
  for (const f of i18nRelabel) {
    try {
      f();
    } catch {}
  }
}
{
  const sel = $('langsel');
  for (const c of I18N_CODES) {
    const o = document.createElement('option');
    o.value = c;
    o.textContent = (c === 'en' ? I18N_EN : i18nLoad(c))['lang.name'] || c;
    sel.appendChild(o);
  }
  uiLang = i18nPick();
  I18N_CUR = uiLang === 'en' ? I18N_EN : i18nLoad(uiLang);
  sel.value = uiLang;
  sel.onchange = () => setLang(sel.value);
  applyI18n();
}
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
  // The preview cold-starts Python + torch + a CUDA context (several seconds on the first video of a
  // session), which SATURATES the GPU and stalls Chromium's compositor - so if it spawns before the
  // file info has actually painted, the info (already probed in ~100ms, DOM updated synchronously
  // above) can't reach the screen until that Python eases, which is the "video info loads slowly"
  // bug. So: reveal the menu + file info now (showWorkspace above, synchronous), paint a loading
  // wheel on the preview panes (showPreviewInitializing, synchronous), then wait for a REAL composited
  // frame (two rAFs = one full paint) so the info is guaranteed on screen BEFORE the heavy preview
  // Python starts. Skipped when a batch auto-advances (no per-item preview).
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

// Sharpen (CAS): a checkbox enables Contrast Adaptive Sharpening and reveals a 0..1 strength slider;
// unchecking hides the slider and disables it. Default is on at full strength (1.0). The strength is
// forwarded to the engine as --sharpen; both the on/off state and the value persist between sessions.
function syncSharpen() {
  const on = $('sharpen').checked;
  $('sharpwrap').style.display = on ? 'flex' : 'none';
  $('sharpnum').textContent = (+$('sharpval').value).toFixed(2);
  syncPreview(); // show/hide the preview panel as sharpen turns on/off
}
const savedSharpOn = localStorage.getItem('sharpenOn');
if (savedSharpOn !== null) $('sharpen').checked = savedSharpOn === '1';
const savedSharpVal = localStorage.getItem('sharpenVal');
if (savedSharpVal !== null) $('sharpval').value = savedSharpVal;
syncSharpen();
$('sharpen').onchange = () => {
  localStorage.setItem('sharpenOn', $('sharpen').checked ? '1' : '0');
  syncSharpen();
  try {
    lvSendOpts();
  } catch {}
};
$('sharpval').oninput = () => {
  localStorage.setItem('sharpenVal', $('sharpval').value);
  syncSharpen();
  try {
    lvSendOpts();
  } catch {}
};

// Restore: engine --restore. Real-ESRGAN's anime-video model cleans compression noise and
// redraws linework on every output frame (before the upscale); fine texture can flatten - it
// is a generative repaint, not a filter. Off by default: it changes the look, and every
// emitted frame pays a second model pass.
// The second row, AnimeJaNai HD: the author's 2x anime model (CC BY-NC-SA 4.0, never bundled), the same pass with
// another model (the host's --restore-model). The two rows exclude each other; AnimeJaNai runs once installed
// (aj-ready), else its setup note shows (one click downloads the author's release, checked in main).
let ajReady: any = { ready: false };
function ajOn() {
  return $('restoreaj').checked && !!ajReady.ready;
}
// The third row, Anime4K: bloc97's Restore CNN UL, the line restore of the Anime4K shader set's Mode A (its weights
// inside the host, always ready; the host's --restore-anime4k). Ticked together with the Anime4K upscaler, the two
// are that mode; alone it keeps the frame's size.
function a4kRestOn() {
  return $('restorea4k').checked;
}
function restoreOn() {
  return $('restore').checked || ajOn() || a4kRestOn();
}
function restoreModelId() {
  return ajOn() ? 'animejanai' : a4kRestOn() ? 'anime4k' : '';
}
// Restore ALWAYS starts off: an expensive, look-altering pass you opt
// into per session, so its state is deliberately NOT persisted; old key retired.
localStorage.removeItem('restoreOn');
async function checkAjReady() {
  $('ajstatus').textContent = tr('common.checking');
  $('ajstatus').style.color = 'var(--sub)';
  try {
    ajReady = await ipcRenderer.invoke('aj-ready');
  } catch {
    ajReady = { ready: false };
  }
  const r = ajReady,
    line = (r.ready ? '✓ ' : '✗ ') + tr('aj.line', { file: r.file || '' });
  if (r.ready) {
    $('ajstatus').textContent = tr('setup.ready');
    $('ajstatus').style.color = 'var(--green)';
    $('ajsteps').innerHTML =
      line +
      '<br>' +
      (r.known === false ? tr('aj.unverified', { sha: String(r.sha256 || '').slice(0, 12) }) : tr('aj.installed'));
  } else {
    $('ajstatus').textContent = tr('dlss.notReady');
    $('ajstatus').style.color = '#e85c5c';
    $('ajsteps').innerHTML =
      line +
      '<br>' +
      tr('aj.howto', {
        licence: ajLink('licence', tr('aj.licLink')),
        link: ajLink('page', tr('aj.repoLink')),
        mb: r.mb || 41,
      });
  }
  $('ajbtns').style.display = r.ready ? 'none' : '';
  $('ajsetup').style.display = $('restoreaj').checked && (!r.ready || r.known === false) ? 'block' : 'none';
  refreshPreviewIfOpen();
  refresh();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
}
// a link main opens (the licence or the author's release page): its text is the language's, its action the code's
function ajLink(which, text) {
  return '<a href="#" onclick="ipcRenderer.invoke(\'aj-open\', \'' + which + '\');return false;">' + text + '</a>';
}
function syncAj() {
  if ($('restoreaj').checked) checkAjReady();
  else $('ajsetup').style.display = 'none';
}
i18nRelabel.push(() => syncAj());
let ajDownloading = false;
ipcRenderer.on('aj-progress', (_e, p) => {
  if (!ajDownloading) return;
  const mb = (x) => (x / 1048576).toFixed(0);
  $('ajstatus').textContent =
    p.stage === 'verify' ? tr('dl.verify') : tr('dl.progress', { got: mb(p.received), total: mb(p.total) });
});
async function doAjDownload() {
  if (ajDownloading) return;
  ajDownloading = true;
  $('ajdl').disabled = true;
  $('ajsetup').style.display = 'block';
  $('ajstatus').textContent = tr('dl.start');
  $('ajstatus').style.color = 'var(--sub)';
  let r: any = {};
  try {
    r = await ipcRenderer.invoke('aj-download');
  } catch (e) {
    r = { ok: false, error: String(e) };
  }
  ajDownloading = false;
  $('ajdl').disabled = false;
  await checkAjReady();
  if (!r.ok) {
    $('ajstatus').textContent = tr('dl.failed');
    $('ajstatus').style.color = '#e85c5c';
    $('ajsteps').innerHTML = '✗ ' + (r.error || tr('dl.failedShort')) + '<br>' + tr('dl.retry');
    $('ajsetup').style.display = 'block';
  }
}
$('ajdl').onclick = doAjDownload;
// one Restore model at a time: ticking a row unticks the other
$('restore').addEventListener('change', () => {
  if ($('restore').checked) {
    if ($('restoreaj').checked) {
      $('restoreaj').checked = false;
      syncAj();
    }
    $('restorea4k').checked = false;
  }
});
$('restoreaj').addEventListener('change', () => {
  if ($('restoreaj').checked) {
    $('restore').checked = false;
    $('restorea4k').checked = false;
  }
  syncAj();
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
});
$('restorea4k').addEventListener('change', () => {
  if ($('restorea4k').checked) {
    $('restore').checked = false;
    if ($('restoreaj').checked) {
      $('restoreaj').checked = false;
      syncAj();
    }
  }
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
});

// The pass order in both modes: on each source or captured frame Restore and the resize to the
// working size (the DLSS mode), DLSS 5, Sharpen and RTX HDR (with the Dolby Vision and HDR10+ metadata of
// a file render), then the interpolation (frame generation last), then the final resize to the output.
// The numbered panels follow it.
function applyOrder() {
  $('nrpanel').before($('restorepanel'));
  $('nrpanel').before($('uppanel'));
  $('nrpanel').after($('sharpenpanel'));
  $('sharpenpanel').after($('hdrpanel'));
  $('hdrpanel').after($('dvpanel'));
  $('dvpanel').after($('hppanel'));
  $('hppanel').after($('interppanel'));
}
localStorage.removeItem('nvorderOn'); // retired: the pass order is no longer a setting
applyOrder();

// "Screen refresh rate" is one of the three Speed modes rather than an override checkbox, so there is
// nothing to re-assert after a video loads: the mode and its value are the user's, independent of
// the source. applyScreenFps is kept as the name every caller already uses (video load, run finish,
// model change) and simply re-renders the Speed row.
let screenHz = 60;
function applyScreenFps() {
  syncTargetUI();
  refresh();
}
ipcRenderer.invoke('refresh-rate').then((hz) => {
  screenHz = +hz || 60;
  $('screenhz').textContent = screenHz;
  if (info) applyScreenFps(); // monitor rate can add decimals (359.99 -> 2), so reformat the target box now that it's known
  try {
    lvSendOpts();
    lvTargetChanged();
  } catch {} // Live's inherited target may just have changed
});

// Upscale-to-resolution: a target height chosen from the dropdown (or the Custom slider) drives an
// arbitrary upscale factor (targetHeight / sourceHeight), keeping the source aspect ratio. RTX VSR
// (the #rtxvsr toggle, once its runtime is installed) does the upscale with NVIDIA AI; with it off or
// absent the engine uses a Lanczos3 resize. On by default; the choice persists. Independent of
// interpolation, so it also applies in sharpen-only mode. There is no integer-scale restriction (RTX
// VSR was probed clean to 16K), but the output is bounded by two real limits: the encoders top out at
// 16K (NVENC caps either dimension at 8192px, verified on the RTX 5090; past that the ENGINE switches
// to a CPU encoder automatically - AV1 to ~12K, H.266/VVC to 16K, verified 15360x8640 - so the GUI
// wall is VVC's), and the engine clamps the upscale factor to 16x. The custom-height slider and
// upDims() honour both, so the long side never exceeds 15360px and the factor never exceeds 16x.
const ENC_MAX = 15360; // 16K long side: the last encoder standing (libvvenc)
const NVENC_MAX = 8192; // above this the engine encodes on the CPU (slower)
const MAX_FACTOR = 16; // engine UPSCALE_F clamp (gmfss_interp.py)
let screenW = 0,
  screenH = 0,
  rtxReady = { vsr: false, hdr: false, bridge: false };
function upTargetH() {
  // chosen output height (0 = off)
  const v = $('upres').value;
  if (v === '0') return 0;
  if (v === 'screen') return screenH || 0;
  if (v === 'custom') return +$('upcustom').value || 0;
  return +v || 0;
}
function upDims() {
  // {w,h} of the resized output (encoder/factor clamped), or null
  if (!info || !info.h || !info.w) return null;
  let th = upTargetH();
  if (!th || th === info.h) return null; // off, or a 1:1 target = nothing to resize
  th = Math.min(th, info.h * MAX_FACTOR, ENC_MAX); // 16x engine factor cap, then the encoder long-side cap
  th = Math.max(th, Math.ceil(info.h / MAX_FACTOR / 2) * 2); // and the 16x floor going down (engine clamp)
  let h = th - (th % 2);
  let w = Math.round((info.w * h) / info.h);
  w = w - (w % 2);
  if (w > ENC_MAX) {
    // wide aspect: width is the binding side, fit it to 8192px
    h = Math.floor((ENC_MAX * info.h) / info.w / 2) * 2;
    w = Math.round((info.w * h) / info.h);
    w = w - (w % 2);
  }
  return { w, h };
}
function upFactor() {
  const d = upDims();
  return d ? d.h / info.h : 0;
} // arbitrary factor (0 = off)
function syncUpscale() {
  a4kAaUi();
  $('upcustomwrap').style.display = $('upres').value === 'custom' ? 'inline-flex' : 'none';
  if (info && info.w && info.h) {
    // bound the custom slider to what this source + encoder can reach
    const m = Math.max(
      720,
      Math.min(Math.floor((ENC_MAX * Math.min(1, info.h / info.w)) / 2) * 2, info.h * MAX_FACTOR),
    );
    if (+$('upcustom').max !== m) {
      $('upcustom').max = m;
      if (+$('upcustom').value > m) $('upcustom').value = m;
    }
  }
  $('upcustomnum').textContent = +$('upcustom').value + 'p';
  let txt = '';
  if ($('upres').value !== '0') {
    // Video only: Live's size is what Display presents
    const d = upDims();
    // Just the resulting dimensions plus encoder-relevant warnings; no resampler names (the
    // engine always picks the best resize).
    if (d)
      txt =
        '→ ' +
        d.w +
        ' × ' +
        d.h +
        (d.w > NVENC_MAX || d.h > NVENC_MAX ? '  ·  ' + tr('up.cpu') : '') +
        (d.w >= ENC_MAX || d.h >= ENC_MAX ? '  ·  ' + tr('up.cap') : '');
    else if (info) txt = tr('up.same');
    else txt = tr('up.load');
  }
  $('upinfo').textContent = txt;
  if (dlssUiReady) syncDlssMode(); // the working size follows the output
  refresh(); // the target height is tagged into the output filename
}

// DLSS mode, Live only (a file render runs every pass at its output size): Live's working size as NVIDIA's share of
// what it presents (the window, or the screen in Fill), Restore, DLSS 5 and the smoothing run at it; the host applies
// the floors (aspect kept, DLSS 5 >= 128 px wide). Persisted, DLAA by default.
var dlssUiReady = false; // var: setMode runs before this block and must skip the sync
// a mode's share in whole percent, as the option labels round it
const DLSS_PCT = { dlaa: 100, quality: 67, balanced: 58, performance: 50, ultra: 33 };
// the slider = the share in whole percent, 1..100 (DLAA at the right end under the default mark): on a mode's
// percent it is that mode (NVIDIA's exact share), anywhere else that percent (the host's --scale 0.01..1), shown in
// the select as Custom (a hidden option: the select picks only the five modes)
function dlssPct() {
  return +$('dlssrange').value;
}
function dlssMode() {
  return Object.keys(DLSS_PCT).find((m) => DLSS_PCT[m] === dlssPct()) || String(dlssPct());
}
function syncDlssMode() {
  const m = dlssMode();
  if (Object.hasOwn(DLSS_PCT, m)) $('dlssmode').value = m;
  else {
    $('dlsscustomopt').textContent = tr('dlssmode.customPct', { pct: m });
    $('dlssmode').value = 'custom';
  }
  $('dlssrangenum').textContent = dlssPct() + '%';
}
{
  // a saved mode or percent; a saved Auto or the old Custom (both retired) opens as DLAA
  const m = localStorage.getItem('dlssMode');
  if (Object.hasOwn(DLSS_PCT, m)) $('dlssrange').value = DLSS_PCT[m];
  else if (/^\d+$/.test(m || '') && +m >= 1 && +m <= 100) $('dlssrange').value = m;
  localStorage.removeItem('dlssCustom'); // the old Custom's own key: the percent is the saved mode now
  localStorage.removeItem('lvFlow'); // retired: the Image scale slider became the DLSS mode
}
// a value that holds 1 s starts the fresh build: the preview's refresh and a running live
// session's relaunch (the working size is a spawn-time argument)
let dlssSettleT = null;
function dlssChanged() {
  localStorage.setItem('dlssMode', dlssMode());
  syncDlssMode();
  upFitHintUi();
  lvSendOpts();
  clearTimeout(dlssSettleT);
  dlssSettleT = setTimeout(() => {
    refreshPreviewIfOpen();
    if (lvState === 'running') ipcRenderer.send('lv-restart');
  }, 1000);
}
$('dlssmode').onchange = () => {
  $('dlssrange').value = DLSS_PCT[$('dlssmode').value];
  dlssChanged();
};
$('dlssrange').oninput = dlssChanged; // a drag stops anywhere from 1 to 100 %
dlssUiReady = true;
syncDlssMode();
function setScreenOptLabel() {
  const o = [...$('upres').options].find((o) => o.value === 'screen');
  if (o) o.textContent = tr('up.opt.screen') + (screenW && screenH ? ' (' + screenW + '×' + screenH + ')' : '');
}
i18nRelabel.push(() => {
  syncUpscale();
  syncDlssMode();
  setScreenOptLabel();
});
// NOTE: the element id is outcodec, NOT codec - id "codec" is taken by the source-codec info span.
// Per-choice guidance so the trade-off is clear without leaving the dropdown. Size claims are
// from a real A/B at the engine's verified visually-lossless settings (the quality-first
// tuning: on the 1080p sample HEVC CQ17-p7-ladder 2.25 MB / AV1 CQ22 2.17 MB / VVC QP17 0.98 MB).
function syncCodec() {
  const c = $('outcodec').value;
  $('codechint').textContent = ['hevc', 'av1', 'vvc'].includes(c) ? tr('codec.' + c + '.hint') : '';
}
i18nRelabel.push(syncCodec);
const savedCodec = localStorage.getItem('codec');
if (savedCodec && [...$('outcodec').options].some((o) => o.value === savedCodec)) $('outcodec').value = savedCodec;
$('outcodec').onchange = () => {
  localStorage.setItem('codec', $('outcodec').value);
  syncCodec();
};
syncCodec();
let savedUpres = localStorage.getItem('upres');
// the 540p and 480p presets are retired: the custom slider reaches every height from 240p, so a saved one becomes Custom
if (savedUpres === '540' || savedUpres === '480') {
  localStorage.setItem('upcustom', savedUpres);
  savedUpres = 'custom';
  localStorage.setItem('upres', savedUpres);
}
if (savedUpres && [...$('upres').options].some((o) => o.value === savedUpres)) $('upres').value = savedUpres;
const savedUpcustom = localStorage.getItem('upcustom');
if (savedUpcustom) $('upcustom').value = savedUpcustom;
$('upres').onchange = () => {
  localStorage.setItem('upres', $('upres').value);
  syncUpscale();
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
};
$('upcustom').oninput = () => {
  localStorage.setItem('upcustom', $('upcustom').value);
  syncUpscale();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
};
$('upcustom').addEventListener('change', refreshPreviewIfOpen);
ipcRenderer.invoke('screen-size').then((s) => {
  screenW = (s && s.width) || 0;
  screenH = (s && s.height) || 0;
  setScreenOptLabel();
  syncUpscale();
});

// NVIDIA RTX: real RTX Video Super Resolution (ON by default) + RTX HDR (OFF by default) via the
// engine/rtxvideo CUDA bridge + NVIDIA's feature DLLs. The feature DLLs are non-redistributable, so the app
// can't bundle them; instead you pick the downloaded RTX Video SDK .zip and the app copies the two
// DLLs into engine/rtxvideo for you. Readiness = the bridge + the feature's model DLL present.
function syncRtx() {
  const on = $('rtxvsr').checked || $('rtxhdr').checked;
  // The setup box appears only when a requested feature's runtime is missing (or while installing);
  // checkRtxReady() collapses it once the files are correctly installed so READY takes no screen space.
  // With no feature requested there is nothing to set up, so hide it outright.
  if (on) checkRtxReady();
  else $('rtxsetup').style.display = 'none';
}
async function checkRtxReady() {
  $('rtxstatus').textContent = tr('common.checking');
  $('rtxstatus').style.color = 'var(--sub)';
  try {
    rtxReady = await ipcRenderer.invoke('rtx-ready');
  } catch {
    rtxReady = { vsr: false, hdr: false, bridge: false };
  }
  const needVsr = $('rtxvsr').checked,
    needHdr = $('rtxhdr').checked;
  const ok = (!needVsr || rtxReady.vsr) && (!needHdr || rtxReady.hdr);
  const lines = [];
  if (needVsr) lines.push((rtxReady.vsr ? '✓ ' : '✗ ') + tr('vsr.name'));
  if (needHdr) lines.push((rtxReady.hdr ? '✓ ' : '✗ ') + tr('rtxhdr.name'));
  if (ok) {
    $('rtxstatus').textContent = tr('setup.ready');
    $('rtxstatus').style.color = 'var(--green)';
    $('rtxsteps').innerHTML = lines.join('<br>') + '<br>' + tr('rtx.installed');
  } else {
    $('rtxstatus').textContent = tr('dlss.notReady');
    $('rtxstatus').style.color = '#e85c5c';
    const detail = rtxReady.bridge === false ? tr('rtx.noBridge') : tr('rtx.howto');
    $('rtxsteps').innerHTML = lines.join('<br>') + '<br>' + detail;
  }
  $('rtxbtns').style.display = ok ? 'none' : ''; // hide Get / Choose once the runtime is ready
  // One setup box serves both RTX features, which now live in different panels: park it under
  // the first checkbox that still needs it (VSR in Upscale, else HDR in HDR).
  const host = (needVsr && !rtxReady.vsr) || !needHdr ? $('uppanel') : $('hdrpanel');
  if ($('rtxsetup').parentElement !== host) host.appendChild($('rtxsetup'));
  // Collapse the whole setup box once the requested runtime is installed, so a READY state takes no
  // screen space; it reappears only while something needed is still missing.
  $('rtxsetup').style.display = ok ? 'none' : 'block';
  syncUpscale(); // the upscale backend label depends on VSR readiness
  syncHdrColor(); // so do the RTX HDR colour controls
  refreshPreviewIfOpen(); // re-render the open preview now that HDR readiness is known
}
async function doInstall(source) {
  $('rtxsetup').style.display = 'block'; // show install progress (checkRtxReady collapses it again on success)
  $('rtxstatus').textContent = tr('common.installing');
  $('rtxstatus').style.color = 'var(--sub)';
  let r: any = {};
  try {
    r = await ipcRenderer.invoke('rtx-install', source);
  } catch (e) {
    r = { ok: false, error: String(e) };
  }
  if (!r.ok) $('rtxsteps').innerHTML = '✗ ' + (r.error || tr('common.installFailed')) + '<br>' + tr('rtx.pickAgain');
  await checkRtxReady();
  if (!r.ok) $('rtxsetup').style.display = 'block'; // keep an install error visible even if no toggle needs the feature
}
$('rtxget').onclick = () => ipcRenderer.invoke('rtx-open-download');
$('rtxbrowsezip').onclick = async () => {
  const p = await ipcRenderer.invoke('rtx-choose');
  if (p) doInstall(p);
}; // selecting a .zip auto-installs
localStorage.removeItem('supersampleOn'); // retired: supersample removed (measured imperceptible)
localStorage.removeItem('encspeed'); // retired: no Encoder speed selector (every render uses the Quality encoder)
localStorage.removeItem('rtxvsrOn'); // retired: its stored choice was made while RTX VSR defaulted off
$('rtxvsr').checked = localStorage.getItem('vsrOn') !== '0'; // default ON: RTX VSR upscales unless unticked
if (localStorage.getItem('rtxhdrOn') === '1') $('rtxhdr').checked = true; // default OFF
// AMD FSR 4 upscaling: bundled (engine/fsrup), so always ready; Anime4K too (its weights are inside the host), ArtCNN too
// (engine/artcnn); Intel XeSS needs Intel's libxess.dll, which its setup row downloads (xess-ready); RTX VSR, AMD FSR 4,
// Anime4K, ArtCNN and Intel XeSS exclude each other (ticking one clears the others), none ticked = Lanczos3
$('fsrup').checked = localStorage.getItem('fsrupOn') === '1'; // default OFF: RTX VSR is the default upscaler
$('a4kup').checked = !$('fsrup').checked && localStorage.getItem('a4kOn') === '1'; // default OFF
$('acnup').checked = !$('fsrup').checked && !$('a4kup').checked && localStorage.getItem('acnOn') === '1'; // default OFF
$('xessup').checked =
  !$('fsrup').checked && !$('a4kup').checked && !$('acnup').checked && localStorage.getItem('xessOn') === '1'; // default OFF
if ($('fsrup').checked || $('a4kup').checked || $('acnup').checked || $('xessup').checked) $('rtxvsr').checked = false;
function fsrUpOn() {
  return $('fsrup').checked && !$('rtxvsr').checked;
}
function a4kUpOn() {
  return $('a4kup').checked && !$('rtxvsr').checked && !$('fsrup').checked;
}
// Anime4K's Mode A+A box (default off): Restore CNN UL on the first x2 pass's 2x frame too; it acts only with the
// Anime4K upscaler, greyed out otherwise (syncUpscale)
$('a4kaa').checked = localStorage.getItem('a4kAA') === '1';
function a4kAaOn() {
  return a4kUpOn() && $('a4kaa').checked;
}
function a4kAaUi() {
  const on = a4kUpOn();
  $('a4kaa').disabled = !on;
  $('a4kaalbl').style.opacity = on ? '1' : '0.45';
  $('a4kaalbl').style.cursor = on ? 'pointer' : 'default';
}
$('a4kaa').onchange = () => {
  localStorage.setItem('a4kAA', $('a4kaa').checked ? '1' : '0');
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
};
function acnUpOn() {
  return $('acnup').checked && !$('rtxvsr').checked && !$('fsrup').checked && !$('a4kup').checked;
}
function xessUpOn() {
  return (
    $('xessup').checked && !$('rtxvsr').checked && !$('fsrup').checked && !$('a4kup').checked && !$('acnup').checked
  );
}
function upOnly(keep) {
  // the upscaler box just ticked stays, the others clear (their stored choices too)
  for (const [id, key] of [
    ['rtxvsr', 'vsrOn'],
    ['fsrup', 'fsrupOn'],
    ['a4kup', 'a4kOn'],
    ['acnup', 'acnOn'],
    ['xessup', 'xessOn'],
  ])
    if (id !== keep && $(id).checked) {
      $(id).checked = false;
      localStorage.setItem(key, '0');
    }
  if (keep !== 'xessup') syncXess(); // its setup row closes with its box
}
$('rtxvsr').onchange = () => {
  localStorage.setItem('vsrOn', $('rtxvsr').checked ? '1' : '0');
  if ($('rtxvsr').checked) upOnly('rtxvsr');
  syncRtx();
  syncUpscale();
  syncVsrQ();
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
};
$('fsrup').onchange = () => {
  localStorage.setItem('fsrupOn', $('fsrup').checked ? '1' : '0');
  if ($('fsrup').checked) upOnly('fsrup');
  syncRtx();
  syncUpscale();
  syncVsrQ();
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
  fsrWarm();
};
$('a4kup').onchange = () => {
  localStorage.setItem('a4kOn', $('a4kup').checked ? '1' : '0');
  if ($('a4kup').checked) upOnly('a4kup');
  syncRtx();
  syncUpscale();
  syncVsrQ();
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
};
$('acnup').onchange = () => {
  localStorage.setItem('acnOn', $('acnup').checked ? '1' : '0');
  if ($('acnup').checked) upOnly('acnup');
  syncRtx();
  syncUpscale();
  syncVsrQ();
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
};
$('xessup').onchange = () => {
  localStorage.setItem('xessOn', $('xessup').checked ? '1' : '0');
  if ($('xessup').checked) upOnly('xessup');
  syncRtx();
  syncUpscale();
  syncVsrQ();
  syncXess();
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
};
// Intel XeSS's setup row (the Dolby Vision tool's flow): shown while the box is ticked and libxess.dll is missing; one
// click downloads Intel's pinned XeSS SDK release (main checks its hash, then copies Intel's files into engine/xess)
let xessReady: any = { ready: false };
function syncXess() {
  if ($('xessup').checked) checkXessReady();
  else $('xesssetup').style.display = 'none';
}
async function checkXessReady() {
  $('xessstatus').textContent = tr('common.checking');
  $('xessstatus').style.color = 'var(--sub)';
  try {
    xessReady = await ipcRenderer.invoke('xess-ready');
  } catch {
    xessReady = { ready: false };
  }
  const line = (xessReady.ready ? '✓ ' : '✗ ') + tr('xess.line');
  if (xessReady.ready) {
    $('xessstatus').textContent = tr('setup.ready');
    $('xessstatus').style.color = 'var(--green)';
    $('xesssteps').innerHTML = line + '<br>' + tr('xess.installed');
  } else {
    $('xessstatus').textContent = tr('dlss.notReady');
    $('xessstatus').style.color = '#e85c5c';
    $('xesssteps').innerHTML =
      line +
      '<br>' +
      tr('xess.howto', {
        ver: xessReady.ver || '',
        mb: xessReady.mb || '73.3',
        link: toolPageLink('xess-open-download'),
      });
  }
  $('xessbtns').style.display = xessReady.ready ? 'none' : '';
  $('xesssetup').style.display = xessReady.ready ? 'none' : 'block';
}
// RTX VSR's quality level 1..4 (default 4 = Ultra, the highest): the slider beside the box, shown while it is ticked; it
// persists, and goes to the host as --vsr-quality when it is not 4
function vsrQuality() {
  const q = Math.round(+$('vsrq').value);
  return q >= 1 && q <= 4 ? q : 4;
}
function syncVsrQ() {
  $('vsrqwrap').style.display = $('rtxvsr').checked ? 'flex' : 'none';
  $('vsrqnum').textContent = String(vsrQuality());
}
const savedVsrQ = localStorage.getItem('vsrQuality');
if (savedVsrQ !== null) $('vsrq').value = savedVsrQ;
syncVsrQ();
$('vsrq').oninput = () => {
  localStorage.setItem('vsrQuality', $('vsrq').value);
  syncVsrQ();
  try {
    lvSendOpts();
  } catch {}
};
$('vsrq').addEventListener('change', () => refreshPreviewIfOpen());
$('rtxhdr').onchange = () => {
  localStorage.setItem('rtxhdrOn', $('rtxhdr').checked ? '1' : '0');
  syncRtx();
  syncHdrColor();
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
};
syncRtx();
i18nRelabel.push(syncRtx);
// Populate readiness on load so the upscale backend label is right even before the RTX panel opens.
ipcRenderer.invoke('rtx-ready').then((r) => {
  if (r) rtxReady = r;
  syncUpscale();
  syncPreview();
  // Live's VSR default depends on rtxReady, which just arrived: re-send + refresh the fill hint
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
});

// NVIDIA DLSS 5 (Neural Rendering; opt-in; both modes: renders run it per output frame, Live
// once per captured frame before the smoothing, lvSendOpts carries it):
// a DLAA-class per-frame pass at the output resolution, run by the native host through the caller
// shim engine/dlssnr/nvngx.dll (ships)
// plus the NR runtime nvngx_dlssnr.dll, which the app never ships (NVIDIA offers no public download,
// the only NVIDIA copy is inside NBA 2K27): one click downloads the community build every DLSS 5 tool
// uses (rhi-repo on GitHub, checksum-verified twice in main), or the user drops a copy in, like the
// RTX Video / NvOFFRUC DLLs.
// Readiness = the shim + runtime present (dlssnr-ready). The two sliders are NVIDIA's global developer
// controls, Structure Intensity and Tone Intensity (0..2, default 1.00); the pass always runs at
// full resolution (no quality selector, by decision).
let dlssnrReady: any = { ready: false };
function nrOn() {
  return $('dlssnr').checked && !!dlssnrReady.ready;
}
function nrStructure() {
  return parseFloat($('nrstructure').value) || 0;
}
function nrTone() {
  return parseFloat($('nrtone').value) || 0;
}
function nrStyle() {
  const b = $('nrstyleseg').querySelector('button.on');
  return b ? parseInt(b.dataset.style, 10) : 0;
} // DLSSNR.Style 0/1/2
function nrPasses() {
  const v = parseInt($('nrpasses').value, 10);
  return v >= 1 && v <= 20 ? v : 1;
} // DLSS 5 chained 1..20 times per frame
function nrMaskOn() {
  return nrOn() && $('nrmask').checked;
} // preview-only heat map of the DLSS 5 change
function syncDlssnr() {
  const on = $('dlssnr').checked;
  $('nrwrap').style.display = on ? 'inline-flex' : 'none';
  if (on) checkDlssnrReady();
  else $('dlssnrsetup').style.display = 'none';
}
async function checkDlssnrReady() {
  $('dlssnrstatus').textContent = tr('common.checking');
  $('dlssnrstatus').style.color = 'var(--sub)';
  try {
    dlssnrReady = await ipcRenderer.invoke('dlssnr-ready');
  } catch {
    dlssnrReady = { ready: false };
  }
  const r = dlssnrReady;
  const lines = [
    (r.host ? '✓ ' : '✗ ') + tr('nr.line.host'),
    (r.runtime ? '✓ ' : '✗ ') + tr('nr.line.runtime', { file: r.file || 'nvngx_dlssnr.dll' }),
    (r.sr ? '✓ ' : '○ ') + tr('nr.line.sr', { file: r.srFile || 'nvngx_dlss.dll' }),
  ];
  if (r.ready) {
    $('dlssnrstatus').textContent = tr('setup.ready');
    $('dlssnrstatus').style.color = 'var(--green)';
    $('dlssnrsteps').innerHTML = lines.join('<br>') + '<br>' + tr('nr.installed');
  } else {
    $('dlssnrstatus').textContent = tr('dlss.notReady');
    $('dlssnrstatus').style.color = '#e85c5c';
    const detail =
      r.host === false
        ? tr('nr.noBridge')
        : tr('nr.howto', { mb: (r.download && r.download.mb) || 111, link: rhiLink() });
    $('dlssnrsteps').innerHTML = lines.join('<br>') + '<br>' + detail;
  }
  $('dlssnrbtns').style.display = r.ready ? 'none' : ''; // hide the buttons once the runtime is in place
  $('dlssnrsetup').style.display = r.ready ? 'none' : 'block';
  refreshPreviewIfOpen(); // the processed pane depends on readiness
  refresh(); // so does the interp-off output name (_dlss5 needs nrOn())
  // the Live hint and the live options read nrOn() = ticked AND ready: a tick (or the boot
  // restore) made before this async answer lands computed them as off, so re-run them now
  // (otherwise the fit hint stays stale after the tick)
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
}
// the DLSS 5 runtime's download page as a link (main opens it): its text is the language's, its action the code's
function rhiLink() {
  return (
    '<a href="#" onclick="ipcRenderer.invoke(\'dlssnr-open-download\');return false;">' + tr('nr.rhiLink') + '</a>'
  );
}
i18nRelabel.push(() => syncDlssnr());
// One-click download (main does fetch + checksum + extract + copy); progress lands in the status line.
let dlssnrDownloading = false;
ipcRenderer.on('dlssnr-progress', (_e, p) => {
  if (!dlssnrDownloading) return;
  const mb = (x) => (x / 1048576).toFixed(0);
  $('dlssnrstatus').textContent =
    p.stage === 'verify' ? tr('dl.verify') : tr('dl.progress', { got: mb(p.received), total: mb(p.total) });
});
async function doDlssnrDownload() {
  if (dlssnrDownloading) return;
  dlssnrDownloading = true;
  $('dlssnrdl').disabled = true;
  $('dlssnrsetup').style.display = 'block';
  $('dlssnrstatus').textContent = tr('dl.start');
  $('dlssnrstatus').style.color = 'var(--sub)';
  let r: any = {};
  try {
    r = await ipcRenderer.invoke('dlssnr-download');
  } catch (e) {
    r = { ok: false, error: String(e) };
  }
  dlssnrDownloading = false;
  $('dlssnrdl').disabled = false;
  await checkDlssnrReady();
  if (!r.ok) {
    $('dlssnrstatus').textContent = tr('dl.failed');
    $('dlssnrstatus').style.color = '#e85c5c';
    $('dlssnrsteps').innerHTML = '✗ ' + (r.error || tr('dl.failedShort')) + '<br>' + tr('dl.retry');
    $('dlssnrsetup').style.display = 'block';
  }
}
$('dlssnrdl').onclick = doDlssnrDownload;
if (localStorage.getItem('dlssnrOn') === '1') $('dlssnr').checked = true; // default OFF
$('dlssnr').onchange = () => {
  localStorage.setItem('dlssnrOn', $('dlssnr').checked ? '1' : '0');
  syncDlssnr();
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
};
for (const [id, key] of [
  ['nrstructure', 'nrStructure'],
  ['nrtone', 'nrTone'],
]) {
  const saved = localStorage.getItem(key);
  if (saved !== null) $(id).value = saved;
  $(id + 'num').textContent = (+$(id).value).toFixed(2);
  $(id).oninput = () => {
    localStorage.setItem(key, $(id).value);
    $(id + 'num').textContent = (+$(id).value).toFixed(2);
  };
  $(id).addEventListener('change', () => {
    refreshPreviewIfOpen();
    try {
      lvSendOpts();
    } catch {}
  });
}
{
  const saved = localStorage.getItem('nrStyle'); // Style selector: persisted, default NVIDIA's Default (0)
  if (saved !== null)
    for (const b of $('nrstyleseg').querySelectorAll('button')) b.classList.toggle('on', b.dataset.style === saved);
}
$('nrstyleseg').onclick = (e) => {
  const b = e.target.closest('button');
  if (!b || b.disabled) return;
  for (const x of $('nrstyleseg').querySelectorAll('button')) x.classList.toggle('on', x === b);
  localStorage.setItem('nrStyle', b.dataset.style);
  refreshPreviewIfOpen();
  try {
    lvSendOpts();
  } catch {}
};
{
  const saved = localStorage.getItem('nrPasses'); // Passes slider 1..20: persisted, default 1
  if (saved !== null && +saved >= 1 && +saved <= 20) $('nrpasses').value = saved;
  $('nrpassesnum').textContent = $('nrpasses').value;
}
$('nrpasses').oninput = () => {
  localStorage.setItem('nrPasses', $('nrpasses').value);
  $('nrpassesnum').textContent = $('nrpasses').value;
};
$('nrpasses').addEventListener('change', () => {
  refreshPreviewIfOpen();
  try {
    lvSendOpts();
  } catch {}
});
if (localStorage.getItem('nrMask') === '1') $('nrmask').checked = true; // default OFF
$('nrmask').onchange = () => {
  localStorage.setItem('nrMask', $('nrmask').checked ? '1' : '0');
  refreshPreviewIfOpen();
};
syncDlssnr();

// Interpolation model: GMFSS (default) vs "NVIDIA Smooth Motion" (NVIDIA Optical Flow FRUC). Two
// checkboxes acting as an exclusive pair (user-chosen UX): ticking one clears the other, and
// unticking the active one hands the choice back to its sibling, so exactly one model is always
// selected. The FRUC runtime (NvOFFRUC.dll) is NVIDIA proprietary + user-installed from the Optical
// Flow SDK .zip. The Smooth Motion checkbox is ALWAYS clickable (ticking it visibly clears GMFSS);
// it just reveals a one-time installer when the runtime is missing, and a render only uses FRUC once
// it is installed - otherwise it falls back to GMFSS (see modelIsFruc + the run payload gate below).
let frucReady = false,
  frucBridge = false,
  dlssReady = false,
  dlssMissing = [];
let dlssDefault = false; // DLSS 4.5 is the default model: its files present, an RTX 40 / 50 GPU, GPU scheduling on
function modelIsFruc() {
  return $('modelfruc').checked && frucReady;
}
function modelIsDlss() {
  return $('modeldlss').checked && dlssReady;
}
// RIFE ships bundled (weights vendored in engine/rife), so it is always ready.
function modelIsRife() {
  return $('modelrife').checked;
}
// The DRBA sub-checkbox (default OFF) adds anime-pacing timing on top of the same RIFE weights.
function modelIsRifeDrba() {
  return modelIsRife() && $('rifedrba').checked;
}
// NVIDIA Optical Flow: the driver's optical-flow hardware run by
// smv-live.exe itself (engine --nvof, live backend nvof), so it is always ready like RIFE.
function modelIsNvof() {
  return $('modelnvof').checked;
}
// AMD FSR frame generation: our bridge ships in engine/fsrfg (engine --fsrfg, live backend fsrfg), always ready
function modelIsFsrfg() {
  return $('modelfsrfg').checked;
}
// AMD FSR 4's first session after a driver update waits ~15 to 30 s for the driver's shader compile: while an FSR route
// is picked, main warms both routes' shaders in a tiny background session once per driver (fsr-warm; a no-op when warm)
function fsrWarm() {
  const fg = modelIsFsrfg(),
    up = $('fsrup').checked;
  if (fg || up) ipcRenderer.send('fsr-warm', { fg, up });
}
// Intel XeSS-FG: our bridge ships in engine/xess (engine --xess-fg, live backend fsrfg + --xess-fg); Intel's
// libxess_fg.dll + libxell.dll come from the Intel XeSS setup's one click (xess-ready's fg), so it is ready once they are
let xefgReady = false;
function modelIsXefg() {
  return $('modelxefg').checked && xefgReady;
}
const MODEL_BOXES = () => [
  $('modelgmfss'),
  $('modelrife'),
  $('modeldlss'),
  $('modelfruc'),
  $('modelfsrfg'),
  $('modelxefg'),
  $('modelnvof'),
];
// The run payload's model ('' = GMFSS, the CLI's default, or interpolation off); Compare's model A reads it too, so a
// compare runs exactly the model a render would.
function renderModel() {
  if (!interpOn()) return '';
  return modelIsRifeDrba()
    ? 'rifedrba'
    : modelIsRife()
      ? 'rife'
      : modelIsDlss()
        ? 'dlssg'
        : modelIsFruc()
          ? 'fruc'
          : modelIsNvof()
            ? 'nvof'
            : modelIsFsrfg()
              ? 'fsrfg'
              : modelIsXefg()
                ? 'xefg'
                : '';
}
function frucSetupHtml() {
  return tr('fruc.intro') + '<br>' + (!frucBridge ? tr('fruc.noBridge') : tr('fruc.howto'));
}
async function refreshFrucState() {
  try {
    const r = await ipcRenderer.invoke('fruc-ready');
    frucReady = !!(r && r.ready);
    frucBridge = !!(r && r.bridge);
  } catch {
    frucReady = false;
    frucBridge = false;
  }
  // DLSS ("DLSS 4.5" Frame Generation) ships bundled, so not-ready means files were deleted from
  // engine/dlssg (a broken install); GPU/driver support is only knowable at render time.
  try {
    const d = await ipcRenderer.invoke('dlssg-ready');
    dlssReady = !!(d && d.ready);
    dlssMissing = (d && d.missing) || [];
    dlssDefault = dlssReady && !!(d && d.supported);
  } catch {
    dlssReady = false;
    dlssMissing = [];
    dlssDefault = false;
  }
  await refreshXefgState();
}
let xefgInfo: any = { fg: false };
async function refreshXefgState() {
  try {
    xefgInfo = await ipcRenderer.invoke('xess-ready');
  } catch {
    xefgInfo = { fg: false };
  }
  xefgReady = !!(xefgInfo && xefgInfo.fg);
}
// Intel XeSS-FG's setup row (the Intel XeSS upscaler's flow): shown while its box is ticked and Intel's two DLLs are
// missing; one click downloads Intel's pinned XeSS SDK release (main checks its hash, then installs Intel's files)
function syncXefgSetup() {
  if (!$('modelxefg').checked || xefgReady) {
    $('xefgsetup').style.display = 'none';
    return;
  }
  $('xefgstatus').textContent = tr('dlss.notReady');
  $('xefgstatus').style.color = '#e85c5c';
  $('xefgsteps').innerHTML =
    '✗ ' +
    tr('xefg.line') +
    '<br>' +
    tr('xess.howto', { ver: xefgInfo.ver || '', mb: xefgInfo.mb || '73.3', link: toolPageLink('xess-open-download') });
  $('xefgbtns').style.display = '';
  $('xefgsetup').style.display = 'block';
}
// Reflect the current model choice: show a setup hint only while the chosen model's runtime is
// missing (FRUC has a one-time DLL installer).
function syncModel() {
  localStorage.setItem(
    'interpModel',
    !interpOn()
      ? 'none' // nothing ticked = interpolation off, remembered
      : $('modelfruc').checked
        ? 'fruc'
        : $('modelfsrfg').checked
          ? 'fsrfg'
          : $('modelxefg').checked
            ? 'xefg'
            : $('modeldlss').checked
              ? 'dlssg'
              : $('modelrife').checked
                ? 'rife'
                : $('modelnvof').checked
                  ? 'nvof'
                  : 'gmfss',
  );
  // The DRBA sub-option only makes sense while RIFE is the chosen model (always ready: bundled).
  $('rifedrbarow').style.display = $('modelrife').checked ? '' : 'none';
  if ($('modeldlss').checked && !dlssReady) {
    $('dlsssteps').innerHTML = tr('dlssg.incomplete', { files: dlssMissing.join(', ') });
    $('dlsssetup').style.display = 'block';
  } else {
    $('dlsssetup').style.display = 'none';
  }
  if ($('modelfruc').checked && !frucReady) {
    $('frucstatus').textContent = tr('dlss.notReady');
    $('frucstatus').style.color = '#e85c5c';
    $('frucsteps').innerHTML = frucSetupHtml();
    $('frucbtns').style.display = '';
    $('frucsetup').style.display = 'block';
  } else {
    $('frucsetup').style.display = 'none';
  }
  syncXefgSetup();
}
// Exclusive-group behaviour: ticking a box clears the others. Unticking the ACTIVE box now leaves
// NOTHING checked, and that is the "no interpolation" state (the group replaced the old separate
// Interpolate checkbox), so syncInterp() runs to grey the Speed controls and retag the output.
function pickModel(box) {
  for (const b of MODEL_BOXES()) if (b !== box) b.checked = false;
  syncModel();
  syncInterp();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {} // live inherits this model choice
  fsrWarm();
  syncCompareUi();
  if (cmpOn()) refreshCompare();
  else refreshPreviewIfOpen(); // Compare's model A is this choice
}
async function doFrucInstall(source) {
  $('frucsetup').style.display = 'block';
  $('frucstatus').textContent = tr('common.installing');
  $('frucstatus').style.color = 'var(--sub)';
  let r: any = {};
  try {
    r = await ipcRenderer.invoke('fruc-install', source);
  } catch (e) {
    r = { ok: false, error: String(e) };
  }
  await refreshFrucState();
  if (frucReady) {
    $('modelfruc').checked = true;
    $('modelgmfss').checked = false;
    localStorage.setItem('interpModel', 'fruc');
    $('frucsetup').style.display = 'none';
  } else if (r.ok) {
    // NvOFFRUC.dll copied fine; readiness just needs the one-time bridge build
    $('frucstatus').textContent = tr('fruc.dllInstalled');
    $('frucstatus').style.color = 'var(--green)';
    $('frucsteps').innerHTML = tr('fruc.lastStep');
  } else {
    $('frucstatus').textContent = tr('dlss.notReady');
    $('frucstatus').style.color = '#e85c5c';
    $('frucsteps').innerHTML = '✗ ' + (r.error || tr('common.installFailed')) + '<br>' + frucSetupHtml();
  }
}
i18nRelabel.push(syncModel);
$('frucget').onclick = () => ipcRenderer.invoke('fruc-open-download');
$('frucbrowsezip').onclick = async () => {
  const p = await ipcRenderer.invoke('fruc-choose');
  if (p) doFrucInstall(p);
}; // selecting a .zip auto-installs
for (const b of ['modelgmfss', 'modelrife', 'modeldlss', 'modelfruc', 'modelfsrfg', 'modelxefg', 'modelnvof'])
  $(b).onchange = () => pickModel($(b));
// The DRBA sub-option persists on its own key (default OFF: max fluidity is the app's default
// philosophy; DRBA deliberately keeps character cadence); toggling it never clears RIFE.
$('rifedrba').checked = localStorage.getItem('rifeDrba') === '1';
$('rifedrba').onchange = () => {
  localStorage.setItem('rifeDrba', $('rifedrba').checked ? '1' : '0');
  try {
    lvSendOpts();
  } catch {}
};
// On load: learn readiness, restore the saved choice, then reflect it (a setup hint shows if the
// chosen model's runtime is missing). Skipped while Interpolate is off: syncInterp() has cleared
// the boxes and re-checking Interpolate restores the choice itself.
function restoreModelChoice() {
  const saved = localStorage.getItem('interpModel');
  if (saved === 'none') {
    // interpolation off: leave the whole group unticked
    for (const b of MODEL_BOXES()) b.checked = false;
    return;
  }
  // no saved choice (a first start): DLSS 4.5 where it runs, else GMFSS
  const box =
    saved === null
      ? dlssDefault
        ? 'modeldlss'
        : 'modelgmfss'
      : saved === 'fruc'
        ? 'modelfruc'
        : saved === 'fsrfg'
          ? 'modelfsrfg'
          : saved === 'xefg'
            ? 'modelxefg'
            : saved === 'dlssg'
              ? 'modeldlss'
              : saved === 'rife'
                ? 'modelrife'
                : saved === 'nvof'
                  ? 'modelnvof'
                  : 'modelgmfss';
  for (const b of MODEL_BOXES()) b.checked = false;
  $(box).checked = true;
}
// Restore FIRST (the saved choice may be "none"), then reflect it: syncInterp greys the Speed
// controls when nothing is ticked.
refreshFrucState().then(() => {
  restoreModelChoice();
  syncModel();
  syncInterp();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
  syncCompareUi(); // Compare's choices follow the runtimes found ready
  // a saved FSR pick warms after the first paint (a spawn during launch stalls the compositor), below normal priority;
  // a window opened behind another one never paints (no rAF), so a timer starts it there
  let bootWarm = false;
  const warmOnce = () => {
    if (!bootWarm) {
      bootWarm = true;
      fsrWarm();
    }
  };
  requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(warmOnce, 2000)));
  setTimeout(warmOnce, 5000);
});

// --- Live mode (real-time DLSS-G overlay; engine/dlssg/smv-live.exe) -------------------------------
// Independent of the file workflow: usable straight from the welcome screen. Two ways in, both
// ending at the exe's --fg mode (whatever window is foreground at spawn goes live): the ` global
// hotkey (instant, main-side), and the Smooth It Live! button, which arms a 5s countdown so the
// user can click the target window first (the Lossless Scaling "Scale" pattern, no keyboard
// needed). main forwards the exe's stderr as lv-out and auto-restarts on exit 4 (target resized).
let lvState = 'idle'; // idle | arming (countdown running) | running
let lvStopReq = false;
let lvTimer = null;
// the hotkey + button share these settings; main mirrors them for hotkey-started sessions
// Live's output target is INHERITED from the Speed selectors (screen-match or the fps box): the
// user sets ONE target for both file renders and live smoothing. Live inherits the Speed MODE,
// not a number derived from a loaded file (that coupling is exactly what made Live target 60
// just because the bundled sample clip is 30 fps).
function liveTargetFps() {
  return Math.round(speedMode() === 'screen' ? screenHz : curFps()); // 359.979 Hz panels read as 360
}
// ONE model choice drives file renders AND live smoothing: live derives its engine from the
// Interpolation model group. Models without a live backend fall back with a visible note
// (FRUC -> GMFSS live; DRBA -> plain RIFE live: the triple-window timing needs a future
// live implementation).
function liveModelInfo() {
  // fixed: the model cannot resample to the exact fps target; it approximates it with a whole
  // multiplier of the captured window's own rate (derived in the exe from the target) - DLSS-G
  // by design (Streamline; past one pass's limit in two passes)
  if (modelIsDlss()) return { model: 'dlssg', name: 'DLSS 4.5', note: '', fixed: true };
  // DRBA live: the rifedrba backend renders the three-frame window, so the picture trails
  // the capture by one source frame
  if (modelIsRifeDrba()) return { model: 'rifedrba', name: 'RIFE (DRBA)', note: 'adds one source frame of delay' };
  if (modelIsRife()) return { model: 'rife', name: 'RIFE', note: '' };
  // Smooth Motion live: every fraction takes the nearest node of the pair's midpoint tree, so
  // the fruc backend is adaptive like rife/gmfss; HDR-capable like rife (the bridge warps 8-bit PQ
  // for the tweens, the real frames keep full precision)
  if (modelIsFruc()) return { model: 'fruc', name: 'Smooth Motion', note: 'tweens at 8-bit precision' };
  // AMD FSR frame generation live: Smooth Motion's route (the nearest node of the pair's midpoint tree), its own bridge
  if (modelIsFsrfg()) return { model: 'fsrfg', name: 'AMD FSR 4', note: '' };
  // Intel XeSS-FG live: AMD FSR frame generation's route with its own bridge (main passes --backend fsrfg --xess-fg)
  if (modelIsXefg()) return { model: 'xefg', name: 'Intel XeSS-FG', note: '' };
  // NVIDIA Optical Flow: adaptive like rife (the splat takes any fraction), native only
  if (modelIsNvof()) return { model: 'nvof', name: 'NVIDIA Optical Flow', note: '' };
  // nothing ticked = interpolation off (same meaning as file renders): the echo backend passes
  // the captured frames through at their own rate and applies the live effects (Restore,
  // Sharpen, Upscale to, RTX VSR, RTX HDR), so the effects work without interpolating.
  if (!interpOn()) return { model: 'echo', name: 'No interpolation', note: '', effectsOnly: true };
  return { model: 'gmfss', name: 'GMFSS', note: '' };
}
// Upscaler policy: live FILL and file renders upscale with RTX VSR by default when its runtime is
// present; the shared RTX VSR checkbox is the opt-out (unticked = Lanczos3 everywhere).
function liveVsrOn() {
  return !!(rtxReady.vsr && $('rtxvsr').checked);
}
// Live TrueHDR: STRICT opt-in via the shared RTX HDR checkbox (a TrueHDR
// expansion is a deliberate look change, so live matches file renders, not the VSR silent
// default). Applies to every model on HDR screens; the exe drops it elsewhere. No
// srcHdr gate here: live has no loaded file, the source is whatever window gets captured.
function liveHdrOn() {
  return !!(rtxReady.hdr && $('rtxhdr').checked);
}
// The exe's loading note / HUD names what is actually ticked: the model, then the live effects it applies,
// e.g. "GMFSS + AnimeJaNai HD + DLSS 5 + RTX VSR" or "DLSS 4.5 + Sharpen"; Restore by its picked row's name, the
// upscaler that takes the live resize when ticked; sharpen = Sharpen's word (English for the exe's log, the GUI
// language's for the HUD)
function liveLoadParts(mi, sharpen: string) {
  const parts = mi.effectsOnly ? [] : [mi.name];
  if (restoreOn()) parts.push(ajOn() ? 'AnimeJaNai HD' : a4kRestOn() ? 'Anime4K Restore' : 'Real-ESRGAN');
  if (nrOn()) parts.push('DLSS 5');
  const up = liveVsrOn()
    ? 'RTX VSR'
    : fsrUpOn()
      ? 'AMD FSR 4.1'
      : a4kUpOn()
        ? a4kAaOn()
          ? 'Anime4K A+A'
          : 'Anime4K'
        : acnUpOn()
          ? 'ArtCNN'
          : xessUpOn()
            ? 'Intel XeSS'
            : '';
  if (up) parts.push(up);
  if ($('sharpen').checked && (parseFloat($('sharpval').value) || 0) > 0) parts.push(sharpen);
  if (liveHdrOn()) parts.push('RTX HDR');
  return parts;
}
function liveLoadLabel(mi) {
  const parts = liveLoadParts(mi, 'Sharpen');
  return parts.length ? parts.join(' + ') : 'passthrough (nothing ticked)';
}
// the HUD's opening note in the GUI language, shown by the exe as is until the first fps reading
function liveHudNote(mi) {
  const parts = liveLoadParts(mi, tr('sharpen.k'));
  return tr('lv.hud.loading', { what: parts.length ? parts.join(' + ') : tr('lv.hud.passthrough') });
}
i18nRelabel.push(() => {
  try {
    lvSendOpts();
  } catch {}
});
function lvSendOpts() {
  const mi = liveModelInfo();
  const hp = hdrColorPayload();
  ipcRenderer.send('lv-opts', {
    // ONE target source: the Speed selectors. Adaptive models (RIFE/GMFSS) resample to
    // it; fixed models (DLSS 4.5) approximate it with a whole multiplier the exe derives
    // from the captured window's measured rate.
    model: mi.model,
    label: liveLoadLabel(mi), // what is ticked, for the exe's loading note + substitution note
    hudnote: liveHudNote(mi),
    note: mi.note, // message/HUD (the raw backend id reads as the wrong model)
    dlssmode: dlssMode(), // the working size's share of the presented size
    fit: $('lvfit').value,
    target: liveTargetFps(),
    // live effects: sharpen inherits the file-render setting; RTX VSR (liveVsrOn above) takes the live
    // resize that enlarges, the host picks it (none enlarges = skipped)
    sharpen: $('sharpen').checked ? parseFloat($('sharpval').value) || 0 : 0,
    rtxvsr: liveVsrOn(),
    vsrquality: vsrQuality(),
    fsrup: fsrUpOn(), // AMD FSR 4 takes the same resize when ticked instead of RTX VSR
    anime4k: a4kUpOn(), // Anime4K the same way
    a4kaa: a4kAaOn(), // its Mode A+A
    artcnn: acnUpOn(), // ArtCNN the same way
    xess: xessUpOn(), // Intel XeSS the same way
    // no "Upscale to": Live's size is what Display presents (the window or the screen), the DLSS mode its speed
    // Restore (AI detail): Real-ESRGAN or AnimeJaNai first on every presented frame
    restore: restoreOn(),
    restoreModel: restoreModelId(),
    // NVIDIA DLSS 5: Neural Rendering once per captured frame (runtime installed), python route
    dlssnr: nrOn(),
    nrstructure: nrStructure(),
    nrtone: nrTone(),
    nrstyle: nrStyle(),
    nrpasses: nrPasses(),
    // live TrueHDR (SDR window -> HDR out) + the RTX HDR tone/colour knobs it inherits
    rtxhdr: liveHdrOn(),
    hdrcolor: hp.color,
    hdrsat: hp.saturation,
    hdrcon: sdkCon(),
    hdrvib: effVibrance(),
    hdrsb: effSatBoost(),
    hud: $('lvhud').checked,
    hudlat: $('lvhudlat').checked,
    gpufit: $('gpufit').checked,
  });
  lvTargetUi(mi);
}
// the line beside Smooth It Live!: the model, how it meets the Speed target, its load time (mi.name / mi.note above stay
// English: they also go to the host's own overlay)
function lvTargetUi(mi = liveModelInfo()) {
  const load =
    mi.model === 'rife'
      ? tr('lv.load.rife')
      : mi.model === 'rifedrba'
        ? tr('lv.load.drba')
        : mi.model === 'gmfss'
          ? tr('lv.load.gmfss')
          : mi.model === 'nvof' || mi.model === 'fsrfg' || mi.model === 'xefg'
            ? tr('lv.load.fast')
            : '';
  // A fixed model cannot hit an arbitrary target exactly: it picks the first whole multiple
  // of the captured window's own rate that reaches it, so say that instead of promising the exact number.
  const lvT = liveTargetFps();
  $('lvtarget').textContent =
    (mi.effectsOnly ? tr('lv.none') : mi.name) +
    ' · ' +
    (mi.effectsOnly
      ? tr('lv.effectsOnly', {
          fps: lvT,
          restore: tr('restore.k'),
          up: tr('up.k'),
          nr: tr('nr.k'),
          sharpen: tr('sharpen.k'),
          hdr: tr('hdr.k'),
        })
      : mi.fixed
        ? tr('lv.fixed', { fps: lvT }) + (mi.model === 'dlssg' ? ' ' + tr('lv.fixed.dlssg') : '')
        : tr('lv.targets', { fps: lvT })) +
    (load ? ' · ' + load : '') +
    (liveHdrOn() && ['rife', 'rifedrba', 'gmfss', 'nvof', 'dlssg'].includes(mi.model) ? ' · ' + tr('lv.rtxhdr') : '') +
    (mi.note ? ' · ' + tr('lv.note.' + mi.model) : '') +
    '  ·  ';
}
// Live's upscalers enlarge only what reaches them: Whole screen captures the screen as the player shows it (a fullscreen
// player has already stretched the video with its own scaler), and Original size at DLAA leaves nothing to enlarge; a
// ticked upscaler says so under its rows, with the setup that hands it the video's own frames (a window at the video's
// size, Fill screen). The names come from the Display row's own labels.
function upFitHintUi() {
  const fit = $('lvfit').value;
  const key = !(liveVsrOn() || fsrUpOn() || a4kUpOn() || acnUpOn() || xessUpOn())
    ? ''
    : fit === 'monitor'
      ? 'up.live.monitor'
      : fit === 'window' && dlssPct() >= 100
        ? 'up.live.window'
        : '';
  $('upfithint').textContent = key
    ? tr(key, {
        whole: tr('live.fit.monitor'),
        orig: tr('live.fit.window'),
        fill: tr('live.fit.fill'),
        display: tr('live.display'),
      })
    : '';
  $('upfithintrow').style.display = key ? '' : 'none';
}
function lvModelUi() {
  // every display mode works with every model (DLSS 4.5 with an effect, Fill included, runs the effects route
  // before its frame generation)
  $('lvfit').disabled = lvState === 'running';
  // Restore applies live too; named here like Upscale to
  const resTxt = restoreOn()
    ? tr('lv.fit.restore', {
        model: tr(ajOn() ? 'restoreaj.label' : a4kRestOn() ? 'restorea4k.label' : 'restore.label'),
      })
    : '';
  // DLSS 5 applies live too: once per captured frame
  const nrTxt = nrOn() ? tr('lv.fit.nr') : '';
  const effTxt = [resTxt, nrTxt].filter(Boolean).join(' · ');
  $('lvfithint').textContent =
    $('lvfit').value === 'fill'
      ? tr('lv.fit.fill') +
        ' · ' +
        (liveVsrOn()
          ? tr('lv.fit.vsr')
          : fsrUpOn()
            ? tr('lv.fit.fsr')
            : a4kUpOn()
              ? tr('lv.fit.a4k')
              : acnUpOn()
                ? tr('lv.fit.acn')
                : xessUpOn()
                  ? tr('lv.fit.xess')
                  : tr('lv.fit.up')) +
        (effTxt ? ' · ' + effTxt : '')
      : $('lvfit').value === 'monitor'
        ? tr('lv.fit.monitor') + (effTxt ? ' · ' + effTxt : '')
        : effTxt;
  upFitHintUi();
}
function lvUi() {
  const b = $('lvgo');
  if (lvState === 'running') b.textContent = tr('lv.stop');
  else if (lvState === 'idle') b.textContent = tr('live.go');
  // arming: the countdown tick owns the button text ("Cancel (N)")
  modeBtnUi(lvState !== 'idle');
  lvModelUi();
}
lvUi();
i18nRelabel.push(() => {
  lvUi();
  lvTargetUi();
});
function lvDisarm() {
  if (lvTimer) {
    clearInterval(lvTimer);
    lvTimer = null;
  }
  if (lvState === 'arming') {
    lvState = 'idle';
    lvUi();
    raw('lvstat', '');
  }
}
// The fps target is a spawn-time argument (--target, and the exe derives the slot
// count from it), so a Speed change during a running session relaunches the session like a DLSS
// mode change: a 700 ms debounce, 'lv-restart', same target window. Only a real
// change of the EFFECTIVE target (what main.ts would pass) restarts anything.
function lvEffTarget() {
  return Math.min(FPS_TOP, Math.max(10, Math.round(liveTargetFps())));
}
let lvSpawnTarget = 0; // effective target the running session was spawned with
let lvTargetRestartT = null;
function lvTargetChanged() {
  if (lvState !== 'running') return;
  clearTimeout(lvTargetRestartT);
  lvTargetRestartT = setTimeout(() => {
    if (lvState === 'running' && lvEffTarget() !== lvSpawnTarget) ipcRenderer.send('lv-restart');
  }, 700);
}
$('lvfit').onchange = () => {
  localStorage.setItem('lvFit', $('lvfit').value);
  lvModelUi();
  lvSendOpts();
};
// on-screen readout toggle (default ON; off spawns the exe with --no-hud)
if (localStorage.getItem('lvHud') === '0') $('lvhud').checked = false;
// latency segment of the same readout (default ON = today's behaviour; off spawns
// --no-hud-latency). Only meaningful while the meter itself is shown, so it greys out
// with the meter rather than pretending to do something.
if (localStorage.getItem('lvHudLat') === '0') $('lvhudlat').checked = false;
function lvHudLatUi() {
  const on = $('lvhud').checked;
  $('lvhudlat').disabled = !on;
  $('lvhudlatlbl').style.opacity = on ? '1' : '0.45';
  $('lvhudlatlbl').style.cursor = on ? 'pointer' : 'default';
}
$('lvhud').onchange = () => {
  localStorage.setItem('lvHud', $('lvhud').checked ? '1' : '0');
  lvHudLatUi();
  lvSendOpts();
};
$('lvhudlat').onchange = () => {
  localStorage.setItem('lvHudLat', $('lvhudlat').checked ? '1' : '0');
  lvSendOpts();
};
// "Fit to the GPU" (default ON): a spawn-time flag on both routes (off = --no-gpu-fit), so a running Live session restarts
if (localStorage.getItem('gpuFit') === '0') $('gpufit').checked = false;
$('gpufit').onchange = () => {
  localStorage.setItem('gpuFit', $('gpufit').checked ? '1' : '0');
  lvSendOpts();
  if (lvState === 'running') ipcRenderer.send('lv-restart');
};
localStorage.removeItem('lvNative'); // retired: the RIFE live route always runs inside smv-live.exe
lvHudLatUi();
// Hotkey recorder: click the button, press the new key or combo; Esc cancels. Electron
// accelerator strings ("`", "F9", "Ctrl+Alt+S"); main answers whether registration
// succeeded (another app can hold the key) and keeps the old key on failure.
let lvKeyCapture = false;
function lvAccelFrom(e) {
  if (['Control', 'Alt', 'Shift', 'Meta'].includes(e.key)) return null; // bare modifier: keep waiting
  const mods = [];
  if (e.ctrlKey) mods.push('Ctrl');
  if (e.altKey) mods.push('Alt');
  if (e.shiftKey && e.key.length > 1) mods.push('Shift'); // printable keys already carry shift in e.key
  const key = e.key === ' ' ? 'Space' : e.key.length === 1 ? e.key.toUpperCase() : e.key;
  return [...mods, key].join('+');
}
$('lvkey').onclick = () => {
  if (lvKeyCapture || lvState !== 'idle') return;
  lvKeyCapture = true;
  const old = $('lvkey').textContent;
  txt('lvkey', () => tr('lv.key.press'));
  txt('lvkeyhint', () => tr('lv.key.esc'));
  const done = (key: string, hint?: () => string) => {
    lvKeyCapture = false;
    raw('lvkey', key);
    if (hint) txt('lvkeyhint', hint);
    else raw('lvkeyhint', '');
    window.removeEventListener('keydown', h, true);
  };
  const h = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === 'Escape') {
      done(old);
      return;
    }
    const acc = lvAccelFrom(e);
    if (!acc) return;
    const ok = await ipcRenderer.invoke('lv-hotkey', acc);
    if (ok) {
      localStorage.setItem('lvHotkey', acc);
      done(acc);
    } else done(old, () => tr('lv.key.fail', { key: acc }));
  };
  window.addEventListener('keydown', h, true);
};
// restore the persisted hotkey at boot (main registers the default ` on its own)
{
  const savedKey = localStorage.getItem('lvHotkey');
  if (savedKey && savedKey !== '`')
    ipcRenderer.invoke('lv-hotkey', savedKey).then((ok) => {
      if (ok) raw('lvkey', savedKey);
    });
}
$('lvgo').onclick = async () => {
  if (lvState === 'running') {
    lvStopReq = true;
    try {
      await ipcRenderer.invoke('lv-stop');
    } catch {} // lv-done resets the UI
    return;
  }
  if (lvState === 'arming') {
    lvDisarm();
    return;
  }
  // arm: give the user 5s to focus the window they want smoothed, then target the foreground
  lvState = 'arming';
  let n = 5;
  const tick = () => {
    if (n <= 0) {
      if (lvTimer) {
        clearInterval(lvTimer);
        lvTimer = null;
      }
      $('lvgo').textContent = tr('lv.starting');
      ipcRenderer.send('lv-start-fg');
      return;
    }
    $('lvgo').textContent = tr('lv.cancelN', { n });
    txt('lvstat', () => tr('lv.clickWindow'));
    n--;
  };
  tick();
  lvTimer = setInterval(tick, 1000);
};
// stderr arrives in ARBITRARY chunk boundaries (the exe's unbuffered stderr can split a
// stats line mid-word), so buffer into complete lines before parsing - a per-chunk regex
// intermittently drops the latency tail into the next chunk.
let lvOutBuf = '';
let lvRefused = false; // a refusal reason owns the status line until the next session
// DLSS 4.5 above one pass: the session's plan, or why it stays at one pass, kept beside the stats until the next session
let lvNote: (() => string) | null = null;
function lvLine(t) {
  // The host's refusal reason ("native host: <reason>", or the engine lookup / build failure) is
  // the ONE thing the user must read: it stays the last word on the panel, because the exe's own
  // generic lines ("this session cannot start in the native host", "live server failed to
  // start") arrive after it and would otherwise replace it, and lv-out never reaches the log panel.
  const rf =
    /native host: (?!this session cannot start)([^\r\n]+)/.exec(t) ||
    /native: (the host could not find or build the engines[^\r\n(]*)/.exec(t);
  if (rf) {
    lvRefused = true;
    raw('lvstat', rf[1].trim().slice(0, 240));
    return;
  }
  if (lvRefused) return;
  const pl = /^DLSS 4\.5 at (\d+)x: (\d+)x in a second process, (?:(\d+)x in a third, )?then (\d+)x/.exec(t);
  const one = /^DLSS 4\.5 stays at (\d+)x: (?:\d+x \(\d+x, then \d+x\) )?([^\r\n]+)/.exec(t);
  const lost = /^DLSS 4\.5's second (and third )?pass(?:es)? ended \(([^)]+)\): back to one pass at (\d+)x/.exec(t);
  // every pass in helpers ("at 36.00x: 6x in a second process, 6x in a third, ..."), and a plan switched in place
  // ("steps to 12.00x in place (2x, 6x, 1x in the running ..."): the multiple and its passes, a 1x pass left out
  const lz =
    /^DLSS 4\.5 at ([\d.]+)x: (\d+)x in a second process(?:, (\d+)x in a third)?(?:, (\d+)x in a fourth)?, /.exec(t);
  const st = /^DLSS 4\.5 steps to ([\d.]+)x in place \(([\dx, ]+) in the running/.exec(t);
  const passNote = (m: string, passes: string[]) => {
    const p = passes.filter((x) => x && x !== '1');
    return 'DLSS 4.5 ' + String(+m) + '×' + (p.length > 1 ? ' (' + p.join('×, ') + '×)' : '');
  };
  if (pl) {
    const [, all, a, b, c] = pl;
    lvNote = () => 'DLSS 4.5 ' + all + '× (' + a + '×, ' + (b ? b + '×, ' : '') + tr('lv.then') + ' ' + c + '×)';
  } else if (lz) {
    const n = passNote(lz[1], [lz[2], lz[3], lz[4]]);
    lvNote = () => n;
  } else if (st) {
    const n = passNote(
      st[1],
      st[2].split(',').map((x) => x.trim().replace(/x$/, '')),
    );
    lvNote = () => n;
  } else if (one) {
    const n = one[1],
      why = one[2].trim().slice(0, 120);
    lvNote = () => tr('lv.stays', { n }) + ': ' + why;
  } else if (lost) {
    const key = lost[1] ? 'lv.lost2' : 'lv.lost1',
      n = lost[3],
      why = lost[2];
    lvNote = () => tr(key, { n, why });
  }
  // surface the exe's 2s stats lines; pass real failures through verbatim (trimmed)
  const m =
    /live: ([\d.]+) captured fps -> ([\d.]+) presented fps.*?latency ~(\d+)ms/.exec(t) ||
    /live: ([\d.]+) captured fps -> ([\d.]+) presented fps/.exec(t);
  // the measured multiplier (out over in, the host's ratio) on every model's line, beside DLSS 4.5's plan
  const rt = /presented fps \(ratio ([\d.]+)/.exec(t);
  if (m) {
    const fin = m[1],
      fout = m[2],
      ms = m[3],
      ratio = rt && +rt[1] > 0 ? ' (' + rt[1] + '×)' : '',
      note = lvNote;
    txt(
      'lvstat',
      () =>
        tr('lv.stats', { fin, fout }) +
        ratio +
        (ms ? '  ·  ' + tr('lv.latency', { ms }) : '') +
        (note ? '  ·  ' + note() : ''),
    );
  }
  // the native host is the only live route: its handoff start and cold engine build mean "loading"
  else if (/engine handoff started|host (GMFSS |Restore )?engine build for/.test(t))
    txt('lvstat', () => tr('lv.loading'));
  else if (
    /FAIL|not supported|no visible window|not a visible window|no frames captured|minimized: waiting|cannot start in the native host|live server (protocol error|stalled)/i.test(
      t,
    )
  )
    raw('lvstat', t.trim().slice(0, 160));
}
ipcRenderer.on('lv-out', (_e, t) => {
  lvOutBuf += t;
  const lines = lvOutBuf.split('\n');
  lvOutBuf = lines.pop(); // keep the trailing partial line for the next chunk
  for (const line of lines) lvLine(line);
});
// sessions can start from main (the ` hotkey) with no renderer involvement: mirror them here
ipcRenderer.on('lv-started', () => {
  if (lvTimer) {
    clearInterval(lvTimer);
    lvTimer = null;
  }
  setMode('live'); // the ` hotkey can start a session with no renderer involvement

  lvState = 'running';
  lvStopReq = false;
  lvRefused = false;
  lvNote = null;
  lvUi();
  try {
    lvSpawnTarget = lvEffTarget();
  } catch {} // what --target this session carries
  txt('lvstat', () => tr('lv.starting'));
});
ipcRenderer.on('lv-done', (_e, code) => {
  lvState = 'idle';
  lvUi();
  // a requested Stop kills the exe (code null via kill(), or nonzero on Windows), and the
  // ` hotkey stops the same way from main without setting lvStopReq: both read as a clean stop
  if (!lvRefused) {
    // a refusal reason stays on the line: an exit code adds nothing to it
    if (lvStopReq || code === 0 || code === null) txt('lvstat', () => tr('lv.stopped'));
    else if (code === 5) txt('lvstat', () => tr('lv.noWindow'));
    else if (code === 7) txt('lvstat', () => tr('lv.closed'));
    else {
      const before = lineOf('lvstat');
      txt('lvstat', () => before() + '  (' + tr('lv.ended', { code }) + ')');
    }
  }
  lvStopReq = false;
});
(async () => {
  try {
    lvReady = await ipcRenderer.invoke('lv-ready');
    if (lvReady) {
      $('lvpanel').style.display = '';
      localStorage.removeItem('lvFlowPreset'); // retired: the 540/720 presets moved to the Upscale to selector
      localStorage.removeItem('lvFscale'); // retired: the Flow scale control is gone
      localStorage.removeItem('lvGmFlow'); // retired: the GMFSS flow scale slider is gone
      const ft = localStorage.getItem('lvFit');
      if (ft && [...$('lvfit').options].some((o) => o.value === ft)) $('lvfit').value = ft;
      lvModelUi();
      lvSendOpts(); // seed the hotkey's settings at boot
    }
  } catch {}
  lvUnavailUi();
})();

// First page = the settings surface without a video: every panel can be set up beside the welcome hint, but nothing
// is probed, sized or previewed (no GPU work) and Smooth It! waits until a picked or dropped video runs loadVideo.
if (!input) {
  showWorkspace(true);
  $('go').disabled = true;
  $('changeout').disabled = true;
  $('outrow').style.display = 'none';
  setStatus('run.status.select');
}

// --- Dolby Vision export tools (mirrors the RTX library flow) ------------------------------------
// DV 8.1 export needs ONE non-shippable open-source binary - dovi_tool - dropped into engine/dvtools;
// the RPU it injects is muxed by the bundled ffmpeg and the DV signaling box is written by the engine
// itself (hdr10_meta.inject_dv_config), so no GPAC/MP4Box is involved. The setup box appears when the
// toggle is on and dovi_tool is missing, and collapses to nothing once it is installed (dv-ready).
let dvReady: any = { dovi: false, ready: false };
// Two Dolby Vision boxes that exclude each other: Profile 8.1 on the HDR10 video, 8.4 with the video written as HLG
// (HDR10+ rides on HDR10 only, so 8.4 and HDR10+ exclude each other too)
function dvOn() {
  return $('dvexport').checked || $('dv84export').checked;
}
function syncDv() {
  if (dvOn()) checkDvReady();
  else $('dvsetup').style.display = 'none';
  updateDvHpHints();
}
// Both dynamic-HDR exports ride on the HDR10 render AND need an MP4 container. When the loaded
// video's tracks force MKV output (subtitles, exotic audio), the engine would skip them with only
// a log notice at render time - so say it up front, next to the checkbox, the moment the output
// name resolves to .mkv. Pure text updates (no IPC), safe to call from refresh() on every change.
function updateDvHpHints() {
  const mkv = (lastOut || '').toLowerCase().endsWith('.mkv');
  const hint = (name, base = 'HDR10') =>
    !$('rtxhdr').checked
      ? tr('common.needsRtxHdr')
      : mkv
        ? tr('dvhp.mkv', { name })
        : tr('dvhp.fallback', { base, name });
  $('dvhint').textContent = hint('Dolby Vision');
  $('dv84hint').textContent = hint('Dolby Vision', 'HLG');
  $('hphint').textContent = hint('HDR10+');
}
async function checkDvReady() {
  $('dvstatus').textContent = tr('common.checking');
  $('dvstatus').style.color = 'var(--sub)';
  try {
    dvReady = await ipcRenderer.invoke('dv-ready');
  } catch {
    dvReady = { dovi: false, ready: false };
  }
  const line = (dvReady.dovi ? '✓ ' : '✗ ') + tr('dv.line');
  if (dvReady.ready) {
    $('dvstatus').textContent = tr('setup.ready');
    $('dvstatus').style.color = 'var(--green)';
    $('dvsteps').innerHTML = line + '<br>' + tr('dv.installed');
  } else {
    $('dvstatus').textContent = tr('dlss.notReady');
    $('dvstatus').style.color = '#e85c5c';
    $('dvsteps').innerHTML =
      line +
      '<br>' +
      tr('dv.howto', { ver: dvReady.ver || '', mb: dvReady.mb || '1.3', link: toolPageLink('dv-open-download') });
  }
  $('dvbtns').style.display = dvReady.ready ? 'none' : ''; // hide the Download button once dovi_tool is installed
  $('dvsetup').style.display = dvReady.ready ? 'none' : 'block';
}
// a tool's pinned release page as a link (main opens it): its text is the language's, its action the code's
function toolPageLink(channel) {
  return '<a href="#" onclick="ipcRenderer.invoke(\'' + channel + '\');return false;">' + tr('aj.repoLink') + '</a>';
}
// One-click download of a pinned tool release (main does fetch + checksum + extract + copy): progress lands in the
// status line, a failure keeps the note open with the retry line; `ready` re-reads the row afterwards.
function toolDownloader(id, channel, ready) {
  let busy = false;
  ipcRenderer.on(channel.replace('download', 'progress'), (_e, p) => {
    if (!busy) return;
    const mb = (x) => (x / 1048576).toFixed(1);
    $(id + 'status').textContent =
      p.stage === 'verify' ? tr('dl.verify') : tr('dl.progress', { got: mb(p.received), total: mb(p.total) });
  });
  return async () => {
    if (busy) return;
    busy = true;
    $(id + 'dl').disabled = true;
    $(id + 'setup').style.display = 'block';
    $(id + 'status').textContent = tr('dl.start');
    $(id + 'status').style.color = 'var(--sub)';
    let r: any = {};
    try {
      r = await ipcRenderer.invoke(channel);
    } catch (e) {
      r = { ok: false, error: String(e) };
    }
    busy = false;
    $(id + 'dl').disabled = false;
    await ready();
    if (!r.ok) {
      $(id + 'status').textContent = tr('dl.failed');
      $(id + 'status').style.color = '#e85c5c';
      $(id + 'steps').innerHTML = '✗ ' + (r.error || tr('dl.failedShort')) + '<br>' + tr('dl.retry');
      $(id + 'setup').style.display = 'block';
    }
  };
}
$('dvdl').onclick = toolDownloader('dv', 'dv-download', () => checkDvReady());
$('xessdl').onclick = toolDownloader('xess', 'xess-download', () => checkXessReady());
// Intel XeSS-FG's row downloads the same pinned SDK release; once Intel's files are in, a render or Live uses it
$('xefgdl').onclick = toolDownloader('xefg', 'xess-download', async () => {
  await refreshXefgState();
  syncModel();
  syncCompareUi();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
  if (xefgReady) {
    $('xefgstatus').textContent = tr('setup.ready');
    $('xefgstatus').style.color = 'var(--green)';
  }
});
syncXess();
i18nRelabel.push(syncXess);
if (localStorage.getItem('dvOn') === '1') $('dvexport').checked = true; // default OFF
if (localStorage.getItem('dv84On') === '1' && !$('dvexport').checked) $('dv84export').checked = true; // default OFF
$('dvexport').onchange = () => {
  localStorage.setItem('dvOn', $('dvexport').checked ? '1' : '0');
  if ($('dvexport').checked && $('dv84export').checked) {
    $('dv84export').checked = false;
    localStorage.setItem('dv84On', '0');
  }
  syncDv();
};
$('dv84export').onchange = () => {
  localStorage.setItem('dv84On', $('dv84export').checked ? '1' : '0');
  if ($('dv84export').checked) {
    if ($('dvexport').checked) {
      $('dvexport').checked = false;
      localStorage.setItem('dvOn', '0');
    }
    if ($('hpexport').checked) {
      $('hpexport').checked = false;
      localStorage.setItem('hpOn', '0');
      syncHp();
    }
  }
  syncDv();
};
$('rtxhdr').addEventListener('change', () => {
  if (dvOn()) syncDv();
}); // hint tracks HDR state
syncDv();
i18nRelabel.push(syncDv);

// --- HDR10+ export tool (mirrors the Dolby Vision flow) ------------------------------------------
// HDR10+ export needs ONE non-shippable open-source binary - hdr10plus_tool - dropped into
// engine/hptools; the engine collects the per-frame brightness stats itself and the tool injects
// the ST 2094-40 SEI (see _hp_export in the engine). The setup box appears when the toggle is on
// and the tool is missing, and collapses once it is installed (hp-ready).
let hpReady: any = { ready: false };
function syncHp() {
  if ($('hpexport').checked) checkHpReady();
  else $('hpsetup').style.display = 'none';
  updateDvHpHints();
}
async function checkHpReady() {
  $('hpstatus').textContent = tr('common.checking');
  $('hpstatus').style.color = 'var(--sub)';
  try {
    hpReady = await ipcRenderer.invoke('hp-ready');
  } catch {
    hpReady = { ready: false };
  }
  const line = (hpReady.ready ? '✓ ' : '✗ ') + tr('hp.line');
  if (hpReady.ready) {
    $('hpstatus').textContent = tr('setup.ready');
    $('hpstatus').style.color = 'var(--green)';
    $('hpsteps').innerHTML = line + '<br>' + tr('hp.installed');
  } else {
    $('hpstatus').textContent = tr('dlss.notReady');
    $('hpstatus').style.color = '#e85c5c';
    $('hpsteps').innerHTML =
      line +
      '<br>' +
      tr('hp.howto', { ver: hpReady.ver || '', mb: hpReady.mb || '0.9', link: toolPageLink('hp-open-download') });
  }
  $('hpbtns').style.display = hpReady.ready ? 'none' : ''; // hide the Download button once the tool is installed
  $('hpsetup').style.display = hpReady.ready ? 'none' : 'block';
}
$('hpdl').onclick = toolDownloader('hp', 'hp-download', () => checkHpReady());
if (localStorage.getItem('hpOn') === '1' && !$('dv84export').checked) $('hpexport').checked = true; // default OFF
$('hpexport').onchange = () => {
  localStorage.setItem('hpOn', $('hpexport').checked ? '1' : '0');
  if ($('hpexport').checked && $('dv84export').checked) {
    $('dv84export').checked = false;
    localStorage.setItem('dv84On', '0');
    syncDv();
  }
  syncHp();
};
$('rtxhdr').addEventListener('change', () => {
  if ($('hpexport').checked) syncHp();
}); // hint tracks HDR state
syncHp();
i18nRelabel.push(syncHp);

// Before/after preview (panel above Smooth It): render ONE frame at the current spatial settings via the
// main process's preview (render/preview.js, the native host's pass chain) and show original vs processed. It applies RTX HDR (when on + ready) and
// Adaptive Sharpen (when on), so the user can see how those settings change the picture before a full render.
// Back / random-frame step through random positions across the clip. HDR is tonemapped to sRGB because
// this canvas cannot show PQ. Images load as object URLs (nodeIntegration) to dodge file:// webSecurity,
// and old URLs are revoked so memory does not grow.
const _fs = require('fs');
let prevTotal = 0,
  prevIdx = 0,
  prevHist = [];
function sharpStrength() {
  return $('sharpen').checked ? parseFloat($('sharpval').value) || 0 : 0;
}
function hdrOn() {
  return $('rtxhdr').checked && rtxReady.hdr && !(info && info.srcHdr);
}
function setPrevImg(el, file, mime?) {
  const old = el.dataset.url;
  if (old) {
    try {
      URL.revokeObjectURL(old);
    } catch {}
  }
  const url = URL.createObjectURL(new Blob([_fs.readFileSync(file)], { type: mime || 'image/png' }));
  el.dataset.url = url;
  el.src = url;
  el.style.visibility = ''; // imgs start hidden so the broken-image icon never shows pre-load
}

// Live progress thumbnail: during a render the engine drops a small PNG of the frame it just
// produced (~1/s, SMV_LIVE_PREVIEW); poll it by mtime and show it under the progress bar.
let livePath = null,
  liveTimer = null,
  liveM = 0;
ipcRenderer.invoke('live-path').then((p) => {
  livePath = p;
});
// Hide/Show toggle: the preview costs a little GPU/CPU per second, so hiding it tells the engine
// (via main's live-off flag) to stop producing it, not just hides the image. Persisted.
let liveHidden = localStorage.getItem('liveHidden') === '1';
function applyLiveOff() {
  ipcRenderer.send('live-off', liveHidden);
} // sync engine to the saved choice
function syncLiveToggle() {
  $('livetoggle').textContent = liveHidden ? tr('run.liveShow') : tr('run.livetoggle');
  if (liveHidden) $('live').style.display = 'none'; // Show reveals again on the next written frame
}
$('livetoggle').onclick = () => {
  liveHidden = !liveHidden;
  localStorage.setItem('liveHidden', liveHidden ? '1' : '0');
  syncLiveToggle();
  applyLiveOff();
};
syncLiveToggle();
i18nRelabel.push(syncLiveToggle);
function updateLive() {
  if (!livePath || liveHidden) return; // hidden: engine produces nothing, keep it hidden
  try {
    const m = _fs.statSync(livePath).mtimeMs;
    if (m === liveM) return; // nothing new yet
    liveM = m;
    setPrevImg($('live'), livePath, 'image/png');
    $('live').style.display = 'block';
  } catch {} // not written yet this run
}
function updatePrevNav() {
  // back only makes sense once a second frame has been generated
  $('prevprev').style.visibility = prevHist.length > 1 ? '' : 'hidden';
}
function syncPreview() {
  // the preview panel appears once a video is loaded
  $('previewpanel').style.display = input ? '' : 'none';
  syncHdrColor(); // HDR colour controls track the HDR toggle / readiness / source
}
let prevBusy = false,
  prevPending = null,
  previewLite = false; // serialize renders. previewLite: skip heavy RTX passes on the auto-preview (set by openPreviewForVideo, one-shot)
let lastPrevKey = '',
  inflightPrevKey = ''; // settings signatures of the shown / mid-render preview
function prevSettingsSig() {
  // everything that changes what the processed pane shows
  const p = hdrColorPayload(); // normalized: the vibrance feature at zero strength equals OFF
  return [
    sharpStrength(),
    restoreOn(),
    restoreModelId(),
    hdrOn(),
    p.color,
    p.saturation,
    effVibrance(),
    effSatBoost(),
    sdkCon(),
    upFactor() || 0,
    !!($('rtxvsr').checked && rtxReady.vsr),
    vsrQuality(),
    fsrUpOn(),
    a4kUpOn(),
    a4kAaOn(),
    acnUpOn(),
    xessUpOn(),
    nrOn(),
    nrStructure(),
    nrTone(),
    nrStyle(),
    nrPasses(),
    nrMaskOn(),
    dlssMode(),
  ];
}
let lastPrevInput = null; // which video the shown original belongs to
async function loadPreview(frame, bg?) {
  // bg: background "refine" pass (the RTX auto-upgrade) - keep the shown image up, no blocking spinner
  if (!input) return;
  if (cmpOn()) return loadCompare(frame == null ? 'mid' : frame); // Compare on: the 2x2 takes the pane
  if (prevBusy) {
    prevPending = frame == null ? 'mid' : frame;
    return;
  }
  prevBusy = true;
  const hdr = hdrOn(),
    sharpen = sharpStrength(),
    sig = prevSettingsSig();
  // The auto-preview on video load runs "lite": it skips the heavy RTX VSR + TrueHDR passes (the bulk of
  // the ~5s cost), so the drop is fast. Adjusting any preview control renders the full RTX version.
  const lite = previewLite;
  previewLite = false;
  const useHdr = lite ? false : hdr;
  const useVsr = lite ? false : !!($('rtxvsr').checked && rtxReady.vsr && upFactor() > 1); // VSR enlarges only
  const useFsr = lite ? false : fsrUpOn() && upFactor() > 1; // AMD FSR 4 the same way
  const useA4k = lite ? false : a4kUpOn() && upFactor() > 1; // Anime4K too
  const useAcn = lite ? false : acnUpOn() && upFactor() > 1; // ArtCNN too
  const useXess = lite ? false : xessUpOn() && upFactor() > 1; // Intel XeSS too
  const useNr = lite ? false : nrOn(); // DLSS 5: its host takes seconds to start, so the lite pass skips it too
  inflightPrevKey = JSON.stringify([input, frame == null ? 'mid' : frame].concat(sig));
  $('hdrpreview').style.display = 'block';
  // On a background refine the lite image is already on screen: leave it up and skip the blocking spinner,
  // so the pane appears instantly and just sharpens to the RTX result when it lands. Otherwise show it.
  if (!bg) {
    if ($('prevproc').clientHeight < 40) $('prevprocwrap').style.minHeight = '120px'; // spinner space on first load
    setSpin($('prevprocwrap'), $('prevspin'), true);
  }
  // The original pane gets the wheel too, but only when the original is actually about to change:
  // first load, a different video, or a frame step. Settings tweaks (and background refines) keep the frame.
  const origChanging =
    !bg &&
    (!$('prevorig').dataset.url || lastPrevInput !== input || (frame != null && frame !== 'mid' && frame !== prevIdx));
  if (origChanging) {
    if ($('prevorig').clientHeight < 40) $('prevorigwrap').style.minHeight = '120px';
    setSpin($('prevorigwrap'), $('prevorigspin'), true);
  }
  let r;
  try {
    r = await ipcRenderer.invoke(
      'preview',
      Object.assign(
        {
          input,
          frame: frame == null ? 'mid' : frame,
          sharpen,
          restore: restoreOn(),
          restoreModel: restoreModelId(),
          hdr: useHdr,
          vibrance: effVibrance(),
          satboost: effSatBoost(),
          contrast: sdkCon(),
          upscale: upFactor() || 0,
          rtxvsr: useVsr,
          vsrquality: vsrQuality(),
          fsrup: useFsr,
          anime4k: useA4k,
          a4kaa: useA4k && a4kAaOn(),
          artcnn: useAcn,
          xess: useXess,
          dlssnr: useNr,
          nrmask: useNr && nrMaskOn(),
          nrstructure: nrStructure(),
          nrtone: nrTone(),
          nrstyle: nrStyle(),
          nrpasses: nrPasses(),
        },
        hdrColorPayload(),
      ),
    );
  } catch (e) {
    r = { error: String(e) };
  }
  setSpin($('prevprocwrap'), $('prevspin'), false);
  $('prevprocwrap').style.minHeight = '';
  setSpin($('prevorigwrap'), $('prevorigspin'), false);
  $('prevorigwrap').style.minHeight = '';
  prevBusy = false;
  if (prevPending != null) {
    const f = prevPending;
    prevPending = null;
    loadPreview(f);
  }
  if (!r || r.error) {
    lastPrevKey = ''; // never dedupe-skip a retry after a failure
    const err = r && r.error;
    txt('prevnote', () => tr('prev.unavail', { err: err || tr('prev.noResult') }));
    return;
  }
  if (r.total) prevTotal = r.total;
  if (typeof r.frame === 'number') prevIdx = r.frame;
  setPrevImg($('prevorig'), r.original);
  const maskShown = !!r.nrmask; // the DLSS 5 change map replaces the processed picture when asked for
  setPrevImg($('prevproc'), maskShown ? r.nrmask : r.processed);
  lastPrevInput = input;
  lastPrevKey = lite ? '' : JSON.stringify([input, prevIdx].concat(sig)); // lite skipped RTX: force a full re-render on the next adjustment
  const upActive = upFactor() > 0; // the processed side is upscaled too, so never call it unchanged
  const vsrOn = useVsr; // what the processed pane actually rendered (lite skips VSR)
  const active = useHdr || sharpen > 0 || restoreOn() || upActive || useNr;
  // Plain Lanczos3 resize with no AI/enhancement pass: both panes are the same resized image, so say so.
  const plainOnly =
    upActive &&
    !vsrOn &&
    !useFsr &&
    !useA4k &&
    !useAcn &&
    !useXess &&
    !restoreOn() &&
    sharpen <= 0 &&
    !useHdr &&
    !useNr;
  const rtxSkipped =
    lite &&
    (hdr ||
      nrOn() ||
      ($('rtxvsr').checked && rtxReady.vsr && upFactor() > 1) ||
      ((fsrUpOn() || a4kUpOn() || acnUpOn() || xessUpOn()) && upFactor() > 1)); // RTX / FSR / Anime4K / ArtCNN / DLSS 5 on in settings but skipped for the fast auto-preview
  const srcHdr = !!(info && info.srcHdr);
  // the pass names ('@key' = a language string, else a name that stays as is)
  const parts = [];
  if (restoreOn()) parts.push('@restore.k');
  if (upActive)
    parts.push(
      vsrOn
        ? 'VSR'
        : useFsr
          ? 'FSR'
          : useA4k
            ? 'Anime4K'
            : useAcn
              ? 'ArtCNN'
              : useXess
                ? 'XeSS'
                : upFactor() < 1
                  ? '@prev.part.down'
                  : '@prev.part.up',
    );
  if (useNr) parts.push('DLSS 5');
  if (sharpen > 0) parts.push('@sharpen.k');
  if (useHdr) parts.push('HDR');
  const note = maskShown
    ? 'prev.note.mask'
    : rtxSkipped
      ? 'prev.note.quick'
      : srcHdr
        ? 'prev.note.srcHdr'
        : !active
          ? 'prev.note.none'
          : plainOnly
            ? upFactor() < 1
              ? 'prev.note.plainDown'
              : 'prev.note.plainUp'
            : useHdr
              ? 'prev.note.hdr'
              : 'prev.note.subtle';
  prevLab = {
    srcHdr,
    active,
    maskShown,
    parts,
    idx: prevIdx,
    total: prevTotal,
    note,
    pct: r.nrmaskPct,
    mean: r.nrmaskMean,
  };
  paintPrevLabels();
  // Auto-upgrade: the quick preview skipped the slow RTX passes for an instant first paint. Now that it's
  // on screen, render the full RTX (VSR/HDR) version of the SAME frame so the pane ends on the real result
  // without the user needing to touch a control - their RTX picks persist across sessions, so many never
  // would. One-shot (the follow-up render isn't lite, so rtxSkipped is false there and it can't loop);
  // skipped if the user already queued another render, which supersedes it.
  if (rtxSkipped && prevPending == null && !prevBusy) loadPreview(prevIdx, true);
}
// the shown preview's labels from its last result (prevLab), so a language switch relabels it with no new render
let prevLab: any = null;
function paintPrevLabels() {
  if (!prevLab) return;
  const L = prevLab;
  $('prevoriglabel').textContent = L.srcHdr ? tr('prev.origHdr') : tr('prev.orig');
  const parts = L.parts.map((p) => (p.startsWith('@') ? tr(p.slice(1)) : p)).join(' + ');
  $('prevproclabel').textContent = !L.active
    ? tr('prev.unchanged')
    : L.maskShown
      ? tr('prev.maskLabel')
      : tr('prev.procParts', { parts });
  $('prevlabel').textContent = tr('prev.frameN', { n: L.idx }) + (L.total ? ' / ' + L.total : '');
  txt('prevnote', () => tr(L.note, { pct: L.pct, mean: L.mean }) + ' · ' + tr('prev.note.click'));
}
i18nRelabel.push(paintPrevLabels);
function randFrame() {
  return prevTotal > 1 ? Math.floor(Math.random() * prevTotal) : 'mid';
}
function refreshPreviewIfOpen() {
  // re-render the shown frame when sharpen / HDR changes
  if (!input) return;
  syncPreview();
  if (cmpOn()) {
    refreshCompare();
    return;
  }
  const frame = prevHist.length ? prevHist[prevHist.length - 1] : 'mid';
  const key = JSON.stringify([input, frame].concat(prevSettingsSig()));
  // Skip when the effective settings already match what is shown (or what is mid-render), so the
  // checkbox handler and checkRtxReady's async refresh don't double-render the same state.
  if (key === (prevBusy ? inflightPrevKey : lastPrevKey)) return;
  loadPreview(frame);
}
// Show the preview panel with a spinning wheel on both panes the INSTANT a video is selected, before
// the slow preview render is even spawned, so the panel reads as "loading" rather than blank/crashed.
// A neutral note replaces the usual result caption until the first frame lands.
function showPreviewInitializing() {
  syncPreview(); // reveal the preview panel
  $('hdrpreview').style.display = 'block';
  for (const [inner, wrap, spin] of [
    ['prevorig', 'prevorigwrap', 'prevorigspin'],
    ['prevproc', 'prevprocwrap', 'prevspin'],
  ]) {
    if ($(inner).clientHeight < 40) $(wrap).style.minHeight = '120px'; // room for the wheel on first load
    setSpin($(wrap), $(spin), true);
  }
  txt('prevnote', () => tr('prev.init'));
}
function openPreviewForVideo() {
  // auto-load the pane when a video is selected
  syncPreview();
  prevHist = [];
  prevTotal = 0;
  prevIdx = 0;
  updatePrevNav();
  previewLite = true; // the auto-preview on load skips the heavy RTX VSR/HDR passes so the drop is fast
  loadPreview('mid').then(() => {
    if (prevTotal && !prevHist.length) prevHist = [prevIdx];
    updatePrevNav();
  });
}
$('prevnext').onclick = async () => {
  await loadPreview(randFrame());
  prevHist.push(prevIdx);
  updatePrevNav();
};
$('prevprev').onclick = async () => {
  if (prevHist.length > 1) {
    prevHist.pop();
    updatePrevNav();
    await loadPreview(prevHist[prevHist.length - 1]);
  }
};
$('sharpval').addEventListener('change', refreshPreviewIfOpen);
$('sharpen').addEventListener('change', refreshPreviewIfOpen);
$('restore').addEventListener('change', () => {
  refreshPreviewIfOpen();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
});
// The interp-off output name tags the active pass (_restored/_dlss5/_sharpened/_hdr), so it must
// re-resolve when any of them toggles (refresh() is a no-op until a video is selected).
for (const _id of ['sharpen', 'sharpval', 'restore', 'restoreaj', 'restorea4k', 'rtxhdr', 'dlssnr'])
  $(_id).addEventListener('change', () => refresh());

// RTX HDR colour: TrueHDR's result as NVIDIA gives it (the engine's raw mode; the same colours as
// NVIDIA's own RTX Video HDR in a player), its Contrast and Saturation sliders = the SDK's, as the
// NVIDIA App's RTX HDR filter. The engine's vivid / rtx modes stay CLI-only. nits = the file
// renders' Peak box (0 = empty: main.ts takes the peak of the display the window is on).
function hdrColorPayload() {
  return { color: 'raw', saturation: sdkSat(), nits: hdrNitsAsked() };
}
function hdrNitsAsked() {
  const v = Math.round(+$('hdrnits').value || 0);
  return v > 0 ? Math.min(2000, Math.max(400, v)) : 0;
}
// The Dynamic Vibrance sliders belong to that feature: both are 0 when it is switched off, and
// the feature at 0/0 is identity (NVIDIA convention, verified in the App).
function effVibrance() {
  return $('hdrdynvib').checked ? (+$('hdrvib').value || 0) / 100 : 0;
}
function effSatBoost() {
  return $('hdrdynvib').checked ? (+$('hdrsb').value || 0) / 100 : 0;
}
function syncHdrColor() {
  const showHdr = $('rtxhdr').checked && rtxReady.hdr && !(info && info.srcHdr);
  $('hdrtonewrap').style.display = showHdr ? 'inline-flex' : 'none'; // RTX HDR's own Contrast + Saturation
  $('hdrpurewrap').style.display = showHdr ? '' : 'none'; // the pure-conversion recipe under the sliders
  $('hdrcolorwrap').style.display = showHdr ? '' : 'none';
  // RTX HDR's sliders are always live; the Dynamic Vibrance sliders belong to that feature and sit
  // greyed out until it is switched on (unchecked = no vibrance, like the App).
  $('hdrcon').disabled = false;
  $('hdrsat').disabled = false;
  const on = $('hdrdynvib').checked;
  $('hdrsb').disabled = !on;
  $('hdrvib').disabled = !on;
  $('hdrconnum').textContent = $('hdrcon').value;
  $('hdrsatnum').textContent = $('hdrsat').value;
  $('hdrsbnum').textContent = $('hdrsb').value;
  $('hdrvibnum').textContent = $('hdrvib').value;
}
// Display scales mirror the NVIDIA App's RTX HDR filter (-100..100, 0 = NVIDIA default); the
// engine keeps the TrueHDR SDK scales (0..200, 100 neutral).
function sdkCon() {
  return 100 + (+$('hdrcon').value || 0);
}
function sdkSat() {
  return 100 + (+$('hdrsat').value || 0);
}
// Dynamic Vibrance ALWAYS starts off: it is a per-session opt-in
// effect, so its on/off state is deliberately NOT persisted (the two slider values below
// still are). The old 'dynVib' key is retired so no stale state lingers.
localStorage.removeItem('dynVib');
const savedHdrSat = localStorage.getItem('hdrSatApp'); // App scale -100..100 (older keys orphaned)
if (savedHdrSat !== null) $('hdrsat').value = savedHdrSat;
const savedHdrNits = localStorage.getItem('hdrNits'); // the Peak box, empty = the display's
if (savedHdrNits) $('hdrnits').value = savedHdrNits;
$('hdrnits').onchange = () => {
  localStorage.setItem('hdrNits', $('hdrnits').value);
  refreshPreviewIfOpen();
};
// the empty box shows what it stands for: the peak of the display the window is on
let displayPeak = null;
function paintDisplayPeak() {
  if (displayPeak != null) $('hdrnits').placeholder = tr('hdr.peak.auto', { n: displayPeak });
}
function showDisplayPeak() {
  ipcRenderer
    .invoke('peak-nits')
    .then((n) => {
      displayPeak = n;
      paintDisplayPeak();
    })
    .catch(() => {});
}
showDisplayPeak();
i18nRelabel.push(paintDisplayPeak);
$('rtxhdr').addEventListener('change', showDisplayPeak);
const savedHdrCon = localStorage.getItem('hdrCon');
if (savedHdrCon !== null) $('hdrcon').value = savedHdrCon;
const savedHdrSB = localStorage.getItem('hdrSB');
if (savedHdrSB !== null) $('hdrsb').value = savedHdrSB;
const savedHdrVib = localStorage.getItem('hdrVib');
if (savedHdrVib !== null) $('hdrvib').value = savedHdrVib;
$('hdrdynvib').onchange = () => {
  syncHdrColor();
  refreshPreviewIfOpen();
  try {
    lvSendOpts();
  } catch {}
};
$('hdrsat').oninput = () => {
  localStorage.setItem('hdrSatApp', $('hdrsat').value);
  $('hdrsatnum').textContent = $('hdrsat').value;
  try {
    lvSendOpts();
  } catch {}
};
$('hdrsat').addEventListener('change', refreshPreviewIfOpen);
$('hdrcon').oninput = () => {
  localStorage.setItem('hdrCon', $('hdrcon').value);
  $('hdrconnum').textContent = $('hdrcon').value;
  try {
    lvSendOpts();
  } catch {}
};
$('hdrcon').addEventListener('change', refreshPreviewIfOpen);
$('hdrsb').oninput = () => {
  localStorage.setItem('hdrSB', $('hdrsb').value);
  $('hdrsbnum').textContent = $('hdrsb').value;
  try {
    lvSendOpts();
  } catch {}
};
$('hdrsb').addEventListener('change', refreshPreviewIfOpen);
$('hdrvib').oninput = () => {
  localStorage.setItem('hdrVib', $('hdrvib').value);
  $('hdrvibnum').textContent = $('hdrvib').value;
  try {
    lvSendOpts();
  } catch {}
};
$('hdrvib').addEventListener('change', refreshPreviewIfOpen);
syncHdrColor();

// 1:1 zoom: the pane shrinks 1080p+ frames ~6x, which visually erases sharpening, so clicking either
// image toggles both to native pixels inside scrollable boxes with mirrored scrolling for A/B compare.
let prevZoom = false,
  prevSyncing = false;
function applyPrevZoom() {
  for (const [w, i] of [
    ['prevorigwrap', 'prevorig'],
    ['prevprocwrap', 'prevproc'],
  ]) {
    const wrap = $(w),
      img = $(i);
    wrap.style.overflow = prevZoom ? 'auto' : 'hidden';
    wrap.style.maxHeight = prevZoom ? '320px' : '';
    img.style.width = prevZoom ? 'auto' : '100%';
    img.style.maxWidth = prevZoom ? 'none' : '';
    img.style.cursor = prevZoom ? 'zoom-out' : 'zoom-in';
  }
}
function prevScrollSync(src, dst) {
  if (prevSyncing) return;
  prevSyncing = true;
  // Proportional, not pixel-absolute: the two panes can hold different native sizes (an
  // upscale or Restore preview's processed side is a larger image than the source), so only
  // mirroring the scroll FRACTION keeps both panes on the same region of the scene. The <1px
  // skip stops the async echo event from ping-ponging rounding jitter between the panes.
  const fx = src.scrollLeft / Math.max(1, src.scrollWidth - src.clientWidth);
  const fy = src.scrollTop / Math.max(1, src.scrollHeight - src.clientHeight);
  const nl = fx * Math.max(0, dst.scrollWidth - dst.clientWidth);
  const nt = fy * Math.max(0, dst.scrollHeight - dst.clientHeight);
  if (Math.abs(dst.scrollLeft - nl) >= 1) dst.scrollLeft = nl;
  if (Math.abs(dst.scrollTop - nt) >= 1) dst.scrollTop = nt;
  prevSyncing = false;
}
function setSpin(wrap, spin, on) {
  // Sized/positioned to the VISIBLE window of the scroll box, so the wheel stays centred in view
  // when the pane is zoomed 1:1 and scrolled (an inset-0 overlay would sit at the scroll origin).
  if (!on) {
    spin.style.display = 'none';
    return;
  }
  spin.style.left = wrap.scrollLeft + 'px';
  spin.style.top = wrap.scrollTop + 'px';
  spin.style.width = wrap.clientWidth + 'px';
  spin.style.height = wrap.clientHeight + 'px';
  spin.style.display = 'flex';
}
function repositionSpins() {
  // keep a visible wheel glued to the viewport while scrolling
  for (const [w, s] of [
    ['prevorigwrap', 'prevorigspin'],
    ['prevprocwrap', 'prevspin'],
  ]) {
    const spin = $(s);
    if (spin.style.display !== 'none') setSpin($(w), spin, true);
  }
}
$('prevorigwrap').addEventListener('scroll', () => {
  prevScrollSync($('prevorigwrap'), $('prevprocwrap'));
  repositionSpins();
});
$('prevprocwrap').addEventListener('scroll', () => {
  prevScrollSync($('prevprocwrap'), $('prevorigwrap'));
  repositionSpins();
});
function togglePrevZoom(e) {
  // Capture the clicked spot as a fraction of the displayed image BEFORE the zoom restyle
  // changes the layout, then centre the 1:1 view on that spot (clicking the dragon's head
  // zooms to the dragon's head, not the frame middle). The other pane follows via the
  // proportional scroll sync, called directly so it lands this frame, not on the async event.
  const img = e.currentTarget;
  const fx = e.offsetX / Math.max(1, img.clientWidth);
  const fy = e.offsetY / Math.max(1, img.clientHeight);
  prevZoom = !prevZoom;
  applyPrevZoom();
  if (prevZoom) {
    const w = img.parentElement;
    w.scrollLeft = fx * img.clientWidth - w.clientWidth / 2;
    w.scrollTop = fy * img.clientHeight - w.clientHeight / 2;
    prevScrollSync(w, w === $('prevorigwrap') ? $('prevprocwrap') : $('prevorigwrap'));
  }
  repositionSpins();
}
$('prevorig').onclick = togglePrevZoom;
$('prevproc').onclick = togglePrevZoom;

// Compare: the Video tab's model (A) beside the one "Compare to" picks (B) on the same pair of real frames. main's
// 'compare' (render/preview.js compare) runs the pair (k, k + 1) through both models at the preview's settings and the
// render's multiplier: left the real frame k over the real frame k + 1, right A's new frame over B's at the same
// moment. ‹ › step through the pair's new frames and on to the next pairs, a click zooms all four together (1:1, 2x,
// 4x, back to fit), holding Space swaps A and B. Not persisted: a new video or a restart starts at Off, so loading a
// video never runs two models.
const CMP_NAMES = {
  gmfss: 'GMFSS',
  nvof: 'NVIDIA Optical Flow',
  rife: 'RIFE',
  rifedrba: 'RIFE',
  dlssg: 'NVIDIA DLSS 4.5',
  fsrfg: 'AMD FSR 4',
  xefg: 'Intel XeSS-FG',
  fruc: 'NVIDIA Smooth Motion',
};
const CMP_ZOOMS = [0, 1, 2, 4]; // 0 = fit, else pixels per image pixel
let cmpRes = null,
  cmpJ = 1,
  cmpWantJ = 0,
  cmpSwap = false,
  cmpZoom = 0,
  cmpSyncing = false;
let cmpKey = '',
  cmpInflight = '',
  cmpTimer = null,
  cmpShown = false;
function cmpA() {
  return interpOn() ? renderModel() || 'gmfss' : '';
}
function cmpOn() {
  return !!$('cmpto').value && !!cmpA();
}
function cmpMulti() {
  return Math.min(100, gridMulti() || Math.max(2, Math.round(outRatio())));
} // a resampled target shows its closest whole multiple
function cmpSig(frame) {
  return JSON.stringify([input, frame, cmpA(), $('cmpto').value, cmpMulti()].concat(prevSettingsSig()));
}
function syncCompareUi() {
  const sel = $('cmpto'),
    a = cmpA(),
    fam = (m) => (m === 'rifedrba' ? 'rife' : m);
  for (const o of sel.options)
    if (o.value)
      o.disabled =
        o.value === fam(a) ||
        (o.value === 'dlssg' && !dlssReady) ||
        (o.value === 'fruc' && !frucReady) ||
        (o.value === 'xefg' && !xefgReady);
  // nothing ticked = nothing to compare; A moved onto B (or B's runtime is missing) = GMFSS, or RIFE when A is GMFSS
  if (!a) sel.value = '';
  else if (sel.value && sel.selectedOptions[0].disabled) sel.value = fam(a) === 'gmfss' ? 'rife' : 'gmfss';
  sel.disabled = !a || goMode !== 'idle';
  $('cmpwrap').title = a ? tr('cmp.tip') : tr('cmp.needModel');
  const on = cmpOn();
  // back to the pair: it (and its note) re-renders on the next refresh, the next compare starts at its frame
  if (cmpShown && !on) {
    lastPrevKey = '';
    cmpRes = null;
  }
  cmpShown = on;
  $('prevpair').style.display = on ? 'none' : 'flex';
  $('prevnav').style.display = on ? 'none' : '';
  $('cmpgrid').style.display = on ? '' : 'none';
  $('cmpnav').style.display = on ? '' : 'none';
}
function cmpSpin(on) {
  for (let i = 0; i < 4; i++) {
    if (on && $('cmpimg' + i).clientHeight < 40) $('cmpw' + i).style.minHeight = '120px'; // spinner space on first load
    if (!on) $('cmpw' + i).style.minHeight = '';
    setSpin($('cmpw' + i), $('cmpspin' + i), on);
  }
}
function cmpOpts(frame, a, b, multi, pairs) {
  // main's 'compare' request at the preview's settings
  return Object.assign(
    {
      input,
      frame,
      modelA: a,
      modelB: b,
      multi,
      pairs,
      sharpen: sharpStrength(),
      restore: restoreOn(),
      restoreModel: restoreModelId(),
      hdr: hdrOn(),
      vibrance: effVibrance(),
      satboost: effSatBoost(),
      contrast: sdkCon(),
      upscale: upFactor() || 0,
      rtxvsr: !!($('rtxvsr').checked && rtxReady.vsr && upFactor() > 1),
      vsrquality: vsrQuality(),
      fsrup: fsrUpOn() && upFactor() > 1,
      anime4k: a4kUpOn() && upFactor() > 1,
      a4kaa: a4kAaOn() && upFactor() > 1,
      artcnn: acnUpOn() && upFactor() > 1,
      xess: xessUpOn() && upFactor() > 1,
      dlssnr: nrOn(),
      nrstructure: nrStructure(),
      nrtone: nrTone(),
      nrstyle: nrStyle(),
      nrpasses: nrPasses(),
    },
    hdrColorPayload(),
  );
}
async function loadCompare(frame) {
  if (!input || !cmpOn()) return;
  if (prevBusy) {
    prevPending = frame;
    return;
  } // the pending request re-enters through loadPreview
  prevBusy = true;
  const key = cmpSig(frame),
    a = cmpA(),
    b = $('cmpto').value,
    multi = cmpMulti(),
    video = input;
  const wantLast = cmpWantJ < 0;
  cmpWantJ = 0; // this run's own step-back request (a queued one keeps its own)
  cmpInflight = key;
  cmpSpin(true);
  txt('prevnote', () => tr('cmp.running'));
  let r;
  try {
    r = await ipcRenderer.invoke('compare', cmpOpts(frame, a, b, multi, 1));
  } catch (e) {
    r = { error: String(e) };
  }
  cmpSpin(false);
  prevBusy = false;
  if (prevPending != null) {
    const f = prevPending;
    prevPending = null;
    loadPreview(f);
  }
  if (input !== video || !cmpOn()) return; // another video or Compare off meanwhile: this result is no longer asked for
  if (!r || r.error) {
    cmpKey = ''; // never dedupe-skip a retry after a failure
    const err = r && r.error;
    txt('prevnote', () => tr('cmp.unavail', { err: err || tr('prev.noResult') }));
    return;
  }
  cmpRes = Object.assign(r, { modelA: a, modelB: b, shown: false });
  cmpKey = key;
  if (r.total) prevTotal = r.total;
  prevIdx = r.frame;
  cmpJ = wantLast ? r.multi - 1 : 1; // stepping back into a pair lands on its last new frame
  showCompare();
}
function showCompare() {
  const r = cmpRes;
  if (!r) return;
  const n = r.multi - 1;
  if (!r.shown) {
    setPrevImg($('cmpimg0'), r.real0);
    setPrevImg($('cmpimg2'), r.real1);
    r.shown = true;
  }
  const [x, y, mx, my] = cmpSwap ? [r.b, r.a, r.modelB, r.modelA] : [r.a, r.b, r.modelA, r.modelB];
  setPrevImg($('cmpimg1'), x[cmpJ - 1]);
  setPrevImg($('cmpimg3'), y[cmpJ - 1]);
  const label = (m) =>
    CMP_NAMES[m] + (m === 'rifedrba' ? ' ' + tr('cmp.noDrba') : '') + ' · ' + tr('cmp.newFrame', { j: cmpJ, n });
  $('cmplab0').textContent = tr('cmp.realN', { n: r.frame });
  $('cmplab1').textContent = label(mx);
  $('cmplab2').textContent = tr('cmp.realN', { n: r.frame + 1 });
  $('cmplab3').textContent = label(my);
  $('cmplabel').textContent =
    tr('cmp.frames', { a: r.frame, b: r.frame + 1 }) + (prevTotal ? ' ' + tr('cmp.of', { total: prevTotal }) : '');
  $('cmpback').disabled = r.frame <= 0 && cmpJ <= 1;
  $('cmpfwd').disabled = prevTotal > 0 && r.frame >= prevTotal - 2 && cmpJ >= n;
  const resampled = !gridMulti(),
    hdrNote = hdrOn() || (info && info.srcHdr),
    m = r.multi;
  txt(
    'prevnote',
    () =>
      (resampled ? tr('cmp.note.resample', { m }) + ' · ' : '') +
      (hdrNote ? tr('prev.note.hdr') + ' · ' : '') +
      tr('cmp.note.zoom'),
  );
}
i18nRelabel.push(() => {
  syncCompareUi();
  if (cmpOn()) showCompare();
});
function refreshCompare() {
  // settings changed: re-run the shown pair once they settle
  clearTimeout(cmpTimer);
  cmpTimer = setTimeout(() => {
    if (!input || !cmpOn()) return;
    const frame = cmpRes ? cmpRes.frame : lastPrevInput === input ? prevIdx : 'mid';
    if (cmpSig(frame) === (prevBusy ? cmpInflight : cmpKey)) return;
    loadCompare(frame);
  }, 300);
}
$('cmpto').onchange = () => {
  syncCompareUi();
  if (cmpOn()) refreshCompare();
  else refreshPreviewIfOpen();
};
$('cmpfwd').onclick = () => {
  const r = cmpRes;
  if (!r) return;
  if (cmpJ < r.multi - 1) {
    cmpJ++;
    showCompare();
  } else if (!prevTotal || r.frame < prevTotal - 2) loadCompare(r.frame + 1);
};
$('cmpback').onclick = () => {
  const r = cmpRes;
  if (!r) return;
  if (cmpJ > 1) {
    cmpJ--;
    showCompare();
  } else if (r.frame > 0) {
    cmpWantJ = -1;
    loadCompare(r.frame - 1);
  }
};
$('cmprand').onclick = () => loadCompare(randFrame());
// the panes' frame size from the result (an image swapping its src reads naturalWidth 0 until it decodes)
function cmpFrameW() {
  return cmpRes ? cmpRes.w : $('cmpimg0').naturalWidth;
}
function cmpFrameH() {
  return cmpRes ? cmpRes.h : $('cmpimg0').naturalHeight;
}
function applyCmpZoom() {
  for (let i = 0; i < 4; i++) {
    const w = $('cmpw' + i),
      img = $('cmpimg' + i);
    w.style.overflow = cmpZoom ? 'auto' : 'hidden';
    w.style.maxHeight = cmpZoom ? '300px' : '';
    img.style.width = cmpZoom ? cmpFrameW() * cmpZoom + 'px' : '100%';
    img.style.maxWidth = cmpZoom ? 'none' : '';
    img.style.imageRendering = cmpZoom >= 2 ? 'pixelated' : '';
    img.style.cursor = cmpZoom === CMP_ZOOMS[CMP_ZOOMS.length - 1] ? 'zoom-out' : 'zoom-in';
  }
}
function cmpScrollSync(src) {
  if (cmpSyncing) return;
  cmpSyncing = true;
  // pixel-absolute: the four panes hold frames of one size, so the same scroll shows the same spot
  for (let i = 0; i < 4; i++) {
    const w = $('cmpw' + i);
    if (w === src) continue;
    if (Math.abs(w.scrollLeft - src.scrollLeft) >= 1) w.scrollLeft = src.scrollLeft;
    if (Math.abs(w.scrollTop - src.scrollTop) >= 1) w.scrollTop = src.scrollTop;
  }
  cmpSyncing = false;
}
for (let i = 0; i < 4; i++) {
  const w = $('cmpw' + i),
    img = $('cmpimg' + i);
  w.addEventListener('scroll', () => {
    cmpScrollSync(w);
    const sp = $('cmpspin' + i);
    if (sp.style.display !== 'none') setSpin(w, sp, true);
  });
  img.onclick = (e) => {
    // the clicked spot in image pixels, read before the restyle moves it; the zoomed view centres on it
    const px = (e.offsetX / Math.max(1, img.clientWidth)) * cmpFrameW();
    const py = (e.offsetY / Math.max(1, img.clientHeight)) * cmpFrameH();
    cmpZoom = CMP_ZOOMS[(CMP_ZOOMS.indexOf(cmpZoom) + 1) % CMP_ZOOMS.length];
    applyCmpZoom();
    if (cmpZoom) {
      w.scrollLeft = px * cmpZoom - w.clientWidth / 2;
      w.scrollTop = py * cmpZoom - w.clientHeight / 2;
      cmpScrollSync(w);
    }
  };
}
// Hold Space: A and B trade panes (and labels) until it is let go; never while a text box or checkbox has the focus
window.addEventListener('keydown', (e) => {
  if (e.code !== 'Space' || uiMode !== 'video' || !cmpOn() || !cmpRes) return;
  const t: any = document.activeElement;
  if (t && /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName)) return;
  e.preventDefault();
  if (t && t.tagName === 'BUTTON') t.blur(); // a focused button would take the Space as a click
  if (!cmpSwap) {
    cmpSwap = true;
    showCompare();
  }
});
window.addEventListener('keyup', (e) => {
  if (e.code !== 'Space' || !cmpSwap) return;
  e.preventDefault();
  cmpSwap = false;
  showCompare();
});
window.addEventListener('blur', () => {
  if (cmpSwap) {
    cmpSwap = false;
    showCompare();
  }
});
// Save image: the four panes as shown (the visible crop at the shown zoom, else the whole frames), with their labels
$('cmpsave').onclick = async () => {
  const r = cmpRes;
  if (!r) return;
  const imgs = [0, 1, 2, 3].map((i) => $('cmpimg' + i));
  await Promise.all(imgs.map((i) => i.decode().catch(() => {}))); // a pane mid-swap draws once it has decoded
  const fw = cmpFrameW(),
    fh = cmpFrameH();
  if (!fw || !fh) return;
  const z = cmpZoom || 1,
    w0 = $('cmpw0');
  const sx = cmpZoom ? w0.scrollLeft / z : 0,
    sy = cmpZoom ? w0.scrollTop / z : 0;
  const cw = cmpZoom ? Math.min(fw - sx, w0.clientWidth / z) : fw,
    ch = cmpZoom ? Math.min(fh - sy, w0.clientHeight / z) : fh;
  const pw = Math.round(cw * z),
    ph = Math.round(ch * z),
    lab = 34,
    gap = 8;
  const cv = document.createElement('canvas');
  cv.width = 2 * pw + gap;
  cv.height = 2 * (ph + lab) + gap;
  const g = cv.getContext('2d');
  g.fillStyle = '#111';
  g.fillRect(0, 0, cv.width, cv.height);
  g.imageSmoothingEnabled = false;
  g.font = '600 20px "Segoe UI", sans-serif';
  g.fillStyle = '#ddd';
  g.textBaseline = 'middle';
  imgs.forEach((img, i) => {
    const x = (i % 2) * (pw + gap),
      y = Math.floor(i / 2) * (ph + lab + gap);
    g.fillText($('cmplab' + i).textContent, x + 4, y + lab / 2);
    g.drawImage(img, sx, sy, cw, ch, x, y + lab, pw, ph);
  });
  const file = await ipcRenderer.invoke(
    'pick-save',
    input.replace(/\.[^.\\/]+$/, '') + '_compare_' + r.frame + '_' + cmpJ + '.png',
    'image',
  );
  if (!file) return;
  const blob: any = await new Promise((res) => cv.toBlob(res, 'image/png'));
  _fs.writeFileSync(file, new Uint8Array(await blob.arrayBuffer()));
  txt('prevnote', () => tr('cmp.saved', { file }));
};
// Save clip: one second from the shown pair as a lossless video of the 2x2 (main's clip-begin / clip-frame /
// clip-end, FFV1), every output frame at the render's output rate; at a real frame's slot both model panes show that
// real frame (the models keep it as it is). The labels and layout of Save image, the whole frames.
async function cmpImg(file) {
  const img = new Image();
  img.src = URL.createObjectURL(new Blob([_fs.readFileSync(file)], { type: 'image/png' }));
  await img.decode();
  return img;
}
$('cmpclip').onclick = async () => {
  const r = cmpRes;
  if (!r || prevBusy) return;
  const n = r.multi,
    pairs = Math.max(1, Math.min(240, Math.round(srcFps()), (prevTotal || r.total) - 1 - r.frame));
  const file = await ipcRenderer.invoke(
    'pick-save',
    input.replace(/\.[^.\\/]+$/, '') + '_compare_' + r.frame + '.mkv',
    'clip',
  );
  if (!file || prevBusy || cmpRes !== r) return;
  prevBusy = true;
  cmpSpin(true);
  txt('prevnote', () => tr('clip.running', { pairs }));
  let err = '',
    began = false,
    reals = [];
  try {
    const c = await ipcRenderer.invoke('compare', cmpOpts(r.frame, r.modelA, r.modelB, n, pairs));
    cmpSpin(false);
    if (!c || c.error) throw new Error((c && c.error) || tr('prev.noResult'));
    const W = c.w,
      H = c.h,
      lab = 34,
      gap = 8,
      cw = 2 * W + gap,
      ch = 2 * (H + lab) + gap;
    const [num, den] = String((info && info.fps) || '24/1')
      .split('/')
      .map(Number);
    began = await ipcRenderer.invoke('clip-begin', { out: file, w: cw, h: ch, fps: num * n + '/' + (den || 1) });
    const cv = document.createElement('canvas');
    cv.width = cw;
    cv.height = ch;
    const g = cv.getContext('2d', { willReadFrequently: true });
    const names = [CMP_NAMES[r.modelA] + (r.modelA === 'rifedrba' ? ' ' + tr('cmp.noDrba') : ''), CMP_NAMES[r.modelB]];
    let realsP = -1;
    for (let o = 0; o <= c.pairs * n; o++) {
      const p = Math.min(Math.floor(o / n), c.pairs - 1),
        j = o - p * n,
        real = j === 0 || j === n;
      if (p !== realsP) {
        reals.forEach((i) => URL.revokeObjectURL(i.src));
        reals = [await cmpImg(c.reals[p]), await cmpImg(c.reals[p + 1])];
        realsP = p;
      }
      const at = p * (n - 1) + j - 1; // the new frame's place in c.a / c.b (outputs o not a multiple of n)
      const right = real ? [reals[j ? 1 : 0], reals[j ? 1 : 0]] : [await cmpImg(c.a[at]), await cmpImg(c.b[at])];
      const what =
        ' · ' + (real ? tr('clip.real', { n: c.frame + p + (j ? 1 : 0) }) : tr('cmp.newFrame', { j, n: n - 1 }));
      const panes = [
        [reals[0], tr('cmp.realN', { n: c.frame + p })],
        [right[0], names[0] + what],
        [reals[1], tr('cmp.realN', { n: c.frame + p + 1 })],
        [right[1], names[1] + what],
      ];
      g.fillStyle = '#111';
      g.fillRect(0, 0, cw, ch);
      g.font = '600 20px "Segoe UI", sans-serif';
      g.fillStyle = '#ddd';
      g.textBaseline = 'middle';
      panes.forEach(([img, text]: any, i) => {
        const x = (i % 2) * (W + gap),
          y = Math.floor(i / 2) * (H + lab + gap);
        g.fillText(text, x + 4, y + lab / 2);
        g.drawImage(img, x, y + lab, W, H);
      });
      if (!real) right.forEach((i) => URL.revokeObjectURL(i.src));
      const at1 = o + 1,
        all = c.pairs * n + 1;
      txt('prevnote', () => tr('clip.frame', { o: at1, total: all }));
      if (!(await ipcRenderer.invoke('clip-frame', new Uint8Array(g.getImageData(0, 0, cw, ch).data.buffer))))
        throw new Error(tr('clip.encStopped'));
    }
  } catch (e) {
    err = (e && e.message) || String(e);
  }
  reals.forEach((i) => URL.revokeObjectURL(i.src));
  cmpSpin(false);
  if (began) {
    const e = await ipcRenderer.invoke('clip-end');
    if (!err && e.rc !== 0) err = tr('clip.encFailed', { rc: e.rc, err: e.error });
  }
  prevBusy = false;
  if (prevPending != null) {
    const f = prevPending;
    prevPending = null;
    loadPreview(f);
  }
  txt('prevnote', () => (err ? tr('clip.notSaved', { err }) : tr('cmp.saved', { file })));
};

// Interpolation on/off is expressed by the MODEL GROUP itself: any model ticked = on, none ticked
// = off. With it off the app only re-encodes (sharpen, upscale, HDR and restore still apply), so
// the fps and screen-rate controls grey out and the output keeps the source frame rate.
// The choice, including the "none" state, persists on the interpModel key.
function syncInterp() {
  const on = interpOn();
  // The model boxes themselves stay ENABLED always - they ARE the toggle. Only their dependent
  // sub-options and setup hints follow the on/off state.
  $('rifedrba').disabled = !on;
  if (!on) {
    $('frucsetup').style.display = 'none';
    $('rifedrbarow').style.display = 'none';
  }
  applyScreenFps(); // recomputes multi/fpsin enable, the fps mode, and the output name
  try {
    lvModelUi();
    lvSendOpts();
  } catch {} // live follows the model choice
}
syncInterp();

// Drop anything onto the window: a video loads for processing, a .zip installs the RTX Video SDK (the
// other tools download in one click). All drops are ignored while a job is running.
document.addEventListener('dragover', (e) => {
  e.preventDefault();
  if ($('cancel').disabled) document.body.classList.add('drag');
});
document.addEventListener('dragleave', (e) => {
  if (e.relatedTarget === null) document.body.classList.remove('drag');
});
document.addEventListener('drop', (e) => {
  e.preventDefault();
  document.body.classList.remove('drag');
  if (!$('cancel').disabled) return; // a job is in progress
  const files = [...(e.dataTransfer.files || [])];
  if (!files.length) return;
  const nm = files[0].name,
    p = webUtils.getPathForFile(files[0]);
  if (/\.zip$/i.test(nm)) {
    // a .zip is the RTX Video SDK: install the RTX runtime
    showWorkspace(); // RTX install feedback lives in #workspace, so reveal it even before a video is loaded
    if ($('rtxsetup').style.display === 'none') $('rtxsetup').style.display = 'block'; // reveal install feedback
    doInstall(p);
  } else {
    // Anything else: load it as a video; several dropped files become a batch queue. Both the
    // Select video button and the file rows are hidden in Live mode, but a drop still fires, so
    // switch to Video mode first or the loaded file would land on an invisible workspace.
    setMode('video');
    queueVideos(files.map((f) => webUtils.getPathForFile(f)));
  }
});

// Resume an interrupted batch: if the previous session died mid-run (crash, forced close), its
// unfinished inputs are still in localStorage (saveBatch). Re-queue them now - the first file loads
// and the rest wait - but do NOT auto-start; the user clicks Smooth It! when ready. The key is
// cleared immediately so a failure during this restore can't loop; startRun re-saves it.
try {
  const pend = JSON.parse(localStorage.getItem('batchPending') || 'null');
  localStorage.removeItem('batchPending');
  if (Array.isArray(pend) && pend.length && pend.every((p) => typeof p === 'string')) {
    log(
      '>> Restored ' +
        pend.length +
        ' unfinished file(s) from the last session; an interrupted render resumes where it left off\n',
    );
    queueVideos(pend);
  }
} catch {
  /* no saved batch */
}

function startRun() {
  if (!input) return;
  const t = targetFps();
  if (!fpsValid(t)) {
    alert(tr('err.fpsRange', { min: sliderMin(), max: sliderMax() }));
    return;
  }
  // 1x (target fps == source fps) adds no frames - it just re-generates every frame as a midpoint tween
  // (softer, half-frame shift), a strictly worse video. Block it with guidance instead of running it.
  if (interpOn() && Math.round(t) <= Math.round(srcFps())) {
    alert(tr('err.oneX', { fps: Math.round(srcFps()) }));
    return;
  }
  // DLSS 4.5 makes only whole multiples; a target between them goes to the render as --fps, which runs the smallest
  // multiple whose every k-th frame lands on it (24 -> 60 = 5x, every 2nd) and says so in the log
  const sharpenStrength = $('sharpen').checked ? parseFloat($('sharpval').value) : 0;
  const dims = upDims(); // resize target ({w,h} or null)
  const factor = upFactor(); // arbitrary resize factor (0 = off, <1 downscales)
  const useRtxVsr = factor > 1 && $('rtxvsr').checked && rtxReady.vsr; // AI upscale of an enlarging resize, else Lanczos3
  const useFsrUp = factor > 1 && fsrUpOn(); // AMD FSR 4 instead of RTX VSR when ticked
  const useA4kUp = factor > 1 && a4kUpOn(); // Anime4K the same way
  const useAcnUp = factor > 1 && acnUpOn(); // ArtCNN the same way
  const useXessUp = factor > 1 && xessUpOn(); // Intel XeSS the same way
  const rtxhdr = hdrOn(); // HDR only when its runtime is installed and the source is SDR
  // Interpolation is the main effect; with it off the run still has work if sharpening, upscaling or
  // HDR is on. Bail only when nothing at all is enabled.
  if (!interpOn() && sharpenStrength <= 0 && factor <= 0 && !rtxhdr && !restoreOn() && !nrOn()) {
    alert(tr('err.nothing'));
    return;
  }
  procStart = 0;
  baseK = 0;
  etaAnchor = 0;
  etaAnchorTime = 0;
  lastFps = 0;
  curDone = 0;
  curPct = 0;
  projBytes = 0;
  clearInterval(etaTimer);
  etaTimer = setInterval(paint, 500); // ticks the ETA down between frames
  // Live thumbnail: only frames written by THIS run may show, so anchor the mtime watermark to
  // whatever is on disk now (a previous run's leftover) and hide the old image until a new write.
  try {
    liveM = _fs.statSync(livePath).mtimeMs;
  } catch {
    liveM = 0;
  }
  $('live').style.display = 'none';
  // Reveal the live area (image appears on the first written frame) so its Hide/Show toggle is
  // reachable for the whole run, and re-assert the engine-side on/off to match the saved choice.
  $('livewrap').style.display = 'flex';
  applyLiveOff();
  clearInterval(liveTimer);
  liveTimer = setInterval(updateLive, 1000);
  fpp = outRatio(); // output frames produced per source frame (1 when not interpolating)
  const srcFrames = info.nb || Math.round(info.dur * srcFps()) || 1;
  totalGen = Math.max(1, Math.round(srcFrames * fpp)); // total frames the conversion will output
  cancelled = false;
  // #go stays enabled and becomes the Pause toggle for the duration of the run (see goMode).
  goMode = 'running';
  $('go').textContent = tr('run.pause');
  $('go').title = tr('run.pause.tip');
  if (!resumeTipShown) {
    resumeTipShown = true; // once per session; batch auto-advance re-enters startRun per file
    log('>> Tip: you can Pause, or even close the app mid-render: the render resumes where it left off\n');
  }
  modeBtnUi(true);
  $('cmpto').disabled = true; // Compare waits for the render (syncCompareUi at engine-done)
  $('pick').disabled = true;
  $('changeout').disabled = true;
  $('out').disabled = true;
  for (const b of MODEL_BOXES()) b.disabled = true;
  $('fpsin').disabled = true;
  $('sharpen').disabled = true;
  $('sharpval').disabled = true;
  $('restore').disabled = true;
  $('dlssmode').disabled = true;
  $('dlssrange').disabled = true;
  $('dlssnr').disabled = true;
  $('nrstructure').disabled = true;
  $('nrtone').disabled = true;
  for (const b of $('nrstyleseg').querySelectorAll('button')) b.disabled = true;
  $('nrpasses').disabled = true;
  $('nrmask').disabled = true;
  $('upres').disabled = true;
  $('upcustom').disabled = true;
  $('outcodec').disabled = true;
  $('rtxvsr').disabled = true;
  $('vsrq').disabled = true;
  $('fsrup').disabled = true;
  $('a4kup').disabled = true;
  $('a4kaa').disabled = true;
  $('acnup').disabled = true;
  $('xessup').disabled = true;
  $('gpufit').disabled = true;
  $('rtxhdr').disabled = true;
  $('hdrdynvib').disabled = true;
  $('hdrsat').disabled = true;
  $('hdrvib').disabled = true;
  $('hdrcon').disabled = true;
  $('hdrsb').disabled = true;
  $('open').disabled = true;
  $('play').disabled = true;
  $('cancel').disabled = false;
  $('playprev').disabled = true;
  $('playprev').style.display = 'none';
  lastPreview = null;
  dlssPreempt = null;
  syncTargetUI();
  // What this run does, for the status / log (mainly relevant when interpolation is off).
  // the pass names ('=name' stays as is, else a language key), read in the language of each repaint
  const passes = [];
  if (restoreOn()) passes.push('run.pass.restore');
  if (factor > 0) passes.push(factor < 1 ? 'run.pass.down' : 'run.pass.up');
  if (rtxhdr) passes.push('=HDR');
  if (sharpenStrength > 0) passes.push('run.pass.sharpen');
  const offLabel = () =>
    (passes.map((p) => (p[0] === '=' ? p.slice(1) : tr(p))).join(' + ') || tr('run.pass.processing')).replace(
      /^./,
      (c) => c.toUpperCase(),
    ) + '...';
  const warm = interpOn(),
    tag = fileTag();
  txt('status', () => tag() + (warm ? tr('run.warming') : offLabel()));
  $('fill').style.width = '0%';
  $('pct').textContent = '0%';
  txt('frames', () => (warm ? tr('run.warmingFrames') : offLabel()));
  const gm = gridMulti();
  const rr = t / srcFps();
  const ratioTxt = gm
    ? gm + '×, on-grid'
    : rr < 9.95
      ? rr.toFixed(1) + '×, resampled'
      : '≈' + Math.round(rr) + '×, resampled';
  const what = interpOn() ? t.toFixed(targetDecimals()) + ' fps (' + ratioTxt + ')' : 'no interpolation';
  const extra =
    (dims
      ? '  ·  ' + (useRtxVsr ? 'RTX VSR' : factor < 1 ? 'downscale' : 'upscale') + ' ' + dims.w + '×' + dims.h
      : '') +
    (rtxhdr ? '  ·  RTX HDR' : '') +
    (sharpenStrength > 0 ? '  ·  Sharpen ' + sharpenStrength : '') +
    (restoreOn() ? '  ·  Restore' : '');
  log('>> ' + input + '  ->  ' + what + extra + '\n');
  const payload: any = { input, multi: gm || 2, output: lastOut, interp: interpOn() }; // on-grid: the derived integer multiplier; resample: --fps overrides this placeholder
  if (interpOn() && !gm) payload.fps = t; // resample only: non-integer-multiple target
  // rifedrba = RIFE with the DRBA sub-option on, rife = plain RIFE (bundled, always ready), dlssg = "DLSS 4.5" (Frame
  // Generation), fruc = "NVIDIA Smooth Motion" (NvOFFRUC), nvof = NVIDIA Optical Flow (engine --nvof), fsrfg = AMD FSR
  // frame generation (engine --fsrfg), xefg = Intel XeSS-FG (engine --xess-fg); none = GMFSS
  const model = renderModel();
  if (model) payload.model = model;
  payload.sharpen = sharpenStrength; // 0 = engine leaves frames untouched
  if (restoreOn()) {
    payload.restore = true;
    payload.restoreModel = restoreModelId();
  } // Real-ESRGAN or AnimeJaNai
  if (nrOn()) {
    payload.dlssnr = true;
    payload.nrstructure = nrStructure();
    payload.nrtone = nrTone();
    payload.nrstyle = nrStyle();
    payload.nrpasses = nrPasses();
  } // DLSS 5 (runtime installed)
  payload.codec = $('outcodec').value; // output codec family (hevc default / av1 / vvc)
  if (factor > 0) {
    payload.upscale = factor; // arbitrary upscale factor (target height / source)
  }
  if (useRtxVsr) {
    payload.rtxvsr = true;
    payload.vsrquality = vsrQuality();
  } // AI upscale via the RTX Video SDK (else Lanczos3)
  if (useFsrUp) payload.fsrup = true; // AMD FSR 4 upscaling (engine/fsrup)
  if (useA4kUp) payload.anime4k = true; // Anime4K upscaling (inside the host)
  if (useA4kUp && a4kAaOn()) payload.a4kaa = true; // its Mode A+A
  if (useAcnUp) payload.artcnn = true; // ArtCNN upscaling (engine/artcnn)
  if (useXessUp) payload.xess = true; // Intel XeSS upscaling (engine/xess)
  if (!$('gpufit').checked) payload.nogpufit = true; // no video memory fit: the same fps, slower when memory runs short
  if (rtxhdr) {
    payload.rtxhdr = true; // HDR10 at the Peak box's nits or the app window's display's
    const hp = hdrColorPayload();
    payload.hdrcolor = hp.color; // raw = NVIDIA's result
    payload.hdrsat = hp.saturation; // SDK Saturation (100 = NVIDIA's default)
    if (hp.nits > 0) payload.hdrnits = hp.nits; // the Peak box (else the display's peak)
    payload.hdrcon = sdkCon(); // SDK Contrast (100 = neutral)
    payload.hdrsb = effSatBoost(); // vibrance uniform gain (0..1)
    payload.hdrvib = effVibrance();
    if (dvOn() && dvReady.ready) {
      payload.dv = true;
      if ($('dv84export').checked) payload.dvprofile = '8.4';
    } // also export Dolby Vision 8.1 / 8.4 (needs HDR)
    if ($('hpexport').checked && !$('dv84export').checked && hpReady.ready) payload.hp = true; // also embed HDR10+ metadata (needs HDR; not on 8.4's HLG)
  }
  // no DLSS mode: a file render runs every pass at the output size (DLAA); the mode is Live's speed lever
  saveBatch([input, ...batch]); // survives a crash/close: the next launch re-queues these files
  ipcRenderer.send('run', payload);
}
// #go dispatches on its current mode: start a fresh run when idle, otherwise toggle Pause/Resume.
$('go').onclick = () => {
  if (goMode === 'idle') {
    startRun();
    return;
  }
  if (goMode === 'running') {
    // Pause: the engine finishes the frames already queued to the encoder, then holds before the
    // next source pair. Stop the ETA countdown (it would keep ticking down through the idle time).
    goMode = 'paused';
    $('go').textContent = tr('run.resume');
    $('go').title = tr('run.resumeNow.tip');
    setStatus('run.status.paused', undefined, true);
    clearInterval(etaTimer);
    etaTimer = null;
    ipcRenderer.send('pause');
  } else {
    // paused -> resume
    goMode = 'running';
    $('go').textContent = tr('run.pause');
    // The pause preview describes the paused-at point; once we resume it is stale, so hide it (the
    // engine overwrites it on the next Pause and deletes it at finalize).
    $('playprev').style.display = 'none';
    $('playprev').disabled = true;
    lastPreview = null;
    $('go').title = tr('run.pause.tip');
    // Re-anchor the pace clock so the paused span doesn't drag the mean fps (and ETA) down; skip
    // if warmup never finished (procStart still 0; the first-frame handler will anchor instead).
    if (procStart) {
      procStart = Date.now();
      baseK = curDone;
    }
    etaTimer = setInterval(paint, 500);
    setStatus('run.status.smoothing', undefined, true);
    ipcRenderer.send('resume');
  }
};
// a switch mid-run relabels the Pause / Resume toggle (the status line follows the run's next event)
i18nRelabel.push(() => {
  if (goMode === 'running') {
    $('go').textContent = tr('run.pause');
    $('go').title = tr('run.pause.tip');
  } else if (goMode === 'paused') {
    $('go').textContent = tr('run.resume');
    $('go').title = tr('run.resumeNow.tip');
  }
});
$('open').onclick = () => {
  if (lastOut) shell.showItemInFolder(lastOut);
};
$('play').onclick = () => {
  if (lastOut) shell.openPath(lastOut);
}; // system default player
$('playprev').onclick = () => {
  if (lastPreview) shell.openPath(lastPreview);
}; // partial preview, sys player
$('changeout').onclick = async () => {
  const chosen = await ipcRenderer.invoke('pick-output', customOut || outName());
  if (chosen) {
    customOut = chosen;
    refresh();
  }
};
$('cancel').onclick = async () => {
  cancelled = true;
  $('cancel').disabled = true;
  setStatus('run.status.cancelling');
  // The bar and live thumbnail describe progress that is being deleted: reset them right away
  // (engine-out drops any PROGRESS still flushing from the dying engine while cancelled, so
  // nothing can repaint the bar between this reset and engine-done).
  clearInterval(etaTimer);
  etaTimer = null;
  $('fill').style.width = '0%';
  $('pct').textContent = '0%';
  raw('frames', '');
  $('livewrap').style.display = 'none';
  $('live').style.display = 'none';
  // invoke resolves only after main killed the engine AND deleted the .part/resume artifacts,
  // so the probe below sees the truth (engine-done skips its own probe on a cancelled run).
  try {
    await ipcRenderer.invoke('cancel');
  } catch {}
  scheduleCheckResume();
};

ipcRenderer.on('engine-out', (_e, t) => {
  // Pause partial-preview signals: on Pause the engine remuxes the frames rendered so far with the
  // source's audio/subtitles into a playable file and logs PREVIEW_READY <path> (or PREVIEW_PENDING
  // when too few frames have flushed yet). Surface a "Play preview" button so the user can judge the
  // result with sound before resuming or cancelling.
  const pr = /PREVIEW_READY\s+([^\r\n]+)/.exec(t);
  if (pr) {
    lastPreview = pr[1].trim();
    $('playprev').disabled = false;
    $('playprev').style.display = '';
    setStatus('run.status.prevReady', undefined, true);
  } else if (/PREVIEW_PENDING/.test(t)) {
    setStatus('run.status.prevPending', undefined, true);
  }
  // DLSS-FG was preempted (RTX Video enhancement) and stopped after its restarts, but banked a clean
  // resumable checkpoint. Remember the % so engine-done presents it as a resumable stop, not a crash.
  const dp = /DLSS_PREEMPTED\s+(\d+)/.exec(t);
  if (dp) dlssPreempt = +dp[1];
  // The engine emits its EXACT output frame count as OUTFRAMES at the end; it's authoritative over the
  // pre-render estimate (container nb_frames * ratio), which can be off by one when nb_frames misreports.
  const of = /OUTFRAMES\s+(\d+)/.exec(t);
  if (of) totalGen = +of[1] || totalGen;
  // SIZE lines carry (bytes so far, projected final bytes); like PROGRESS, a chunk can batch
  // several, so keep the newest projection for the frames line.
  let sm = null,
    sg = /SIZE\s+\d+\s+(\d+)/g,
    sx;
  while ((sx = sg.exec(t)) !== null) sm = sx;
  if (sm) projBytes = +sm[1] || 0;
  // A single stderr chunk can batch several PROGRESS lines; take the last so the bar and the
  // frame/elapsed feeding the ETA reflect the newest count, never a stale one from the batch.
  let m = null,
    g = /PROGRESS\s+(\d+)\/(\d+)/g,
    x;
  while ((x = g.exec(t)) !== null) m = x;
  if (m && !cancelled) {
    const done = +m[1],
      tot = +m[2] || 1;
    // Clamp to [0,1]: a missing/zero denominator (e.g. an MKV with no nb_frames) would
    // otherwise make frac huge and send the bar past 100% (the runaway 9999999% bug).
    const frac = Math.min(1, Math.max(0, done / tot));
    const pct = 100 * frac; // 0% at first generated frame -> 100% at last
    $('fill').style.width = pct + '%';
    $('pct').textContent = Math.round(pct) + '%'; // the percentage now lives inside the bar
    curDone = done;
    curPct = pct;
    const now = Date.now();
    if (!procStart) {
      // First generated frame is in, so warmup (model load + first inference) is over. Anchor the
      // clock here so the average pace excludes that one-time startup cost and this first frame.
      procStart = now;
      baseK = done;
      setStatus('run.status.smoothing', undefined, true);
      paint();
      return;
    }
    // ETA from the mean pace over EVERY frame since warmup (not a trailing window): all frames
    // generated since the anchor over all time since the anchor. Steady and self-correcting as the
    // run goes on; the 0.5s timer counts the displayed seconds down between PROGRESS frames.
    const elapsed = (now - procStart) / 1000,
      generated = done - baseK;
    if (elapsed > 0 && generated > 0) {
      lastFps = (generated * fpp) / elapsed; // mean output frames/sec across the whole run
      etaAnchor = lastFps > 0 ? ((tot - done) * fpp) / lastFps : 0;
      etaAnchorTime = now;
    }
    paint();
  } else if (!m) {
    // PROGRESS chunks are never logged; while cancelled they are dropped entirely
    if (/\[trt\] building/.test(t)) setStatus('run.status.building');
    // OUTFRAMES is machine-only, and the engine's "done ..." line duplicates the ">> Saved: <path>"
    // the GUI prints on engine-done - strip both from the GUI log (they still show in CLI output).
    const shown = t
      .replace(/OUTFRAMES\s+\d+\n?/g, '')
      .replace(/SIZE\s+\d+\s+\d+\n?/g, '')
      .replace(/^done \d+ (frames|pairs)[^\n]*\n?/gm, '');
    if (shown.trim() && !/Optical Flow Grid Size/.test(shown)) log(shown);
  }
});
// Launch-time update notice (main's checkForUpdate): a one-line link under the subtitle; the check
// is silent unless a newer GitHub release exists, so this stays hidden for up-to-date installs.
ipcRenderer.on('update-available', (_e, u) => {
  if (!u || !u.version) return;
  $('updatever').textContent = u.version;
  $('updatelink').onclick = (ev) => {
    ev.preventDefault();
    shell.openExternal(u.url);
  };
  $('update').style.display = 'block';
});

ipcRenderer.on('engine-done', (_e, code) => {
  // Reset the Pause/Resume toggle back to a fresh "Smooth It!" first, so every exit path
  // (done / cancelled / failed) restores it and the batch auto-advance below re-enters startRun.
  goMode = 'idle';
  $('go').textContent = tr('run.go');
  $('go').title = ''; // checkResume below re-fills the tooltip
  // The run ended (done/cancelled/failed): the engine deletes the partial preview at finalize, so hide
  // the "Play preview" button - the finished "Play video" takes over on success.
  $('playprev').style.display = 'none';
  $('playprev').disabled = true;
  lastPreview = null;
  clearInterval(etaTimer);
  etaTimer = null; // stop the ETA countdown
  // Final thumbnail pull (the last written frame), except on cancel: the click hid the preview.
  clearInterval(liveTimer);
  liveTimer = null;
  if (!cancelled) updateLive();
  modeBtnUi(lvState !== 'idle');
  $('go').disabled = false;
  $('pick').disabled = false;
  $('changeout').disabled = false;
  $('out').disabled = false;
  for (const b of MODEL_BOXES()) b.disabled = false;
  $('fpsin').disabled = false;
  $('sharpen').disabled = false;
  $('sharpval').disabled = false;
  $('restore').disabled = false;
  $('dlssmode').disabled = false;
  $('dlssrange').disabled = false;
  $('dlssnr').disabled = false;
  $('nrstructure').disabled = false;
  $('nrtone').disabled = false;
  for (const b of $('nrstyleseg').querySelectorAll('button')) b.disabled = false;
  $('nrpasses').disabled = false;
  $('nrmask').disabled = false;
  $('upres').disabled = false;
  $('upcustom').disabled = false;
  $('outcodec').disabled = false;
  $('rtxvsr').disabled = false;
  $('vsrq').disabled = false;
  $('fsrup').disabled = false;
  $('a4kup').disabled = false;
  a4kAaUi();
  $('acnup').disabled = false;
  $('xessup').disabled = false;
  $('gpufit').disabled = false;
  $('rtxhdr').disabled = !!(info && info.srcHdr);
  $('hdrdynvib').disabled = false;
  $('hdrcon').disabled = false;
  syncHdrColor();
  $('cancel').disabled = true;
  syncInterp(); // re-assert the interp / screen-rate state after the run re-enabled the inputs
  syncCompareUi();
  if (cancelled) {
    setStatus('run.status.cancelled');
    log('>> Cancelled\n');
    if (batch.length) log('>> Batch cleared (' + batch.length + ' queued files not processed)\n');
    batch = [];
    batchTotal = 1;
    batchFailed = [];
    saveBatch(null);
  } else if (code === 0) {
    $('fill').style.width = '100%';
    $('pct').textContent = '100%';
    const all = totalGen;
    txt('frames', () => tr('run.frames', { done: all, total: all }));
    setStatus('run.status.done');
    $('open').disabled = false;
    $('play').disabled = false;
    log('>> Saved: ' + lastOut + '\n');
    if (batch.length) advanceBatch();
    else finishBatch();
  } else if (dlssPreempt !== null) {
    // Not a failure: DLSS-FG was preempted by RTX Video enhancement and stopped after its restarts,
    // but the render banked a clean, fully-resumable checkpoint. Present it as a resumable stop; the
    // checkResume below turns #go into "Resume" so the user closes the video and continues.
    setStatus('run.status.preempted', { pct: dlssPreempt });
    log(
      '>> DLSS Frame Generation was preempted (RTX Video enhancement); ~' +
        dlssPreempt +
        '% saved and resumable. Close the interfering video, then Resume (or use GMFSS).\n',
    );
    if (batch.length) {
      log('>> Continuing with the next queued file\n');
      advanceBatch();
    } else if (batchTotal > 1) finishBatch();
    else saveBatch(null);
  } else {
    setStatus('run.status.failed', { code });
    log('>> Failed (exit ' + code + ')\n');
    // A failed file no longer kills the rest of an unattended batch: note it and keep going.
    if (batchTotal > 1) batchFailed.push(input);
    if (batch.length) {
      log('>> Continuing with the next queued file\n');
      advanceBatch();
    } else if (batchTotal > 1) finishBatch();
    else saveBatch(null);
  }
  // A finished/failed run changes resumability; re-label #go. Cancelled runs are probed by the
  // Cancel click handler instead, after main's artifact cleanup resolves (probing here would
  // still see the resume files and wrongly announce "Interrupted render found").
  if (!cancelled) scheduleCheckResume();
});
// Batch auto-advance: load the next queued file and start it with the same settings. batchAuto
// suppresses the modal probe alert and the preview refresh while the queue advances unattended;
// an unreadable file counts as failed and the queue keeps moving.
function advanceBatch() {
  const next = batch.shift();
  saveBatch([next, ...batch]);
  batchAuto = true;
  loadVideo(next).then((ok) => {
    batchAuto = false;
    if (ok) $('go').onclick();
    else {
      batchFailed.push(next);
      if (batch.length) advanceBatch();
      else finishBatch();
    }
  });
}
// Whole queue drained: per-batch summary (incl. failures), native notification, reset + unpersist.
function finishBatch() {
  const fails = batchFailed.length;
  if (batchTotal > 1) {
    setStatus(fails ? 'run.batch.doneFails' : 'run.batch.done', { ok: batchTotal - fails, total: batchTotal, fails });
    if (fails) log('>> Failed files:\n' + batchFailed.map((f) => '>>   ' + f).join('\n') + '\n');
  }
  ipcRenderer.send(
    'render-complete',
    batchTotal > 1
      ? tr(fails ? 'notify.batchFails' : 'notify.batch', { ok: batchTotal - fails, total: batchTotal, fails })
      : tr('notify.done'),
  );
  batchTotal = 1;
  batchFailed = [];
  saveBatch(null);
}
