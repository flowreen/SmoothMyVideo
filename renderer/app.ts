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
