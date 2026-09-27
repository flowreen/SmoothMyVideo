const { ipcRenderer, shell, webUtils, webFrame } = require('electron');
// Announce every page load (app launch OR a reload). On a reload this lets main kill any engine
// orphaned by the previous page - e.g. an accidental Ctrl+R mid-render - so the app comes up clean
// instead of the refreshed UI racing a still-running engine (the 1%->15%->1% progress ping-pong).
ipcRenderer.send('renderer-ready');

// --- UI zoom: Ctrl+= / Ctrl+- / Ctrl+0 and Ctrl+scrollwheel, browser-style ------------------------
// Chromium zoom levels: factor = 1.2^level; half-level steps ~= 10% per notch, clamped to a range
// where the tall single-column layout stays usable. Persisted so the app reopens at the same zoom.
function applyZoom(z){
  z = Math.max(-4, Math.min(4, z));
  webFrame.setZoomLevel(z);
  localStorage.setItem('zoomLevel', String(z));
}
applyZoom(parseFloat(localStorage.getItem('zoomLevel')) || 0);
window.addEventListener('keydown', (e) => {
  if(!e.ctrlKey || e.altKey || e.metaKey) return;
  if(e.key === '=' || e.key === '+'){ applyZoom(webFrame.getZoomLevel() + 0.5); e.preventDefault(); }
  else if(e.key === '-'){ applyZoom(webFrame.getZoomLevel() - 0.5); e.preventDefault(); }
  else if(e.key === '0'){ applyZoom(0); e.preventDefault(); }
});
window.addEventListener('wheel', (e) => {
  if(!e.ctrlKey) return;
  e.preventDefault();   // never page-scroll while zooming (needs the non-passive listener)
  applyZoom(webFrame.getZoomLevel() + (e.deltaY < 0 ? 0.5 : -0.5));
}, { passive: false });
const path = require('path');
let info = null, input = null, lastOut = null, customOut = null, fpp = 1, totalGen = 1, procStart = 0, baseK = 0, cancelled = false;
let isExample = false;   // the bundled first-page example clip is loaded (Smooth It! gated)
let lastPreview = null;   // path of the latest pause-generated partial preview (PREVIEW_READY), or null
let dlssPreempt = null;   // pct if DLSS-FG stopped on RTX Video preemption (a clean resumable stop), else null
// Batch queue: inputs waiting after the current one. Filled by a multi-select pick or a multi-file
// drop; consumed one at a time on engine-done (same settings for every file, default output names).
let batch = [], batchTotal = 1, batchAuto = false, batchFailed = [];
// The unfinished inputs (current file first) are mirrored to localStorage while a run is live, so a
// crash or accidental close mid-batch can be resumed: the next launch re-queues them (no auto-start).
function saveBatch(list){ try {
  if(list && list.length) localStorage.setItem('batchPending', JSON.stringify(list));
  else localStorage.removeItem('batchPending');
} catch {} }
// #go doubles as the run's Pause/Resume toggle: 'idle' = "Smooth It!" (starts a run), 'running' =
// "Pause" (holds generation), 'paused' = "Resume" (continues). engine-done resets it to 'idle'.
let goMode = 'idle';
function fileTag(){ return batchTotal > 1 ? 'File ' + (batchTotal - batch.length) + '/' + batchTotal + ' · ' : ''; }
// ETA is a live countdown. Each PROGRESS frame re-anchors it to the average pace since warmup,
// and a timer ticks the displayed seconds down between frames so it never sits frozen. curDone/
// curPct hold the last frame numbers so the timer can repaint the line with no new PROGRESS.
let etaAnchor = 0, etaAnchorTime = 0, lastFps = 0, etaTimer = null, curDone = 0, curPct = 0, projBytes = 0;
function fmtEta(s){ s=Math.max(0,Math.round(s)); const h=Math.floor(s/3600), m=Math.floor(s%3600/60), ss=s%60;
  return h>0 ? h+'h '+String(m).padStart(2,'0')+'m '+String(ss).padStart(2,'0')+'s'
       : m>0 ? m+'m '+String(ss).padStart(2,'0')+'s' : ss+'s'; }
const MONTHS = ['January','February','March','April','May','June','July','August','September','October','November','December'];
function fmtFinish(s){ const d = new Date(Date.now() + Math.max(0,s)*1000);   // wall-clock end time
  return String(d.getHours()).padStart(2,'0')+':'+String(d.getMinutes()).padStart(2,'0')+' '
       + String(d.getDate()).padStart(2,'0')+'-'+MONTHS[d.getMonth()]+'-'+d.getFullYear(); }

const $ = (id) => document.getElementById(id);
// Stored-settings schema version: bump whenever a localStorage key or its scale/meaning changes,
// and the first launch of the new build wipes every stored setting once, so stale keys can never
// conflict with the new UI (e.g. an HDR slider rescale or a key rename).
const SETTINGS_VERSION = '3';   // 3 = the unified target-fps slider (key fpsTarget) replaced the multiplier dropdown
if(localStorage.getItem('settingsVersion') !== SETTINGS_VERSION){
  localStorage.clear();
  localStorage.setItem('settingsVersion', SETTINGS_VERSION);
}
// --- UI mode (Video / Live) ----------------------------------------------------------------
// ONE class on <html> drives the split: mode-video hides every .lonly element, mode-live every
// .vonly one (see the CSS). The head script paints it before first paint; this keeps it in sync,
// persists the choice, and re-writes the key after a settings-version wipe above. Anything with
// neither class (model group, Speed, Image scale, RTX, Sharpen) is shared by both modes.
let uiMode = document.documentElement.classList.contains('mode-live') ? 'live' : 'video';
let lvReady = false;   // main's lv-ready answer; false = show the unavailable note in Live mode
function lvUnavailUi(){ $('lvunavail').style.display = (uiMode === 'live' && !lvReady) ? '' : 'none'; }
function setMode(m){
  uiMode = (m === 'live') ? 'live' : 'video';
  document.documentElement.className = 'mode-' + uiMode;
  $('modevideo').classList.toggle('on', uiMode === 'video');
  $('modelive').classList.toggle('on', uiMode === 'live');
  try { localStorage.setItem('uiMode', uiMode); } catch {}
  lvUnavailUi();
  // the numbered panels follow the processing order, which differs in one place: live runs RTX HDR
  // on each captured frame right after DLSS 5, before the smoothing; a file render runs it last
  if(uiMode === 'live') $('nrpanel').after($('hdrpanel')); else $('sharpenpanel').after($('hdrpanel'));
  applyOrder();   // the Order checkbox (file renders only)
}
// Switching mid-job would hide the running thing, so both buttons grey out while a live session
// or a file render is going (called from lvUi and from the run start/end enable sweeps).
function modeBtnUi(busy){
  $('modevideo').disabled = !!busy; $('modelive').disabled = !!busy;
  $('modevideo').title = busy ? 'stop the current session first' : 'Smooth a video file and save the result';
  $('modelive').title = busy ? 'stop the current session first' : 'Smooth any window on screen in real time';
}
// the Size hint under Upscale reads differently per mode (syncUpscale is defined further down,
// so it runs on the click, not at boot where the boot-time syncUpscale call covers it)
$('modevideo').onclick = () => { setMode('video'); try { syncUpscale(); } catch {} };
$('modelive').onclick = () => { setMode('live'); try { syncUpscale(); } catch {} };
setMode(uiMode);

function log(t){ $('log').textContent += t; $('log').scrollTop = $('log').scrollHeight; }
// Reveal the operational UI (controls, preview, progress, log). The bundled example clip shows it
// from launch WITH the welcome/drop hint still visible; loading a real video (or an RTX install)
// hides the welcome panel too.
function showWorkspace(keepWelcome){
  if(!keepWelcome) $('welcome').style.display = 'none';
  $('workspace').style.display = '';
}
// Repaint the frames/speed/ETA line. Called on every PROGRESS frame and on each timer tick; the
// tick recomputes the remaining seconds from the anchor so the ETA counts down between frames.
function paint(){
  if(!procStart) return;                              // still warming up: keep the warmup message
  let speed = '   ·   measuring speed...';
  if(lastFps > 0){
    const remain = Math.max(0, etaAnchor - (Date.now()-etaAnchorTime)/1000);
    speed = '   ·   '+Math.round(lastFps)+' fps   ·   ETA '+fmtEta(remain)+'  ('+fmtFinish(remain)+')';
  }
  // Projected final size (engine SIZE lines: bytes so far, linearly extrapolated to 100%), so a
  // long render reveals its disk cost in the first minutes instead of at the end.
  const size = projBytes > 0 ? '   ·   ~'+fmtBytes(projBytes) : '';
  $('frames').textContent = 'Frames: '+Math.round(curDone*fpp)+' / '+totalGen+speed+size;
}
function fmtBytes(b){ return b >= 1e9 ? (b/1e9).toFixed(b >= 1e10 ? 0 : 1)+' GB' : Math.round(b/1e6)+' MB'; }
// Interpolation is ON when ANY model is ticked. The model checkboxes ARE the master toggle (there
// is no separate Interpolate box): unticking them all means "keep the source frames", which greys
// the Speed controls. Ids are inlined rather than using MODEL_BOXES() so this stays callable before
// that const is initialised.
function interpOn(){
  return ['modelgmfss','modelrife','modeldlss','modelfruc','modelnvof','modellsfg']
    .some(id => { const e = $(id); return !!e && e.checked; });
}
function srcFps(){ const [n,d]=(((info && info.fps) || '24/1')).split('/'); return (+n)/((+d)||1); }

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
function speedMode(){ return localStorage.getItem('speedMode') === 'screen' ? 'screen' : 'fps'; }
function curFps(){ const v = +localStorage.getItem('speedFps'); return v > 0 ? v : SPEED_DEF_FPS; }
// The effective target the ENGINE renders to. Every existing consumer (gridMulti, outRatio,
// fpsValid, outName, the run payload) reads this one function, so the mode change stays contained.
function targetFps(){
  if(!interpOn()) return srcFps();          // nothing ticked = output keeps the source rate
  return speedMode() === 'screen' ? screenHz : curFps();
}
function sliderMin(){ return info ? Math.max(1, Math.round(srcFps())) : 1; }        // never target below the source rate
function sliderMax(){ return info ? srcFps() * 100 : 2400; }   // hard cap at exactly 100x the EXACT source (24000/1001 -> 2397.602, kept on-grid; not a rounded 2400)
// Decimals to display: match the source's natural precision (23.976 -> 3, 24 -> 0, 29.97 -> 2), taking
// the MORE precise of source and monitor rate so screen-match values also render exactly (24 src +
// 359.99 Hz -> 2). r_frame_rate is a rational (24000/1001), so read precision off a 3-dp rounding.
function decimalsOf(x){ const s = (+x).toFixed(3).replace(/0+$/, '').replace(/\.$/, '');
  const i = s.indexOf('.'); return i < 0 ? 0 : s.length - 1 - i; }
function targetDecimals(){ return info ? Math.max(decimalsOf(srcFps()), decimalsOf(screenHz)) : 0; }
function fpsValid(t){ if(!interpOn()) return true; return t >= sliderMin() && t <= sliderMax(); }
function outRatio(){ if(!interpOn()) return 1; return targetFps()/srcFps(); }
// On-grid eligibility: if the integer target is an exact integer multiple of the (fractional) source
// fps, run the GMFSS integer --multi path (real endpoints, exact for 23.976); else resample via --fps.
// Returns the multiplier N (>=2) for the on-grid case, or 0 for the resample case. Uses the fractional
// srcFps so 48 -> N=2 on a 23.976 source (round(23.976*2)=48).
function gridMulti(){ if(!info) return 0; const s = srcFps(); const t = targetFps(); const N = Math.round(t/s);
  return (N >= 2 && Math.abs(s*N - t) < 0.02) ? N : 0; }   // on-grid when the target is an exact multiple (tolerance covers the 3-dp display rounding)
// 3-decimal precision, so an exact multiple of a fractional source rate is expressible
// (47.952 = 2x of 23.976 stays on the integer --multi path).
// The top clamps to 100x the loaded source (sliderMax, floored to 3 dp so 9999 lands exactly on-grid:
// 2397.602 on a 23.976 source), 2400 with no video.
function setFps(v){ if(isNaN(v)) return;
  const top = Math.floor(sliderMax() * 1000) / 1000;
  localStorage.setItem('speedFps', String(Math.min(top, Math.max(1, Math.round(v * 1000) / 1000))));
  syncTargetUI(); refresh(); try{ lvSendOpts(); lvTargetChanged(); }catch{} }
function setSpeedMode(m){ localStorage.setItem('speedMode', m);
  syncTargetUI(); refresh(); try{ lvSendOpts(); lvTargetChanged(); }catch{} }
// Ratio text for the current target: the exact integer when on-grid (2, 3, ...), else
// the fractional ratio (e.g. 2.08 for a 50 fps target on a 24 source, which the engine resamples).
function ratioText(){ const gm = gridMulti();
  return gm ? String(gm) : (targetFps()/srcFps()).toFixed(2).replace(/0$/, ''); }
// Show ONLY the active mode's input, filled from that mode's own value, and grey everything while a
// run owns the controls or no model is ticked. Skip rewriting an input while it is focused, so a
// two-keystroke entry ("6" -> "60") is not reset mid-type.
function syncTargetUI(){
  const m = speedMode(), on = interpOn(), dis = !on || goMode === 'running';
  for(const [id, mode] of [['smfps','fps'], ['smscreen','screen']]){
    const b = $(id); if(!b) continue;
    b.classList.toggle('on', m === mode); b.disabled = !on;
  }
  const show = (id, vis) => { const e = $(id); if(e) e.style.display = vis ? '' : 'none'; };
  show('fpswrap', m === 'fps'); show('screenwrap', m === 'screen');
  ['fpsin','fpsup','fpsdn','fpsround'].forEach(id => { const e = $(id); if(e) e.disabled = dis; });
  // Nothing ticked = no rate change to describe: the fps box empties.
  if($('fpsin') && document.activeElement !== $('fpsin')) $('fpsin').value = on ? curFps() : '';
  // The round button: only for a resampled target, snaps it to the closest whole multiple of the
  // EXACT source rate (14.97x of 23.976 -> 15x = 359.64), so nobody types 3 decimals to get on-grid.
  const rn = roundMulti();
  show('fpsround', m === 'fps' && on && !isExample && rn > 0);
  if(rn && $('fpsround')) $('fpsround').title = 'Round to ' + rn + '× of the source (' + roundFps(rn)
    + ' fps): the closest whole multiple, on-grid real frames';
}
// The closest whole multiple (>= 2) for a resampled target, else 0 (already on-grid / no video).
function roundMulti(){ if(!info || gridMulti()) return 0;
  const N = Math.max(2, Math.round(targetFps() / srcFps()));
  return fpsValid(roundFps(N)) ? N : 0; }
function roundFps(N){ return Math.round(srcFps() * N * 1000) / 1000; }   // setFps's 3-dp rounding
$('fpsround').onclick = () => { const N = roundMulti(); if(N) setFps(srcFps() * N); };
function nudgeFps(d){ setFps(curFps() + d); }
// Press-and-hold auto-repeat: one step on press, then repeat every 60ms after a 350ms hold. Each step
// re-reads the current target, so holding + walks the value up (and stops at the clamp) instead of once.
function holdRepeat(btn, fn){ let to, iv;
  const stop = () => { clearTimeout(to); clearInterval(iv); };
  const step = () => { if(btn.disabled){ stop(); return; } fn(); };
  btn.addEventListener('mousedown', e => { if(btn.disabled) return; e.preventDefault(); step();
    to = setTimeout(() => { iv = setInterval(step, 60); }, 350); });
  ['mouseup', 'mouseleave'].forEach(ev => btn.addEventListener(ev, stop));
}
// The steppers walk the persisted value; neither reads the loaded video.
holdRepeat($('fpsup'),  () => setFps(curFps() + 1));
holdRepeat($('fpsdn'),  () => setFps(curFps() - 1));
$('speedseg').onclick = e => { const b = e.target.closest('button'); if(!b || b.disabled) return;
  setSpeedMode(b.id === 'smscreen' ? 'screen' : 'fps'); };
// Interp-off runs are named for the pass that actually runs (priority: restore > DLSS 5 > sharpen > HDR;
// an upscale keeps its _<h>p tag instead, as before). The '_unchanged' fallback covers the
// nothing-enabled combo, which is only ever a display placeholder: startRun refuses it with the
// "Nothing to do" alert, so no file by that name is produced - the tag just tells the user the
// current settings would change nothing, instead of claiming a sharpen that will not happen.
function passTag(){
  if(restoreOn()) return '_restored';
  if(nrOn()) return '_dlss5';
  if($('sharpen').checked && (parseFloat($('sharpval').value) || 0) > 0) return '_sharpened';
  if(hdrOn()) return '_hdr';
  return '_unchanged';
}
function outName(){ if(!input) return ''; const i=input.lastIndexOf('.'); const b=i<0?input:input.slice(0,i);
  const d = upDims();                                   // upscale target ({w,h} or null); tag the new height
  const res = d ? '_' + d.h + 'p' : '';
  const ext = (info && info.needMkv) ? '.mkv' : '.mp4';   // subtitles/extra tracks ride in mkv
  return interpOn() ? (b + '_' + Math.round(targetFps()) + 'fps' + res + ext)
                    : (b + (d ? res : passTag()) + ext); }
// A custom name that STILL CARRIES an auto tag (_<n>fps or _<n>p) stays in sync with the settings:
// the user's folder, base words and extension are preserved, and the _<n>fps / _<n>p suffix is
// regenerated from the current fps + upscale - so changing the fps updates the fps tag, and enabling
// or changing the upscale resolution updates (or appends) the resolution tag, layered on top of the
// user's edit. A name the user renamed with neither tag left is treated as fully custom and never rewritten.
function retagCustom(name){
  if(!/_\d+fps(?=[._]|$)|_\d+p(?=[._]|$)/.test(name)) return name;
  const slash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'));
  const dir = name.slice(0, slash + 1);                 // '' when there is no directory part
  const rest = name.slice(slash + 1);
  const dot = rest.lastIndexOf('.');
  const ext = dot >= 1 ? rest.slice(dot) : '';          // keep the user's chosen extension
  let base = dot >= 1 ? rest.slice(0, dot) : rest;
  base = base.replace(/_\d+fps/g, '').replace(/_\d+p(?=_|$)/g, '')
             .replace(/_sharpened|_restored|_dlss5|_unchanged|_hdr(?=_|$)/g, '');
  const d = upDims();
  const res = d ? '_' + d.h + 'p' : '';
  const tags = interpOn() ? ('_' + Math.round(targetFps()) + 'fps' + res) : (d ? res : passTag());
  return dir + base + tags + ext;
}
function refresh(){ if(!input) return;
  const t = targetFps();
  const N = gridMulti();
  const kind = N ? 'on-grid · real frames'                                  // exact integer multiple of the source
    : Math.round(t) <= Math.round(srcFps()) ? 'no new frames'               // degenerate: output fps = source fps
    : 'resampled';                                                          // non-integer multiple: engine uses --fps
  // The hint carries the derived other side of the relationship: the implied ratio to the source.
  // Hidden while the bundled example is loaded: its 30 fps would leak into the UI as "4x of 30"
  // and read as the app's choice (the same coupling that hides the example's metadata rows).
  const derived = ratioText() + '× of ' + (+srcFps().toFixed(targetDecimals())) + ' fps';
  $('ratehint').textContent = !interpOn() || isExample ? '' : derived + ' · ' + kind;
  syncTargetUI();
  if(!fpsValid(t)){ $('out').value = ''; $('out').placeholder = 'target fps must be ' + sliderMin() + '-' + sliderMax(); lastOut = null; return; }
  $('out').placeholder = 'output file path';
  if(customOut) customOut = retagCustom(customOut);   // re-sync the _fps/_p tags in the user's name
  lastOut = customOut || outName(); $('out').value = lastOut;
  if(goMode === 'idle') $('out').disabled = false;   // editable while idle; startRun locks it
  updateDvHpHints();   // the DV/HDR10+ "skipped for MKV" hints track the resolved output name
  scheduleCheckResume();   // the Resume detection tracks the resolved output name too
}
// Crash/exit resume detection: an interrupted render leaves its stage-1 video + sidecar next to
// the output (see the engine's resume block); when the resolved output name has one, #go becomes
// "Resume" and the render continues where it left off (the engine re-validates the settings
// signature itself, so a stale match here degrades to a fresh render with a log notice, never a
// wrong file). Debounced: refresh() fires on every slider scrub.
let resumeInfo = null, resumeTimer = null, resumeLoggedFor = null, resumeSeq = 0, resumeTipShown = false;
function scheduleCheckResume(){ clearTimeout(resumeTimer); resumeTimer = setTimeout(checkResume, 150); }
async function checkResume(){
  const seq = ++resumeSeq;
  let r = null;
  try { if(input && lastOut) r = await ipcRenderer.invoke('check-resume', lastOut); } catch {}
  if(seq !== resumeSeq) return;          // a newer check superseded this one mid-flight
  resumeInfo = r;
  if(goMode !== 'idle') return;          // never touch the button while it is the Pause toggle
  $('go').textContent = resumeInfo ? 'Resume' : 'Smooth It!';
  if(resumeInfo){
    const pct = resumeInfo.total ? Math.round(100*resumeInfo.pair/resumeInfo.total) : 0;
    $('go').title = 'Continues the interrupted render from where it left off (~'+pct+'% is already done). '
                  + 'You can pause or close the app again at any time and resume later.';
    $('status').textContent = 'Interrupted render found: click Resume to continue from ~'+pct+'%.';
    if(resumeLoggedFor !== lastOut){
      resumeLoggedFor = lastOut;
      log('>> Interrupted render found for this file; it will resume from ~'+pct+'% instead of starting over\n');
    }
  } else {
    if(resumeLoggedFor === lastOut) resumeLoggedFor = null;   // gone (e.g. cancelled): a future interruption logs anew
    $('go').title = 'Starts the render. You can pause or even close the app mid-render: it resumes where it left off.';
  }
}
// Editing the output path inline: a non-empty value becomes the custom output (kept across setting
// changes, like the Change... dialog); clearing it on blur reverts to the auto-generated name.
$('out').oninput = () => { const v = $('out').value.trim(); customOut = v || null; lastOut = v || outName(); };
$('out').onchange = () => { if(!$('out').value.trim()){ customOut = null; refresh(); } };

async function loadVideo(f, asExample){
  if(!f) return false;
  if(asExample && input) return false;   // a real video won the boot race; keep it
  isExample = !!asExample;
  input = f; customOut = null;
  $('path').textContent = asExample ? 'Example clip (Big Buck Bunny) - the settings below preview on it until you pick a video' : f;
  if(!asExample) try { localStorage.setItem('dir', path.dirname(f)); } catch {}
  const p = await ipcRenderer.invoke('probe', f);
  // A modal alert would stall an unattended batch, so mid-batch probe failures go to the log.
  if(p.error){ if(batchAuto) log('>> Could not read '+f+': '+p.error+'\n'); else alert('Could not read video: '+p.error); return false; }
  const streams = p.streams || [];
  const s = streams.find(x => x.codec_type === 'video') || streams[0] || {};
  // Track passthrough: subtitles or mp4-incompatible audio push the output container to .mkv
  // (mirrors the engine's own MP4_AUDIO_OK rule, so the name shown matches what it writes).
  const MP4A = ['aac','ac3','eac3','mp3','alac','opus','flac'];
  const auds = streams.filter(x => x.codec_type === 'audio');
  const subs = streams.filter(x => x.codec_type === 'subtitle');
  info = { fps: s.r_frame_rate, w: s.width, h: s.height, nb: +(s.nb_frames||0),
           dur: +((p.format||{}).duration||0), codec: s.codec_name,
           transfer: s.color_transfer || '',
           nAud: auds.length, nSub: subs.length,
           needMkv: subs.length > 0 || auds.some(a => !MP4A.includes((a.codec_name||'').toLowerCase())) };
  info.srcHdr = info.transfer === 'smpte2084' || info.transfer === 'arib-std-b67';
  // TrueHDR converts SDR only; for an HDR source the engine carries the HDR through untouched.
  $('rtxhdr').disabled = info.srcHdr;
  $('rtxhdrhint').textContent = info.srcHdr ? '(source is already HDR, carried through as-is)'
                                            : '(convert SDR video to HDR)';
  showWorkspace(asExample);   // the example keeps the welcome/drop hint on screen
  // The bundled example autoloads to keep the settings surface + preview alive, but its probe
  // data (1280x720 / 30fps / h264) is the sample's, not the user's - showing it reads as "a
  // video is loaded". Hide the metadata rows while the example is up (same gate as #outrow).
  $('info').style.display = asExample ? 'none' : 'block';
  $('res').textContent = s.width+' x '+s.height;
  $('fps').textContent = srcFps().toFixed(decimalsOf(srcFps()));   // natural precision: 24 -> "24", 23.976 -> "23.976"
  $('fps').title = (info.fps ? info.fps + ' = ' : '') + (+srcFps().toFixed(6)) + ' fps (shown rounded)';   // reveal the exact rational, e.g. 24000/1001 = 23.976024
  const d=info.dur|0; $('dur').textContent = (d/3600|0)+':'+String((d%3600/60|0)).padStart(2,'0')+':'+String(d%60).padStart(2,'0');
  $('codec').textContent = (s.codec_name || '?')
    + (info.nAud > 1 || info.nSub ? '  ·  ' + info.nAud + ' audio · ' + info.nSub
       + ' subtitle' + (info.nSub === 1 ? '' : 's') + ' (kept' + (info.needMkv ? ', MKV output)' : ')') : '');
  $('go').disabled = asExample; $('changeout').disabled = asExample; $('open').disabled = true; $('play').disabled = true;
  $('outrow').style.display = asExample ? 'none' : '';   // no output path for the example
  $('live').style.display = 'none';   // stale live thumbnail belongs to the previous video
  $('status').textContent = asExample
    ? 'This is the bundled example. Select your own video, then click Smooth It!'
    : 'Ready. Pick a target fps, then click Smooth It!';
  applyScreenFps();      // re-asserts the interp / screen-rate state and recomputes the output name
  syncUpscale();         // recompute the upscale target + output dims for this video's resolution
  // The preview cold-starts Python + torch + a CUDA context (several seconds on the first video of a
  // session), which SATURATES the GPU and stalls Chromium's compositor - so if it spawns before the
  // file info has actually painted, the info (already probed in ~100ms, DOM updated synchronously
  // above) can't reach the screen until that Python eases, which is the "video info loads slowly"
  // bug. So: reveal the menu + file info now (showWorkspace above, synchronous), paint a loading
  // wheel on the preview panes (showPreviewInitializing, synchronous), then wait for a REAL composited
  // frame (two rAFs = one full paint) so the info is guaranteed on screen BEFORE the heavy preview
  // Python starts. Skipped when a batch auto-advances (no per-item preview).
  if(!batchAuto){
    showPreviewInitializing();
    requestAnimationFrame(() => requestAnimationFrame(openPreviewForVideo));
  }
  return true;
}
// One or more inputs: the first loads now, the rest queue for the same settings back to back.
function queueVideos(paths){
  const list = (paths || []).filter(Boolean);
  if(!list.length) return;
  batch = list.slice(1); batchTotal = list.length; batchFailed = [];
  if(batchTotal > 1) log('>> Queued '+batchTotal+' files (same settings, default output names)\n');
  loadVideo(list[0]);
}
$('pick').onclick = async () => { queueVideos(await ipcRenderer.invoke('pick-video', localStorage.getItem('dir') || undefined)); };
// The fps box holds the user's ABSOLUTE target. Commit on change only, so a partial entry on the way
// to a bigger number is never clamped mid-type. It is independent of any loaded video.
$('fpsin').onchange = () => { const v = parseFloat($('fpsin').value);
  if(isNaN(v)){ $('fpsin').value = curFps(); return; } setFps(v); $('fpsin').value = curFps(); };

// Sharpen (CAS): a checkbox enables Contrast Adaptive Sharpening and reveals a 0..1 strength slider;
// unchecking hides the slider and disables it. Default is on at full strength (1.0). The strength is
// forwarded to the engine as --sharpen; both the on/off state and the value persist between sessions.
function syncSharpen(){
  const on = $('sharpen').checked;
  $('sharpwrap').style.display = on ? 'flex' : 'none';
  $('sharpnum').textContent = (+$('sharpval').value).toFixed(2);
  syncPreview();        // show/hide the preview panel as sharpen turns on/off
}
const savedSharpOn = localStorage.getItem('sharpenOn');
if(savedSharpOn !== null) $('sharpen').checked = savedSharpOn === '1';
const savedSharpVal = localStorage.getItem('sharpenVal');
if(savedSharpVal !== null) $('sharpval').value = savedSharpVal;
syncSharpen();
$('sharpen').onchange = () => { localStorage.setItem('sharpenOn', $('sharpen').checked ? '1' : '0'); syncSharpen(); try{ lvSendOpts(); }catch{} };
$('sharpval').oninput = () => { localStorage.setItem('sharpenVal', $('sharpval').value); syncSharpen(); try{ lvSendOpts(); }catch{} };

// Restore: engine --restore. Real-ESRGAN's anime-video model cleans compression noise and
// redraws linework on every output frame (before the upscale); fine texture can flatten - it
// is a generative repaint, not a filter. Off by default: it changes the look, and every
// emitted frame pays a second model pass.
function restoreOn(){ return $('restore').checked; }
// Restore ALWAYS starts off: an expensive, look-altering pass you opt
// into per session, so its state is deliberately NOT persisted; old key retired.
localStorage.removeItem('restoreOn');

// Order: engine --nvidia-order (file renders). NVIDIA's game order runs Restore and the upscale on
// each source frame first, then DLSS 5 and the interpolation at the output size (slower: the model
// works on the bigger frames); off = the faster default order. The numbered panels follow the
// order that is picked, and the pass hints switch with it (body.nvorder).
function nvOrderOn(){ return uiMode === 'video' && $('nvorder').checked; }
function applyOrder(){
  const nvo = nvOrderOn();
  document.body.classList.toggle('nvorder', nvo);
  if(nvo){ $('nrpanel').before($('restorepanel')); $('nrpanel').before($('uppanel')); }
  else { $('interppanel').after($('restorepanel')); $('restorepanel').after($('uppanel')); }
}
if(localStorage.getItem('nvorderOn') === '1') $('nvorder').checked = true;   // default OFF
$('nvorder').onchange = () => { localStorage.setItem('nvorderOn', $('nvorder').checked ? '1' : '0'); applyOrder(); refreshPreviewIfOpen(); };
applyOrder();

// "Match screen" is now one of the three Speed modes rather than an override checkbox, so there is
// nothing to re-assert after a video loads: the mode and its value are the user's, independent of
// the source. applyScreenFps is kept as the name every caller already uses (video load, run finish,
// model change) and simply re-renders the Speed row.
let screenHz = 60;
function applyScreenFps(){ syncTargetUI(); refresh(); }
// One-time migration off the old scheme (fpsTarget = an ABSOLUTE target derived from whichever video
// was loaded, plus a separate screenfps checkbox). Translating beats bumping SETTINGS_VERSION, which
// would wipe every unrelated preference.
if(localStorage.getItem('speedMode') === null){
  // Only an explicit screen-match choice carries over as a MODE. The old fpsTarget was DERIVED from
  // whichever video was loaded (2x of the 30 fps sample clip = 60), so promoting it to an absolute
  // target would perpetuate the exact coupling this replaces. Keep it only as the fps box's seed.
  const old = +localStorage.getItem('fpsTarget');
  if(old > 0 && localStorage.getItem('speedFps') === null) localStorage.setItem('speedFps', String(Math.round(old)));
  localStorage.setItem('speedMode', localStorage.getItem('screenfps') === '1' ? 'screen' : 'fps');
}
// A persisted 'mult' choice (the old Multiplier mode) becomes the fps default
// (translating beats a SETTINGS_VERSION bump, which would wipe every unrelated preference).
if(localStorage.getItem('speedMode') === 'mult') localStorage.setItem('speedMode', 'fps');
localStorage.removeItem('speedMult');
localStorage.removeItem('screenfps'); localStorage.removeItem('fpsTarget');
ipcRenderer.invoke('refresh-rate').then(hz => {
  screenHz = (+hz) || 60; $('screenhz').textContent = screenHz;
  if(info) applyScreenFps();   // monitor rate can add decimals (359.99 -> 2), so reformat the target box now that it's known
  try{ lvSendOpts(); lvTargetChanged(); }catch{}  // Live's inherited target may just have changed
});

// Upscale-to-resolution: a target height chosen from the dropdown (or the Custom slider) drives an
// arbitrary upscale factor (targetHeight / sourceHeight), keeping the source aspect ratio. RTX VSR
// (the #rtxvsr toggle, once its runtime is installed) does the upscale with NVIDIA AI; with it off or
// absent the engine uses a bicubic resize. Off by default; the choice persists. Independent of
// interpolation, so it also applies in sharpen-only mode. There is no integer-scale restriction (RTX
// VSR was probed clean to 16K), but the output is bounded by two real limits: the encoders top out at
// 16K (NVENC caps either dimension at 8192px, verified on the RTX 5090; past that the ENGINE switches
// to a CPU encoder automatically - AV1 to ~12K, H.266/VVC to 16K, verified 15360x8640 - so the GUI
// wall is VVC's), and the engine clamps the upscale factor to 16x. The custom-height slider and
// upDims() honour both, so the long side never exceeds 15360px and the factor never exceeds 16x.
const ENC_MAX = 15360;                           // 16K long side: the last encoder standing (libvvenc)
const NVENC_MAX = 8192;                          // above this the engine encodes on the CPU (slower)
const MAX_FACTOR = 16;                           // engine UPSCALE_F clamp (gmfss_interp.py)
let screenW = 0, screenH = 0, rtxReady = { vsr:false, hdr:false, bridge:false };
function upTargetH(){                            // chosen output height (0 = off)
  const v = $('upres').value;
  if(v === '0') return 0;
  if(v === 'screen') return screenH || 0;
  if(v === 'custom') return +$('upcustom').value || 0;
  return +v || 0;
}
function upDims(){                               // {w,h} of the resized output (encoder/factor clamped), or null
  if(!info || !info.h || !info.w) return null;
  let th = upTargetH();
  if(!th || th === info.h) return null;         // off, or a 1:1 target = nothing to resize
  th = Math.min(th, info.h * MAX_FACTOR, ENC_MAX);   // 16x engine factor cap, then the encoder long-side cap
  th = Math.max(th, Math.ceil(info.h / MAX_FACTOR / 2) * 2);   // and the 16x floor going down (engine clamp)
  let h = th - (th % 2);
  let w = Math.round(info.w * h / info.h); w = w - (w % 2);
  if(w > ENC_MAX){                              // wide aspect: width is the binding side, fit it to 8192px
    h = Math.floor(ENC_MAX * info.h / info.w / 2) * 2;
    w = Math.round(info.w * h / info.h); w = w - (w % 2);
  }
  return { w, h };
}
function upFactor(){ const d = upDims(); return d ? d.h / info.h : 0; }   // arbitrary factor (0 = off)
function syncUpscale(){
  $('upcustomwrap').style.display = $('upres').value === 'custom' ? 'inline-flex' : 'none';
  if(info && info.w && info.h){                  // bound the custom slider to what this source + encoder can reach
    const m = Math.max(720, Math.min(Math.floor(ENC_MAX * Math.min(1, info.h / info.w) / 2) * 2, info.h * MAX_FACTOR));
    if(+$('upcustom').max !== m){ $('upcustom').max = m; if(+$('upcustom').value > m) $('upcustom').value = m; }
  }
  $('upcustomnum').textContent = (+$('upcustom').value) + 'p';
  let txt = '';
  if(uiMode === 'live'){
    // Live has no loaded file: the height itself is the setting (the internal render size)
    const th = upTargetH();
    txt = th ? '→ rendered at ' + th + 'p first, then fitted to the window or screen' : '';
  } else if($('upres').value !== '0'){
    const d = upDims();
    // Just the resulting dimensions plus encoder-relevant warnings; no resampler names (the
    // engine always picks the best resize).
    if(d) txt = '→ ' + d.w + ' × ' + d.h
                + ((d.w > NVENC_MAX || d.h > NVENC_MAX) ? '  ·  CPU encode at this size (slower)' : '')
                + ((d.w >= ENC_MAX || d.h >= ENC_MAX) ? '  ·  at the 16K encoder cap' : '');
    else if(info) txt = 'source is already exactly at that height, 1:1, nothing to resize';
    else txt = 'load a video to preview the result';
  }
  $('upinfo').textContent = txt;
  refresh();             // the target height is tagged into the output filename
}
function setScreenOptLabel(){ const o = [...$('upres').options].find(o => o.value === 'screen');
  if(o) o.textContent = 'Match screen' + (screenW && screenH ? ' (' + screenW + '×' + screenH + ')' : ''); }
// NOTE: the element id is outcodec, NOT codec - id "codec" is taken by the source-codec info span.
// Per-choice guidance so the trade-off is clear without leaving the dropdown. Size claims are
// from a real A/B at the engine's verified visually-lossless settings (the quality-first
// tuning: on the 1080p sample HEVC CQ17-p7-ladder 2.25 MB / AV1 CQ22 2.17 MB / VVC QP17 0.98 MB).
const CODEC_HINTS = {
  hevc: 'safest choice: TVs, phones, editors and players all take it; audio is copied',
  av1:  'encodes 1.4 to 1.6x faster than HEVC on the same GPU at higher fidelity; similar size on clean anime, up to 2.5x larger on fast noisy content; royalty-free, plays in every modern browser, but devices from before ~2020 may not decode it',
  vvc:  'smallest files of the three (roughly half of HEVC), but the CPU encode is slow and almost no player supports H.266 yet (archival)',
};
function syncCodec(){ $('codechint').textContent = CODEC_HINTS[$('outcodec').value] || ''; }
const savedCodec = localStorage.getItem('codec');
if(savedCodec && [...$('outcodec').options].some(o => o.value === savedCodec)) $('outcodec').value = savedCodec;
$('outcodec').onchange = () => { localStorage.setItem('codec', $('outcodec').value); syncCodec(); };
syncCodec();
let savedUpres = localStorage.getItem('upres');
if(savedUpres === '540'){ savedUpres = '480'; localStorage.setItem('upres', savedUpres); }   // 540p retired (no 540p anime exists), 480p took its slot
if(savedUpres && [...$('upres').options].some(o => o.value === savedUpres)) $('upres').value = savedUpres;
const savedUpcustom = localStorage.getItem('upcustom'); if(savedUpcustom) $('upcustom').value = savedUpcustom;
$('upres').onchange = () => { localStorage.setItem('upres', $('upres').value); syncUpscale(); refreshPreviewIfOpen(); try{ lvModelUi(); lvSendOpts(); }catch{} };
$('upcustom').oninput = () => { localStorage.setItem('upcustom', $('upcustom').value); syncUpscale(); try{ lvModelUi(); lvSendOpts(); }catch{} };
$('upcustom').addEventListener('change', refreshPreviewIfOpen);
ipcRenderer.invoke('screen-size').then(s => { screenW = (s&&s.width)||0; screenH = (s&&s.height)||0; setScreenOptLabel(); syncUpscale(); });

// NVIDIA RTX (opt-in): real RTX Video Super Resolution + RTX HDR via the engine/rtxvideo CUDA bridge
// + NVIDIA's feature DLLs. Both OFF by default. The feature DLLs are non-redistributable, so the app
// can't bundle them; instead you pick the downloaded RTX Video SDK .zip and the app copies the two
// DLLs into engine/rtxvideo for you. Readiness = the bridge + the feature's model DLL present.
function syncRtx(){
  const on = $('rtxvsr').checked || $('rtxhdr').checked;
  // The setup box appears only when a requested feature's runtime is missing (or while installing);
  // checkRtxReady() collapses it once the files are correctly installed so READY takes no screen space.
  // With no feature requested there is nothing to set up, so hide it outright.
  if(on) checkRtxReady();
  else $('rtxsetup').style.display = 'none';
}
async function checkRtxReady(){
  $('rtxstatus').textContent = 'checking...'; $('rtxstatus').style.color = 'var(--sub)';
  try { rtxReady = await ipcRenderer.invoke('rtx-ready'); } catch { rtxReady = { vsr:false, hdr:false, bridge:false }; }
  const needVsr = $('rtxvsr').checked, needHdr = $('rtxhdr').checked;
  const ok = (!needVsr || rtxReady.vsr) && (!needHdr || rtxReady.hdr);
  const lines = [];
  if(needVsr) lines.push((rtxReady.vsr ? '✓' : '✗') + ' RTX Video Super Resolution');
  if(needHdr) lines.push((rtxReady.hdr ? '✓' : '✗') + ' RTX HDR');
  if(ok){
    $('rtxstatus').textContent = '✓ READY'; $('rtxstatus').style.color = 'var(--green)';
    $('rtxsteps').innerHTML = lines.join('<br>') + '<br>The RTX Video runtime is installed.';
  } else {
    $('rtxstatus').textContent = '✗ NOT READY'; $('rtxstatus').style.color = '#e85c5c';
    const detail = rtxReady.bridge === false
      ? 'The RTX bridge (rtxvideo_cuda.dll) is missing from engine/rtxvideo. Rebuild it (see DEVELOPMENT.md, "Building the native bridges").'
      : 'Click <b>Get from NVIDIA</b>, then <b>Choose .zip&hellip;</b> and pick the downloaded RTX_Video_SDK&hellip;.zip. It installs automatically.';
    $('rtxsteps').innerHTML = lines.join('<br>') + '<br>' + detail;
  }
  $('rtxbtns').style.display = ok ? 'none' : '';   // hide Get / Choose once the runtime is ready
  // One setup box serves both RTX features, which now live in different panels: park it under
  // the first checkbox that still needs it (VSR in Upscale, else HDR in HDR).
  const host = ((needVsr && !rtxReady.vsr) || !needHdr) ? $('uppanel') : $('hdrpanel');
  if($('rtxsetup').parentElement !== host) host.appendChild($('rtxsetup'));
  // Collapse the whole setup box once the requested runtime is installed, so a READY state takes no
  // screen space; it reappears only while something needed is still missing.
  $('rtxsetup').style.display = ok ? 'none' : 'block';
  syncUpscale();         // the upscale backend label depends on VSR readiness
  refreshPreviewIfOpen(); // re-render the open preview now that HDR readiness is known
}
async function doInstall(source){
  $('rtxsetup').style.display = 'block';   // show install progress (checkRtxReady collapses it again on success)
  $('rtxstatus').textContent = 'installing...'; $('rtxstatus').style.color = 'var(--sub)';
  let r = {}; try { r = await ipcRenderer.invoke('rtx-install', source); } catch(e){ r = { ok:false, error:String(e) }; }
  if(!r.ok) $('rtxsteps').innerHTML = '✗ ' + (r.error || 'install failed')
      + '<br>Pick the downloaded RTX_Video_SDK&hellip;.zip again with <b>Choose .zip&hellip;</b>.';
  await checkRtxReady();
  if(!r.ok) $('rtxsetup').style.display = 'block';   // keep an install error visible even if no toggle needs the feature
}
$('rtxget').onclick = () => ipcRenderer.invoke('rtx-open-download');
$('rtxbrowsezip').onclick = async () => { const p = await ipcRenderer.invoke('rtx-choose','zip'); if(p) doInstall(p); };  // selecting a .zip auto-installs
localStorage.removeItem('supersampleOn');   // retired: supersample removed (measured imperceptible)
localStorage.removeItem('encspeed');   // retired: no Encoder speed selector (every render uses the Quality encoder)
if(localStorage.getItem('rtxvsrOn') === '1') $('rtxvsr').checked = true;   // default OFF
if(localStorage.getItem('rtxhdrOn') === '1') $('rtxhdr').checked = true;   // default OFF
$('rtxvsr').onchange = () => { localStorage.setItem('rtxvsrOn', $('rtxvsr').checked ? '1' : '0'); syncRtx(); syncUpscale(); refreshPreviewIfOpen(); try{ lvModelUi(); lvSendOpts(); }catch{} };
$('rtxhdr').onchange = () => { localStorage.setItem('rtxhdrOn', $('rtxhdr').checked ? '1' : '0'); syncRtx(); refreshPreviewIfOpen(); try{ lvModelUi(); lvSendOpts(); }catch{} };
syncRtx();
// Populate readiness on load so the upscale backend label is right even before the RTX panel opens.
ipcRenderer.invoke('rtx-ready').then(r => { if(r) rtxReady = r; syncUpscale(); syncPreview();
  // Live's VSR default depends on rtxReady, which just arrived: re-send + refresh the fill hint
  try{ lvModelUi(); lvSendOpts(); }catch{} });

// NVIDIA DLSS 5 (Neural Rendering; opt-in; both modes: renders run it per output frame, Live
// once per captured frame before the smoothing, lvSendOpts carries it):
// a DLAA-class per-frame pass at the output resolution, hosted by engine/dlssnr/dlssnr.exe (ships)
// plus the NR runtime nvngx_dlssnr.dll, which the app never ships (NVIDIA offers no public download,
// the only NVIDIA copy is inside NBA 2K27): one click downloads the community build every DLSS 5 tool
// uses (rhi-repo on GitHub, checksum-verified twice in main), or the user drops a copy in, like the
// RTX Video / NvOFFRUC DLLs.
// Readiness = host + runtime present (dlssnr-ready). The two sliders are NVIDIA's global developer
// controls, Structure Intensity and Tone Intensity (0..2, default 1.00); the pass always runs at
// DLAA quality (no quality selector, by decision).
let dlssnrReady = { ready:false };
function nrOn(){ return $('dlssnr').checked && !!dlssnrReady.ready; }
function nrStructure(){ return parseFloat($('nrstructure').value) || 0; }
function nrTone(){ return parseFloat($('nrtone').value) || 0; }
function nrStyle(){ const b = $('nrstyleseg').querySelector('button.on'); return b ? parseInt(b.dataset.style, 10) : 1; }   // DLSSNR.Style 0/1/2
function nrMaskOn(){ return nrOn() && $('nrmask').checked; }   // preview-only heat map of the DLSS 5 change
function syncDlssnr(){
  const on = $('dlssnr').checked;
  $('nrwrap').style.display = on ? 'inline-flex' : 'none';
  if(on) checkDlssnrReady();
  else $('dlssnrsetup').style.display = 'none';
}
async function checkDlssnrReady(){
  $('dlssnrstatus').textContent = 'checking...'; $('dlssnrstatus').style.color = 'var(--sub)';
  try { dlssnrReady = await ipcRenderer.invoke('dlssnr-ready'); } catch { dlssnrReady = { ready:false }; }
  const r = dlssnrReady;
  const lines = [(r.host ? '✓' : '✗') + ' DLSS 5 host (dlssnr.exe + nvngx.dll, bundled)',
                 (r.runtime ? '✓' : '✗') + ' ' + (r.file || 'nvngx_dlssnr.dll') + ' (the Neural Rendering runtime, not included)',
                 (r.sr ? '✓' : '○') + ' ' + (r.srFile || 'nvngx_dlss.dll') + ' (DLSS SR runtime from the same pack, optional)'];
  if(r.ready){
    $('dlssnrstatus').textContent = '✓ READY'; $('dlssnrstatus').style.color = 'var(--green)';
    $('dlssnrsteps').innerHTML = lines.join('<br>') + '<br>The DLSS 5 runtime is installed.';
  } else {
    $('dlssnrstatus').textContent = '✗ NOT READY'; $('dlssnrstatus').style.color = '#e85c5c';
    const detail = r.host === false
      ? 'The DLSS 5 host (dlssnr.exe / nvngx.dll) is missing from engine/dlssnr. Rebuild it (see DEVELOPMENT.md, "Building the native bridges").'
      : 'NVIDIA does not publish <code>nvngx_dlssnr.dll</code> (no SDK, no driver copy, no NVIDIA App override); it only ships inside NBA 2K27. '
        + '<b>Download the DLSS 5 runtime</b> fetches the community build every DLSS 5 tool uses (' + ((r.download && r.download.mb) || 111) + ' MB zip from '
        + '<a href="#" onclick="ipcRenderer.invoke(\'dlssnr-open-download\');return false;">rhi-repo on GitHub</a>, RTX 40 + 50), verifies its checksum and installs it. '
        + 'Or, if you already have the file, drop it on this window or click <b>Choose nvngx_dlssnr.dll&hellip;</b>.';
    $('dlssnrsteps').innerHTML = lines.join('<br>') + '<br>' + detail;
  }
  $('dlssnrbtns').style.display = r.ready ? 'none' : '';   // hide the buttons once the runtime is in place
  $('dlssnrsetup').style.display = r.ready ? 'none' : 'block';
  refreshPreviewIfOpen();   // the processed pane depends on readiness
  refresh();                // so does the interp-off output name (_dlss5 needs nrOn())
  // the Live hint and the live options read nrOn() = ticked AND ready: a tick (or the boot
  // restore) made before this async answer lands computed them as off, so re-run them now
  // (otherwise the fit hint stays stale after the tick)
  try{ lvModelUi(); lvSendOpts(); }catch{}
}
function dlssnrInstallNote(r){   // one line about what got installed (known build or not)
  return r.known ? 'Installed build: ' + r.known + '.' : 'Installed an unverified build (SHA256 ' + String(r.sha256 || '').slice(0, 12) + '&hellip;), not the verified build; it may still work.';
}
async function doDlssnrInstall(source){
  $('dlssnrsetup').style.display = 'block';
  $('dlssnrstatus').textContent = 'installing...'; $('dlssnrstatus').style.color = 'var(--sub)';
  let r = {}; try { r = await ipcRenderer.invoke('dlssnr-install', source); } catch(e){ r = { ok:false, error:String(e) }; }
  if(!r.ok) $('dlssnrsteps').innerHTML = '✗ ' + (r.error || 'install failed')
      + '<br>Pick the file again with <b>Choose nvngx_dlssnr.dll&hellip;</b>.';
  await checkDlssnrReady();
  if(!r.ok) $('dlssnrsetup').style.display = 'block';   // keep an install error visible
  else if(!r.known){ $('dlssnrsteps').innerHTML += '<br>' + dlssnrInstallNote(r); $('dlssnrsetup').style.display = 'block'; }
}
// One-click download (main does fetch + checksum + extract + copy); progress lands in the status line.
let dlssnrDownloading = false;
ipcRenderer.on('dlssnr-progress', (_e, p) => {
  if(!dlssnrDownloading) return;
  const mb = x => (x / 1048576).toFixed(0);
  $('dlssnrstatus').textContent = p.stage === 'verify' ? 'verifying checksum...' : 'downloading... ' + mb(p.received) + ' / ' + mb(p.total) + ' MB';
});
async function doDlssnrDownload(){
  if(dlssnrDownloading) return;
  dlssnrDownloading = true; $('dlssnrdl').disabled = true; $('dlssnrbrowse').disabled = true;
  $('dlssnrsetup').style.display = 'block';
  $('dlssnrstatus').textContent = 'downloading...'; $('dlssnrstatus').style.color = 'var(--sub)';
  let r = {}; try { r = await ipcRenderer.invoke('dlssnr-download'); } catch(e){ r = { ok:false, error:String(e) }; }
  dlssnrDownloading = false; $('dlssnrdl').disabled = false; $('dlssnrbrowse').disabled = false;
  await checkDlssnrReady();
  if(!r.ok){
    $('dlssnrstatus').textContent = '✗ DOWNLOAD FAILED'; $('dlssnrstatus').style.color = '#e85c5c';
    $('dlssnrsteps').innerHTML = '✗ ' + (r.error || 'download failed') + '<br>Check the connection and try again, or get the file yourself from '
      + '<a href="#" onclick="ipcRenderer.invoke(\'dlssnr-open-download\');return false;">rhi-repo on GitHub</a> and use <b>Choose nvngx_dlssnr.dll&hellip;</b>.';
    $('dlssnrsetup').style.display = 'block';
  }
}
$('dlssnrdl').onclick = doDlssnrDownload;
$('dlssnrbrowse').onclick = async () => { const p = await ipcRenderer.invoke('dlssnr-choose'); if(p) doDlssnrInstall(p); };
if(localStorage.getItem('dlssnrOn') === '1') $('dlssnr').checked = true;   // default OFF
$('dlssnr').onchange = () => { localStorage.setItem('dlssnrOn', $('dlssnr').checked ? '1' : '0'); syncDlssnr(); refreshPreviewIfOpen(); try{ lvModelUi(); lvSendOpts(); }catch{} };
for(const [id, key] of [['nrstructure', 'nrStructure'], ['nrtone', 'nrTone']]){
  const saved = localStorage.getItem(key); if(saved !== null) $(id).value = saved;
  $(id + 'num').textContent = (+$(id).value).toFixed(2);
  $(id).oninput = () => { localStorage.setItem(key, $(id).value); $(id + 'num').textContent = (+$(id).value).toFixed(2); };
  $(id).addEventListener('change', () => { refreshPreviewIfOpen(); try{ lvSendOpts(); }catch{} });
}
{ const saved = localStorage.getItem('nrStyle');   // Style selector: persisted, default Natural (1)
  if(saved !== null) for(const b of $('nrstyleseg').querySelectorAll('button')) b.classList.toggle('on', b.dataset.style === saved); }
$('nrstyleseg').onclick = e => { const b = e.target.closest('button'); if(!b || b.disabled) return;
  for(const x of $('nrstyleseg').querySelectorAll('button')) x.classList.toggle('on', x === b);
  localStorage.setItem('nrStyle', b.dataset.style); refreshPreviewIfOpen(); try{ lvSendOpts(); }catch{} };
if(localStorage.getItem('nrMask') === '1') $('nrmask').checked = true;   // default OFF
$('nrmask').onchange = () => { localStorage.setItem('nrMask', $('nrmask').checked ? '1' : '0'); refreshPreviewIfOpen(); };
syncDlssnr();

// Interpolation model: GMFSS (default) vs "NVIDIA Smooth Motion" (NVIDIA Optical Flow FRUC). Two
// checkboxes acting as an exclusive pair (user-chosen UX): ticking one clears the other, and
// unticking the active one hands the choice back to its sibling, so exactly one model is always
// selected. The FRUC runtime (NvOFFRUC.dll) is NVIDIA proprietary + user-installed from the Optical
// Flow SDK .zip. The Smooth Motion checkbox is ALWAYS clickable (ticking it visibly clears GMFSS);
// it just reveals a one-time installer when the runtime is missing, and a render only uses FRUC once
// it is installed - otherwise it falls back to GMFSS (see modelIsFruc + the run payload gate below).
let frucReady = false, frucBridge = false, dlssReady = false, dlssMissing = [];
function modelIsFruc(){ return $('modelfruc').checked && frucReady; }
function modelIsDlss(){ return $('modeldlss').checked && dlssReady; }
// RIFE ships bundled (weights vendored in engine/rife), so it is always ready.
function modelIsRife(){ return $('modelrife').checked; }
// The DRBA sub-checkbox (default OFF) adds anime-pacing timing on top of the same RIFE weights.
function modelIsRifeDrba(){ return modelIsRife() && $('rifedrba').checked; }
// Frame Blend (engine --lsfg, flow-warp) uses the bundled RIFE weights, so it is always
// ready: nothing to install, no GPU feature check.
function modelIsLsfg(){ return $('modellsfg').checked; }
// NVIDIA Optical Flow (direct): the driver's optical-flow hardware run by
// smv-live.exe itself (engine --nvof, live backend nvof), so it is always ready like Frame Blend.
function modelIsNvof(){ return $('modelnvof').checked; }
const MODEL_BOXES = () => [$('modelgmfss'), $('modelrife'), $('modeldlss'), $('modelfruc'), $('modelnvof'), $('modellsfg')];
function frucSetupHtml(){
  const detail = !frucBridge
    ? 'The bridge (nvoffruc_bridge.dll) is missing from engine/nvoffruc. Build it once (see DEVELOPMENT.md, "Building the native bridges").'
    : 'Click <b>Get from NVIDIA</b>, then <b>Choose .zip&hellip;</b> and pick the downloaded Optical_Flow_SDK&hellip;.zip. It installs automatically. Until then GMFSS is used.';
  return "NVIDIA Smooth Motion uses NVIDIA's Optical Flow runtime (a one-time download, not bundled).<br>" + detail;
}
async function refreshFrucState(){
  try { const r = await ipcRenderer.invoke('fruc-ready'); frucReady = !!(r && r.ready); frucBridge = !!(r && r.bridge); }
  catch { frucReady = false; frucBridge = false; }
  // DLSS ("DLSS 4.5" Frame Generation) ships bundled, so not-ready means files were deleted from
  // engine/dlssg (a broken install); GPU/driver support is only knowable at render time.
  try { const d = await ipcRenderer.invoke('dlssg-ready'); dlssReady = !!(d && d.ready); dlssMissing = (d && d.missing) || []; }
  catch { dlssReady = false; dlssMissing = []; }
}
// Reflect the current model choice: show a setup hint only while the chosen model's runtime is
// missing (FRUC has a one-time DLL installer).
function syncModel(){
  localStorage.setItem('interpModel',
    !interpOn() ? 'none' :                          // nothing ticked = interpolation off, remembered
    $('modelfruc').checked ? 'fruc' :
    $('modeldlss').checked ? 'dlssg' : $('modelrife').checked ? 'rife' :
    $('modellsfg').checked ? 'lsfg' : $('modelnvof').checked ? 'nvof' : 'gmfss');
  // The DRBA sub-option only makes sense while RIFE is the chosen model (always ready: bundled).
  $('rifedrbarow').style.display = $('modelrife').checked ? '' : 'none';
  if($('modeldlss').checked && !dlssReady){
    $('dlsssteps').innerHTML = 'The bundled DLSS runtime is incomplete: <b>' + dlssMissing.join(', ')
      + '</b> missing from engine/dlssg. Reinstall the app (or rebuild per DEVELOPMENT.md). '
      + 'Until then GMFSS is used.';
    $('dlsssetup').style.display = 'block';
  } else {
    $('dlsssetup').style.display = 'none';
  }
  // DLSS-FG has no arbitrary-timestep API: it makes only the evenly spaced frames between
  // consecutive source frames, i.e. whole 2x-6x multipliers (multi-frame generation). Picking it
  // with an off-grid or out-of-range target snaps the fps box to exactly 2x the source (the
  // steppers stay usable; a bad target is blocked with guidance at Smooth It! time).
  // isExample guard: the snap must never rewrite the user's target from the sample clip's rate.
  if($('modeldlss').checked && dlssReady && info && !isExample){
    const gm = gridMulti();
    if((gm < 2 || gm > 6) && speedMode() === 'fps') setFps(srcFps() * 2);
  }
  if($('modelfruc').checked && !frucReady){
    $('frucstatus').textContent = '✗ NOT READY'; $('frucstatus').style.color = '#e85c5c';
    $('frucsteps').innerHTML = frucSetupHtml();
    $('frucbtns').style.display = '';
    $('frucsetup').style.display = 'block';
  } else {
    $('frucsetup').style.display = 'none';
  }
}
// Exclusive-group behaviour: ticking a box clears the others. Unticking the ACTIVE box now leaves
// NOTHING checked, and that is the "no interpolation" state (the group replaced the old separate
// Interpolate checkbox), so syncInterp() runs to grey the Speed controls and retag the output.
function pickModel(box){
  for(const b of MODEL_BOXES()) if(b !== box) b.checked = false;
  syncModel();
  syncInterp();
  try{ lvModelUi(); lvSendOpts(); }catch{}   // live inherits this model choice
}
async function doFrucInstall(source){
  $('frucsetup').style.display = 'block';
  $('frucstatus').textContent = 'installing...'; $('frucstatus').style.color = 'var(--sub)';
  let r = {}; try { r = await ipcRenderer.invoke('fruc-install', source); } catch(e){ r = { ok:false, error:String(e) }; }
  await refreshFrucState();
  if(frucReady){ $('modelfruc').checked = true; $('modelgmfss').checked = false; localStorage.setItem('interpModel','fruc'); $('frucsetup').style.display = 'none'; }
  else if(r.ok){   // NvOFFRUC.dll copied fine; readiness just needs the one-time bridge build
    $('frucstatus').textContent = '✓ DLL installed'; $('frucstatus').style.color = 'var(--green)';
    $('frucsteps').innerHTML = 'NvOFFRUC.dll installed. Last step: build <b>nvoffruc_bridge.dll</b> into '
      + 'engine/nvoffruc once (see DEVELOPMENT.md, "Building the native bridges"). Until then GMFSS is used.';
  } else {
    $('frucstatus').textContent = '✗ NOT READY'; $('frucstatus').style.color = '#e85c5c';
    $('frucsteps').innerHTML = '✗ ' + (r.error || 'install failed') + '<br>' + frucSetupHtml();
  }
}
$('frucget').onclick = () => ipcRenderer.invoke('fruc-open-download');
$('frucbrowsezip').onclick = async () => { const p = await ipcRenderer.invoke('fruc-choose'); if(p) doFrucInstall(p); };  // selecting a .zip auto-installs
for(const b of ['modelgmfss','modelrife','modeldlss','modelfruc','modelnvof','modellsfg']) $(b).onchange = () => pickModel($(b));
// The DRBA sub-option persists on its own key (default OFF: max fluidity is the app's default
// philosophy; DRBA deliberately keeps character cadence); toggling it never clears RIFE.
$('rifedrba').checked = localStorage.getItem('rifeDrba') === '1';
$('rifedrba').onchange = () => { localStorage.setItem('rifeDrba', $('rifedrba').checked ? '1' : '0'); try{ lvSendOpts(); }catch{} };
// On load: learn readiness, restore the saved choice, then reflect it (a setup hint shows if the
// chosen model's runtime is missing). Skipped while Interpolate is off: syncInterp() has cleared
// the boxes and re-checking Interpolate restores the choice itself.
function restoreModelChoice(){
  let saved = localStorage.getItem('interpModel');
  // Migration from the separate-Interpolate-checkbox era: interp='0' meant "no interpolation", which
  // is now expressed as no model ticked. Translate it once instead of bumping SETTINGS_VERSION,
  // which would wipe every other saved preference.
  if(localStorage.getItem('interp') === '0'){
    saved = 'none'; localStorage.setItem('interpModel', 'none');
    localStorage.removeItem('interp');
  }
  localStorage.removeItem('svpNvof');   // retired: the SVP models are gone
  if(saved === 'svp' || saved === 'svpnvof'){  // the default model takes over
    saved = 'gmfss'; localStorage.setItem('interpModel', 'gmfss');
  }
  if(saved === 'none'){     // interpolation off: leave the whole group unticked
    for(const b of MODEL_BOXES()) b.checked = false;
    return;
  }
  const box = saved === 'fruc' ? 'modelfruc'
            : saved === 'dlssg' ? 'modeldlss' : saved === 'rife' ? 'modelrife'
            : saved === 'lsfg' ? 'modellsfg' : saved === 'nvof' ? 'modelnvof' : 'modelgmfss';
  for(const b of MODEL_BOXES()) b.checked = false;
  $(box).checked = true;
}
// Restore FIRST (the saved choice may be "none"), then reflect it: syncInterp greys the Speed
// controls when nothing is ticked.
refreshFrucState().then(() => { restoreModelChoice(); syncModel(); syncInterp(); try{ lvModelUi(); lvSendOpts(); }catch{} });

// --- Live mode (real-time DLSS-G overlay; engine/dlssg/smv-live.exe) -------------------------------
// Independent of the file workflow: usable straight from the welcome screen. Two ways in, both
// ending at the exe's --fg mode (whatever window is foreground at spawn goes live): the ` global
// hotkey (instant, main-side), and the Smooth It Live! button, which arms a 5s countdown so the
// user can click the target window first (the Lossless Scaling "Scale" pattern, no keyboard
// needed). main forwards the exe's stderr as lv-out and auto-restarts on exit 4 (target resized).
let lvState = 'idle';   // idle | arming (countdown running) | running
let lvStopReq = false;
let lvTimer = null;
// the hotkey + button share these settings; main mirrors them for hotkey-started sessions
// Live's output target is INHERITED from the Speed selectors (screen-match or the fps box): the
// user sets ONE target for both file renders and live smoothing. Live inherits the Speed MODE,
// not a number derived from a loaded file (that coupling is exactly what made Live target 60
// just because the bundled sample clip is 30 fps).
function liveTargetFps(){
  return Math.round(speedMode() === 'screen' ? screenHz : curFps());   // 359.979 Hz panels read as 360
}
// ONE model choice drives file renders AND live smoothing: live derives its engine from the
// Interpolation model group. Models without a live backend fall back with a visible note
// (FRUC -> GMFSS live; DRBA -> plain RIFE live: the triple-window timing needs a future
// live implementation).
function liveModelInfo(){
  // fixed: the model cannot resample to the exact fps target; it approximates it with a whole
  // multiplier of the captured window's own rate (derived in the exe from the target) - DLSS-G
  // by design (Streamline, capped at 6x)
  if(modelIsDlss()) return { model:'dlssg', name:'DLSS 4.5', note:'', fixed:true };
  // DRBA live: the rifedrba backend renders the three-frame window, so the picture trails
  // the capture by one source frame
  if(modelIsRifeDrba()) return { model:'rifedrba', name:'RIFE (DRBA)', note:'adds one source frame of delay' };
  if(modelIsRife()) return { model:'rife', name:'RIFE', note:'' };
  // Smooth Motion live: every fraction takes the nearest node of the pair's midpoint tree, so
  // the fruc backend is adaptive like rife/gmfss; HDR-capable like rife (the bridge warps 8-bit PQ
  // for the tweens, the real frames keep full precision)
  if(modelIsFruc()) return { model:'fruc', name:'Smooth Motion', note:'tweens at 8-bit precision' };
  // Frame Blend is adaptive like rife/gmfss (no fixed flag): the server backend resamples
  // to the fps target the same way, which is what makes it a like-for-like pacing comparison.
  if(modelIsLsfg()) return { model:'blend', name:'Frame Blend', note:'flow at the Image scale, full resolution warps' };
  // NVIDIA Optical Flow (direct): adaptive like rife (the splat takes any fraction), native only
  if(modelIsNvof()) return { model:'nvof', name:'NVIDIA Optical Flow', note:'' };
  // nothing ticked = interpolation off (same meaning as file renders): the echo backend passes
  // the captured frames through at their own rate and applies the live effects (Restore,
  // Sharpen, Upscale to, RTX VSR, RTX HDR), so the effects work without interpolating.
  if(!interpOn()) return { model:'echo', name:'No interpolation', note:'', effectsOnly:true };
  return { model:'gmfss', name:'GMFSS', note:'' };
}
// Upscaler policy: live FILL upscales with RTX VSR BY DEFAULT when its runtime is
// present (live output is ephemeral, so a better silent default is fine). The shared RTX VSR
// checkbox stays the opt-out: explicitly unchecked ('0') means bicubic everywhere. FILE
// renders remain strict opt-in (checkbox checked) so render output never changes silently.
function liveVsrOn(){ return !!(rtxReady.vsr && localStorage.getItem('rtxvsrOn') !== '0'); }
// Live TrueHDR: STRICT opt-in via the shared RTX HDR checkbox (a TrueHDR
// expansion is a deliberate look change, so live matches file renders, not the VSR silent
// default). Applies to the server models on HDR screens; the exe drops it elsewhere. No
// srcHdr gate here: live has no loaded file, the source is whatever window gets captured.
function liveHdrOn(){ return !!(rtxReady.hdr && $('rtxhdr').checked); }
// The exe's loading note / HUD names what is actually ticked: the model, then the live effects the server
// models apply (DLSS 4.5 takes none), e.g. "GMFSS + DLSS 5" or "DLSS 5 + Sharpen".
function liveLoadLabel(mi){
  const parts = mi.effectsOnly ? [] : [mi.name];
  if(mi.model !== 'dlssg'){
    if(restoreOn()) parts.push('Restore');
    if(nrOn()) parts.push('DLSS 5');
    if($('sharpen').checked && (parseFloat($('sharpval').value) || 0) > 0) parts.push('Sharpen');
    const upH = upTargetH();
    if(upH) parts.push((liveVsrOn() ? 'RTX VSR ' : 'upscale ') + upH + 'p');
    if(liveHdrOn()) parts.push('RTX HDR');
  }
  return parts.length ? parts.join(' + ') : 'passthrough (nothing ticked)';
}
function lvSendOpts(){
  const mi = liveModelInfo();
  const hp = hdrColorPayload();
  ipcRenderer.send('lv-opts', {
    // ONE target source: the Speed selectors. Adaptive models (RIFE/GMFSS/blend) resample to
    // it; fixed models (DLSS 4.5) approximate it with a whole multiplier the exe derives
    // from the captured window's measured rate.
    model: mi.model,
    label: liveLoadLabel(mi),   // what is ticked, for the exe's loading note + substitution note
    note: mi.note,    // message/HUD (the raw backend id reads as the wrong model)
    flow: +$('lvflow').value,
    fit: $('lvfit').value,
    target: liveTargetFps(),
    // live effects: sharpen inherits the file-render setting; VSR defaults ON for the live
    // fill upscale (liveVsrOn above; the server ignores it outside fill / on non-enlarging fits)
    sharpen: $('sharpen').checked ? parseFloat($('sharpval').value) || 0 : 0,
    rtxvsr: liveVsrOn(),
    // "Upscale to" height (0 = off): live renders the model frame at that size first (VSR
    // when enlarging), then fits it to the window or monitor. Match screen = this screen.
    uph: upTargetH(),
    // Restore (AI detail): Real-ESRGAN first on every presented frame, python route only
    restore: restoreOn(),
    // NVIDIA DLSS 5: Neural Rendering once per captured frame (runtime installed), python route
    dlssnr: nrOn(), nrstructure: nrStructure(), nrtone: nrTone(), nrstyle: nrStyle(),
    // live TrueHDR (SDR window -> HDR out) + the RTX HDR tone/colour knobs it inherits
    rtxhdr: liveHdrOn(),
    hdrcolor: hp.color,
    hdrsat: hp.saturation,
    hdrcon: sdkCon(),
    hdrvib: effVibrance(),
    hdrsb: effSatBoost(),
    hud: $('lvhud').checked,
    hudlat: $('lvhudlat').checked,
  });
  const load = mi.model === 'rife' ? 'starts in ~3s, under 1s again while its engines stay loaded (first time at a new window size ~45s)'
    : mi.model === 'rifedrba' ? 'starts in ~3s (first time at a new window size ~55s)'
    : mi.model === 'gmfss' ? 'loads ~4s on start, under 1s again while it stays loaded (first time at a new window size ~30s)'
    : mi.model === 'nvof' ? 'starts in a few seconds, nothing to build' : '';
  // A fixed model cannot hit an arbitrary target exactly: it picks the nearest whole multiple
  // of the captured window's own rate, so say that instead of promising the exact number.
  const lvT = liveTargetFps();
  $('lvtarget').textContent = mi.name
    + (mi.effectsOnly ? ' · effects only (Restore, Sharpen, Upscale to, RTX VSR, RTX HDR) at the source rate, capped by the ' + lvT + ' fps Speed setting; tick a model to smooth'
       : mi.fixed ? ' · nearest whole multiple of the captured rate to ' + lvT + ' fps'
                  + (mi.model === 'dlssg' ? ' (up to 6x)' : '')
       : ' · targets ' + lvT + ' fps (the Speed setting below)')
    + (load ? ' · ' + load : '')
    + (liveHdrOn() && ['rife', 'rifedrba', 'gmfss', 'blend', 'nvof'].includes(mi.model) ? ' · RTX HDR (on HDR screens)' : '')
    + (mi.note ? ' · ' + mi.note : '') + '  ·  ';
}
// Image scale = the resolution the whole pipeline processes at, upscaled back on output:
// universal for the server models with real compute (RIFE/GMFSS); DLSS-G has no such input.
function lvModelUi(){
  const mi = liveModelInfo();
  const m = mi.model;
  const server = m !== 'dlssg';       // everything but DLSS-G runs through the server
  // models with an Image scale input: the whole pipeline runs at the reduced size (blend too)
  const hasFlow = m === 'rife' || m === 'rifedrba' || m === 'gmfss' || m === 'blend' || m === 'fruc' || m === 'nvof';
  // stays enabled while live runs: moving it relaunches the session with the new scale
  $('lvflow').disabled = !hasFlow;
  $('lvflownum').textContent = $('lvflow').value + '%';
  $('lvflowhint').textContent = hasFlow
    ? 'whole pipeline at lower res, upscaled back: big speedup, softer picture (live and file renders)'
    : 'not applicable to this model';
  // Fill screen needs the server route (the upscale runs there); Whole screen (monitor
  // capture, 1:1) works with EVERY model incl. DLSS-G. For dlssg the select stays enabled
  // but a fill selection is bounced back to window with a hint.
  $('lvfit').disabled = lvState === 'running';
  if(!server && $('lvfit').value === 'fill') $('lvfit').value = 'window';
  // the Upscale to selector applies live too (server models): rendered at that height first,
  // then fitted; named here so the inherited setting is visible from the Live panel
  const upH = server ? upTargetH() : 0;
  const upTxt = upH ? (liveVsrOn() ? 'RTX VSR' : 'upscale') + ' to ' + upH + 'p first' : '';
  // Restore applies live too (server models, python route); named here like Upscale to
  const resTxt = server && restoreOn() ? 'Restore (AI detail) first, heavy' : '';
  // DLSS 5 applies live too (server models, python route): once per captured frame
  const nrTxt = server && nrOn() ? 'DLSS 5 on every captured frame' : '';
  const effTxt = [resTxt, nrTxt].filter(Boolean).join(' · ');
  $('lvfithint').textContent =
    $('lvfit').value === 'fill' ? 'enlarges the window to fill its monitor, aspect kept · '
                                  + (upTxt || (liveVsrOn() ? 'RTX VSR upscale' : 'upscale'))
                                  + (effTxt ? ' · ' + effTxt : '')
    : $('lvfit').value === 'monitor' ? 'smooths everything on the monitor the window is on'
                                       + (upTxt ? ' · ' + upTxt : '') + (effTxt ? ' · ' + effTxt : '')
    : !server ? 'Fill screen needs the RIFE or GMFSS model' : [upTxt, effTxt].filter(Boolean).join(' · ');
}
function lvUi(){
  const b = $('lvgo');
  if(lvState === 'running') b.textContent = 'Stop';
  else if(lvState === 'idle') b.textContent = 'Smooth It Live!';
  // arming: the countdown tick owns the button text ("Cancel (N)")
  modeBtnUi(lvState !== 'idle');
  lvModelUi();
}
function lvDisarm(){
  if(lvTimer){ clearInterval(lvTimer); lvTimer = null; }
  if(lvState === 'arming'){ lvState = 'idle'; lvUi(); $('lvstat').textContent = ''; }
}
// mid-run scale changes: the scale is a spawn-time arg, so a running session is relaunched
// (same target window) once the user settles on a value; debounced so a drag = one restart
let lvFlowRestartT = null;
$('lvflow').oninput = () => {
  localStorage.setItem('lvFlow', $('lvflow').value);
  lvModelUi();
  lvSendOpts();
  if(lvState === 'running'){
    clearTimeout(lvFlowRestartT);
    lvFlowRestartT = setTimeout(() => {
      if(lvState === 'running') ipcRenderer.send('lv-restart');
    }, 700);
  }
};
// The fps target is a spawn-time argument (--target, and the exe derives the slot
// count from it), so a Speed change during a running session takes the same relaunch path as
// the Image scale slider: 700 ms debounce, 'lv-restart', same target window. Only a real
// change of the EFFECTIVE target (what main.ts would pass) restarts anything.
function lvEffTarget(){ return Math.min(1000, Math.max(10, Math.round(liveTargetFps()))); }
let lvSpawnTarget = 0;      // effective target the running session was spawned with
let lvTargetRestartT = null;
function lvTargetChanged(){
  if(lvState !== 'running') return;
  clearTimeout(lvTargetRestartT);
  lvTargetRestartT = setTimeout(() => {
    if(lvState === 'running' && lvEffTarget() !== lvSpawnTarget) ipcRenderer.send('lv-restart');
  }, 700);
}
$('lvfit').onchange = () => {
  localStorage.setItem('lvFit', $('lvfit').value);
  lvModelUi();
  lvSendOpts();
};
// on-screen readout toggle (default ON; off spawns the exe with --no-hud)
if(localStorage.getItem('lvHud') === '0') $('lvhud').checked = false;
// latency segment of the same readout (default ON = today's behaviour; off spawns
// --no-hud-latency). Only meaningful while the meter itself is shown, so it greys out
// with the meter rather than pretending to do something.
if(localStorage.getItem('lvHudLat') === '0') $('lvhudlat').checked = false;
function lvHudLatUi(){
  const on = $('lvhud').checked;
  $('lvhudlat').disabled = !on;
  $('lvhudlatlbl').style.opacity = on ? '1' : '0.45';
  $('lvhudlatlbl').style.cursor = on ? 'pointer' : 'default';
}
$('lvhud').onchange = () => { localStorage.setItem('lvHud', $('lvhud').checked ? '1' : '0'); lvHudLatUi(); lvSendOpts(); };
$('lvhudlat').onchange = () => { localStorage.setItem('lvHudLat', $('lvhudlat').checked ? '1' : '0'); lvSendOpts(); };
localStorage.removeItem('lvNative');   // retired: the RIFE live route always runs inside smv-live.exe
lvHudLatUi();
// Hotkey recorder: click the button, press the new key or combo; Esc cancels. Electron
// accelerator strings ("`", "F9", "Ctrl+Alt+S"); main answers whether registration
// succeeded (another app can hold the key) and keeps the old key on failure.
let lvKeyCapture = false;
function lvAccelFrom(e){
  if(['Control','Alt','Shift','Meta'].includes(e.key)) return null;   // bare modifier: keep waiting
  const mods = [];
  if(e.ctrlKey) mods.push('Ctrl');
  if(e.altKey) mods.push('Alt');
  if(e.shiftKey && e.key.length > 1) mods.push('Shift');   // printable keys already carry shift in e.key
  const key = e.key === ' ' ? 'Space' : e.key.length === 1 ? e.key.toUpperCase() : e.key;
  return [...mods, key].join('+');
}
$('lvkey').onclick = () => {
  if(lvKeyCapture || lvState !== 'idle') return;
  lvKeyCapture = true;
  const old = $('lvkey').textContent;
  $('lvkey').textContent = 'press a key…';
  $('lvkeyhint').textContent = 'Esc cancels';
  const done = (txt, hint) => { lvKeyCapture = false; $('lvkey').textContent = txt; $('lvkeyhint').textContent = hint || ''; window.removeEventListener('keydown', h, true); };
  const h = async (e) => {
    e.preventDefault(); e.stopPropagation();
    if(e.key === 'Escape'){ done(old); return; }
    const acc = lvAccelFrom(e);
    if(!acc) return;
    const ok = await ipcRenderer.invoke('lv-hotkey', acc);
    if(ok){ localStorage.setItem('lvHotkey', acc); done(acc); }
    else done(old, 'could not register ' + acc + ' (taken by another app?)');
  };
  window.addEventListener('keydown', h, true);
};
// restore the persisted hotkey at boot (main registers the default ` on its own)
{
  const savedKey = localStorage.getItem('lvHotkey');
  if(savedKey && savedKey !== '`')
    ipcRenderer.invoke('lv-hotkey', savedKey).then(ok => { if(ok) $('lvkey').textContent = savedKey; });
}
$('lvgo').onclick = async () => {
  if(lvState === 'running'){
    lvStopReq = true;
    try { await ipcRenderer.invoke('lv-stop'); } catch {}   // lv-done resets the UI
    return;
  }
  if(lvState === 'arming'){ lvDisarm(); return; }
  // arm: give the user 5s to focus the window they want smoothed, then target the foreground
  lvState = 'arming';
  let n = 5;
  const tick = () => {
    if(n <= 0){
      if(lvTimer){ clearInterval(lvTimer); lvTimer = null; }
      $('lvgo').textContent = 'starting…';
      ipcRenderer.send('lv-start-fg');
      return;
    }
    $('lvgo').textContent = 'Cancel (' + n + ')';
    $('lvstat').textContent = 'click the window you want smoothed…';
    n--;
  };
  tick();
  lvTimer = setInterval(tick, 1000);
};
// stderr arrives in ARBITRARY chunk boundaries (the exe's unbuffered stderr can split a
// stats line mid-word), so buffer into complete lines before parsing - a per-chunk regex
// intermittently drops the latency tail into the next chunk.
let lvOutBuf = '';
let lvRefused = false;   // a refusal reason owns the status line until the next session
function lvLine(t){
  // The host's refusal reason ("native host: <reason>", or the engine lookup / build failure) is
  // the ONE thing the user must read: it stays the last word on the panel, because the exe's own
  // generic lines ("this session cannot start in the native host", "live server failed to
  // start") arrive after it and would otherwise replace it, and lv-out never reaches the log panel.
  const rf = /native host: (?!this session cannot start)([^\r\n]+)/.exec(t)
          || /native: (the host could not find or build the engines[^\r\n(]*)/.exec(t);
  if(rf){ lvRefused = true; $('lvstat').textContent = rf[1].trim().slice(0, 240); return; }
  if(lvRefused) return;
  // surface the exe's 2s stats lines; pass real failures through verbatim (trimmed)
  const m = /live: ([\d.]+) captured fps -> ([\d.]+) presented fps.*?latency ~(\d+)ms/.exec(t)
         || /live: ([\d.]+) captured fps -> ([\d.]+) presented fps/.exec(t);
  if(m) $('lvstat').textContent = m[1] + ' fps in → ' + m[2] + ' fps out' + (m[3] ? '  ·  ~' + m[3] + ' ms latency' : '');
  // the native host is the only live route: its handoff start and cold engine build mean "loading"
  else if(/engine handoff started|host (GMFSS |Restore )?engine build for/.test(t)) $('lvstat').textContent = 'loading the model…';
  else if(/FAIL|not supported|no visible window|not a visible window|no frames captured|cannot start in the native host|live server (protocol error|stalled)/i.test(t))
    $('lvstat').textContent = t.trim().slice(0, 160);
}
ipcRenderer.on('lv-out', (_e, t) => {
  lvOutBuf += t;
  const lines = lvOutBuf.split('\n');
  lvOutBuf = lines.pop();          // keep the trailing partial line for the next chunk
  for(const line of lines) lvLine(line);
});
// sessions can start from main (the ` hotkey) with no renderer involvement: mirror them here
ipcRenderer.on('lv-started', () => {
  if(lvTimer){ clearInterval(lvTimer); lvTimer = null; }
  setMode('live');   // the ` hotkey can start a session with no renderer involvement

  lvState = 'running'; lvStopReq = false; lvRefused = false; lvUi();
  try{ lvSpawnTarget = lvEffTarget(); }catch{}   // what --target this session carries
  $('lvstat').textContent = 'starting…';
});
ipcRenderer.on('lv-done', (_e, code) => {
  lvState = 'idle'; lvUi();
  // a requested Stop kills the exe (code null via kill(), or nonzero on Windows), and the
  // ` hotkey stops the same way from main without setting lvStopReq: both read as a clean stop
  if(!lvRefused){   // a refusal reason stays on the line: an exit code adds nothing to it
    if(lvStopReq || code === 0 || code === null) $('lvstat').textContent = 'stopped';
    else if(code === 5) $('lvstat').textContent = 'no smoothable window was focused - click the window you want smoothed during the countdown';
    else $('lvstat').textContent += '  (live mode ended, exit ' + code + ')';
  }
  lvStopReq = false;
});
(async () => {
  try {
    lvReady = await ipcRenderer.invoke('lv-ready');
    if(lvReady){
      $('lvpanel').style.display = '';
      const fl = +localStorage.getItem('lvFlow');
      if(fl >= 1 && fl <= 100) $('lvflow').value = fl;
      localStorage.removeItem('lvFlowPreset');   // retired: the 540/720 presets moved to the Upscale to selector
      localStorage.removeItem('lvFscale');       // retired: the Flow scale control is gone
      const ft = localStorage.getItem('lvFit');
      if(ft && [...$('lvfit').options].some(o => o.value === ft)) $('lvfit').value = ft;
      lvModelUi();
      lvSendOpts();   // seed the hotkey's settings at boot
    }
  } catch {}
  lvUnavailUi();
})();

// First page = the full settings surface: load the bundled example clip through the normal
// video path (probe + all panels + the before/after preview) so every visual option shows its
// effect before the user picks a file. Smooth It! stays gated until a real video is loaded;
// picking/dropping one replaces the example via the same loadVideo flow.
(async () => {
  try {
    if(input) return;   // a video arrived first (drag-drop straight onto the launching window)
    const ex = await ipcRenderer.invoke('example-path');
    if(ex && !input) await loadVideo(ex, true);
  } catch {}
})();

// --- Dolby Vision export tools (mirrors the RTX library flow) ------------------------------------
// DV 8.1 export needs ONE non-shippable open-source binary - dovi_tool - dropped into engine/dvtools;
// the RPU it injects is muxed by the bundled ffmpeg and the DV signaling box is written by the engine
// itself (hdr10_meta.inject_dv_config), so no GPAC/MP4Box is involved. The setup box appears when the
// toggle is on and dovi_tool is missing, and collapses to nothing once it is installed (dv-ready).
let dvReady = { dovi:false, ready:false };
function syncDv(){
  if($('dvexport').checked) checkDvReady();
  else $('dvsetup').style.display = 'none';
  updateDvHpHints();
}
// Both dynamic-HDR exports ride on the HDR10 render AND need an MP4 container. When the loaded
// video's tracks force MKV output (subtitles, exotic audio), the engine would skip them with only
// a log notice at render time - so say it up front, next to the checkbox, the moment the output
// name resolves to .mkv. Pure text updates (no IPC), safe to call from refresh() on every change.
function updateDvHpHints(){
  const mkv = (lastOut || '').toLowerCase().endsWith('.mkv');
  const hint = (name) => !$('rtxhdr').checked ? '(needs RTX HDR on)'
    : mkv ? '(skipped for this video: its tracks need MKV output, ' + name + ' needs MP4)'
    : '(HDR10-compatible, falls back to HDR10)';
  $('dvhint').textContent = hint('Dolby Vision');
  $('hphint').textContent = hint('HDR10+');
}
async function checkDvReady(){
  $('dvstatus').textContent = 'checking...'; $('dvstatus').style.color = 'var(--sub)';
  try { dvReady = await ipcRenderer.invoke('dv-ready'); } catch { dvReady = { dovi:false, ready:false }; }
  const line = (dvReady.dovi ? '✓' : '✗') + ' dovi_tool (RPU generate + inject)';
  if(dvReady.ready){
    $('dvstatus').textContent = '✓ READY'; $('dvstatus').style.color = 'var(--green)';
    $('dvsteps').innerHTML = line + '<br>The Dolby Vision export tool is installed.';
  } else {
    $('dvstatus').textContent = '✗ NOT READY'; $('dvstatus').style.color = '#e85c5c';
    // Name the exact Windows download, since the release page also lists macOS/Linux/ARM builds.
    $('dvsteps').innerHTML = line
      + '<br>Click <b>Get dovi_tool</b> and pick the <b>x64 Windows</b> build '
      + '(<code>&hellip;x86_64-pc-windows-msvc.zip</code>), then <b>Choose .zip&hellip;</b> and pick it; it installs automatically.';
  }
  $('dvbtns').style.display = dvReady.ready ? 'none' : '';   // hide Get / Choose once dovi_tool is installed
  $('dvsetup').style.display = dvReady.ready ? 'none' : 'block';
}
async function doDvInstall(source){
  $('dvsetup').style.display = 'block';
  $('dvstatus').textContent = 'installing...'; $('dvstatus').style.color = 'var(--sub)';
  let r = {}; try { r = await ipcRenderer.invoke('dv-install', source); } catch(e){ r = { ok:false, error:String(e) }; }
  if(!r.ok) $('dvsteps').innerHTML = '✗ ' + (r.error || 'install failed')
      + '<br>Pick the dovi_tool&hellip;.zip again with <b>Choose .zip&hellip;</b>.';
  await checkDvReady();
  if(!r.ok) $('dvsetup').style.display = 'block';   // keep an install error visible
}
$('dvget').onclick = () => ipcRenderer.invoke('dv-open-download');
$('dvbrowsezip').onclick = async () => { const p = await ipcRenderer.invoke('dv-choose'); if(p) doDvInstall(p); };   // selecting auto-installs
if(localStorage.getItem('dvOn') === '1') $('dvexport').checked = true;   // default OFF
$('dvexport').onchange = () => { localStorage.setItem('dvOn', $('dvexport').checked ? '1' : '0'); syncDv(); };
$('rtxhdr').addEventListener('change', () => { if($('dvexport').checked) syncDv(); });   // hint tracks HDR state
syncDv();

// --- HDR10+ export tool (mirrors the Dolby Vision flow) ------------------------------------------
// HDR10+ export needs ONE non-shippable open-source binary - hdr10plus_tool - dropped into
// engine/hptools; the engine collects the per-frame brightness stats itself and the tool injects
// the ST 2094-40 SEI (see _hp_export in the engine). The setup box appears when the toggle is on
// and the tool is missing, and collapses once it is installed (hp-ready).
let hpReady = { ready:false };
function syncHp(){
  if($('hpexport').checked) checkHpReady();
  else $('hpsetup').style.display = 'none';
  updateDvHpHints();
}
async function checkHpReady(){
  $('hpstatus').textContent = 'checking...'; $('hpstatus').style.color = 'var(--sub)';
  try { hpReady = await ipcRenderer.invoke('hp-ready'); } catch { hpReady = { ready:false }; }
  const line = (hpReady.ready ? '✓' : '✗') + ' hdr10plus_tool (ST 2094-40 SEI inject)';
  if(hpReady.ready){
    $('hpstatus').textContent = '✓ READY'; $('hpstatus').style.color = 'var(--green)';
    $('hpsteps').innerHTML = line + '<br>The HDR10+ export tool is installed.';
  } else {
    $('hpstatus').textContent = '✗ NOT READY'; $('hpstatus').style.color = '#e85c5c';
    // Name the exact Windows download, since the release page also lists macOS/Linux/ARM builds.
    $('hpsteps').innerHTML = line
      + '<br>Click <b>Get hdr10plus_tool</b> and pick the <b>x64 Windows</b> build '
      + '(<code>&hellip;x86_64-pc-windows-msvc.zip</code>), then <b>Choose .zip&hellip;</b> and pick it; it installs automatically.';
  }
  $('hpbtns').style.display = hpReady.ready ? 'none' : '';   // hide Get / Choose once the tool is installed
  $('hpsetup').style.display = hpReady.ready ? 'none' : 'block';
}
async function doHpInstall(source){
  $('hpsetup').style.display = 'block';
  $('hpstatus').textContent = 'installing...'; $('hpstatus').style.color = 'var(--sub)';
  let r = {}; try { r = await ipcRenderer.invoke('hp-install', source); } catch(e){ r = { ok:false, error:String(e) }; }
  if(!r.ok) $('hpsteps').innerHTML = '✗ ' + (r.error || 'install failed')
      + '<br>Pick the hdr10plus_tool&hellip;.zip again with <b>Choose .zip&hellip;</b>.';
  await checkHpReady();
  if(!r.ok) $('hpsetup').style.display = 'block';   // keep an install error visible
}
$('hpget').onclick = () => ipcRenderer.invoke('hp-open-download');
$('hpbrowsezip').onclick = async () => { const p = await ipcRenderer.invoke('hp-choose'); if(p) doHpInstall(p); };   // selecting auto-installs
if(localStorage.getItem('hpOn') === '1') $('hpexport').checked = true;   // default OFF
$('hpexport').onchange = () => { localStorage.setItem('hpOn', $('hpexport').checked ? '1' : '0'); syncHp(); };
$('rtxhdr').addEventListener('change', () => { if($('hpexport').checked) syncHp(); });   // hint tracks HDR state
syncHp();

// Before/after preview (panel above Smooth It): render ONE frame at the current spatial settings via the
// main process's preview (render/preview.js, the native host's pass chain) and show original vs processed. It applies RTX HDR (when on + ready) and FSR/CAS
// sharpen (when on), so the user can see how those settings change the picture before a full render.
// Back / random-frame step through random positions across the clip. HDR is tonemapped to sRGB because
// this canvas cannot show PQ. Images load as object URLs (nodeIntegration) to dodge file:// webSecurity,
// and old URLs are revoked so memory does not grow.
const _fs = require('fs');
let prevTotal = 0, prevIdx = 0, prevHist = [];
function sharpStrength(){ return $('sharpen').checked ? (parseFloat($('sharpval').value) || 0) : 0; }
function hdrOn(){ return $('rtxhdr').checked && rtxReady.hdr && !(info && info.srcHdr); }
function setPrevImg(el, file, mime){
  const old = el.dataset.url; if(old){ try { URL.revokeObjectURL(old); } catch {} }
  const url = URL.createObjectURL(new Blob([_fs.readFileSync(file)], { type: mime || 'image/png' }));
  el.dataset.url = url; el.src = url;
  el.style.visibility = '';        // imgs start hidden so the broken-image icon never shows pre-load
}

// Live progress thumbnail: during a render the engine drops a small PNG of the frame it just
// produced (~1/s, SMV_LIVE_PREVIEW); poll it by mtime and show it under the progress bar.
let livePath = null, liveTimer = null, liveM = 0;
ipcRenderer.invoke('live-path').then(p => { livePath = p; });
// Hide/Show toggle: the preview costs a little GPU/CPU per second, so hiding it tells the engine
// (via main's live-off flag) to stop producing it, not just hides the image. Persisted.
let liveHidden = localStorage.getItem('liveHidden') === '1';
function applyLiveOff(){ ipcRenderer.send('live-off', liveHidden); }   // sync engine to the saved choice
function syncLiveToggle(){
  $('livetoggle').textContent = liveHidden ? 'Show preview' : 'Hide preview (frees resources)';
  if(liveHidden) $('live').style.display = 'none';   // Show reveals again on the next written frame
}
$('livetoggle').onclick = () => {
  liveHidden = !liveHidden;
  localStorage.setItem('liveHidden', liveHidden ? '1' : '0');
  syncLiveToggle(); applyLiveOff();
};
syncLiveToggle();
function updateLive(){
  if(!livePath || liveHidden) return;             // hidden: engine produces nothing, keep it hidden
  try {
    const m = _fs.statSync(livePath).mtimeMs;
    if(m === liveM) return;                       // nothing new yet
    liveM = m;
    setPrevImg($('live'), livePath, 'image/png');
    $('live').style.display = 'block';
  } catch {}                                      // not written yet this run
}
function updatePrevNav(){          // back only makes sense once a second frame has been generated
  $('prevprev').style.visibility = prevHist.length > 1 ? '' : 'hidden';
}
function syncPreview(){            // the preview panel appears once a video is loaded
  $('previewpanel').style.display = input ? '' : 'none';
  syncHdrColor();                  // HDR colour controls track the HDR toggle / readiness / source
}
let prevBusy = false, prevPending = null, previewLite = false;   // serialize renders. previewLite: skip heavy RTX passes on the auto-preview (set by openPreviewForVideo, one-shot)
let lastPrevKey = '', inflightPrevKey = ''; // settings signatures of the shown / mid-render preview
function prevSettingsSig(){                 // everything that changes what the processed pane shows
  const p = hdrColorPayload();   // normalized: the vibrance feature at zero strength equals OFF
  return [sharpStrength(), restoreOn(), hdrOn(), p.color, p.saturation, effVibrance(), effSatBoost(), sdkCon(),
          upFactor() || 0, !!($('rtxvsr').checked && rtxReady.vsr), nrOn(), nrStructure(), nrTone(), nrStyle(), nrMaskOn(),
          nvOrderOn()];
}
let lastPrevInput = null;                    // which video the shown original belongs to
async function loadPreview(frame, bg){   // bg: background "refine" pass (the RTX auto-upgrade) - keep the shown image up, no blocking spinner
  if(!input) return;
  if(prevBusy){ prevPending = (frame == null ? 'mid' : frame); return; }
  prevBusy = true;
  const hdr = hdrOn(), sharpen = sharpStrength(), sig = prevSettingsSig();
  // The auto-preview on video load runs "lite": it skips the heavy RTX VSR + TrueHDR passes (the bulk of
  // the ~5s cost), so the drop is fast. Adjusting any preview control renders the full RTX version.
  const lite = previewLite; previewLite = false;
  const useHdr = lite ? false : hdr;
  const useVsr = lite ? false : !!($('rtxvsr').checked && rtxReady.vsr && upFactor() > 1);   // VSR upscales only
  const useNr = lite ? false : nrOn();     // DLSS 5: its host takes seconds to start, so the lite pass skips it too
  inflightPrevKey = JSON.stringify([input, (frame == null ? 'mid' : frame)].concat(sig));
  $('hdrpreview').style.display = 'block';
  // On a background refine the lite image is already on screen: leave it up and skip the blocking spinner,
  // so the pane appears instantly and just sharpens to the RTX result when it lands. Otherwise show it.
  if(!bg){
    if($('prevproc').clientHeight < 40) $('prevprocwrap').style.minHeight = '120px';  // spinner space on first load
    setSpin($('prevprocwrap'), $('prevspin'), true);
  }
  // The original pane gets the wheel too, but only when the original is actually about to change:
  // first load, a different video, or a frame step. Settings tweaks (and background refines) keep the frame.
  const origChanging = !bg && (!$('prevorig').dataset.url || lastPrevInput !== input
    || (frame != null && frame !== 'mid' && frame !== prevIdx));
  if(origChanging){
    if($('prevorig').clientHeight < 40) $('prevorigwrap').style.minHeight = '120px';
    setSpin($('prevorigwrap'), $('prevorigspin'), true);
  }
  let r; try { r = await ipcRenderer.invoke('preview', Object.assign(
    { input, frame: (frame == null ? 'mid' : frame), sharpen, restore: restoreOn(), hdr: useHdr,
      vibrance: effVibrance(), satboost: effSatBoost(), contrast: sdkCon(),
      upscale: upFactor() || 0,
      rtxvsr: useVsr, dlssnr: useNr, nrmask: useNr && nrMaskOn(), nrstructure: nrStructure(), nrtone: nrTone(), nrstyle: nrStyle(),
      nvorder: nvOrderOn() }, hdrColorPayload())); }
  catch(e){ r = { error: String(e) }; }
  setSpin($('prevprocwrap'), $('prevspin'), false);
  $('prevprocwrap').style.minHeight = '';
  setSpin($('prevorigwrap'), $('prevorigspin'), false);
  $('prevorigwrap').style.minHeight = '';
  prevBusy = false;
  if(prevPending != null){ const f = prevPending; prevPending = null; loadPreview(f); }
  if(!r || r.error){ lastPrevKey = '';   // never dedupe-skip a retry after a failure
    $('prevnote').textContent = 'preview unavailable: ' + ((r && r.error) || 'no result'); return; }
  if(r.total) prevTotal = r.total;
  if(typeof r.frame === 'number') prevIdx = r.frame;
  setPrevImg($('prevorig'), r.original);
  const maskShown = !!r.nrmask;   // the DLSS 5 change map replaces the processed picture when asked for
  setPrevImg($('prevproc'), maskShown ? r.nrmask : r.processed);
  lastPrevInput = input;
  lastPrevKey = lite ? '' : JSON.stringify([input, prevIdx].concat(sig));   // lite skipped RTX: force a full re-render on the next adjustment
  const upActive = upFactor() > 0;   // the processed side is upscaled too, so never call it unchanged
  const vsrOn = useVsr;              // what the processed pane actually rendered (lite skips VSR)
  const active = useHdr || sharpen > 0 || restoreOn() || upActive || useNr;
  // Plain bicubic upscale with no AI/enhancement pass: both panes are the same bicubic image, so say so.
  const bicubicOnly = upActive && !vsrOn && !restoreOn() && sharpen <= 0 && !useHdr && !useNr;
  const rtxSkipped = lite && (hdr || nrOn() || ($('rtxvsr').checked && rtxReady.vsr && upFactor() > 1));   // RTX / DLSS 5 on in settings but skipped for the fast auto-preview
  const srcHdr = !!(info && info.srcHdr);
  $('prevoriglabel').textContent = srcHdr ? 'Original (HDR, tonemapped)' : 'Original';
  const parts = []; if(restoreOn()) parts.push('Restore');
  if(upActive) parts.push(vsrOn ? 'VSR' : upFactor() < 1 ? 'downscale' : 'upscale');
  if(useNr) parts.push('DLSS 5');
  if(sharpen > 0) parts.push('FSR'); if(useHdr) parts.push('HDR');
  $('prevproclabel').textContent = !active ? 'Unchanged'
    : maskShown ? 'DLSS 5 changed pixels'
    : 'Processed (' + parts.join(' + ') + ')';
  $('prevlabel').textContent = 'frame ' + prevIdx + (prevTotal ? ' / ' + prevTotal : '');
  $('prevnote').textContent = (maskShown ? 'DLSS 5 changed ' + r.nrmaskPct + '% of the pixels, average change ' + r.nrmaskMean + '/255 · bright = large change, dark = untouched'
    : rtxSkipped ? 'quick preview shown; refining to the full RTX VSR/HDR/DLSS 5 version…'
    : srcHdr ? 'source is already HDR (shown tonemapped); RTX HDR does not apply'
    : !active ? 'no Restore, FSR, DLSS 5 or RTX HDR enabled, the output will match the source'
    : bicubicOnly ? (upFactor() < 1 ? 'plain downscale, both panes match (RTX VSR is for upscaling)'
                                    : 'plain upscale, no AI detail added (enable RTX VSR for that; both panes match)')
    : useHdr ? 'HDR is tonemapped to show on this SDR screen'
    : 'detail changes are subtle when the frame is shrunk to fit') + ' · click an image for 1:1 pixels';
  // Auto-upgrade: the quick preview skipped the slow RTX passes for an instant first paint. Now that it's
  // on screen, render the full RTX (VSR/HDR) version of the SAME frame so the pane ends on the real result
  // without the user needing to touch a control - their RTX picks persist across sessions, so many never
  // would. One-shot (the follow-up render isn't lite, so rtxSkipped is false there and it can't loop);
  // skipped if the user already queued another render, which supersedes it.
  if(rtxSkipped && prevPending == null && !prevBusy) loadPreview(prevIdx, true);
}
function randFrame(){ return prevTotal > 1 ? Math.floor(Math.random() * prevTotal) : 'mid'; }
function refreshPreviewIfOpen(){           // re-render the shown frame when sharpen / HDR changes
  if(!input) return;
  syncPreview();
  const frame = prevHist.length ? prevHist[prevHist.length - 1] : 'mid';
  const key = JSON.stringify([input, frame].concat(prevSettingsSig()));
  // Skip when the effective settings already match what is shown (or what is mid-render), so the
  // checkbox handler and checkRtxReady's async refresh don't double-render the same state.
  if(key === (prevBusy ? inflightPrevKey : lastPrevKey)) return;
  loadPreview(frame);
}
// Show the preview panel with a spinning wheel on both panes the INSTANT a video is selected, before
// the slow preview render is even spawned, so the panel reads as "loading" rather than blank/crashed.
// A neutral note replaces the usual result caption until the first frame lands.
function showPreviewInitializing(){
  syncPreview();                                  // reveal the preview panel
  $('hdrpreview').style.display = 'block';
  for(const [inner, wrap, spin] of [['prevorig','prevorigwrap','prevorigspin'],
                                     ['prevproc','prevprocwrap','prevspin']]){
    if($(inner).clientHeight < 40) $(wrap).style.minHeight = '120px';   // room for the wheel on first load
    setSpin($(wrap), $(spin), true);
  }
  $('prevnote').textContent = 'Initializing preview…';
}
function openPreviewForVideo(){            // auto-load the pane when a video is selected
  syncPreview(); prevHist = []; prevTotal = 0; prevIdx = 0; updatePrevNav();
  previewLite = true;   // the auto-preview on load skips the heavy RTX VSR/HDR passes so the drop is fast
  loadPreview('mid').then(() => { if(prevTotal && !prevHist.length) prevHist = [prevIdx]; updatePrevNav(); });
}
$('prevnext').onclick = async () => { await loadPreview(randFrame()); prevHist.push(prevIdx); updatePrevNav(); };
$('prevprev').onclick = async () => { if(prevHist.length > 1){ prevHist.pop(); updatePrevNav(); await loadPreview(prevHist[prevHist.length - 1]); } };
$('sharpval').addEventListener('change', refreshPreviewIfOpen);
$('sharpen').addEventListener('change', refreshPreviewIfOpen);
$('restore').addEventListener('change', () => { refreshPreviewIfOpen(); try{ lvModelUi(); lvSendOpts(); }catch{} });
// The interp-off output name tags the active pass (_restored/_dlss5/_sharpened/_hdr), so it must
// re-resolve when any of them toggles (refresh() is a no-op until a video is selected).
for(const _id of ['sharpen', 'sharpval', 'restore', 'rtxhdr', 'dlssnr'])
  $(_id).addEventListener('change', () => refresh());

// RTX HDR colour controls: Colour mode (Faithful = vivid 1.0, the accurate default; NVIDIA vivid =
// the rtx mode where the Saturation slider behaves like real RTX TrueHDR but hue-corrected), plus a
// Vibrance boost for muted colours. The engine's raw mode (the unmodified model output, cyan cast
// and all) is deliberately NOT offered here - it is a known-defective rendition kept at the CLI
// (--hdr-color raw) as a debug/reference only.
function hdrColorPayload(){
  // The RTX HDR Saturation slider spans true-to-source (-100) to NVIDIA-max (+100), always
  // hue-corrected: above -100 it drives NVIDIA's TrueHDR saturation via the rtx colour mode.
  // At -100 (SDK 0) it routes to the exact source-chroma path: the rtx safety floor (per-pixel
  // max against the source) would otherwise leave a small excess (~0.33/255 mean pixel diff).
  const sat = sdkSat();
  if(sat > 0) return { color: 'rtx', saturation: sat };
  return { color: 'vivid', saturation: 0 };   // Saturation at -100 = true to source, exact
}
// The Dynamic Vibrance sliders belong to that feature: both are 0 when it is switched off, and
// the feature at 0/0 is identity (NVIDIA convention, verified in the App).
function effVibrance(){ return $('hdrdynvib').checked ? (+$('hdrvib').value || 0) / 100 : 0; }
function effSatBoost(){ return $('hdrdynvib').checked ? (+$('hdrsb').value || 0) / 100 : 0; }
function syncHdrColor(){
  const showHdr = $('rtxhdr').checked && rtxReady.hdr && !(info && info.srcHdr);
  $('hdrtonewrap').style.display = showHdr ? 'inline-flex' : 'none';  // RTX HDR's own Contrast + Saturation
  $('hdrpurewrap').style.display = showHdr ? '' : 'none';   // the pure-conversion recipe under the sliders
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
function sdkCon(){ return 100 + (+$('hdrcon').value || 0); }
function sdkSat(){ return 100 + (+$('hdrsat').value || 0); }
// Dynamic Vibrance ALWAYS starts off: it is a per-session opt-in
// effect, so its on/off state is deliberately NOT persisted (the two slider values below
// still are). The old 'dynVib' key is retired so no stale state lingers.
localStorage.removeItem('dynVib');
const savedHdrSat = localStorage.getItem('hdrSatApp');   // App scale -100..100 (older keys orphaned)
if(savedHdrSat !== null) $('hdrsat').value = savedHdrSat;
const savedHdrCon = localStorage.getItem('hdrCon'); if(savedHdrCon !== null) $('hdrcon').value = savedHdrCon;
const savedHdrSB = localStorage.getItem('hdrSB'); if(savedHdrSB !== null) $('hdrsb').value = savedHdrSB;
const savedHdrVib = localStorage.getItem('hdrVib'); if(savedHdrVib !== null) $('hdrvib').value = savedHdrVib;
$('hdrdynvib').onchange = () => { syncHdrColor(); refreshPreviewIfOpen(); try{ lvSendOpts(); }catch{} };
$('hdrsat').oninput = () => { localStorage.setItem('hdrSatApp', $('hdrsat').value); $('hdrsatnum').textContent = $('hdrsat').value; try{ lvSendOpts(); }catch{} };
$('hdrsat').addEventListener('change', refreshPreviewIfOpen);
$('hdrcon').oninput = () => { localStorage.setItem('hdrCon', $('hdrcon').value); $('hdrconnum').textContent = $('hdrcon').value; try{ lvSendOpts(); }catch{} };
$('hdrcon').addEventListener('change', refreshPreviewIfOpen);
$('hdrsb').oninput = () => { localStorage.setItem('hdrSB', $('hdrsb').value); $('hdrsbnum').textContent = $('hdrsb').value; try{ lvSendOpts(); }catch{} };
$('hdrsb').addEventListener('change', refreshPreviewIfOpen);
$('hdrvib').oninput = () => { localStorage.setItem('hdrVib', $('hdrvib').value); $('hdrvibnum').textContent = $('hdrvib').value; try{ lvSendOpts(); }catch{} };
$('hdrvib').addEventListener('change', refreshPreviewIfOpen);
syncHdrColor();

// 1:1 zoom: the pane shrinks 1080p+ frames ~6x, which visually erases sharpening, so clicking either
// image toggles both to native pixels inside scrollable boxes with mirrored scrolling for A/B compare.
let prevZoom = false, prevSyncing = false;
function applyPrevZoom(){
  for(const [w, i] of [['prevorigwrap', 'prevorig'], ['prevprocwrap', 'prevproc']]){
    const wrap = $(w), img = $(i);
    wrap.style.overflow = prevZoom ? 'auto' : 'hidden';
    wrap.style.maxHeight = prevZoom ? '320px' : '';
    img.style.width = prevZoom ? 'auto' : '100%';
    img.style.maxWidth = prevZoom ? 'none' : '';
    img.style.cursor = prevZoom ? 'zoom-out' : 'zoom-in';
  }
}
function prevScrollSync(src, dst){
  if(prevSyncing) return; prevSyncing = true;
  // Proportional, not pixel-absolute: the two panes can hold different native sizes (an
  // upscale or Restore preview's processed side is a larger image than the source), so only
  // mirroring the scroll FRACTION keeps both panes on the same region of the scene. The <1px
  // skip stops the async echo event from ping-ponging rounding jitter between the panes.
  const fx = src.scrollLeft / Math.max(1, src.scrollWidth - src.clientWidth);
  const fy = src.scrollTop / Math.max(1, src.scrollHeight - src.clientHeight);
  const nl = fx * Math.max(0, dst.scrollWidth - dst.clientWidth);
  const nt = fy * Math.max(0, dst.scrollHeight - dst.clientHeight);
  if(Math.abs(dst.scrollLeft - nl) >= 1) dst.scrollLeft = nl;
  if(Math.abs(dst.scrollTop - nt) >= 1) dst.scrollTop = nt;
  prevSyncing = false;
}
function setSpin(wrap, spin, on){
  // Sized/positioned to the VISIBLE window of the scroll box, so the wheel stays centred in view
  // when the pane is zoomed 1:1 and scrolled (an inset-0 overlay would sit at the scroll origin).
  if(!on){ spin.style.display = 'none'; return; }
  spin.style.left = wrap.scrollLeft + 'px'; spin.style.top = wrap.scrollTop + 'px';
  spin.style.width = wrap.clientWidth + 'px'; spin.style.height = wrap.clientHeight + 'px';
  spin.style.display = 'flex';
}
function repositionSpins(){          // keep a visible wheel glued to the viewport while scrolling
  for(const [w, s] of [['prevorigwrap', 'prevorigspin'], ['prevprocwrap', 'prevspin']]){
    const spin = $(s); if(spin.style.display !== 'none') setSpin($(w), spin, true);
  }
}
$('prevorigwrap').addEventListener('scroll', () => { prevScrollSync($('prevorigwrap'), $('prevprocwrap')); repositionSpins(); });
$('prevprocwrap').addEventListener('scroll', () => { prevScrollSync($('prevprocwrap'), $('prevorigwrap')); repositionSpins(); });
function togglePrevZoom(e){
  // Capture the clicked spot as a fraction of the displayed image BEFORE the zoom restyle
  // changes the layout, then centre the 1:1 view on that spot (clicking the dragon's head
  // zooms to the dragon's head, not the frame middle). The other pane follows via the
  // proportional scroll sync, called directly so it lands this frame, not on the async event.
  const img = e.currentTarget;
  const fx = e.offsetX / Math.max(1, img.clientWidth);
  const fy = e.offsetY / Math.max(1, img.clientHeight);
  prevZoom = !prevZoom; applyPrevZoom();
  if(prevZoom){ const w = img.parentElement;
    w.scrollLeft = fx * img.clientWidth - w.clientWidth / 2;
    w.scrollTop = fy * img.clientHeight - w.clientHeight / 2;
    prevScrollSync(w, w === $('prevorigwrap') ? $('prevprocwrap') : $('prevorigwrap')); }
  repositionSpins();
}
$('prevorig').onclick = togglePrevZoom;
$('prevproc').onclick = togglePrevZoom;

// Interpolation on/off is expressed by the MODEL GROUP itself: any model ticked = on, none ticked
// = off. With it off the app only re-encodes (sharpen, upscale, HDR and restore still apply), so
// the fps and screen-rate controls grey out and the output keeps the source frame rate.
// The choice, including the "none" state, persists on the interpModel key.
function syncInterp(){
  const on = interpOn();
  // The model boxes themselves stay ENABLED always - they ARE the toggle. Only their dependent
  // sub-options and setup hints follow the on/off state.
  $('rifedrba').disabled = !on;
  if(!on){
    $('frucsetup').style.display = 'none';
    $('rifedrbarow').style.display = 'none';
  }
  applyScreenFps();      // recomputes multi/fpsin enable, the fps mode, and the output name
  try{ lvModelUi(); lvSendOpts(); }catch{}   // live follows the model choice
}
syncInterp();

// Drop anything onto the window: a video loads for processing, a tool archive installs a library.
// Routing by filename since all three installers take a .zip: an hdr10plus_tool release goes to the
// HDR10+ installer, a dovi_tool release (its name, or a bare .exe) to the Dolby Vision installer;
// any other .zip is treated as the RTX Video SDK. All drops are ignored while a job is running.
document.addEventListener('dragover', (e) => { e.preventDefault(); if($('cancel').disabled) document.body.classList.add('drag'); });
document.addEventListener('dragleave', (e) => { if(e.relatedTarget === null) document.body.classList.remove('drag'); });
document.addEventListener('drop', (e) => {
  e.preventDefault(); document.body.classList.remove('drag');
  if(!$('cancel').disabled) return;            // a job is in progress
  const files = [...(e.dataTransfer.files || [])];
  if(!files.length) return;
  const nm = files[0].name, p = webUtils.getPathForFile(files[0]);
  if(/nvngx_dlssnr\.dll$/i.test(nm) || (/\.zip$/i.test(nm) && /dlss[ _-]?5|dlssnr/i.test(nm))){   // the DLSS 5 runtime: install into engine/dlssnr
    showWorkspace();
    $('dlssnr').checked = true; localStorage.setItem('dlssnrOn', '1'); syncDlssnr();   // engage the row so its setup box is coherent
    doDlssnrInstall(p);
  } else if(/hdr10plus_tool/i.test(nm)){       // the hdr10plus_tool download: install into engine/hptools
    showWorkspace();
    $('hpexport').checked = true; localStorage.setItem('hpOn', '1');   // engage the HDR10+ panel so its setup box is coherent
    $('hpsetup').style.display = 'block';       // reveal install feedback
    doHpInstall(p);
  } else if(/dovi_tool/i.test(nm) || /\.exe$/i.test(nm)){   // the dovi_tool download: install into engine/dvtools
    showWorkspace();
    $('dvexport').checked = true; localStorage.setItem('dvOn', '1');   // engage the DV panel so its setup box is coherent
    $('dvsetup').style.display = 'block';       // reveal install feedback
    doDvInstall(p);
  } else if(/\.zip$/i.test(nm)){               // otherwise a .zip is the RTX Video SDK: install the RTX runtime
    showWorkspace();   // RTX install feedback lives in #workspace, so reveal it even before a video is loaded
    if($('rtxsetup').style.display === 'none') $('rtxsetup').style.display = 'block';   // reveal install feedback
    doInstall(p);
  } else {
    // Anything else: load it as a video; several dropped files become a batch queue. Both the
    // Select video button and the file rows are hidden in Live mode, but a drop still fires, so
    // switch to Video mode first or the loaded file would land on an invisible workspace.
    setMode('video');
    queueVideos(files.map(f => webUtils.getPathForFile(f)));
  }
});

// Resume an interrupted batch: if the previous session died mid-run (crash, forced close), its
// unfinished inputs are still in localStorage (saveBatch). Re-queue them now - the first file loads
// and the rest wait - but do NOT auto-start; the user clicks Smooth It! when ready. The key is
// cleared immediately so a failure during this restore can't loop; startRun re-saves it.
try {
  const pend = JSON.parse(localStorage.getItem('batchPending') || 'null');
  localStorage.removeItem('batchPending');
  if(Array.isArray(pend) && pend.length && pend.every(p => typeof p === 'string')){
    log('>> Restored ' + pend.length + ' unfinished file(s) from the last session; an interrupted render resumes where it left off\n');
    queueVideos(pend);
  }
} catch { /* no saved batch */ }

function startRun(){
  if(!input) return;
  const t = targetFps();
  if(!fpsValid(t)){ alert('Enter a target fps between ' + sliderMin() + ' and ' + sliderMax() + '.'); return; }
  // 1x (target fps == source fps) adds no frames - it just re-generates every frame as a midpoint tween
  // (softer, half-frame shift), a strictly worse video. Block it with guidance instead of running it.
  if(interpOn() && Math.round(t) <= Math.round(srcFps())){
    alert('A 1× target adds no frames (output fps = source fps). Raise the target above ' + Math.round(srcFps())
        + ' fps to smooth, or turn off Interpolate to just sharpen / re-encode.'); return; }
  // DLSS-FG only generates the evenly spaced frames between consecutive frames: whole 2x-6x
  // multipliers on the source grid (multi-frame generation). There is no timestep to ask for
  // anything else, so block with guidance (mirrors the 1x block). The per-GPU ceiling (RTX 40 =
  // 2x only) is enforced by the engine at render start with its own clear error.
  if(interpOn() && modelIsDlss() && (gridMulti() < 2 || gridMulti() > 6)){
    alert('DLSS 4.5 interpolates at whole multipliers from 2× to 6× of the source fps ('
        + (2*srcFps()).toFixed(targetDecimals()) + '–' + (6*srcFps()).toFixed(targetDecimals())
        + ' fps for this video). Pick one of those, or GMFSS for any other target.'); return; }
  const sharpenStrength = $('sharpen').checked ? parseFloat($('sharpval').value) : 0;
  const dims = upDims();                               // resize target ({w,h} or null)
  const factor = upFactor();                           // arbitrary resize factor (0 = off, <1 downscales)
  const useRtxVsr = factor > 1 && $('rtxvsr').checked && rtxReady.vsr;   // AI upscale, else bicubic; never VSR on a downscale
  const rtxhdr = hdrOn();  // HDR only when its runtime is installed and the source is SDR
  // Interpolation is the main effect; with it off the run still has work if sharpening, upscaling or
  // HDR is on. Bail only when nothing at all is enabled.
  if(!interpOn() && sharpenStrength <= 0 && factor <= 0 && !rtxhdr && !restoreOn() && !nrOn()){
    alert('Nothing to do: turn on Interpolate, pick an Upscale resolution, enable FSR sharpening, Restore or DLSS 5, or turn on RTX HDR (and install its runtime).');
    return;
  }
  procStart = 0; baseK = 0; etaAnchor = 0; etaAnchorTime = 0; lastFps = 0; curDone = 0; curPct = 0; projBytes = 0;
  clearInterval(etaTimer); etaTimer = setInterval(paint, 500);   // ticks the ETA down between frames
  // Live thumbnail: only frames written by THIS run may show, so anchor the mtime watermark to
  // whatever is on disk now (a previous run's leftover) and hide the old image until a new write.
  try { liveM = _fs.statSync(livePath).mtimeMs; } catch { liveM = 0; }
  $('live').style.display = 'none';
  // Reveal the live area (image appears on the first written frame) so its Hide/Show toggle is
  // reachable for the whole run, and re-assert the engine-side on/off to match the saved choice.
  $('livewrap').style.display = 'flex';
  applyLiveOff();
  clearInterval(liveTimer); liveTimer = setInterval(updateLive, 1000);
  fpp = outRatio();                                    // output frames produced per source frame (1 when not interpolating)
  const srcFrames = info.nb || Math.round(info.dur*srcFps()) || 1;
  totalGen = Math.max(1, Math.round(srcFrames*fpp));   // total frames the conversion will output
  cancelled = false;
  // #go stays enabled and becomes the Pause toggle for the duration of the run (see goMode).
  goMode = 'running'; $('go').textContent = 'Pause';
  $('go').title = 'Pauses the render. You can also just close the app - the render resumes where it left off next time.';
  if(!resumeTipShown){
    resumeTipShown = true;   // once per session; batch auto-advance re-enters startRun per file
    log('>> Tip: you can Pause, or even close the app mid-render: the render resumes where it left off\n');
  }
  modeBtnUi(true);
  $('pick').disabled = true; $('changeout').disabled = true; $('out').disabled = true; for(const b of MODEL_BOXES()) b.disabled = true; $('fpsin').disabled = true; $('sharpen').disabled = true; $('sharpval').disabled = true; $('restore').disabled = true; $('nvorder').disabled = true; $('dlssnr').disabled = true; $('nrstructure').disabled = true; $('nrtone').disabled = true; for(const b of $('nrstyleseg').querySelectorAll('button')) b.disabled = true; $('nrmask').disabled = true; $('upres').disabled = true; $('upcustom').disabled = true; $('outcodec').disabled = true; $('rtxvsr').disabled = true; $('rtxhdr').disabled = true; $('hdrdynvib').disabled = true; $('hdrsat').disabled = true; $('hdrvib').disabled = true; $('hdrcon').disabled = true; $('hdrsb').disabled = true; $('open').disabled = true; $('play').disabled = true; $('cancel').disabled = false; $('playprev').disabled = true; $('playprev').style.display = 'none'; lastPreview = null; dlssPreempt = null; syncTargetUI();
  // What this run does, for the status / log (mainly relevant when interpolation is off).
  const passes = []; if(restoreOn()) passes.push('restoring'); if(factor > 0) passes.push(factor < 1 ? 'downscaling' : 'upscaling'); if(rtxhdr) passes.push('HDR'); if(sharpenStrength > 0) passes.push('sharpening');
  const offLabel = (passes.join(' + ') || 'processing').replace(/^./, c => c.toUpperCase()) + '...';
  $('status').textContent = fileTag() + (interpOn() ? 'Warming up the model...' : offLabel);
  $('fill').style.width = '0%'; $('pct').textContent = '0%';
  $('frames').textContent = interpOn() ? 'Warming up (loading model, first frames)...' : offLabel;
  const gm = gridMulti();
  const rr = t/srcFps();
  const ratioTxt = gm ? gm+'×, on-grid' : (rr < 9.95 ? rr.toFixed(1)+'×, resampled' : '≈'+Math.round(rr)+'×, resampled');
  const what = interpOn() ? (t.toFixed(targetDecimals())+' fps ('+ratioTxt+')') : 'no interpolation';
  const extra = (dims ? '  ·  '+(useRtxVsr ? 'RTX VSR' : factor < 1 ? 'downscale' : 'upscale')+' '+dims.w+'×'+dims.h : '')
              + (rtxhdr ? '  ·  RTX HDR' : '')
              + (sharpenStrength>0 ? '  ·  FSR '+sharpenStrength : '')
              + (restoreOn() ? '  ·  Restore' : '');
  log('>> '+input+'  ->  '+what+extra+'\n');
  const payload = { input, multi: gm || 2, output: lastOut, interp: interpOn() };   // on-grid: the derived integer multiplier; resample: --fps overrides this placeholder
  if(interpOn() && !gm) payload.fps = t;                                             // resample only: non-integer-multiple target
  if(interpOn() && modelIsRifeDrba()) payload.model = 'rifedrba'; // RIFE with the DRBA sub-option on
  else if(interpOn() && modelIsRife()) payload.model = 'rife';     // plain RIFE (bundled, always ready)
  else if(interpOn() && modelIsDlss()) payload.model = 'dlssg';   // "DLSS 4.5" (Frame Generation) backend
  else if(interpOn() && modelIsFruc()) payload.model = 'fruc';   // "NVIDIA Smooth Motion" (NvOFFRUC) backend
  else if(interpOn() && modelIsLsfg()) payload.model = 'lsfg'; // Frame Blend (flow-warp, engine --lsfg)
  else if(interpOn() && modelIsNvof()) payload.model = 'nvof'; // NVIDIA Optical Flow (direct), engine --nvof
  payload.sharpen = sharpenStrength;   // 0 = engine leaves frames untouched
  if(restoreOn()) payload.restore = true;   // AI detail restoration (Real-ESRGAN animevideov3)
  if(nvOrderOn()) payload.nvorder = true;   // NVIDIA order: Restore and the upscale before DLSS 5 and the model
  if(nrOn()){ payload.dlssnr = true; payload.nrstructure = nrStructure(); payload.nrtone = nrTone(); payload.nrstyle = nrStyle(); }   // DLSS 5 (runtime installed)
  payload.codec = $('outcodec').value; // output codec family (hevc default / av1 / vvc)
  if(factor > 0){
    payload.upscale = factor;                          // arbitrary upscale factor (target height / source)
  }
  if(useRtxVsr) payload.rtxvsr = true;                 // AI upscale via the RTX Video SDK (else bicubic)
  if(rtxhdr){ payload.rtxhdr = true;     // HDR10; engine masters at a fixed 1000-nit peak
    const hp = hdrColorPayload();        // zero-strength Dynamic Vibrance routes to the source path
    payload.hdrcolor = hp.color;                                   // vivid (default) / rtx
    payload.hdrsat = hp.saturation;                                // SDK Saturation (rtx mode)
    payload.hdrcon = sdkCon();                                     // SDK Contrast (100 = neutral)
    payload.hdrsb = effSatBoost();                                 // vibrance uniform gain (0..1)
    payload.hdrvib = effVibrance();
    if($('dvexport').checked && dvReady.ready) payload.dv = true;   // also export Dolby Vision 8.1 (needs HDR)
    if($('hpexport').checked && hpReady.ready) payload.hp = true;   // also embed HDR10+ metadata (needs HDR)
  }
  // Image scale slider (shared with Live): the engine processes the whole video at the
  // reduced size and its upscale pass restores the output size (GMFSS default, RIFE, DRBA,
  // Frame Blend).
  const flowPct = +$('lvflow').value;
  if(interpOn() && flowPct < 100 && (!payload.model || payload.model === 'rife' || payload.model === 'rifedrba'
                                     || payload.model === 'lsfg'))
    payload.flowscale = flowPct;
  saveBatch([input, ...batch]);   // survives a crash/close: the next launch re-queues these files
  ipcRenderer.send('run', payload);
}
// #go dispatches on its current mode: start a fresh run when idle, otherwise toggle Pause/Resume.
$('go').onclick = () => {
  if(goMode === 'idle'){ startRun(); return; }
  if(goMode === 'running'){
    // Pause: the engine finishes the frames already queued to the encoder, then holds before the
    // next source pair. Stop the ETA countdown (it would keep ticking down through the idle time).
    goMode = 'paused'; $('go').textContent = 'Resume';
    $('go').title = 'Continues rendering now. Closing the app instead is also fine - the render resumes where it left off next time.';
    $('status').textContent = fileTag() + 'Paused; click Resume to continue (or close the app and resume another time).';
    clearInterval(etaTimer); etaTimer = null;
    ipcRenderer.send('pause');
  } else { // paused -> resume
    goMode = 'running'; $('go').textContent = 'Pause';
    // The pause preview describes the paused-at point; once we resume it is stale, so hide it (the
    // engine overwrites it on the next Pause and deletes it at finalize).
    $('playprev').style.display = 'none'; $('playprev').disabled = true; lastPreview = null;
    $('go').title = 'Pauses the render. You can also just close the app - the render resumes where it left off next time.';
    // Re-anchor the pace clock so the paused span doesn't drag the mean fps (and ETA) down; skip
    // if warmup never finished (procStart still 0; the first-frame handler will anchor instead).
    if(procStart){ procStart = Date.now(); baseK = curDone; }
    etaTimer = setInterval(paint, 500);
    $('status').textContent = fileTag() + 'Smoothing on the GPU...';
    ipcRenderer.send('resume');
  }
};
$('open').onclick = () => { if(lastOut) shell.showItemInFolder(lastOut); };
$('play').onclick = () => { if(lastOut) shell.openPath(lastOut); };   // system default player
$('playprev').onclick = () => { if(lastPreview) shell.openPath(lastPreview); };  // partial preview, sys player
$('changeout').onclick = async () => {
  const chosen = await ipcRenderer.invoke('pick-output', customOut || outName());
  if(chosen){ customOut = chosen; refresh(); }
};
$('cancel').onclick = async () => {
  cancelled = true; $('cancel').disabled = true; $('status').textContent = 'Cancelling...';
  // The bar and live thumbnail describe progress that is being deleted: reset them right away
  // (engine-out drops any PROGRESS still flushing from the dying engine while cancelled, so
  // nothing can repaint the bar between this reset and engine-done).
  clearInterval(etaTimer); etaTimer = null;
  $('fill').style.width = '0%'; $('pct').textContent = '0%'; $('frames').textContent = '';
  $('livewrap').style.display = 'none'; $('live').style.display = 'none';
  // invoke resolves only after main killed the engine AND deleted the .part/resume artifacts,
  // so the probe below sees the truth (engine-done skips its own probe on a cancelled run).
  try { await ipcRenderer.invoke('cancel'); } catch {}
  scheduleCheckResume();
};

ipcRenderer.on('engine-out', (_e, t) => {
  // Pause partial-preview signals: on Pause the engine remuxes the frames rendered so far with the
  // source's audio/subtitles into a playable file and logs PREVIEW_READY <path> (or PREVIEW_PENDING
  // when too few frames have flushed yet). Surface a "Play preview" button so the user can judge the
  // result with sound before resuming or cancelling.
  const pr = /PREVIEW_READY\s+([^\r\n]+)/.exec(t);
  if(pr){
    lastPreview = pr[1].trim();
    $('playprev').disabled = false; $('playprev').style.display = '';
    $('status').textContent = fileTag() + 'Paused. A preview of the part rendered so far (with sound & subtitles) is ready - click "Play preview".';
  } else if(/PREVIEW_PENDING/.test(t)){
    $('status').textContent = fileTag() + 'Paused; preview not ready yet (too few frames rendered) - Resume, then Pause again a bit later.';
  }
  // DLSS-FG was preempted (RTX Video enhancement) and stopped after its restarts, but banked a clean
  // resumable checkpoint. Remember the % so engine-done presents it as a resumable stop, not a crash.
  const dp = /DLSS_PREEMPTED\s+(\d+)/.exec(t);
  if(dp) dlssPreempt = +dp[1];
  // The engine emits its EXACT output frame count as OUTFRAMES at the end; it's authoritative over the
  // pre-render estimate (container nb_frames * ratio), which can be off by one when nb_frames misreports.
  const of = /OUTFRAMES\s+(\d+)/.exec(t);
  if(of) totalGen = +of[1] || totalGen;
  // SIZE lines carry (bytes so far, projected final bytes); like PROGRESS, a chunk can batch
  // several, so keep the newest projection for the frames line.
  let sm = null, sg = /SIZE\s+\d+\s+(\d+)/g, sx;
  while((sx = sg.exec(t)) !== null) sm = sx;
  if(sm) projBytes = +sm[1] || 0;
  // A single stderr chunk can batch several PROGRESS lines; take the last so the bar and the
  // frame/elapsed feeding the ETA reflect the newest count, never a stale one from the batch.
  let m = null, g = /PROGRESS\s+(\d+)\/(\d+)/g, x;
  while((x = g.exec(t)) !== null) m = x;
  if(m && !cancelled){ const done=+m[1], tot=+m[2]||1;
    // Clamp to [0,1]: a missing/zero denominator (e.g. an MKV with no nb_frames) would
    // otherwise make frac huge and send the bar past 100% (the runaway 9999999% bug).
    const frac = Math.min(1, Math.max(0, done/tot));
    const pct = 100*frac;                              // 0% at first generated frame -> 100% at last
    $('fill').style.width = pct+'%';
    $('pct').textContent = Math.round(pct)+'%';          // the percentage now lives inside the bar
    curDone = done; curPct = pct;
    const now = Date.now();
    if(!procStart){
      // First generated frame is in, so warmup (model load + first inference) is over. Anchor the
      // clock here so the average pace excludes that one-time startup cost and this first frame.
      procStart = now; baseK = done;
      $('status').textContent = fileTag() + 'Smoothing on the GPU...';
      paint();
      return;
    }
    // ETA from the mean pace over EVERY frame since warmup (not a trailing window): all frames
    // generated since the anchor over all time since the anchor. Steady and self-correcting as the
    // run goes on; the 0.5s timer counts the displayed seconds down between PROGRESS frames.
    const elapsed = (now-procStart)/1000, generated = done-baseK;
    if(elapsed > 0 && generated > 0){
      lastFps = generated*fpp/elapsed;                 // mean output frames/sec across the whole run
      etaAnchor = lastFps > 0 ? (tot-done)*fpp/lastFps : 0;
      etaAnchorTime = now;
    }
    paint();
  } else if(!m) {   // PROGRESS chunks are never logged; while cancelled they are dropped entirely
    if(/\[trt\] building/.test(t)) $('status').textContent = 'Building TensorRT engines (one time for this resolution)...';
    // OUTFRAMES is machine-only, and the engine's "done ..." line duplicates the ">> Saved: <path>"
    // the GUI prints on engine-done - strip both from the GUI log (they still show in CLI output).
    const shown = t.replace(/OUTFRAMES\s+\d+\n?/g, '').replace(/SIZE\s+\d+\s+\d+\n?/g, '')
      .replace(/^done \d+ (frames|pairs)[^\n]*\n?/gm, '');
    if(shown.trim() && !/Optical Flow Grid Size/.test(shown)) log(shown);
  }
});
// Launch-time update notice (main's checkForUpdate): a one-line link under the subtitle; the check
// is silent unless a newer GitHub release exists, so this stays hidden for up-to-date installs.
ipcRenderer.on('update-available', (_e, u) => {
  if(!u || !u.version) return;
  $('updatever').textContent = u.version;
  $('updatelink').onclick = (ev) => { ev.preventDefault(); shell.openExternal(u.url); };
  $('update').style.display = 'block';
});

ipcRenderer.on('engine-done', (_e, code) => {
  // Reset the Pause/Resume toggle back to a fresh "Smooth It!" first, so every exit path
  // (done / cancelled / failed) restores it and the batch auto-advance below re-enters startRun.
  goMode = 'idle'; $('go').textContent = 'Smooth It!'; $('go').title = '';   // checkResume below re-fills the tooltip
  // The run ended (done/cancelled/failed): the engine deletes the partial preview at finalize, so hide
  // the "Play preview" button - the finished "Play video" takes over on success.
  $('playprev').style.display = 'none'; $('playprev').disabled = true; lastPreview = null;
  clearInterval(etaTimer); etaTimer = null;            // stop the ETA countdown
  // Final thumbnail pull (the last written frame), except on cancel: the click hid the preview.
  clearInterval(liveTimer); liveTimer = null; if(!cancelled) updateLive();
  modeBtnUi(lvState !== 'idle');
  $('go').disabled=false; $('pick').disabled=false; $('changeout').disabled=false; $('out').disabled=false; for(const b of MODEL_BOXES()) b.disabled=false; $('fpsin').disabled=false; $('sharpen').disabled=false; $('sharpval').disabled=false; $('restore').disabled=false; $('nvorder').disabled=false; $('dlssnr').disabled=false; $('nrstructure').disabled=false; $('nrtone').disabled=false; for(const b of $('nrstyleseg').querySelectorAll('button')) b.disabled=false; $('nrmask').disabled=false; $('upres').disabled=false; $('upcustom').disabled=false; $('outcodec').disabled=false; $('rtxvsr').disabled=false; $('rtxhdr').disabled=!!(info&&info.srcHdr); $('hdrdynvib').disabled=false; $('hdrcon').disabled=false; syncHdrColor(); $('cancel').disabled=true;
  syncInterp();         // re-assert the interp / screen-rate state after the run re-enabled the inputs
  if(cancelled){ $('status').textContent='Cancelled.'; log('>> Cancelled\n');
    if(batch.length) log('>> Batch cleared ('+batch.length+' queued files not processed)\n');
    batch = []; batchTotal = 1; batchFailed = []; saveBatch(null); }
  else if(code===0){ $('fill').style.width='100%'; $('pct').textContent='100%'; $('frames').textContent='Frames: '+totalGen+' / '+totalGen;
    $('status').textContent='Done!'; $('open').disabled=false; $('play').disabled=false; log('>> Saved: '+lastOut+'\n');
    if(batch.length) advanceBatch(); else finishBatch(); }
  else if(dlssPreempt !== null){
    // Not a failure: DLSS-FG was preempted by RTX Video enhancement and stopped after its restarts,
    // but the render banked a clean, fully-resumable checkpoint. Present it as a resumable stop; the
    // checkResume below turns #go into "Resume" so the user closes the video and continues.
    $('status').textContent = 'Stopped: DLSS was preempted (RTX Video). ~'+dlssPreempt+'% saved - close the video and click Resume.';
    log('>> DLSS Frame Generation was preempted (RTX Video enhancement); ~'+dlssPreempt+'% saved and resumable. Close the interfering video, then Resume (or use GMFSS).\n');
    if(batch.length){ log('>> Continuing with the next queued file\n'); advanceBatch(); }
    else if(batchTotal > 1) finishBatch();
    else saveBatch(null); }
  else { $('status').textContent='Failed (exit '+code+'). See log.'; log('>> Failed (exit '+code+')\n');
    // A failed file no longer kills the rest of an unattended batch: note it and keep going.
    if(batchTotal > 1) batchFailed.push(input);
    if(batch.length){ log('>> Continuing with the next queued file\n'); advanceBatch(); }
    else if(batchTotal > 1) finishBatch();
    else saveBatch(null); }
  // A finished/failed run changes resumability; re-label #go. Cancelled runs are probed by the
  // Cancel click handler instead, after main's artifact cleanup resolves (probing here would
  // still see the resume files and wrongly announce "Interrupted render found").
  if(!cancelled) scheduleCheckResume();
});
// Batch auto-advance: load the next queued file and start it with the same settings. batchAuto
// suppresses the modal probe alert and the preview refresh while the queue advances unattended;
// an unreadable file counts as failed and the queue keeps moving.
function advanceBatch(){
  const next = batch.shift();
  saveBatch([next, ...batch]);
  batchAuto = true;
  loadVideo(next).then(ok => { batchAuto = false;
    if(ok) $('go').onclick();
    else { batchFailed.push(next);
      if(batch.length) advanceBatch(); else finishBatch(); } });
}
// Whole queue drained: per-batch summary (incl. failures), native notification, reset + unpersist.
function finishBatch(){
  const fails = batchFailed.length;
  if(batchTotal > 1){
    $('status').textContent = 'Done! ' + (batchTotal - fails) + '/' + batchTotal + ' files processed'
      + (fails ? ', ' + fails + ' failed (see log)' : '.');
    if(fails) log('>> Failed files:\n' + batchFailed.map(f => '>>   ' + f).join('\n') + '\n');
  }
  ipcRenderer.send('render-complete', batchTotal > 1
    ? ((batchTotal - fails) + '/' + batchTotal + ' files done' + (fails ? ', ' + fails + ' failed' : ''))
    : 'Render complete');
  batchTotal = 1; batchFailed = []; saveBatch(null);
}
