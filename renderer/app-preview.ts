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
    nrSkin(),
    nrAutoMask(),
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
          nrskin: nrSkin(),
          nrautomask: nrAutoMask(),
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
// renders' Peak box (0 = empty: the main process takes the peak of the display the window is on).
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

// 1:1 zoom: the pane shrinks 1080p+ frames, which visually erases sharpening, so clicking either image toggles
// both layers to the result's native pixels inside scrollable boxes with mirrored scrolling. The original is drawn
// at the result's size (an upscale's result is larger than its source), so the two sides of the divider cover the
// same part of the scene.
let prevZoom = false,
  prevSyncing = false;
function applyPrevZoom() {
  const procW = $('prevproc').naturalWidth;
  $('prevpair').classList.toggle('zoomed', prevZoom);
  if (procW) $('prevpair').style.setProperty('--natw', procW + 'px');
  for (const [w, i] of [
    ['prevorigwrap', 'prevorig'],
    ['prevprocwrap', 'prevproc'],
  ]) {
    const wrap = $(w),
      img = $(i);
    wrap.style.overflow = prevZoom ? 'auto' : 'hidden';
    wrap.style.maxHeight = prevZoom ? 'var(--prevh)' : '';
    img.style.width = prevZoom ? (i === 'prevorig' && procW ? procW + 'px' : 'auto') : '100%';
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
  // The wheels sit above both layers of the pane, sized to the visible window of the scroll box (without its
  // scrollbars), so a wheel stays centred in view when the pane is zoomed 1:1. The two would overlap: the
  // original's hides while the result's shows.
  spin.style.display = on ? 'flex' : 'none';
  if (on) {
    spin.style.left = '0';
    spin.style.top = '0';
    spin.style.width = wrap.clientWidth + 'px';
    spin.style.height = wrap.clientHeight + 'px';
  }
  $('prevorigspin').firstElementChild.style.visibility = $('prevspin').style.display === 'none' ? '' : 'hidden';
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
$('prevproc').addEventListener('load', () => {
  const img = $('prevproc');
  if (img.naturalWidth && img.naturalHeight) {
    $('prevpair').style.setProperty('--ar', String(img.naturalWidth / img.naturalHeight)); // the pane takes its shape
  }
  if (prevZoom) applyPrevZoom(); // a new result may have another size: the original follows it
});

// The before / after divider: --split on the pane is the share of its width that shows the original. The handle
// drags it, the arrow keys step it, Home and End jump to the sides.
function setPrevSplit(pct) {
  const v = Math.min(100, Math.max(0, pct));
  $('prevpair').style.setProperty('--split', v + '%');
  $('prevsplit').setAttribute('aria-valuenow', String(Math.round(v)));
}
function prevSplitAt(clientX) {
  const r = $('prevpair').getBoundingClientRect();
  setPrevSplit(((clientX - r.left) / Math.max(1, r.width)) * 100);
}
$('prevsplit').addEventListener('pointerdown', (e) => {
  e.preventDefault();
  $('prevsplit').setPointerCapture(e.pointerId);
  $('prevsplit').focus();
  prevSplitAt(e.clientX);
});
$('prevsplit').addEventListener('pointermove', (e) => {
  if ($('prevsplit').hasPointerCapture(e.pointerId)) prevSplitAt(e.clientX);
});
$('prevsplit').addEventListener('keydown', (e) => {
  const step = { ArrowLeft: -2, ArrowRight: 2, Home: -100, End: 100 }[e.key];
  if (!step) return;
  e.preventDefault();
  setPrevSplit((parseFloat($('prevpair').style.getPropertyValue('--split')) || 50) + step);
});
