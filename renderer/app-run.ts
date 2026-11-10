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

// Drop videos onto the window: they load for processing. All drops are ignored while a job is running.
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
  // Several dropped files become a batch queue. Both the Select video button and the file rows are hidden
  // in Live mode, but a drop still fires, so switch to Video mode first or the loaded file would land on an
  // invisible workspace.
  setMode('video');
  queueVideos(files.map((f) => webUtils.getPathForFile(f)));
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
  $('nrskin').disabled = true;
  $('nrautomask').disabled = true;
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
    payload.nrskin = nrSkin();
    payload.nrautomask = nrAutoMask();
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
    // OUTFRAMES is machine-only, and the engine's "done ..." line duplicates the ">> Saved: <path>"
    // the GUI prints on engine-done - strip both from the GUI log (they still show in CLI output).
    const shown = t
      .replace(/OUTFRAMES\s+\d+\n?/g, '')
      .replace(/SIZE\s+\d+\s+\d+\n?/g, '')
      .replace(/^done \d+ (frames|pairs)[^\n]*\n?/gm, '');
    if (shown.trim() && !/Optical Flow Grid Size/.test(shown)) log(shown);
  }
});
