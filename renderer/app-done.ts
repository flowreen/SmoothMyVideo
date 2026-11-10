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
  $('nrautomask').disabled = false;
  syncNrSkin(); // Skin Structure stays off while the Auto mask is
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
