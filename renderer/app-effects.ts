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
const MAX_FACTOR = 16; // the engine clamps the upscale factor here
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
