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
  nvvfg: 'NVIDIA Video Frame Generation',
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
        (o.value === 'xefg' && !xefgReady) ||
        (o.value === 'nvvfg' && !vfgReady);
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
  $('prevpair').style.display = on ? 'none' : 'block';
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
      nrskin: nrSkin(),
      nrautomask: nrAutoMask(),
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
