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
function nrSkin() {
  const v = parseFloat($('nrskin').value);
  return v >= 0 && v !== 1 ? v : -1;
} // DLSSNR.SkinStructureStrength: the slider's 1.00 (its default) = -1 = NVIDIA's default, skin follows Structure
function nrAutoMask() {
  return $('nrautomask').checked ? 1 : 0;
} // DLSSNR.UseAutoMask, NVIDIA's character mask (Skin Structure acts only with it)
function syncNrSkin() {
  $('nrskin').disabled = !$('nrautomask').checked;
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
  ['nrskin', 'nrSkin'],
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
if (localStorage.getItem('nrAutoMask') === '0') $('nrautomask').checked = false; // default ON, NVIDIA's mask
syncNrSkin();
$('nrautomask').onchange = () => {
  localStorage.setItem('nrAutoMask', $('nrautomask').checked ? '1' : '0');
  syncNrSkin();
  refreshPreviewIfOpen();
  try {
    lvSendOpts();
  } catch {}
};
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
