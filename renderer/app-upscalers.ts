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
