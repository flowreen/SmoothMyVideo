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
