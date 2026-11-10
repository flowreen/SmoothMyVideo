// Interpolation model: GMFSS (default) vs "NVIDIA Smooth Motion" (NVIDIA Optical Flow FRUC). Two
// checkboxes acting as an exclusive pair (user-chosen UX): ticking one clears the other, and
// unticking the active one hands the choice back to its sibling, so exactly one model is always
// selected. The FRUC runtime (NvOFFRUC.dll) is NVIDIA proprietary + user-installed from the Optical
// Flow SDK .zip. The Smooth Motion checkbox is ALWAYS clickable (ticking it visibly clears GMFSS);
// it just reveals a one-time installer when the runtime is missing, and a render only uses FRUC once
// it is installed - otherwise it falls back to GMFSS (see modelIsFruc + the run payload gate below).
let frucReady = false,
  frucBridge = false,
  dlssReady = false,
  dlssMissing = [];
let dlssDefault = false; // DLSS 4.5 is the default model: its files present, an RTX 40 / 50 GPU, GPU scheduling on
function modelIsFruc() {
  return $('modelfruc').checked && frucReady;
}
function modelIsDlss() {
  return $('modeldlss').checked && dlssReady;
}
// RIFE ships bundled (weights vendored in engine/rife), so it is always ready.
function modelIsRife() {
  return $('modelrife').checked;
}
// The DRBA sub-checkbox (default OFF) adds anime-pacing timing on top of the same RIFE weights.
function modelIsRifeDrba() {
  return modelIsRife() && $('rifedrba').checked;
}
// NVIDIA Optical Flow: the driver's optical-flow hardware run by
// smv-live.exe itself (engine --nvof, live backend nvof), so it is always ready like RIFE.
function modelIsNvof() {
  return $('modelnvof').checked;
}
// AMD FSR frame generation: our bridge ships in engine/fsrfg (engine --fsrfg, live backend fsrfg), always ready
function modelIsFsrfg() {
  return $('modelfsrfg').checked;
}
// AMD FSR 4's first session after a driver update waits ~15 to 30 s for the driver's shader compile: while an FSR route
// is picked, main warms both routes' shaders in a tiny background session once per driver (fsr-warm; a no-op when warm)
function fsrWarm() {
  const fg = modelIsFsrfg(),
    up = $('fsrup').checked;
  if (fg || up) ipcRenderer.send('fsr-warm', { fg, up });
}
// Intel XeSS-FG: our bridge ships in engine/xess (engine --xess-fg, live backend fsrfg + --xess-fg); Intel's
// libxess_fg.dll + libxell.dll come from the Intel XeSS setup's one click (xess-ready's fg), so it is ready once they are
let xefgReady = false;
function modelIsXefg() {
  return $('modelxefg').checked && xefgReady;
}
// NVIDIA Video Frame Generation: NVIDIA's Video Effects runtime in engine/nvvfx (engine --nvvfg), installed by its
// setup row's one click (nvvfg-ready), so it is ready once the four files are there (live backend nvvfg)
let vfgReady = false;
function modelIsVfg() {
  return $('modelnvvfg').checked && vfgReady;
}
const MODEL_BOXES = () => [
  $('modelgmfss'),
  $('modelrife'),
  $('modeldlss'),
  $('modelfruc'),
  $('modelfsrfg'),
  $('modelxefg'),
  $('modelnvvfg'),
  $('modelnvof'),
];
// The run payload's model ('' = GMFSS, the CLI's default, or interpolation off); Compare's model A reads it too, so a
// compare runs exactly the model a render would.
function renderModel() {
  if (!interpOn()) return '';
  return modelIsRifeDrba()
    ? 'rifedrba'
    : modelIsRife()
      ? 'rife'
      : modelIsDlss()
        ? 'dlssg'
        : modelIsFruc()
          ? 'fruc'
          : modelIsNvof()
            ? 'nvof'
            : modelIsFsrfg()
              ? 'fsrfg'
              : modelIsXefg()
                ? 'xefg'
                : modelIsVfg()
                  ? 'nvvfg'
                  : '';
}
function frucSetupHtml() {
  return tr('fruc.intro') + '<br>' + (!frucBridge ? tr('fruc.noBridge') : tr('fruc.howto'));
}
async function refreshFrucState() {
  try {
    const r = await ipcRenderer.invoke('fruc-ready');
    frucReady = !!(r && r.ready);
    frucBridge = !!(r && r.bridge);
  } catch {
    frucReady = false;
    frucBridge = false;
  }
  // DLSS ("DLSS 4.5" Frame Generation) ships bundled, so not-ready means files were deleted from
  // engine/dlssg (a broken install); GPU/driver support is only knowable at render time.
  try {
    const d = await ipcRenderer.invoke('dlssg-ready');
    dlssReady = !!(d && d.ready);
    dlssMissing = (d && d.missing) || [];
    dlssDefault = dlssReady && !!(d && d.supported);
  } catch {
    dlssReady = false;
    dlssMissing = [];
    dlssDefault = false;
  }
  await refreshXefgState();
  await refreshVfgState();
}
let vfgInfo: any = { ready: false };
async function refreshVfgState() {
  try {
    vfgInfo = await ipcRenderer.invoke('nvvfg-ready');
  } catch {
    vfgInfo = { ready: false };
  }
  vfgReady = !!(vfgInfo && vfgInfo.ready);
}
// NVIDIA Video Frame Generation's setup row: shown while its box is ticked and the runtime is missing; one click
// downloads NVIDIA's pinned package (main checks its hash, then installs the runtime's four files)
function syncVfgSetup() {
  if (!$('modelnvvfg').checked || vfgReady) {
    $('nvvfgsetup').style.display = 'none';
    return;
  }
  $('nvvfgstatus').textContent = tr('dlss.notReady');
  $('nvvfgstatus').style.color = '#e85c5c';
  $('nvvfgsteps').innerHTML =
    '✗ ' +
    tr('nvvfg.line') +
    '<br>' +
    tr('nvvfg.howto', {
      ver: vfgInfo.ver || '',
      mb: vfgInfo.mb || '415.6',
      link: toolPageLink('rtx-open-download', 'pypi.nvidia.com'),
    });
  $('nvvfgbtns').style.display = '';
  $('nvvfgsetup').style.display = 'block';
}
let xefgInfo: any = { fg: false };
async function refreshXefgState() {
  try {
    xefgInfo = await ipcRenderer.invoke('xess-ready');
  } catch {
    xefgInfo = { fg: false };
  }
  xefgReady = !!(xefgInfo && xefgInfo.fg);
}
// Intel XeSS-FG's setup row (the Intel XeSS upscaler's flow): shown while its box is ticked and Intel's two DLLs are
// missing; one click downloads Intel's pinned XeSS SDK release (main checks its hash, then installs Intel's files)
function syncXefgSetup() {
  if (!$('modelxefg').checked || xefgReady) {
    $('xefgsetup').style.display = 'none';
    return;
  }
  $('xefgstatus').textContent = tr('dlss.notReady');
  $('xefgstatus').style.color = '#e85c5c';
  $('xefgsteps').innerHTML =
    '✗ ' +
    tr('xefg.line') +
    '<br>' +
    tr('xess.howto', { ver: xefgInfo.ver || '', mb: xefgInfo.mb || '73.3', link: toolPageLink('xess-open-download') });
  $('xefgbtns').style.display = '';
  $('xefgsetup').style.display = 'block';
}
// Reflect the current model choice: show a setup hint only while the chosen model's runtime is
// missing (FRUC has a one-time DLL installer).
function syncModel() {
  localStorage.setItem(
    'interpModel',
    !interpOn()
      ? 'none' // nothing ticked = interpolation off, remembered
      : $('modelfruc').checked
        ? 'fruc'
        : $('modelfsrfg').checked
          ? 'fsrfg'
          : $('modelxefg').checked
            ? 'xefg'
            : $('modelnvvfg').checked
              ? 'nvvfg'
              : $('modeldlss').checked
                ? 'dlssg'
                : $('modelrife').checked
                  ? 'rife'
                  : $('modelnvof').checked
                    ? 'nvof'
                    : 'gmfss',
  );
  // The DRBA sub-option only makes sense while RIFE is the chosen model (always ready: bundled).
  $('rifedrbarow').style.display = $('modelrife').checked ? '' : 'none';
  if ($('modeldlss').checked && !dlssReady) {
    $('dlsssteps').innerHTML = tr('dlssg.incomplete', { files: dlssMissing.join(', ') });
    $('dlsssetup').style.display = 'block';
  } else {
    $('dlsssetup').style.display = 'none';
  }
  if ($('modelfruc').checked && !frucReady) {
    $('frucstatus').textContent = tr('dlss.notReady');
    $('frucstatus').style.color = '#e85c5c';
    $('frucsteps').innerHTML = frucSetupHtml();
    $('frucbtns').style.display = '';
    $('frucsetup').style.display = 'block';
  } else {
    $('frucsetup').style.display = 'none';
  }
  syncXefgSetup();
  syncVfgSetup();
}
// Exclusive-group behaviour: ticking a box clears the others. Unticking the ACTIVE box now leaves
// NOTHING checked, and that is the "no interpolation" state (the group replaced the old separate
// Interpolate checkbox), so syncInterp() runs to grey the Speed controls and retag the output.
function pickModel(box) {
  for (const b of MODEL_BOXES()) if (b !== box) b.checked = false;
  syncModel();
  syncInterp();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {} // live inherits this model choice
  fsrWarm();
  syncCompareUi();
  if (cmpOn()) refreshCompare();
  else refreshPreviewIfOpen(); // Compare's model A is this choice
}
async function doFrucInstall(source) {
  $('frucsetup').style.display = 'block';
  $('frucstatus').textContent = tr('common.installing');
  $('frucstatus').style.color = 'var(--sub)';
  let r: any = {};
  try {
    r = await ipcRenderer.invoke('fruc-install', source);
  } catch (e) {
    r = { ok: false, error: String(e) };
  }
  await refreshFrucState();
  if (frucReady) {
    $('modelfruc').checked = true;
    $('modelgmfss').checked = false;
    localStorage.setItem('interpModel', 'fruc');
    $('frucsetup').style.display = 'none';
  } else if (r.ok) {
    // NvOFFRUC.dll copied fine; readiness just needs the one-time bridge build
    $('frucstatus').textContent = tr('fruc.dllInstalled');
    $('frucstatus').style.color = 'var(--green)';
    $('frucsteps').innerHTML = tr('fruc.lastStep');
  } else {
    $('frucstatus').textContent = tr('dlss.notReady');
    $('frucstatus').style.color = '#e85c5c';
    $('frucsteps').innerHTML = '✗ ' + (r.error || tr('common.installFailed')) + '<br>' + frucSetupHtml();
  }
}
i18nRelabel.push(syncModel);
$('frucget').onclick = () => ipcRenderer.invoke('fruc-open-download');
$('frucbrowsezip').onclick = async () => {
  const p = await ipcRenderer.invoke('fruc-choose');
  if (p) doFrucInstall(p);
}; // selecting a .zip auto-installs
for (const b of MODEL_BOXES()) b.onchange = () => pickModel(b);
// The DRBA sub-option persists on its own key (default OFF: max fluidity is the app's default
// philosophy; DRBA deliberately keeps character cadence); toggling it never clears RIFE.
$('rifedrba').checked = localStorage.getItem('rifeDrba') === '1';
$('rifedrba').onchange = () => {
  localStorage.setItem('rifeDrba', $('rifedrba').checked ? '1' : '0');
  try {
    lvSendOpts();
  } catch {}
};
// On load: learn readiness, restore the saved choice, then reflect it (a setup hint shows if the
// chosen model's runtime is missing). Skipped while Interpolate is off: syncInterp() has cleared
// the boxes and re-checking Interpolate restores the choice itself.
function restoreModelChoice() {
  const saved = localStorage.getItem('interpModel');
  if (saved === 'none') {
    // interpolation off: leave the whole group unticked
    for (const b of MODEL_BOXES()) b.checked = false;
    return;
  }
  // no saved choice (a first start): DLSS 4.5 where it runs, else GMFSS
  const box =
    saved === null
      ? dlssDefault
        ? 'modeldlss'
        : 'modelgmfss'
      : saved === 'fruc'
        ? 'modelfruc'
        : saved === 'fsrfg'
          ? 'modelfsrfg'
          : saved === 'xefg'
            ? 'modelxefg'
            : saved === 'nvvfg'
              ? 'modelnvvfg'
              : saved === 'dlssg'
                ? 'modeldlss'
                : saved === 'rife'
                  ? 'modelrife'
                  : saved === 'nvof'
                    ? 'modelnvof'
                    : 'modelgmfss';
  for (const b of MODEL_BOXES()) b.checked = false;
  $(box).checked = true;
}
// Restore FIRST (the saved choice may be "none"), then reflect it: syncInterp greys the Speed
// controls when nothing is ticked.
refreshFrucState().then(() => {
  restoreModelChoice();
  syncModel();
  syncInterp();
  try {
    lvModelUi();
    lvSendOpts();
  } catch {}
  syncCompareUi(); // Compare's choices follow the runtimes found ready
  // a saved FSR pick warms after the first paint (a spawn during launch stalls the compositor), below normal priority;
  // a window opened behind another one never paints (no rAF), so a timer starts it there
  let bootWarm = false;
  const warmOnce = () => {
    if (!bootWarm) {
      bootWarm = true;
      fsrWarm();
    }
  };
  requestAnimationFrame(() => requestAnimationFrame(() => setTimeout(warmOnce, 2000)));
  setTimeout(warmOnce, 5000);
});
