// --- Live mode (real-time DLSS-G overlay; engine/dlssg/smv-live.exe) -------------------------------
// Independent of the file workflow: usable straight from the welcome screen. Two ways in, both
// ending at the exe's --fg mode (whatever window is foreground at spawn goes live): the ` global
// hotkey (instant, main-side), and the Smooth It Live! button, which arms a 5s countdown so the
// user can click the target window first (the Lossless Scaling "Scale" pattern, no keyboard
// needed). main forwards the exe's stderr as lv-out and auto-restarts on exit 4 (target resized).
let lvState = 'idle'; // idle | arming (countdown running) | running
let lvStopReq = false;
let lvTimer = null;
// the hotkey + button share these settings; main mirrors them for hotkey-started sessions
// Live's output target is INHERITED from the Speed selectors (screen-match or the fps box): the
// user sets ONE target for both file renders and live smoothing. Live inherits the Speed MODE,
// not a number derived from a loaded file (that coupling is exactly what made Live target 60
// just because the bundled sample clip is 30 fps).
function liveTargetFps() {
  return Math.round(speedMode() === 'screen' ? screenHz : curFps()); // 359.979 Hz panels read as 360
}
// ONE model choice drives file renders AND live smoothing: live derives its engine from the
// Interpolation model group. Models without a live backend fall back with a visible note
// (FRUC -> GMFSS live; DRBA -> plain RIFE live: the triple-window timing needs a future
// live implementation).
function liveModelInfo() {
  // fixed: the model cannot resample to the exact fps target; it approximates it with a whole
  // multiplier of the captured window's own rate (derived in the exe from the target) - DLSS-G
  // by design (past one pass's limit in two or three passes)
  if (modelIsDlss()) return { model: 'dlssg', name: 'DLSS 4.5', note: '', fixed: true };
  // DRBA live: the rifedrba backend renders the three-frame window, so the picture trails
  // the capture by one source frame
  if (modelIsRifeDrba()) return { model: 'rifedrba', name: 'RIFE (DRBA)', note: 'adds one source frame of delay' };
  if (modelIsRife()) return { model: 'rife', name: 'RIFE', note: '' };
  // Smooth Motion live: every fraction takes the nearest node of the pair's midpoint tree, so
  // the fruc backend is adaptive like rife/gmfss; HDR-capable like rife (the bridge warps 8-bit PQ
  // for the tweens, the real frames keep full precision)
  if (modelIsFruc()) return { model: 'fruc', name: 'Smooth Motion', note: 'tweens at 8-bit precision' };
  // AMD FSR frame generation live: Smooth Motion's route (the nearest node of the pair's midpoint tree), its own bridge
  if (modelIsFsrfg()) return { model: 'fsrfg', name: 'AMD FSR 4', note: '' };
  // Intel XeSS-FG live: AMD FSR frame generation's route with its own bridge (main passes --backend fsrfg --xess-fg)
  if (modelIsXefg()) return { model: 'xefg', name: 'Intel XeSS-FG', note: '' };
  // NVIDIA Video Frame Generation live: Smooth Motion's route with NVIDIA's own runtime; it makes a frame at any time
  // between a pair, so the nvvfg backend is adaptive like rife; an HDR display's frames go through the model at 10 bits
  if (modelIsVfg()) return { model: 'nvvfg', name: 'NVIDIA Video Frame Generation', note: '' };
  // NVIDIA Optical Flow: adaptive like rife (the splat takes any fraction), native only
  if (modelIsNvof()) return { model: 'nvof', name: 'NVIDIA Optical Flow', note: '' };
  // nothing ticked = interpolation off (same meaning as file renders): the echo backend passes
  // the captured frames through at their own rate and applies the live effects (Restore,
  // Sharpen, Upscale to, RTX VSR, RTX HDR), so the effects work without interpolating.
  if (!interpOn()) return { model: 'echo', name: 'No interpolation', note: '', effectsOnly: true };
  return { model: 'gmfss', name: 'GMFSS', note: '' };
}
// Upscaler policy: live FILL and file renders upscale with RTX VSR by default when its runtime is
// present; the shared RTX VSR checkbox is the opt-out (unticked = Lanczos3 everywhere).
function liveVsrOn() {
  return !!(rtxReady.vsr && $('rtxvsr').checked);
}
// Live TrueHDR: STRICT opt-in via the shared RTX HDR checkbox (a TrueHDR
// expansion is a deliberate look change, so live matches file renders, not the VSR silent
// default). Applies to every model on HDR screens; the exe drops it elsewhere. No
// srcHdr gate here: live has no loaded file, the source is whatever window gets captured.
function liveHdrOn() {
  return !!(rtxReady.hdr && $('rtxhdr').checked);
}
// The exe's loading note / HUD names what is actually ticked: the model, then the live effects it applies,
// e.g. "GMFSS + AnimeJaNai HD + DLSS 5 + RTX VSR" or "DLSS 4.5 + Sharpen"; Restore by its picked row's name, the
// upscaler that takes the live resize when ticked; sharpen = Sharpen's word (English for the exe's log, the GUI
// language's for the HUD)
function liveLoadParts(mi, sharpen: string) {
  const parts = mi.effectsOnly ? [] : [mi.name];
  if (restoreOn()) parts.push(ajOn() ? 'AnimeJaNai HD' : a4kRestOn() ? 'Anime4K Restore' : 'Real-ESRGAN');
  if (nrOn()) parts.push('DLSS 5');
  const up = liveVsrOn()
    ? 'RTX VSR'
    : fsrUpOn()
      ? 'AMD FSR 4.1'
      : a4kUpOn()
        ? a4kAaOn()
          ? 'Anime4K A+A'
          : 'Anime4K'
        : acnUpOn()
          ? 'ArtCNN'
          : xessUpOn()
            ? 'Intel XeSS'
            : '';
  if (up) parts.push(up);
  if ($('sharpen').checked && (parseFloat($('sharpval').value) || 0) > 0) parts.push(sharpen);
  if (liveHdrOn()) parts.push('RTX HDR');
  return parts;
}
function liveLoadLabel(mi) {
  const parts = liveLoadParts(mi, 'Sharpen');
  return parts.length ? parts.join(' + ') : 'passthrough (nothing ticked)';
}
// the HUD's opening note in the GUI language, shown by the exe as is until the first fps reading
function liveHudNote(mi) {
  const parts = liveLoadParts(mi, tr('sharpen.k'));
  return tr('lv.hud.loading', { what: parts.length ? parts.join(' + ') : tr('lv.hud.passthrough') });
}
i18nRelabel.push(() => {
  try {
    lvSendOpts();
  } catch {}
});
function lvSendOpts() {
  const mi = liveModelInfo();
  const hp = hdrColorPayload();
  ipcRenderer.send('lv-opts', {
    // ONE target source: the Speed selectors. Adaptive models (RIFE/GMFSS) resample to
    // it; fixed models (DLSS 4.5) approximate it with a whole multiplier the exe derives
    // from the captured window's measured rate.
    model: mi.model,
    live: uiMode === 'live', // main keeps a spare DLSS 4.5 host while the Live tab shows
    label: liveLoadLabel(mi), // what is ticked, for the exe's loading note + substitution note
    hudnote: liveHudNote(mi),
    note: mi.note, // message/HUD (the raw backend id reads as the wrong model)
    dlssmode: dlssMode(), // the working size's share of the presented size
    fit: $('lvfit').value,
    target: liveTargetFps(),
    // live effects: sharpen inherits the file-render setting; RTX VSR (liveVsrOn above) takes the live
    // resize that enlarges, the host picks it (none enlarges = skipped)
    sharpen: $('sharpen').checked ? parseFloat($('sharpval').value) || 0 : 0,
    rtxvsr: liveVsrOn(),
    vsrquality: vsrQuality(),
    fsrup: fsrUpOn(), // AMD FSR 4 takes the same resize when ticked instead of RTX VSR
    anime4k: a4kUpOn(), // Anime4K the same way
    a4kaa: a4kAaOn(), // its Mode A+A
    artcnn: acnUpOn(), // ArtCNN the same way
    xess: xessUpOn(), // Intel XeSS the same way
    // no "Upscale to": Live's size is what Display presents (the window or the screen), the DLSS mode its speed
    // Restore (AI detail): Real-ESRGAN or AnimeJaNai first on every presented frame
    restore: restoreOn(),
    restoreModel: restoreModelId(),
    // NVIDIA DLSS 5: Neural Rendering once per captured frame (runtime installed)
    dlssnr: nrOn(),
    nrstructure: nrStructure(),
    nrtone: nrTone(),
    nrskin: nrSkin(),
    nrautomask: nrAutoMask(),
    nrstyle: nrStyle(),
    nrpasses: nrPasses(),
    // live TrueHDR (SDR window -> HDR out) + the RTX HDR tone/colour knobs it inherits
    rtxhdr: liveHdrOn(),
    hdrcolor: hp.color,
    hdrsat: hp.saturation,
    hdrcon: sdkCon(),
    hdrvib: effVibrance(),
    hdrsb: effSatBoost(),
    hud: $('lvhud').checked,
    hudlat: $('lvhudlat').checked,
    gpufit: $('gpufit').checked,
  });
  lvTargetUi(mi);
}
// the line beside Smooth It Live!: the model, how it meets the Speed target, its load time (mi.name / mi.note above stay
// English: they also go to the host's own overlay)
function lvTargetUi(mi = liveModelInfo()) {
  const load =
    mi.model === 'rife'
      ? tr('lv.load.rife')
      : mi.model === 'rifedrba'
        ? tr('lv.load.drba')
        : mi.model === 'gmfss'
          ? tr('lv.load.gmfss')
          : mi.model === 'nvof' || mi.model === 'fsrfg' || mi.model === 'xefg' || mi.model === 'nvvfg'
            ? tr('lv.load.fast')
            : '';
  // A fixed model cannot hit an arbitrary target exactly: it picks the first whole multiple
  // of the captured window's own rate that reaches it, so say that instead of promising the exact number.
  const lvT = liveTargetFps();
  $('lvtarget').textContent =
    (mi.effectsOnly ? tr('lv.none') : mi.name) +
    ' · ' +
    (mi.effectsOnly
      ? tr('lv.effectsOnly', {
          fps: lvT,
          restore: tr('restore.k'),
          up: tr('up.k'),
          nr: tr('nr.k'),
          sharpen: tr('sharpen.k'),
          hdr: tr('hdr.k'),
        })
      : mi.fixed
        ? tr('lv.fixed', { fps: lvT }) + (mi.model === 'dlssg' ? ' ' + tr('lv.fixed.dlssg') : '')
        : tr('lv.targets', { fps: lvT })) +
    (load ? ' · ' + load : '') +
    (liveHdrOn() && ['rife', 'rifedrba', 'gmfss', 'nvof', 'dlssg'].includes(mi.model) ? ' · ' + tr('lv.rtxhdr') : '') +
    (mi.note ? ' · ' + tr('lv.note.' + mi.model) : '') +
    '  ·  ';
}
// Live's upscalers enlarge only what reaches them: Whole screen captures the screen as the player shows it (a fullscreen
// player has already stretched the video with its own scaler), and Original size at DLAA leaves nothing to enlarge; a
// ticked upscaler says so under its rows, with the setup that hands it the video's own frames (a window at the video's
// size, Fill screen). The names come from the Display row's own labels. Under Live DLSS 4.5 the ticked upscaler does
// not run at all (every model interpolates before any upscale, and DLSS 4.5's presented frames are enlarged by the
// host's Lanczos), which that line says instead.
function upFitHintUi() {
  const fit = $('lvfit').value;
  const key = !(liveVsrOn() || fsrUpOn() || a4kUpOn() || acnUpOn() || xessUpOn())
    ? ''
    : liveModelInfo().model === 'dlssg'
      ? 'up.live.dlssg'
      : fit === 'monitor'
        ? 'up.live.monitor'
        : fit === 'window' && dlssPct() >= 100
          ? 'up.live.window'
          : '';
  $('upfithint').textContent = key
    ? tr(key, {
        whole: tr('live.fit.monitor'),
        orig: tr('live.fit.window'),
        fill: tr('live.fit.fill'),
        display: tr('live.display'),
      })
    : '';
  $('upfithintrow').style.display = key ? '' : 'none';
}
function lvModelUi() {
  // every display mode works with every model (DLSS 4.5 with an effect, Fill included, runs the effects route
  // before its frame generation)
  $('lvfit').disabled = lvState === 'running';
  // Restore applies live too; named here like Upscale to
  const resTxt = restoreOn()
    ? tr('lv.fit.restore', {
        model: tr(ajOn() ? 'restoreaj.label' : a4kRestOn() ? 'restorea4k.label' : 'restore.label'),
      })
    : '';
  // DLSS 5 applies live too: once per captured frame
  const nrTxt = nrOn() ? tr('lv.fit.nr') : '';
  const effTxt = [resTxt, nrTxt].filter(Boolean).join(' · ');
  $('lvfithint').textContent =
    $('lvfit').value === 'fill'
      ? tr('lv.fit.fill') +
        ' · ' +
        (liveVsrOn()
          ? tr('lv.fit.vsr')
          : fsrUpOn()
            ? tr('lv.fit.fsr')
            : a4kUpOn()
              ? tr('lv.fit.a4k')
              : acnUpOn()
                ? tr('lv.fit.acn')
                : xessUpOn()
                  ? tr('lv.fit.xess')
                  : tr('lv.fit.up')) +
        (effTxt ? ' · ' + effTxt : '')
      : $('lvfit').value === 'monitor'
        ? tr('lv.fit.monitor') + (effTxt ? ' · ' + effTxt : '')
        : effTxt;
  upFitHintUi();
}
function lvUi() {
  const b = $('lvgo');
  if (lvState === 'running') b.textContent = tr('lv.stop');
  else if (lvState === 'idle') b.textContent = tr('live.go');
  // arming: the countdown tick owns the button text ("Cancel (N)")
  modeBtnUi(lvState !== 'idle');
  lvModelUi();
}
lvUi();
i18nRelabel.push(() => {
  lvUi();
  lvTargetUi();
});
function lvDisarm() {
  if (lvTimer) {
    clearInterval(lvTimer);
    lvTimer = null;
  }
  if (lvState === 'arming') {
    lvState = 'idle';
    lvUi();
    raw('lvstat', '');
  }
}
// The fps target is a spawn-time argument (--target, and the exe derives the slot
// count from it), so a Speed change during a running session relaunches the session like a DLSS
// mode change: a 700 ms debounce, 'lv-restart', same target window. Only a real
// change of the EFFECTIVE target (what main-live.ts would pass) restarts anything.
function lvEffTarget() {
  return Math.min(FPS_TOP, Math.max(10, Math.round(liveTargetFps())));
}
let lvSpawnTarget = 0; // effective target the running session was spawned with
let lvTargetRestartT = null;
function lvTargetChanged() {
  if (lvState !== 'running') return;
  clearTimeout(lvTargetRestartT);
  lvTargetRestartT = setTimeout(() => {
    if (lvState === 'running' && lvEffTarget() !== lvSpawnTarget) ipcRenderer.send('lv-restart');
  }, 700);
}
$('lvfit').onchange = () => {
  localStorage.setItem('lvFit', $('lvfit').value);
  lvModelUi();
  lvSendOpts();
};
// on-screen readout toggle (default ON; off spawns the exe with --no-hud)
if (localStorage.getItem('lvHud') === '0') $('lvhud').checked = false;
// latency segment of the same readout (default ON = today's behaviour; off spawns
// --no-hud-latency). Only meaningful while the meter itself is shown, so it greys out
// with the meter rather than pretending to do something.
if (localStorage.getItem('lvHudLat') === '0') $('lvhudlat').checked = false;
function lvHudLatUi() {
  const on = $('lvhud').checked;
  $('lvhudlat').disabled = !on;
  $('lvhudlatlbl').style.opacity = on ? '1' : '0.45';
  $('lvhudlatlbl').style.cursor = on ? 'pointer' : 'default';
}
$('lvhud').onchange = () => {
  localStorage.setItem('lvHud', $('lvhud').checked ? '1' : '0');
  lvHudLatUi();
  lvSendOpts();
};
$('lvhudlat').onchange = () => {
  localStorage.setItem('lvHudLat', $('lvhudlat').checked ? '1' : '0');
  lvSendOpts();
};
// "Fit to the GPU" (default ON): a spawn-time flag on both routes (off = --no-gpu-fit), so a running Live session restarts
if (localStorage.getItem('gpuFit') === '0') $('gpufit').checked = false;
$('gpufit').onchange = () => {
  localStorage.setItem('gpuFit', $('gpufit').checked ? '1' : '0');
  lvSendOpts();
  if (lvState === 'running') ipcRenderer.send('lv-restart');
};
localStorage.removeItem('lvNative'); // retired: the RIFE live route always runs inside smv-live.exe
lvHudLatUi();
// Hotkey recorder: click the button, press the new key or combo; Esc cancels. Electron
// accelerator strings ("`", "F9", "Ctrl+Alt+S"); main answers whether registration
// succeeded (another app can hold the key) and keeps the old key on failure.
let lvKeyCapture = false;
function lvAccelFrom(e) {
  if (['Control', 'Alt', 'Shift', 'Meta'].includes(e.key)) return null; // bare modifier: keep waiting
  const mods = [];
  if (e.ctrlKey) mods.push('Ctrl');
  if (e.altKey) mods.push('Alt');
  if (e.shiftKey && e.key.length > 1) mods.push('Shift'); // printable keys already carry shift in e.key
  const key = e.key === ' ' ? 'Space' : e.key.length === 1 ? e.key.toUpperCase() : e.key;
  return [...mods, key].join('+');
}
$('lvkey').onclick = () => {
  if (lvKeyCapture || lvState !== 'idle') return;
  lvKeyCapture = true;
  const old = $('lvkey').textContent;
  txt('lvkey', () => tr('lv.key.press'));
  txt('lvkeyhint', () => tr('lv.key.esc'));
  const done = (key: string, hint?: () => string) => {
    lvKeyCapture = false;
    raw('lvkey', key);
    if (hint) txt('lvkeyhint', hint);
    else raw('lvkeyhint', '');
    window.removeEventListener('keydown', h, true);
  };
  const h = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.key === 'Escape') {
      done(old);
      return;
    }
    const acc = lvAccelFrom(e);
    if (!acc) return;
    const ok = await ipcRenderer.invoke('lv-hotkey', acc);
    if (ok) {
      localStorage.setItem('lvHotkey', acc);
      done(acc);
    } else done(old, () => tr('lv.key.fail', { key: acc }));
  };
  window.addEventListener('keydown', h, true);
};
// restore the persisted hotkey at boot (main registers the default ` on its own)
{
  const savedKey = localStorage.getItem('lvHotkey');
  if (savedKey && savedKey !== '`')
    ipcRenderer.invoke('lv-hotkey', savedKey).then((ok) => {
      if (ok) raw('lvkey', savedKey);
    });
}
$('lvgo').onclick = async () => {
  if (lvState === 'running') {
    lvStopReq = true;
    try {
      await ipcRenderer.invoke('lv-stop');
    } catch {} // lv-done resets the UI
    return;
  }
  if (lvState === 'arming') {
    lvDisarm();
    return;
  }
  // arm: give the user 5s to focus the window they want smoothed, then target the foreground
  lvState = 'arming';
  let n = 5;
  const tick = () => {
    if (n <= 0) {
      if (lvTimer) {
        clearInterval(lvTimer);
        lvTimer = null;
      }
      $('lvgo').textContent = tr('lv.starting');
      ipcRenderer.send('lv-start-fg');
      return;
    }
    $('lvgo').textContent = tr('lv.cancelN', { n });
    txt('lvstat', () => tr('lv.clickWindow'));
    n--;
  };
  tick();
  lvTimer = setInterval(tick, 1000);
};
// stderr arrives in ARBITRARY chunk boundaries (the exe's unbuffered stderr can split a
// stats line mid-word), so buffer into complete lines before parsing - a per-chunk regex
// intermittently drops the latency tail into the next chunk.
let lvOutBuf = '';
let lvRefused = false; // a refusal reason owns the status line until the next session
// DLSS 4.5: the session's plan, or why none runs, kept beside the stats until the next session
let lvNote: (() => string) | null = null;
let lvMemLess = false; // this session's plan was cut to one pass for the free video memory: said beside its multiple
function lvLine(t) {
  // The host's refusal reason ("native host: <reason>", or the engine lookup / build failure) is
  // the ONE thing the user must read: it stays the last word on the panel, because the exe's own
  // generic lines ("this session cannot start in the native host", "live server failed to
  // start") arrive after it and would otherwise replace it, and lv-out never reaches the log panel.
  const rf =
    /native host: (?!this session cannot start)([^\r\n]+)/.exec(t) ||
    /native: (the host could not find or build the engines[^\r\n(]*)/.exec(t);
  if (rf) {
    lvRefused = true;
    raw('lvstat', rf[1].trim().slice(0, 240));
    return;
  }
  if (lvRefused) return;
  // the plan presenting ("at 36.00x: 6x in a second stage, 6x in a third, ..."), and a plan switched in place
  // ("steps to 12.00x in place (2x, 6x, 1x in the running ..."): the multiple and its passes, a 1x pass left out
  const at =
    /^DLSS 4\.5 at ([\d.]+)x: (\d+)x in a second stage(?:, (\d+)x in a third)?(?:, (\d+)x in a fourth)?, /.exec(t);
  const st = /^DLSS 4\.5 steps to ([\d.]+)x in place \(([\dx, ]+) in the running/.exec(t);
  // the session has no pass of its own to fall back to: its plan ended (made again, or for good), or the free
  // video memory held one pass only, or none
  const gone =
    /^DLSS 4\.5's pass(?:es)? (?:ended \(([^)]+)\)|did not start): the captured frames are shown as they are (until|from)/.exec(
      t,
    );
  const memOne =
    /^DLSS 4\.5: [\d.]+x needs ~\d+ MiB of video memory for its passes, \d+ MiB are free; one pass at/.test(t);
  const memNone = /^DLSS 4\.5: [\d.]+x needs ~\d+ MiB of video memory, \d+ MiB are free: the captured frames/.test(t);
  if (memOne) lvMemLess = true;
  const passNote = (m: string, passes: string[]) => {
    const p = passes.filter((x) => x && x !== '1');
    const less = lvMemLess;
    return () =>
      'DLSS 4.5 ' +
      String(+m) +
      '×' +
      (p.length > 1 ? ' (' + p.join('×, ') + '×)' : '') +
      (less ? ' · ' + tr('lv.memLess') : '');
  };
  if (at) {
    lvNote = passNote(at[1], [at[2], at[3], at[4]]);
  } else if (st) {
    lvNote = passNote(
      st[1],
      st[2].split(',').map((x) => x.trim().replace(/x$/, '')),
    );
  } else if (gone) {
    const why = (gone[1] || 'it did not start').slice(0, 120),
      key = gone[2] === 'until' ? 'lv.again' : 'lv.gone';
    lvNote = () => tr(key, { why });
  } else if (memNone) {
    lvNote = () => tr('lv.memNone');
  }
  // surface the exe's 2s stats lines; pass real failures through verbatim (trimmed)
  const m =
    /live: ([\d.]+) captured fps -> ([\d.]+) presented fps.*?latency ~(\d+)ms/.exec(t) ||
    /live: ([\d.]+) captured fps -> ([\d.]+) presented fps/.exec(t);
  // the measured multiplier (out over in, the host's ratio) on every model's line, beside DLSS 4.5's plan
  const rt = /presented fps \(ratio ([\d.]+)/.exec(t);
  if (m) {
    const fin = m[1],
      fout = m[2],
      ms = m[3],
      ratio = rt && +rt[1] > 0 ? ' (' + rt[1] + '×)' : '',
      note = lvNote;
    txt(
      'lvstat',
      () =>
        tr('lv.stats', { fin, fout }) +
        ratio +
        (ms ? '  ·  ' + tr('lv.latency', { ms }) : '') +
        (note ? '  ·  ' + note() : ''),
    );
  }
  // the native host is the only live route: its handoff start and cold engine build mean "loading"
  else if (/engine handoff started|host (GMFSS |Restore )?engine build for/.test(t))
    txt('lvstat', () => tr('lv.loading'));
  else if (
    /FAIL|not supported|no visible window|not a visible window|no frames captured|minimized: waiting|cannot start in the native host|live server (protocol error|stalled)/i.test(
      t,
    )
  )
    raw('lvstat', t.trim().slice(0, 160));
}
ipcRenderer.on('lv-out', (_e, t) => {
  lvOutBuf += t;
  const lines = lvOutBuf.split('\n');
  lvOutBuf = lines.pop(); // keep the trailing partial line for the next chunk
  for (const line of lines) lvLine(line);
});
// sessions can start from main (the ` hotkey) with no renderer involvement: mirror them here
ipcRenderer.on('lv-started', () => {
  if (lvTimer) {
    clearInterval(lvTimer);
    lvTimer = null;
  }
  setMode('live'); // the ` hotkey can start a session with no renderer involvement

  lvState = 'running';
  lvStopReq = false;
  lvRefused = false;
  lvNote = null;
  lvMemLess = false;
  lvUi();
  try {
    lvSpawnTarget = lvEffTarget();
  } catch {} // what --target this session carries
  txt('lvstat', () => tr('lv.starting'));
});
ipcRenderer.on('lv-done', (_e, code) => {
  lvState = 'idle';
  lvUi();
  // a requested Stop kills the exe (code null via kill(), or nonzero on Windows), and the
  // ` hotkey stops the same way from main without setting lvStopReq: both read as a clean stop
  if (!lvRefused) {
    // a refusal reason stays on the line: an exit code adds nothing to it
    if (lvStopReq || code === 0 || code === null) txt('lvstat', () => tr('lv.stopped'));
    else if (code === 5) txt('lvstat', () => tr('lv.noWindow'));
    else if (code === 7) txt('lvstat', () => tr('lv.closed'));
    else {
      const before = lineOf('lvstat');
      txt('lvstat', () => before() + '  (' + tr('lv.ended', { code }) + ')');
    }
  }
  lvStopReq = false;
});
(async () => {
  try {
    lvReady = await ipcRenderer.invoke('lv-ready');
    if (lvReady) {
      $('lvpanel').style.display = '';
      localStorage.removeItem('lvFlowPreset'); // retired: the 540/720 presets moved to the Upscale to selector
      localStorage.removeItem('lvFscale'); // retired: the Flow scale control is gone
      localStorage.removeItem('lvGmFlow'); // retired: the GMFSS flow scale slider is gone
      const ft = localStorage.getItem('lvFit');
      if (ft && [...$('lvfit').options].some((o) => o.value === ft)) $('lvfit').value = ft;
      lvModelUi();
      lvSendOpts(); // seed the hotkey's settings at boot
    }
  } catch {}
  lvUnavailUi();
})();

// First page = the settings surface without a video: every panel can be set up beside the welcome hint, but nothing
// is probed, sized or previewed (no GPU work) and Smooth It! waits until a picked or dropped video runs loadVideo.
if (!input) {
  showWorkspace(true);
  $('go').disabled = true;
  $('changeout').disabled = true;
  $('outrow').style.display = 'none';
  setStatus('run.status.select');
}
