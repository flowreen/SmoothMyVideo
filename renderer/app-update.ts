// Update notice and automatic update (main's checkForUpdate / autoUpdate): the one-line hint under the subtitle while a
// newer release is out or downloading (with the Automatic updates tick), the header's Update button once the new
// version is unpacked (its click restarts into it; left alone, the next launch applies it). Hidden for an up-to-date
// install.
let updVersion = '';
let updUrl = '';
function updLine(key: string, vars: Record<string, string | number> = {}, link = true) {
  txt('updtxt', () => tr(key, { v: updVersion, ...vars }));
  $('updatelink').style.display = link ? '' : 'none';
  $('update').style.display = 'block';
}
ipcRenderer.on('update-available', (_e, u) => {
  if (!u || !u.version) return;
  updVersion = u.version;
  updUrl = u.url;
  updLine('update.out');
});
ipcRenderer.on('update-progress', (_e, p) => {
  if (!p || !updVersion) return;
  if (p.stage === 'download')
    updLine('update.dl', { pct: p.total ? Math.floor((100 * p.received) / p.total) : 0 }, false);
  else updLine('update.unpack', {}, false);
});
ipcRenderer.on('update-ready', (_e, u) => {
  if (!u || !u.version) return;
  updVersion = u.version;
  if ($('updatelink').style.display === 'none') $('update').style.display = 'none'; // the download line gives way
  txt('updbtn', () => tr('update.btn', { v: updVersion }));
  $('updbtn').style.display = '';
  $('updbtn').disabled = false;
});
ipcRenderer.on('update-failed', (_e, u) => {
  // the download or unpack failed or stopped, or the swap did not happen (retry: the button offers it again)
  if (!u || !updVersion) return;
  updLine(u.retry ? 'update.failed' : 'update.out');
});
// automatic update off in this copy (main's updateBlocker: no file list, a write-protected folder, too little disk
// space, the unpacker blocked, two failures of this version): the notice names why, the tick greys out, the link stays
const UPD_WHY: Record<string, string> = {
  list: 'update.why.list',
  locked: 'update.why.locked',
  space: 'update.why.space',
  unpack: 'update.why.unpack',
  failed: 'update.why.failed',
};
ipcRenderer.on('update-blocked', (_e, u) => {
  if (!u || !u.version || !UPD_WHY[u.why]) return;
  updVersion = u.version;
  txt('updtxt', () => tr('update.off', { v: updVersion, why: tr(UPD_WHY[u.why]) }));
  $('updatelink').style.display = '';
  $('update').style.display = 'block';
  $('updauto').disabled = true;
});
$('updatelink').onclick = (ev: Event) => {
  ev.preventDefault();
  if (updUrl) shell.openExternal(updUrl);
};
$('updbtn').onclick = () => {
  $('updbtn').disabled = true;
  ipcRenderer.send('update-apply');
};
// the Automatic updates tick (default on, remembered): main hears every change; off stops a download under way
const updAuto = $('updauto');
updAuto.checked = localStorage.getItem('autoUpdate') !== '0';
updAuto.onchange = () => {
  localStorage.setItem('autoUpdate', updAuto.checked ? '1' : '0');
  ipcRenderer.send('update-auto', updAuto.checked);
};
ipcRenderer.send('update-auto', updAuto.checked);
