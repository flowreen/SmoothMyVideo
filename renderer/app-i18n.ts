// --- UI text: every visible string lives in i18n\<code>.json -------------------------------------
// en.json is the source and the fallback for a key another language lacks. index.html marks its texts with data-i18n
// (the element's HTML) and data-i18n-title / -ph / -aria / -alt (that attribute); the code asks tr(key, {name: value})
// for every text it writes, and each code-owned text registers in i18nRelabel so a switch repaints it from the
// current state. The language = the user's pick (localStorage uiLang), else the first of Windows' languages SMV
// has, else English.
const fs = require('fs');
const { fileURLToPath } = require('url');
const I18N_DIR = path.join(path.dirname(fileURLToPath(location.href)), 'i18n');
// the selector offers the languages whose file is present (en always)
const I18N_CODES = ['en', 'ja', 'zh-CN', 'ko', 'es', 'pt-BR', 'fr', 'de', 'id', 'hi', 'ar'].filter(
  (c) => c === 'en' || fs.existsSync(path.join(I18N_DIR, c + '.json')),
);
function i18nLoad(code: string): Record<string, string> {
  try {
    return JSON.parse(fs.readFileSync(path.join(I18N_DIR, code + '.json'), 'utf8'));
  } catch {
    return {};
  }
}
const I18N_EN = i18nLoad('en');
let I18N_CUR: Record<string, string> = I18N_EN;
let uiLang = 'en';
function i18nPick(): string {
  const saved = localStorage.getItem('uiLang');
  if (saved && I18N_CODES.includes(saved)) return saved;
  for (const l of navigator.languages && navigator.languages.length
    ? navigator.languages
    : [navigator.language || 'en']) {
    const s = String(l).toLowerCase();
    if (s.startsWith('ja')) return 'ja';
    if (s === 'zh' || s.startsWith('zh-cn') || s.startsWith('zh-sg') || s.startsWith('zh-hans')) return 'zh-CN';
    if (s.startsWith('ko')) return 'ko';
    if (s.startsWith('es')) return 'es';
    if (s.startsWith('pt')) return 'pt-BR';
    if (s.startsWith('fr')) return 'fr';
    if (s.startsWith('de')) return 'de';
    if (s.startsWith('id') || s.startsWith('in')) return 'id';
    if (s.startsWith('hi')) return 'hi';
    if (s.startsWith('ar')) return 'ar';
    if (s.startsWith('en')) return 'en';
  }
  return 'en';
}
function tr(key: string, vars?: Record<string, string | number>): string {
  let s = I18N_CUR[key] ?? I18N_EN[key] ?? key;
  if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m));
  return s;
}
const I18N_ATTRS: [string, string, string][] = [
  ['i18nTitle', 'data-i18n-title', 'title'],
  ['i18nPh', 'data-i18n-ph', 'placeholder'],
  ['i18nAria', 'data-i18n-aria', 'aria-label'],
  ['i18nAlt', 'data-i18n-alt', 'alt'],
];
// Arabic reads right to left: <html dir> mirrors the layout (index.html keeps the pictures, the log and the paths
// left to right)
const I18N_RTL = ['ar'];
function applyI18n() {
  document.documentElement.lang = uiLang;
  document.documentElement.dir = I18N_RTL.includes(uiLang) ? 'rtl' : 'ltr';
  document.querySelectorAll('[data-i18n]').forEach((e: any) => {
    e.innerHTML = tr(e.dataset.i18n);
  });
  for (const [ds, sel, attr] of I18N_ATTRS)
    document.querySelectorAll('[' + sel + ']').forEach((e: any) => e.setAttribute(attr, tr(e.dataset[ds])));
}
const i18nRelabel: (() => void)[] = [];
// The lines an event writes (the render status, the frames line, the Live status, the preview note, the hotkey hint):
// txt(id, build) writes build() and keeps it, so a switch rebuilds the line in the new language even when no later
// event comes (Stopped, Done, Paused, a saved clip); raw(id, s) writes the host's own words as they are and drops it.
const i18nLine = new Map<string, () => string>();
function txt(id: string, build: () => string) {
  i18nLine.set(id, build);
  $(id).textContent = build();
}
function raw(id: string, s: string) {
  i18nLine.delete(id);
  $(id).textContent = s;
}
// the line's current text as a builder, for an event that appends to it
function lineOf(id: string): () => string {
  const f = i18nLine.get(id);
  if (f) return f;
  const s = $(id).textContent;
  return () => s;
}
function setLang(code: string) {
  uiLang = I18N_CODES.includes(code) ? code : 'en';
  I18N_CUR = uiLang === 'en' ? I18N_EN : i18nLoad(uiLang);
  try {
    localStorage.setItem('uiLang', uiLang);
  } catch {}
  applyI18n();
  for (const [id, f] of i18nLine) {
    try {
      $(id).textContent = f();
    } catch {}
  }
  for (const f of i18nRelabel) {
    try {
      f();
    } catch {}
  }
}
{
  const sel = $('langsel');
  for (const c of I18N_CODES) {
    const o = document.createElement('option');
    o.value = c;
    o.textContent = (c === 'en' ? I18N_EN : i18nLoad(c))['lang.name'] || c;
    sel.appendChild(o);
  }
  uiLang = i18nPick();
  I18N_CUR = uiLang === 'en' ? I18N_EN : i18nLoad(uiLang);
  sel.value = uiLang;
  sel.onchange = () => setLang(sel.value);
  applyI18n();
}
