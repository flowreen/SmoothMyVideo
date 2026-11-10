// The "Environment variables" table of docs/DEVELOPMENT.md is the one list of the SMV_* levers, and this holds it
// complete: every lever a source reads is named in that file, and every SMV_* name that file holds occurs in a source.
// A lever read is its name as a whole quoted string (the argument of an environment lookup), a `process.env.` access
// or, in a .bat / .cmd file, a %NAME% expansion or a `defined NAME` test. Scanned: every engine/*/build_src but the
// vendored SDK folders, our vkd3d / dxil-spirv patches, src, scripts, the renderer's .ts files and the launcher.
// Exit 1 on a difference. Usage: node scripts/check-levers.ts [FILE]   (FILE = another copy of DEVELOPMENT.md)
import fs from 'node:fs';
import path from 'node:path';

const ROOT = path.join(import.meta.dirname, '..');
const DOC = process.argv[2] ?? path.join(ROOT, 'docs', 'DEVELOPMENT.md');
const VENDORED = new Set(['cuda', 'cuda_shim', 'nvofa']);
const SOURCE = /\.(cpp|h|inl|bat|cmd|ts|patch)$/;
const NAME = /SMV_[A-Z0-9_]+/g;
const READ = /["'](SMV_[A-Z0-9_]+)["']|process\.env\.(SMV_[A-Z0-9_]+)/g;
const READ_BATCH = /%(SMV_[A-Z0-9_]+)%|\bdefined\s+(SMV_[A-Z0-9_]+)/g;

function sources(): string[] {
  const out: string[] = [];
  const walk = (dir: string, deep: boolean) => {
    for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (deep && !VENDORED.has(entry.name)) walk(file, true);
      } else if (SOURCE.test(entry.name)) out.push(file);
    }
  };
  for (const entry of fs.readdirSync(path.join(ROOT, 'engine'), { withFileTypes: true })) {
    const dir = path.join(ROOT, 'engine', entry.name, 'build_src');
    if (entry.isDirectory() && fs.existsSync(dir)) walk(dir, true);
  }
  walk(path.join(ROOT, 'engine', 'fsrfg', 'source'), false);
  walk(path.join(ROOT, 'src'), true);
  walk(path.join(ROOT, 'scripts'), false);
  walk(path.join(ROOT, 'renderer'), false);
  out.push(path.join(ROOT, 'SmoothMyVideo.cmd'));
  return out.filter((file) => path.basename(file) !== 'check-levers.ts');
}

const readAt = new Map<string, string>();
const named = new Set<string>();
for (const file of sources()) {
  const text = fs.readFileSync(file, 'utf8');
  for (const name of text.match(NAME) ?? []) named.add(name);
  for (const m of text.matchAll(/\.(bat|cmd)$/.test(file) ? READ_BATCH : READ)) {
    const name = m[1] ?? m[2];
    if (!readAt.has(name)) readAt.set(name, path.relative(ROOT, file).replaceAll('\\', '/'));
  }
}

const documented = new Set(fs.readFileSync(DOC, 'utf8').match(NAME) ?? []);
const inSources = (name: string) =>
  name.endsWith('_') ? [...named].some((other) => other.startsWith(name)) : named.has(name);
const undocumented = [...readAt.keys()].filter((name) => !documented.has(name)).sort();
const unread = [...documented].filter((name) => !inSources(name)).sort();

console.log(`levers read in the sources: ${readAt.size}, SMV_* names in docs/DEVELOPMENT.md: ${documented.size}`);
for (const name of undocumented) console.log(`not in docs/DEVELOPMENT.md: ${name} (read in ${readAt.get(name)})`);
for (const name of unread) console.log(`in docs/DEVELOPMENT.md, in no source: ${name}`);
console.log(undocumented.length || unread.length ? 'levers: FAIL' : 'levers: OK');
process.exit(undocumented.length || unread.length ? 1 : 0);
