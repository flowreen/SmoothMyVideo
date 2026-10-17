// Post-edit scan for the renderer (run after EVERY edit of renderer/index.html or renderer/app.ts, see the project
// rules): builds app.js from app.ts first (scripts/build-renderer.ts), then duplicate element ids, dangling $('id')
// references from either file, every <script src> resolving to a file, and node --check on the inline scripts and on
// app.js. Exit 1 on any finding. Usage: node scripts/scan-index.ts
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const ROOT = path.join(import.meta.dirname, '..');
const RDIR = path.join(ROOT, 'renderer');
const repr = (a: string[]) => `[${a.map((s) => `'${s}'`).join(', ')}]`;
const orNone = (a: string[]) => (a.length ? repr(a) : 'none');
const isFile = (p: string) => fs.existsSync(p) && fs.statSync(p).isFile();

const build = spawnSync(
  process.execPath,
  ['--disable-warning=ExperimentalWarning', path.join(ROOT, 'scripts', 'build-renderer.ts')],
  { stdio: 'inherit' },
);
if (build.status !== 0) process.exit(build.status ?? 1);

const html = fs.readFileSync(path.join(RDIR, 'index.html'), 'utf8');
const srcs = [...html.matchAll(/<script[^>]*\bsrc="([^"]+)"/g)].map((m) => m[1]);
const missingSrc = srcs.filter((s) => !isFile(path.join(RDIR, s)));
console.log(`<script src>: ${repr(srcs)}`);
console.log(`missing script files: ${orNone(missingSrc)}`);
const externals = new Map(
  srcs.filter((s) => !missingSrc.includes(s)).map((s) => [s, fs.readFileSync(path.join(RDIR, s), 'utf8')]),
);

const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]);
const dups = [...new Set(ids.filter((id, i) => ids.indexOf(id) !== i))].sort();
console.log(`ids: ${ids.length} total, ${new Set(ids).size} unique`);
console.log(`dup ids: ${orNone(dups)}`);

const scanned = html + [...externals.values()].join('\n');
const refs = [...new Set([...scanned.matchAll(/\$\('([^']+)'\)/g)].map((m) => m[1]))].sort();
const dangling = refs.filter((r) => !ids.includes(r));
console.log(`$() refs: ${refs.length} unique`);
console.log(`dangling $(): ${orNone(dangling)}`);

const fails: string[] = [];
const inline = [...html.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map((m) => m[1]).filter((s) => s.trim());
const tmp = path.join(os.tmpdir(), 'smv_index_inline.js');
fs.writeFileSync(tmp, inline.join('\n;\n'));
const checks: [string, string][] = [
  ['inline', tmp],
  ...[...externals.keys()].map((s): [string, string] => [s, path.join(RDIR, s)]),
];
for (const [name, file] of checks) {
  const r = spawnSync(process.execPath, ['--check', file], { encoding: 'utf8' });
  console.log(`node --check ${name}: ${r.status === 0 ? 'OK' : 'FAIL\n' + (r.stderr ?? '').slice(0, 2000)}`);
  if (r.status !== 0) fails.push(name);
}
process.exit(dups.length || dangling.length || missingSrc.length || fails.length ? 1 : 0);
