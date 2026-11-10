// The window's program is renderer/app.ts and its parts renderer/app-<name>.ts, in the order of the "files" list in
// renderer/tsconfig.json. This joins them into ONE classic script, renderer/app.js (git-ignored, loaded by
// index.html), with a blank line between two parts. Node's own type stripper blanks every type annotation with spaces
// and changes nothing else, so each part's text is in app.js exactly as written. Type checking is tsc's job
// (renderer/tsconfig.json, no emit): tsc's own emit would prepend "use strict", which changes how a classic script
// runs.
// Usage: node scripts/build-renderer.ts             builds app.js
//        node scripts/build-renderer.ts --where N   the part and the line that line N of app.js comes from
import fs from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import path from 'node:path';

const dir = path.join(import.meta.dirname, '..', 'renderer');
const names: string[] = JSON.parse(fs.readFileSync(path.join(dir, 'tsconfig.json'), 'utf8')).files;
const parts = names.map((name) => ({ name, text: fs.readFileSync(path.join(dir, name), 'utf8') }));

const where = process.argv.indexOf('--where');
if (where > 0) {
  let line = Number(process.argv[where + 1]);
  let found = Number.isInteger(line) && line > 0 ? '' : 'N is a line number of app.js, from 1';
  for (const [i, part] of parts.entries()) {
    if (found) break;
    const lines = part.text.split('\n').length - 1;
    if (line <= lines) found = `renderer/${part.name}:${line}`;
    else if (line === lines + 1 && i + 1 < parts.length) found = `the blank line after renderer/${part.name}`;
    line -= lines + 1;
  }
  console.log(found || 'past the end of app.js');
  process.exit(found.startsWith('renderer/') ? 0 : 1);
}

fs.writeFileSync(
  path.join(dir, 'app.js'),
  parts.map((part) => stripTypeScriptTypes(part.text, { mode: 'strip' })).join('\n'),
);
