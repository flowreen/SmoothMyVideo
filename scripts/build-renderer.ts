// renderer/app.ts -> renderer/app.js (git-ignored, loaded by index.html): Node's own type stripper blanks every type
// annotation with spaces and changes nothing else, so app.js is the program exactly as written, line for line
// (DevTools line numbers match app.ts). Type checking is tsc's job (tsconfig.renderer.json, no emit): tsc's own emit
// would prepend "use strict", which changes how a classic script runs.
import fs from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import path from 'node:path';

const dir = path.join(import.meta.dirname, '..', 'renderer');
const src = fs.readFileSync(path.join(dir, 'app.ts'), 'utf8');
fs.writeFileSync(path.join(dir, 'app.js'), stripTypeScriptTypes(src, { mode: 'strip' }));
