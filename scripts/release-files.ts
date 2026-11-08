// Writes an app folder's resources\release-files.txt: "<sha256> <size> <path relative to the folder>" a line, sorted by
// path, after one '#' line. pack-zip.ts runs it on the staged release folder. The automatic update checks a downloaded
// version's files against its list, and its swap replaces only the files the installed version's list names, so
// whatever else a user keeps in the folder stays.
// Usage: node scripts/release-files.ts <app folder> <version>
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

const [folder, version] = process.argv.slice(2);
if (!folder || !version || !fs.existsSync(path.join(folder, 'resources'))) {
  console.error('usage: node scripts/release-files.ts <app folder holding resources\\> <version>');
  process.exit(1);
}
const listPath = path.join(folder, 'resources', 'release-files.txt');

function listFiles(dir: string): string[] {
  const out: string[] = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...listFiles(p));
    else if (p !== listPath) out.push(path.relative(folder, p));
  }
  return out;
}
const listed = listFiles(folder).sort();
const lines = listed.map((r) => {
  const b = fs.readFileSync(path.join(folder, r));
  return `${crypto.createHash('sha256').update(b).digest('hex')} ${b.length} ${r}`;
});
fs.writeFileSync(listPath, [`# SmoothMyVideo ${version} release files: sha256 size path`, ...lines].join('\n') + '\n');
console.log(`release-files.txt: ${listed.length} files listed`);
