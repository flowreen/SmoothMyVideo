// `node dist/render/cli.js <render.py's arguments>` (priority 24 step 6e): the render's command
// line, speaking render.py's stderr protocol and exit codes; every render runs in this process
// (native.ts) on the native host.
// Like render.py it appends its session to THE log (%TEMP%\smv-engine.log, truncated past 8 MB):
// a header line, every line it writes, and a crash's full stack. The ffmpeg / host children write
// their own stderr straight to the GUI (python's fd-level tee also copied that into the log; a
// node tee would have to pipe it through this process, which blocks on the resident host's
// control pipe, so a chatty child could stall).
import * as fs from 'fs';
import { pyArgv, renderRoute, SESSION_LOG, stamp } from './native';

const LOG_CAP = 8 * 1024 * 1024;

function openLog(argv: string[]): number | null {
  try {
    try {
      if (fs.statSync(SESSION_LOG).size > LOG_CAP) fs.writeFileSync(SESSION_LOG, '');
    } catch {
      /* except OSError: pass */
    }
    const fd = fs.openSync(SESSION_LOG, 'a');
    fs.writeSync(fd, `[${stamp()}] ${pyArgv(argv)}\n`);
    return fd;
  } catch {
    return null; // no session log this run; the render must not care
  }
}

if (require.main === module) {
  const argv = process.argv.slice(2);
  const log = openLog(argv);
  const say = (s: string) => {
    fs.writeSync(2, s);
    if (log !== null) {
      try {
        fs.writeSync(log, s);
      } catch {
        /* except OSError: pass */
      }
    }
  };
  renderRoute(argv, say).then((code) => {
    process.exitCode = code;
  });
}
