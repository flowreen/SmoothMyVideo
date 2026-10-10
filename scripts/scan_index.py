"""Post-edit scan for the renderer (run after EVERY edit of renderer/index.html or renderer/app.ts,
see project rules): builds app.js from app.ts first (scripts/build-renderer.ts), then duplicate element
ids, dangling $('id') references from either file, every <script src> resolving to a file, and
node --check on the inline scripts and on app.js."""
import re, subprocess, sys, os, tempfile

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
subprocess.run(["node", "--disable-warning=ExperimentalWarning", os.path.join(root, "scripts", "build-renderer.ts")],
               check=True)
rdir = os.path.join(root, "renderer")
html = open(os.path.join(rdir, "index.html"), encoding="utf-8").read()
srcs = re.findall(r'<script[^>]*\bsrc="([^"]+)"', html)
missing_src = [s for s in srcs if not os.path.isfile(os.path.join(rdir, s))]
print(f"<script src>: {srcs}")
print("missing script files:", missing_src if missing_src else "none")
externals = {s: open(os.path.join(rdir, s), encoding="utf-8").read()
             for s in srcs if s not in missing_src}

ids = re.findall(r'\bid="([^"]+)"', html)
dups = sorted({i for i in ids if ids.count(i) > 1})
print(f"ids: {len(ids)} total, {len(set(ids))} unique")
print("dup ids:", dups if dups else "none")

refs = sorted(set(re.findall(r"\$\('([^']+)'\)", html + "\n".join(externals.values()))))
missing = [r for r in refs if r not in ids]
print(f"$() refs: {len(refs)} unique")
print("dangling $():", missing if missing else "none")

fails = []
inline = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", html, re.S)
inline = [s for s in inline if s.strip()]
tmp = os.path.join(tempfile.gettempdir(), "smv_index_inline.js")
open(tmp, "w", encoding="utf-8").write("\n;\n".join(inline))
checks = [("inline", tmp)] + [(s, os.path.join(rdir, s)) for s in externals]
for name, path in checks:
    r = subprocess.run(["node", "--check", path], capture_output=True, text=True)
    print(f"node --check {name}:", "OK" if r.returncode == 0 else "FAIL\n" + r.stderr[:2000])
    if r.returncode:
        fails.append(name)
sys.exit(1 if (dups or missing or missing_src or fails) else 0)
