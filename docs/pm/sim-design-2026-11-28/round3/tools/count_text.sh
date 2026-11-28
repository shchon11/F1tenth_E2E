#!/bin/bash
# usage: count_text.sh <html>...   visible characters (no whitespace) and text pieces in the rendered DOM
TMP="${TENTH_TMP:-/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad}/chrome3"
mkdir -p "$TMP"
for f in "$@"; do
  google-chrome --headless=new --no-sandbox --disable-gpu --user-data-dir="$TMP" --virtual-time-budget=3000 --dump-dom "file://$(realpath "$f")" 2>/dev/null > "$TMP/dom.html" &
  PID=$!; for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 0.5; done; kill -0 $PID 2>/dev/null && kill $PID
  python3 -I - "$TMP/dom.html" "$f" <<'PY'
import re,sys,html
s=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
s=re.sub(r'<(script|style|title|head)\b.*?</\1>','',s,flags=re.S|re.I)
t=html.unescape(re.sub(r'<[^>]+>','\n',s))
chunks=[c.strip() for c in t.split('\n') if c.strip()]
n=sum(len(re.sub(r'\s+','',c)) for c in chunks)
print(f"{sys.argv[2].split('/')[-1]:24s} chars={n:4d} pieces={len(chunks):3d}")
PY
done
