#!/bin/bash
# usage: count_text.sh <html>...   prints the number of visible characters (no whitespace) in the rendered DOM
R=/home/shchon11/F1tenth_E2E/docs/pm/sim-design-2026-11-28/round2
mkdir -p "$R/_tmp/chrome"
for f in "$@"; do
  google-chrome --headless=new --no-sandbox --disable-gpu --user-data-dir="$R/_tmp/chrome" --virtual-time-budget=4000 --dump-dom "file://$(realpath "$f")" 2>/dev/null > "$R/_tmp/dom.html" &
  PID=$!; for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 0.5; done; kill -0 $PID 2>/dev/null && kill $PID
  python3 -I - "$R/_tmp/dom.html" "$f" <<'PY'
import re,sys,html
s=open(sys.argv[1],encoding='utf-8',errors='ignore').read()
s=re.sub(r'<(script|style|title|head)\b.*?</\1>','',s,flags=re.S|re.I)
t=html.unescape(re.sub(r'<[^>]+>','\n',s))
chunks=[c.strip() for c in t.split('\n') if c.strip()]
n=sum(len(re.sub(r'\s+','',c)) for c in chunks)
print(f"{sys.argv[2].split('/')[-1]:28s} chars={n:5d} text_pieces={len(chunks):4d}")
PY
done
