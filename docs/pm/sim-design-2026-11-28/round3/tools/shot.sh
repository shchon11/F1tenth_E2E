#!/bin/bash
# usage: shot.sh <out dir> <html>...     one headless chrome at a time, CPU only; a leftover is killed by PID
R="$(cd "$(dirname "$0")/.." && pwd)"
TMP="${TENTH_TMP:-/tmp/claude-1000/-home-shchon11-F1tenth-E2E/363f6b70-010e-4309-a11c-ec7f99fed255/scratchpad}/chrome3"
mkdir -p "$TMP"
out="$(realpath -m "$1")"; shift; mkdir -p "$out"
for f in "$@"; do
  html="$(realpath "$f")"; png="$out/$(basename "${f%.html}").png"
  google-chrome --headless=new --no-sandbox --disable-gpu --disable-software-rasterizer --hide-scrollbars \
    --user-data-dir="$TMP" --force-device-scale-factor=1 --window-size=1920,1200 \
    --virtual-time-budget=4000 --screenshot="$png" "file://$html" >/dev/null 2>&1 &
  PID=$!
  for i in $(seq 1 80); do kill -0 $PID 2>/dev/null || break; sleep 0.5; done
  if kill -0 $PID 2>/dev/null; then kill $PID; sleep 1; kill -9 $PID 2>/dev/null; echo "killed $PID"; fi
  ls -la "$png" 2>/dev/null | awk '{print $5, $9}'
done
