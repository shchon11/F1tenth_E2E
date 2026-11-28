#!/bin/bash
# usage: shot.sh <html> <png> [width] [height]
# one headless chrome at a time; exits by itself; leftover is killed by PID
R=/home/shchon11/F1tenth_E2E/docs/pm/sim-design-2026-11-28/round2
html="$(realpath "$1")"; png="$(realpath -m "$2")"; W="${3:-1920}"; H="${4:-1200}"
mkdir -p "$R/_tmp/chrome"
google-chrome --headless=new --no-sandbox --disable-gpu --disable-software-rasterizer --hide-scrollbars \
  --user-data-dir="$R/_tmp/chrome" --force-device-scale-factor=1 --window-size=$W,$H \
  --virtual-time-budget=4000 --screenshot="$png" "file://$html" >/dev/null 2>&1 &
PID=$!
for i in $(seq 1 60); do kill -0 $PID 2>/dev/null || break; sleep 0.5; done
if kill -0 $PID 2>/dev/null; then kill $PID; sleep 1; kill -9 $PID 2>/dev/null; echo "killed $PID"; fi
ls -la "$png" 2>/dev/null | awk '{print $5, $9}'
