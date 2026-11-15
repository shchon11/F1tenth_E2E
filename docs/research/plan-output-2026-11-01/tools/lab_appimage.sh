#!/bin/bash
# (11-16) lab PC: f1sim 콘솔 AppImage를 만들고 앱 메뉴·바탕화면에 넣음. 시뮬레이션은 이 PC의 venv Python(torch)으로
set -e
C=~/F1tenth/console_main; OUT=~/Applications/f1sim-Console-x86_64.AppImage; VPY=/home/shchon11/F1tenth/.venv/bin/python3
mkdir -p ~/Applications
echo "$(date +%T) AppImage 만들기"
# built beside and moved over: the running console keeps the old file open ("Text file busy" when written in place)
cd $C && PATH=/home/shchon11/F1tenth/.venv/bin:$PATH bash packaging/build-appimage.sh --output $OUT.new && mv -f $OUT.new $OUT
ID=io.f1sim.Console; B=$C/f1sim/f1sim/assets/branding; ICONS=~/.local/share/icons/hicolor
mkdir -p ~/.local/share/applications $ICONS/scalable/apps
cp $B/f1sim.svg $ICONS/scalable/apps/$ID.svg
for n in 16 24 32 48 64 128 256 512; do [ -f $B/f1sim-$n.png ] && mkdir -p $ICONS/${n}x${n}/apps && cp $B/f1sim-$n.png $ICONS/${n}x${n}/apps/$ID.png; done
# the entry: the AppImage's window, the worker under this PC's venv (torch for its driver), the copied checkpoints, cached racelines
sed -e "s|^Exec=.*|Exec=env F1SIM_WORKER_PYTHON=$VPY F1SIM_RUNS=/home/shchon11/f1sim_runs F1SIM_RACELINE_CACHE_ONLY=1 $OUT %f|" \
    -e "s|^Icon=.*|Icon=$ID|" $C/packaging/$ID.desktop > ~/.local/share/applications/$ID.desktop
chmod +x ~/.local/share/applications/$ID.desktop
cp ~/.local/share/applications/$ID.desktop ~/Desktop/$ID.desktop && chmod +x ~/Desktop/$ID.desktop
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus
gio set ~/Desktop/$ID.desktop metadata::trusted true 2>/dev/null || true
update-desktop-database ~/.local/share/applications 2>/dev/null || true
gtk-update-icon-cache -f -t $ICONS 2>/dev/null || true
ls -la $OUT; grep ^Exec ~/.local/share/applications/$ID.desktop
echo "$(date +%T) 끝"
