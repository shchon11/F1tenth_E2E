#!/usr/bin/env bash
# 설치본 폴더(~/Applications/TENTH, 이미 파일이 놓인 상태) 안에서 도는 마무리: 설정 파일(없을 때만), 앱 메뉴 항목, 바탕화면 아이콘(신뢰 표시).
# 설치본에는 install.sh 이름으로 들어간다. update.sh / install_lab.sh가 부른다.
#
#   ~/Applications/TENTH/install.sh --python P --gpu N [--device cuda] [--data D] [--external E] [--server-home H]
#                                   [--server-mem 3G] [--app-mem 3G] [--min-free 3] [--app-gpu-index I] [--bank-root B]
#   --bank-root B : library 사본의 /data/compbench/banks 경로를 B로 바꾼다(그 PC에 /data/compbench가 없을 때)
#   설정 파일 ~/.config/tenth/config.yaml 이 이미 있으면 건드리지 않는다(사용자가 고친 값 보존). --force-config로 새로 씀.
set -euo pipefail
APP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=""; GPU="0"; DEVICE="cuda"; DATA="$APP/data"; EXT=""; SHOME=""; SMEM=""; AMEM=""; MINFREE="0"; AGPU=""; BANKS=""; FORCE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --python) PY="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --data) DATA="$2"; shift 2 ;;
    --external) EXT="$2"; shift 2 ;;
    --server-home) SHOME="$2"; shift 2 ;;
    --server-mem) SMEM="$2"; shift 2 ;;
    --app-mem) AMEM="$2"; shift 2 ;;
    --min-free) MINFREE="$2"; shift 2 ;;
    --app-gpu-index) AGPU="$2"; shift 2 ;;
    --bank-root) BANKS="$2"; shift 2 ;;
    --force-config) FORCE=1; shift ;;
    *) echo "모르는 옵션: $1" >&2; exit 2 ;;
  esac
done

# 1. 자료 사본의 상자 배치 경로(이 PC에 맞게)
if [ -n "$BANKS" ]; then
  sed -i "s#/data/compbench/banks#$BANKS#g" "$APP"/library/courses/*.course.yaml
fi

# 2. 설정 파일(한 곳)
CFG="$HOME/.config/tenth/config.yaml"
mkdir -p "$(dirname "$CFG")"
if [ ! -f "$CFG" ] || [ "$FORCE" = 1 ]; then
  [ -n "$PY" ] || { echo "--python 이 필요합니다(설정 파일이 없음)" >&2; exit 2; }
  cat > "$CFG" <<EOF
# TENTH 실행 설정. 실행기(~/Applications/TENTH/tenth-launch)가 열 때마다 읽는다. 고친 값은 다시 설치해도 유지된다.
python: $PY          # 시뮬 서버를 돌릴 파이썬(torch가 있는 venv)
device: $DEVICE                 # cuda | cpu (cpu는 실시간의 약 0.4배)
gpu: "$GPU"                    # 서버가 쓸 GPU(CUDA_VISIBLE_DEVICES 값)
app_gpu_index: "$AGPU"           # 앱(Godot) 그리기 GPU. 비우면 Godot이 고름
library: $APP/library   # 코스·규칙·드라이버·물리 자료
data: $DATA                # 결과(results/)·리플레이(replays/) 저장
external: "$EXT"   # 팀 지도 폴더(F1SIM_EXTERNAL). 비우면 기본
server_home: "$SHOME"           # 서버의 HOME. 비우면 보통 홈. 체크포인트는 <홈>/f1sim_runs, 캐시는 <홈>/.cache/f1sim 에서 찾는다
port: 47800                  # 쓰이고 있으면 다음 빈 번호
server_memory_max: "$SMEM"       # 서버 메모리 상한(systemd-run). 비우면 상한 없음
app_memory_max: "$AMEM"
min_free_gb: $MINFREE               # 시작 전 필요한 가용 메모리(GB)
ready_timeout_s: 300
log_dir: $APP/logs
EOF
  echo "설정 파일 씀: $CFG"
else
  echo "설정 파일 그대로 둠: $CFG"
fi

# 3. 앱 메뉴 항목 + 바탕화면 아이콘(신뢰 표시). 옛 콘솔(io.f1sim.Console)은 건드리지 않는다.
ENTRY="$HOME/.local/share/applications/tenth.desktop"
mkdir -p "$(dirname "$ENTRY")"
cat > "$ENTRY" <<EOF
[Desktop Entry]
Type=Application
Version=1.0
Name=TENTH
GenericName=1/10 레이싱 시뮬레이터
Comment=1/10 자율주행 레이싱 시뮬레이터 (시뮬 서버 + 화면 앱)
Exec=$APP/tenth-launch
Icon=$APP/icon.svg
Terminal=false
Categories=Game;Simulation;
StartupNotify=false
EOF
chmod +x "$ENTRY"
DESK="$(xdg-user-dir DESKTOP 2>/dev/null || echo "$HOME/Desktop")"
if [ -d "$DESK" ]; then
  cp "$ENTRY" "$DESK/tenth.desktop"
  chmod +x "$DESK/tenth.desktop"
  export DBUS_SESSION_BUS_ADDRESS="${DBUS_SESSION_BUS_ADDRESS:-unix:path=/run/user/$(id -u)/bus}"
  gio set "$DESK/tenth.desktop" metadata::trusted true 2>/dev/null || echo "주의: 바탕화면 아이콘 신뢰 표시 실패(처음 누를 때 '실행 허용'을 고르십시오)"
  echo "바탕화면 아이콘: $DESK/tenth.desktop (신뢰: $(gio info -a metadata::trusted "$DESK/tenth.desktop" 2>/dev/null | sed -n 's/.*trusted: //p'))"
fi
update-desktop-database "$HOME/.local/share/applications" 2>/dev/null || true
mkdir -p "$APP/logs" "$APP/run"
rm -rf "$APP/run/capture_host"          # 자동 확인용 하드 링크(옛 실행 파일을 붙잡고 있음). 다음 flow가 새로 만든다
echo "설치 끝: $APP ($(head -1 "$APP/VERSION"))"
