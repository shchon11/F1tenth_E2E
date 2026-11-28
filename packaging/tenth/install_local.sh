#!/usr/bin/env bash
# 이 PC에 설치: game/build/dist/TENTH -> ~/Applications/TENTH (앱이 열려 있으면 멈춤), 설정·앱 메뉴·바탕화면 아이콘.
# 서버는 저장소 venv(.venv), GPU 1 = 5060(이 PC는 번호가 거꾸로: 0 = 4070), 체크포인트·캐시는 보통 홈(~/f1sim_runs, ~/.cache/f1sim),
# 결과는 /data/tenth, 메모리 상한 3G(다른 작업과 함께 돌므로).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DIST="$REPO/game/build/dist/TENTH"
APP="$HOME/Applications/TENTH"
[ -x "$DIST/tenth.x86_64" ] || { echo "빌드가 없습니다: $DIST (packaging/tenth/build.sh 먼저)" >&2; exit 1; }
mkdir -p "$APP/run"
# 실행기가 쥔 잠금을 잡은 채로 바꾼다: 앱이 열려 있으면 바로 실패
flock -n "$APP/run/launcher.lock" rsync -a --delete --exclude /logs/ --exclude /run/ --exclude /data/ "$DIST/" "$APP/" \
  || { echo "TENTH가 실행 중입니다. 앱을 닫고 다시 돌리십시오." >&2; exit 1; }
"$APP/install.sh" --python "$REPO/.venv/bin/python" --gpu 1 --device cuda --data /data/tenth --external "$REPO/external" \
  --server-mem 3G --app-mem 3G --min-free 3
