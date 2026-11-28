#!/usr/bin/env bash
# 연구실 PC 설치·갱신(다시 돌리면 바뀐 것만 보냄). 먼저 packaging/tenth/build.sh.
#   - 설치본: game/build/dist/TENTH -> lab:~/Applications/TENTH (앱이 열려 있으면 멈춤)
#   - 서버 자료: lab_manifest.py 목록(체크포인트·상자 배치·지도·캐시) -> lab:~/F1tenth/tenth/ (서버의 HOME으로 씀)
#     기존 ~/F1tenth/console_main, ~/F1tenth/F1tenth_E2E, ~/f1sim_runs, ~/.cache/f1sim 은 건드리지 않는다.
#   - 서버 파이썬: 그 PC의 ~/F1tenth/.venv(torch 2.10), GPU 0(4060 Ti), 메모리 상한 10G
# 터널은 6~9 MB/s, ssh를 한꺼번에 12개 넘게 열면 끊긴다 -> rsync 한 줄씩, 끊기면 이어받기(--partial)로 3번까지.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DIST="$REPO/game/build/dist/TENTH"
H="${LAB_H:-shchon11@10.100.0.2}"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15"
[ -x "$DIST/tenth.x86_64" ] || { echo "빌드가 없습니다: $DIST (packaging/tenth/build.sh 먼저)" >&2; exit 1; }
say() { echo "[lab $(date +%T)] $*"; }
retry() { for t in 1 2 3; do "$@" && return 0; say "실패($t번째) -> 다시"; sleep 3; done; return 1; }

say "연구실 PC 확인"
$SSH "$H" 'mkdir -p ~/Applications/TENTH/run ~/F1tenth/tenth/data && flock -n ~/Applications/TENTH/run/launcher.lock true' \
  || { echo "연구실 PC의 TENTH가 실행 중입니다(또는 접속 실패). 닫고 다시 돌리십시오." >&2; exit 1; }

say "설치본 보내기($(du -sh "$DIST" | cut -f1))"
retry rsync -a --delete --partial --exclude /logs/ --exclude /run/ --exclude /data/ -e "$SSH" "$DIST/" "$H:Applications/TENTH/"

say "서버 자료 목록 만들기"
STAGE="$REPO/game/build/lab_data"
rm -rf "$STAGE"; mkdir -p "$STAGE"
"$REPO/.venv/bin/python" "$HERE/lab_manifest.py" "$DIST" | while IFS=$'\t' read -r src dst; do
  mkdir -p "$STAGE/$(dirname "$dst")"
  ln -s "$src" "$STAGE/$dst"
done
say "서버 자료 보내기($(du -shL "$STAGE" | cut -f1)) -> ~/F1tenth/tenth/"
retry rsync -a -L --partial -e "$SSH" "$STAGE/" "$H:F1tenth/tenth/"

say "연구실 PC에서 설정·앱 메뉴·바탕화면 아이콘"
$SSH "$H" 'bash ~/Applications/TENTH/install.sh --python ~/F1tenth/.venv/bin/python --gpu 0 --device cuda \
  --data ~/F1tenth/tenth/data --external ~/F1tenth/tenth/external --server-home ~/F1tenth/tenth \
  --server-mem 10G --min-free 4 --bank-root "$HOME/F1tenth/tenth/banks"'
say "끝"
