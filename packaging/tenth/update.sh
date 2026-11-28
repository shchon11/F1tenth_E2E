#!/usr/bin/env bash
# 한 줄 갱신: 다시 빌드하고 설치한다.
#   packaging/tenth/update.sh [local|lab|both] [--worktree]     # 기본 both. --worktree = 커밋 안 한 변경까지 넣음(시험용)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="both"; WT=""
for a in "$@"; do
  case "$a" in
    local|lab|both) TARGET="$a" ;;
    --worktree) WT="--worktree" ;;
    *) echo "사용법: update.sh [local|lab|both] [--worktree]" >&2; exit 2 ;;
  esac
done
bash "$HERE/build.sh" $WT
case "$TARGET" in
  local) bash "$HERE/install_local.sh" ;;
  lab)   bash "$HERE/install_lab.sh" ;;
  both)  bash "$HERE/install_local.sh"; bash "$HERE/install_lab.sh" ;;
esac
