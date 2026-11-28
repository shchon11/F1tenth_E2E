#!/usr/bin/env bash
# TENTH 설치본 빌드: Godot 앱을 명령줄로 릴리스 내보내기 하고(에디터 GUI 없음), 서버 코드·자료 사본과 함께 한 폴더로 묶는다.
#
#   packaging/tenth/build.sh              # 커밋된 HEAD로 빌드 -> game/build/dist/TENTH/
#   packaging/tenth/build.sh --worktree   # 작업 트리(커밋 안 한 변경 포함)로 빌드(시험용)
#
# 산출물(game/build/dist/TENTH/, git 무시):
#   tenth.x86_64 + tenth.pck   Godot 4.7.2 릴리스 내보내기(템플릿 SHA-512 고정)
#   tenth-launch               실행기(서버 -> 준비되면 앱 -> 앱을 닫으면 서버 정리)
#   install.sh                 그 PC에서 설정 파일·앱 메뉴·바탕화면 아이콘을 마무리(install_here.sh)
#   server/f1sim/              시뮬 서버 코드 사본(f1sim 패키지), library/ 자료 사본
#   icon.svg, icon.png, VERSION, LICENSES/
# 템플릿: 공식 릴리스의 export_templates.tpz를 받아 SHA-512를 확인하고 linux x86_64 두 개만
#   game/.godot_bin/xdg_data/godot/export_templates/4.7.2.stable/ 에 푼다(git 무시). 내보낼 때만 XDG_DATA_HOME을 그리로 돌린다.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
GAME="$REPO/game"
GODOT="$GAME/tools/godot.sh"
VER="4.7.2-stable"; TVER="4.7.2.stable"
TPZ_SHA512="ca4d71c4d7b81dfc15d1a98baa07534aa95b03fdda78a0075b06672e1648d2e5f40980c9adc28d23e1b92e732ee7bf3461997aa804af74ec2fcd7a93ccb84079"
TPZ_URL="https://github.com/godotengine/godot/releases/download/$VER/Godot_v${VER}_export_templates.tpz"
XDG="$GAME/.godot_bin/xdg_data"
TDIR="$XDG/godot/export_templates/$TVER"
DL="$GAME/.godot_bin/dl"
BUILD="$GAME/build"
STAGE="$BUILD/stage"
DIST="$BUILD/dist/TENTH"
MODE=head
[ "${1:-}" = "--worktree" ] && MODE=worktree

say() { echo "[build $(date +%T)] $*"; }

# ---------------------------------------------------------------- 1. 내보내기 템플릿(버전·해시 고정)
templates_ok() {
  [ -f "$TDIR/.verified" ] && (cd "$TDIR" && sha256sum -c --quiet .verified 2>/dev/null)
}
if ! templates_ok; then
  mkdir -p "$DL" "$TDIR"
  TPZ="$DL/Godot_v${VER}_export_templates.tpz"
  if [ ! -f "$TPZ" ]; then
    say "내보내기 템플릿 내려받기(1.28 GB, 공식 릴리스): $TPZ_URL"
    curl -sSL -C - -o "$TPZ.part" "$TPZ_URL"
    mv "$TPZ.part" "$TPZ"
  fi
  say "템플릿 SHA-512 확인"
  echo "$TPZ_SHA512  $TPZ" | sha512sum -c --quiet - || { echo "템플릿 해시가 다릅니다: $TPZ (지우고 다시 돌리십시오)" >&2; exit 1; }
  rm -rf "$TDIR"; mkdir -p "$TDIR"
  unzip -q -j -o "$TPZ" templates/linux_release.x86_64 templates/linux_debug.x86_64 templates/version.txt -d "$TDIR"
  chmod +x "$TDIR"/linux_*.x86_64
  [ "$(cat "$TDIR/version.txt")" = "$TVER" ] || { echo "템플릿 version.txt가 $TVER이 아닙니다" >&2; exit 1; }
  (cd "$TDIR" && sha256sum linux_release.x86_64 linux_debug.x86_64 version.txt > .verified)
  echo "tpz_sha512 $TPZ_SHA512" > "$TDIR/.source"
  rm -f "$TPZ"                                   # 1.28 GB는 지운다. 푼 파일의 sha256은 .verified로 매번 확인
  say "템플릿 준비됨: $TDIR"
fi

# ---------------------------------------------------------------- 2. 소스 스냅샷(HEAD 또는 작업 트리)
cd "$REPO"
COMMIT="$(git rev-parse --short=12 HEAD)"
PATHS=(game f1sim/f1sim library)
DIRTY="$(git status --porcelain -- "${PATHS[@]}" | grep -v '^?? game/export_presets.cfg$' || true)"
rm -rf "$STAGE"; mkdir -p "$STAGE"
if [ "$MODE" = head ]; then
  say "HEAD $COMMIT 스냅샷"
  git archive HEAD "${PATHS[@]}" | tar -x -C "$STAGE"
  # 내보내기 설정은 패키징이 소유한다: 아직 커밋 전이면 작업 트리의 것을 쓴다
  cp game/export_presets.cfg "$STAGE/game/export_presets.cfg"
  if [ -n "$DIRTY" ]; then
    say "주의: 아래 변경은 커밋되지 않아 이번 빌드에 들어가지 않습니다(--worktree로 넣을 수 있음)"
    echo "$DIRTY" | sed 's/^/    /'
  fi
else
  say "작업 트리 스냅샷(HEAD $COMMIT + 커밋 안 한 변경)"
  git ls-files -co --exclude-standard -- "${PATHS[@]}" | tar -c -T - | tar -x -C "$STAGE"
fi
find "$STAGE" -name __pycache__ -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------- 3. Godot 가져오기 + 릴리스 내보내기(헤드리스)
say "Godot 가져오기(헤드리스)"
"$GODOT" --headless --import --path "$STAGE/game" > "$BUILD/import.log" 2>&1 || true
"$GODOT" --headless --import --path "$STAGE/game" > "$BUILD/import2.log" 2>&1 || true   # 첫 회의 테마 오류 줄은 두 번째에 없어진다
if grep -E "^(ERROR|SCRIPT ERROR)" "$BUILD/import2.log" >/dev/null; then
  echo "가져오기 오류(두 번째):" >&2; grep -E "^(ERROR|SCRIPT ERROR)" "$BUILD/import2.log" | head -20 >&2; exit 1
fi
OUTB="$BUILD/linux"
rm -rf "$OUTB"; mkdir -p "$OUTB"
say "릴리스 내보내기 -> $OUTB/tenth.x86_64"
XDG_DATA_HOME="$XDG" "$GODOT" --headless --path "$STAGE/game" --export-release "Linux" "$OUTB/tenth.x86_64" > "$BUILD/export.log" 2>&1 || {
  echo "내보내기 실패. 로그: $BUILD/export.log" >&2; tail -30 "$BUILD/export.log" >&2; exit 1; }
[ -x "$OUTB/tenth.x86_64" ] && [ -s "$OUTB/tenth.pck" ] || { echo "내보내기 산출물이 없습니다(로그: $BUILD/export.log)" >&2; tail -30 "$BUILD/export.log" >&2; exit 1; }
# 글꼴이 pck에 들어갔는지(내보낸 빌드의 화면 글꼴 = Pretendard)
for f in PretendardVariable.ttf IBMPlexSansKR-Regular.woff2 f1tenth_car.glb; do
  grep -q -a "$f" "$OUTB/tenth.pck" || { echo "pck에 $f가 없습니다" >&2; exit 1; }
done

# ---------------------------------------------------------------- 4. 설치본 폴더 조립
say "설치본 폴더 조립 -> $DIST"
rm -rf "$DIST"; mkdir -p "$DIST/server" "$DIST/LICENSES"
cp "$OUTB/tenth.x86_64" "$OUTB/tenth.pck" "$DIST/"
cp -r "$STAGE/f1sim/f1sim" "$DIST/server/f1sim"
rm -rf "$DIST/server/f1sim/server/tests"
cp -r "$STAGE/library" "$DIST/library"
install -m 755 "$HERE/tenth_launch.py" "$DIST/tenth-launch"
install -m 755 "$HERE/install_here.sh" "$DIST/install.sh"
cp "$HERE/server_boot.py" "$DIST/server/tenth_boot.py"      # 편집 설치(f1sim -e)의 찾기 고리를 떼고 사본만 쓴다
cp "$HERE/icon.svg" "$HERE/icon.png" "$DIST/"
cp "$STAGE/game/assets/fonts/Pretendard-LICENSE.txt" "$DIST/LICENSES/"
GODOT_LICENSE_NOTE="Godot Engine 4.7.2 (MIT) — https://godotengine.org/license"
echo "$GODOT_LICENSE_NOTE" > "$DIST/LICENSES/GODOT.txt"
{
  echo "commit $COMMIT"
  echo "source $MODE"
  echo "built $(date '+%Y-%m-%d %H:%M:%S')"
  echo "godot $VER (templates $(sed -n 's/^tpz_sha512 //p' "$TDIR/.source" | cut -c1-16)…)"
  echo "pck_sha256 $(sha256sum "$DIST/tenth.pck" | cut -c1-16)"
  [ "$MODE" = head ] && [ -n "$DIRTY" ] && echo "not_included $(echo "$DIRTY" | wc -l) uncommitted paths"
} > "$DIST/VERSION"
du -sh "$DIST" | sed 's/^/    /'
say "끝: $(tr '\n' ' ' < "$DIST/VERSION")"
