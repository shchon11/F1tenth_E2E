#!/usr/bin/env bash
# 고정된 Godot 바이너리를 부르는 유일한 입구. 버전을 바꿀 때는 이 파일의 두 줄만 고친다.
# 바이너리가 없으면 공식 배포본을 내려받고 SHA-512를 확인한다(sudo 불필요).
set -euo pipefail
GODOT_VERSION="4.7.2-stable"
GODOT_SHA512="9aa00f7a605200940bce3027a567b782f49bd8e940dd06ae9e987bd65aee1b1467edd56ed84fcdcbdd44354bf613bdbb4e5d2913e925850368e150c59ed54c65"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${GODOT_BIN_DIR:-$HERE/.godot_bin}"
BIN="$BIN_DIR/Godot_v${GODOT_VERSION}_linux.x86_64"

if [[ ! -x "$BIN" ]]; then
  mkdir -p "$BIN_DIR/dl"
  ZIP="$BIN_DIR/dl/Godot_v${GODOT_VERSION}_linux.x86_64.zip"
  if [[ ! -f "$ZIP" ]]; then
    curl -sSL -o "$ZIP" "https://github.com/godotengine/godot/releases/download/${GODOT_VERSION}/Godot_v${GODOT_VERSION}_linux.x86_64.zip"
  fi
  echo "$GODOT_SHA512  $ZIP" | sha512sum -c --quiet -
  unzip -o -q "$ZIP" -d "$BIN_DIR"
  chmod +x "$BIN"
fi
exec "$BIN" "$@"
