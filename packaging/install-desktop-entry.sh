#!/usr/bin/env bash
# Put f1sim Console in this user's application menu.
#
# Per-user, under $XDG_DATA_HOME (~/.local/share): no root, nothing outside the home directory,
# and nothing that another user on the machine can see or has to clean up. Run it again after
# moving the checkout or rebuilding the virtualenv and it rewrites the entry.
#
#   packaging/install-desktop-entry.sh             # use whatever `f1sim-console` is on PATH
#   packaging/install-desktop-entry.sh --venv .venv   # pin it to a virtualenv in the checkout
#   packaging/install-desktop-entry.sh --uninstall
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ID="io.f1sim.Console"
DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
APPS="$DATA/applications"
ICONS="$DATA/icons/hicolor"

VENV=""
UNINSTALL=0
while [ $# -gt 0 ]; do
    case "$1" in
        --venv) VENV="$2"; shift 2 ;;
        --uninstall) UNINSTALL=1; shift ;;
        -h|--help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ "$UNINSTALL" = 1 ]; then
    rm -f "$APPS/$ID.desktop"
    find "$ICONS" -name "$ID.png" -delete 2>/dev/null || true
    rm -f "$ICONS/scalable/apps/$ID.svg"
    command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
    echo "removed $ID"
    exit 0
fi

# What the entry should run. A virtualenv is the common case and the one that breaks when the
# menu entry says a bare `f1sim-console` that is only on PATH inside an activated shell -- so when
# there is one, the entry names its interpreter by absolute path and no activation is needed.
if [ -n "$VENV" ]; then
    VENV="$(cd "$VENV" && pwd)"
    if [ -x "$VENV/bin/f1sim-console" ]; then
        EXEC="$VENV/bin/f1sim-console"
    elif [ -x "$VENV/bin/python" ]; then
        EXEC="$VENV/bin/python -m f1sim.viewer.console"
    else
        echo "no interpreter in $VENV/bin" >&2; exit 1
    fi
elif [ -x "$REPO/.venv/bin/f1sim-console" ]; then
    EXEC="$REPO/.venv/bin/f1sim-console"
elif [ -x "$REPO/.venv/bin/python" ]; then
    EXEC="$REPO/.venv/bin/python -m f1sim.viewer.console"
elif command -v f1sim-console >/dev/null; then
    EXEC="$(command -v f1sim-console)"
else
    echo "f1sim-console not found. Install the package first (see README), or pass --venv PATH." >&2
    exit 1
fi

BRANDING="$REPO/f1sim/f1sim/assets/branding"
[ -d "$BRANDING" ] || { echo "icons missing: $BRANDING" >&2; exit 1; }

mkdir -p "$APPS"
sed "s|^Exec=.*|Exec=$EXEC %f|" "$HERE/$ID.desktop" > "$APPS/$ID.desktop"
chmod 644 "$APPS/$ID.desktop"

for n in 16 24 32 48 64 128 256 512; do
    src="$BRANDING/f1sim-$n.png"
    [ -f "$src" ] || continue
    mkdir -p "$ICONS/${n}x${n}/apps"
    cp -f "$src" "$ICONS/${n}x${n}/apps/$ID.png"
done
if [ -f "$BRANDING/f1sim.svg" ]; then
    mkdir -p "$ICONS/scalable/apps"
    cp -f "$BRANDING/f1sim.svg" "$ICONS/scalable/apps/$ID.svg"
fi

# Best effort: both tools are optional, and a desktop that has neither still picks the entry up on
# the next login.
command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -qtf "$ICONS" 2>/dev/null || true

echo "installed $ID"
echo "  runs:  $EXEC"
echo "  entry: $APPS/$ID.desktop"
echo "It may take a moment to appear, or log out and back in."
