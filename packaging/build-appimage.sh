#!/usr/bin/env bash
# Build a single-file f1sim Console AppImage.
#
# What you get is a portable GUI: one executable that runs on any reasonably recent x86-64 Linux
# without a virtualenv, a PYTHONPATH or a system Python of the right version.
#
# What it deliberately does NOT bundle is torch and CUDA. Those are several gigabytes, they have to
# match the driver on the machine that runs them, and bundling one build would make the AppImage
# both enormous and wrong on most machines. The console is split in two for exactly this reason:
# the window never imports torch, and the simulation runs in a worker process. So the AppImage
# carries the GUI, and on first run it asks for -- or is told -- which Python to run the worker
# with. Everything that needs a GPU goes through that interpreter, which is the machine's own.
#
#   packaging/build-appimage.sh                       # -> dist/f1sim-Console-x86_64.AppImage
#   packaging/build-appimage.sh --output /tmp/out.AppImage
#
# Needs: python3 (3.10+), and either `appimagetool` on PATH or network access to fetch it once.
# Nothing else, and nothing about the machine it is built on ends up inside the image.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
ID="io.f1sim.Console"
ARCH="$(uname -m)"
OUT="$REPO/dist/f1sim-Console-$ARCH.AppImage"
KEEP=0

while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUT="$2"; shift 2 ;;
        --keep-appdir) KEEP=1; shift ;;
        -h|--help) sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }

BUILD="$(mktemp -d)"
APPDIR="$BUILD/AppDir"
cleanup() { [ "$KEEP" = 1 ] || rm -rf "$BUILD"; }
trap cleanup EXIT

echo "==> staging AppDir"
mkdir -p "$APPDIR/usr/bin" "$APPDIR/usr/lib" "$APPDIR/usr/share/applications" \
         "$APPDIR/usr/share/icons/hicolor/scalable/apps"

# The GUI and its pure-Python dependencies, installed into the image. `--no-deps` for f1sim itself
# because its declared dependencies include torch: the console package is imported by the window,
# but the window's own import path does not touch it.
python3 -m pip install --quiet --upgrade --target "$APPDIR/usr/lib/python" \
    PyQt5 numpy pyyaml pillow
python3 -m pip install --quiet --no-deps --target "$APPDIR/usr/lib/python" "$REPO/f1sim"

BRANDING="$APPDIR/usr/lib/python/f1sim/assets/branding"
[ -f "$BRANDING/f1sim.svg" ] || { echo "branding assets did not install" >&2; exit 1; }

cp "$BRANDING/f1sim.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/$ID.svg"
cp "$BRANDING/f1sim.svg" "$APPDIR/$ID.svg"
for n in 16 24 32 48 64 128 256 512; do
    src="$BRANDING/f1sim-$n.png"
    [ -f "$src" ] || continue
    mkdir -p "$APPDIR/usr/share/icons/hicolor/${n}x${n}/apps"
    cp "$src" "$APPDIR/usr/share/icons/hicolor/${n}x${n}/apps/$ID.png"
done
cp "$BRANDING/f1sim-256.png" "$APPDIR/.DirIcon" 2>/dev/null || true

# AppImage wants the desktop file at the AppDir root as well as in usr/share.
sed 's|^Exec=.*|Exec=AppRun %f|' "$HERE/$ID.desktop" > "$APPDIR/$ID.desktop"
cp "$APPDIR/$ID.desktop" "$APPDIR/usr/share/applications/$ID.desktop"

cat > "$APPDIR/AppRun" <<'APPRUN'
#!/bin/bash
# f1sim Console launcher inside the AppImage.
#
# The image carries the window; the simulation worker runs under the machine's own Python, because
# that is the one with a torch built for its driver. `$F1SIM_WORKER_PYTHON` names it; without one
# the console looks for a sensible interpreter and, failing that, opens anyway and says what to do.
HERE="$(dirname "$(readlink -f "${0}")")"
export PYTHONPATH="$HERE/usr/lib/python${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HERE/usr/bin:$PATH"
# Do not let a bundled Qt fight the host's: the image ships its own plugins and nothing else.
unset QT_PLUGIN_PATH QT_QPA_PLATFORM_PLUGIN_PATH

if [ -z "${F1SIM_WORKER_PYTHON:-}" ]; then
    for cand in python3 python; do
        if command -v "$cand" >/dev/null && "$cand" -c "import torch" 2>/dev/null; then
            export F1SIM_WORKER_PYTHON="$(command -v "$cand")"
            break
        fi
    done
fi

exec python3 -m f1sim.viewer.console "$@"
APPRUN
chmod +x "$APPDIR/AppRun"

# The tool, fetched once into the build directory rather than installed on the machine.
TOOL="$(command -v appimagetool || true)"
if [ -z "$TOOL" ]; then
    echo "==> fetching appimagetool"
    URL="https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-$ARCH.AppImage"
    if ! curl -fsSL "$URL" -o "$BUILD/appimagetool"; then
        echo "could not download appimagetool from $URL" >&2
        echo "Install it yourself and re-run, or build without it: the AppDir is at $APPDIR" >&2
        KEEP=1
        exit 1
    fi
    chmod +x "$BUILD/appimagetool"
    TOOL="$BUILD/appimagetool"
fi

mkdir -p "$(dirname "$OUT")"
echo "==> building $OUT"
# ARCH is what appimagetool stamps into the image; it does not read it from the host.
ARCH="$ARCH" "$TOOL" --no-appstream "$APPDIR" "$OUT"
chmod +x "$OUT"
echo
echo "built: $OUT"
echo "Run it directly. For training and replay it needs a Python with torch;"
echo "set F1SIM_WORKER_PYTHON=/path/to/python if it does not find one."
