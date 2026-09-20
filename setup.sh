#!/usr/bin/env bash
# Set up f1sim on this machine, whatever machine it is.
#
# Makes a virtualenv in the checkout, installs the right torch for whatever hardware it finds
# (CUDA, ROCm or CPU), installs f1sim into it, and checks the result actually runs. It changes
# nothing outside the checkout unless you ask for the menu entry.
#
#   ./setup.sh                    # detect hardware, full install
#   ./setup.sh --cpu              # CPU-only torch even if a GPU is present
#   ./setup.sh --no-viewer        # skip the GUI dependencies (headless training box)
#   ./setup.sh --desktop          # also add f1sim Console to the application menu
#   ./setup.sh --venv ~/envs/f1   # put the virtualenv somewhere else
#
# Safe to re-run: it upgrades in place rather than starting over.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$REPO/.venv"
FLAVOUR=""          # "", cpu, cu121, cu124, rocm -- empty means detect
VIEWER=1
DESKTOP=0

say()  { printf '\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m error\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --cpu) FLAVOUR="cpu"; shift ;;
        --cuda) FLAVOUR="${2:-cu124}"; shift 2 ;;
        --rocm) FLAVOUR="rocm"; shift ;;
        --no-viewer) VIEWER=0; shift ;;
        --desktop) DESKTOP=1; shift ;;
        --venv) VENV="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option: $1  (--help for the list)" ;;
    esac
done

# ---------------------------------------------------------------- python
PY=""
for cand in python3.12 python3.11 python3.10 python3; do
    command -v "$cand" >/dev/null || continue
    if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 10) else 1)'; then
        PY="$cand"; break
    fi
done
[ -n "$PY" ] || die "need Python 3.10 or newer; found none. On Debian/Ubuntu: sudo apt install python3 python3-venv"
say "python: $($PY -V) at $(command -v "$PY")"

if ! "$PY" -c 'import venv' 2>/dev/null; then
    die "the venv module is missing. On Debian/Ubuntu: sudo apt install python3-venv"
fi

# ---------------------------------------------------------------- hardware
if [ -z "$FLAVOUR" ]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        # Pick the wheel index from the driver, not from a version hard-coded here: a cu124 build
        # will not load on a driver too old for it, and that failure is a confusing one.
        DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)"
        DRV="${DRV:-0}"
        if   [ "$DRV" -ge 550 ]; then FLAVOUR="cu124"
        elif [ "$DRV" -ge 525 ]; then FLAVOUR="cu121"
        else warn "NVIDIA driver $DRV is older than CUDA 12 wheels need; using CPU torch."; FLAVOUR="cpu"
        fi
        say "found NVIDIA GPU(s), driver $DRV -> torch $FLAVOUR"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/     /'
    elif command -v rocminfo >/dev/null; then
        FLAVOUR="rocm"; say "found ROCm -> torch rocm"
    else
        FLAVOUR="cpu"
        say "no GPU found -> CPU torch (training will be slow; the viewer and the tests are fine)"
    fi
fi

case "$FLAVOUR" in
    cpu)   INDEX="https://download.pytorch.org/whl/cpu" ;;
    cu121) INDEX="https://download.pytorch.org/whl/cu121" ;;
    cu124) INDEX="https://download.pytorch.org/whl/cu124" ;;
    rocm)  INDEX="https://download.pytorch.org/whl/rocm6.2" ;;
    *)     die "unknown torch flavour: $FLAVOUR" ;;
esac

# ---------------------------------------------------------------- venv
if [ ! -x "$VENV/bin/python" ]; then
    say "creating virtualenv at $VENV"
    "$PY" -m venv "$VENV"
else
    say "reusing virtualenv at $VENV"
fi
VPY="$VENV/bin/python"
"$VPY" -m pip install --quiet --upgrade pip wheel

if ! "$VPY" -c 'import torch' 2>/dev/null; then
    say "installing torch ($FLAVOUR) -- this is the big one, give it a few minutes"
    "$VPY" -m pip install --index-url "$INDEX" torch
else
    say "torch already present: $("$VPY" -c 'import torch; print(torch.__version__)')"
fi

say "installing f1sim"
EXTRAS="gym,learn,dev"
[ "$VIEWER" = 1 ] && EXTRAS="$EXTRAS,viewer"
"$VPY" -m pip install --quiet -e "$REPO/f1sim[$EXTRAS]"
[ "$VIEWER" = 1 ] && "$VPY" -m pip install --quiet PyQt5

# ---------------------------------------------------------------- check
say "checking the install"
"$VPY" - <<'CHECK'
import sys
import torch
print(f"     torch {torch.__version__}  cuda={torch.cuda.is_available()}", end="")
if torch.cuda.is_available():
    print(f"  devices={torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})")
else:
    print()
import f1sim
from f1sim import tracks
print(f"     f1sim from {f1sim.__file__}")
print(f"     {len(list(tracks.REAL))} measured maps in the catalogue")
try:
    from PyQt5 import QtWidgets          # noqa: F401
    print("     PyQt5 present: the console will run")
except ImportError:
    print("     PyQt5 missing: headless only (re-run without --no-viewer for the GUI)")
CHECK

if ! command -v ffmpeg >/dev/null; then
    warn "ffmpeg not found: everything works except recording video from the console."
    warn "  Debian/Ubuntu: sudo apt install ffmpeg    Fedora: sudo dnf install ffmpeg"
fi

if [ "$DESKTOP" = 1 ]; then
    say "adding the application menu entry"
    "$REPO/packaging/install-desktop-entry.sh" --venv "$VENV"
fi

cat <<EOF

$(say "done")

    source $VENV/bin/activate
    f1sim-console                       # the GUI: drive, train and edit maps

  Everything the GUI does is also a command, e.g.
    python -m f1sim.learn.evaluate --help

  Where things are kept (override any of them with the environment variable):
    runs and checkpoints   \$F1SIM_RUNS    (default ~/f1sim_runs)
    maps you import        \$F1SIM_MAPS    (default ~/.f1sim/maps)
    scenes you draw        \$F1SIM_SCENES  (default ~/f1sim_scenes)

  Add the launcher icon later with:  packaging/install-desktop-entry.sh --venv $VENV
EOF
