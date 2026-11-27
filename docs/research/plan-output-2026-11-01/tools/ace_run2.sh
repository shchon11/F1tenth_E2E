#!/bin/bash
# (11-27) plan_rl on the lab server ACE (4x RTX A6000 48 GB, 503 GB RAM; user: GPU 3 only, GPUs 0-2 are other people's).
# ACE is reached only through the lab PC (its key lives there): asus -> shchon11@10.100.0.2 (stage ~/F1tenth/ace_stage) -> `ssh ACE`.
# The account seonghj is SHARED and its home disk is nearly full: everything we write is under /nas/workspace/shchon11_f1sim (NFS,
# 2.5 TB; user 11-27: nothing in home). env.sh points HOME, caches, TMPDIR, PIP_CACHE_DIR, F1SIM_RUNS, F1SIM_MAPS, F1SIM_EXTERNAL
# there and sets CUDA_VISIBLE_DEVICES=3 (PCI order = nvidia-smi's GPU 3). Never touch anything else (not /nas/workspace/seonghj or
# shchoi), never kill a PID we did not start. The NFS squashes owners: copies onto it use rsync -rlt, not -a (chgrp fails).
# Nothing runs unless you call a command:
#   ace_run.sh setup                      (re)write /nas/workspace/shchon11_f1sim/env.sh + dirs; build venv/ if missing (torch 2.11.0+cu128 = asus;
#                                         ACE driver 535 / CUDA 12.2 runs it by minor-version compatibility, checked 11-27)
#                                         (then `ace_run.sh data` once)
#   ace_run.sh sync                       code -> ACE: <SRC>/f1sim + tools/ (working tree, rsync --delete); SRC = $ACE_SRC or the main repo
#   ace_run.sh send PATH...               files/dirs under ~/f1sim_runs (a parent checkpoint, a prop bank) -> ACE runs/ at the same place
#   ace_run.sh data                       external/ (maps, baselines; the lab's ~/F1tenth/F1tenth_E2E/external without .git, which
#                                         lacks baselines/forzaeth_ftg -> that from asus), ~/.f1sim/maps + map caches (centerlines,
#                                         map_lut, racelines of the registry maps) -> ACE. goal_field caches are built on first use.
#   ace_run.sh launch NAME ARGS...        sync, send every ~/f1sim_runs file named in ARGS, start plan_rl --name NAME ARGS on GPU 3
#                                         (setsid nohup nice); asus paths in ARGS are rewritten to ACE's; log runs/NAME.log, pid runs/NAME.pid
#   ace_run.sh status [NAME]              GPU 3, our footprint, our processes, the log tail
#   ace_run.sh fetch NAME [EVERY_S]       ACE runs/NAME/ -> ~/f1sim_runs/NAME/, log -> out/rq/NAME.log; with EVERY_S: again every EVERY_S s
#                                         until the run's process is gone (run it under setsid for a long run)
#   ace_run.sh stop NAME                  kill the run's own PID (checked: plan_rl --name NAME)
#   ace_run.sh clean NAME                 delete ACE runs/NAME/ and runs/NAME.* (refused while it runs; fetch first)
# Measured 11-27 (scan31b2 u600 fork, T-maze world, --envs 2048, GPU 3 SHARED with another user's job at ~100 % util): peak 29.5 GB
# GPU at horizon 256 (--horizon-auto max), 2.5 GB RSS, ~1800 decisions/s (u1 128 steps 117+32 s, u2 256 steps 226+67 s; asus tm39a2
# 416 envs ~1380/s). Another user's job queue (dispatch over GPUs 0-3) also lands on GPU 3: check `status` before a long run.
# Env: ACE_SRC (repo to ship, default the main checkout), ACE_ENV='K=V K2=V2' (exported in the run), ACE_FORCE=1 (reuse a name),
#      ACE_MAX_GB (refuse to launch when /nas/workspace/shchon11_f1sim is bigger, default 100), LAB_H (lab PC address).
set -o pipefail
LAB=${LAB_H:-shchon11@10.100.0.2}
O="-o BatchMode=yes -o ConnectTimeout=20"
MAIN=/home/shchon11/F1tenth_E2E
SRC=${ACE_SRC:-$MAIN}
TREL=docs/research/plan-output-2026-11-01/tools
OUTRQ=$MAIN/docs/research/plan-output-2026-11-01/out/rq
RUNS=/home/shchon11/f1sim_runs
STAGE=F1tenth/ace_stage                      # on the lab PC, relative to its home
AB=/nas/workspace/shchon11_f1sim              # on ACE
AR=$AB/runs
MAXGB=${ACE_MAX_GB:-100}
GPU=${ACE_GPU:-3}                            # (11-27) the user allowed GPU 2 too
case "$GPU" in 2|3) ;; *) echo "ace_run: ACE_GPU must be 2 or 3" >&2; exit 1 ;; esac

lab() { ssh $O "$LAB" "$@" < /dev/null; }
ace() { ssh $O "$LAB" "ssh $O ACE bash -s"; }          # the script comes on stdin (no quoting through two shells)
die() { echo "ace_run: $*" >&2; exit 1; }

sync_code() {
  [ -d "$SRC/f1sim/f1sim" ] || die "no f1sim package under $SRC"
  lab "mkdir -p ~/$STAGE/F1tenth_E2E/$TREL" || die "lab PC unreachable ($LAB)"
  rsync -az --delete -e "ssh $O" --exclude __pycache__ --exclude '*.egg-info' --exclude '*.pyc' \
    "$SRC/f1sim/" "$LAB:$STAGE/F1tenth_E2E/f1sim/" || die "rsync code -> lab"
  rsync -az --delete -e "ssh $O" --exclude __pycache__ --exclude '*.pyc' \
    "$SRC/$TREL/" "$LAB:$STAGE/F1tenth_E2E/$TREL/" || die "rsync tools -> lab"
  local stamp
  stamp="$(date '+%F %T') $(git -C "$SRC" rev-parse --short HEAD) $(git -C "$SRC" rev-parse --abbrev-ref HEAD) src=$SRC dirty_files=$(git -C "$SRC" status --short -- f1sim $TREL | wc -l)"
  echo "$stamp" | ssh $O "$LAB" "cat > ~/$STAGE/F1tenth_E2E/SYNCED"
  lab "cd ~/$STAGE && rsync -rlt --delete --exclude __pycache__ F1tenth_E2E/f1sim/ ACE:$AB/F1tenth_E2E/f1sim/ && rsync -rlt --delete --exclude __pycache__ F1tenth_E2E/$TREL/ ACE:$AB/F1tenth_E2E/$TREL/ && rsync -rlt F1tenth_E2E/SYNCED ACE:$AB/F1tenth_E2E/SYNCED" \
    || die "rsync lab -> ACE"
  echo "synced: $stamp"
}

send_paths() {                                  # paths under ~/f1sim_runs, same relative place on ACE
  local p rel
  for p in "$@"; do
    p=$(readlink -f "$p"); case "$p" in "$RUNS"/*) ;; *) die "not under $RUNS: $p" ;; esac
    [ -e "$p" ] || die "missing: $p"
    rel=${p#"$RUNS"/}
    lab "mkdir -p ~/$STAGE/runs/$(dirname "$rel")" &&
    rsync -az -e "ssh $O" "$p" "$LAB:$STAGE/runs/$(dirname "$rel")/" &&
    lab "cd ~/$STAGE/runs && ssh $O ACE 'mkdir -p $AR/$(dirname "$rel")' && rsync -rlt '$rel' 'ACE:$AR/$(dirname "$rel")/'" || die "send $rel"
    echo "sent $rel"
  done
}

send_data() {
  local maps
  maps=$(ls ~/.f1sim/maps)
  lab "mkdir -p ~/$STAGE/home/.f1sim/maps ~/$STAGE/home/.cache/f1sim/racelines"
  rsync -az -e "ssh $O" ~/.f1sim/maps/ "$LAB:$STAGE/home/.f1sim/maps/" || die "maps"
  rsync -az -e "ssh $O" ~/.cache/f1sim/centerlines ~/.cache/f1sim/map_lut "$LAB:$STAGE/home/.cache/f1sim/" || die "caches"
  for m in $maps; do
    ls ~/.cache/f1sim/racelines/"${m}"_*.csv >/dev/null 2>&1 &&
      rsync -az -e "ssh $O" ~/.cache/f1sim/racelines/"${m}"_*.csv "$LAB:$STAGE/home/.cache/f1sim/racelines/"
  done
  lab "mkdir -p ~/$STAGE/F1tenth_E2E/external/baselines"
  rsync -az -e "ssh $O" "$MAIN/external/baselines/forzaeth_ftg" "$LAB:$STAGE/F1tenth_E2E/external/baselines/" || die "forzaeth_ftg"
  lab "ssh $O ACE 'mkdir -p $AB/F1tenth_E2E/external' && rsync -rlt --exclude .git ~/F1tenth/F1tenth_E2E/external/ ACE:$AB/F1tenth_E2E/external/ && cd ~/$STAGE && rsync -rlt F1tenth_E2E/external/ ACE:$AB/F1tenth_E2E/external/ && rsync -rlt home/ ACE:$AB/home/" || die "lab -> ACE"
  echo "data sent (external, maps: $maps)"
}

launch() {
  local name=$1; shift
  [ -n "$name" ] || die "launch NAME ARGS..."
  case "$name" in *[!A-Za-z0-9_.-]*) die "name: letters, digits, _ . - only" ;; esac
  if [ -z "$ACE_FORCE" ] && [ -e "$RUNS/$name" ]; then die "$RUNS/$name exists here: pick an unused name (ACE_FORCE=1 overrides)"; fi
  local a args=() files=()
  for a in "$@"; do                             # asus paths -> ACE paths; checkpoints etc. under ~/f1sim_runs get sent
    if [ -e "$a" ] && [[ "$(readlink -f "$a")" == "$RUNS"/* ]]; then files+=("$a"); fi
    a=${a//"$SRC/"/"$AB/F1tenth_E2E/"}; a=${a//"$MAIN/"/"$AB/F1tenth_E2E/"}; a=${a//"$RUNS/"/"$AR/"}
    a=${a//"$RUNS"/"$AR"}
    args+=("$a")
  done
  sync_code
  [ ${#files[@]} -gt 0 ] && send_paths "${files[@]}"
  local sh kv
  for kv in $ACE_ENV; do case "${kv%%=*}" in CUDA_VISIBLE_DEVICES|HOME|F1SIM_RUNS) die "ACE_ENV may not set ${kv%%=*}" ;; esac; done
  sh=$(mktemp)
  {
    echo "#!/bin/bash"
    echo "# ace_run.sh launch $name ($(date '+%F %T'))"
    echo "source $AB/env.sh"
    echo "export CUDA_VISIBLE_DEVICES=$GPU            # (11-27) ACE_GPU: 3 (default) or 2, both allowed by the user"
    for kv in $ACE_ENV; do printf 'export %s=%q\n' "${kv%%=*}" "${kv#*=}"; done
    echo "echo \$\$ > $AR/$name.pid"
    echo "cp $AB/F1tenth_E2E/SYNCED $AR/$name.code"
    echo "cd $AB/F1tenth_E2E/f1sim"
    printf 'exec nice -n 10 python -u -m f1sim.learn.plan_rl --name %q --runs %q' "$name" "$AR"
    printf ' %q' "${args[@]}"
    echo
  } > "$sh"
  ace <<EOF
set -e
cd $AR
if [ -z "$ACE_FORCE" ] && { [ -e $AR/$name ] || [ -e $AR/$name.log ]; }; then echo "ACE already has $name"; exit 3; fi
used=\$(du -s -BG $AB | cut -f1 | tr -d G)
if [ "\$used" -gt $MAXGB ]; then echo "/nas/workspace/shchon11_f1sim is \${used} GB > $MAXGB GB: clean up first"; exit 4; fi
cat > $AR/$name.sh <<'SH'
$(cat "$sh")
SH
chmod +x $AR/$name.sh
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i $GPU | sed 's/^/GPU /'
(setsid nohup $AR/$name.sh > $AR/$name.log 2>&1 < /dev/null &)
sleep 30
p=\$(cat $AR/$name.pid 2>/dev/null)
if [ -n "\$p" ] && ps -p "\$p" > /dev/null; then echo "alive after 30 s: pid \$p"; tail -5 $AR/$name.log
else echo "DIED within 30 s:"; tail -30 $AR/$name.log; exit 5; fi
EOF
  local rc=$?
  rm -f "$sh"
  [ $rc -eq 0 ] || exit $rc
  echo "launched $name on ACE GPU $GPU -- log $AR/$name.log; fetch with: $0 fetch $name 600"
}

status() {
  local name=$1
  ace <<EOF
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed 's/^/GPU /'
nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader
echo "ours: \$(du -sh $AB | cut -f1) in $AB; NAS free \$(df -h --output=avail $AB | tail -1); from us in home: \$(ls -d /home/seonghj/f1sim_shchon11 2>/dev/null || echo none)"
for f in $AR/*.pid; do [ -e "\$f" ] || continue; p=\$(cat "\$f"); n=\$(basename "\$f" .pid)
  if ps -p "\$p" -o args= 2>/dev/null | grep -q "plan_rl --name \$n "; then echo "running \$n pid \$p (\$(ps -p \$p -o etime=,rss= | awk '{print \$1", "int(\$2/1048576)" GB RSS"}'))"; fi; done
[ -n "$name" ] && { echo "--- $name.log"; tail -8 $AR/$name.log; }
EOF
}

fetch_once() {
  local name=$1
  # (11-27) the live view's files (rewritten several times a second; ace_live_play.py replays them here) are not fetched: rsync
  # failed on them ("failed verification", code 23) and that ended the fetch loop
  lab "mkdir -p ~/$STAGE/back && rsync -a --exclude 'live_tap*' --exclude 'live_log*' ACE:$AR/$name ACE:$AR/$name.log ~/$STAGE/back/ 2>/dev/null; rsync -a ACE:$AR/$name.code ~/$STAGE/back/ 2>/dev/null; true"
  if [ -e "$RUNS/$name" ] && [ ! -e "$RUNS/$name/.from_ace" ]; then die "$RUNS/$name exists and did not come from ACE: not mixing runs"; fi
  mkdir -p "$RUNS/$name" "$OUTRQ" && touch "$RUNS/$name/.from_ace"
  rsync -az --exclude 'live_tap*' --exclude 'live_log*' -e "ssh $O" "$LAB:$STAGE/back/$name/" "$RUNS/$name/" || die "fetch $name"
  rsync -az -e "ssh $O" "$LAB:$STAGE/back/$name.log" "$OUTRQ/$name.log"
  rsync -az -e "ssh $O" "$LAB:$STAGE/back/$name.code" "$RUNS/$name/ace_code.txt" 2>/dev/null
  echo "$(date +%T) fetched $name: $(ls "$RUNS/$name" | grep -c '\.pt$') checkpoints, log $OUTRQ/$name.log"
}

setup() {
  ace <<'EOF'
set -e
B=/nas/workspace/shchon11_f1sim
mkdir -p $B/F1tenth_E2E/external $B/runs $B/tmp $B/home/.cache/f1sim $B/home/.f1sim/maps $B/home/.config/matplotlib
ln -sfn $B/runs $B/home/f1sim_runs
cat > $B/env.sh <<'ENV'
# source me (bash): the F1TENTH sim env on ACE (written by ace_run.sh setup). The account seonghj is SHARED and its home disk is
# nearly full -- every path we write is under /nas/workspace/shchon11_f1sim (NFS; HOME is redirected there, so ~/.cache/f1sim,
# ~/.triton, inductor, pip, TMPDIR all stay inside).
B=/nas/workspace/shchon11_f1sim
export HOME=$B/home XDG_CACHE_HOME=$B/home/.cache TMPDIR=$B/tmp PIP_CACHE_DIR=$B/home/.cache/pip
export TRITON_CACHE_DIR=$B/home/.triton/cache TORCHINDUCTOR_CACHE_DIR=$B/home/.cache/f1sim/inductor
export CUDA_CACHE_PATH=$B/home/.nv/ComputeCache MPLCONFIGDIR=$B/home/.config/matplotlib
export F1SIM_RUNS=$B/runs F1SIM_MAPS=$B/home/.f1sim/maps F1SIM_EXTERNAL=$B/F1tenth_E2E/external F1SIM_SCENES=$B/home/f1sim_scenes
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 PYTHONNOUSERSITE=1 WANDB_MODE=disabled PYTHONPATH=$B/F1tenth_E2E/f1sim
[ -f $B/venv/bin/activate ] && source $B/venv/bin/activate
ENV
source $B/env.sh
if [ ! -x $B/venv/bin/python ]; then
  /usr/bin/python3 -m venv $B/venv && source $B/venv/bin/activate
  python -m pip install -q --no-cache-dir -U pip "setuptools<80" wheel
  python -m pip install -q --no-cache-dir torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
  python -m pip install -q --no-cache-dir numpy==1.26.4 scipy==1.15.3 scikit-image==0.25.2 PyYAML==6.0.2 pillow==12.3.0 \
    matplotlib==3.10.8 websockets==16.1.1 h5py==3.16.0 trimesh==5.1.0 mapbox_earcut==2.0.0 gymnasium==1.3.0 networkx==3.4.2
fi
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
EOF
}

clean() {
  local name=$1
  [ -n "$name" ] || die "clean NAME"
  case "$name" in *[!A-Za-z0-9_.-]*|.*) die "bad name" ;; esac
  ace <<EOF
p=\$(cat $AR/$name.pid 2>/dev/null)
if [ -n "\$p" ] && ps -p "\$p" -o args= | grep -q "plan_rl --name $name "; then echo "$name is running (pid \$p): not deleting"; exit 1; fi
rm -rf -- "$AR/$name" "$AR/$name.log" "$AR/$name.pid" "$AR/$name.sh" "$AR/$name.code" "$AR/$name.mem"
echo "deleted $name on ACE; ours now \$(du -sh $AB | cut -f1)"
EOF
}

alive() {
  ace <<EOF | grep -q yes
p=\$(cat $AR/$1.pid 2>/dev/null) && ps -p "\$p" -o args= | grep -q "plan_rl --name $1 " && echo yes
EOF
}

case "$1" in
  setup)  setup ;;
  sync)   sync_code ;;
  clean)  clean "$2" ;;
  send)   shift; send_paths "$@" ;;
  data)   send_data ;;
  launch) shift; launch "$@" ;;
  status) status "$2" ;;
  fetch)  [ -n "$2" ] || die "fetch NAME [EVERY_S]"
          if [ -n "$3" ]; then while alive "$2"; do fetch_once "$2"; sleep "$3"; done; fi
          fetch_once "$2" ;;
  stop)   [ -n "$2" ] || die "stop NAME"
          ace <<EOF
p=\$(cat $AR/$2.pid 2>/dev/null) || { echo "no pid file for $2"; exit 1; }
if ps -p "\$p" -o args= | grep -q "plan_rl --name $2 "; then kill "\$p" && echo "stopped $2 (pid \$p)"; else echo "pid \$p is not $2's plan_rl: not killing"; fi
EOF
          ;;
  *)      sed -n '2,28p' "$0"; exit 1 ;;
esac
